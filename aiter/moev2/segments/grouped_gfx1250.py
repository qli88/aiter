# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`grouped_gfx1250`: the gfx1250 grouped MoE (route, quant, both TDM batched GEMMs and
the gather-reduce), legacy's `grouped_gemm_gfx1250_a8w4`.

It runs interleaved (GUGU) g1u1 MXFP4 weights with fp4 or fp8 activations. Its tile
table (`tuned_grouped_fmoe.csv`, `AITER_CONFIG_GROUPED_FMOE`) and the `AITER_TDM_*`
tile overrides are read inside the launch, as in legacy; no `tuned_fmoe.csv` row names
it. Output: the routed result (`routed_out`). The `p2p_scatter` output that feeds the
gfx1250 combine (`stage2_scatter`) is step 4.6b.
"""

import os

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from .base import Cap, Segment
from .mxfp4 import mxfp4, mxfp8

ACTIVATIONS = (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)
# The weight dtypes it reads as MXFP4: ATOM's loader keeps the packed bytes as uint8.
WEIGHT_DTYPES = (dtypes.fp4x2, dtypes.u8)


def _grouped():
    from aiter.ops.flydsl import grouped_moe_gfx1250

    return grouped_moe_gfx1250


def dispatch_context():
    """The FlyDSL dispatch context `MegaMoEGfx1250` sets around its `fused_moe` call
    (`set_flydsl_dispatch_context`), or None."""
    return _grouped()._flydsl_dispatch_context()


def enabled() -> bool:
    """`grouped_gemm_gfx1250_a8w4`'s switches: AITER_USE_GROUPED_GEMM (on by default on
    a gfx1250 build) and the AITER_DISABLE_GROUPED_A8W4 kill switch."""
    grouped = _grouped()
    return (
        grouped._use_grouped_gemm_enabled()
        and os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") != "1"
    )


def build_arch_named() -> bool:
    """`grouped_gemm_gfx1250_a8w4`'s arch check: the build arch, the build variables
    (`GPU_ARCHS`, `TARGET_ARCH`, `AITER_GPU_ARCHS`) or AITER_FORCE_GFX1250 name gfx1250
    (todo 5.26: it does not ask the device)."""
    from aiter.jit.utils.chip_info import get_gfx

    gfx_env = ";".join(
        str(os.environ.get(k, "")).lower()
        for k in ("GPU_ARCHS", "TARGET_ARCH", "AITER_GPU_ARCHS")
    )
    return (
        get_gfx() == "gfx1250"
        or "gfx1250" in gfx_env
        or os.environ.get("AITER_FORCE_GFX1250", "0") in _grouped()._TRUTHY_ENV
    )


class GroupedGfx1250(Segment):
    """No resolver dispatches it (no ranking entry or `tuned_fmoe.csv` row names it):
    path `grouped_gfx1250` runs it where legacy's branch does, so it owns no kernel-name
    family and `reach` is not checked."""

    name = "grouped_gfx1250"
    covers = (Op.SORT, Op.REDUCE)

    @property
    def capabilities(self):
        return (
            Cap(
                arch=frozenset({"gfx1250"}),
                act_in=frozenset({mxfp4(), mxfp8()}),
                w1=frozenset({mxfp4()}),
                w2=frozenset({mxfp4()}),
                out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
                activation=frozenset({"silu", "swiglu", "situv2"}),
                gate_mode=frozenset({"interleave"}),
                g1u1=frozenset({True}),
                features=frozenset(
                    {
                        "bias1",
                        "bias2",
                        "doweight_stage1",
                        "ep_mask",
                        "num_local_tokens",
                        "act_scales",
                    }
                ),
                pads=True,
            ),
        )

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        return Buf("routed_out", ScaleSpec(p.out_dtype))

    def build(self, p, params):
        return self.launch

    @staticmethod
    def unsupported(c, q_dtype_a) -> str | None:
        """Why `grouped_gemm_gfx1250_a8w4` returns None for the call (`executor.Call`,
        its activation format `q_dtype_a`), leaving it to the 2-stage dispatch, or
        None: it runs the call. It ignores the padding, `a2_scale` and `block_size_M`
        (todo 5.39)."""
        w1 = c["w1"]
        if c.gate_mode != GateMode.INTERLEAVE:
            return "gate/up not interleaved"
        if not enabled():
            return "AITER_USE_GROUPED_GEMM off, or AITER_DISABLE_GROUPED_A8W4=1"
        if not c.isG1U1 or c.quant_type != QuantType.per_1x32:
            return "not g1u1 per_1x32"
        if c.activation not in ACTIVATIONS:
            return f"activation {c.activation}"
        if w1.dtype not in WEIGHT_DTYPES or q_dtype_a not in (dtypes.fp4x2, dtypes.fp8):
            return f"{q_dtype_a} activations into {w1.dtype} weights"
        if c["w1_scale"] is None or c["w2_scale"] is None:
            return "weight scales missing"
        if not build_arch_named():
            return "gfx1250 not named by the build"
        if c.M and c["expert_mask"] is not None and c["doweight_stage1"]:
            return "doweight_stage1 with expert_mask"
        return None

    @staticmethod
    def launch(c, q_dtype_a, stage2_scatter=None):
        """`_fused_moe_impl`'s call of `grouped_gemm_gfx1250_a8w4`; its result (None
        where `unsupported` gives a reason; every None comes before a launch)."""
        beta, linear_beta = c["beta"], c["linear_beta"]
        return _grouped().grouped_gemm_gfx1250_a8w4(
            c["hidden_states"],
            c["w1"],
            c["w2"],
            c["topk_weight"],
            c["topk_ids"],
            E=c.E,
            model_dim=c.model_dim,
            inter_dim=c.inter_dim,
            dtype=c.dtype,
            activation=c.activation,
            quant_type=c.quant_type,
            q_dtype_a=q_dtype_a,
            q_dtype_w=c["w1"].dtype,
            isG1U1=c.isG1U1,
            doweight_stage1=c["doweight_stage1"],
            w1_scale=c["w1_scale"],
            w2_scale=c["w2_scale"],
            a1_scale=c["a1_scale"],
            expert_mask=c["expert_mask"],
            hidden_pad=c["hidden_pad"],
            intermediate_pad=c["intermediate_pad"],
            bias1=c["bias1"],
            bias2=c["bias2"],
            swiglu_limit=c["swiglu_limit"],
            num_local_tokens=c["num_local_tokens"],
            situ_beta=1.0 if beta is None else float(beta),
            situ_linear_beta=1.0 if linear_beta is None else float(linear_beta),
            stage2_scatter=stage2_scatter,
        )


GROUPED = GroupedGfx1250()
