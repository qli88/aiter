# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`grouped_gfx1250`: the gfx1250 grouped MoE (route, quant, both TDM batched GEMMs and
the gather-reduce), legacy's `grouped_gemm_gfx1250_a8w4`.

It runs interleaved (GUGU) g1u1 MXFP4 weights with fp4 or fp8 activations. Its tile
table (`tuned_grouped_fmoe.csv`, `AITER_CONFIG_GROUPED_FMOE`) and the `AITER_TDM_*`
tile overrides are read inside the launch, as in legacy; no `tuned_fmoe.csv` row names
it. Two outputs, one launch: `GROUPED` writes the routed result (`routed_out`, local
MoE); `GROUPED_P2P` reads the tokens `comm.dispatch_gfx1250` received and writes
`p2p_scatter`: GEMM2 scatters into the peers' combine slots (`stage2_scatter`), which
`comm.combine_gfx1250` reduces (the `MegaMoEGfx1250` pipeline).
"""

import os

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from .base import Cap, Segment
from .mxfp4 import mxfp4, mxfp8

# `comm.combine_gfx1250`'s payload formats, by `combine_quant` (the `p2p_scatter` format).
COMBINE_FORMATS = {
    "none": ScaleSpec(dtypes.bf16),
    "mxfp8": mxfp8(),
    "mxfp4": mxfp4(),
}
# The rows of the dispatched tokens it reads: one per received token, or grouped per
# expert by the compact plan (`stage1_fused`).
TOKEN_ROWS = frozenset({"received", "compact_plan"})

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


def disabled() -> bool:
    """The AITER_DISABLE_GROUPED_A8W4=1 kill switch."""
    return os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1"


def enabled() -> bool:
    """`grouped_gemm_gfx1250_a8w4`'s switches: AITER_USE_GROUPED_GEMM (on by default on
    a gfx1250 build) and the AITER_DISABLE_GROUPED_A8W4 kill switch."""
    return _grouped()._use_grouped_gemm_enabled() and not disabled()


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
    path `grouped_gfx1250` runs `GROUPED` where legacy's branch does, path
    `mega_moe_gfx1250` runs `GROUPED_P2P` between the gfx1250 dispatch and combine. It
    owns no kernel-name family and `reach` is not checked.

    `p2p`: the output is `p2p_scatter` (it covers SORT..GEMM2; the combine reduces), in
    the format of the plan's `combine_quant` (also a parameter of this segment, as the
    epilogue packs it); its local parameter is `recv_token_bound` (the received rows
    the GEMM is shown, static per bucket)."""

    name = "grouped_gfx1250"

    def __init__(self, p2p: bool = False):
        self.p2p = p2p
        self.covers = (Op.SORT, Op.GEMM2) if p2p else (Op.SORT, Op.REDUCE)
        if p2p:
            self.structure_params = frozenset({"combine_quant"})

    @property
    def capabilities(self):
        features = {
            "bias1",
            "bias2",
            "doweight_stage1",
            "ep_mask",
            "num_local_tokens",
            "act_scales",
        }
        if self.p2p:
            features = (features - {"doweight_stage1"}) | {"scatter"}
        return (
            Cap(
                arch=frozenset({"gfx1250"}),
                act_in=frozenset({mxfp4(), mxfp8()}),
                w1=frozenset({mxfp4()}),
                w2=frozenset({mxfp4()}),
                out_dtype=frozenset(
                    {dtypes.bf16} if self.p2p else {dtypes.bf16, dtypes.fp16}
                ),
                activation=frozenset({"silu", "swiglu", "situv2"}),
                gate_mode=frozenset({"interleave"}),
                g1u1=frozenset({True}),
                features=frozenset(features),
                pads=True,
                parallel=self.p2p,
            ),
        )

    def reads(self, p):
        if self.p2p:
            return (Buf("tokens", None, rows=TOKEN_ROWS),)
        return (CALL_INPUT,)

    def produces(self, p, params):
        if self.p2p:
            return Buf("p2p_scatter", COMBINE_FORMATS[params.combine_quant])
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
    def launch(c, q_dtype_a, stage2_scatter=None, dispatch_context=None):
        """`_fused_moe_impl`'s call of `grouped_gemm_gfx1250_a8w4`; its result (None
        where `unsupported` gives a reason; every None comes before a launch).
        `stage2_scatter`: the combine slots GEMM2 writes (`p2p_scatter`);
        `dispatch_context`: the FlyDSL dispatch's context (its compact plan), which
        legacy passes through `set_flydsl_dispatch_context`."""
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
            dispatch_context=dispatch_context,
        )


GROUPED = GroupedGfx1250()
GROUPED_P2P = GroupedGfx1250(p2p=True)
