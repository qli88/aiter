# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK 2-stage kernels (`moe_ck2stages_gemm1_*` / `moe_ck2stages_gemm2_*`)."""

import functools
from dataclasses import replace

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes

from ..dispatch import kernel_names
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec, scale_spec
from .base import Cap, Params, Segment
from .defaults import TUNE_BLOCK_MS, tunable


def ck_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    kernelName="",
    sorted_weights=None,
    quant_type=aiter.QuantType.No,
    activation=ActivationType.Gelu,
    splitk=1,
    use_non_temporal_load=False,
    dtype=None,
):
    token_num = hidden_states.shape[0]
    is_splitk = quant_type == QuantType.per_1x128 and splitk > 1
    if is_splitk:
        # CK kernel zeros this buffer via hipMemsetAsync when KBatch > 1
        sorted_size = min(token_num * topk * block_m, sorted_token_ids.shape[0])
        tmp_out = torch.empty(
            (sorted_size, w1.shape[1]), dtype=dtypes.fp32, device=out.device
        )
    else:
        tmp_out = out
    aiter.ck_moe_stage1_fwd(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        block_m,
        sorted_weights,
        quant_type,
        activation,
        splitk if is_splitk else 0,
        use_non_temporal_load,
        out.dtype,
    )
    if is_splitk:
        valid_out = tmp_out[: token_num * topk, :]
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, valid_out.view(dtypes.fp32))
        elif activation == ActivationType.Gelu:
            aiter.gelu_and_mul(out, valid_out.view(dtypes.fp32))
        else:
            raise ValueError(
                f"Unsupported activation for split-k post-activation: {activation}"
            )
    return out


def per_tensor_fp8():
    """fp8 per-tensor `ScaleSpec` of the running arch (e4m3fnuz on gfx942)."""
    return scale_spec(QuantType.per_Tensor, dtypes.fp8)


def _cap(specs, weights=None):
    weights = specs if weights is None else weights
    return Cap(
        arch=frozenset({"gfx942", "gfx950"}),
        act_in=specs,
        w1=weights,
        w2=weights,
        out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
        activation=frozenset({"silu", "gelu"}),
        gate_mode=frozenset({"separated"}),
        features=frozenset(
            {"ep_mask", "num_local_tokens", "doweight_stage1", "act_scales"}
        ),
    )


def per_token_fp8():
    """fp8 per-token `ScaleSpec` of the running arch."""
    return scale_spec(QuantType.per_Token, dtypes.fp8)


def blockscale_fp8():
    """fp8 `ScaleSpec` with 1x128 activation groups (128x128 weight blocks)."""
    return scale_spec(QuantType.per_1x128, dtypes.fp8)


def a8w8_per_tensor_caps():
    return (_cap(frozenset({per_tensor_fp8()})),)


def a8w8_per_token_caps():
    return (_cap(frozenset({per_token_fp8()})),)


def a8w8_blockscale_caps():
    return (_cap(frozenset({blockscale_fp8()})),)


def per_tensor_int8():
    return scale_spec(QuantType.per_Tensor, dtypes.i8)


def per_token_int8():
    return scale_spec(QuantType.per_Token, dtypes.i8)


def per_token_int4():
    """int4 weights packed eight to a `uint32`, per-token scales (fp8 activations)."""
    return scale_spec(QuantType.per_Token, torch.uint32)


def int8_per_tensor_caps():
    return (_cap(frozenset({per_tensor_int8()})),)


def int8_per_token_caps():
    return (_cap(frozenset({per_token_int8()})),)


def int4_per_token_caps():
    return (_cap(frozenset({per_token_fp8()}), frozenset({per_token_int4()})),)


def a16w16_caps():
    """bf16/fp16 activations and weights, unquantized (`QuantType.No`)."""
    return tuple(
        _cap(frozenset({scale_spec(QuantType.No, d)}))
        for d in (dtypes.bf16, dtypes.fp16)
    )


def a4w4_caps():
    """MXFP4 activations and weights, as the a4w4 path runs them: gfx950, no
    num_local_tokens or caller-quantized activations."""
    from .mxfp4 import a4w4_caps

    return tuple(
        replace(cap, features=cap.features | {"doweight_stage1"}) for cap in a4w4_caps()
    )


def a8w4_caps():
    """mxfp8 activations into MXFP4 weights, as the a8w4 path runs them: gfx950, gate
    and up separated."""
    from .mxfp4 import a8w4_caps

    return tuple(
        replace(
            cap,
            gate_mode=frozenset({"separated"}),
            features=cap.features | {"doweight_stage1"},
        )
        for cap in a8w4_caps()
    )


def mxfp8_caps():
    """mxfp8 activations and weights, gate and up separated: legacy's CK branch takes
    them, though `gen_instances.py` builds no f8 1x32 instance ahead of time."""
    from .mxfp4 import mxfp8_caps

    return tuple(
        replace(
            cap,
            gate_mode=frozenset({"separated"}),
            features=cap.features | {"doweight_stage1"},
        )
        for cap in mxfp8_caps()
    )


def ck_2stage_caps():
    """The `gen_instances.py` build list: f8 and i8 (and f8 x i4) per-tensor/per-token,
    f8 1x128, fp4 1x32 (fp4 or fp8 activations), and unquantized f16/b16; and the f8
    1x32 calls legacy hands the CK branch."""
    return (
        a8w8_per_tensor_caps()
        + a8w8_per_token_caps()
        + a8w8_blockscale_caps()
        + int8_per_tensor_caps()
        + int8_per_token_caps()
        + int4_per_token_caps()
        + a4w4_caps()
        + a8w4_caps()
        + mxfp8_caps()
        + a16w16_caps()
    )


def ck_stage1_caps():
    """Stage 1 always splits w1 into gate and up (`N = w1.size(1) / 2`): g1u1 only."""
    return tuple(replace(cap, g1u1=frozenset({True})) for cap in ck_2stage_caps())


def _ck_dtype(dtype):
    """The codegen name of a dtype (legacy `dtype2str_dict`), for either fp8; None for
    a dtype the codegen has no name for."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return "f8"
    return {
        dtypes.fp16: "f16",
        dtypes.bf16: "b16",
        dtypes.i8: "i8",
        dtypes.fp4x2: "fp4x2",
        torch.uint32: "i4",
        torch.int4: "i4",
    }.get(dtype)


def _ck_tunable(p) -> bool:
    """The problems the legacy CK generator lists instances for without raising:
    per-1x32 only as fp4 x fp4 without Swiglu (fp8 x fp4 goes to CK-Tile), and the
    activations the codegen maps (`ACT_OP_MAP`)."""
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_ck2stages_common import (
        ACT_OP_MAP,
    )

    if p.quant_type == QuantType.per_1x32 and not (
        p.act_in.dtype == p.w1.dtype == dtypes.fp4x2 and p.activation.kind != "swiglu"
    ):
        return False
    return p.activation.kind in ACT_OP_MAP


@functools.cache
def _ck_instances(stage, arch, a, b, c, quant, act, doweight):
    """`(name, MPerBlock)` of the CK instances the codegen builds for one call."""
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_ck2stages_common import (
        get_gemm1_kernels_list,
        get_gemm2_kernels_list,
    )

    if stage == 1:
        _, kernels = get_gemm1_kernels_list(
            a, b, c, False, quant, act, doweight, True, arch=arch
        )
    else:
        _, kernels = get_gemm2_kernels_list(
            a, b, c, False, quant, not doweight, True, arch=arch
        )
    return tuple((k.name, k.MPerBlock) for k in kernels.values())


def ck_param_space(segment, p, stage):
    names = tuple(_ck_dtype(d) for d in (p.act_in.dtype, p.w1.dtype, p.out_dtype))
    if not (segment.accepts(p) and tunable(p) and _ck_tunable(p)) or None in names:
        return ()
    instances = _ck_instances(
        stage,
        p.arch,
        *names,
        int(p.quant_type),
        p.activation.kind,
        "doweight_stage1" in p.features,
    )
    nts = (False, True) if p.quant_type == QuantType.per_1x128 else (False,)
    return tuple(
        Params(kernel=name, block_m=block_m, nt=nt)
        for block_m in TUNE_BLOCK_MS
        for name, m_per_block in instances
        if m_per_block == block_m
        for nt in nts
    )


class CkGemm1(Segment):
    name = "ck.gemm1"
    prefixes = (kernel_names.CK2STAGES,)
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        return ck_stage1_caps()

    def owns(self, kernel) -> bool:
        """An empty kernelName1 runs the CK heuristic kernel."""
        return kernel is None or (
            kernel.family == kernel_names.CK2STAGES and kernel.stage == 1
        )

    def declines_row(self, p, row):
        kernel_name2 = str(row.get("kernelName2", "") or "")
        if (
            p.quant_type == QuantType.per_1x32
            and p.any_shuffled
            and not kernel_name2.startswith("flydsl_")
        ):
            # get_2stage_cfgs keeps such a row only after its default MXFP4 choices
            # (FlyDSL, CK-Tile split-K) pass on it.
            return "tuned CK row for shuffled MXFP4 weights"
        return None

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in))

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(p.out_dtype))

    def build(self, p, params):
        return functools.partial(
            ck_moe_stage1,
            kernelName=params.kernel,
            activation=p.activation.activation,
            quant_type=p.quant_type,
            dtype=p.out_dtype,
            splitk=int(params.ksplit),
            use_non_temporal_load=params.nt,
        )

    def param_space(self, p):
        return ck_param_space(self, p, 1)


class CkGemm2(Segment):
    name = "ck.gemm2"
    prefixes = (kernel_names.CK2STAGES,)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return ck_2stage_caps()

    def owns(self, kernel) -> bool:
        return kernel is None or (
            kernel.family == kernel_names.CK2STAGES and kernel.stage == 2
        )

    def reads(self, p):
        return (Buf("sorted", None), Buf("inter", p.act_in))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def build(self, p, params):
        return functools.partial(
            aiter.ck_moe_stage2_fwd,
            kernelName=params.kernel,
            activation=p.activation.activation,
            quant_type=p.quant_type,
            use_non_temporal_load=params.nt,
        )

    def param_space(self, p):
        return ck_param_space(self, p, 2)
