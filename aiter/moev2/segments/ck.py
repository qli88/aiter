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
from .base import Cap, Segment


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


def ck_2stage_caps():
    """The `gen_instances.py` build list: f8 and i8 (and f8 x i4) per-tensor/per-token,
    f8 1x128, and unquantized f16/b16."""
    return (
        a8w8_per_tensor_caps()
        + a8w8_per_token_caps()
        + a8w8_blockscale_caps()
        + int8_per_tensor_caps()
        + int8_per_token_caps()
        + int4_per_token_caps()
        + a16w16_caps()
    )


def ck_stage1_caps():
    """Stage 1 always splits w1 into gate and up (`N = w1.size(1) / 2`): g1u1 only."""
    return tuple(replace(cap, g1u1=frozenset({True})) for cap in ck_2stage_caps())


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
