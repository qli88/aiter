# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK 2-stage kernels (`moe_ck2stages_gemm1_*` / `moe_ck2stages_gemm2_*`)."""

import functools

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


def a8w8_per_tensor_caps():
    fp8 = frozenset({per_tensor_fp8()})
    return (
        Cap(
            arch=frozenset({"gfx942", "gfx950"}),
            act_in=fp8,
            w1=fp8,
            w2=fp8,
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "gelu"}),
            gate_mode=frozenset({"separated"}),
            features=frozenset(
                {"ep_mask", "num_local_tokens", "doweight_stage1", "act_scales"}
            ),
        ),
    )


class CkGemm1(Segment):
    name = "ck.gemm1"
    prefixes = (kernel_names.CK2STAGES,)
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        return a8w8_per_tensor_caps()

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
        return a8w8_per_tensor_caps()

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
