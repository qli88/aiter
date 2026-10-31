# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP activation quantization before GEMM1 and between the GEMMs (`aiter.get_hip_quant`).

Covers the quant types whose activations are quantized by a standalone kernel and the
intermediate is quantized after GEMM1 (the generic branches of the legacy
`fused_moe_2stages`), and its MXFP4 (`per_1x32` fp4) branches, which quantize into the
sorted layout.
"""

import functools

from aiter import (
    ActivationType,
    QuantType,
    dtypes,
    fused_dynamic_mxfp4_quant_moe_sort,
    mxfp4_moe_sort_fwd,
)
from aiter import get_hip_quant as get_quant
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from .base import Cap, Segment

_FEATURES = frozenset({"ep_mask", "num_local_tokens", "doweight_stage1", "act_scales"})
_EXTRAS = frozenset({"disable_inline_sort"})


class QuantA1(Segment):
    name = "quant.a1"
    covers = (Op.QUANT_A1, Op.QUANT_A1)
    capabilities = (Cap(features=_FEATURES, extras=_EXTRAS),)

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        return Buf("tokens", p.act_in)

    def run(
        self,
        quant_type,
        hidden_states,
        a1_scale,
        q_dtype_a,
        num_local_tokens,
        transpose_scale=False,
    ):
        """`(a1, a1_scale)`: quantize unless the activations already are `q_dtype_a`."""
        if hidden_states.dtype != q_dtype_a:
            quant_func = get_quant(quant_type)
            if transpose_scale:
                quant_func = functools.partial(quant_func, transpose_scale=True)
            return quant_func(
                hidden_states,
                scale=a1_scale,
                quant_dtype=q_dtype_a,
                num_rows=num_local_tokens,
            )
        assert (
            a1_scale is not None or quant_type == QuantType.No
        ), "a1_scale must be provided for quantized input for fused_moe"
        return hidden_states, a1_scale

    def run_mxfp4(
        self,
        metadata,
        hidden_states,
        a1_scale,
        q_dtype_a,
        dtype,
        w1_dtype,
        activation,
        gate_mode,
        is_shuffled,
        sorted_ids,
        num_valid_ids,
        model_dim,
        topk,
        block_size_M,
        num_local_tokens,
        sorted_weights,
        routing_num_experts,
    ):
        """`(a1, a1_scale)` for MXFP4 weights: the `per_1x32` branches of the legacy
        `fused_moe_2stages`, which sort the scales for the stage 1 kernel."""
        token_num = hidden_states.shape[0]
        if not metadata.prequant:
            return hidden_states, None
        if (
            dtype in [dtypes.bf16, dtypes.fp16]
            and w1_dtype == dtypes.fp4x2
            and (
                q_dtype_a in [dtypes.bf16, dtypes.fp16]
                and (
                    activation
                    in (
                        ActivationType.Swiglu,
                        ActivationType.Situv2,
                        ActivationType.Relu2,
                    )
                    or gate_mode == GateMode.INTERLEAVE
                )
                or (q_dtype_a in [dtypes.fp4x2] and metadata.ksplit > 1 and is_shuffled)
            )
        ):
            return hidden_states.to(dtype), None
        if hidden_states.dtype == dtypes.fp4x2 and a1_scale is not None:
            # Already fp4 (e.g. from an FP4 dispatch): only sort the scale.
            return hidden_states, mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
        return fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M,
            num_rows=num_local_tokens,
            sorted_weights=sorted_weights,
            num_experts_upper_bound=routing_num_experts,
        )


class QuantInter(Segment):
    name = "quant.inter"
    covers = (Op.QUANT_INTER, Op.QUANT_INTER)
    capabilities = (Cap(features=_FEATURES, extras=_EXTRAS),)

    def reads(self, p):
        # Any unquantized intermediate: `asm.stage1` writes bf16 even for fp16 outputs.
        return (Buf("inter", None),)

    def produces(self, p, params):
        return Buf("inter", p.act_in)

    def run(
        self,
        quant_type,
        a2,
        a2_scale,
        q_dtype_a,
        num_local_tokens,
        token_num,
        topk,
        inter_dim,
    ):
        a2, a2_scale = get_quant(quant_type)(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        return a2.view(token_num, topk, inter_dim), a2_scale

    def run_mxfp4(
        self,
        a2,
        sorted_ids,
        num_valid_ids,
        token_num,
        topk,
        inter_dim,
        block_size_M,
        num_local_tokens,
        sorted_weights,
        routing_num_experts,
    ):
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2.view(-1, inter_dim),
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M,
            num_rows=num_local_tokens,
            sorted_weights=sorted_weights,
            num_experts_upper_bound=routing_num_experts,
        )
        return a2.view(token_num, topk, -1), a2_scale
