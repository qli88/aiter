# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP activation quantization before GEMM1 and between the GEMMs (`aiter.get_hip_quant`).

Covers the quant types whose activations are quantized by a standalone kernel and the
intermediate is quantized after GEMM1 (the generic branches of the legacy
`fused_moe_2stages`), and its `per_1x32` branches (`quant_a1_mxfp4`, `quant_inter_mxfp4`),
which quantize into the sorted layout.
"""

import functools
import os

import torch

from aiter import (
    ActivationType,
    QuantType,
    dtypes,
    fused_dynamic_mxfp4_quant_moe_sort,
    fused_dynamic_mxfp8_quant_moe_sort,
    mxfp4_moe_sort_fwd,
)
from aiter import get_hip_quant as get_quant
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from .base import Cap, Params, Segment
from .ck import blockscale_fp8

_FEATURES = frozenset(
    {"ep_mask", "num_local_tokens", "doweight_stage1", "act_scales", "bias1", "bias2"}
)
_EXTRAS = frozenset({"disable_inline_sort", "tuned_table"})
# Debug: unit scales instead of the mxfp8 quant (the legacy module reads the same flag).
_MOE_A8W4_BYPASS_QUANT = os.environ.get("AITER_MOE_A8W4_BYPASS_QUANT", "0") == "1"


class QuantA1(Segment):
    name = "quant.a1"
    covers = (Op.QUANT_A1, Op.QUANT_A1)
    capabilities = (Cap(features=_FEATURES, pads=True, extras=_EXTRAS, shared=True),)
    scales = "plain"

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        return Buf("tokens", p.act_in, scales=self.scales)

    def default_params(self, p):
        return Params() if self.reach == "ranked" else None

    def run(self, quant_type, hidden_states, a1_scale, q_dtype_a, num_local_tokens):
        """`(a1, a1_scale)`: quantize unless the activations already are `q_dtype_a`
        (then they and their scales pass unchanged, whatever `scales` says)."""
        if hidden_states.dtype != q_dtype_a:
            quant_func = get_quant(quant_type)
            if self.scales == "transposed":
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


class QuantA1Transposed(QuantA1):
    """`quant.a1` writing the 1x128 scales transposed, for `asm.stage1_1x128`."""

    name = "quant.a1_transposed"
    reach = "tuned"
    scales = "transposed"

    @property
    def capabilities(self):
        return (
            Cap(
                act_in=frozenset({blockscale_fp8()}),
                features=_FEATURES,
                pads=True,
                extras=_EXTRAS,
                shared=True,
            ),
        )


def _keeps_bf16(dtype, w1_dtype, q_dtype_a, activation, ksplit, is_shuffled):
    """The legacy condition under which neither GEMM input is quantized: bf16/fp16
    activations into fp4 weights with Swiglu, SiTUv2 or Relu2, or fp4 activations into
    a split-K GEMM on preshuffled weights (both quantize inline)."""
    return (
        dtype in [dtypes.bf16, dtypes.fp16]
        and w1_dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and activation
            in (ActivationType.Swiglu, ActivationType.Situv2, ActivationType.Relu2)
            or (ksplit > 1 and is_shuffled)
        )
    )


def quant_a1_mxfp4(
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
    *,
    prequant,
):
    """`(a1, a1_scale)` for `per_1x32`: the legacy `fused_moe_2stages` branches, which
    quantize into (or sort the scales of) the sorted layout of the stage 1 kernel.
    `prequant` False: the stage 1 quantizes the activations itself (legacy
    `metadata.prequant` after its MXMOE override), so they pass unchanged."""
    token_num = hidden_states.shape[0]
    if not prequant:
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
    if (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1_dtype in (dtypes.fp4x2, dtypes.fp8)
    ):
        # mxfp8 activations + mxfp4 weights (a8w4) or mxfp8 weights.
        if _MOE_A8W4_BYPASS_QUANT:
            # Debug bypass: skip real quant, feed unit scales.
            a1 = hidden_states.to(dtypes.fp8)
            return a1, torch.ones(
                [sorted_ids.shape[0], a1.shape[-1] // 32],
                dtype=dtypes.fp8_e8m0,
                device=a1.device,
            )
        if hidden_states.dtype == dtypes.fp8 and a1_scale is not None:
            # Already mxfp8 (e.g. from a quantizing dispatch): only sort the scale.
            return hidden_states, mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
        return fused_dynamic_mxfp8_quant_moe_sort(
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
    if w1_dtype == dtypes.i4x2:
        # a16wi4: bf16 activations into int4 weights, not quantized.
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


def quant_inter_mxfp4(
    a2,
    a2_scale,
    a1_scale,
    metadata,
    q_dtype_a,
    dtype,
    w1_dtype,
    activation,
    is_shuffled,
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
    """`(a2, a2_scale)` for `per_1x32` when stage 1 returned a plain intermediate: the
    legacy `fused_moe_2stages` branches after the fused-output ones."""
    if _keeps_bf16(
        dtype, w1_dtype, q_dtype_a, activation, metadata.ksplit, is_shuffled
    ):
        return a2, None
    if (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1_dtype in (dtypes.fp4x2, dtypes.fp8)
    ):
        # a8w4 / mxfp8: quantize the stage 1 output to mxfp8 for stage 2.
        if _MOE_A8W4_BYPASS_QUANT:
            return a2.to(dtypes.fp8), a1_scale
        a2, a2_scale = fused_dynamic_mxfp8_quant_moe_sort(
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
    if w1_dtype == dtypes.i4x2:
        # a16wi4: the stage 1 output is bf16, not quantized.
        return a2, None
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


class QuantInter(Segment):
    name = "quant.inter"
    covers = (Op.QUANT_INTER, Op.QUANT_INTER)
    capabilities = (Cap(features=_FEATURES, pads=True, extras=_EXTRAS, shared=True),)

    def reads(self, p):
        # Any unquantized intermediate: `asm.stage1` writes bf16 even for fp16 outputs.
        return (Buf("inter", None),)

    def produces(self, p, params):
        return Buf("inter", p.act_in)

    def default_params(self, p):
        return Params()

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
