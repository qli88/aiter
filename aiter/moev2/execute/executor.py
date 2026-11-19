# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run a resolved plan on one call.

Statement for statement this is the part of the legacy `_fused_moe_impl` and
`fused_moe_2stages` that the plan's segments reach, in the same order: the same
validation, buffers, kernel launches and benchmark hooks.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import functools
from dataclasses import dataclass, replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.moe_common import (
    GateMode,
    _flydsl_stage2_fp8_enabled,
    _mxfp4_intermediate_enabled,
)
from aiter.ops.flydsl.mxfp4_kname import _parse_mxfp4_g1_kname, parse_g2_kname_any

from .. import hooks
from ..dispatch.ops import Op
from ..policy import _normalize_mxfp4_activation_params
from ..segments.flydsl import (
    _flydsl_route_output_nbytes,
    _flydsl_stage1_wrapper,
    _flydsl_stage2_wrapper,
    _flydsl_v2_stage2_wrapper,
    _is_flydsl_stage2,
    _stage2_runtime_uses_route_reduce,
    _validate_flydsl_route_output_nbytes,
)
from ..segments.mxmoe import (
    _is_mxfp4_inline_sort,
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_a4w4_stage2_fw,
)
from ..segments.sort import (
    AUX_SORT_OPUS,
    _fused_decode_sort_quant,
    _is_fused_decode_sort_quant_enabled,
)
from ..shapes import _is_row_major_2d, get_inter_dim
from .output import (
    _return_output,
    _validate_output_buffer_metadata,
    _validate_output_buffer_no_overlap,
)


@dataclass
class Call:
    """A `fused_moe_` call after the legacy prologue (enums decoded, shapes derived)."""

    args: dict
    activation: ActivationType
    quant_type: QuantType
    gate_mode: GateMode
    block_size_M: int | None
    M: int
    topk: int
    E: int
    model_dim: int
    inter_dim: int
    isG1U1: bool
    global_E: int
    dtype: torch.dtype

    def __getitem__(self, name):
        return self.args[name]


def prologue(call, quant_remap, *, prepare=True) -> Call:
    """The validation and derived values of `_fused_moe_impl`, up to dispatch.

    ``prepare=False`` builds a side-effect-free dispatch preview. A path that accepts the
    preview calls the full prologue before execution; a declined call reaches legacy
    without materializing a strided activation or validating its output buffer twice.
    """
    activation = ActivationType(call["activation"])
    quant_type = QuantType(call["quant_type"])
    gate_mode = GateMode(call["gate_mode"])
    block_size_M = call["block_size_M"]
    if block_size_M == -1:
        block_size_M = None
    hidden_states, w1, w2 = call["hidden_states"], call["w1"], call["w2"]
    M, topk = call["topk_ids"].shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if (
        prepare
        and not hidden_states.is_contiguous()
        and not _is_row_major_2d(hidden_states)
    ):
        hidden_states = hidden_states.contiguous()
        call = {**call, "hidden_states": hidden_states}

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]
    global_E = E
    if call["expert_mask"] is not None:
        global_E = call["expert_mask"].numel()
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    assert dtype in [
        dtypes.fp16,
        dtypes.bf16,
    ], f"Fused_moe unsupported out dtype: {dtype}"
    if prepare:
        output = call["output"]
        _validate_output_buffer_metadata(
            output, (M, model_dim), dtype, hidden_states.device
        )
        _validate_output_buffer_no_overlap(output, hidden_states)
    quant_type = quant_remap.get(quant_type, quant_type)
    return Call(
        call,
        activation,
        quant_type,
        gate_mode,
        block_size_M,
        M,
        topk,
        E,
        model_dim,
        inter_dim,
        isG1U1,
        global_E,
        dtype,
    )


def run_whole_graph(c: Call, metadata, block_size_M, q_dtype_a):
    request = FusedMoeRequest(
        hidden_states=c["hidden_states"],
        w1=c["w1"],
        w2=c["w2"],
        topk_weight=c["topk_weight"],
        topk_ids=c["topk_ids"],
        expert_mask=c["expert_mask"],
        activation=c.activation,
        quant_type=c.quant_type,
        doweight_stage1=c["doweight_stage1"],
        w1_scale=c["w1_scale"],
        w2_scale=c["w2_scale"],
        a1_scale=c["a1_scale"],
        a2_scale=c["a2_scale"],
        block_size_m=(int(block_size_M) if block_size_M is not None else None),
        ksplit=metadata.ksplit,
        num_local_tokens=c["num_local_tokens"],
        moe_sorting_dispatch_policy=c["moe_sorting_dispatch_policy"],
        dtype=c.dtype,
        hidden_pad=c["hidden_pad"],
        intermediate_pad=c["intermediate_pad"],
        bias1=c["bias1"],
        bias2=c["bias2"],
        swiglu_limit=c["swiglu_limit"],
        beta=c["beta"],
        linear_beta=c["linear_beta"],
        gate_mode=c.gate_mode,
        q_dtype_a=q_dtype_a,
        q_dtype_w=c["w1"].dtype,
        w1_global_scale=c["w1_global_scale"],
        w2_global_scale=c["w2_global_scale"],
    )
    full_call = functools.partial(metadata.full_impl, request)
    hooks.record_kernel("whole_graph", full_call)
    full_output = full_call()
    return _return_output(full_output, c["output"])


def _e8m0_view(w, scale):
    # Only one-byte packed scales may be reinterpreted as E8M0.
    return (
        scale.view(dtypes.fp8_e8m0)
        if w.dtype in (dtypes.fp4x2, dtypes.fp8)
        and scale is not None
        and scale.element_size() == 1
        else scale
    )


def run_two_stage(c: Call, metadata, plan, q_dtype_a, after_sort=None):
    """Sort, quantize, GEMM1, quantize, GEMM2: `_fused_moe_impl` + `fused_moe_2stages`.

    `after_sort()` runs where `fused_moe_2stages` starts, after the sort.
    """
    if not c["hidden_states"].is_contiguous():
        c = replace(
            c, args={**c.args, "hidden_states": c["hidden_states"].contiguous()}
        )
    block_size_M = metadata.block_m if c.block_size_M is None else c.block_size_M
    if metadata.full_impl is not None:
        return run_whole_graph(c, metadata, block_size_M, q_dtype_a)
    # Ensure block_size_M is int (metadata.block_m from CSV may be float)
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    sort = plan.segment("sort").segment
    quant_a1 = plan.segment("quant.a1").segment
    gemm1 = plan.starting(Op.GEMM1).segment
    # A stage 1 that also quantizes its output writes into its own buffer layout.
    fused_inter_quant = gemm1.covers[1] == Op.QUANT_INTER
    quant_inter = None if fused_inter_quant else plan.segment("quant.inter").segment

    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    topk_ids, topk_weight = c["topk_ids"], c["topk_weight"]
    num_local_tokens = c["num_local_tokens"]
    output = c["output"]
    doweight_stage1 = c["doweight_stage1"]
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = sort.run(
        topk_ids,
        topk_weight,
        c.global_E,
        c.model_dim,
        c.dtype,
        block_size_M,
        c["expert_mask"],
        num_local_tokens,
        c["moe_sorting_dispatch_policy"],
        output,
    )
    if after_sort is not None:
        after_sort()

    token_num, _ = hidden_states.shape
    topk = c.topk
    inter_dim = c.inter_dim
    dtype = moe_out.dtype
    device = hidden_states.device
    if moe_out.numel() == 0:
        moe_out = (
            output
            if output is not None
            else torch.empty((token_num, c.model_dim), dtype=dtype, device=device)
        )
    a1, a1_scale = quant_a1.run(
        c.quant_type,
        hidden_states,
        c["a1_scale"],
        q_dtype_a,
        num_local_tokens,
        transpose_scale=getattr(gemm1, "transpose_a1_scale", False),
    )
    if fused_inter_quant:
        a2 = gemm1.out_buffer(a1, a1_scale, token_num, topk, inter_dim, q_dtype_a)
    else:
        # Stage1 does not write slots with topk_ids == -1 or masked out by
        # expert_mask, but dynamic per-tensor a2 quant reads the whole buffer,
        # so stale memory in those slots can set the scale.
        _a2_alloc = (
            torch.zeros
            if c.quant_type == QuantType.per_Tensor and c["a2_scale"] is None
            else torch.empty
        )
        a2 = _a2_alloc(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=device,
        )
    _stage1_call = functools.partial(
        metadata.stage1,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        None if metadata.fuse_quant else a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=_e8m0_view(w1, c["w1_scale"]),
        sorted_weights=sorted_weights if doweight_stage1 else None,
    )
    hooks.record_kernel("stage1", _stage1_call)
    a2 = _stage1_call()
    if fused_inter_quant:
        a2, a2_scale = gemm1.split_out(a2, a1, a1_scale, token_num, topk, inter_dim)
    else:
        a2, a2_scale = quant_inter.run(
            c.quant_type,
            a2,
            c["a2_scale"],
            q_dtype_a,
            num_local_tokens,
            token_num,
            topk,
            inter_dim,
        )
    _stage2_call = functools.partial(
        metadata.stage2,
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=_e8m0_view(w2, c["w2_scale"]),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
    )
    hooks.record_kernel("stage2", _stage2_call)
    _stage2_call()
    return _return_output(moe_out, output)


def run_mxfp4(c: Call, metadata, plan, q_dtype_a, after_sort=None):
    """`run_two_stage` for MXFP4 weights: the `per_1x32` branches of `_fused_moe_impl`
    and `fused_moe_2stages` (route-reduce check, auxiliary or fused decode sort, sorted
    quantization, the FlyDSL and MXMOE stage arguments)."""
    if not _is_mxfp4_inline_sort(metadata) and not c["hidden_states"].is_contiguous():
        c = replace(
            c, args={**c.args, "hidden_states": c["hidden_states"].contiguous()}
        )
    block_size_M = metadata.block_m if c.block_size_M is None else c.block_size_M
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    sort = plan.starting(Op.SORT).segment
    quant_a1 = plan.segment("quant.a1").segment
    quant_inter = plan.segment("quant.inter")

    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    topk_ids, topk_weight = c["topk_ids"], c["topk_weight"]
    expert_mask = c["expert_mask"]
    num_local_tokens = c["num_local_tokens"]
    output = c["output"]
    doweight_stage1 = c["doweight_stage1"]
    M, topk, model_dim, dtype = c.M, c.topk, c.model_dim, c.dtype
    stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
    stage2_func = getattr(metadata.stage2, "func", metadata.stage2)

    runtime_stage2_reduce = _stage2_runtime_uses_route_reduce(
        metadata.stage2, M, model_dim
    )
    if _is_flydsl_stage2(metadata.stage2) and runtime_stage2_reduce:
        output_element_size = torch.empty((), dtype=dtype).element_size()
        _validate_flydsl_route_output_nbytes(
            _flydsl_route_output_nbytes(
                metadata.stage2, M, topk, model_dim, output_element_size
            )
        )
    prequantized_a1 = None
    prequantized_a1_scale = None
    m_indices = None
    reverse_sorted = None
    fused_sort_quant = not metadata.output_aux and _is_fused_decode_sort_quant_enabled(
        hidden_states,
        topk_ids,
        topk_weight,
        global_experts=c.global_E,
        block_m=block_size_M,
        quant_type=c.quant_type,
        q_dtype_a=q_dtype_a,
        q_dtype_w=w1.dtype,
        activation=c.activation,
        is_g1u1=c.isG1U1,
        expert_mask=expert_mask,
        need_local_topk_ids=False,
        run_1stage=metadata.run_1stage,
        stage1_is_flydsl=stage1_func is _flydsl_stage1_wrapper,
        hidden_pad=c["hidden_pad"],
        ksplit=metadata.ksplit,
    )
    if metadata.output_aux:
        stage2_kwargs = metadata.stage2.keywords
        kernel_name2 = stage2_kwargs.get("kernelName2") or stage2_kwargs.get(
            "kernelName", ""
        )
        g2 = parse_g2_kname_any(kernel_name2)
        if expert_mask is not None and not g2["v2"]:
            raise NotImplementedError(
                f"MXMOE GEMM2 {kernel_name2!r} does not support expert-parallel; "
                "use a flydsl_moe2_layout_ GEMM2."
            )
        atomic = g2["atomic"]
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            m_indices,
            reverse_sorted,
        ) = sort.run(
            topk_ids,
            topk_weight,
            c.global_E,
            model_dim,
            dtype,
            block_size_M,
            expert_mask,
            num_local_tokens,
            output_aux=metadata.output_aux,
            accumulate=atomic and not runtime_stage2_reduce,
            output=output,
        )[
            :7
        ]
    elif fused_sort_quant:
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            prequantized_a1,
            prequantized_a1_scale,
        ) = _fused_decode_sort_quant(
            hidden_states,
            topk_ids,
            topk_weight,
            model_dim=model_dim,
            global_experts=c.global_E,
            block_m=block_size_M,
            moebuf_dtype=dtype,
            accumulate=not runtime_stage2_reduce,
            compact_scale=False,
            num_local_tokens=num_local_tokens,
        )
    else:
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            sort.run(
                topk_ids,
                topk_weight,
                c.global_E,
                model_dim,
                dtype,
                block_size_M,
                expert_mask,
                num_local_tokens,
                c["moe_sorting_dispatch_policy"],
                output,
                accumulate=not runtime_stage2_reduce,
            )
        )
    if after_sort is not None:
        after_sort()

    token_num, _ = hidden_states.shape
    inter_dim = c.inter_dim
    moe_out = moe_buf
    dtype = moe_out.dtype
    device = hidden_states.device
    if moe_out.numel() == 0:
        moe_out = (
            output
            if output is not None
            else torch.empty((token_num, model_dim), dtype=dtype, device=device)
        )
    is_shuffled = getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
    config_situ_beta, config_situ_linear_beta, normalized_swiglu_limit = (
        _normalize_mxfp4_activation_params(
            c.activation, c["beta"], c["linear_beta"], c["swiglu_limit"]
        )
    )
    if (
        stage1_func is _mxfp4_a4w4_stage1_fw
        and metadata.output_aux == AUX_SORT_OPUS
        # block_m 16 is the only inline-quant a4w4 GEMM1: it reads raw bf16.
        and int(metadata.block_m) != 16
        and _parse_mxfp4_g1_kname(metadata.stage1.keywords["kernelName1"])["a_dtype"]
        == "fp4"
    ):
        metadata = replace(metadata, prequant=True)
    if prequantized_a1 is not None:
        a1, a1_scale = prequantized_a1, prequantized_a1_scale
    else:
        a1, a1_scale = quant_a1.run_mxfp4(
            metadata,
            hidden_states,
            c["a1_scale"],
            q_dtype_a,
            dtype,
            w1.dtype,
            c.activation,
            c.gate_mode,
            is_shuffled,
            sorted_ids,
            num_valid_ids,
            model_dim,
            topk,
            block_size_M,
            num_local_tokens,
            sorted_weights,
            c.global_E,
        )
    a2 = torch.empty((token_num, topk, inter_dim), dtype=dtype, device=device)
    extra_stage1_args = {}
    extra_stage2_args = {}
    if stage1_func is _flydsl_stage1_wrapper:
        extra_stage1_args["swiglu_limit"] = c["swiglu_limit"]
    elif stage1_func is _mxfp4_a4w4_stage1_fw:
        extra_stage1_args["swiglu_limit"] = normalized_swiglu_limit
    if stage1_func is _flydsl_stage1_wrapper and metadata.skip_inter_quant:
        extra_stage1_args["v2_output_layout"] = True
    if stage1_func is _flydsl_stage1_wrapper:
        extra_stage1_args["situ_beta"] = config_situ_beta
        extra_stage1_args["situ_linear_beta"] = config_situ_linear_beta
    elif stage1_func is _mxfp4_a4w4_stage1_fw:
        # MXMOE's activation comes from its name and betas enter its compile key.
        extra_stage1_args["situ_beta"] = config_situ_beta
        extra_stage1_args["situ_linear_beta"] = config_situ_linear_beta
    if (
        stage2_func
        in (_flydsl_stage2_wrapper, _flydsl_v2_stage2_wrapper, _mxfp4_a4w4_stage2_fw)
        and expert_mask is not None
    ):
        extra_stage2_args["expert_mask"] = expert_mask
        extra_stage2_args["topk_ids"] = topk_ids
    stage2_keywords = getattr(metadata.stage2, "keywords", None) or {}
    if stage2_keywords.get("skip_dropped_routes"):
        # The reduction reads topk_ids to leave out dropped routes.
        extra_stage2_args["topk_ids"] = topk_ids
    uses_flydsl_v2_stage2 = stage2_func is _flydsl_v2_stage2_wrapper or (
        stage2_func is _mxfp4_a4w4_stage2_fw
        and str(stage2_keywords.get("kernelName2", "")).startswith(
            "flydsl_moe2_layout_"
        )
    )
    if stage2_func in (
        _flydsl_stage2_wrapper,
        _flydsl_v2_stage2_wrapper,
        _mxfp4_a4w4_stage2_fw,
    ):
        extra_stage2_args["use_valid_token_count"] = num_local_tokens is not None
    if (
        uses_flydsl_v2_stage2
        and not doweight_stage1
        and (_flydsl_stage2_fp8_enabled() or _mxfp4_intermediate_enabled())
    ):
        # FP8/FP4 route-output reduction applies the route weights after GEMM2.
        extra_stage2_args["topk_weights"] = topk_weight
    if m_indices is not None:
        extra_stage1_args["m_indices"] = m_indices
        extra_stage1_args["moe_buf"] = moe_buf
        extra_stage2_args["reverse_sorted"] = reverse_sorted
    _stage1_call = functools.partial(
        metadata.stage1,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        None if metadata.fuse_quant else a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=_e8m0_view(w1, c["w1_scale"]),
        sorted_weights=sorted_weights if doweight_stage1 else None,
        **extra_stage1_args,
    )
    hooks.record_kernel("stage1", _stage1_call)
    a2 = _stage1_call()
    a2_scale = c["a2_scale"]
    if isinstance(a2, tuple) and (
        stage1_func is _mxfp4_a4w4_stage1_fw
        or metadata.skip_inter_quant
        or m_indices is not None
    ):
        a2, a2_scale = a2[0], a2[1]
    elif metadata.fuse_quant == "fp4" and isinstance(a2, tuple):
        a2_raw, a2_scale = a2[0], a2[1]
        _fp4_bytes = token_num * topk * (inter_dim // 2)
        a2 = (
            a2_raw.view(-1)
            .view(torch.uint8)[:_fp4_bytes]
            .view(dtypes.fp4x2)
            .reshape(token_num, topk, -1)
        )
    elif metadata.fuse_quant == "fp8" and isinstance(a2, tuple):
        a2, a2_scale = a2[0], a2[1]
        a2 = a2.view(token_num, topk, -1)
    elif (
        dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and c.activation
            in (ActivationType.Swiglu, ActivationType.Situv2, ActivationType.Relu2)
            or (metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a2_scale = None
    else:
        a2, a2_scale = quant_inter.segment.run_mxfp4(
            a2,
            sorted_ids,
            num_valid_ids,
            token_num,
            topk,
            inter_dim,
            block_size_M,
            num_local_tokens,
            sorted_weights,
            c.global_E,
        )
    _stage2_call = functools.partial(
        metadata.stage2,
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=_e8m0_view(w2, c["w2_scale"]),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
        **extra_stage2_args,
    )
    hooks.record_kernel("stage2", _stage2_call)
    _stage2_call()
    return _return_output(moe_out, output)
