# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run a resolved plan on one call.

Statement for statement this is the part of the legacy `_fused_moe_impl` and
`fused_moe_2stages` that the plan's segments reach, in the same order: the same
validation, buffers, kernel launches and benchmark hooks.
"""

import functools
from dataclasses import dataclass, replace

import torch

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.flydsl.mxfp4_kname import parse_g2_kname_any
from aiter.ops.opus import moe_stage2_a8w4 as _opus_a8w4
from aiter.ops.opus.moe_stage1_a8w4 import (
    opus_a8w4_stage1_wrapper as _opus_a8w4_stage1_wrapper,
)

from .. import hooks
from ..dispatch.ops import Op
from ..policy import _normalize_mxfp4_activation_params
from ..segments.ck_tile import cktile_moe_stage1, cktile_moe_stage2
from ..segments.flydsl import (
    _can_reroute_mxfp4_to_flydsl,
    _flydsl_route_output_nbytes,
    _flydsl_stage1_wrapper,
    _flydsl_stage2_fp8_enabled,
    _flydsl_stage2_wrapper,
    _flydsl_v2_stage2_wrapper,
    _is_flydsl_stage2,
    _needs_swiglu_bias_support,
    _normalize_bias_for_kernel,
    _stage2_runtime_uses_route_reduce,
    _validate_flydsl_route_output_nbytes,
)
from ..segments.mxmoe import _mxfp4_a4w4_stage1_fw, _mxfp4_a4w4_stage2_fw
from ..segments.opus import _opus_stage2_fp8_enabled
from ..segments.quant import quant_a1_mxfp4, quant_inter_mxfp4
from ..segments.sort import (
    AUX_SORT_OPUS,
    _fused_decode_sort_quant,
    _is_fused_decode_sort_quant_enabled,
)
from ..shapes import get_inter_dim
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


def prologue(call, quant_remap) -> Call:
    """The validation and derived values of `_fused_moe_impl`, up to the dispatch key."""
    activation = ActivationType(call["activation"])
    quant_type = QuantType(call["quant_type"])
    gate_mode = GateMode(call["gate_mode"])
    block_size_M = call["block_size_M"]
    if block_size_M == -1:
        block_size_M = None
    hidden_states, w1, w2 = call["hidden_states"], call["w1"], call["w2"]
    M, topk = call["topk_ids"].shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

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
    full_output = metadata.full_impl(
        FusedMoeRequest(
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
        )
    )
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


def run_one_stage(c: Call, metadata, plan, q_dtype_a):
    """Sort, then the 1-stage kernel: `_fused_moe_impl` with `metadata.run_1stage`."""
    block_size_M = metadata.block_m if c.block_size_M is None else c.block_size_M
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    assert not metadata.flat or get_gfx() in (
        "gfx942",
        "gfx950",
    ), f"FLAT fmoe asm kernels are gfx942/gfx950-only; refusing to launch on {get_gfx()}. "
    sort = plan.segments[0].segment
    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    topk_ids = c["topk_ids"]
    num_local_tokens = c["num_local_tokens"]
    output = c["output"]
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = sort.run(
        topk_ids,
        c["topk_weight"],
        c.global_E,
        c.model_dim,
        c.dtype,
        block_size_M,
        c["expert_mask"],
        num_local_tokens,
        c["moe_sorting_dispatch_policy"],
        None if metadata.flat else output,
    )
    _stage1_call = functools.partial(
        metadata.stage1,
        hidden_states,
        w1,
        w2,
        c.topk,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        c.isG1U1,
        block_size_M,
        q_dtype_a=q_dtype_a,
        q_dtype_w=w1.dtype,
        w1_scale=c["w1_scale"],
        w2_scale=c["w2_scale"],
        a1_scale=c["a1_scale"],
        a2_scale=c["a2_scale"],
        num_local_tokens=num_local_tokens,
        M=c.M,
        device=topk_ids.device,
        doweight_stage1=c["doweight_stage1"],
    )
    hooks.record_kernel("stage1", _stage1_call)
    return _return_output(_stage1_call(), output)


def run_two_stage(c: Call, metadata, plan, q_dtype_a, after_sort=None):
    """Sort, quantize, GEMM1, quantize, GEMM2: `_fused_moe_impl` + `fused_moe_2stages`.

    `after_sort()` runs where `fused_moe_2stages` starts, after the sort.
    """
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
        a2 = torch.empty(
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


def a16w4_unsupported(c: Call, q_dtype_a) -> str | None:
    """Why legacy `_fused_moe_impl` raises NotImplementedError for an a16w4 call (bf16
    activations into MXFP4 weights, SiTUv2 or Swiglu re-routed to FlyDSL) before the
    dispatch, or None."""
    w1, w2 = c["w1"], c["w2"]
    is_situv2 = (
        c.quant_type == QuantType.per_1x32
        and w1.dtype == dtypes.fp4x2
        and q_dtype_a == dtypes.bf16
        and c.activation == ActivationType.Situv2
    )
    is_swiglu_rerouted = (
        c.inter_dim % 256 != 0
        and q_dtype_a == dtypes.bf16
        and _can_reroute_mxfp4_to_flydsl(
            model_dim=c.model_dim,
            inter_dim=c.inter_dim,
            dtype=c.dtype,
            q_dtype_a=q_dtype_a,
            q_dtype_w=w1.dtype,
            q_type=c.quant_type,
            activation=c.activation,
            is_shuffled=getattr(w1, "is_shuffled", False)
            and getattr(w2, "is_shuffled", False),
            use_g1u1=c.isG1U1,
            doweight_stage1=c["doweight_stage1"],
            gate_mode=c.gate_mode,
            gate_up_interleave=getattr(w1, "gate_up_interleave", None),
        )
    )
    if not (is_situv2 or is_swiglu_rerouted):
        return None
    is_shuffled = getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
    for bad, why in (
        (
            get_gfx() not in ("gfx942", "gfx950"),
            f"requires the FlyDSL kernel on CDNA gfx942/gfx950 (gfx={get_gfx()!r})",
        ),
        (c["bias1"] is not None or c["bias2"] is not None, "per-expert bias"),
        (c["expert_mask"] is not None, "expert-parallel masking"),
        (
            not (is_shuffled and c.isG1U1 and not c["doweight_stage1"]),
            "non-preshuffled / non-g1u1 weights or doweight_stage1=True",
        ),
    ):
        if bad:
            return why
    return None


def run_mxfp4(c: Call, metadata, plan, q_dtype_a, after_sort=None):
    """`run_two_stage` for `per_1x32`: the legacy `_fused_moe_impl` after the dispatch and
    `fused_moe_2stages`, branch for branch (route-reduce check, auxiliary or fused decode
    sort, local topk ids for split-K bias under EP, sorted quantization, the FlyDSL,
    MXMOE, Opus and CK-Tile stage arguments)."""
    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
    stage2_func = getattr(metadata.stage2, "func", metadata.stage2)
    if (
        w1.dtype == dtypes.fp4x2
        and c.inter_dim % 256 != 0
        and stage2_func is cktile_moe_stage2
    ):
        raise NotImplementedError(
            "No safe MXFP4 MoE backend for this shape and layout: CK-Tile "
            f"stage2 mis-indexes scales at inter_dim={c.inter_dim}, and the "
            "FlyDSL fallback cannot accept this invocation."
        )
    block_size_M = metadata.block_m if c.block_size_M is None else c.block_size_M
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    sort = plan.starting(Op.SORT).segment

    topk_ids, topk_weight = c["topk_ids"], c["topk_weight"]
    expert_mask = c["expert_mask"]
    num_local_tokens = c["num_local_tokens"]
    output = c["output"]
    doweight_stage1 = c["doweight_stage1"]
    M, topk, model_dim, dtype = c.M, c.topk, c.model_dim, c.dtype
    activation, quant_type = c.activation, c.quant_type
    need_bias_support = _needs_swiglu_bias_support(dtype, quant_type)
    need_local_topk_ids = (
        not metadata.run_1stage
        and need_bias_support
        and metadata.has_bias
        and metadata.ksplit > 1
        and stage1_func in (_flydsl_stage1_wrapper, cktile_moe_stage1)
        and expert_mask is not None
    )

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
    local_topk_ids = None
    fused_sort_quant = not metadata.output_aux and _is_fused_decode_sort_quant_enabled(
        hidden_states,
        topk_ids,
        topk_weight,
        global_experts=c.global_E,
        block_m=block_size_M,
        quant_type=quant_type,
        q_dtype_a=q_dtype_a,
        q_dtype_w=w1.dtype,
        activation=activation,
        is_g1u1=c.isG1U1,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
        need_local_topk_ids=need_local_topk_ids,
        run_1stage=metadata.run_1stage,
        stage1_is_flydsl=stage1_func is _flydsl_stage1_wrapper,
        hidden_pad=c["hidden_pad"],
        ksplit=metadata.ksplit,
    )
    if metadata.output_aux:
        # Only the layout-v2 scatter GEMM2 consumes the masked aux sort (remote
        # routes get reverse_sorted = -1).
        if expert_mask is not None and stage2_func is not _flydsl_v2_stage2_wrapper:
            raise NotImplementedError(
                "MXFP4 a4w4 FlyDSL port does not support expert-parallel yet "
                "(only the layout-v2 scatter GEMM2 handles the masked aux sort)."
            )
        stage2_kwargs = metadata.stage2.keywords
        kernel_name2 = stage2_kwargs.get("kernelName2") or stage2_kwargs.get(
            "kernelName", ""
        )
        atomic = parse_g2_kname_any(kernel_name2)["atomic"]
        sorting_ret = sort.run(
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
            return_local_topk_ids=need_local_topk_ids,
        )
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            m_indices,
            reverse_sorted,
        ) = sorting_ret[:7]
        local_topk_ids = sorting_ret[7] if need_local_topk_ids else None
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
        )
    else:
        sorting_ret = sort.run(
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
            return_local_topk_ids=need_local_topk_ids,
        )
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            sorting_ret[:5]
        )
        if need_local_topk_ids:
            local_topk_ids = sorting_ret[5]
    _opus_a8w4.check_route_bucket_metadata(metadata, sorted_expert_ids, logger)
    if after_sort is not None:
        after_sort()

    # fused_moe_2stages, with its `topk_ids` argument.
    topk_ids = local_topk_ids if local_topk_ids is not None else topk_ids
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
    beta, linear_beta, swiglu_limit = c["beta"], c["linear_beta"], c["swiglu_limit"]
    config_situ_beta, config_situ_linear_beta, normalized_swiglu_limit = (
        _normalize_mxfp4_activation_params(activation, beta, linear_beta, swiglu_limit)
    )
    if (
        stage1_func is _mxfp4_a4w4_stage1_fw
        and metadata.output_aux == AUX_SORT_OPUS
        # block_m 16 is the only inline-quant a4w4 GEMM1: it reads raw bf16.
        and int(metadata.block_m) != 16
    ):
        metadata = replace(metadata, prequant=True)
    if prequantized_a1 is not None:
        a1, a1_scale = prequantized_a1, prequantized_a1_scale
    else:
        a1, a1_scale = quant_a1_mxfp4(
            metadata,
            hidden_states,
            c["a1_scale"],
            q_dtype_a,
            dtype,
            w1.dtype,
            activation,
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
    # The a16w4 FlyDSL stage 1 allocates its own sorted intermediate.
    is_a16w4_port = (
        dtype in (dtypes.bf16, dtypes.fp16)
        and w1.dtype == dtypes.fp4x2
        and q_dtype_a == dtypes.bf16
        and stage1_func is _flydsl_stage1_wrapper
    )
    a2 = (
        None
        if is_a16w4_port
        else torch.empty((token_num, topk, inter_dim), dtype=dtype, device=device)
    )
    extra_stage1_args = plan.starting(Op.GEMM1).segment.call_args(c)
    extra_stage2_args = plan.starting(Op.GEMM2).segment.call_args(c)
    if not metadata.run_1stage and need_bias_support:
        if metadata.has_bias:
            extra_stage1_args["bias1"] = _normalize_bias_for_kernel(c["bias1"])
            if stage1_func in (_flydsl_stage1_wrapper, cktile_moe_stage1):
                extra_stage1_args["topk_ids"] = topk_ids
        if metadata.stage2_has_bias:
            extra_stage2_args["bias2"] = _normalize_bias_for_kernel(c["bias2"])
    if stage1_func in (_flydsl_stage1_wrapper, _opus_a8w4_stage1_wrapper):
        # These clamp silu whenever a finite limit is configured: the caller's limit.
        extra_stage1_args["swiglu_limit"] = swiglu_limit
    elif stage1_func is _mxfp4_a4w4_stage1_fw:
        extra_stage1_args["swiglu_limit"] = normalized_swiglu_limit
    if stage1_func is _flydsl_stage1_wrapper and metadata.skip_inter_quant:
        extra_stage1_args["v2_output_layout"] = True
    if stage1_func is _flydsl_stage1_wrapper:
        extra_stage1_args["situ_beta"] = config_situ_beta
        extra_stage1_args["situ_linear_beta"] = config_situ_linear_beta
    elif stage1_func is _mxfp4_a4w4_stage1_fw:
        # MXMOE's activation comes from its name and betas enter its compile key.
        extra_stage1_args["situ_beta"] = (
            DEFAULT_SITUV2_BETA
            if activation == ActivationType.Situv2 and beta is None
            else config_situ_beta
        )
        extra_stage1_args["situ_linear_beta"] = (
            DEFAULT_SITUV2_LINEAR_BETA
            if activation == ActivationType.Situv2 and linear_beta is None
            else config_situ_linear_beta
        )
    elif stage1_func is _opus_a8w4_stage1_wrapper:
        if metadata.skip_inter_quant:
            extra_stage1_args["output_sorted"] = True
        extra_stage1_args["situ_beta"] = 4.0 if beta is None else float(beta)
        extra_stage1_args["situ_linear_beta"] = (
            25.0 if linear_beta is None else float(linear_beta)
        )
    if (
        stage2_func in (_flydsl_stage2_wrapper, _flydsl_v2_stage2_wrapper)
        and expert_mask is not None
    ):
        extra_stage2_args["expert_mask"] = expert_mask
        extra_stage2_args["topk_ids"] = topk_ids
    stage2_keywords = getattr(metadata.stage2, "keywords", None) or {}
    uses_flydsl_v2_stage2 = stage2_func is _flydsl_v2_stage2_wrapper or (
        stage2_func is _mxfp4_a4w4_stage2_fw
        and str(stage2_keywords.get("kernelName2", "")).startswith(
            "flydsl_moe2_layout_"
        )
    )
    if uses_flydsl_v2_stage2:
        extra_stage2_args["use_valid_token_count"] = num_local_tokens is not None
        if not doweight_stage1 and _flydsl_stage2_fp8_enabled():
            # FP8 route-output reduction applies the route weights after GEMM2.
            extra_stage2_args["topk_weights"] = topk_weight
    if stage2_func is _opus_a8w4.opus_a8w4_stage2_wrapper:
        extra_stage2_args["stage2_fp8_enabled"] = _opus_stage2_fp8_enabled()
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
    else:
        a2, a2_scale = quant_inter_mxfp4(
            a2,
            a2_scale,
            a1_scale,
            metadata,
            q_dtype_a,
            dtype,
            w1.dtype,
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
