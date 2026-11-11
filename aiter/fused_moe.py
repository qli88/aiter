# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
import re
from collections.abc import Callable
from dataclasses import replace

import torch

import aiter

# from aiter import get_torch_quant as get_quant
from aiter import (
    ActivationType,
    QuantType,
    dtypes,
    fused_dynamic_mxfp4_quant_moe_sort,
    fused_dynamic_mxfp8_quant_moe_sort,
    logger,
    mxfp4_moe_sort_fwd,
)
from aiter import get_hip_quant as get_quant
from aiter.fused_moe_registry import (  # noqa: F401
    BoundFusedMoeImpl,
    FusedMoeImplResolutionError,
    FusedMoeRequest,
    resolve_fused_moe_impl,
)
from aiter.jit.core import AITER_CONFIGS, AITER_CSRC_DIR, PY, bd_dir, mp_lock
from aiter.jit.utils.chip_info import (  # noqa: F401
    get_cu_num,
    get_gfx,
    get_gfx_runtime,
    gfx_from_cu_num,
)
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.moev2 import hooks as moev2_hooks
from aiter.moev2 import paths as moev2_paths
from aiter.moev2.dispatch import tuned as moev2_tuned
from aiter.moev2.dispatch.tuned import _ACT_TYPE_DISABLED_KEY
from aiter.moev2.execute.metadata import MOEMetadata
from aiter.moev2.execute.output import (
    _return_output,
    _validate_output_buffer_metadata,
    _validate_output_buffer_no_overlap,
)
from aiter.moev2.policy import (  # noqa: F401
    _SWIGLU_MXFP4_BF16_BOUND,
    _bound_split,
    _normalize_mxfp4_activation_params,
    quant_remap,
    resolve_activation_dtype,
)
from aiter.moev2.reference.torch_moe import (  # noqa: F401
    _int4_per_1x32_weight_expert_to_f32,
    _mxfp4_per_1x32_weight_expert_to_f32,
    situv2,
    swiglu,
    torch_moe,
    torch_moe_act,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.moev2.routing.topk import fused_topk  # noqa: F401
from aiter.moev2.segments.asm import asm_stage1, fused_moe_1stage
from aiter.moev2.segments.ck import ck_moe_stage1
from aiter.moev2.segments.ck_tile import (
    _is_cktile_mxfp4_stage2_name,
    cktile_moe_stage1,
    cktile_moe_stage2,
)
from aiter.moev2.segments.defaults import (
    BLOCK_SIZE_M,
    get_block_size_M,
    get_ksplit,
    use_nt,
)
from aiter.moev2.segments.flydsl import (  # noqa: F401
    _can_reroute_mxfp4_to_flydsl,
    _empty_bf16,
    _empty_u8,
    _flydsl_mxfp4_layout_is_compatible,
    _flydsl_route_output_nbytes,
    _flydsl_stage1_wrapper,
    _flydsl_stage2_fp8_enabled,
    _flydsl_stage2_wrapper,
    _flydsl_v2_stage2_wrapper,
    _get_flydsl_moe_kernels,
    _get_padding_for_flydsl,
    _is_flydsl_stage2,
    _mxfp4_intermediate_enabled,
    _mxfp4_scale_u8,
    _needs_swiglu_bias_support,
    _normalize_bias_for_kernel,
    _stage2_honors_runtime_reduce_override,
    _stage2_runtime_uses_route_reduce,
    _stage2_uses_fp8_route_output,
    _validate_flydsl_route_output_nbytes,
    stage2_uses_route_reduce,
)
from aiter.moev2.segments.mxmoe import (  # noqa: F401
    _is_inline_sort_cfg,
    _is_inline_sort_kname,
    _is_mxfp4_inline_sort,
    _make_mxfp4_metadata,
    _mxfp4_a4w4_stage1,
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_a4w4_stage2,
    _mxfp4_a4w4_stage2_fw,
    _mxfp4_inline_sort_unsupported,
    _padded_scale_cols,
)
from aiter.moev2.segments.opus import _opus_stage2_fp8_enabled
from aiter.moev2.segments.sort import (
    _FUSED_DECODE_SORT_QUANT_MAX_M,
    _FUSED_DECODE_SORT_QUANT_ROUTES,
    _MOE_SORT_BACKEND,
    AUX_SORT_OPUS,
    _adaptive_moe_sort,
    _aux_uses_opus,
    _fused_decode_sort_quant,
    _moe_prepare_unsorted_input,
    _mxfp4_aux_instance_supported,
)
from aiter.moev2.shapes import (  # noqa: F401
    _PADDED_M_TIERS,
    get_inter_dim,
    get_padded_M,
    nextPow2,
)
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
    get_flydsl_activation_name,
)
from aiter.ops.flydsl.mxfp4_kname import (
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    parse_flydsl_v2_gemm2_kernel,
    parse_g2_kname_any,
)
from aiter.ops.opus import moe_stage2_a8w4 as _opus_a8w4
from aiter.ops.opus.moe_stage1_a8w4 import (
    opus_a8w4_stage1_wrapper as _opus_a8w4_stage1_wrapper,
)

_USE_CK_MOE_SORTING = os.environ.get("AITER_USE_CK_MOE_SORTING", "0") == "1"
_USE_FLYDSL_MOE_SORTING = os.environ.get("AITER_USE_FLYDSL_MOE_SORTING", "0") == "1"
_MOE_A8W4_BYPASS_QUANT = os.environ.get("AITER_MOE_A8W4_BYPASS_QUANT", "0") == "1"
# Internal benchmark/test hook (not a public switch): set True to force the
# generic sort+quant path so base-vs-head latency can be measured from one
# build. Production leaves this False and selects the fused kernel purely from
# the hardware/shape/dtype/route gate below.
_fused_decode_sort_quant_disabled = False


def _is_fused_decode_sort_quant_enabled(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    global_experts: int,
    block_m: int,
    quant_type: QuantType,
    q_dtype_a: torch.dtype,
    q_dtype_w: torch.dtype,
    activation: ActivationType,
    is_g1u1: bool,
    expert_mask: torch.Tensor | None,
    num_local_tokens: torch.Tensor | None,
    need_local_topk_ids: bool,
    run_1stage: bool,
    stage1_is_flydsl: bool,
    hidden_pad: int,
    ksplit: int,
) -> bool:
    """Gate the direct-scale HIP route-sort/quant kernel to supported decode.

    The shape bounds mirror the kernel's own contract (``model_dim=4096``,
    ``block_m=32``, ``M`` in ``[1, 128]``, an instantiated ``(E, topk)``
    route). The remaining conditions are integration requirements: they
    describe the stage-1 path whose activation quantization this kernel
    replaces. The fused path is selected automatically whenever every
    condition matches; it falls back to the generic sort + quant otherwise.
    """
    if _fused_decode_sort_quant_disabled:
        return False
    try:
        runtime_is_gfx950 = get_gfx_runtime() == "gfx950"
    except (KeyError, OSError, RuntimeError):
        return False
    tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
    topk = topk_ids.shape[1] if topk_ids.ndim == 2 else 0
    return (
        runtime_is_gfx950
        and hidden_states.dtype == dtypes.bf16
        and 1 <= tokens <= _FUSED_DECODE_SORT_QUANT_MAX_M
        and hidden_states.shape[1] == 4096
        and hidden_states.is_contiguous()
        and topk_ids.dtype == dtypes.i32
        and (global_experts, topk) in _FUSED_DECODE_SORT_QUANT_ROUTES
        and topk_ids.shape[0] == tokens
        and topk_ids.is_contiguous()
        and topk_weights.dtype == dtypes.fp32
        and topk_weights.is_contiguous()
        and block_m == 32
        and quant_type == QuantType.per_1x32
        and q_dtype_a == dtypes.fp4x2
        and q_dtype_w == dtypes.fp4x2
        and activation == ActivationType.Silu
        and is_g1u1
        and expert_mask is None
        and num_local_tokens is None
        and not need_local_topk_ids
        and not run_1stage
        and stage1_is_flydsl
        and hidden_pad == 0
        # ksplit > 1 makes fused_moe_2stages hand stage 1 unquantized BF16
        # activations (the kernel quantizes inline), so prequantizing here
        # would feed the wrong dtype.
        and ksplit <= 1
    )


# Optional hook for collecting per-stage benchmark callables.
kernel_bench_callable = None


def _moe_sorting_impl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    dispatch_policy,
    use_opus,
    return_local_topk_ids=False,
    accumulate=True,
    output_aux=False,
    output=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape

    if (
        output_aux
        and expert_mask is None
        and not _aux_uses_opus(output_aux, block_size, M * topk, num_experts)
        and _mxfp4_aux_instance_supported(
            num_experts, topk, model_dim, block_size, accumulate
        )
        and _MOE_SORT_BACKEND not in ("opus", "ck")
    ):
        # adaptive (fused) sort emits the a4w4 extras (m_indices + reverse_sorted)
        # plus the atomic zero-init; opus single-pass aux is the env-gated fallback.
        # It has no expert_mask support, so EP always takes the Opus aux sort.
        # `output` not threaded here: this buffer also feeds stage1 as moe_buf.
        return _adaptive_moe_sort(
            topk_ids,
            topk_weights,
            num_experts,
            topk,
            block_size,
            model_dim,
            atomic=accumulate,
            emit_aux=True,
            moebuf_dtype=moebuf_dtype,
        )

    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    max_num_tokens_padded = max_num_m_blocks * block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    # moe_buf shape depends on the downstream stage2 path:
    #  - accumulate (or EP w/ expert_mask): stage2 atomically accumulates into [M, model_dim].
    #  - else (FlyDSL stage2 reduce mode without mask, or the non-atomic
    #    output_aux scatter, which writes every output row itself): caller owns
    #    the intermediate; allocate a placeholder here.
    # A caller buffer can stand in: the sort kernel zeroes what it is handed.
    if accumulate or (expert_mask is not None and not output_aux):
        moe_buf = (
            output
            if output is not None
            else torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        )
    else:
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)
    local_topk_ids = torch.empty_like(topk_ids) if return_local_topk_ids else None
    if return_local_topk_ids:
        # CK sorting does not emit local ids; use Opus so callers do not need a slow
        # Python-side remap or a hard failure when local expert ids are required.
        use_opus = True

    aux_m_indices = None
    aux_reverse_sorted = None
    if output_aux:
        use_opus = True
        dispatch_policy = (
            0 if _aux_uses_opus(output_aux, block_size, M * topk, num_experts) else 1
        )
        aux_m_indices = torch.empty(
            max_num_tokens_padded, dtype=dtypes.i32, device=device
        )
        aux_reverse_sorted = torch.empty(M * topk, dtype=dtypes.i32, device=device)

    if use_opus:
        ws_size = aiter.moe_sorting_opus_get_workspace_size(
            M, num_experts, topk, dispatch_policy, topk_ids.get_device()
        )
        workspace = (
            torch.empty(ws_size, dtype=torch.uint8, device=device)
            if ws_size > 0
            else None
        )
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            workspace,
            dispatch_policy,
            local_topk_ids,
            aux_m_indices,
            aux_reverse_sorted,
        )
    else:
        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            dispatch_policy,
        )
    ret = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    if output_aux:
        ret = (*ret, aux_m_indices, aux_reverse_sorted)
    if return_local_topk_ids:
        return (*ret, local_topk_ids)
    return ret


def _flydsl_moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    accumulate=True,
    output=None,
):
    """FlyDSL sorting dispatch -- called outside torch_compile_guard."""
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd

    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    max_num_tokens_padded = max_num_m_blocks * block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    # moe_buf shape mirrors _moe_sorting_impl: full [M, model_dim] when stage2
    # accumulates (or EP w/ expert_mask), else a (0,0) placeholder for FlyDSL
    # stage2 reduce mode. The kernel no-ops its zero pass on an empty buffer
    # (moe_buf_elems == 0), so reduce mode skips zeroing the [M, model_dim]
    # buffer entirely — the caller owns the [M, topk, model_dim] intermediate.
    # As in _moe_sorting_impl, a caller buffer can stand in.
    if (expert_mask is not None) or accumulate:
        moe_buf = (
            output
            if output is not None
            else torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        )
    else:
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)

    flydsl_moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        expert_mask,
        num_local_tokens,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
    return_local_topk_ids=False,
    accumulate=True,
    flat=False,
    output_aux=False,
    output=None,
):
    if (
        not _USE_CK_MOE_SORTING
        and _USE_FLYDSL_MOE_SORTING
        and not return_local_topk_ids
        and not flat
        and not output_aux
        and dispatch_policy == 0
    ):
        return _flydsl_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            accumulate=accumulate,
            output=output,
        )
    # FLAT kernel: in-kernel routing (manifest flat=1); pass through unsorted topk.
    # `output` cannot be written in place: FLAT needs extra bytes appended to moe_buf.
    if flat:
        assert output is None, "FLAT over-allocates moe_buf; the caller copies"
        return _moe_prepare_unsorted_input(
            topk_ids, topk_weights, model_dim, moebuf_dtype
        )
    try:
        return _moe_sorting_impl(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            use_opus=not _USE_CK_MOE_SORTING,
            return_local_topk_ids=return_local_topk_ids,
            accumulate=accumulate,
            output_aux=output_aux,
            output=output,
        )
    except Exception as e:
        logger.error(f"Error in moe_sorting: {e}")
        max_num_tokens_padded = int(
            topk_ids.numel() + num_experts * block_size - topk_ids.shape[1]
        )
        topk = topk_ids.shape[1]
        logger.error(
            f"Moe_sorting info: {max_num_tokens_padded=} {block_size=} {num_experts=} {topk=} {topk_ids.shape=}"
        )
        raise


def get_topk_valid_mask(
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build valid_mask [token_num, topk] for EP mode.

    valid_mask[t, k] = 1 if topk_ids[t, k] points to a local (non-fake) expert,
    else 0. When expert_mask is None (non-EP), returns all-ones.
    """
    if expert_mask is None:
        return torch.ones(topk_ids.shape, dtype=dtypes.i32, device=topk_ids.device)
    return expert_mask[topk_ids]


def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask: torch.Tensor | None = None,  # EP
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    w1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M=None,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    stage2_scatter: Stage2ScatterContext | None = None,
    # Optional [M, model_dim] destination for the result, to save the caller a
    # copy. Must be contiguous, match shape/dtype/device and not overlap
    # hidden_states, or the call raises; when given it is what gets returned.
    output: torch.Tensor | None = None,
    quant_type_a: QuantType | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    # False explicitly selects unclamped shared-expert SiLU semantics.
    clamp_shared: bool = True,
):
    """Run fused MoE, dispatching to FHMoE when shared-expert tensors are present.

    Supplying any shared tensor (or ``shared_expert_id``) requires all four
    shared weight/scale tensors and dispatches through the gfx950 FHMoE path.
    Inputs must satisfy the contracts documented in :mod:`aiter.fhmoe`.

    ``clamp_shared`` is an FHMoE model-semantic input. The default ``True``
    applies the configured SiLU clamp to routed and shared experts. ``False``
    changes only the shared expert to unclamped SiLU; routed experts still use
    ``swiglu_limit``. Clamped and unclamped stage-1 kernels have distinct AOT
    cache identities, so deployments changing this setting must rebuild their
    FlyDSL AOT cache.

    """
    if not torch.compiler.is_compiling() and moev2_paths.any_active():
        call = dict(locals())
        v2_path = moev2_paths.route_to_v2(call, "api")
        if v2_path is not None:
            out = v2_path.run(call)
            if out is not moev2_paths.DECLINED:
                return out
    has_shared_expert = (
        any(
            tensor is not None
            for tensor in (shared_w1, shared_w2, shared_w1_scale, shared_w2_scale)
        )
        or shared_expert_id != -1
    )
    if not clamp_shared and not has_shared_expert:
        raise ValueError("clamp_shared=False requires FHMoE shared-expert dispatch")
    if has_shared_expert:
        routed_mxfp8 = w1 is not None and w1.dtype == dtypes.fp8
        unsupported = []
        if stage2_scatter is not None:
            unsupported.append("stage2_scatter")
        if quant_type_a is not None:
            unsupported.append("quant_type_a")
        fhmoe_quant_dtype_a = (
            dtypes.fp8 if GateMode(gate_mode) == GateMode.INTERLEAVE else dtypes.fp4x2
        )
        if quant_dtype_a is not None and quant_dtype_a != fhmoe_quant_dtype_a:
            unsupported.append("quant_dtype_a")
        if quant_dtype_a2 is not None:
            unsupported.append("quant_dtype_a2")
        # The MXFP8 path does not consume caller-prequantized activations and
        # requires an explicit preshuffle marker. Preserve the existing DSV4
        # MXFP4 path, which accepts activation scales and trusts its layout.
        if routed_mxfp8 and a1_scale is not None:
            unsupported.append("a1_scale")
        if routed_mxfp8 and a2_scale is not None:
            unsupported.append("a2_scale")
        if unsupported:
            raise NotImplementedError(
                "FHMoE shared-expert dispatch does not support "
                + ", ".join(unsupported)
            )
        weights = (w1, w2, shared_w1, shared_w2)
        if routed_mxfp8 and any(
            weight is not None and not getattr(weight, "is_shuffled", False)
            for weight in weights
        ):
            raise ValueError(
                "FHMoE weights must be AITER-preshuffled and marked "
                "is_shuffled=True; re-run the shuffle after tensor copies"
            )
        from aiter.fhmoe import _fhmoe

        return _fhmoe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            expert_mask=expert_mask,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_size_M=block_size_M,
            num_local_tokens=num_local_tokens,
            moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
            dtype=dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
            swiglu_limit=swiglu_limit,
            gate_mode=gate_mode,
            shared_w1=shared_w1,
            shared_w2=shared_w2,
            shared_w1_scale=shared_w1_scale,
            shared_w2_scale=shared_w2_scale,
            shared_expert_id=shared_expert_id,
            clamp_shared=clamp_shared,
            output=output,
        )
    if not block_size_M:
        block_size_M = -1
    enable_ep_scatter = stage2_scatter is not None
    scatter_source_map = stage2_scatter.source_token_map if enable_ep_scatter else None
    return fused_moe_(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=activation.value,
        quant_type=quant_type.value,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        ep_arena_handle=stage2_scatter.arena_handle if enable_ep_scatter else 0,
        ep_combine_input_offset=(
            stage2_scatter.combine_input_offset if enable_ep_scatter else 0
        ),
        ep_slot_stride_bytes=(
            stage2_scatter.slot_stride_bytes if enable_ep_scatter else 0
        ),
        ep_max_tokens_per_rank=(
            stage2_scatter.max_tokens_per_rank if enable_ep_scatter else 0
        ),
        ep_world_size=stage2_scatter.world_size if enable_ep_scatter else 0,
        ep_combine_quant=(
            int(stage2_scatter.combine_quant_bits) if enable_ep_scatter else 0
        ),
        ep_source_token_map=scatter_source_map,
        output=output,
        quant_type_a=None if quant_type_a is None else quant_type_a.value,
        quant_dtype_a=quant_dtype_a,
        quant_dtype_a2=quant_dtype_a2,
    )


def fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2: torch.Tensor,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,  # EP
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    # following for quant
    w1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M: int = -1,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    beta: float | None = None,
    linear_beta: float | None = None,
    gate_mode: str = GateMode.SEPARATED.value,
    ep_arena_handle: int = 0,
    ep_combine_input_offset: int = 0,
    ep_slot_stride_bytes: int = 0,
    ep_max_tokens_per_rank: int = 0,
    ep_world_size: int = 0,
    ep_combine_quant: int = 0,
    ep_source_token_map: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: int | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
) -> torch.Tensor:
    device = topk_ids.device
    M, _topk = topk_ids.shape
    dtype = hidden_states.dtype if dtype is None else dtype
    model_dim = w2.shape[1]
    if ep_source_token_map is not None and output is not None:
        raise RuntimeError(
            "output= is incompatible with stage2_scatter: GEMM2 P2P-writes the "
            "route-weighted results into the peers' combine buffers and returns an "
            "uninitialized placeholder"
        )
    # Unconditional: the runtime returns `output` on every path, and which kernel
    # writes it depends on env vars and tuned metadata the fake cannot see.
    if output is not None:
        _validate_output_buffer_metadata(
            output, (M, model_dim), dtype, hidden_states.device
        )
        return output
    return torch.empty((M, model_dim), dtype=dtype, device=device)


@torch_compile_guard(gen_fake=fused_moe_fake)
def fused_moe_(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2: torch.Tensor,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,  # EP
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    # following for quant
    w1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M: int = -1,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    beta: float | None = None,
    linear_beta: float | None = None,
    gate_mode: str = GateMode.SEPARATED.value,
    ep_arena_handle: int = 0,
    ep_combine_input_offset: int = 0,
    ep_slot_stride_bytes: int = 0,
    ep_max_tokens_per_rank: int = 0,
    ep_world_size: int = 0,
    ep_combine_quant: int = 0,
    ep_source_token_map: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: int | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
) -> torch.Tensor:
    if moev2_paths.any_active():
        call = dict(locals())
        v2_path = moev2_paths.route_to_v2(call, "op")
        if v2_path is not None:
            with moev2_hooks.legacy_globals(
                kernel_bench_callable, _fused_decode_sort_quant_disabled
            ):
                out = v2_path.run(call)
            if out is not moev2_paths.DECLINED:
                return out
    stage2_scatter = None
    if ep_source_token_map is not None:
        stage2_scatter = Stage2ScatterContext(
            arena_handle=ep_arena_handle,
            combine_input_offset=ep_combine_input_offset,
            slot_stride_bytes=ep_slot_stride_bytes,
            max_tokens_per_rank=ep_max_tokens_per_rank,
            world_size=ep_world_size,
            source_token_map=ep_source_token_map,
            combine_quant_bits=int(ep_combine_quant),
        )
    return _fused_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        stage2_scatter=stage2_scatter,
        output=output,
        quant_type_a=quant_type_a,
        quant_dtype_a=quant_dtype_a,
        quant_dtype_a2=quant_dtype_a2,
    )


def _fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M: int = -1,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    beta: float | None = None,
    linear_beta: float | None = None,
    gate_mode: str = GateMode.SEPARATED.value,
    stage2_scatter: Stage2ScatterContext | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: int | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    *,
    _q_dtype_a: torch.dtype | None = None,
    _metadata_transform: Callable | None = None,
    _metadata_config_file: str | None = None,
    _stage1_extra_args: dict | None = None,
    _stage2_extra_args: dict | None = None,
    _stage2_override: Callable | None = None,
    _enforce_metadata_block_size: bool = False,
) -> torch.Tensor:
    # We do such convert since custom_op schema restriction on block_size_M, and Enum type
    activation = ActivationType(activation)
    quant_type = QuantType(quant_type)
    gate_mode = GateMode(gate_mode)
    if quant_type_a is not None and QuantType(quant_type_a) != quant_type:
        raise NotImplementedError(
            f"quant_type_a={QuantType(quant_type_a)!s} != quant_type={quant_type!s}: "
            "mixed activation/weight quant granularity is not supported"
        )
    if block_size_M == -1:
        block_size_M = None
    """user API"""
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]
    isShuffled = getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
    # gfx1250: GUGU row-interleave, not 16-row block shuffle.
    if get_gfx() == "gfx1250" and isG1U1 and gate_mode != GateMode.INTERLEAVE:
        raise ValueError(
            "gfx1250 fused_moe requires gu_interleave weight layout "
            f"(gate_mode={GateMode.INTERLEAVE.value!r}), got {gate_mode.value!r}"
        )

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    dtype = hidden_states.dtype if dtype is None else dtype
    assert dtype in [
        dtypes.fp16,
        dtypes.bf16,
    ], f"Fused_moe unsupported out dtype: {dtype}"
    if stage2_scatter is not None and output is not None:
        raise RuntimeError(
            "output= is incompatible with stage2_scatter: GEMM2 P2P-writes the "
            "route-weighted results into the peers' combine buffers and returns an "
            "uninitialized placeholder"
        )
    _validate_output_buffer_metadata(
        output, (M, model_dim), dtype, hidden_states.device
    )
    _validate_output_buffer_no_overlap(output, hidden_states)
    quant_type = quant_remap.get(quant_type, quant_type)
    q_dtype_w = w1.dtype
    q_dtype_a = resolve_activation_dtype(
        quant_type,
        q_dtype_w,
        activation=activation,
        gate_mode=gate_mode,
        M=M,
        hidden_dtype=hidden_states.dtype,
        has_a1_scale=a1_scale is not None,
        gfx=get_gfx_runtime(),
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
    )

    if quant_dtype_a is not None:
        q_dtype_a = quant_dtype_a
    elif _q_dtype_a is not None:
        q_dtype_a = _q_dtype_a

    grouped_a8w4_out = None
    from aiter.ops.flydsl.grouped_moe_gfx1250 import grouped_gemm_gfx1250_a8w4

    # grouped_gemm_gfx1250_a8w4 reads GUGU (gate/up row-interleaved) w1 only,
    # so it is reachable exclusively from GateMode.INTERLEAVE. SEPARATED
    # weights fall through to the generic MoE below.
    if gate_mode == GateMode.INTERLEAVE:
        grouped_a8w4_out = grouped_gemm_gfx1250_a8w4(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            E=E,
            model_dim=model_dim,
            inter_dim=inter_dim,
            dtype=dtype,
            activation=activation,
            quant_type=quant_type,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            isG1U1=isG1U1,
            doweight_stage1=doweight_stage1,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            # A quantizing EP dispatch puts the caller's e8m0 row here;
            # the gfx1250 path uses it to skip a quant it would redo.
            a1_scale=a1_scale,
            expert_mask=expert_mask,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
            swiglu_limit=swiglu_limit,
            num_local_tokens=num_local_tokens,
            situ_beta=1.0 if beta is None else float(beta),
            situ_linear_beta=1.0 if linear_beta is None else float(linear_beta),
            stage2_scatter=stage2_scatter,
        )

    if grouped_a8w4_out is not None:
        return _return_output(grouped_a8w4_out, output)

    # a16w4-SiTUv2 (bf16 A x MXFP4 W); SiTUv2 gate distinguishes gpt-oss (Swiglu -> cktile).
    _is_a16w4_situv2 = (
        quant_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
        and q_dtype_a == dtypes.bf16
        and activation == ActivationType.Situv2
    )
    # Validate the Swiglu shapes that get re-routed onto the a16w4 FlyDSL port
    # against the runtime-only constraints that get_2stage_cfgs cannot see.
    _is_a16w4_swiglu_rerouted = (
        inter_dim % 256 != 0
        and q_dtype_a == dtypes.bf16
        and _can_reroute_mxfp4_to_flydsl(
            model_dim=model_dim,
            inter_dim=inter_dim,
            dtype=dtype,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            q_type=quant_type,
            activation=activation,
            is_shuffled=getattr(w1, "is_shuffled", False)
            and getattr(w2, "is_shuffled", False),
            use_g1u1=isG1U1,
            doweight_stage1=doweight_stage1,
            gate_mode=gate_mode,
            gate_up_interleave=getattr(w1, "gate_up_interleave", None),
        )
    )
    if _is_a16w4_situv2 or _is_a16w4_swiglu_rerouted:
        _a16w4_why = (
            "SiTUv2" if _is_a16w4_situv2 else f"Swiglu with inter_dim={inter_dim}"
        )
        for _bad, _why in (
            (
                get_gfx() not in ("gfx942", "gfx950"),
                (
                    f"requires the FlyDSL kernel on CDNA gfx942/gfx950 "
                    f"(gfx={get_gfx()!r})"
                ),
            ),
            (bias1 is not None or bias2 is not None, "per-expert bias"),
            (expert_mask is not None, "expert-parallel masking"),
            (
                not (isShuffled and isG1U1 and not doweight_stage1),
                "non-preshuffled / non-g1u1 weights or doweight_stage1=True",
            ),
        ):
            if _bad:
                raise NotImplementedError(
                    f"a16w4 (bf16 A x MXFP4 W) {_a16w4_why} is not supported: {_why}."
                )

    config_situ_beta, config_situ_linear_beta, _ = _normalize_mxfp4_activation_params(
        activation, beta, linear_beta, swiglu_limit
    )

    def _resolve_metadata(disable_inline_sort=False):
        metadata = get_2stage_cfgs(
            get_padded_M(M),
            model_dim,
            inter_dim,
            E,
            topk,
            dtype,
            q_dtype_a,
            q_dtype_w,
            quant_type,
            isG1U1,
            activation,
            doweight_stage1,
            hidden_pad,
            intermediate_pad,
            isShuffled,
            gate_mode,
            is_ep=expert_mask is not None,
            has_stage1_bias=bias1 is not None,
            has_stage2_bias=bias2 is not None,
            situ_beta=config_situ_beta,
            situ_linear_beta=config_situ_linear_beta,
            swiglu_limit=swiglu_limit,
            opus_weights_shuffled=getattr(w1, "is_shuffled", False)
            and getattr(w2, "is_shuffled", False),
            gate_up_interleave=getattr(w1, "gate_up_interleave", None),
            config_file=_metadata_config_file,
            _disable_inline_sort=disable_inline_sort,
            q_dtype_a2=quant_dtype_a2,
            input_dtype=hidden_states.dtype,
            has_stage2_scatter=stage2_scatter is not None,
            has_activation_scales=a1_scale is not None or a2_scale is not None,
            has_num_local_tokens=num_local_tokens is not None,
        )
        return (
            metadata if _metadata_transform is None else _metadata_transform(metadata)
        )

    metadata = _resolve_metadata()

    # A tuned fast row that this invocation cannot run is discarded, not raised
    # on: re-resolve with that family disabled and fall back to legacy.
    use_inline_sort = _is_mxfp4_inline_sort(metadata)
    if use_inline_sort:
        reason = _mxfp4_inline_sort_unsupported(
            metadata,
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_ids,
            gate_mode=gate_mode,
            num_local_tokens=num_local_tokens,
            expert_mask=expert_mask,
            bias1=bias1,
            bias2=bias2,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            stage2_scatter=stage2_scatter,
            block_size_M=block_size_M,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )
        if reason:
            logger.warning(
                f"[fused_moe] inline-sort config is unsupported ({reason}); "
                "using default heuristics"
            )
            metadata = _resolve_metadata(disable_inline_sort=True)
            use_inline_sort = False

    stage2_func = getattr(metadata.stage2, "func", metadata.stage2)
    if (
        q_dtype_w == dtypes.fp4x2
        and inter_dim % 256 != 0
        and stage2_func is cktile_moe_stage2
    ):
        raise NotImplementedError(
            "No safe MXFP4 MoE backend for this shape and layout: CK-Tile "
            f"stage2 mis-indexes scales at inter_dim={inter_dim}, and the "
            "FlyDSL fallback cannot accept this invocation."
        )

    if (
        _enforce_metadata_block_size
        and block_size_M is not None
        and int(block_size_M) != int(metadata.block_m)
    ):
        raise ValueError(
            f"FHMoE block_size_M={block_size_M} does not match the selected "
            f"kernel block size {metadata.block_m}"
        )
    block_size_M = metadata.block_m if block_size_M is None else block_size_M
    if metadata.full_impl is not None:
        full_output = metadata.full_impl(
            FusedMoeRequest(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weight=topk_weight,
                topk_ids=topk_ids,
                expert_mask=expert_mask,
                activation=activation,
                quant_type=quant_type,
                doweight_stage1=doweight_stage1,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_size_m=(int(block_size_M) if block_size_M is not None else None),
                ksplit=metadata.ksplit,
                num_local_tokens=num_local_tokens,
                moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
                dtype=dtype,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                bias1=bias1,
                bias2=bias2,
                swiglu_limit=swiglu_limit,
                beta=beta,
                linear_beta=linear_beta,
                gate_mode=gate_mode,
                q_dtype_a=q_dtype_a,
                q_dtype_w=q_dtype_w,
            )
        )
        return _return_output(full_output, output)

    # Ensure block_size_M is int (metadata.block_m from CSV may be float)
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
    need_bias_support = _needs_swiglu_bias_support(dtype, quant_type)
    need_local_topk_ids = (
        not metadata.run_1stage
        and need_bias_support
        and metadata.has_bias
        and metadata.ksplit > 1
        and stage1_func in (_flydsl_stage1_wrapper, cktile_moe_stage1)
        and expert_mask is not None
    )
    assert not metadata.flat or get_gfx() in (
        "gfx942",
        "gfx950",
    ), f"FLAT fmoe asm kernels are gfx942/gfx950-only; refusing to launch on {get_gfx()}. "

    sort_m_indices = None
    sort_reverse_sorted = None
    runtime_stage2_reduce = _stage2_runtime_uses_route_reduce(
        metadata.stage2,
        M,
        model_dim,
    )
    uses_common_route_reduce = (
        _is_flydsl_stage2(metadata.stage2) and runtime_stage2_reduce
    )
    if uses_common_route_reduce:
        output_element_size = torch.empty((), dtype=dtype).element_size()
        _validate_flydsl_route_output_nbytes(
            _flydsl_route_output_nbytes(
                metadata.stage2,
                M,
                topk,
                model_dim,
                output_element_size,
            )
        )
    prequantized_a1 = None
    prequantized_a1_scale = None
    # The a4w4 aux-sort families already fuse activation quantization into GEMM1,
    # so the fused sort+quant kernel has nothing left to save there.
    fused_sort_quant = not metadata.output_aux and _is_fused_decode_sort_quant_enabled(
        hidden_states,
        topk_ids,
        topk_weight,
        global_experts=global_E,
        block_m=block_size_M,
        quant_type=quant_type,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        activation=activation,
        is_g1u1=isG1U1,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
        need_local_topk_ids=need_local_topk_ids,
        run_1stage=metadata.run_1stage,
        stage1_is_flydsl=stage1_func is _flydsl_stage1_wrapper,
        hidden_pad=hidden_pad,
        ksplit=metadata.ksplit,
    )
    # compact-scale variant not ported (its stage-1 change targets the
    # refactored mixed_moe_gemm_2stage_common); keep the fused sort+quant on
    # the expanded-scale path only.
    fused_compact_scale = False
    if metadata.output_aux:
        # Only the layout-v2 scatter GEMM2 consumes the masked aux sort (remote
        # routes get reverse_sorted = -1). The MXMOE a4w4 port would silently
        # route tokens to the wrong experts, so fail loudly there.
        if expert_mask is not None and (
            getattr(metadata.stage2, "func", metadata.stage2)
            is not _flydsl_v2_stage2_wrapper
        ):
            raise NotImplementedError(
                "MXFP4 a4w4 FlyDSL port does not support expert-parallel yet "
                "(only the layout-v2 scatter GEMM2 handles the masked aux sort)."
            )
        _stage2_kwargs = metadata.stage2.keywords
        _kn2 = _stage2_kwargs.get("kernelName2") or _stage2_kwargs.get("kernelName", "")
        _atomic = parse_g2_kname_any(_kn2)["atomic"]
        # BM16's adaptive sort already emits routes and zeroes the output without
        # quantizing. Keep the Opus crossover for the configured aux pipeline.
        sorting_ret = moe_sorting(
            topk_ids,
            topk_weight,
            global_E,
            model_dim,
            dtype,
            block_size_M,
            expert_mask,
            num_local_tokens,
            return_local_topk_ids=need_local_topk_ids,
            accumulate=_atomic and not runtime_stage2_reduce,
            output_aux=metadata.output_aux,
            output=output,
        )
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            sort_m_indices,
            sort_reverse_sorted,
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
            global_experts=global_E,
            block_m=block_size_M,
            moebuf_dtype=dtype,
            accumulate=not runtime_stage2_reduce,
            compact_scale=fused_compact_scale,
        )
        local_topk_ids = None
    else:
        sorting_ret = moe_sorting(
            topk_ids,
            topk_weight,
            global_E,
            model_dim,
            dtype,
            block_size_M,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            return_local_topk_ids=need_local_topk_ids,
            accumulate=not runtime_stage2_reduce,
            flat=metadata.flat,
            output=None if metadata.flat else output,
        )
        if need_local_topk_ids:
            (
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf,
                local_topk_ids,
            ) = sorting_ret
        else:
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
                sorting_ret
            )
            local_topk_ids = None
    _opus_a8w4.check_route_bucket_metadata(metadata, sorted_expert_ids, logger)

    if metadata.run_1stage:
        if _stage2_override is not None:
            raise RuntimeError("_stage2_override requires a two-stage MoE config")
        _stage1_call = functools.partial(
            metadata.stage1,
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            isG1U1,
            block_size_M,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            M=M,
            device=topk_ids.device,
            doweight_stage1=doweight_stage1,
        )
        if kernel_bench_callable is not None:
            kernel_bench_callable.append(("stage1", _stage1_call))
        return _return_output(_stage1_call(), output)
    else:
        ret = fused_moe_2stages(
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            isG1U1,
            block_size_M,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            # following for cktile support
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
            topk_ids=local_topk_ids if local_topk_ids is not None else topk_ids,
            topk_weights=topk_weight,
            # only for flydsl dsv4
            swiglu_limit=swiglu_limit,
            beta=beta,
            linear_beta=linear_beta,
            gate_mode=gate_mode,
            expert_mask=expert_mask,
            m_indices=sort_m_indices,
            reverse_sorted=sort_reverse_sorted,
            # Reuse the capability-validated row selected above. Re-looking it
            # up inside fused_moe_2stages could resurrect a tuned row that was
            # deliberately discarded before sorting.
            _metadata_transform=lambda _: metadata,
            _metadata_config_file=_metadata_config_file,
            _stage1_extra_args=_stage1_extra_args,
            _stage2_extra_args=_stage2_extra_args,
            output=output,
            _stage2_override=_stage2_override,
            routing_num_experts=global_E,
            prequantized_a1=prequantized_a1,
            prequantized_a1_scale=prequantized_a1_scale,
        )
        return _return_output(ret, output)


cfg_2stages = None
cfg_2stages_by_file = {}
# fmt: off
fused_moe_1stage_dict = {
    "gfx942":
    {
        # activation,                    quant_type,        dtype,    q_dtype_a,    q_dtype_w,   isG1U1,    doweight_stage1,      API
        (ActivationType.Silu,          QuantType.No,  dtypes.bf16,   dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,          QuantType.No,  dtypes.fp16,   dtypes.fp16,   dtypes.fp16,   False,   False) : aiter.fmoe,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,   dtypes.i4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,    QuantType.per_1x32,  dtypes.bf16,  dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False,   False) : aiter.fmoe_int8_g1u0,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False,   False) : aiter.fmoe_int8_g1u0,
    },
    "gfx950":
    {
        (ActivationType.Silu,    QuantType.per_1x32,   dtypes.bf16,   dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Situv2,  QuantType.per_1x32,   dtypes.bf16,   dtypes.bf16,   dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Gelu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,    dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   True)  : aiter.fmoe_g1u1_tkw1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
    }
}
# fmt: on


@functools.lru_cache(maxsize=2048)
def get_2stage_cfgs(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    activation,
    doweight_stage1,
    hidden_pad,
    intermediate_pad,
    is_shuffled=True,
    gate_mode=GateMode.SEPARATED.value,
    is_ep=False,
    has_stage1_bias=False,
    has_stage2_bias=False,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=None,
    opus_weights_shuffled=None,
    gate_up_interleave=None,
    config_file=None,
    _disable_inline_sort=False,
    q_dtype_a2=None,
    input_dtype=None,
    has_stage2_scatter=False,
    has_activation_scales=False,
    has_num_local_tokens=False,
):
    gate_mode = GateMode(gate_mode)
    cktile_mxfp4_unsafe = q_dtype_w == dtypes.fp4x2 and inter_dim % 256 != 0
    flydsl_can_take_over = _can_reroute_mxfp4_to_flydsl(
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        q_type=q_type,
        activation=activation,
        is_shuffled=bool(opus_weights_shuffled),
        use_g1u1=use_g1u1,
        doweight_stage1=doweight_stage1,
        gate_mode=gate_mode,
        gate_up_interleave=gate_up_interleave,
    )
    if activation == ActivationType.Relu2:
        if use_g1u1:
            raise NotImplementedError(
                "ActivationType.Relu2 is gate-only (non-gated) and does not "
                "support use_g1u1=True; pass use_g1u1=False."
            )
        if doweight_stage1:
            raise NotImplementedError(
                "ActivationType.Relu2 CK-Tile stage1 instances are generated "
                "with MulRoutedWeight=False; doweight_stage1=True is not "
                "supported and would silently drop routed weights."
            )
        if has_stage1_bias or has_stage2_bias:
            raise NotImplementedError(
                "ActivationType.Relu2 CK-Tile path does not support "
                "per-expert bias (has_stage1_bias/has_stage2_bias); the "
                "returned metadata hardcodes has_bias=False."
            )
        if gate_mode != GateMode.SEPARATED:
            raise NotImplementedError(
                "ActivationType.Relu2 CK-Tile stage1 instance "
                "(kFFN_gemm1_gate_only) only supports the gate-only/"
                f"separated weight layout; gate_mode={gate_mode.value!r} "
                "is not supported."
            )
    # Configs are keyed on (gfx, cu_num, ...) so archs that share a cu_num
    # (e.g. gfx950 vs gfx1250, both report 256 CU) don't collide. Legacy CSVs
    # without a `gfx` column are backfilled from cu_num at load time via
    # ``_ensure_gfx_column`` (see gfx_from_cu_num).
    _INDEX_COLS = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    if config_file is not None:
        _INDEX_COLS.extend(
            ["shared_expert_id", "hidden_pad", "intermediate_pad", "gate_mode"]
        )

    def get_cfg_2stages(tune_file):
        return moev2_tuned.read_table(tune_file, _INDEX_COLS, config_file is not None)

    global cfg_2stages
    tune_file = config_file or AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    config_path = os.path.dirname(tune_file)
    untune_file = os.path.join(config_path, "untuned_fmoe.csv")
    profile_file = os.path.join(config_path, "profile_fmoe.csv")
    if config_file is None:
        if cfg_2stages is None:
            cfg_2stages = get_cfg_2stages(tune_file)
        active_cfg_2stages = cfg_2stages
    else:
        active_cfg_2stages = cfg_2stages_by_file.get(tune_file)
        if active_cfg_2stages is None:
            active_cfg_2stages = get_cfg_2stages(tune_file)
            cfg_2stages_by_file[tune_file] = active_cfg_2stages
    cu_num = get_cu_num()
    gfx = get_gfx_runtime()
    keys = (
        gfx,
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        activation,
        str(dtype),
        str(q_dtype_a),
        str(q_dtype_w),
        str(q_type),
        use_g1u1,
        doweight_stage1,
    )
    keys_disabled = (
        gfx,
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        _ACT_TYPE_DISABLED_KEY,
        str(dtype),
        str(q_dtype_a),
        str(q_dtype_w),
        str(q_type),
        use_g1u1,
        doweight_stage1,
    )
    if config_file is not None:
        fhmoe_keys = (expert - 1, hidden_pad, intermediate_pad, str(gate_mode))
        keys += fhmoe_keys
        keys_disabled += fhmoe_keys

    def MainFunc():
        with open(untune_file, "a") as f:
            if os.path.getsize(untune_file) == 0:
                f.write(
                    "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
                )
            q_dtype_ws = q_dtype_w if q_dtype_w != torch.uint32 else "torch.int4"
            f.write(
                f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},{dtype},{q_dtype_a},{q_dtype_ws},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
            )
        logger.info("\033[34m Start tuning fmoe")
        os.system(
            f"{PY} {AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i {untune_file} -o {tune_file} -o2 {profile_file} --last"
        )

    def FinalFunc():
        logger.info(
            f"[Hint] tuned configs are saved in {tune_file}, you can set AITER_CONFIG_FMOE to this file to use tuned configs"
        )
        logger.info("\033[0m")

    def _lookup_cfg(c2s):
        if not c2s:
            return None
        primary, fallback = c2s
        lookup_keys = keys[:7] + (str(activation),) + keys[8:]
        result = primary.get(lookup_keys, None)
        if result is None and config_file is None:
            result = fallback.get(keys_disabled, None)
        # Tier fallback: if current tier not found, try smaller tiers in descending order
        if result is None and config_file is None and token > _PADDED_M_TIERS[0]:
            tier_idx = _PADDED_M_TIERS.index(token) if token in _PADDED_M_TIERS else -1
            for fallback_tier in reversed(_PADDED_M_TIERS[:tier_idx]):
                # keys layout: (gfx, cu_num, token, ...); replace token (idx 2).
                keys_fb = lookup_keys[:2] + (fallback_tier,) + lookup_keys[3:]
                keys_fb_disabled = (
                    keys_disabled[:2] + (fallback_tier,) + keys_disabled[3:]
                )
                result = primary.get(keys_fb, None)
                if result is None:
                    result = fallback.get(keys_fb_disabled, None)
                if result is not None:
                    break
        return result

    cfg = _lookup_cfg(active_cfg_2stages)
    if (
        cfg is None
        and config_file is None
        and os.environ.get("AITER_ONLINE_TUNE", "0") == "1"
    ):
        lock_name = re.sub(r"[^\w.\-]", "_", str(keys))
        lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{lock_name}")
        mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
        cfg_2stages = get_cfg_2stages(tune_file)
        cfg = _lookup_cfg(cfg_2stages)
        if cfg is None:
            logger.warning(f"Fmoe tuning not support for {keys}")

    if cfg is not None:
        kn1 = str(cfg.get("kernelName1", "") or "").strip()
        kn2 = str(cfg.get("kernelName2", "") or "").strip()
        has_opus = kn1.startswith("opus_") or kn2.startswith("opus_")
        if opus_weights_shuffled is None:
            opus_weights_shuffled = is_shuffled
        if has_opus and not opus_weights_shuffled:
            cfg = None
            logger.warning(
                f"[fused_moe] discarding Opus tuned config for {keys}: "
                "both w1 and w2 must be marked is_shuffled=True; using default "
                "heuristics"
            )
        elif kn1.startswith("opus_") and activation not in (
            ActivationType.Silu,
            ActivationType.Swiglu,
            ActivationType.Situv2,
        ):
            cfg = None
            logger.warning(
                f"[fused_moe] discarding Opus tuned config for unsupported "
                f"activation {activation}; using default heuristics"
            )
        elif (
            kn1.startswith("flydsl_") or kn2.startswith("flydsl_")
        ) and activation not in (
            ActivationType.Silu,
            ActivationType.Swiglu,
            ActivationType.Situv2,
        ):
            cfg = None
            logger.warning(
                f"[fused_moe] discarding FlyDSL tuned config for unsupported "
                f"activation {activation}; using default heuristics"
            )
        elif _disable_inline_sort and _is_inline_sort_cfg(kn1, kn2):
            cfg = None
            logger.warning("[fused_moe] discarding tuned inline-sort config")
        elif _is_inline_sort_cfg(kn1, kn2) and is_ep:
            cfg = None
            logger.warning(
                "[fused_moe] discarding tuned inline-sort config with expert_mask"
            )
        elif _is_inline_sort_cfg(kn1, kn2):
            inline_metadata = _make_mxfp4_metadata(
                kn1,
                kn2,
                gate_mode,
                cfg.get("ksplit", 0),
                block_m=cfg.get("block_m", BLOCK_SIZE_M),
                run_1stage=cfg.get("run_1stage", False),
                has_stage1_bias=has_stage1_bias,
                has_stage2_bias=has_stage2_bias,
            )
            if not _is_mxfp4_inline_sort(inline_metadata):
                cfg = None
                _disable_inline_sort = True
                logger.warning(
                    "[fused_moe] discarding tuned inline-sort config with "
                    "incomplete metadata"
                )

    if cfg is not None and _is_mxfp4_kname(kn1):
        parsed_g1 = _parse_mxfp4_g1_kname(kn1)
        configured_act = parsed_g1["act"]
        # Gelu/GeluTanh are real ActivationType values that MXMOE has no kernel
        # for. Folding them to "silu" would match a SiLU-tuned row and silently
        # run the wrong activation, so ask for a name that refuses instead.
        try:
            expected_act = get_flydsl_activation_name(activation)
        except ValueError:
            expected_act = None
        reject_reason = None
        if expected_act is None:
            reject_reason = f"no MXMOE kernel for activation {activation!r}"
        elif configured_act != expected_act:
            reject_reason = (
                f"activation {configured_act!r} does not match runtime {expected_act!r}"
            )
        elif swiglu_limit and expected_act not in ("silu", "swiglu"):
            reject_reason = (
                f"MXMOE cannot apply swiglu_limit={swiglu_limit!r} to "
                f"activation {expected_act!r}"
            )
        elif not aiter.is_mxfp4_moe_shape_supported(expert, model_dim, inter_dim, topk):
            reject_reason = (
                "generated MXFP4 auxiliary kernels do not cover "
                f"(expert={expert}, model_dim={model_dim}, "
                f"inter_dim={inter_dim}, topk={topk})"
            )
        elif has_stage1_bias and not parsed_g1.get("enable_bias", False):
            reject_reason = "stage1 bias is present but kernelName1 lacks '_bias'"
        elif has_stage2_bias and parse_flydsl_v2_gemm2_kernel(kn2) is None:
            reject_reason = (
                f"stage2 bias requires a flydsl_moe2_layout_ kernel, got {kn2!r}"
            )
        elif is_ep:
            reject_reason = "the MXMOE output_aux sort drops expert_mask"
        if reject_reason is not None:
            cfg = None
            logger.warning(
                f"[fused_moe] discarding MXMOE config: {reject_reason}; "
                f"using default heuristics"
            )

    if cfg is not None:
        kn2 = str(cfg.get("kernelName2", "") or "").strip()
        if cktile_mxfp4_unsafe and _is_cktile_mxfp4_stage2_name(kn2):
            cfg = None
            logger.warning(
                "[fused_moe] discarding unsafe CK-Tile MXFP4 stage2 config "
                f"for inter_dim={inter_dim}; using default heuristics"
            )
        elif kn2.startswith("opus_"):
            opus_supported, opus_reason = _opus_a8w4.cfg_is_supported(
                kn2,
                cfg=cfg,
                gfx=gfx,
                block_m=cfg.get("block_m", BLOCK_SIZE_M),
                is_ep=is_ep,
                has_stage2_bias=has_stage2_bias,
            )
            if not opus_supported:
                cfg = None
                logger.warning(
                    f"[fused_moe] Opus stage2 config unsupported ({opus_reason}); "
                    "using default heuristics"
                )

    bypass_tuned_config = int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0"))
    kernel_name1 = kn1 if cfg is not None else ""
    weights_shuffled = (
        is_shuffled if opus_weights_shuffled is None else opus_weights_shuffled
    )
    try:
        full_impl = (
            None if bypass_tuned_config else resolve_fused_moe_impl(kernel_name1)
        )
    except FusedMoeImplResolutionError as error:
        logger.warning(f"[fused_moe] {error}; using default heuristics.")
        cfg = None
        full_impl = None
    if full_impl is not None and kernel_name1.startswith("impl__flydsl_"):
        unsupported = None
        if not weights_shuffled:
            unsupported = "both w1 and w2 must be marked is_shuffled=True"
        elif has_stage1_bias or has_stage2_bias:
            unsupported = "per-expert bias"
        elif doweight_stage1:
            unsupported = "doweight_stage1=True"
        elif input_dtype is not None and input_dtype != dtypes.bf16:
            unsupported = f"activation dtype {input_dtype}"
        elif has_stage2_scatter:
            unsupported = "stage2_scatter"
        elif has_activation_scales:
            unsupported = "prequantized activations"
        elif has_num_local_tokens:
            unsupported = "num_local_tokens"
        elif hidden_pad or intermediate_pad:
            unsupported = "hidden/intermediate padding"
        elif gate_mode is not GateMode.SEPARATED:
            unsupported = f"gate mode {gate_mode.value!r}"
        elif activation not in (ActivationType.Silu, ActivationType.Swiglu):
            unsupported = f"activation {activation}"
        if unsupported is not None:
            cfg = None
            full_impl = None
            logger.warning(
                f"[fused_moe] discarding FlyDSL whole-graph config for {keys}: "
                f"unsupported {unsupported}; using default heuristics"
            )
    if is_ep and full_impl is not None:
        cfg = None
        full_impl = None
    if full_impl is not None:
        block_m = int(cfg.get("block_m", BLOCK_SIZE_M))
        ksplit = int(cfg.get("ksplit", 0))
        logger.info(f"[fused_moe] using {kernel_name1!r} for {keys}")
        return MOEMetadata(
            None,
            None,
            block_m,
            ksplit,
            full_impl=full_impl,
        )
    if config_file is not None and (cfg is None or bypass_tuned_config):
        raise NotImplementedError(
            "The dedicated FHMoE path requires an exact tuned config row for "
            f"{keys} in {tune_file}"
        )

    # Asm 1-stage kernels exist for Silu/Gelu and for gfx950 MXFP4 SiTUv2 FLAT.
    if (
        cfg is not None
        and cfg.get("run_1stage", False)
        and activation
        not in (ActivationType.Silu, ActivationType.Gelu, ActivationType.Situv2)
    ):
        cfg = None
        logger.warning(
            f"[fused_moe] discarding 1-stage tuned config for unsupported "
            f"activation {activation}; using default heuristics"
        )

    use_non_temporal_load = False
    if cfg is None or bypass_tuned_config:
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        run_1stage_xbf16 = False
        # No tuned config => default host moe_sort. For FLAT, run tuner and set flat=1.
        cfg_flat = 0
        if (
            activation,
            q_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            use_g1u1,
            doweight_stage1,
        ) in fused_moe_1stage_dict.get(get_gfx(), {}):
            if q_type == QuantType.per_1x128:
                xbf16 = (
                    get_gfx() == "gfx950"
                    and int(os.environ.get("AITER_XBFLOAT16", "0")) == 1
                )
                # for fp8 blockscale, ck has better performance so disable assembly kernel
                run_1stage = (
                    token > 32
                    and inter_dim % 128 == 0
                    and aiter.fmoe_fp8_blockscale_g1u1_has_kernel(
                        activation.value, xbf16, BLOCK_SIZE_M, inter_dim, model_dim
                    )
                    == 1
                )
                run_1stage_xbf16 = run_1stage and xbf16
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.i8:
                run_1stage = token > 32
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:
                run_1stage = token > 16 or inter_dim % 128 != 0
            elif q_type != QuantType.per_1x32:
                run_1stage = token < 256

        block_m = (
            BLOCK_SIZE_M
            if run_1stage
            else (
                # block_m=16 CK stage1 uses KPerBlock=256 (K=model_dim)
                (64 if token > 32 else 16 if model_dim % 256 == 0 else 32)
                if q_type == QuantType.per_1x128
                else get_block_size_M(token, topk, expert, inter_dim)
            )
        )
        ksplit = (
            ksplit
            if (run_1stage)
            else (
                get_ksplit(token, topk, expert, inter_dim, model_dim)
                if q_type in [QuantType.per_1x128, QuantType.per_1x32]
                else ksplit
            )
        )
        use_non_temporal_load = use_nt(token, topk, expert)
        aiter.logger.info(
            f"run_1stage = {run_1stage}, xbf16 = {run_1stage_xbf16}, ksplit = {ksplit} q_type = {q_type} block_m = {block_m} use_nt = {use_non_temporal_load}, estimated_m_per_expert = {token * topk // expert}"
        )
    else:
        block_m = cfg["block_m"]
        nt_override = int(os.environ.get("AITER_USE_NT", "-1"))
        if nt_override != -1:
            use_non_temporal_load = bool(nt_override)
        elif "nt" in cfg:
            try:
                use_non_temporal_load = bool(int(float(cfg["nt"])))
            except (TypeError, ValueError):
                # blank or malformed column: keep the pre-column behaviour
                use_non_temporal_load = False
        if int(os.environ.get("AITER_KSPLIT", "0")) != -1:
            ksplit = cfg["ksplit"]
        else:
            ksplit = 0
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
        run_1stage = cfg.get("run_1stage", False)
        if not is_shuffled and not run_1stage:
            logger.warning(
                f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
                "Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
                "Running with preshuffle_off may produce incorrect results."
            )
        if "xbf16" in cfg:
            run_1stage_xbf16 = run_1stage and bool(int(cfg["xbf16"]))
        else:
            run_1stage_xbf16 = run_1stage and "blockscaleBf16" in str(kernelName1)
        if "flat" in cfg:
            cfg_flat = int(cfg["flat"]) if run_1stage else 0
        else:
            cfg_flat = 0

    is_opus_cfg = cfg is not None and _opus_a8w4.is_opus_a8w4_stage2_kernel(
        cfg.get("kernelName2", "")
    )
    route_bucket_metadata = _opus_a8w4.route_bucket_metadata(cfg) if is_opus_cfg else {}
    opus_stage2_launch = (
        _opus_a8w4.parse_stage2_config(cfg, block_m) if is_opus_cfg else None
    )

    want_fp8_inter = (
        q_dtype_a2 == dtypes.fp8
        if q_dtype_a2 is not None
        else os.environ.get("AITER_SITUV2_A4W4_FP8_INTER", "0") == "1"
    )
    if (
        want_fp8_inter
        and q_dtype_a == dtypes.fp4x2
        and isinstance(kernelName2, str)
        and kernelName2.startswith("flydsl_moe2_layout_afp4_")
        and isinstance(kernelName1, str)
        and kernelName1.startswith("flydsl_moe1_")
    ):
        kernelName2 = kernelName2.replace("_afp4_", "_afp8_", 1)

    tag = f"({kernelName1=}, {kernelName2=})"
    logger.info(
        f"[fused_moe] using {'1stage' if run_1stage else '2stage'}{' xbf16' if run_1stage_xbf16 else ''} {'default' if cfg is None else tag} for {keys} "
    )

    def get_block_m() -> int:
        if q_dtype_a == dtypes.fp8:
            return 32
        else:
            return 16 if token < 2048 else 32 if token < 16384 else 64

    if _is_mxfp4_kname(kernelName1):
        # gate_mode is a runtime weight-layout property, not a tuning key: route
        # any a4w4 kernelName to the port; the bound interleave flag picks the
        # compiled il/sep variant at runtime.
        return _make_mxfp4_metadata(
            kernelName1,
            kernelName2,
            gate_mode,
            ksplit,
            has_stage1_bias=has_stage1_bias,
            has_stage2_bias=has_stage2_bias,
        )

    if run_1stage:
        # never hard code block_m for 1-stage since it can be tuned by kernel itself, and we have different heuristics for different quant types
        # # TODO: enable this approach for other quant types and archs
        # if q_type == QuantType.per_1x128 and get_gfx() == "gfx950":
        #     tkn_per_epr = token * topk // expert
        #     block_m = 64 if tkn_per_epr > 32 else block_m
        return MOEMetadata(
            functools.partial(
                fused_moe_1stage,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                xbf16=run_1stage_xbf16,
                flat=cfg_flat,
            ),
            None,
            block_m,
            ksplit,
            run_1stage,
            flat=cfg_flat,
            **route_bucket_metadata,
        )
    is_flydsl1 = isinstance(kernelName1, str) and kernelName1.startswith("flydsl_")
    is_flydsl2 = isinstance(kernelName2, str) and kernelName2.startswith("flydsl_")
    is_flydsl2_layout = isinstance(kernelName2, str) and kernelName2.startswith(
        "flydsl_moe2_layout_"
    )
    is_opus1 = isinstance(kernelName1, str) and kernelName1.startswith("opus_moe1_")
    is_cktile2 = isinstance(kernelName2, str) and kernelName2.startswith("cktile_")
    is_opus2 = _opus_a8w4.is_opus_a8w4_stage2_kernel(kernelName2)
    if is_opus1 or is_flydsl1 or is_flydsl2:
        enable_bias = (
            activation == ActivationType.Swiglu
            and _needs_swiglu_bias_support(dtype, q_type)
            and q_dtype_w == dtypes.fp4x2
        )
        _s1_fp4q = is_flydsl1 and "_fp4" in kernelName1.split("_t")[-1]
        if is_opus1:
            stage1_func = functools.partial(
                _opus_a8w4_stage1_wrapper,
                kernelName=kernelName1,
                activation=activation,
                block_m=block_m,
                inter_dim_pad=intermediate_pad,
            )
        elif is_flydsl1:
            stage1_func = functools.partial(
                _flydsl_stage1_wrapper,
                kernelName=kernelName1,
                activation=activation,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            )
        else:
            stage1_func = functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dtype=dtype,
                splitk=ksplit,
                use_non_temporal_load=use_non_temporal_load,
            )

        flydsl_v2_stage2_cfg = None
        if is_flydsl2_layout:
            flydsl_v2_stage2_cfg = parse_flydsl_v2_gemm2_kernel(kernelName2)
            if flydsl_v2_stage2_cfg is None:
                raise ValueError(f"Invalid FlyDSL v2 GEMM2 kernel name: {kernelName2}")
            stage2_func = functools.partial(
                _flydsl_v2_stage2_wrapper,
                kernelName=kernelName2,
                model_dim=model_dim,
                inter_dim=inter_dim,
                num_experts=expert,
            )
        elif is_flydsl2:
            stage2_func = functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kernelName2,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            )
        elif is_opus2:
            stage2_func = functools.partial(
                _opus_a8w4.opus_a8w4_stage2_wrapper,
                kernelName=kernelName2,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
                launch=opus_stage2_launch,
            )
        elif is_cktile2:
            stage2_func = functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            )
        else:
            stage2_func = functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type,
                use_non_temporal_load=use_non_temporal_load,
            )
        _s1_fp8q = is_opus1 or (is_flydsl1 and "_fp8" in kernelName1.split("_t")[-1])
        _fuse_quant = "fp8" if _s1_fp8q else ("fp4" if _s1_fp4q else "")
        v2_scatter = False
        if flydsl_v2_stage2_cfg is not None:
            stage1_func.keywords["out_dtype"] = flydsl_v2_stage2_cfg["a_dtype"]
            _fuse_quant = flydsl_v2_stage2_cfg["a_dtype"]
            v2_scatter = flydsl_v2_stage2_cfg["epilog"] == "scatter"
        return MOEMetadata(
            stage1_func,
            stage2_func,
            block_m,
            int(ksplit),
            run_1stage,
            has_bias=enable_bias and (is_opus1 or is_flydsl1),
            fuse_quant=_fuse_quant,
            stage2_has_bias=enable_bias and (is_flydsl2 or is_cktile2),
            skip_inter_quant="_moe2_layout_" in str(kernelName2),
            output_aux=AUX_SORT_OPUS if v2_scatter else False,
            **route_bucket_metadata,
        )
    # CK-Tile's 2-stage MXFP4 stage-2 (moe_cktile2stages_gemm2) reduces over
    # inter_dim and indexes its e8m0 weight scales in groups of 8 blocks
    # (8 * 32 = 256 elements). When inter_dim is not a multiple of 256 the host
    # pads the scale group dimension (shuffle_scale rounds it up to a multiple
    # of 8) and the kernel reads the wrong groups, silently returning a badly
    # wrong result. Stage-1 is unaffected -- it reduces over model_dim.
    # Correct shapes keep their existing routing. Unsafe shapes leave CK-Tile
    # only when the target FlyDSL kernel accepts the shape and weight layout.
    # Calls with no safe fallback are rejected before the CK-Tile launch.
    cktile_mxfp4_ok = not (cktile_mxfp4_unsafe and flydsl_can_take_over)
    if (
        gate_mode != GateMode.SEPARATED
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
        and q_dtype_w == dtypes.fp4x2
        and (gate_up_interleave is None or gate_up_interleave is True)
        and cktile_mxfp4_ok
    ):
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=1,
                dtype=dtype,
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            get_block_m(),
            ksplit,
            run_1stage=False,
            has_bias=has_stage1_bias,
            stage2_has_bias=has_stage2_bias,
        )
    swiglu_mxfp4_bf16_cktile = (
        q_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
        and q_dtype_w == dtypes.fp4x2
        and is_shuffled
        and cktile_mxfp4_ok
        and (
            gate_up_interleave is False
            # Callers that never tag w1.gate_up_interleave (None) keep the rule
            # from before #6478: bf16/fp16 activations on the standard layout.
            or (gate_up_interleave is None and q_dtype_a in [dtypes.bf16, dtypes.fp16])
        )
    )
    if (
        activation == ActivationType.Relu2
        and not use_g1u1
        and q_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
        and q_dtype_a == dtypes.bf16
        and dtype == dtypes.bf16
        and not has_stage2_scatter
        and not has_activation_scales
        and bool(opus_weights_shuffled)
        and get_gfx_runtime() == "gfx950"
    ):
        _cktile_block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64,
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=1,
                dtype=dtype,
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            _cktile_block_m,
            1,
            run_1stage,
            has_bias=False,
            stage2_has_bias=False,
        )
    if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
        # Untuned a16wi4 fallback: one shape-safe config on the shared a16w-mix port.
        # Tiles belong in the tuned CSV, not in a heuristic here. ksplit is 0 because
        # the port has no grid split-K (it uses intra-block k_wave); asking for it
        # would set up partials the kernel never writes. block_m MUST equal tile_m --
        # it sizes both moe_sorting and the gemm tile.
        _out_str = "bf16"
        _tile_m = 16 if token < 2048 else 32 if token < 16384 else 64
        _tile_n = _tile_k = 128
        from aiter.ops.flydsl.moe_kernels import flydsl_kernel_name

        kn1 = flydsl_kernel_name(1, "bf16", "int4", _out_str, _tile_m, _tile_n, _tile_k)
        kn2 = flydsl_kernel_name(
            2, "bf16", "int4", _out_str, _tile_m, _tile_n, _tile_k, "atomic"
        )
        logger.warning(
            f"[fused_moe] no tuned FlyDSL config for {keys}, "
            f"using untuned a16wi4 fallback ({kn1=}, {kn2=})"
        )
        return MOEMetadata(
            functools.partial(
                _flydsl_stage1_wrapper,
                kernelName=kn1,
                activation=activation,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            ),
            functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kn2,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            ),
            _tile_m,
            0,
            False,
        )
    # Debug: AITER_FLYDSL_FORCE=1 is for debug use.
    _flydsl_force = os.environ.get("AITER_FLYDSL_FORCE", "1") == "1"
    # a16w4-SiTUv2 (bf16 A x mxfp4 W) -> ported 2-stage path; SiTUv2 gate distinguishes gpt-oss (Swiglu -> cktile).
    _is_a16w4_situv2 = (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Situv2
        and q_dtype_a == dtypes.bf16
        and q_dtype_w == dtypes.fp4x2
        and is_shuffled
        and use_g1u1
        and not doweight_stage1
    )
    # bf16 A x mxfp4 W with Swiglu normally takes the CK-Tile branch above. When
    # cktile_mxfp4_ok is False that branch is skipped, and the shape needs the
    # same a16w4 FlyDSL kernels as SiTUv2 -- there is no A quantisation either
    # way, so _a_type below is "bf16" for both. Gated on `not cktile_mxfp4_ok`,
    # so it is False for every shape CK-Tile already handles.
    _is_a16w4_swiglu_rerouted = (
        not cktile_mxfp4_ok
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
        and q_dtype_a == dtypes.bf16
        and q_dtype_w == dtypes.fp4x2
        and is_shuffled
        and use_g1u1
        and not doweight_stage1
    )
    _is_a16w4 = _is_a16w4_situv2 or _is_a16w4_swiglu_rerouted
    use_mxfp4_flydsl = _is_a16w4 or (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and (
            activation in (ActivationType.Swiglu, ActivationType.Situv2)
            or _flydsl_force
        )
        and (
            q_dtype_a in (dtypes.fp4x2, dtypes.fp8)
            and q_dtype_w in (dtypes.fp4x2, dtypes.fp8)
        )
        and _flydsl_mxfp4_layout_is_compatible(q_dtype_a, gate_mode, gate_up_interleave)
        and is_shuffled
        and use_g1u1
        and not doweight_stage1
    )
    # The fallback's layout GEMM2 writes bf16 only, and its output_aux sort drops expert_mask.
    _mxmoe_fallback_ok = (
        dtype == dtypes.bf16
        and not is_ep
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Situv2
        and q_dtype_a == dtypes.fp4x2
        and q_dtype_w == dtypes.fp4x2
        and is_shuffled
        and use_g1u1
        and not doweight_stage1
        and gate_mode != GateMode.INTERLEAVE
        and not (has_stage1_bias or has_stage2_bias)
        and hidden_pad == 0
        and intermediate_pad == 0
        and model_dim % 256 == 0
        and inter_dim % 128 == 0
        and aiter.is_mxfp4_moe_shape_supported(expert, model_dim, inter_dim, topk)
        and os.environ.get("AITER_MXMOE_FALLBACK", "1") == "1"
    )
    if _mxmoe_fallback_ok and cfg is None:
        _bm = 64 if token < 512 else 128
        _rows_per_expert = -(-token * topk // expert)
        _g1_swz = min(6, max(1, -(-_rows_per_expert // _bm)))
        _g1_sfx = f"_xcd{_g1_swz}" if _g1_swz > 1 else ""
        _kn1 = f"flydsl_mxmoe_g1_a4w4_{_bm}x256x256_situv2{_g1_sfx}"
        _kn2 = f"flydsl_moe2_layout_afp4_wfp4_bf16_t{_bm}x256x128_reduce_sbm{_bm}"
        logger.warning(
            f"[fused_moe] no tuned FlyDSL config for {keys}, "
            f"using heuristic MXMOE fallback (kn1={_kn1!r}, kn2={_kn2!r})"
        )
        return _make_mxfp4_metadata(_kn1, _kn2, gate_mode, 0, block_m=_bm)

    if use_mxfp4_flydsl:
        from aiter.ops.flydsl.moe_kernels import (
            flydsl_kernel_name,
            get_flydsl_kernel_params,
            pick_flydsl_stage2_tile_k,
        )

        _out_type = "bf16" if dtype == dtypes.bf16 else "f16"
        # a-dtype routing:
        #   "bf16" => a16w4 bf16/fp4 (no A quant; SiTUv2, or Swiglu re-routed
        #             off CK-Tile because inter_dim is not 256-aligned)
        #   "fp4"  => a4w4 fp4/fp4
        #   "fp8"  => a8w4 fp8/fp4 (or a8w8 with w=fp8)
        if _is_a16w4:
            _a_type = "bf16"
        elif q_dtype_a == dtypes.fp4x2:
            _a_type = "fp4"
        else:
            _a_type = "fp8"
        # w-dtype "fp4" => mxfp4 weight; "fp8" => mxfp8 weight (a8w8).
        _w_type = "fp8" if q_dtype_w == dtypes.fp8 else "fp4"
        _s2_tk = pick_flydsl_stage2_tile_k(inter_dim)
        # Per token tier: (tile_m, stage1 suffix, stage2 suffix).
        if token < 2048:
            _tile_m, _s1_sfx, _s2_sfx = 32, "_w2", "_bnt2"
        elif token < 4096:
            _tile_m, _s1_sfx, _s2_sfx = 64, "_w3_bnt0", ""
        elif token < 16384:
            _tile_m, _s1_sfx, _s2_sfx = 128, "_w2_bnt0", ""
        else:
            _tile_m, _s1_sfx, _s2_sfx = 64, "_w4_bnt0", ""
        _base_kn1 = flydsl_kernel_name(
            1, _a_type, _w_type, _out_type, _tile_m, 128, 256
        )
        _base_kn2 = flydsl_kernel_name(
            2, _a_type, _w_type, _out_type, _tile_m, 128, _s2_tk, "atomic"
        )
        kn1 = f"{_base_kn1}{_s1_sfx}"
        kn2 = f"{_base_kn2}{_s2_sfx}"

        # fp8 stage1 kernel names always carry a "_gui" suffix
        # (moe_kernels.py:114-115). Append it before the lookup so the fp8
        # variants resolve; fp4 names are unchanged.
        if _a_type == "fp8":
            kn1 = f"{kn1}_gui"
            _base_kn1 = f"{_base_kn1}_gui"
        if get_flydsl_kernel_params(kn1) is None:
            kn1 = _base_kn1
        if get_flydsl_kernel_params(kn2) is None:
            kn2 = _base_kn2

        logger.warning(
            f"[fused_moe] no tuned FlyDSL config for {keys}, "
            f"using heuristic FlyDSL fallback ({kn1=}, {kn2=})"
        )
        enable_bias = (
            activation == ActivationType.Swiglu
            and _needs_swiglu_bias_support(dtype, q_type)
        )
        return MOEMetadata(
            functools.partial(
                _flydsl_stage1_wrapper,
                kernelName=kn1,
                activation=activation,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            ),
            functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kn2,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            ),
            _tile_m,
            -1,  # split_k = -1
            False,
            has_bias=enable_bias,
            stage2_has_bias=enable_bias,
        )
    if (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and q_dtype_w in [dtypes.fp4x2]
        and is_shuffled
        and not (activation == ActivationType.Swiglu and q_dtype_a == dtypes.fp4x2)
        and (ksplit > 1 or swiglu_mxfp4_bf16_cktile)
    ):
        # GPT-OSS Swiglu can use bf16/fp16 activations for small batches while
        # keeping the generic preshuffled fp4 weights. CK2stages has no
        # heuristic kernel for that A16W4 combination, so use CK-Tile.
        # Use CK-Tile's split-k epilogue for the generic preshuffled MXFP4
        # layout. The non-split gate/up epilogue is reserved for legacy A16W4.
        _min_split_k = 2 if swiglu_mxfp4_bf16_cktile else 1
        _split_k = max(int(ksplit), _min_split_k)
        _cktile_block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=_split_k,
                dtype=dtype,
                post_activation_layout=(
                    "standard" if swiglu_mxfp4_bf16_cktile else "auto"
                ),
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            _cktile_block_m,
            _split_k,
            run_1stage,
            has_bias=has_stage1_bias,
            stage2_has_bias=has_stage2_bias,
        )

    if (
        activation == ActivationType.Swiglu
        and q_dtype_w == dtypes.fp4x2
        and q_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and not kernelName1
    ):
        logger.warning(
            "[fused_moe] SwiGLU MXFP4 with unshuffled weights not supported "
            "by CK2stages codegen; routing to CK-Tile (ROCM-25478)"
        )
        _split_k = max(int(ksplit), 2)
        _cktile_block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=_split_k,
                dtype=dtype,
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            _cktile_block_m,
            _split_k,
            run_1stage,
            has_bias=has_stage1_bias,
            stage2_has_bias=has_stage2_bias,
        )

    if (kernelName1 and "ck2stages" in kernelName1) or (
        not kernelName1
        and (
            (q_type == QuantType.per_1x128 and doweight_stage1)
            or q_dtype_w
            in [
                dtypes.bf16,
                dtypes.fp16,
                torch.uint32,
                dtypes.fp4x2,
                dtypes.fp8,
            ]
        )
    ):
        if kernelName2 and kernelName2.startswith("flydsl_"):
            stage2_func = functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kernelName2,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            )
        elif _opus_a8w4.is_opus_a8w4_stage2_kernel(kernelName2):
            stage2_func = functools.partial(
                _opus_a8w4.opus_a8w4_stage2_wrapper,
                kernelName=kernelName2,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
                launch=opus_stage2_launch,
            )
        elif kernelName2 and kernelName2.startswith("cktile_"):
            stage2_func = functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            )
        else:
            stage2_func = functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type,
                use_non_temporal_load=use_non_temporal_load,
            )
        return MOEMetadata(
            functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dtype=dtype,
                splitk=int(ksplit),
                use_non_temporal_load=use_non_temporal_load,
            ),
            stage2_func,
            block_m,
            int(ksplit),
            run_1stage,
            **route_bucket_metadata,
        )

    # TODO: remove when stage2 support more size
    tmpList = [16, 32, 64, 128]
    if block_m not in tmpList:
        tag = ""
        block_m = ([el for el in tmpList if block_m < el] + [128])[0]

    if _opus_a8w4.is_opus_a8w4_stage2_kernel(kernelName2):
        stage2_func = functools.partial(
            _opus_a8w4.opus_a8w4_stage2_wrapper,
            kernelName=kernelName2,
            inter_dim_pad=intermediate_pad,
            model_dim_pad=hidden_pad,
            launch=opus_stage2_launch,
        )
    elif kernelName2 and kernelName2.startswith("cktile_"):
        stage2_func = functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=activation,
        )
    else:
        stage2_func = functools.partial(
            aiter.ck_moe_stage2_fwd,
            kernelName=kernelName2,
            activation=activation,
            quant_type=q_type,
            use_non_temporal_load=use_non_temporal_load,
        )
    return MOEMetadata(
        functools.partial(
            asm_stage1,
            kernelName=kernelName1,
            activation=activation,
            quant_type=q_type,
        ),
        stage2_func,
        block_m,
        ksplit,
        run_1stage,
        **route_bucket_metadata,
    )


_get_2stage_cfgs_cache_clear = get_2stage_cfgs.cache_clear


def _clear_dispatch_caches():
    # aiter.moev2 caches the same decisions for the calls it runs.
    _get_2stage_cfgs_cache_clear()
    moev2_tuned.clear_cache()


get_2stage_cfgs.cache_clear = _clear_dispatch_caches


def fused_moe_2stages(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_out,
    isG1U1,
    block_size_M,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: torch.Tensor | None = None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    topk_ids=None,
    topk_weights=None,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode=GateMode.SEPARATED.value,
    expert_mask=None,
    m_indices=None,
    reverse_sorted=None,
    _metadata_transform: Callable | None = None,
    _metadata_config_file: str | None = None,
    _stage1_extra_args: dict | None = None,
    _stage2_extra_args: dict | None = None,
    output=None,
    _stage2_override: Callable | None = None,
    routing_num_experts: int | None = None,
    prequantized_a1: torch.Tensor | None = None,
    prequantized_a1_scale: torch.Tensor | None = None,
):
    quant_func = get_quant(quant_type)
    gate_mode = GateMode(gate_mode)
    token_num, _ = hidden_states.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = moe_out.dtype
    device = hidden_states.device
    _sort_moe_buf = moe_out
    if moe_out.numel() == 0:
        # Reduce mode: the real output is born here, so it can be the caller's
        # buffer. Accumulate mode is handled in moe_sorting (zeroed there).
        moe_out = (
            output
            if output is not None
            else torch.empty((token_num, model_dim), dtype=dtype, device=device)
        )
    is_shuffled = getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
    config_situ_beta, config_situ_linear_beta, normalized_swiglu_limit = (
        _normalize_mxfp4_activation_params(activation, beta, linear_beta, swiglu_limit)
    )
    metadata = get_2stage_cfgs(
        get_padded_M(token_num),  # consider token_num > 1024 as prefill
        model_dim,
        inter_dim,
        E,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        quant_type,
        isG1U1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        is_shuffled,
        gate_mode,
        is_ep=expert_mask is not None,
        has_stage1_bias=bias1 is not None,
        has_stage2_bias=bias2 is not None,
        situ_beta=config_situ_beta,
        situ_linear_beta=config_situ_linear_beta,
        swiglu_limit=swiglu_limit,
        opus_weights_shuffled=getattr(w1, "is_shuffled", False)
        and getattr(w2, "is_shuffled", False),
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
        config_file=_metadata_config_file,
        input_dtype=hidden_states.dtype,
    )
    if _metadata_transform is not None:
        metadata = _metadata_transform(metadata)
    if (
        getattr(metadata.stage1, "func", metadata.stage1) is _mxfp4_a4w4_stage1_fw
        and metadata.output_aux == AUX_SORT_OPUS
        # Prequant is a property of GEMM1, not of the sort: block_m 16 is the
        # only inline-quant ("_f16in") a4w4 variant, and that kernel reads raw
        # bf16 hidden_states and ignores the A buffers entirely. Handing it a
        # prequantized fp4 A plus scales faults with an illegal address. Keep
        # this keyed on block_m rather than on _aux_uses_opus, which now sends
        # block_m 16 to Opus for the sort alone.
        and int(metadata.block_m) != 16
    ):
        # Main's fused prequant already writes the E8M0 scale layout consumed by
        # replacement GEMM1 on the default Opus auxiliary path.
        metadata = replace(metadata, prequant=True)
    if prequantized_a1 is not None:
        if quant_type != QuantType.per_1x32:
            raise ValueError(
                "prequantized_a1 is only supported for QuantType.per_1x32, "
                f"got {quant_type}"
            )
        if prequantized_a1_scale is None:
            raise ValueError(
                "prequantized_a1_scale must be supplied alongside prequantized_a1"
            )
        a1 = prequantized_a1
        a1_scale = prequantized_a1_scale
    elif not metadata.prequant:
        a1 = hidden_states
        a1_scale = None
    elif (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and (
                activation
                in (ActivationType.Swiglu, ActivationType.Situv2, ActivationType.Relu2)
                or gate_mode == GateMode.INTERLEAVE
            )
            or (q_dtype_a in [dtypes.fp4x2] and metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a1 = hidden_states.to(dtype)
        a1_scale = None
    elif (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype in (dtypes.fp4x2, dtypes.fp8)
    ):
        # mxfp8 activations + mxfp4 weights (a8w4) OR mxfp8 weights (a8w8).
        if _MOE_A8W4_BYPASS_QUANT:
            # Debug bypass: skip real quant, feed unit scales.
            a1 = hidden_states.to(dtypes.fp8)
            M = sorted_ids.shape[0]
            a1_scale = torch.ones(
                [M, a1.shape[-1] // 32], dtype=dtypes.fp8_e8m0, device=a1.device
            )
        elif hidden_states.dtype == dtypes.fp8 and a1_scale is not None:
            # MXFP8 dispatch passthrough: activations already carry group-32 e8m0
            # scales, so skip the quant and only sort the scale, exactly as the
            # fp4x2 pre-quantized branch below does.
            a1 = hidden_states
            a1_scale = mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
        else:
            # stage1 input is not topk-replicated, so M==token_num and the HIP
            # launcher infers TOPK=1 from input.numel() / (cols * token_num).
            a1, a1_scale = fused_dynamic_mxfp8_quant_moe_sort(
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

    elif quant_type == QuantType.per_1x32 and w1.dtype == dtypes.i4x2:
        # a16wi4: bf16 activations, int4 weights; no activation quantization needed
        a1 = hidden_states.to(dtype)
        a1_scale = None
    elif quant_type == QuantType.per_1x32:
        if hidden_states.dtype == dtypes.fp4x2 and a1_scale is not None:
            # Input is already quantized to fp4x2 (e.g., from FP4 dispatch),
            # skip re-quantization, only sort the scale
            a1 = hidden_states
            a1_scale = mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
        else:
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
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
    elif hidden_states.dtype != q_dtype_a:
        if quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
            quant_func = functools.partial(quant_func, transpose_scale=True)
        a1, a1_scale = quant_func(
            hidden_states,
            scale=a1_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
        )
    else:
        assert (
            a1_scale is not None or quant_type == QuantType.No
        ), "a1_scale must be provided for quantized input for fused_moe"
        a1 = hidden_states
    # a16w4 (bf16 A x mxfp4 W) SiTUv2: stage1 allocates its own sorted
    # [sorted_size, inter_dim] bf16 intermediate and ignores this `out` buffer, so
    # skip the throwaway (token_num, topk, inter_dim) alloc (up to ~tens of MB at
    # prefill). Gate matches the a16w4 branch of _flydsl_moe_stage1_impl.
    _is_a16w4_port = (
        quant_type == QuantType.per_1x32
        and dtype in (dtypes.bf16, dtypes.fp16)
        and getattr(w1, "dtype", None) == dtypes.fp4x2
        and q_dtype_a == dtypes.bf16
        and getattr(metadata.stage1, "func", metadata.stage1) is _flydsl_stage1_wrapper
    )
    if _is_a16w4_port:
        a2 = None
    elif quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        ratio = a1_scale.element_size() // a1.element_size()
        a2 = torch.empty(
            (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
            dtype=q_dtype_a,
            device=device,
        )
    else:
        _a2_dtype = dtype
        a2 = torch.empty(
            (token_num, topk, inter_dim),
            dtype=_a2_dtype,
            device=device,
        )
    extra_stage1_args = dict(_stage1_extra_args or {})
    extra_stage2_args = dict(_stage2_extra_args or {})
    need_bias_support = _needs_swiglu_bias_support(dtype, quant_type)
    stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
    stage2_func = getattr(metadata.stage2, "func", metadata.stage2)
    if not metadata.run_1stage and need_bias_support:
        if metadata.has_bias:
            extra_stage1_args["bias1"] = _normalize_bias_for_kernel(bias1)
            if stage1_func in (_flydsl_stage1_wrapper, cktile_moe_stage1):
                extra_stage1_args["topk_ids"] = topk_ids
        if metadata.stage2_has_bias:
            extra_stage2_args["bias2"] = _normalize_bias_for_kernel(bias2)
    if stage1_func in (_flydsl_stage1_wrapper, _opus_a8w4_stage1_wrapper):
        # Hand these two the caller's limit unchanged. They clamp silu whenever a
        # finite limit is configured (runtime_swiglu_limit in moe_kernels.py), and
        # the torch reference does the same, so normalizing non-Swiglu to None
        # would silently drop the clamp on paths outside this change.
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
    # EP: forward expert_mask + topk_ids to the flydsl stage2 wrapper so it can
    # switch to reduce mode and fuse the validity gather in compile_moe_reduction.
    if (
        stage2_func
        in (
            _flydsl_stage2_wrapper,
            _flydsl_v2_stage2_wrapper,
        )
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
            extra_stage2_args["topk_weights"] = topk_weights
    if stage2_func is _opus_a8w4.opus_a8w4_stage2_wrapper:
        extra_stage2_args["stage2_fp8_enabled"] = _opus_stage2_fp8_enabled()
    if m_indices is not None:
        extra_stage1_args["m_indices"] = m_indices
        extra_stage1_args["moe_buf"] = _sort_moe_buf
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
        w1_scale=(
            # Only one-byte packed scales may be reinterpreted as E8M0.
            w1_scale.view(dtypes.fp8_e8m0)
            if w1.dtype in (dtypes.fp4x2, dtypes.fp8)
            and w1_scale is not None
            and w1_scale.element_size() == 1
            else w1_scale
        ),
        sorted_weights=sorted_weights if doweight_stage1 else None,
        **extra_stage1_args,
    )
    if kernel_bench_callable is not None:
        kernel_bench_callable.append(("stage1", _stage1_call))
    a2 = _stage1_call()
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
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and activation
            in (ActivationType.Swiglu, ActivationType.Situv2, ActivationType.Relu2)
            or (metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a2_scale = None
    elif (
        quant_type == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype in (dtypes.fp4x2, dtypes.fp8)
    ):
        # a8w4 / mxfp8: quantize stage1 output to mxfp8 for stage2's fp8 operand.
        if not _MOE_A8W4_BYPASS_QUANT:
            a2 = a2.view(-1, inter_dim)
            a2, a2_scale = fused_dynamic_mxfp8_quant_moe_sort(
                a2,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                topk=topk,
                block_size=block_size_M,
                num_rows=num_local_tokens,
                sorted_weights=sorted_weights,
                num_experts_upper_bound=routing_num_experts,
            )
            a2 = a2.view(token_num, topk, -1)
        else:
            a2 = a2.to(dtypes.fp8)
            a2_scale = a1_scale
    elif quant_type == QuantType.per_1x32 and w1.dtype == dtypes.i4x2:
        # a16wi4: stage1 output is bf16, no inter-stage quantization
        a2_scale = None
    elif quant_type == QuantType.per_1x32:
        a2 = a2.view(-1, inter_dim)
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M,
            num_rows=num_local_tokens,
            sorted_weights=sorted_weights,
            num_experts_upper_bound=routing_num_experts,
        )
        a2 = a2.view(token_num, topk, -1)
    elif quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        a2_v = a2[:token_num, :, :]
        a2_scale = (
            a2[token_num:, ...]
            .view(-1)[: token_num * topk * inter_dim * ratio // 128]
            .view(dtypes.fp32)
            .view(token_num, -1)
        )
        a2 = a2_v
    else:
        a2, a2_scale = quant_func(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2 = a2.view(token_num, topk, inter_dim)

    stage2_sorted_weights = sorted_weights if not doweight_stage1 else None
    stage2_args = (
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
    )
    stage2_kwargs = dict(
        w2_scale=(
            # See stage1 w1_scale note: only reinterpret packed (e8m0) scales;
            # per_Token fp8 uses an fp32 scale and must be passed through as-is
            # (PR #3811 regression fix).
            w2_scale.view(dtypes.fp8_e8m0)
            if w2.dtype in (dtypes.fp4x2, dtypes.fp8)
            and w2_scale is not None
            and w2_scale.element_size() == 1
            else w2_scale
        ),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=stage2_sorted_weights,
        **extra_stage2_args,
    )
    if _stage2_override is None:
        _stage2_call = functools.partial(
            metadata.stage2,
            *stage2_args,
            **stage2_kwargs,
        )
    else:
        _stage2_call = functools.partial(
            _stage2_override,
            ordinary_stage2=metadata.stage2,
            stage2_args=stage2_args,
            stage2_kwargs=stage2_kwargs,
        )
    if kernel_bench_callable is not None:
        kernel_bench_callable.append(("stage2", _stage2_call))
    stage2_output = _stage2_call()
    return moe_out if _stage2_override is None else stage2_output
