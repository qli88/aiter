# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

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
from aiter.fhmoe_contract import (  # noqa: F401
    FHMOE_BUFFER_MAX_BYTES,
    fhmoe_reduce_output_nbytes,
)
from aiter.fused_moe_registry import (  # noqa: F401
    BoundFusedMoeImpl,
    FusedMoeImplResolutionError,
    FusedMoeRequest,
    make_fused_moe_impl_kernel_name,
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

# NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
# Please do not modify this code without coordinating with the moev2 owners
# (@qli88). See aiter/moev2/README.md.
# Besides the moev2 hooks, paths and tuned-table modules, the names below moved from
# here to aiter.moev2 and are re-exported under their old names; the legacy code below
# runs most of them. Changes must keep legacy behavior
# (python tests/moe_dispatch/snapshot.py --check).
from aiter.moev2 import hooks as moev2_hooks
from aiter.moev2 import paths as moev2_paths
from aiter.moev2.dispatch import tuned as moev2_tuned
from aiter.moev2.dispatch.tuned import _ACT_TYPE_DISABLED_KEY
from aiter.moev2.execute.metadata import AsmMoeMode, MOEMetadata
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
    _clamp_swiglu_gate_up,
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
from aiter.moev2.segments import sort as moev2_sort
from aiter.moev2.segments.asm import (  # noqa: F401
    _ASM_ACT_TAG,
    _retarget_asm_activation,
    asm_stage1,
    fused_moe_1stage,
)
from aiter.moev2.segments.ck import _require_shuffled_gfx950_fp8, ck_moe_stage1
from aiter.moev2.segments.ck_tile import (  # noqa: F401
    _cktile_gemm1_k_batch,
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
    _STAGE2_SCRATCH,
    _can_reroute_mxfp4_to_flydsl,
    _empty_u8,
    _flydsl_mxfp4_layout_is_compatible,
    _flydsl_route_output_nbytes,
    _flydsl_stage1_wrapper,
    _flydsl_stage2_wrapper,
    _flydsl_v2_stage2_per_slot,
    _flydsl_v2_stage2_wrapper,
    _get_flydsl_moe_kernels,
    _get_padding_for_flydsl,
    _is_flydsl_stage2,
    _mxfp4_scale_u8,
    _needs_swiglu_bias_support,
    _normalize_bias_for_kernel,
    _stage2_honors_runtime_reduce_override,
    _stage2_runtime_uses_route_reduce,
    _stage2_scratch,
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
    _mxmoe_gemm2_sort_block_m,
    _padded_scale_cols,
)
from aiter.moev2.segments.opus import _opus_stage2_fp8_enabled

# Re-exported for existing imports. Tests and benchmarks that override the sort flags or
# helpers must patch aiter.moev2.segments.sort, which every caller below reads.
from aiter.moev2.segments.sort import (  # noqa: F401
    _FUSED_DECODE_SORT_QUANT_MAX_M,
    _FUSED_DECODE_SORT_QUANT_ROUTES,
    _MOE_SORT_BACKEND,
    _USE_CK_MOE_SORTING,
    _USE_FLYDSL_MOE_SORTING,
    AUX_SORT_OPUS,
    _adaptive_moe_sort,
    _adaptive_output_usable,
    _aux_uses_opus,
    _empty_bf16,
    _flydsl_moe_sorting,
    _fused_decode_sort_quant,
    _moe_prepare_unsorted_input,
    _moe_sorting_impl,
    _mxfp4_aux_instance_supported,
    moe_sorting,
)
from aiter.moev2.shapes import (  # noqa: F401
    _PADDED_M_TIERS,
    _is_row_major_2d,
    get_inter_dim,
    get_padded_M,
    nextPow2,
)
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
    _flydsl_stage2_fp8_enabled,
    _mxfp4_intermediate_enabled,
    get_flydsl_activation_name,
)
from aiter.ops.flydsl.mxfp4_kname import (  # noqa: F401
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    _parse_mxfp4_g2_kname,
    native_scale_layout_for,
    parse_flydsl_v2_gemm2_kernel,
    parse_g2_kname_any,
)
from aiter.ops.moe_limits import RUNTIME_SILU_LIMIT_QUANT_TYPES, normalize_swiglu_limit
from aiter.ops.moe_mxfp4_aux import (  # noqa: F401
    _MissingMxfp4MoeAuxInstances,
    is_mxfp4_moe_scatter_supported,
)
from aiter.ops.opus import moe_stage2_a8w4 as _opus_a8w4
from aiter.ops.opus.moe_stage1_a8w4 import (
    opus_a8w4_stage1_wrapper as _opus_a8w4_stage1_wrapper,
)
from aiter.plugin import route as _route
from aiter.presorted_moe import (  # noqa: F401  register_presorted_moe is public
    aux_sort_capacity,
    bytes_overlap,
    drop_presort_after,
    drop_presort_on_error,
    match_presorted_moe,
    pop_presorted_moe,
    presorted_moe_ignored,
    register_presorted_moe,
)

_SILU_VALUE = int(ActivationType.Silu)


def swiglu_limit_tag(swiglu_limit):
    """`_tag` value of tuned fmoe rows that were tuned with this swiglu_limit."""
    swiglu_limit = normalize_swiglu_limit(swiglu_limit)
    return f"swiglu_limit={float(swiglu_limit):g}" if swiglu_limit else ""


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


# FLAT 1stage asm kernels ingest raw topk_ids / topk_weights through the
# sorted_* kernarg slots and accumulate via global_atomic_pk_add_bf16, so
# moe_sorting is a pass-through for them. The manifest value is flat=1
# (one token per TG) or flat=2 (EMSORT); either nonzero value takes this path.


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


@drop_presort_after
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
    herd_topk: int = 0,
    herd_routed_scaling_factor: float = 1.0,
    deterministic: bool = False,
    # NVFP4 weights: fp32 1 / weight_global_scale, [E, 2] (gate, up) and [E]
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
):
    """Run fused MoE, dispatching to FHMoE when shared-expert tensors are present.

    ``deterministic=True`` gives bitwise the same output for a token whatever the
    batch, and from run to run: the FlyDSL kernels with one K structure for
    every token count, no split-K, and the non-atomic reduce epilogue in stage 2
    (the top-k partials are summed in a fixed order). It changes only the order
    of the sums. Because the default picks the activation dtype by token count,
    which would make a token's result depend on its batch, the caller chooses
    one activation dtype with ``quant_dtype_a``, which is required with
    ``deterministic=True``: today ``dtypes.fp8`` (MXFP8, the precision the
    default uses from 256 tokens). It is supported on gfx950 for MXFP4 weights
    (``QuantType.per_1x32``, preshuffled gate/up-interleaved) with SwiGLU or SiLU, bf16
    output, without expert parallelism.

    Supplying any shared tensor (or ``shared_expert_id``) requires all four
    shared weight/scale tensors and dispatches through the gfx950 FHMoE path.
    Inputs must satisfy the contracts documented in :mod:`aiter.fhmoe`.

    ``clamp_shared`` is an FHMoE model-semantic input. The default ``True``
    applies the configured SiLU clamp to routed and shared experts. ``False``
    changes only the shared expert to unclamped SiLU; routed experts still use
    ``swiglu_limit``. Clamped and unclamped stage-1 kernels have distinct AOT
    cache identities, so deployments changing this setting must rebuild their
    FlyDSL AOT cache.

    ``w1_global_scale`` / ``w2_global_scale`` with fp4x2 weights select the
    gfx950 NVFP4 MoE (:mod:`aiter.ops.moe_nvfp4`): w1 / w2 in the
    ``shuffle_weight(w, (16, 32))`` layout, E4M3 ``w1_scale`` / ``w2_scale`` in
    the ``shuffle_scale_nvfp4_moe`` layout.
    """
    if herd_topk and (w1_global_scale is not None or w2_global_scale is not None):
        raise ValueError("herd_topk does not support NVFP4 global scales")
    # NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
    # Please do not modify this code without coordinating with the moev2 owners
    # (@qli88). See aiter/moev2/README.md.
    # Entry "api": an enabled path may run this call on aiter.moev2.
    # aiter.moev2.api.op_call copies how this function calls fused_moe_; mirror
    # changes to those arguments there.
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
        if herd_topk:
            raise ValueError("herd_topk does not support fused shared experts")
        if deterministic:
            raise NotImplementedError(
                "fused_moe(deterministic=True) does not support shared experts"
            )
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
        if w1_global_scale is not None or w2_global_scale is not None:
            unsupported.append("w1_global_scale / w2_global_scale")
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
        splitk=splitk,
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
        herd_topk=herd_topk,
        herd_routed_scaling_factor=herd_routed_scaling_factor,
        deterministic=deterministic,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
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
    splitk: int = 0,
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
    herd_topk: int = 0,
    herd_routed_scaling_factor: float = 1.0,
    deterministic: bool = False,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
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


def fused_moe_g1u1_bf16_small_m_direct(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    model_dim = w2.shape[1]
    inter_dim = w2.shape[2]
    if output is None:
        output = torch.empty((M, model_dim), dtype=dtype, device=hidden_states.device)
    act_workspace = torch.empty(
        (M * topk, inter_dim), dtype=dtype, device=hidden_states.device
    )
    aiter.fmoe_g1u1_bf16_small_m(
        output,
        hidden_states.contiguous(),
        w1.contiguous(),
        w2.contiguous(),
        topk_ids.contiguous(),
        topk_weight.contiguous(),
        act_workspace,
        topk,
        bool(getattr(w1, "is_shuffled", False)),
        bool(getattr(w2, "is_shuffled", False)),
    )
    return output


_BF16_G1U1_SMALL_M_DIRECT_SHUFFLE_LAYOUT = (16, 16)
_BF16_G1U1_SMALL_M_DIRECT_ALLOWED_M = (1, 2, 3, 4, 8, 16)
_BF16_G1U1_SMALL_M_DIRECT_MODEL_DIM = 2048
_BF16_G1U1_SMALL_M_DIRECT_INTER_DIMS = (128, 256)
_BF16_G1U1_SMALL_M_DIRECT_EXPERTS = 256
_BF16_G1U1_SMALL_M_DIRECT_TOPK = 8


def _supports_bf16_g1u1_small_m_direct_shuffle(weight: torch.Tensor) -> bool:
    if not getattr(weight, "is_shuffled", False):
        return True
    return (
        getattr(weight, "aiter_shuffle_layout", None)
        == _BF16_G1U1_SMALL_M_DIRECT_SHUFFLE_LAYOUT
        and not getattr(weight, "gate_up_interleave", False)
        and not getattr(weight, "aiter_shuffle_is_guinterleave", False)
    )


def _can_use_bf16_g1u1_small_m_direct(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    dtype: torch.dtype,
    doweight_stage1: bool,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    num_local_tokens: torch.Tensor | None,
    moe_sorting_dispatch_policy: int,
    expert_mask: torch.Tensor | None,
    hidden_pad: int,
    intermediate_pad: int,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    splitk: int = 0,
    swiglu_limit: float | None = None,
    beta: float | None = None,
    linear_beta: float | None = None,
    gate_mode: str | GateMode | None = GateMode.SEPARATED.value,
    stage2_scatter: Stage2ScatterContext | None = None,
) -> bool:
    M = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    inter_dim = w2.shape[2]
    return (
        get_gfx_runtime() == "gfx1201"
        and quant_type == QuantType.No
        and activation == ActivationType.Silu
        and dtype == dtypes.bf16
        and M in _BF16_G1U1_SMALL_M_DIRECT_ALLOWED_M
        and topk == _BF16_G1U1_SMALL_M_DIRECT_TOPK
        and inter_dim in _BF16_G1U1_SMALL_M_DIRECT_INTER_DIMS
        and hidden_states.shape == (M, _BF16_G1U1_SMALL_M_DIRECT_MODEL_DIM)
        and w1.shape
        == (
            _BF16_G1U1_SMALL_M_DIRECT_EXPERTS,
            2 * inter_dim,
            _BF16_G1U1_SMALL_M_DIRECT_MODEL_DIM,
        )
        and w2.shape
        == (
            _BF16_G1U1_SMALL_M_DIRECT_EXPERTS,
            _BF16_G1U1_SMALL_M_DIRECT_MODEL_DIM,
            inter_dim,
        )
        and topk_weight.shape == topk_ids.shape
        and hidden_states.is_cuda
        and w1.is_cuda
        and w2.is_cuda
        and topk_ids.is_cuda
        and topk_weight.is_cuda
        and w1.device == hidden_states.device
        and w2.device == hidden_states.device
        and topk_ids.device == hidden_states.device
        and topk_weight.device == hidden_states.device
        and hidden_states.dtype == dtypes.bf16
        and w1.dtype == dtypes.bf16
        and w2.dtype == dtypes.bf16
        and topk_ids.dtype == dtypes.i32
        and topk_weight.dtype == dtypes.fp32
        and expert_mask is None
        and stage2_scatter is None
        and not doweight_stage1
        and w1_scale is None
        and w2_scale is None
        and a1_scale is None
        and a2_scale is None
        and num_local_tokens is None
        and moe_sorting_dispatch_policy == 0
        and hidden_pad == 0
        and intermediate_pad == 0
        and bias1 is None
        and bias2 is None
        and splitk in (0, 1)
        and swiglu_limit is None
        and beta is None
        and linear_beta is None
        and gate_mode in (None, GateMode.SEPARATED, GateMode.SEPARATED.value)
        and _supports_bf16_g1u1_small_m_direct_shuffle(w1)
        and _supports_bf16_g1u1_small_m_direct_shuffle(w2)
    )


# Version 2: v1 ended at `deterministic`; v2 adds `w1_global_scale` and
# `w2_global_scale`, which turn fp4x2 weights from MXFP4 into NVFP4. A v1
# provider built for 40 arguments cannot bind 42, and a forwarder that binds
# them would serve an NVFP4 call as MXFP4, so v1 is not superseded.
@torch_compile_guard(gen_fake=fused_moe_fake)
@_route(op="fused_moe", version=2)
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
    splitk: int = 0,
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
    herd_topk: int = 0,
    herd_routed_scaling_factor: float = 1.0,
    deterministic: bool = False,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route each token to its top-k experts, run both expert projections and
    combine them, weighted by ``topk_weight``.

    Returns a new ``[M, model_dim]`` tensor of ``dtype`` (``hidden_states``'
    dtype when None), ``M = topk_ids.shape[0]``, ``model_dim = w2.shape[1]``.
    Given ``output`` -- contiguous, of exactly that shape, dtype and device, not
    overlapping ``hidden_states`` -- the result is written into it and
    ``output`` itself is returned. Nothing else is written through the
    arguments, except under the stage-2 scatter below.

    This is the layer beneath the public ``fused_moe``, whose signature stays
    as callers know it. That function sends a call with shared experts (any
    ``shared_*``) to FHMoE without coming here, turns ``block_size_M=None``
    into -1 and unpacks ``stage2_scatter`` into the ``ep_*`` fields. What
    arrives here, for a provider:

    * ``activation`` and ``quant_type`` are ints, the ``.value`` of
      ``ActivationType`` and ``QuantType``; so is ``quant_type_a`` when given.
    * ``block_size_M=-1`` leaves the tile to AITER; any other value pins it.
      ``moe_sorting_dispatch_policy`` only schedules AITER's own sort.
    * ``ep_source_token_map`` set is expert parallelism's stage-2 scatter,
      described by the other ``ep_*`` fields (all 0 without it): GEMM2
      P2P-writes the route-weighted rows into the peers' combine buffers, and
      the returned tensor is an uninitialised placeholder that does not hold
      the result. ``output`` is refused with it. A provider that does not do
      that write must decline these calls.
    * The activation dtype is not in the signature. AITER chooses it from the
      weights, ``activation``, ``gate_mode`` and the token count M, and the
      environment moves it: ``AITER_SITUV2_A8W4`` / ``AITER_SITUV2_A4W4``
      (fp8 / fp4 SiTUv2 activations, bf16 otherwise),
      ``AITER_BF16_FP8_MOE_BOUND`` (bf16 below that many tokens, fp8 from it)
      and ``GPTOSS_SWIGLU_MXFP4_BF16_BOUND`` (bf16 below, fp4 from it).
      ``resolve_activation_dtype`` answers it as this op does, and
      ``quant_dtype_a`` overrides it.
    * ``splitk`` is the caller's split-K request; 0 and 1 are no split. AITER
      reads it here only to keep its gfx1201 small-M kernel off split calls,
      and it moves no more than the order of the K sums, so a provider may
      ignore it.
    * ``swiglu_limit`` of None, 0 (vLLM's spelling) or inf is no clamp for
      SiLU.
    * ``deterministic=True`` asks for a result bitwise independent of the
      batch and of the run (see the public ``fused_moe``). A provider whose
      reduction order is not fixed must decline it.
    * ``herd_topk`` (gfx1250) says ``topk_ids`` and ``topk_weight`` hold K+1
      HERD candidates per token, scaled by ``herd_routed_scaling_factor``;
      with ``herd_topk=0`` that factor is unused.
    * ``w1_global_scale`` / ``w2_global_scale`` set (fp32 ``[E, 2]`` gate/up
      and ``[E]``, each ``1 / weight_global_scale``) make fp4x2 weights NVFP4:
      E4M3 per-16 block scales in ``w1_scale`` / ``w2_scale``, multiplied by
      the per-expert global scale. Both None is MXFP4 as before. A provider
      without NVFP4 must decline calls that set them.

    A caller may hand the next call on its thread a precomputed sort with
    ``register_presorted_moe``. The registration is consumed below this route,
    on every native call, and the public ``fused_moe`` clears it when the call
    ends, so a call that a provider serves leaves none behind. A provider may
    ignore it. A caller of this layer directly gets no such clean-up.
    """
    # NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
    # Please do not modify this code without coordinating with the moev2 owners
    # (@qli88). See aiter/moev2/README.md.
    # Entry "op": an enabled path may run this call on aiter.moev2 instead of
    # _fused_moe_impl; DECLINED hands it back.
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
        splitk=splitk,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        stage2_scatter=stage2_scatter,
        output=output,
        quant_type_a=quant_type_a,
        quant_dtype_a=quant_dtype_a,
        quant_dtype_a2=quant_dtype_a2,
        herd_topk=herd_topk,
        herd_routed_scaling_factor=herd_routed_scaling_factor,
        deterministic=deterministic,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
    )


# NOTE: aiter.moev2 (under development, not graduated) keeps a copy of this logic: the
# validation and derived values, the get_2stage_cfgs key and the launches, in
# aiter/moev2/execute/executor.py (prologue, run_two_stage, run_whole_graph) and
# aiter/moev2/api.py (dispatch_key). Mirror changes there or coordinate with the moev2
# owners (@qli88), then run the tests/moe_dispatch checks in aiter/moev2/README.md.
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
    splitk: int = 0,
    swiglu_limit: float | None = None,
    beta: float | None = None,
    linear_beta: float | None = None,
    gate_mode: str = GateMode.SEPARATED.value,
    stage2_scatter: Stage2ScatterContext | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: int | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    herd_topk: int = 0,
    herd_routed_scaling_factor: float = 1.0,
    deterministic: bool = False,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
    *,
    _q_dtype_a: torch.dtype | None = None,
    _metadata_transform: Callable | None = None,
    _metadata_config_file: str | None = None,
    _stage1_extra_args: dict | None = None,
    _stage2_extra_args: dict | None = None,
    _stage2_override: Callable | None = None,
    _enforce_metadata_block_size: bool = False,
    _config_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    # Consume a caller's presort first, so every path (including early returns
    # and errors) clears it.
    presorted_aux = pop_presorted_moe()
    # vLLM sends 0.0 for "no limit". Check floats and the raw int first:
    # pybind enum comparisons are much slower than int compares.
    silu_limit = (
        swiglu_limit is not None
        and swiglu_limit != 0
        and swiglu_limit != math.inf
        and (
            activation == _SILU_VALUE
            if type(activation) is int
            else ActivationType(activation) == ActivationType.Silu
        )
    )
    # We do such convert since custom_op schema restriction on block_size_M, and Enum type
    activation = ActivationType(activation)
    quant_type = QuantType(quant_type)
    if silu_limit:
        quant_type, quant_type_a, swiglu_limit = _normalize_moe_dispatch_contract(
            quant_type, activation, swiglu_limit, quant_type_a
        )
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
    herd_candidate_ids, herd_candidate_weights = topk_ids, topk_weight
    if herd_topk:
        # Only the gfx1250 integration passes K+1 candidates to MoE.
        # Other architectures finish HERD in the router and pass K routes.
        if get_gfx() != "gfx1250":
            raise ValueError("herd_topk is supported only on gfx1250")
        if topk != herd_topk + 1 or topk_weight.shape != topk_ids.shape:
            raise ValueError(
                "herd_topk requires K+1 candidate IDs and weights per token"
            )
        if expert_mask is not None or stage2_scatter is not None or doweight_stage1:
            raise ValueError(
                "HERD routing requires unmasked routes and doweight_stage1=False"
            )
        from aiter.ops.flydsl.kernels.moe_herd_route_psum import HERD_ROUTE_BLOCK

        if M > HERD_ROUTE_BLOCK:
            raise ValueError(
                f"herd_topk takes at most {HERD_ROUTE_BLOCK} tokens, got {M}; "
                "AITER_FLYDSL_HERD_MAX_M must not exceed it"
            )
        topk = herd_topk
    config_M, config_topk = (M, topk) if _config_shape is None else _config_shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if not hidden_states.is_contiguous() and not _is_row_major_2d(hidden_states):
        hidden_states = hidden_states.contiguous()

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]
    isShuffled = getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
    # gfx1250: GUGU row-interleave, not 16-row block shuffle.
    if (
        get_gfx() in ("gfx1250", "gfx1250-strict")
        and isG1U1
        and gate_mode != GateMode.INTERLEAVE
    ):
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
    if (w1_global_scale is not None or w2_global_scale is not None) and (
        q_dtype_w != dtypes.fp4x2 or gate_mode != GateMode.SEPARATED
    ):
        raise ValueError(
            "w1_global_scale / w2_global_scale (NVFP4 MoE) need fp4x2 weights and"
            f" gate_mode={GateMode.SEPARATED.value!r}, got {q_dtype_w} and"
            f" {gate_mode.value!r}"
        )
    q_dtype_a = resolve_activation_dtype(
        quant_type,
        q_dtype_w,
        activation=activation,
        gate_mode=gate_mode,
        M=config_M,
        hidden_dtype=hidden_states.dtype,
        has_a1_scale=a1_scale is not None,
        gfx=get_gfx_runtime(),
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
    )

    if quant_dtype_a is not None:
        q_dtype_a = quant_dtype_a
    elif _q_dtype_a is not None:
        q_dtype_a = _q_dtype_a

    if deterministic:
        # The default picks the activation dtype by token count (bf16, then fp4
        # or fp8 above a bound), which would make results depend on the batch.
        # The caller fixes one dtype for every token count; it is quantized per
        # 1x32 block within a token's row, so a row's result does not depend on
        # the other rows.
        if quant_dtype_a is None:
            raise ValueError(
                "fused_moe(deterministic=True) needs quant_dtype_a: the activation "
                "dtype for every token count (dtypes.fp8, MXFP8, is supported)"
            )
        q_dtype_a = quant_dtype_a
        _check_deterministic_supported(
            dtype=dtype,
            quant_type=quant_type,
            q_dtype_w=q_dtype_w,
            q_dtype_a=q_dtype_a,
            q_dtype_a2=quant_dtype_a2,
            activation=activation,
            gate_mode=gate_mode,
            gate_up_interleave=getattr(w1, "gate_up_interleave", None),
            # isShuffled is true when either weight is preshuffled; the
            # deterministic kernels need both.
            is_shuffled=getattr(w1, "is_shuffled", False)
            and getattr(w2, "is_shuffled", False),
            is_g1u1=isG1U1,
            doweight_stage1=doweight_stage1,
            expert_mask=expert_mask,
            stage2_scatter=stage2_scatter,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_size_M=block_size_M,
            tokens=M,
            model_dim=model_dim,
            inter_dim=inter_dim,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            has_bias1=bias1 is not None,
            has_bias2=bias2 is not None,
            weight_bytes=max(
                w1.numel() * w1.element_size(), w2.numel() * w2.element_size()
            ),
        )

    grouped_a8w4_out = None
    from aiter.ops.flydsl.grouped_moe_gfx1250 import grouped_gemm_gfx1250_a8w4

    default_situ_beta, default_situ_linear_beta = (
        (DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA)
        if activation == ActivationType.Situv2
        else (1.0, 1.0)
    )

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
            situ_beta=default_situ_beta if beta is None else float(beta),
            situ_linear_beta=(
                default_situ_linear_beta if linear_beta is None else float(linear_beta)
            ),
            stage2_scatter=stage2_scatter,
            herd_topk=herd_topk,
            herd_routed_scaling_factor=herd_routed_scaling_factor,
            output=output,
        )

    if grouped_a8w4_out is not None:
        return _return_output(grouped_a8w4_out, output)
    if herd_topk and get_gfx() == "gfx1250":
        raise ValueError(
            "gfx1250 HERD needs the grouped MoE path (per_1x32 MXFP4 weights, "
            "gate_mode=INTERLEAVE)"
        )

    if _can_use_bf16_g1u1_small_m_direct(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        activation=activation,
        quant_type=quant_type,
        dtype=dtype,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        expert_mask=expert_mask,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        stage2_scatter=stage2_scatter,
        splitk=splitk,
    ):
        return fused_moe_g1u1_bf16_small_m_direct(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            dtype=dtype,
            output=output,
        )

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

    def _resolve_metadata(disable_inline_sort=False, exclude_mxmoe=False):
        metadata = get_2stage_cfgs(
            get_padded_M(config_M),
            model_dim,
            inter_dim,
            E,
            config_topk,
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
            _exclude_mxmoe=exclude_mxmoe,
            q_dtype_a2=quant_dtype_a2,
            input_dtype=hidden_states.dtype,
            has_stage2_scatter=stage2_scatter is not None,
            has_activation_scales=a1_scale is not None or a2_scale is not None,
            _has_input_activation_scales=a1_scale is not None,
            has_num_local_tokens=num_local_tokens is not None,
            has_global_scale=w1_global_scale is not None or w2_global_scale is not None,
        )
        return (
            metadata if _metadata_transform is None else _metadata_transform(metadata)
        )

    if deterministic:
        metadata = _deterministic_metadata(
            M, activation, dtype, quant_type, hidden_pad, intermediate_pad
        )
    else:
        metadata = _resolve_metadata(disable_inline_sort=bool(herd_topk))
        reason = ""
        if _is_mxmoe_opus_metadata(metadata):
            reason = _mxmoe_opus_call_unsupported(
                metadata,
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                a1_scale,
                topk_ids,
                block_size_M=block_size_M,
            )
        elif (
            getattr(metadata.stage1, "func", metadata.stage1) is _mxfp4_a4w4_stage1_fw
            and not _is_mxfp4_inline_sort(metadata)
            and block_size_M is not None
            and int(block_size_M) != int(metadata.block_m)
        ):
            reason = f"block_size_M={block_size_M} differs from configured BM={metadata.block_m}"
        if reason:
            logger.warning(
                f"[fused_moe] MXMOE pair is unsupported ({reason}); "
                "using default heuristics"
            )
            metadata = _resolve_metadata(
                disable_inline_sort=bool(herd_topk), exclude_mxmoe=True
            )
            if block_size_M is not None and int(block_size_M) != int(metadata.block_m):
                raise NotImplementedError(
                    f"No fallback backend supports block_size_M={block_size_M}; "
                    f"resolved backend requires block_m={metadata.block_m}"
                )
        if herd_topk and (
            metadata.flat or metadata.output_aux or metadata.full_impl is not None
        ):
            raise ValueError(
                "gfx950 HERD needs a MoE backend using the standard sorted-token layout"
            )

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
    if not use_inline_sort and not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

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
    # a16w16 tiles size moe_sorting and both gemms. A caller block_size_M that
    # disagrees used to let stage1 run on the wrong layout; stage2 then raised
    # FlyDSL stage2 sorting layout mismatch. Reject here, same as FHMoE.
    # Not a general FlyDSL rule: some a16w4 names advertise sort_block_m !=
    # tile_m via _sbm. a16w16 has no such suffix, so tile_m is the sort size.
    _a16w16_kn1 = str(
        getattr(metadata.stage1, "keywords", {}).get("kernelName", "") or ""
    )
    if (
        _a16w16_kn1.startswith("flydsl_moe1_abf16_wbf16_")
        and block_size_M is not None
        and int(block_size_M) != int(metadata.block_m)
    ):
        raise ValueError(
            f"a16w16 block_size_M={block_size_M} does not match kernel tile "
            f"{metadata.block_m}; moe_sorting and both gemm tiles share this size"
        )
    block_size_M = metadata.block_m if block_size_M is None else block_size_M
    if _stage2_override is not None and (
        metadata.run_1stage or metadata.full_impl is not None
    ):
        raise RuntimeError("_stage2_override requires a two-stage MoE config")
    if presorted_aux and not metadata.output_aux:
        presorted_moe_ignored("the selected MoE pipeline takes no BM=16 aux sort")
    asm_mode = AsmMoeMode(metadata.flat)
    if metadata.full_impl is not None:
        request = FusedMoeRequest(
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
            w1_global_scale=w1_global_scale,
            w2_global_scale=w2_global_scale,
        )
        full_call = functools.partial(metadata.full_impl, request)
        # NOTE: aiter.moev2 is under active development (MoE v2 migration, not
        # graduated). Please do not modify this code without coordinating with the moev2
        # owners (@qli88). See aiter/moev2/README.md.
        # The parity harness (tests/moe_dispatch) compares this launch with v2's.
        if kernel_bench_callable is not None:
            kernel_bench_callable.append(("whole_graph", full_call))
        full_output = full_call()
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
    assert asm_mode.requires_host_sort or get_gfx() in (
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
        need_local_topk_ids=need_local_topk_ids,
        run_1stage=metadata.run_1stage,
        stage1_is_flydsl=stage1_func is _flydsl_stage1_wrapper,
        hidden_pad=hidden_pad,
        ksplit=metadata.ksplit,
    )
    fused_sort_quant = fused_sort_quant and not herd_topk
    # compact-scale variant not ported (its stage-1 change targets the
    # refactored mixed_moe_gemm_2stage_common); keep the fused sort+quant on
    # the expanded-scale path only.
    fused_compact_scale = False
    if herd_topk:
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            topk_ids,
            topk_weight,
        ) = moev2_sort._flydsl_moe_sorting(
            herd_candidate_ids,
            herd_candidate_weights,
            global_E,
            model_dim,
            dtype,
            block_size_M,
            None,
            None,
            accumulate=not runtime_stage2_reduce,
            output=output,
            herd_topk=herd_topk,
            herd_routed_scaling_factor=herd_routed_scaling_factor,
        )
        local_topk_ids = None
    elif metadata.output_aux:
        _stage2_kwargs = metadata.stage2.keywords
        _kn2 = _stage2_kwargs.get("kernelName2") or _stage2_kwargs.get("kernelName", "")
        _is_opus = _is_mxmoe_opus_metadata(metadata)
        _g2 = None if _is_opus else parse_g2_kname_any(_kn2)
        if expert_mask is not None and (_is_opus or not _g2["v2"]):
            raise NotImplementedError(
                f"MXMOE GEMM2 {_kn2!r} does not support expert-parallel; "
                "use a flydsl_moe2_layout_ GEMM2."
            )
        _atomic = (
            not stage2_uses_route_reduce(metadata.stage2) if _is_opus else _g2["atomic"]
        )
        # The presort layout has to match both the selected kernels' tile size
        # and the block size the stages are given, which a caller's
        # block_size_M can override.
        presorted = match_presorted_moe(
            presorted_aux,
            topk_ids,
            topk_weight,
            block_m=metadata.block_m,
            num_experts=global_E,
            topk=topk,
            stage_block_m=block_size_M,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            output=output,
        )
        if presorted is not None:
            sorted_ids = presorted["sorted_token_ids"]
            sorted_weights = presorted["sorted_weights"]
            sorted_expert_ids = presorted["sorted_expert_ids"]
            num_valid_ids = presorted["num_valid_ids"]
            sort_m_indices = presorted["m_indices"]
            sort_reverse_sorted = presorted["reverse_sorted"]
            if _atomic and not runtime_stage2_reduce:
                # Atomic GEMM2 accumulates with 4-byte atomics and faults on an
                # output at an odd bf16 element, so such an output gets a fresh
                # moe_buf and _return_output copies back.
                in_place = output is not None and output.data_ptr() % 4 == 0
                moe_buf = (
                    output
                    if in_place
                    else torch.empty(
                        (M, model_dim), dtype=dtype, device=topk_ids.device
                    )
                )
                # Atomic GEMM2 needs a zeroed destination; skip the zeroing only
                # if the caller zeroed this exact output. A buffer allocated here
                # can reuse a freed block at the same address, so always zero it.
                if not (
                    in_place
                    and presorted.get("moe_buf_zeroed")
                    and int(presorted.get("moe_buf_ptr", 0)) == output.data_ptr()
                ):
                    moe_buf.zero_()
            else:
                moe_buf = torch.empty((0, 0), dtype=dtype, device=topk_ids.device)
            local_topk_ids = None
        else:
            # BM16's adaptive sort already emits routes and zeroes the output
            # without quantizing. Keep the Opus crossover for the configured aux
            # pipeline.
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
            num_local_tokens=num_local_tokens,
        )
        local_topk_ids = None
    else:
        sort_output = output if asm_mode.requires_host_sort else None
        if asm_mode is AsmMoeMode.POST_REDUCE:
            # The reducer overwrites the final output, including masked rows.
            # An empty buffer also suppresses the sort's EP zero-fill.
            sort_output = torch.empty((0, 0), dtype=dtype, device=topk_ids.device)
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
            accumulate=not runtime_stage2_reduce
            and asm_mode is not AsmMoeMode.POST_REDUCE,
            flat=asm_mode,
            output=sort_output,
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
        if asm_mode is AsmMoeMode.POST_REDUCE:
            moe_buf = (
                output
                if output is not None
                else torch.empty((M, model_dim), dtype=dtype, device=topk_ids.device)
            )
    _opus_a8w4.check_route_bucket_metadata(metadata, sorted_expert_ids, logger)

    if metadata.run_1stage:
        if silu_limit and normalize_swiglu_limit(swiglu_limit) is not None:
            raise NotImplementedError(
                f"1-stage MoE cannot apply swiglu_limit={swiglu_limit!r}; "
                "the selected kernel has no clamp."
            )
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
            expert_mask=expert_mask,
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
            deterministic=deterministic,
            output=output,
            _stage2_override=_stage2_override,
            routing_num_experts=global_E,
            prequantized_a1=prequantized_a1,
            prequantized_a1_scale=prequantized_a1_scale,
        )
        return _return_output(ret, output)


@dataclass
class MoeStage2Inputs:
    """Returned by ``fused_moe_prepare_stage2`` for ``fused_moe_run_stage2``.
    Only the stage1 output ``a2`` / ``a2_scale`` is public."""

    # One row per route or per sorted slot; the last dim is the intermediate.
    a2: torch.Tensor
    a2_scale: torch.Tensor | None
    _stage2: Callable
    # (w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, moe_out, topk);
    # moe_out is the [token_num, model_dim] output prepared by the sorting.
    _args: tuple
    _kwargs: dict
    # Set once stage2 has written moe_out.
    _moe_out_written: bool = False


def _stage2_override_kind(stage2: Callable) -> str | None:
    """Return "v1" or "v2" if ``stage2`` is a FlyDSL MXFP4 kernel that
    supports per-slot output and replaced operands, else None."""
    func = getattr(stage2, "func", stage2)
    kernel_name = getattr(stage2, "keywords", {}).get("kernelName", "")
    if func is _flydsl_stage2_wrapper:
        params = _get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel_name)
        fp4 = params is not None and params["a_dtype"] == params["b_dtype"] == "fp4"
        return "v1" if fp4 else None
    if func is _flydsl_v2_stage2_wrapper:
        cfg = parse_flydsl_v2_gemm2_kernel(kernel_name)
        fp4 = cfg is not None and cfg["a_dtype"] == cfg["b_dtype"] == "fp4"
        return "v2" if fp4 and cfg["epilog"] != "scatter" else None
    return None


def supports_stage2_overrides(metadata: "MOEMetadata") -> bool:
    """Whether ``fused_moe_run_stage2`` supports ``per_slot`` and replaced
    operands for ``metadata`` (from ``get_2stage_cfgs``). Nothing is run."""
    if (
        metadata.run_1stage
        or metadata.full_impl is not None
        or metadata.stage2 is None
        or metadata.output_aux
        or metadata.stage2_has_bias
        # Split-K stage1 leaves the intermediate unquantized.
        or metadata.ksplit > 1
    ):
        return False
    return _stage2_override_kind(metadata.stage2) is not None


def _captured_stage2_supports_overrides(inputs: "MoeStage2Inputs") -> bool:
    """``supports_stage2_overrides`` for a captured stage2: split-K leaves
    ``a2_scale`` None, and a stage2 bias or the aux sort is in the kwargs."""
    kwargs = inputs._kwargs
    return (
        _stage2_override_kind(inputs._stage2) is not None
        and inputs.a2_scale is not None
        and kwargs.get("bias2") is None
        and kwargs.get("reverse_sorted") is None
    )


def _scaled_dim(dim: int, cols: int, ref_cols: int) -> int:
    """Logical width of ``cols`` stored columns, if ``ref_cols`` stored
    columns hold ``dim`` elements."""
    if (dim * cols) % ref_cols:
        raise ValueError(
            f"{cols} stored columns do not hold whole elements "
            f"({ref_cols} hold {dim})"
        )
    return dim * cols // ref_cols


def _check_stage2_operands(
    inputs: "MoeStage2Inputs",
    w2: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a2: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
) -> None:
    """Raise ``ValueError`` if a replacing stage2 operand does not fit the
    one it replaces (see ``fused_moe_run_stage2``). Shuffled E8M0 scales pad
    rows and groups, so only their minimum size is checked."""
    old_a2, old_a2_scale = inputs.a2, inputs.a2_scale
    old_w2 = inputs._args[1]
    old_w2_scale = inputs._kwargs.get("w2_scale")
    new_a2 = old_a2 if a2 is None else a2
    new_w2 = old_w2 if w2 is None else w2

    def fail(msg: str) -> None:
        raise ValueError(f"fused_moe_run_stage2: {msg}")

    for name, new in (
        ("w2", w2),
        ("w2_scale", w2_scale),
        ("a2", a2),
        ("a2_scale", a2_scale),
    ):
        if new is None:
            continue
        if new.device != old_a2.device:
            fail(f"{name} is on {new.device}, stage1 ran on {old_a2.device}")
        if not new.is_contiguous():
            fail(f"{name} is not contiguous")
    for name, new in (("w2_scale", w2_scale), ("a2_scale", a2_scale)):
        if new is not None and new.element_size() != 1:
            fail(f"{name} is a {new.dtype} tensor, not E8M0")
    for name, new, old in (("a2", a2, old_a2), ("w2", w2, old_w2)):
        if new is not None and (new.dtype != old.dtype or new.dim() != old.dim()):
            fail(
                f"{name} is a {new.dim()}-d {new.dtype} tensor; the one it "
                f"replaces is a {old.dim()}-d {old.dtype} tensor"
            )
    if a2 is not None and a2.shape[:-1] != old_a2.shape[:-1]:
        fail(
            f"a2 has rows {tuple(a2.shape[:-1])}, stage1 produced "
            f"{tuple(old_a2.shape[:-1])}"
        )
    if w2 is not None and w2.shape[0] != old_w2.shape[0]:
        fail(f"w2 has {w2.shape[0]} experts, stage1 ran {old_w2.shape[0]}")
    # The last dims of w2 and a2 are both the reduction dim (K).
    if new_w2.shape[-1] * old_a2.shape[-1] != new_a2.shape[-1] * old_w2.shape[-1]:
        fail(
            f"w2 last dim {new_w2.shape[-1]} does not match a2 last dim "
            f"{new_a2.shape[-1]} (stage1: {old_w2.shape[-1]} and "
            f"{old_a2.shape[-1]})"
        )
    if a2 is not None and a2.shape[-1] != old_a2.shape[-1] and a2_scale is None:
        fail("an a2 of another width needs its a2_scale")
    if w2 is not None and w2.shape != old_w2.shape and w2_scale is None:
        fail("a w2 of another shape needs its w2_scale")
    if w2_scale is not None:
        if w2 is None and (
            old_w2_scale is None or w2_scale.shape != old_w2_scale.shape
        ):
            fail(
                f"w2_scale has shape {tuple(w2_scale.shape)}, the one it "
                f"replaces {None if old_w2_scale is None else tuple(old_w2_scale.shape)}"
            )
        # [E * model_dim, groups] or [E, model_dim, groups], rows padded.
        scale_rows = math.prod(w2_scale.shape[:-1])
        if scale_rows < new_w2.shape[0] * new_w2.shape[1]:
            fail(
                f"w2_scale has {scale_rows} rows, w2 needs at least "
                f"{new_w2.shape[0] * new_w2.shape[1]}"
            )
    if a2_scale is not None and (
        old_a2_scale is None or a2_scale.shape[0] != old_a2_scale.shape[0]
    ):
        fail(
            f"a2_scale has {a2_scale.shape[0]} rows, stage1 produced "
            f"{None if old_a2_scale is None else old_a2_scale.shape[0]}"
        )
    # One E8M0 scale per 32 FP4 values (2 per byte) of K.
    groups = -(-new_a2.shape[-1] * 2 // 32)
    for name, new in (("w2_scale", w2_scale), ("a2_scale", a2_scale)):
        if new is not None and new.shape[-1] < groups:
            fail(
                f"{name} has {new.shape[-1]} groups, the reduction dim needs "
                f"{groups}"
            )


@drop_presort_on_error
def fused_moe_prepare_stage2(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask: torch.Tensor | None = None,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M=None,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = GateMode.SEPARATED.value,
    *,
    quant_type_a: QuantType | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    deterministic: bool = False,
    config_shape: tuple[int, int] | None = None,
) -> MoeStage2Inputs:
    """Run the sorting and stage1 of ``fused_moe`` and return the stage2
    inputs, so the caller can work on them (e.g. gather ``a2`` across ranks)
    before ``fused_moe_run_stage2``. Not related to ``fused_moe_1stage``.

    Takes the arguments of ``fused_moe`` up to ``gate_mode`` (same order and
    defaults), plus keyword-only ``quant_type_a``, ``quant_dtype_a``,
    ``quant_dtype_a2`` and ``deterministic``. Shared experts (FHMoE),
    ``stage2_scatter``, ``herd_topk`` and ``output`` are not supported; pass
    ``out`` to ``fused_moe_run_stage2`` instead of ``output``.

    ``config_shape=(token_num, topk)`` selects kernels and tuned config as if
    ``topk_ids`` had that shape, e.g. the config of the same routes as a
    repeated input with top-1.

    Raises ``RuntimeError`` if the selected pipeline has no separate stage2
    (1-stage, full or direct implementations); for some of them the MoE has
    already run by then.
    """
    captured = []

    def capture_stage2(*, ordinary_stage2, stage2_args, stage2_kwargs):
        captured.append((ordinary_stage2, stage2_args, stage2_kwargs))

    _fused_moe_impl(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType(activation).value,
        quant_type=QuantType(quant_type).value,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M or -1,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        splitk=splitk,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        quant_type_a=None if quant_type_a is None else QuantType(quant_type_a).value,
        quant_dtype_a=quant_dtype_a,
        quant_dtype_a2=quant_dtype_a2,
        deterministic=deterministic,
        _stage2_override=capture_stage2,
        _config_shape=config_shape,
    )
    if not captured:
        raise RuntimeError(
            "fused_moe_prepare_stage2: the selected MoE pipeline has no "
            "separate stage2"
        )
    ((stage2, args, kwargs),) = captured
    kwargs = dict(kwargs)
    return MoeStage2Inputs(
        a2=args[0],
        a2_scale=kwargs.pop("a2_scale"),
        _stage2=stage2,
        _args=tuple(args[1:]),
        _kwargs=kwargs,
    )


def fused_moe_run_stage2(
    inputs: MoeStage2Inputs,
    *,
    out: torch.Tensor | None = None,
    per_slot: bool = False,
    w2: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a2: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the stage2 of ``fused_moe`` on ``inputs`` from
    ``fused_moe_prepare_stage2``. Returns the ``[token_num, model_dim]``
    output (``out`` if given).

    Options, if ``supports_stage2_overrides`` is true for the config (else
    ``NotImplementedError``):

    * ``per_slot``: write each route's weighted output to its own row of
      ``out[token_num, topk, model_dim]`` instead of summing. Under EP, rows
      of masked experts are zero if ``out`` is allocated here, and untouched
      in a caller's ``out``.
    * ``w2`` / ``w2_scale``: use these down weights (same layout and experts,
      any ``model_dim``).
    * ``a2`` / ``a2_scale``: use this intermediate (same layout and rows, any
      width), e.g. ``inputs.a2`` gathered across ranks.

    Replacing operands must be contiguous, on the same device, of the same
    dtype, with matching K for ``w2`` and ``a2``, E8M0 scales with enough
    rows and groups, and a new scale if the shape changes; else
    ``ValueError``.

    ``inputs`` can be run several times (e.g. with different ``w2`` rows).
    Without ``out``, the summed output is the sorting's buffer, so it can
    only be returned once per ``inputs``. ``out`` must be contiguous and must
    not overlap ``a2``.
    """
    stage2 = inputs._stage2
    w1, w2_, sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk = inputs._args
    w2_ = w2_ if w2 is None else w2
    a2_ = inputs.a2 if a2 is None else a2
    kwargs = dict(inputs._kwargs)
    kwargs["a2_scale"] = inputs.a2_scale if a2_scale is None else a2_scale
    if per_slot or any(t is not None for t in (w2, w2_scale, a2, a2_scale)):
        if not _captured_stage2_supports_overrides(inputs):
            raise NotImplementedError(
                f"stage2 {stage2} does not support per_slot or replaced "
                "operands; see supports_stage2_overrides"
            )
        _check_stage2_operands(inputs, w2, w2_scale, a2, a2_scale)
        if w2_scale is not None:
            kwargs["w2_scale"] = w2_scale.view(dtypes.fp8_e8m0)
        # v2 takes the GEMM shape from its config, v1 from the operands.
        config = getattr(stage2, "keywords", {})
        if w2 is not None and "model_dim" in config:
            kwargs["model_dim"] = w2.shape[1]
        if a2 is not None and "inter_dim" in config:
            kwargs["inter_dim"] = _scaled_dim(
                config["inter_dim"], a2.shape[-1], inputs.a2.shape[-1]
            )
    token_num, dtype, device = moe_buf.shape[0], moe_buf.dtype, moe_buf.device
    model_dim = w2_.shape[1]
    if out is not None and bytes_overlap(out, a2_):
        raise ValueError("fused_moe_run_stage2: out overlaps a2")
    if per_slot:
        shape = (token_num, topk, model_dim)
        if out is None:
            out = torch.zeros(shape, dtype=dtype, device=device)
        _validate_output_buffer_metadata(out, shape, dtype, device)
        moe_out = moe_buf
        kwargs["per_slot_out"] = out
    elif out is None and w2 is None:
        if inputs._moe_out_written:
            raise RuntimeError(
                "fused_moe_run_stage2: the summed output of these inputs was "
                "already returned; pass out= to run stage2 again"
            )
        moe_out = out = moe_buf
    else:
        if out is None:
            out = torch.empty((token_num, model_dim), dtype=dtype, device=device)
        _validate_output_buffer_metadata(out, (token_num, model_dim), dtype, device)
        if not _stage2_runtime_uses_route_reduce(stage2, token_num, model_dim):
            # Accumulating stage2 kernels add into their output.
            out.zero_()
        moe_out = out
    stage2(
        a2_,
        w1,
        w2_,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        **kwargs,
    )
    if moe_out is moe_buf and not per_slot:
        inputs._moe_out_written = True
    return out


cfg_2stages = None
cfg_2stages_by_file = {}
cfg_2stages_by_limit_tag = {}

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
        (ActivationType.Swiglu,  QuantType.per_1x32,   dtypes.bf16,   dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Swiglu,  QuantType.per_1x32,   dtypes.bf16,    dtypes.bf16,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Gelu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,    dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   True)  : aiter.fmoe_g1u1_tkw1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
    }
}
# fmt: on


def _normalize_moe_dispatch_contract(
    quant_type, activation, swiglu_limit, quant_type_a=None
):
    quant_type = quant_remap.get(quant_type, quant_type)
    if quant_type_a is not None:
        quant_type_a = quant_remap.get(QuantType(quant_type_a), QuantType(quant_type_a))
    if activation == ActivationType.Silu:
        swiglu_limit = normalize_swiglu_limit(swiglu_limit)
    return quant_type, quant_type_a, swiglu_limit


def _mxmoe_opus_cfg_unsupported(
    p1: dict,
    kernel_name2: str,
    cfg: dict,
    *,
    gfx: str,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    dtype: torch.dtype,
    q_dtype_a: torch.dtype,
    q_dtype_w: torch.dtype,
    q_type: QuantType,
    use_g1u1: bool,
    doweight_stage1: bool,
    gate_mode: GateMode,
    hidden_pad: int,
    intermediate_pad: int,
    is_ep: bool,
    has_stage2_bias: bool,
    q_dtype_a2: torch.dtype | None,
    input_dtype: torch.dtype | None,
    has_stage2_scatter: bool,
    has_num_local_tokens: bool,
) -> str:
    """Static common contract for the explicit MXMOE -> OPUS selection."""
    if (p1["a_dtype"], p1["out_dtype"]) != ("fp8", "fp8"):
        return "GEMM1 requires A8W4 and FP8out"
    if dtype != dtypes.bf16 or q_dtype_a != dtypes.fp8:
        return "requires BF16 final output and explicit FP8 activation precision"
    if q_dtype_w != dtypes.fp4x2 or q_type != QuantType.per_1x32:
        return "requires per-1x32 MXFP4 weights"
    if not use_g1u1 or gate_mode != GateMode.INTERLEAVE or doweight_stage1:
        return "requires G1U1, INTERLEAVE W1 and doweight_stage1=False"
    if hidden_pad or intermediate_pad:
        return "hidden/intermediate padding is not supported"
    if has_stage2_scatter or has_num_local_tokens:
        return "stage2_scatter/num_local_tokens is not supported"
    if not aiter.is_mxfp4_moe_shape_supported(expert, model_dim, inter_dim, topk):
        return "generated MXFP4 auxiliary shape is not registered"
    if q_dtype_a2 not in (None, dtypes.fp8):
        return "requires FP8 intermediate precision"
    if input_dtype not in (None, dtypes.bf16, dtypes.fp8):
        return "requires raw BF16 or prequantized FP8 input"
    if p1["inline_quant"] and input_dtype == dtypes.fp8:
        return "inline GEMM1 reads BF16 and cannot accept FP8 input"
    supported, reason = _opus_a8w4.cfg_is_supported(
        kernel_name2,
        cfg=cfg,
        gfx=gfx,
        block_m=cfg.get("block_m", BLOCK_SIZE_M),
        is_ep=is_ep,
        has_stage2_bias=has_stage2_bias,
    )
    if not supported:
        return reason
    launch = _opus_a8w4.parse_stage2_config(cfg, cfg.get("block_m", BLOCK_SIZE_M))
    if p1["BM"] != launch.sort_block_m:
        return f"GEMM1 BM={p1['BM']} requires OPUS SBM={p1['BM']}"
    if inter_dim < 256 or inter_dim % 128:
        return "OPUS requires inter_dim >= 256 and divisible by 128"
    if model_dim % launch.instance.block_n:
        return f"model_dim must be divisible by OPUS block_n={launch.instance.block_n}"
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    try:
        _assert_supported(
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            BM=p1["BM"],
            BN=p1["BN"],
            BK=p1["BK"],
            use_nt=p1["use_nt"],
            inline_quant=p1["inline_quant"],
            prefetch_hidden=p1["prefetch_hidden"],
            a_dtype=p1["a_dtype"],
            out_dtype=p1["out_dtype"],
            act=p1["act"],
            num_waves=p1["num_waves"],
            k_wave=p1["k_wave"],
            native_scale_layout=False,
        )
    except NotImplementedError as exc:
        return str(exc)
    if p1["splitk"]:
        return "split-K GEMM1 is not supported"
    return ""


def _is_mxmoe_opus_metadata(metadata: MOEMetadata) -> bool:
    return (
        getattr(metadata.stage1, "func", metadata.stage1) is _mxfp4_a4w4_stage1_fw
        and getattr(metadata.stage2, "func", metadata.stage2)
        is _opus_a8w4.opus_a8w4_stage2_wrapper
    )


def _mxmoe_opus_call_unsupported(
    metadata: MOEMetadata,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,
    *,
    block_size_M: int | None,
) -> str:
    """Check tensor preparation and the actual sort before each public call."""
    if block_size_M is not None and int(block_size_M) != int(metadata.block_m):
        return f"block_size_M={block_size_M} differs from configured BM/SBM={metadata.block_m}"
    if not (getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False)):
        return "both weights must be marked is_shuffled=True"
    if w1.dtype != dtypes.fp4x2 or w2.dtype != dtypes.fp4x2:
        return "both weights must retain packed MXFP4 dtype"
    tensors = [hidden_states, w1, w2, w1_scale, w2_scale, topk_ids]
    if w1_scale is None or w2_scale is None:
        return "both weights require E8M0 scales"
    if any(
        scale.dtype not in (torch.uint8, dtypes.fp8_e8m0)
        for scale in (w1_scale, w2_scale)
    ):
        return "weight scales require one-byte E8M0 storage"
    experts, hidden, inter = w1.shape[0], w2.shape[1], w2.shape[-1] * 2
    if hidden_states.shape[1] != hidden or w1.shape[-1] * 2 != hidden:
        return "activation and packed W1 dimensions do not match model_dim"
    if w1_scale.numel() < experts * 2 * inter * _padded_scale_cols(hidden):
        return "W1 scales are smaller than the padded preshuffle stride"
    if (
        w2_scale.dim() != 2
        or w2_scale.shape[0] != experts * hidden
        or w2_scale.shape[1] != _padded_scale_cols(inter)
    ):
        return (
            "W2 scales require contiguous [expert * model_dim, padded group32] storage"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "weights, scales, activation and routing must be contiguous"
    if len({tensor.device for tensor in tensors}) != 1:
        return "cross-device tensors"
    if hidden_states.dtype == dtypes.fp8:
        if a1_scale is None:
            raise ValueError(
                "MXFP8 input requires original per-token E8M0 activation scales"
            )
        expected = (hidden_states.shape[0], hidden_states.shape[1] // 32)
        if tuple(a1_scale.shape) != expected or a1_scale.dtype not in (
            torch.uint8,
            dtypes.fp8_e8m0,
        ):
            raise ValueError(
                f"MXFP8 input requires per-token group32 E8M0 scales of shape {expected}"
            )
        if not a1_scale.is_contiguous() or a1_scale.device != hidden_states.device:
            raise ValueError(
                "MXFP8 activation scales must be contiguous and on the input device"
            )
    M, topk = topk_ids.shape
    generated_sort = not moev2_sort._aux_uses_opus(
        metadata.output_aux, metadata.block_m, M * topk, w1.shape[0]
    ) and moev2_sort._MOE_SORT_BACKEND not in ("opus", "ck")
    if generated_sort and not moev2_sort._mxfp4_aux_instance_supported(
        w1.shape[0],
        topk,
        w2.shape[1],
        int(metadata.block_m),
        not stage2_uses_route_reduce(metadata.stage2),
    ):
        raise _MissingMxfp4MoeAuxInstances(
            [
                (
                    w1.shape[0],
                    topk,
                    w2.shape[1],
                    not stage2_uses_route_reduce(metadata.stage2),
                )
            ]
        )
    return ""


# Widest row fused_dynamic_mx_quant_moe_sort quantizes (32 elements per thread,
# 256 threads; csrc/kernels/quant_kernels.cu).
_MX_QUANT_SORT_MAX_COLS = 8192


def _check_deterministic_supported(
    *,
    dtype,
    quant_type,
    q_dtype_w,
    q_dtype_a,
    q_dtype_a2,
    activation,
    gate_mode,
    gate_up_interleave,
    is_shuffled,
    is_g1u1,
    doweight_stage1,
    expert_mask,
    stage2_scatter,
    a1_scale,
    a2_scale,
    block_size_M,
    tokens,
    model_dim,
    inter_dim,
    hidden_pad,
    intermediate_pad,
    has_bias1,
    has_bias2,
    weight_bytes,
):
    """Raise NotImplementedError naming what fused_moe(deterministic=True) cannot run."""
    layout = {True: "gate/up-interleaved", False: "separated"}.get(
        gate_up_interleave, "untagged"
    )
    unsupported = [
        why
        for bad, why in (
            (
                not (quant_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp4x2),
                f"weights other than MXFP4 ({quant_type!s}, {q_dtype_w})",
            ),
            (
                # MXFP4 activations lose accuracy against the default.
                q_dtype_a != dtypes.fp8,
                f"activation dtype {q_dtype_a} (MXFP8 only)",
            ),
            (
                q_dtype_a2 is not None and q_dtype_a2 != q_dtype_a,
                (
                    f"stage-2 activation dtype {q_dtype_a2} (must match the stage-1 "
                    f"activation dtype {q_dtype_a})"
                ),
            ),
            (dtype != dtypes.bf16, f"output dtype {dtype} (bf16 only)"),
            (
                # SwiGLU and SiLU are where the default also quantizes the
                # activations (from 256 tokens); SiTUv2's default keeps bf16.
                activation not in (ActivationType.Swiglu, ActivationType.Silu),
                f"activation {activation!s} (SwiGLU or SiLU only)",
            ),
            (
                # The MXFP8-activation (a8w4) kernels take the gate/up-interleaved
                # preshuffle only.
                gate_mode != GateMode.INTERLEAVE
                or not _flydsl_mxfp4_layout_is_compatible(
                    dtypes.fp8, gate_mode, gate_up_interleave
                ),
                (
                    f"{layout} weights with gate_mode={gate_mode.value!r} "
                    "(gate/up-interleaved only)"
                ),
            ),
            (not is_shuffled, "w1 and w2 not both preshuffled"),
            (
                not (is_g1u1 and not doweight_stage1),
                "non-g1u1 weights or doweight_stage1=True",
            ),
            (expert_mask is not None, "expert-parallel masking"),
            (stage2_scatter is not None, "stage2_scatter"),
            (
                a1_scale is not None or a2_scale is not None,
                "prequantized activations",
            ),
            (
                block_size_M is not None
                and int(block_size_M) != _deterministic_block_m(tokens),
                (
                    f"block_size_M={block_size_M} "
                    f"(uses {_deterministic_block_m(tokens)})"
                ),
            ),
            (
                # Only multiples of 512 have been validated with the deterministic
                # kernels.
                model_dim % 512 != 0,
                f"hidden size {model_dim} (a multiple of 512 only)",
            ),
            (
                # Both stages quantize their activations to MXFP8 at every token
                # count; at small batches that is the fused quantize-and-sort
                # kernel, which takes at most 8,192 columns.
                max(model_dim, inter_dim) > _MX_QUANT_SORT_MAX_COLS,
                (
                    f"hidden size {model_dim} / intermediate size {inter_dim} "
                    f"(at most {_MX_QUANT_SORT_MAX_COLS})"
                ),
            ),
            (
                # Without a stage-2 bias the padded output columns are not stored
                # before the top-k reduction reads them.
                hidden_pad > 0 and not has_bias2,
                "hidden padding without a stage-2 bias",
            ),
            (
                # Without a stage-1 bias stage 1 skips the padded intermediate
                # columns, while stage 2 with a bias reads them.
                intermediate_pad > 0 and has_bias2 and not has_bias1,
                "intermediate padding with a stage-2 bias but no stage-1 bias",
            ),
            (
                # The kernels address each weight with one 32-bit buffer
                # descriptor, so experts past 4 GiB would read other experts.
                weight_bytes > FHMOE_BUFFER_MAX_BYTES,
                f"weights of {weight_bytes} bytes (at most 4 GiB per projection)",
            ),
            (get_gfx_runtime() != "gfx950", f"gfx={get_gfx_runtime()!r}"),
        )
        if bad
    ]
    if unsupported:
        raise NotImplementedError(
            "fused_moe(deterministic=True) does not support: " + "; ".join(unsupported)
        )


def _deterministic_block_m(tokens: int) -> int:
    # At tile_m 32 the a8w4 (MXFP8-activation) stage 1 with tile_n 128 gives a
    # different result from run to run on gfx950, so the tile height starts at 64.
    return 64 if tokens <= 2048 else 128


def _deterministic_metadata(
    tokens, activation, dtype, quant_type, hidden_pad, intermediate_pad
):
    """FlyDSL a8w4 (MXFP8 x MXFP4) kernels with one K structure for every token count.

    The tuned MXFP4 configurations change the stage-1 K waves, the fused fp4
    output and the stage-2 kernel family (v1 / v2 layout, tile_k 128 / 256) with
    the token count, and each changes how a row's sums are split. Deterministic
    mode always runs stage 1 with a single K wave (k_wave 1; the _w2 in the
    kernel name is waves_per_eu 2) and tile_k 256, and stage 2 with the
    non-atomic reduce epilogue instead of atomic adds of the top-k partials,
    without split-K, skipping dropped routes in the reduction. Only the tile
    height grows with the token count, which does not change a row's reduction.
    """
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    block_m = _deterministic_block_m(tokens)
    # The a8w4 stage-1 kernels read gate/up-interleaved weights (_gui). The tile
    # width only splits the output columns, so a row's K reduction is the same at
    # both widths; the wider tile is faster from tile_m 128.
    tile_n = 128 if block_m <= 64 else 256
    kn1 = f"flydsl_moe1_afp8_wfp4_bf16_t{block_m}x{tile_n}x256_w2_gui"
    kn2 = f"flydsl_moe2_afp8_wfp4_bf16_t{block_m}x128x256_reduce"
    for kn in (kn1, kn2):
        params = get_flydsl_kernel_params(kn)
        if params is None or params.get("k_batch", 1) != 1:
            raise NotImplementedError(f"{kn} is not registered without split-K")
    enable_bias = _needs_swiglu_bias_support(dtype, quant_type)
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
            # The sort leaves out dropped routes (topk_ids -1), so their rows of
            # the per-slot output are never written; the reduction skips them.
            skip_dropped_routes=True,
        ),
        block_m,
        -1,  # no split-K
        False,
        has_bias=enable_bias,
        stage2_has_bias=enable_bias,
    )


def _swiglu_limit_is_active(swiglu_limit) -> bool:
    """True only when the caller asked for a finite positive clamp bound.

    ``None``, ``0.0``, and ``+inf`` are dispatch sentinels: vLLM's ROCm adapter
    sends ``0.0`` when the config limit is unset, and those calls must stay on
    CK. This is not kernel clamp encoding. Once a call is on a16wmix,
    ``runtime_swiglu_limit`` maps a SwiGLU ``None``/``0.0`` to 7.0 (only ``+inf``
    turns that clamp off). Rerouting those sentinels would start clamping
    existing SwiGLU callers at 7, so they must not count as active here.
    """
    if swiglu_limit is None:
        return False
    try:
        value = float(swiglu_limit)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


# a16wmix gemm1 clamps SwiGLU natively and SiLU via clamp_gate_up then
# silu_mul_batch (the shared helper stays unclamped for mixed_moe).
_A16WMIX_BF16_ACTIVATIONS = (ActivationType.Silu, ActivationType.Swiglu)
# The port's only arch split is gfx942 vs everything else (gfx950 16x16x32
# MFMA, buffer_atomic_pk_add_bf16). gfx90a compiles with
# "s_waitcnt is not supported on target arch 'gfx90a'". Same CDNA pair as the
# a16w4 FlyDSL guard below.
_A16WMIX_BF16_ARCHS = ("gfx942", "gfx950")


def _can_reroute_bf16_to_a16wmix(
    *,
    gfx,
    model_dim,
    inter_dim,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    activation,
    is_shuffled,
    use_g1u1,
    doweight_stage1,
    swiglu_limit,
    has_stage1_bias,
    hidden_pad,
    intermediate_pad,
):
    """Whether untuned/CK BF16 QuantType.No can take the a16wmix a16w16 port.

    Not a default-path swap: ``None`` / ``0.0`` / ``+inf`` stay on CK, so
    existing callers stay bit-identical. The finite limit is the gate; there is
    no ``AITER_*`` override. Weights must be preshuffled; a16wmix consumes
    shuffle_weight (16,16). ``gfx`` is the live device from
    ``get_gfx_runtime()``.
    """
    # gfx90a and RDNA must not fall through to the gfx950 compile path.
    if gfx not in _A16WMIX_BF16_ARCHS:
        return False
    if not (
        dtype == dtypes.bf16
        and q_type == QuantType.No
        and q_dtype_a == dtypes.bf16
        and q_dtype_w == dtypes.bf16
        and activation in _A16WMIX_BF16_ACTIVATIONS
        and use_g1u1
        and is_shuffled
        and not doweight_stage1
        and not has_stage1_bias
        and not hidden_pad
        and not intermediate_pad
        and model_dim % 128 == 0
        and inter_dim % 128 == 0
    ):
        return False
    return _swiglu_limit_is_active(swiglu_limit)


# NOTE: aiter.moev2 (under development, not graduated) keeps a copy of this logic for
# the kernel families its paths cover (listed in aiter/moev2/README.md): the tuned
# lookup, row checks and log lines in aiter/moev2/dispatch/resolver.py, each default
# branch in aiter/moev2/dispatch/ranking.py. Mirror changes there or coordinate with the
# moev2 owners (@qli88), then run the tests/moe_dispatch checks in aiter/moev2/README.md.
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
    _exclude_mxmoe=False,
    _has_input_activation_scales=False,
    has_global_scale=False,
):
    gate_mode = GateMode(gate_mode)
    if q_dtype_w == dtypes.fp4x2 and has_global_scale:
        from aiter.ops.moe_nvfp4 import _fused_moe_nvfp4

        if has_stage2_scatter:
            raise ValueError("NVFP4 fused_moe: stage2_scatter is not supported")
        return MOEMetadata(None, None, 32, 0, full_impl=_fused_moe_nvfp4)
    cktile_mxfp4_unsafe = q_dtype_w == dtypes.fp4x2 and inter_dim % 256 != 0
    if q_type in RUNTIME_SILU_LIMIT_QUANT_TYPES and activation == ActivationType.Silu:
        swiglu_limit = normalize_swiglu_limit(swiglu_limit)
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

    # NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
    # Please do not modify this code without coordinating with the moev2 owners
    # (@qli88). See aiter/moev2/README.md.
    # Tuned tables are read by aiter.moev2.dispatch.tuned, shared with aiter.moev2.
    def get_cfg_2stages(tune_file, tag=None):
        return moev2_tuned.read_table(
            tune_file, _INDEX_COLS, config_file is not None, tag
        )

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
    limit_tag = (
        swiglu_limit_tag(swiglu_limit)
        if config_file is None
        and q_type in RUNTIME_SILU_LIMIT_QUANT_TYPES
        and activation == ActivationType.Silu
        else ""
    )
    if limit_tag:
        limit_tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_SWIGLU_LIMIT_FILE
        # Rows tuned by gemm_moe_tune.py --swiglu_limit, tagged with the limit.
        limit_cfgs = cfg_2stages_by_limit_tag.get((limit_tune_file, limit_tag))
        if limit_cfgs is None:
            limit_cfgs = get_cfg_2stages(limit_tune_file, tag=limit_tag)
            cfg_2stages_by_limit_tag[(limit_tune_file, limit_tag)] = limit_cfgs
        limit_cfg = _lookup_cfg(limit_cfgs)
        if limit_cfg is not None:
            if (
                limit_cfg.get("run_1stage", False)
                or int(limit_cfg.get("ksplit", 0) or 0) > 1
                or "_silu_" not in str(limit_cfg.get("kernelName1", ""))
                or "Splitk" in str(limit_cfg.get("kernelName1", ""))
            ):
                raise ValueError(
                    "Incompatible runtime SiLU limit tuning row; retune with "
                    "non-split CK SiLU instances"
                )
            cfg = limit_cfg
        else:
            logger.info(
                f"[fused_moe] no row for {keys} with {limit_tag}; using default config"
            )
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
        # Invalid MXMOE names must fail before a backend/layout check can
        # silently discard the row and substitute a heuristic kernel.
        parsed_g1 = _parse_mxfp4_g1_kname(kn1) if _is_mxfp4_kname(kn1) else None
        if (
            parsed_g1 is not None
            and _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2)
            and q_dtype_a2 not in (None, dtypes.fp8)
        ):
            # The existing A8W4 heuristic fallback consumes FP8 intermediate.
            # Discarding this row cannot satisfy a different explicit request.
            raise NotImplementedError(
                f"No fallback backend for MXMOE -> OPUS supports requested intermediate precision {q_dtype_a2}; requires FP8"
            )
        if (
            parsed_g1 is not None
            and _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2)
            and input_dtype == dtypes.fp8
            and not _has_input_activation_scales
        ):
            raise ValueError(
                "MXFP8 input requires original per-token E8M0 activation scales"
            )
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

    if (
        cfg is not None
        and cfg.get("run_1stage", False)
        and kn1.startswith("_ZN5aiter")
        and activation == ActivationType.Situv2
        and q_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
    ):
        if is_shuffled:
            if not opus_weights_shuffled:
                raise ValueError(
                    "SiTUv2 MXFP4 requires both w1 and w2 to use the same "
                    "preshuffled weight layout"
                )
            if gate_mode != GateMode.SEPARATED or gate_up_interleave is True:
                raise ValueError("SiTUv2 A16W4 requires separated preshuffled weights")
            cfg = None
            logger.warning(
                "[fused_moe] discarding 1-stage SiTUv2 ASM config for preshuffled "
                "MXFP4 weights; using the two-stage layout-compatible path"
            )
        elif (
            situ_beta != DEFAULT_SITUV2_BETA
            or situ_linear_beta != DEFAULT_SITUV2_LINEAR_BETA
        ):
            raise ValueError(
                "1-stage SiTUv2 ASM has fixed beta=4 and linear_beta=25; "
                "use separated preshuffled weights for runtime betas"
            )

    if (
        cfg is not None
        and cfg.get("flat", AsmMoeMode.SORTED) == AsmMoeMode.POST_REDUCE
        and (
            gate_mode != GateMode.SEPARATED
            or gate_up_interleave is True
            or not opus_weights_shuffled
        )
    ):
        if config_file is not None:
            raise ValueError("Post-reduction requires separated, shuffled weights")
        cfg = None
        logger.warning(
            "[fused_moe] discarding post-reduction config for incompatible weight layout; "
            "using default heuristics"
        )

    if cfg is not None and _is_mxfp4_kname(kn1):
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
        elif _exclude_mxmoe:
            reject_reason = "excluded for this invocation"
        elif _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2):
            reject_reason = (
                _mxmoe_opus_cfg_unsupported(
                    parsed_g1,
                    kn2,
                    cfg,
                    gfx=gfx,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                    expert=expert,
                    topk=topk,
                    dtype=dtype,
                    q_dtype_a=q_dtype_a,
                    q_dtype_w=q_dtype_w,
                    q_type=q_type,
                    use_g1u1=use_g1u1,
                    doweight_stage1=doweight_stage1,
                    gate_mode=gate_mode,
                    hidden_pad=hidden_pad,
                    intermediate_pad=intermediate_pad,
                    is_ep=is_ep,
                    has_stage2_bias=has_stage2_bias,
                    q_dtype_a2=q_dtype_a2,
                    input_dtype=input_dtype,
                    has_stage2_scatter=has_stage2_scatter,
                    has_num_local_tokens=has_num_local_tokens,
                )
                or None
            )
        elif parsed_g1["inline_quant"] and input_dtype not in (None, dtypes.bf16):
            reject_reason = (
                f"inline GEMM1 reads BF16 and cannot accept {input_dtype} input"
            )
        elif not (
            int(cfg.get("block_m", BLOCK_SIZE_M))
            == parsed_g1["BM"]
            == _mxmoe_gemm2_sort_block_m(kn2, parsed_g1["BM"])
        ):
            reject_reason = (
                f"sort block_m={cfg.get('block_m', BLOCK_SIZE_M)} and GEMM2 {kn2!r} "
                f"must both match GEMM1 BM={parsed_g1['BM']}"
            )
        # BM16 makes the masked Opus aux sort use dispatch_policy=1, which hangs
        # at >= 819 experts (#8014).
        elif is_ep and parsed_g1["BM"] == 16:
            reject_reason = "BM16 MXMOE is not supported under EP yet"
        # MXMOE under EP is validated only for SiTUv2 with an atomic or reduce
        # layout GEMM2.
        elif is_ep and activation != ActivationType.Situv2:
            reject_reason = (
                f"MXMOE under EP is only validated for SiTUv2, got {activation}"
            )
        # Under EP `expert` is the local count, and the masked Opus sort plus the
        # prequant GEMM1 replace every NE-keyed aux kernel.
        elif is_ep and parse_flydsl_v2_gemm2_kernel(kn2) is None:
            reject_reason = f"EP requires a flydsl_moe2_layout_ GEMM2, got {kn2!r}"
        elif is_ep and parse_flydsl_v2_gemm2_kernel(kn2)["epilog"] == "scatter":
            reject_reason = "the scatter GEMM2 epilog is not validated under EP"
        elif not is_ep and not aiter.is_mxfp4_moe_shape_supported(
            expert, model_dim, inter_dim, topk
        ):
            reject_reason = (
                "generated MXFP4 auxiliary kernels do not cover "
                f"(expert={expert}, model_dim={model_dim}, "
                f"inter_dim={inter_dim}, topk={topk})"
            )
        elif has_stage2_bias and parse_flydsl_v2_gemm2_kernel(kn2) is None:
            reject_reason = (
                f"stage2 bias requires a flydsl_moe2_layout_ kernel, got {kn2!r}"
            )
        if reject_reason is not None:
            cfg = None
            logger.warning(
                f"[fused_moe] discarding MXMOE config ({kn1!r}, {kn2!r}): {reject_reason}; "
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

    # Runtime SiLU clamping is supported only by the non-split CK stage-1 epilogue
    # and, for gfx942 per-token FP8, the FlyDSL backend, so it must reject ASM
    # 1-stage and other FlyDSL whole-graph rows before they resolve.
    clamp_silu_limit = (
        bool(swiglu_limit)
        and q_type in RUNTIME_SILU_LIMIT_QUANT_TYPES
        and activation == ActivationType.Silu
    )
    flydsl_gfx942_row = (
        gfx == "gfx942"
        and q_type == QuantType.per_Token
        and q_dtype_w == dtypes.fp8
        and cfg is not None
        and str(cfg.get("kernelName1", "")).startswith("impl__flydsl_gfx942__")
    )
    if clamp_silu_limit and cfg is not None and not flydsl_gfx942_row:
        cfg_ksplit = int(cfg.get("ksplit", 0) or 0)
        if (
            cfg.get("run_1stage", False)
            or "ck2stages" not in str(cfg.get("kernelName1", ""))
            or cfg_ksplit > 1
            or "_silu_" not in str(cfg.get("kernelName1", ""))
        ):
            logger.warning(
                f"[fused_moe] discarding tuned config {cfg.get('kernelName1')!r} "
                f"for {keys}: it cannot apply swiglu_limit={swiglu_limit!r}; "
                "using default heuristics"
            )
            cfg = None

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
    # Only the clamped SiLU takes num_local_tokens on a FlyDSL row; the other
    # rows keep the default heuristics for it.
    clamped_silu = (
        gfx == "gfx942"
        and _swiglu_limit_is_active(swiglu_limit)
        and activation == ActivationType.Silu
    )
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
        elif has_num_local_tokens and not clamped_silu:
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
    # Calls CK's runtime-limit stage-1 can clamp stay on it: the whole-graph
    # default clamps the BF16 activations it reads and reports stage1=None.
    # Same condition as the NotImplementedError gate below (FP8 is already
    # required here), which defines what that stage-1 accepts.
    ck_can_clamp_silu = inter_dim % 128 == 0 and q_dtype_a == dtypes.fp8
    if (
        full_impl is None
        and (cfg is None or bypass_tuned_config or cfg.get("run_1stage", False))
        and clamped_silu
        and not ck_can_clamp_silu
        and q_type == QuantType.per_Token
        and q_dtype_w == dtypes.fp8
        and use_g1u1
        and weights_shuffled
        and gate_mode is GateMode.SEPARATED
        and input_dtype in (None, dtypes.bf16)
        and not (
            doweight_stage1
            or has_stage1_bias
            or has_stage2_bias
            or has_activation_scales
            or has_stage2_scatter
            or hidden_pad
            or intermediate_pad
        )
    ):
        from aiter.ops.flydsl.fused_moe_gfx942 import default_config

        flydsl_config = default_config(token)
        kernel_name1 = make_fused_moe_impl_kernel_name("flydsl_gfx942", flydsl_config)
        full_impl = resolve_fused_moe_impl(kernel_name1)
        cfg = {"block_m": int(flydsl_config.split("_", 1)[0]), "ksplit": 0}
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
    # No FlyDSL backend took the clamp, so CK stage 1 must.
    if clamp_silu_limit and q_type != QuantType.per_1x128:
        if q_dtype_w != dtypes.fp8 or q_dtype_a != dtypes.fp8:
            raise NotImplementedError(
                f"fused_moe cannot apply swiglu_limit={swiglu_limit!r} for {q_type} "
                f"SiLU with weight dtype {q_dtype_w} / activation dtype {q_dtype_a}; "
                "the runtime-limit CK instances are FP8 only."
            )
        if inter_dim % 128 != 0:
            raise NotImplementedError(
                f"fused_moe cannot apply swiglu_limit={swiglu_limit!r} for {q_type} "
                f"SiLU at inter_dim={inter_dim}; the runtime-limit CK stage1 needs a "
                "multiple of 128."
            )
    if config_file is not None and (cfg is None or bypass_tuned_config):
        raise NotImplementedError(
            "The dedicated FHMoE path requires an exact tuned config row for "
            f"{keys} in {tune_file}"
        )

    # Asm 1-stage kernels exist for Silu/Gelu and for gfx950 MXFP4 SiTUv2 and
    # SwiGLU OAI FLAT.
    if (
        cfg is not None
        and cfg.get("run_1stage", False)
        and activation
        not in (
            ActivationType.Silu,
            ActivationType.Gelu,
            ActivationType.Situv2,
            ActivationType.Swiglu,
        )
    ):
        cfg = None
        logger.warning(
            f"[fused_moe] discarding 1-stage tuned config for unsupported "
            f"activation {activation}; using default heuristics"
        )

    # The SwiGLU OAI FLAT asm kernels hard-code alpha=1.702 / limit=7.0, take
    # GGUU (separated) gate/up weights and have no per-expert bias input. The
    # 1-stage launch does not forward bias or swiglu_limit, so a row that does
    # not match the call would silently compute a different function.
    # They also read raw top-k routing (flat=1). Only a named flat=1 row makes
    # fused_moe skip the host moe_sorting; an unnamed row would let fmoe_g1u1
    # pick a FLAT kernel heuristically and hand it host-sorted ids.
    if (
        cfg is not None
        and cfg.get("run_1stage", False)
        and activation == ActivationType.Swiglu
    ):
        swiglu_kernel1 = cfg.get("kernelName1", "")
        if not isinstance(swiglu_kernel1, str):  # pandas reads a blank cell as NaN
            swiglu_kernel1 = ""
        try:
            swiglu_flat = int(cfg.get("flat", AsmMoeMode.SORTED))
        except (TypeError, ValueError):  # blank (NaN) flat cell
            swiglu_flat = AsmMoeMode.SORTED
        unsupported = None
        if not swiglu_kernel1.strip():
            unsupported = "empty kernelName1 (FLAT kernels need an explicit name)"
        elif swiglu_flat != AsmMoeMode.NO_SORT:
            unsupported = f"flat={cfg.get('flat')} (FLAT kernels need flat=1)"
        elif has_stage1_bias or has_stage2_bias:
            unsupported = "per-expert bias"
        elif swiglu_limit is not None and float(swiglu_limit) != 7.0:
            unsupported = f"swiglu_limit={swiglu_limit} (kernel uses 7.0)"
        elif gate_mode != GateMode.SEPARATED:
            unsupported = f"gate mode {gate_mode.value!r}"
        elif gate_up_interleave:
            unsupported = "gate/up-interleaved (GUGU) weights"
        if unsupported is not None:
            cfg = None
            logger.warning(
                f"[fused_moe] discarding SwiGLU OAI 1-stage asm config for {keys}: "
                f"unsupported {unsupported}; using default heuristics"
            )

    use_non_temporal_load = False
    if cfg is None or bypass_tuned_config:
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        run_1stage_xbf16 = False
        # No tuned config => default host moe_sort (flat=0). A tuned FLAT row
        # sets flat=1 (one token per TG) or flat=2 (EMSORT).
        cfg_flat = AsmMoeMode.SORTED
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
                    not clamp_silu_limit
                    and token > 32
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
        # A tuned CK row does not use the heuristic above. is_shuffled is the
        # OR, so one tagged weight used to skip the preshuffle warning too.
        tuned_names = (
            str(cfg.get("kernelName1", "") or ""),
            str(cfg.get("kernelName2", "") or ""),
        )
        if not cfg.get("run_1stage", False) and any(
            "moe_ck2stages" in name for name in tuned_names
        ):
            _require_shuffled_gfx950_fp8(gfx, q_dtype_w, q_type, weights_shuffled)
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
        if run_1stage:
            retargeted = _retarget_asm_activation(kernelName1, activation)
            if retargeted != kernelName1:
                logger.info(
                    f"[fused_moe] 1-stage kernel {kernelName1!r} is for a "
                    f"different activation; using {retargeted!r} for {activation}"
                )
                kernelName1 = retargeted
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
            cfg_flat = AsmMoeMode(int(cfg["flat"])) if run_1stage else AsmMoeMode.SORTED
        else:
            cfg_flat = AsmMoeMode.SORTED

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

    if clamp_silu_limit:
        # Deliberately override tuned, heuristic and AITER_KSPLIT selections.
        ksplit = 0
        run_1stage = False

    reroute_a16wmix_bf16 = _can_reroute_bf16_to_a16wmix(
        gfx=gfx,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        q_type=q_type,
        activation=activation,
        is_shuffled=bool(weights_shuffled),
        use_g1u1=use_g1u1,
        doweight_stage1=doweight_stage1,
        swiglu_limit=swiglu_limit,
        has_stage1_bias=has_stage1_bias,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    ) and not (isinstance(kernelName1, str) and kernelName1.startswith("flydsl_"))

    # The a16wmix reroute below logs its own kernel names; this line would name
    # the CK/1-stage choice it is about to discard.
    if not reroute_a16wmix_bf16:
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
            block_m=block_m,
            opus_launch=opus_stage2_launch,
            route_bucket_metadata=route_bucket_metadata,
        )

    if reroute_a16wmix_bf16:
        # Untuned a16w16 fallback (and override of a CK/asm tuned row) when a
        # finite swiglu_limit is configured. Tiles belong in the CSV; this is
        # the shape-safe default until a tuner row exists. block_m MUST equal
        # tile_m -- it sizes both moe_sorting and the gemm tile. A caller
        # block_size_M that disagrees is rejected in _fused_moe_impl, because
        # the only other check (FHMoE) does not cover this path and stage1
        # would otherwise run on the mismatched layout.
        from aiter.ops.flydsl.moe_kernels import flydsl_kernel_name

        _out_str = "bf16"
        _tile_m = 16 if token < 2048 else 32 if token < 16384 else 64
        _tile_n = _tile_k = 128
        kn1 = flydsl_kernel_name(1, "bf16", "bf16", _out_str, _tile_m, _tile_n, _tile_k)
        kn2 = flydsl_kernel_name(
            2, "bf16", "bf16", _out_str, _tile_m, _tile_n, _tile_k, "atomic"
        )
        logger.warning(
            "[fused_moe] no FlyDSL a16w16 config for %s, "
            "using untuned fallback (kn1=%s, kn2=%s)",
            keys,
            kn1,
            kn2,
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
    # wrong result. Stage-1 reduces over model_dim, which must be a multiple
    # of its 256-wide K tile; cktile_moe_stage1 rejects any other model_dim.
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
    # The fallback's layout GEMM2 writes bf16 only, and it is not validated under EP.
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
            # Plain fp8 CK instances read the shuffle_weight() layout.
            _require_shuffled_gfx950_fp8(gfx, q_dtype_w, q_type, weights_shuffled)
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


# NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
# Please do not modify this code without coordinating with the moev2 owners
# (@qli88). See aiter/moev2/README.md.
# aiter.moev2 reads the same default tuned table (share_default_table) and caches the
# same decisions, so get_2stage_cfgs.cache_clear clears both.
def _default_cfg_2stages():
    global cfg_2stages
    if cfg_2stages is None:
        cfg_2stages = moev2_tuned.read_table(
            AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
            moev2_tuned.index_columns(False),
            False,
        )
    return cfg_2stages


moev2_tuned.share_default_table(_default_cfg_2stages)
_get_2stage_cfgs_cache_clear = get_2stage_cfgs.cache_clear


def _clear_dispatch_caches():
    # aiter.moev2 caches the same decisions for the calls it runs.
    _get_2stage_cfgs_cache_clear()
    moev2_tuned.clear_cache()


get_2stage_cfgs.cache_clear = _clear_dispatch_caches


# NOTE: aiter.moev2 (under development, not graduated) keeps a copy of this logic in
# aiter/moev2/execute/executor.py (run_two_stage). Mirror changes there or coordinate
# with the moev2 owners (@qli88), then run the tests/moe_dispatch checks in
# aiter/moev2/README.md.
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
    deterministic: bool = False,
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
        _has_input_activation_scales=a1_scale is not None,
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
        and _parse_mxfp4_g1_kname(metadata.stage1.keywords["kernelName1"])["a_dtype"]
        == "fp4"
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
                # As for MXFP4 below: the row is shared by the token's routes.
                sorted_weights=None if deterministic else sorted_weights,
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
                # A token's activation row is shared by its top-k routes; with
                # routing weights, zero-weight routes write zeros into that shared
                # row, so the result would depend on which route writes last.
                sorted_weights=None if deterministic else sorted_weights,
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
        # Stage1 does not write slots with topk_ids == -1 or masked out by
        # expert_mask, but dynamic per-tensor a2 quant reads the whole buffer,
        # so stale memory in those slots can set the scale.
        _a2_alloc = (
            torch.zeros
            if quant_type == QuantType.per_Tensor and a2_scale is None
            else torch.empty
        )
        a2 = _a2_alloc(
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
    active_silu_limit = (
        normalize_swiglu_limit(swiglu_limit)
        if swiglu_limit and activation.value == _SILU_VALUE
        else None
    )
    if stage1_func in (_flydsl_stage1_wrapper, _opus_a8w4_stage1_wrapper):
        # Hand these two the caller's limit unchanged. They clamp silu whenever a
        # finite limit is configured (runtime_swiglu_limit in moe_kernels.py), and
        # the torch reference does the same, so normalizing non-Swiglu to None
        # would silently drop the clamp on paths outside this change.
        extra_stage1_args["swiglu_limit"] = swiglu_limit
    elif stage1_func is _mxfp4_a4w4_stage1_fw:
        extra_stage1_args["swiglu_limit"] = normalized_swiglu_limit
    elif active_silu_limit is not None:
        if stage1_func in (ck_moe_stage1, asm_stage1):
            extra_stage1_args["swiglu_limit"] = active_silu_limit
        elif stage1_func is cktile_moe_stage1:
            raise NotImplementedError(
                f"CK-Tile MoE stage1 cannot apply swiglu_limit={swiglu_limit!r}; "
                "the selected kernel has no clamp."
            )
    if stage1_func is _flydsl_stage1_wrapper and metadata.skip_inter_quant:
        extra_stage1_args["v2_output_layout"] = True
    if stage1_func is _flydsl_stage1_wrapper:
        extra_stage1_args["situ_beta"] = config_situ_beta
        extra_stage1_args["situ_linear_beta"] = config_situ_linear_beta
    elif stage1_func is _mxfp4_a4w4_stage1_fw:
        # MXMOE's activation comes from its name and betas enter its compile key.
        extra_stage1_args["situ_beta"] = config_situ_beta
        extra_stage1_args["situ_linear_beta"] = config_situ_linear_beta
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
            _mxfp4_a4w4_stage2_fw,
        )
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
    elif (
        quant_type == QuantType.No
        and stage1_func is _flydsl_stage1_wrapper
        and w1.dtype == dtypes.bf16
    ):
        # a16w16: stage1 returns bf16 [sorted_size, inter_dim]; no inter-stage quant
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
