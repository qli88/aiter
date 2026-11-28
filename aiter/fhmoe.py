# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused heterogeneous MoE (FHMoE) MXFP4/FP8 and MXFP8/FP8 dispatch.

The notation names routed/shared storage respectively. Shared ``FP8`` means
E4M3 payloads with per-32 E8M0 block scales, stored separately from routed
weights.

FHMoE requires gfx950, SiLU, per-1x32 quantization, contiguous preshuffled
weights/scales, all four shared tensors, and a dummy final routed row identified
by ``shared_expert_id == E - 1``. Expert bias and expert-mask dispatch are not
supported.

MXFP8/FP8 rows in ``tuned_fhmoe.csv`` are selected from request geometry,
dtypes, padding, and gate layout; model identity is not inferred.
``clamp_shared`` is an explicit caller-provided semantic. The existing DSV4
I384 MXFP4/FP8 compatibility path is retained unchanged. Unknown MXFP8
geometries may use generic fallback, while partial matches to dedicated
geometries fail closed on contract mismatches.
"""

import functools
import os
from dataclasses import replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fhmoe_contract import (
    FHMOE_BUFFER_MAX_BYTES,
)
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.jit.utils.torch_guard import torch_compile_guard

# NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
# Please do not modify this code without coordinating with the moev2 owners
# (@qli88). See aiter/moev2/README.md.
# Besides the moev2 hooks and paths, the names below moved from here to aiter.moev2
# and are re-exported under their old names; the legacy code below runs most of them.
# Changes must keep legacy behavior (python tests/moe_dispatch/snapshot.py --check).
from aiter.moev2 import hooks as moev2_hooks
from aiter.moev2 import paths as moev2_paths
from aiter.moev2.segments.flydsl_fhmoe import (  # noqa: F401
    _config_has_mxfp8_fhmoe_contract,
    _config_has_mxfp8_fhmoe_geometry,
    _dtype_name,
    _flydsl_fhmoe_stage1_wrapper,
    _flydsl_fhmoe_stage2_wrapper,
    _is_dsv4_i384_fhmoe_contract,
    _is_dsv4_i384_metadata,
    _is_fhmoe_metadata_compatible,
    _valid_fhmoe_kernel_geometry,
    _validate_fhmoe_contract,
    _validate_fhmoe_metadata_consistency,
)
from aiter.ops.flydsl.moe_common import GateMode
from aiter.presorted_moe import drop_presort_on_error


def _use_fhmoe_wrappers(metadata):
    from aiter.fused_moe import _flydsl_stage1_wrapper, _flydsl_stage2_wrapper

    stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
    stage2_func = getattr(metadata.stage2, "func", metadata.stage2)
    if (
        metadata.run_1stage
        or stage1_func is not _flydsl_stage1_wrapper
        or stage2_func is not _flydsl_stage2_wrapper
    ):
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require the "
            "two-stage FlyDSL path"
        )
    _validate_fhmoe_metadata_consistency(metadata)
    stage1 = functools.partial(
        _flydsl_fhmoe_stage1_wrapper,
        *metadata.stage1.args,
        **metadata.stage1.keywords,
    )
    stage2 = functools.partial(
        _flydsl_fhmoe_stage2_wrapper,
        *metadata.stage2.args,
        **metadata.stage2.keywords,
    )
    return replace(metadata, stage1=stage1, stage2=stage2)


def _fhmoe_config_file() -> str:
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE


def _dsv4_i384_fhmoe_config_file() -> str:
    """Compatibility wrapper for existing DSV4 capability tests."""
    return _fhmoe_config_file()


def _default_fhmoe_bucket(tokens: int, get_padded_m) -> int:
    return get_padded_m(tokens)


def _supports_fhmoe_config(
    max_tokens: int,
    config_file: str,
    *,
    shape: tuple[int, int, int, int],
    q_dtype_w: torch.dtype,
    bucket_fn,
    validate_fn,
) -> bool:
    try:
        from aiter.fused_moe import get_2stage_cfgs, get_padded_M

        required_tokens = {
            bucket_fn(1 << exponent, get_padded_M)
            for exponent in range(max_tokens.bit_length())
        }
        required_tokens.add(bucket_fn(max_tokens, get_padded_M))
        model_dim, inter_dim, experts, topk = shape
        for token in required_tokens:
            metadata = get_2stage_cfgs(
                token,
                model_dim,
                inter_dim,
                experts,
                topk,
                torch.bfloat16,
                dtypes.fp8,
                q_dtype_w,
                QuantType.per_1x32,
                True,
                ActivationType.Silu,
                False,
                0,
                0,
                True,
                GateMode.INTERLEAVE,
                config_file=config_file,
            )
            metadata = _use_fhmoe_wrappers(metadata)
            if not validate_fn(metadata):
                return False
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


@functools.cache
def _supports_dsv4_i384_fhmoe_config(max_tokens: int, config_file: str) -> bool:
    return _supports_fhmoe_config(
        max_tokens,
        config_file,
        shape=(7168, 384, 385, 7),
        q_dtype_w=dtypes.fp4x2,
        bucket_fn=_default_fhmoe_bucket,
        validate_fn=_is_dsv4_i384_metadata,
    )


def supports_dsv4_i384_fhmoe(max_tokens: int) -> bool:
    """Check complete tuned DSV4 I384 CSV coverage through ``max_tokens``."""
    if type(max_tokens) is not int or max_tokens <= 0:
        return False

    try:
        if int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")) != 0:
            return False
        config_file = _dsv4_i384_fhmoe_config_file()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return _supports_dsv4_i384_fhmoe_config(max_tokens, config_file)


@functools.cache
def _supports_mxfp8_fhmoe_config(
    max_tokens: int,
    config_file: str,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
    force_reduce: bool,
    stage2_fp8: bool,
) -> bool:
    try:
        from aiter.fused_moe import (
            _flydsl_route_output_nbytes,
            _stage2_runtime_uses_route_reduce,
            get_2stage_cfgs,
            get_padded_M,
        )

        max_bucket = get_padded_M(max_tokens)
        max_metadata = None
        required_tokens = {
            get_padded_M(1 << exponent) for exponent in range(max_tokens.bit_length())
        }
        required_tokens.add(max_bucket)
        for token in required_tokens:
            metadata = get_2stage_cfgs(
                token,
                model_dim,
                inter_dim,
                experts,
                topk,
                dtype,
                dtypes.fp8,
                dtypes.fp8,
                QuantType.per_1x32,
                True,
                ActivationType.Silu,
                doweight_stage1,
                hidden_pad,
                intermediate_pad,
                True,
                gate_mode,
                config_file=config_file,
            )
            metadata = _use_fhmoe_wrappers(metadata)
            if not _is_fhmoe_metadata_compatible(
                metadata,
                model_dim=model_dim,
                dtype=dtype,
                q_dtype_a=dtypes.fp8,
                q_dtype_w=dtypes.fp8,
                gate_mode=gate_mode,
            ):
                return False
            if token == max_bucket:
                max_metadata = metadata
        if max_metadata is None:
            return False
        if _stage2_runtime_uses_route_reduce(
            max_metadata.stage2,
            max_tokens,
            model_dim,
            force_reduce=force_reduce,
            stage2_fp8=stage2_fp8,
        ):
            output_element_size = torch.empty((), dtype=dtype).element_size()
            if (
                _flydsl_route_output_nbytes(
                    max_metadata.stage2,
                    max_tokens,
                    topk,
                    model_dim,
                    output_element_size,
                    stage2_fp8=stage2_fp8,
                )
                > FHMOE_BUFFER_MAX_BYTES
            ):
                return False
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def supports_mxfp8_fhmoe_config(
    max_tokens: int,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
) -> bool:
    """Return whether dedicated MXFP8/FP8 coverage is executable through M."""
    if type(max_tokens) is not int or max_tokens <= 0:
        return False
    try:
        if get_gfx_runtime() != "gfx950":
            return False
        if int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")) != 0:
            return False
        config_file = _fhmoe_config_file()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return _supports_mxfp8_fhmoe_config(
        max_tokens,
        config_file,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        gate_mode=gate_mode,
        doweight_stage1=doweight_stage1,
        force_reduce=os.environ.get("AITER_FLYDSL_FORCE_REDUCE", "0") == "1",
        stage2_fp8=os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1",
    )


def _resolve_mxfp8_fhmoe_config_file(
    max_tokens: int,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
) -> str | None:
    """Resolve a dedicated MXFP8 config, failing closed on partial matches."""
    config_file = _fhmoe_config_file()
    has_geometry = _config_has_mxfp8_fhmoe_geometry(
        config_file,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    )
    if not has_geometry:
        return None
    has_contract = _config_has_mxfp8_fhmoe_contract(
        config_file,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        gate_mode=gate_mode,
        doweight_stage1=doweight_stage1,
    )
    if not has_contract:
        raise ValueError(
            "MXFP8/FP8 FHMoE request partially matches a dedicated CSV "
            "contract but differs in dtype, padding, gate mode, or "
            "doweight_stage1"
        )
    if not supports_mxfp8_fhmoe_config(
        max_tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        gate_mode=gate_mode,
        doweight_stage1=doweight_stage1,
    ):
        raise NotImplementedError(
            "The dedicated MXFP8/FP8 FHMoE config does not cover "
            f"this request at M={max_tokens}"
        )
    return config_file


def _uses_dsv4_fhmoe_config(
    token_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
    routed_mxfp4: bool = True,
) -> bool:
    """Return whether dispatch should use the dedicated DSV4 FHMoE table."""
    return _is_dsv4_i384_fhmoe_contract(
        model_dim,
        inter_dim,
        experts,
        topk,
        hidden_pad,
        intermediate_pad,
        gate_mode,
        doweight_stage1,
        routed_mxfp4,
    ) and supports_dsv4_i384_fhmoe(token_num)


def fhmoe_fake(
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
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    output: torch.Tensor | None = None,
    # Model semantic forwarded by fused_moe; False leaves shared SiLU unclamped.
    clamp_shared: bool = True,
) -> torch.Tensor:
    del (
        w1,
        topk_weight,
        expert_mask,
        activation,
        quant_type,
        doweight_stage1,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        block_size_M,
        num_local_tokens,
        moe_sorting_dispatch_policy,
        hidden_pad,
        intermediate_pad,
        bias1,
        bias2,
        swiglu_limit,
        gate_mode,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        shared_expert_id,
        clamp_shared,
    )
    if output is not None:
        return output
    output_dtype = hidden_states.dtype if dtype is None else dtype
    return torch.empty(
        (topk_ids.shape[0], w2.shape[1]),
        dtype=output_dtype,
        device=topk_ids.device,
    )


@drop_presort_on_error
def _fhmoe_impl(
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
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    output: torch.Tensor | None = None,
    # Keep this schema default aligned with fused_moe's public API.
    clamp_shared: bool = True,
) -> torch.Tensor:
    # NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
    # Please do not modify this code without coordinating with the moev2 owners
    # (@qli88). See aiter/moev2/README.md.
    # Entry "fhmoe": an enabled path may run this call on aiter.moev2 instead of the
    # code below; DECLINED hands it back.
    if moev2_paths.any_active("fhmoe"):
        call = dict(locals())
        v2_path = moev2_paths.route_to_v2(call, "fhmoe")
        if v2_path is not None:
            from aiter import fused_moe as legacy

            with moev2_hooks.legacy_globals(
                legacy.kernel_bench_callable, legacy._fused_decode_sort_quant_disabled
            ):
                out = v2_path.run(call)
            if out is not moev2_paths.DECLINED:
                return out
    from aiter.fused_moe import _fused_moe_impl

    activation_enum = ActivationType(activation)
    quant_type_enum = QuantType(quant_type)
    gate_mode_enum = GateMode(gate_mode)
    _validate_fhmoe_contract(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        shared_expert_id,
        activation_enum,
        quant_type_enum,
        gate_mode_enum,
        expert_mask,
        bias1,
        bias2,
    )
    q_dtype_a = dtypes.fp8 if gate_mode_enum == GateMode.INTERLEAVE else dtypes.fp4x2
    if not clamp_shared and w1.dtype != dtypes.fp8:
        raise ValueError(
            "clamp_shared=False requires MXFP8 routed weights; "
            "MXFP4/FP8 FHMoE keeps the shared expert clamped"
        )
    experts = w1.shape[0]
    model_dim = w2.shape[1]
    inter_dim = w2.shape[2] * (model_dim // w1.shape[-1])
    topk = topk_ids.shape[1]
    metadata_config_file = None
    if _is_dsv4_i384_fhmoe_contract(
        model_dim,
        inter_dim,
        experts,
        topk,
        hidden_pad,
        intermediate_pad,
        gate_mode_enum,
        doweight_stage1,
        routed_mxfp4=w1.dtype == dtypes.fp4x2,
    ):
        if not supports_dsv4_i384_fhmoe(hidden_states.shape[0]):
            raise NotImplementedError(
                "The active FHMoE config does not cover this DSV4 I384 "
                f"token shape: M={hidden_states.shape[0]}"
            )
        from aiter.jit.core import AITER_CONFIGS

        metadata_config_file = AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE
    elif w1.dtype == dtypes.fp8:
        request_dtype = hidden_states.dtype if dtype is None else dtype
        metadata_config_file = _resolve_mxfp8_fhmoe_config_file(
            hidden_states.shape[0],
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            dtype=request_dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            gate_mode=gate_mode_enum,
            doweight_stage1=doweight_stage1,
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
        gate_mode=gate_mode,
        output=output,
        _q_dtype_a=q_dtype_a,
        _metadata_transform=_use_fhmoe_wrappers,
        _metadata_config_file=metadata_config_file,
        _enforce_metadata_block_size=True,
        _stage1_extra_args={
            "shared_w1": shared_w1,
            "shared_w1_scale": shared_w1_scale,
            "shared_expert_id": shared_expert_id,
            "swiglu_limit": swiglu_limit,
            "clamp_shared": clamp_shared,
        },
        _stage2_extra_args={
            "shared_w2": shared_w2,
            "shared_w2_scale": shared_w2_scale,
            "shared_expert_id": shared_expert_id,
        },
    )


@torch_compile_guard(gen_fake=fhmoe_fake)
def fhmoe_(
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
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    output: torch.Tensor | None = None,
    # Keep this schema default aligned with fused_moe's public API.
    clamp_shared: bool = True,
) -> torch.Tensor:
    return _fhmoe_impl(
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
        output=output,
        clamp_shared=clamp_shared,
    )


def _fhmoe(**kwargs) -> torch.Tensor:
    """Call the heterogeneous custom op using the public wrapper arguments."""
    kwargs["activation"] = kwargs["activation"].value
    kwargs["quant_type"] = kwargs["quant_type"].value
    block_size_M = kwargs.get("block_size_M")
    kwargs["block_size_M"] = -1 if not block_size_M else block_size_M
    return fhmoe_(**kwargs)
