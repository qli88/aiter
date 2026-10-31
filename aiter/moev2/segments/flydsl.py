# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL 2-stage kernels (`flydsl_moe{1,2}_*`, `flydsl_moe2_layout_*`).

The stage wrappers the legacy code builds its `MOEMetadata` from, and the stage-2
route-reduce queries; aiter/fused_moe.py imports them from here.
"""

import functools
import os
from collections.abc import Callable

import torch

import aiter
from aiter import ActivationType, dtypes
from aiter.fhmoe_contract import FHMOE_BUFFER_MAX_BYTES, fhmoe_reduce_output_nbytes
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel
from aiter.ops.opus import moe_stage2_a8w4 as _opus_a8w4


@functools.lru_cache(maxsize=1)
def _get_flydsl_moe_kernels():
    from aiter.ops.flydsl import moe_kernels

    return moe_kernels


def _is_flydsl_stage2(stage2: Callable) -> bool:
    func = getattr(stage2, "func", stage2)
    return func is _flydsl_stage2_wrapper or getattr(func, "_is_flydsl_stage2", False)


def _stage2_honors_runtime_reduce_override(stage2: Callable) -> bool:
    """Return whether stage2 reaches common runtime reduce resolution."""
    if not _is_flydsl_stage2(stage2):
        return False
    kernel_name = getattr(stage2, "keywords", {}).get("kernelName", "")
    parsed = _get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel_name)
    if parsed is None:
        raise ValueError(
            "Runtime reduce overrides require a recognized FlyDSL stage2 "
            f"kernel, got {kernel_name!r}"
        )
    # The a16w-mix launcher returns before the environment override. MXFP4
    # always atomically scatters; INT4 follows only its configured kernel mode.
    return not (
        parsed.get("a_dtype") == "bf16" and parsed.get("b_dtype") in ("fp4", "int4")
    )


def stage2_uses_route_reduce(stage2: Callable) -> bool:
    """Return True when stage2 writes per-slot route output then reduces it."""
    func = getattr(stage2, "func", stage2)
    kernel_name = getattr(stage2, "keywords", {}).get("kernelName", "")
    if _is_flydsl_stage2(func):
        parsed = _get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel_name)
        if parsed is None:
            return False
        # a16w4 (bf16 A x mxfp4 W) down-proj only supports atomic scatter into a
        # pre-zeroed buffer; the ported gemm2 launcher ignores the kernelName's
        # mode, so a "reduce"-named a16w4 config must still get accumulate=True
        # sorting (moe_buf zeroed). Treat any a16w4 stage2 as atomic here.
        if parsed.get("a_dtype") == "bf16" and parsed.get("b_dtype") == "fp4":
            return False
        return parsed.get("mode", "atomic") == "reduce"
    if func is _opus_a8w4.opus_a8w4_stage2_wrapper:
        return _opus_a8w4.stage2_uses_route_reduce(stage2)
    if func is _flydsl_v2_stage2_wrapper:
        cfg = parse_flydsl_v2_gemm2_kernel(kernel_name)
        return cfg is not None and cfg.get("epilog") == "reduce"
    return False


def _stage2_runtime_uses_route_reduce(
    stage2: Callable,
    token_num: int,
    model_dim: int,
    *,
    force_reduce: bool | None = None,
    stage2_fp8: bool | None = None,
) -> bool:
    """Return the effective route-output mode used by stage2 at runtime."""
    if stage2_uses_route_reduce(stage2):
        return True
    if not _stage2_honors_runtime_reduce_override(stage2):
        return False
    kernel_name = getattr(stage2, "keywords", {}).get("kernelName", "")
    parsed = _get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel_name)
    assert parsed is not None
    mode = _get_flydsl_moe_kernels().resolve_flydsl_stage2_mode(
        parsed.get("mode", "atomic"),
        token_num,
        model_dim,
        2,
        parsed.get("b_dtype", ""),
        False,
        force_reduce=force_reduce,
        stage2_fp8=stage2_fp8,
    )
    return mode == "reduce"


def _stage2_uses_fp8_route_output(
    stage2: Callable,
    *,
    stage2_fp8: bool | None = None,
    return_per_slot: bool = False,
) -> bool:
    """Return whether runtime reduce output uses packed FP8 value/scale rows."""
    if stage2_fp8 is None:
        stage2_fp8 = os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1"
    if not stage2_fp8 or return_per_slot:
        return False
    if not _stage2_honors_runtime_reduce_override(stage2):
        return False
    kernel_name = getattr(stage2, "keywords", {}).get("kernelName", "")
    parsed = _get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel_name)
    assert parsed is not None
    return parsed.get("b_dtype") in ("fp4", "fp8")


def _flydsl_route_output_nbytes(
    stage2: Callable,
    token_num: int,
    topk: int,
    model_dim: int,
    output_element_size: int,
    *,
    stage2_fp8: bool | None = None,
    return_per_slot: bool = False,
) -> int:
    """Return bytes allocated by the effective common stage2 reduce output."""
    if _stage2_uses_fp8_route_output(
        stage2,
        stage2_fp8=stage2_fp8,
        return_per_slot=return_per_slot,
    ):
        from aiter.ops.flydsl.kernels.mxfp4_gemm_common import fp8out_row_bytes

        kernels = _get_flydsl_moe_kernels()
        row_bytes = fp8out_row_bytes(
            model_dim,
            scale_blk=kernels._S2_LEGACY_FP8_SCALE_BLK,
            pitch_align=kernels._S2_LEGACY_FP8_PITCH_ALIGN,
        )
        return int(token_num) * int(topk) * int(row_bytes)
    return fhmoe_reduce_output_nbytes(
        token_num,
        topk,
        model_dim,
        output_element_size,
    )


def _validate_flydsl_route_output_nbytes(route_output_nbytes: int) -> None:
    if route_output_nbytes > FHMOE_BUFFER_MAX_BYTES:
        raise NotImplementedError(
            "FlyDSL stage2 reduce output exceeds the current 32-bit "
            f"buffer-addressing limit: {route_output_nbytes} bytes"
        )


def _normalize_bias_for_kernel(
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if bias is None:
        return bias
    if bias.dtype != torch.float32:
        raise TypeError(f"MoE bias must be fp32, got {bias.dtype}")
    return bias


# TODO: remove this function once kernel handles padding in the runtime
def _get_padding_for_flydsl(
    inter_dim_pad,
    model_dim_pad,
    bias: torch.Tensor | None = None,
):
    if bias is not None:
        return 0, 0
    return inter_dim_pad, model_dim_pad


def _flydsl_stage1_wrapper(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    activation=ActivationType.Silu,
    w1_scale=None,
    a1_scale=None,
    sorted_weights=None,
    out_scale=None,
    out_scale_sorted=None,
    bias1=None,
    topk_ids=None,
    swiglu_limit: float | None = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    out_dtype: str | None = None,
    v2_output_layout: bool = False,
    **_kwargs,
):
    inter_dim_pad, model_dim_pad = _get_padding_for_flydsl(
        inter_dim_pad, model_dim_pad, bias1
    )
    moe_kernels = _get_flydsl_moe_kernels()
    parsed = moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    if out_dtype is not None:
        parsed = {**parsed, "out_dtype": out_dtype}
    if activation == ActivationType.Swiglu:
        act = "swiglu"
    elif activation == ActivationType.Situv2:
        act = "situv2"
    elif activation == ActivationType.Silu:
        act = "silu"
    else:
        raise ValueError(f"Unsupported activation for FlyDSL MoE stage1: {activation}")
    _a_scale_one = parsed.get("a_scale_one", False)
    return moe_kernels.flydsl_moe_stage1(
        a=hidden_states,
        w1=w1,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        use_async_copy=True,
        k_batch=parsed.get("k_batch", 1),
        # None, matching stage 2: the int4 registry emits no waves_per_eu key.
        waves_per_eu=parsed.get("waves_per_eu", None),
        b_nt=parsed.get("b_nt", 2),
        gate_mode=parsed.get("gate_mode", "separated"),
        inter_dim_pad=inter_dim_pad,
        model_dim_pad=model_dim_pad,
        bias=_normalize_bias_for_kernel(bias1),
        topk_ids=topk_ids,
        a_scale_one=_a_scale_one,
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
        swiglu_limit=swiglu_limit,
        k_wave=parsed.get("k_wave", 1),
        v2_output_layout=v2_output_layout,
    )


def _flydsl_stage2_wrapper(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    bias2=None,
    block_m=None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    expert_mask=None,
    topk_ids=None,
    **_kwargs,
):
    inter_dim_pad, model_dim_pad = _get_padding_for_flydsl(
        inter_dim_pad, model_dim_pad, bias2
    )
    # `parsed` is the static per-kernel params dict registered at
    # import time by `get_flydsl_stage2_kernels` (see
    # aiter/ops/flydsl/moe_kernels.py) and pre-populated into
    # `_KERNEL_PARAMS` by `_register_all_configs()`. Variant-specific
    # knobs that live on the kernel name (e.g. the
    # `..._persist_async_w4_cumul3` production variant adds
    # `use_async_copy=True / waves_per_eu=4 / cu_num_mul=3`) are
    # already baked into this dict, so the `parsed.get(..., default)`
    # calls below pick up the registered values for that kernel name
    # rather than always falling back to defaults.
    moe_kernels = _get_flydsl_moe_kernels()
    parsed = moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    effective_sort_block_m = parsed.get("sort_block_m", 0) or parsed["tile_m"]
    if block_m is not None and int(block_m) != effective_sort_block_m:
        raise ValueError(
            "FlyDSL stage2 sorting layout mismatch: "
            f"moe_sorting uses block_m={int(block_m)}, but {kernelName} expects "
            f"sort_block_m={effective_sort_block_m}. Select a kernel with "
            f"_sbm{int(block_m)}."
        )
    return moe_kernels.flydsl_moe_stage2(
        inter_states=inter_states,
        w2=w2,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        mode=parsed.get("mode", "atomic"),
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=parsed.get("sort_block_m", 0),
        waves_per_eu=parsed.get("waves_per_eu", None),
        use_async_copy=parsed.get("use_async_copy", False),
        cu_num_mul=parsed.get("cu_num_mul", 1),
        b_nt=parsed.get("b_nt", 0),
        persist=parsed.get("persist", None),
        inter_dim_pad=inter_dim_pad,
        model_dim_pad=model_dim_pad,
        bias=bias2,
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
        expert_mask=expert_mask,
        topk_ids=topk_ids,
    )


def _empty_bf16(device):
    return torch.empty((0,), dtype=dtypes.bf16, device=device)


def _empty_u8(device):
    return torch.empty((0,), dtype=torch.uint8, device=device)


def _mxfp4_scale_u8(scale):
    """FlyDSL can't ingest fp4/e8m0 dtype codes via DLPack, so pass a uint8 view (the
    same reinterpret_cast HIP does). Returns the uint8 view, or the input (already
    uint8, or None) unchanged.

    Not memoized: tensors hash by identity, so per-call intermediates never hit and
    the cache would pin them on the device instead."""
    if scale is not None and scale.element_size() == 1 and scale.dtype != torch.uint8:
        return scale.view(torch.uint8)
    return scale


def _flydsl_stage2_fp8_enabled():
    return os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1"


def _mxfp4_intermediate_enabled():
    # Lossy before-sum 4-bit quant of the scatter intermediate (ok for gsm8k, degrades other evals).
    return os.environ.get("AITER_MXFP4_INTERMEDIATE", "0") == "1"


def _flydsl_v2_stage2_wrapper(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    model_dim=0,
    inter_dim=0,
    num_experts=0,
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    bias2=None,
    block_m=None,
    expert_mask=None,
    topk_ids=None,
    topk_weights=None,
    reverse_sorted=None,
    use_valid_token_count=False,
    **_kwargs,
):
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import (
        _validate_v2_gemm2_dtypes,
        mxfp4_moe_gemm2,
    )

    cfg = parse_flydsl_v2_gemm2_kernel(kernelName)
    if cfg is None:
        raise ValueError(f"Invalid FlyDSL v2 GEMM2 kernel name: {kernelName}")
    _validate_v2_gemm2_dtypes(cfg["a_dtype"], cfg["b_dtype"])

    bm = cfg["tile_m"]
    bn = cfg["tile_n"]
    bk = cfg["tile_k"]
    sbm = cfg["sort_block_m"] or (int(block_m) if block_m else bm)
    if cfg["sort_block_m"] and block_m is not None and int(block_m) != sbm:
        raise ValueError(
            "FlyDSL v2 stage2 sorting layout mismatch: moe_sorting uses "
            f"block_m={int(block_m)}, but the kernel expects sort_block_m={sbm}."
        )
    epilog = cfg["epilog"]
    max_sorted = inter_states.shape[0]

    token_num = out.shape[0]
    model_dim_runtime = out.shape[1]
    target = out
    _s2_fp8_inter = epilog == "reduce" and _flydsl_stage2_fp8_enabled()
    if _s2_fp8_inter:
        _s2_fp8_inter = sorted_weights is not None and topk_weights is not None
    _defer_w = _s2_fp8_inter
    _fp8_scale_blk = None
    _fp8_pitch_align = None
    _s2_fp4_scatter = (
        epilog == "scatter"
        and bn in (128, 256)
        and bias2 is None
        and _mxfp4_intermediate_enabled()
    )
    target_scale = None
    if epilog == "scatter":
        if reverse_sorted is None or sorted_weights is None:
            raise ValueError(
                "epilog='scatter' FlyDSL GEMM2 requires reverse_sorted and sorted_weights"
            )
        if _s2_fp4_scatter:
            target = torch.empty(
                (max_sorted, model_dim_runtime // 2),
                dtype=torch.uint8,
                device=out.device,
            )
            target_scale = torch.empty(
                (max_sorted, model_dim_runtime // 32),
                dtype=torch.uint8,
                device=out.device,
            )
        else:
            target = torch.empty(
                (max_sorted, model_dim_runtime), dtype=out.dtype, device=out.device
            )
    elif epilog == "reduce":
        if _s2_fp8_inter:
            from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
                FP8OUT_PITCH_ALIGN,
                FP8OUT_SCALE_BLK_MIN,
                fp8out_row_bytes,
                fp8out_scale_blk,
            )

            if model_dim_runtime % FP8OUT_SCALE_BLK_MIN != 0:
                raise ValueError(
                    "AITER_FLYDSL_STAGE2_FP8 requires model_dim to be divisible "
                    f"by {FP8OUT_SCALE_BLK_MIN}"
                )
            _fp8_scale_blk = fp8out_scale_blk(model_dim_runtime)
            _fp8_pitch_align = FP8OUT_PITCH_ALIGN

            target = torch.empty(
                (
                    token_num * topk,
                    fp8out_row_bytes(
                        model_dim_runtime,
                        scale_blk=_fp8_scale_blk,
                        pitch_align=_fp8_pitch_align,
                    ),
                ),
                dtype=torch.uint8,
                device=out.device,
            )
        else:
            target = torch.empty(
                (token_num, topk, model_dim_runtime),
                dtype=out.dtype,
                device=out.device,
            )
        if expert_mask is not None:
            # EP sorting omits remote and fake routes, so GEMM2 intentionally
            # leaves their per-route slots unwritten. Zero them before the
            # masked reduction to prevent speculative loads of stale NaN data.
            target.zero_()
    mxfp4_moe_gemm2(
        inter_sorted_quant=_mxfp4_scale_u8(inter_states),
        inter_sorted_shuffled_scale=_mxfp4_scale_u8(a2_scale),
        w2_u8=_mxfp4_scale_u8(w2),
        w2_scale_u8=_mxfp4_scale_u8(w2_scale),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        out=target,
        M_logical=token_num,
        max_sorted=max_sorted,
        NE=num_experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        BM=bm,
        BN=bn,
        BK=bk,
        use_nt=cfg["use_nt"],
        a_dtype=cfg["a_dtype"],
        b_dtype=cfg["b_dtype"],
        epilog=epilog,
        SBM=sbm,
        persist=cfg["persist"],
        g2_bf16_lds=cfg["bf16_lds"],
        g2_spart=cfg["spart"],
        out_dtype="fp8" if _s2_fp8_inter else ("fp4" if _s2_fp4_scatter else "bf16"),
        bias=bias2,
        is_ep=expert_mask is not None,
        out_scale=target_scale,
        use_valid_token_count=use_valid_token_count,
    )
    if _s2_fp4_scatter:
        aiter.mxfp4_moe_scatter_reduce_q(
            flat_out_q=target,
            flat_out_scale=target_scale,
            reverse_sorted=reverse_sorted,
            sorted_weights=sorted_weights,
            out=out,
            NE=num_experts,
            TOPK=topk,
            D_HIDDEN=model_dim_runtime,
            MB=bm,
        )
        return out
    if epilog == "scatter":
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=target,
            reverse_sorted=reverse_sorted,
            sorted_weights=sorted_weights,
            out=out,
            NE=num_experts,
            TOPK=topk,
            D_HIDDEN=model_dim_runtime,
            MB=bm,
        )
        return out
    if epilog == "reduce":
        from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

        _run_moe_reduction(
            target,
            out,
            token_num,
            topk,
            model_dim_runtime,
            expert_mask=expert_mask,
            topk_ids=topk_ids,
            is_fp8=_s2_fp8_inter,
            topk_weights=topk_weights if _defer_w else None,
            fp8_scale_blk=_fp8_scale_blk,
            fp8_pitch_align=_fp8_pitch_align,
        )
    return out


def _flydsl_mxfp4_layout_is_compatible(q_dtype_a, gate_mode, gate_up_interleave=None):
    if gate_up_interleave is not None:
        return (gate_up_interleave is False and gate_mode == GateMode.SEPARATED) or (
            gate_up_interleave is True and gate_mode == GateMode.INTERLEAVE
        )
    return (
        q_dtype_a in (dtypes.bf16, dtypes.fp4x2) and gate_mode == GateMode.SEPARATED
    ) or (q_dtype_a == dtypes.fp8 and gate_mode == GateMode.INTERLEAVE)
