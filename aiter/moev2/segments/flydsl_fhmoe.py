# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segments of the FHMoE FlyDSL kernels (`aiter.fhmoe`): the shared expert is the last
routed row (`shared_expert_id == E - 1`), its fp8 1x32 weights passed beside the routed
MXFP4 or MXFP8 ones.

The FHMoE stages run the kernels of the plain FlyDSL stages (`flydsl.gemm1`,
`flydsl.gemm1_q`, `flydsl.gemm2`) through the FHMoE wrappers, as the legacy
`_use_fhmoe_wrappers` swaps them in; the shared-expert tensors come from the call
(`call_args`). The contract checks and the dedicated-table helpers live here too, and
`aiter.fhmoe` imports them.
"""

import csv
import functools

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from ..problem import scale_spec
from .base import Cap
from .mxfp4 import (
    FlydslGemm1,
    FlydslGemm1Quant,
    FlydslGemm2,
    a4w4_caps,
    a8w4_caps,
    mxfp8_caps,
)


def _validate_fhmoe_contract(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    shared_w1: torch.Tensor | None,
    shared_w2: torch.Tensor | None,
    shared_w1_scale: torch.Tensor | None,
    shared_w2_scale: torch.Tensor | None,
    shared_expert_id: int,
    activation: ActivationType,
    quant_type: QuantType,
    gate_mode: GateMode,
    expert_mask: torch.Tensor | None,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
) -> None:
    shared_args = (shared_w1, shared_w2, shared_w1_scale, shared_w2_scale)
    has_shared_expert = all(arg is not None for arg in shared_args)
    if any(arg is not None for arg in shared_args) and not has_shared_expert:
        raise ValueError(
            "shared_w1, shared_w2, shared_w1_scale, and shared_w2_scale "
            "must be provided together"
        )
    if not has_shared_expert:
        raise ValueError(
            "shared_expert_id requires shared_w1, shared_w2, and their scales"
        )

    E = w1.shape[0]
    model_dim = w2.shape[1]
    inter_dim = w2.shape[2] * (model_dim // w1.shape[-1])
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts currently require gfx950"
        )
    if quant_type != QuantType.per_1x32:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require "
            "per_1x32 quantization"
        )
    if activation != ActivationType.Silu:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts currently require SiLU"
        )
    if gate_mode not in (GateMode.INTERLEAVE, GateMode.SEPARATED):
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require "
            "interleaved or separated gate/up weights"
        )
    if expert_mask is not None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts do not yet "
            "support expert masks"
        )
    if bias1 is not None or bias2 is not None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts do not support expert biases"
        )
    if shared_expert_id != E - 1:
        raise ValueError(
            "The heterogeneous FlyDSL path requires a dummy final routed "
            f"weight row and shared_expert_id == E - 1; got {shared_expert_id=} "
            f"and E={E}"
        )
    if w1.dtype != w2.dtype:
        raise ValueError(
            "Heterogeneous routed W1/W2 must use the same dtype, got "
            f"{w1.dtype} and {w2.dtype}"
        )
    if w1.dtype not in (dtypes.fp4x2, dtypes.fp8):
        raise ValueError("Heterogeneous routed weights must use MXFP4 or MXFP8")
    if w1.dtype == dtypes.fp8 and gate_mode != GateMode.INTERLEAVE:
        raise ValueError(
            "Heterogeneous MXFP8 routed weights require interleaved gate/up layout"
        )
    if w1.shape[1] != 2 * inter_dim:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require "
            "gate and up projections"
        )
    assert shared_w1 is not None and shared_w2 is not None
    assert shared_w1_scale is not None and shared_w2_scale is not None
    if shared_w1.dtype != dtypes.fp8 or shared_w2.dtype != dtypes.fp8:
        raise ValueError("Heterogeneous shared weights must use FP8 E4M3")
    if shared_w1.shape != (1, w1.shape[1], model_dim):
        raise ValueError(
            f"Expected shared_w1 shape {(1, w1.shape[1], model_dim)}, "
            f"got {tuple(shared_w1.shape)}"
        )
    if shared_w2.shape != (1, model_dim, inter_dim):
        raise ValueError(
            f"Expected shared_w2 shape {(1, model_dim, inter_dim)}, "
            f"got {tuple(shared_w2.shape)}"
        )

    scale_dtypes = (dtypes.fp8_e8m0, torch.uint8)
    scale_tensors = (w1_scale, w2_scale, shared_w1_scale, shared_w2_scale)
    if any(scale is None or scale.dtype not in scale_dtypes for scale in scale_tensors):
        raise ValueError(
            "Heterogeneous routed/shared scales must use FP8 E8M0 or raw "
            "uint8 E8M0 storage"
        )

    stage1_scale_shape = (
        ((2 * inter_dim + 255) // 256) * 256,
        ((model_dim // 32 + 7) // 8) * 8,
    )
    stage2_scale_k = ((inter_dim + 255) // 256) * 256
    stage2_scale_shape = (
        ((model_dim + 255) // 256) * 256,
        ((stage2_scale_k // 32 + 7) // 8) * 8,
    )
    if tuple(shared_w1_scale.shape) != stage1_scale_shape:
        raise ValueError(
            f"Expected preshuffled shared_w1_scale shape {stage1_scale_shape}, "
            f"got {tuple(shared_w1_scale.shape)}"
        )
    if tuple(shared_w2_scale.shape) != stage2_scale_shape:
        raise ValueError(
            f"Expected preshuffled shared_w2_scale shape {stage2_scale_shape}, "
            f"got {tuple(shared_w2_scale.shape)}"
        )

    assert w1_scale is not None and w2_scale is not None
    expected_w1_scale_numel = E * 2 * inter_dim * (model_dim // 32)
    expected_w2_scale_numel = E * model_dim * (stage2_scale_k // 32)
    if w1_scale.ndim not in (1, 2) or w2_scale.ndim not in (1, 2):
        raise ValueError(
            "Preshuffled routed scales must be flattened or 2D; raw "
            f"[E, N, K/32] layouts are unsupported, got {w1_scale.ndim}D/"
            f"{w2_scale.ndim}D"
        )
    if w1_scale.numel() != expected_w1_scale_numel:
        raise ValueError(
            "Expected preshuffled routed w1_scale to contain "
            f"{expected_w1_scale_numel} elements, got {w1_scale.numel()}"
        )
    if w2_scale.numel() != expected_w2_scale_numel:
        raise ValueError(
            "Expected preshuffled routed w2_scale to contain "
            f"{expected_w2_scale_numel} elements, got {w2_scale.numel()}"
        )

    tensors = (
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
    )
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError(
            "Heterogeneous routed/shared weights and scales must be on the "
            "hidden-state device"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "Heterogeneous routed/shared weights and scales must be contiguous"
        )


def _flydsl_fhmoe_stage1_wrapper(
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
    block_m=None,
    swiglu_limit: float | None = None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    shared_w1=None,
    shared_w1_scale=None,
    shared_expert_id: int = -1,
    clamp_shared: bool = True,
    v2_output_layout: bool = False,
    **_kwargs,
):
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.fhmoe import flydsl_fhmoe_stage1

    parsed = moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    act = "swiglu" if activation == ActivationType.Swiglu else "silu"
    return flydsl_fhmoe_stage1(
        a=hidden_states,
        w1=w1,
        shared_w1=shared_w1,
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
        w1_scale=w1_scale,
        shared_w1_scale=shared_w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        use_async_copy=True,
        k_batch=parsed.get("k_batch", 1),
        waves_per_eu=parsed.get("waves_per_eu", 3),
        b_nt=parsed.get("b_nt", 2),
        gate_mode=parsed.get("gate_mode", "separated"),
        inter_dim_pad=inter_dim_pad,
        model_dim_pad=model_dim_pad,
        bias=bias1,
        topk_ids=topk_ids,
        a_scale_one=parsed.get("a_scale_one", False),
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
        swiglu_limit=swiglu_limit,
        k_wave=parsed.get("k_wave", 1),
        v2_output_layout=v2_output_layout,
        shared_expert_id=shared_expert_id,
        clamp_shared=clamp_shared,
    )


def _flydsl_fhmoe_stage2_wrapper(
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
    shared_w2=None,
    shared_w2_scale=None,
    shared_expert_id: int = -1,
    **_kwargs,
):
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.fhmoe import flydsl_fhmoe_stage2

    parsed = moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    return flydsl_fhmoe_stage2(
        inter_states=inter_states,
        w2=w2,
        shared_w2=shared_w2,
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
        shared_w2_scale=shared_w2_scale,
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
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
        bias=bias2,
        expert_mask=expert_mask,
        topk_ids=topk_ids,
        shared_expert_id=shared_expert_id,
    )


_flydsl_fhmoe_stage2_wrapper._is_flydsl_stage2 = True


def _validate_fhmoe_metadata_consistency(metadata):
    """Validate shared stage identity and sorting-block metadata."""
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    stage1 = get_flydsl_kernel_params(metadata.stage1.keywords.get("kernelName", ""))
    stage2 = get_flydsl_kernel_params(metadata.stage2.keywords.get("kernelName", ""))
    if stage1 is None or stage2 is None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require valid FlyDSL kernels"
        )
    stage2_sort_block = int(stage2.get("sort_block_m", 0) or stage2["tile_m"])
    if (
        stage1["stage"] != 1
        or stage2["stage"] != 2
        or int(stage1["tile_m"]) != int(metadata.block_m)
        or stage2_sort_block != int(metadata.block_m)
    ):
        raise NotImplementedError(
            "FHMoE metadata block_m must match stage1 tile_m and stage2 sort_block_m"
        )
    return stage1, stage2


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == dtypes.fp8:
        return "fp8"
    if dtype == dtypes.fp4x2:
        return "fp4"
    if dtype == dtypes.bf16:
        return "bf16"
    if dtype == dtypes.fp16:
        return "f16"
    return str(dtype)


def _is_fhmoe_metadata_compatible(
    metadata,
    *,
    model_dim: int,
    dtype: torch.dtype,
    q_dtype_a: torch.dtype,
    q_dtype_w: torch.dtype,
    gate_mode: GateMode,
) -> bool:
    """Validate a dedicated row against model-agnostic FHMoE request fields."""
    stage1, stage2 = _validate_fhmoe_metadata_consistency(metadata)
    a_dtype = _dtype_name(q_dtype_a)
    b_dtype = _dtype_name(q_dtype_w)
    out_dtype = _dtype_name(dtype)
    stage2_a_dtype = metadata.fuse_quant or a_dtype
    return (
        stage1["a_dtype"] == a_dtype
        and stage1["b_dtype"] == stage2["b_dtype"] == b_dtype
        and stage2["a_dtype"] == stage2_a_dtype
        and stage2["out_dtype"] == out_dtype
        and stage1.get("gate_mode") == gate_mode.value
        and _valid_fhmoe_kernel_geometry(stage1, stage2, model_dim=model_dim)
    )


def _is_dsv4_i384_metadata(metadata) -> bool:
    """Validate that a dedicated DSV4 row names compatible FlyDSL kernels."""
    stage1, stage2 = _validate_fhmoe_metadata_consistency(metadata)
    return (
        stage1["a_dtype"] == stage2["a_dtype"] == "fp8"
        and stage1["b_dtype"] == stage2["b_dtype"] == "fp4"
        and stage2["out_dtype"] == "bf16"
        and stage1.get("gate_mode") == "interleave"
        and _valid_fhmoe_kernel_geometry(stage1, stage2, model_dim=7168)
    )


def _valid_fhmoe_kernel_geometry(stage1, stage2, *, model_dim: int) -> bool:
    """Return whether split-K/wave and stage-2 sorting geometry can compile."""
    stage1_k_wave = int(stage1.get("k_wave", 1))
    stage1_k_batch = int(stage1.get("k_batch", 1))
    if stage1_k_batch != 1 or stage1_k_wave < 1:
        return False
    stage1_k_dim = model_dim // stage1_k_batch
    stage2_sort_block = int(stage2.get("sort_block_m", 0) or stage2["tile_m"])
    return (
        stage1_k_dim % stage1_k_wave == 0
        and (stage1_k_dim // stage1_k_wave) % int(stage1["tile_k"]) == 0
        and stage2_sort_block % int(stage2["tile_m"]) == 0
    )


@functools.cache
def _config_has_mxfp8_fhmoe_geometry(
    config_file: str,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
) -> bool:
    """Return whether the dedicated CSV declares this MXFP8/FP8 geometry."""
    with open(config_file, newline="") as stream:
        rows = csv.DictReader(stream)
        return any(
            int(row["model_dim"]) == model_dim
            and int(row["inter_dim"]) == inter_dim
            and int(row["expert"]) == experts
            and int(row["topk"]) == topk
            and int(row["shared_expert_id"]) == experts - 1
            and row["q_dtype_w"] == str(dtypes.fp8)
            for row in rows
        )


@functools.cache
def _config_has_mxfp8_fhmoe_contract(
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
) -> bool:
    """Return whether the dedicated CSV declares this full MXFP8 contract."""
    if not _config_has_mxfp8_fhmoe_geometry(
        config_file,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    ):
        return False
    with open(config_file, newline="") as stream:
        rows = csv.DictReader(stream)
        return any(
            int(row["model_dim"]) == model_dim
            and int(row["inter_dim"]) == inter_dim
            and int(row["expert"]) == experts
            and int(row["topk"]) == topk
            and int(row["shared_expert_id"]) == experts - 1
            and row["act_type"] == str(ActivationType.Silu)
            and row["dtype"] == str(dtype)
            and row["q_dtype_a"] == str(dtypes.fp8)
            and row["q_dtype_w"] == str(dtypes.fp8)
            and row["q_type"] == str(QuantType.per_1x32)
            and int(row["use_g1u1"]) == 1
            and int(row["doweight_stage1"]) == int(doweight_stage1)
            and int(row["hidden_pad"]) == hidden_pad
            and int(row["intermediate_pad"]) == intermediate_pad
            and row["gate_mode"] == str(gate_mode)
            for row in rows
        )


def _is_dsv4_i384_fhmoe_contract(
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
    return (
        (model_dim, inter_dim, experts, topk) == (7168, 384, 385, 7)
        and routed_mxfp4
        and hidden_pad == 0
        and intermediate_pad == 0
        and gate_mode == GateMode.INTERLEAVE
        and not doweight_stage1
    )


def shared_weights():
    """The shared expert's weight format: fp8 with one e8m0 scale per 32."""
    return scale_spec(QuantType.per_1x32, dtypes.fp8)


def fhmoe_caps():
    """The plain FlyDSL formats with the shared expert as the last routed row: Silu
    only, without expert masks or bias, and keyed on a dedicated table when the call
    names one (`_validate_fhmoe_contract` checks the rest before the dispatch)."""
    return tuple(
        Cap(
            arch=cap.arch,
            act_in=cap.act_in,
            w1=cap.w1,
            w2=cap.w2,
            out_dtype=cap.out_dtype,
            activation=frozenset({"silu"}),
            gate_mode=cap.gate_mode,
            g1u1=cap.g1u1,
            pads=True,
            extras=cap.extras | {"tuned_table"},
            shared=True,
        )
        for cap in a4w4_caps() + a8w4_caps() + mxfp8_caps()
    )


class _FhmoeStage1:
    """The FHMoE stage 1 over a plain FlyDSL stage 1 (first base after this one)."""

    @property
    def capabilities(self):
        return fhmoe_caps()

    def build(self, p, params):
        stage1 = super().build(p, params)
        return functools.partial(
            _flydsl_fhmoe_stage1_wrapper, *stage1.args, **stage1.keywords
        )

    def call_args(self, c):
        return {
            "shared_w1": c["shared_w1"],
            "shared_w1_scale": c["shared_w1_scale"],
            "shared_expert_id": c["shared_expert_id"],
            "swiglu_limit": c["swiglu_limit"],
            "clamp_shared": c["clamp_shared"],
        }


class FlydslFhmoeGemm1(_FhmoeStage1, FlydslGemm1):
    """`flydsl.gemm1` with the shared expert."""

    name = "flydsl_fhmoe.gemm1"


class FlydslFhmoeGemm1Quant(_FhmoeStage1, FlydslGemm1Quant):
    """`flydsl.gemm1_q` with the shared expert."""

    name = "flydsl_fhmoe.gemm1_q"


class FlydslFhmoeGemm2(FlydslGemm2):
    """`flydsl.gemm2` with the shared expert."""

    name = "flydsl_fhmoe.gemm2"

    @property
    def capabilities(self):
        return fhmoe_caps()

    def build(self, p, params):
        stage2 = super().build(p, params)
        return functools.partial(
            _flydsl_fhmoe_stage2_wrapper, *stage2.args, **stage2.keywords
        )

    def call_args(self, c):
        return {
            "shared_w2": c["shared_w2"],
            "shared_w2_scale": c["shared_w2_scale"],
            "shared_expert_id": c["shared_expert_id"],
        }
