# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus a8w4 stages (`aiter.ops.opus`): mxfp8 activations into MXFP4 weights.

`_opus_stage2_fp8_enabled` moved here verbatim from aiter/fused_moe.py, which imports it
back.

Segments, for tuned rows only (legacy has no Opus default): `opus.gemm1` writes the
mxfp8 intermediate itself, for a stage 2 that reads it (Opus, FlyDSL, CK-Tile; the v2
layout for `opus_moe2_layout_*`); `opus.gemm1_layout` writes the sorted input of a
FlyDSL layout GEMM2; `opus.gemm2` runs an Opus stage 2 (`opus_moe2_*`,
`opus_moe2_layout_*`; a layout name after a FlyDSL stage 1 needs one quantizing its
output). The kernels take the row's `block_m` (stage 1) and sort `block_m` (stage 2).
"""

import functools
import os

from aiter import ActivationType, dtypes
from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel
from aiter.ops.opus import moe_stage2_a8w4 as _opus_a8w4
from aiter.ops.opus.moe_stage1_a8w4 import (
    opus_a8w4_stage1_wrapper as _opus_a8w4_stage1_wrapper,
)

from ..dispatch import kernel_names, overrides
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from .base import Params, RowRejection, Segment
from .defaults import BLOCK_SIZE_M, TUNE_BLOCK_MS, default_block_m, tunable
from .mxfp4 import (
    a8w4_caps,
    fused_inter,
    layout_gemm2_inter,
    mxfp4_row_rejection,
    swiglu_bias,
    with_doweight,
)


def _opus_stage2_fp8_enabled():
    return os.environ.get("AITER_OPUS_STAGE2_FP8", "1") == "1"


def _row_block_m(row):
    return int(row.get("block_m", BLOCK_SIZE_M))


@functools.lru_cache(maxsize=2048)
def _opus_candidates(model_dim, inter_dim, token, stage1) -> tuple:
    """`(stage, kernel, block_m)` of the legacy tuner's `gen_opus_2stages_task` for a
    tuned a8w4 key (`stage1`: its stage 1 filters pass); each stage 1 entry stands for
    both of legacy's tasks (token-major and sorted output)."""
    from csrc.opus_moe.opus_moe_common import (
        OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT as contract,
    )
    from csrc.opus_moe.opus_moe_common import (
        get_opus_a8w4_stage2_kernels,
        opus_a8w4_stage1_instances_for_shape,
    )

    if inter_dim % (contract.bk_logical // contract.fp4_values_per_byte):
        return ()
    insts = opus_a8w4_stage1_instances_for_shape(
        model_dim=model_dim, logical_inter_dim=inter_dim, inter_dim_pad=0
    )
    stage2 = get_opus_a8w4_stage2_kernels(token=token)
    out = []
    for block_m in TUNE_BLOCK_MS:
        if stage1:
            out += [
                (1, inst.profile_name, block_m)
                for inst in insts
                if int(inst.block_m) == block_m
            ]
        for name, k in stage2.items():
            block_n = int(k.get("kernel_block_n"))
            if k["sort_block_m"] != block_m or block_n <= 0 or model_dim % block_n:
                continue
            layout = name.replace("opus_moe2_", "opus_moe2_layout_", 1)
            out += [(2, name, block_m), (2, layout, block_m)]
    return tuple(out)


def opus_param_space(segment, p) -> tuple:
    """The Opus candidates `segment` runs: its stage. Legacy's stage 1 filters: bf16
    output, Silu/Swiglu/SiTUv2, no `doweight_stage1`."""
    if not (
        segment.accepts(p)
        and tunable(p)
        and p.act_in.dtype == dtypes.fp8
        and p.w1.dtype == dtypes.fp4x2
    ):
        return ()
    stage = 1 if Op.GEMM1 in segment.covers else 2
    stage1 = (
        p.out_dtype == dtypes.bf16
        and p.activation.activation
        in (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)
        and "doweight_stage1" not in p.features
    )
    s = p.shape
    return tuple(
        Params(kernel=name, block_m=block_m)
        for st, name, block_m in _opus_candidates(s.model_dim, s.inter_dim, s.m, stage1)
        if st == stage and segment.owns(kernel_names.parse(name))
    )


class _OpusStage1(Segment):
    prefixes = (kernel_names.OPUS,)
    structure_params = frozenset({"block_m"})

    @property
    def capabilities(self):
        return a8w4_caps()

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.OPUS
            and kernel.stage == 1
        )

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def row_params(self, p, row):
        return {"block_m": _row_block_m(row)}

    def build(self, p, params):
        return functools.partial(
            _opus_a8w4_stage1_wrapper,
            kernelName=params.kernel,
            activation=p.activation.activation,
            block_m=params.block_m,
            inter_dim_pad=p.layout.intermediate_pad,
        )

    def metadata_fields(self, p, params):
        return {"has_bias": swiglu_bias(p)}

    def param_space(self, p):
        return opus_param_space(self, p)


class OpusGemm1(_OpusStage1):
    """An Opus stage 1 writing the mxfp8 intermediate (in the v2 layout for an Opus
    layout GEMM2)."""

    name = "opus.gemm1"
    reach = "tuned"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    def produces(self, p, params):
        return fused_inter(p.act_in, params.get("kernel2"))


class OpusGemm1Layout(_OpusStage1):
    """An Opus stage 1 writing the sorted, quantized input of a FlyDSL layout GEMM2."""

    name = "opus.gemm1_layout"
    reach = "tuned"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    def reads(self, p):
        return (
            Buf("sorted", None, rows=frozenset({"expert_sorted", "aux_sorted"})),
            Buf("tokens", p.act_in),
        )

    def produces(self, p, params):
        return Buf(
            "inter",
            layout_gemm2_inter(params.get("kernel2")),
            rows="expert_sorted",
            scales="v2",
        )

    def build(self, p, params):
        stage1 = super().build(p, params)
        stage1.keywords["out_dtype"] = self._a_dtype(params)
        return stage1

    @staticmethod
    def _a_dtype(params):
        return parse_flydsl_v2_gemm2_kernel(params.kernel2)["a_dtype"]


class OpusGemm2(Segment):
    """An Opus stage 2, after a FlyDSL, Opus, CK or CK-Tile stage 1 (a layout name
    after a FlyDSL one: `flydsl.gemm1_q`)."""

    name = "opus.gemm2"
    reach = "tuned"
    prefixes = (kernel_names.OPUS,)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return with_doweight(a8w4_caps())

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.OPUS
            and kernel.stage == 2
            and _opus_a8w4.is_opus_a8w4_stage2_kernel(kernel.format())
        )

    def reads(self, p):
        return (Buf("sorted", None), Buf("inter", p.act_in))

    def reads_with(self, p, params):
        """A layout name after a FlyDSL or Opus stage 1 reads the v2 layout (legacy
        `skip_inter_quant`); after another stage 1 legacy quantizes the intermediate
        as for any other name."""
        sort, inter = self.reads(p)
        if params.get("kernel1", "").startswith(("flydsl_", "opus_moe1_")):
            return (sort, fused_inter(inter.fmt, params.get("kernel")))
        return (sort, inter)

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def declines_row(self, p, row):
        if not overrides.bypass_tuned() or self.check_row(p, row) is not None:
            return None
        # Bypassing the row, legacy still parses its Opus launch, with the default
        # block_m.
        s = p.shape
        block_m = default_block_m(
            p.quant_type, s.m, s.topk, s.experts, s.inter_dim, s.model_dim
        )
        try:
            _opus_a8w4.parse_stage2_config(row, block_m)
        except ValueError as error:
            return f"legacy raises for the bypassed Opus row: {error}"
        return None

    def check_row(self, p, row, kernel_name1=""):
        rejection = mxfp4_row_rejection(p, row)
        if rejection is not None:
            return rejection
        kn2 = str(row.get("kernelName2", "") or "").strip()
        supported, reason = _opus_a8w4.cfg_is_supported(
            kn2,
            cfg=row,
            gfx=p.arch,
            block_m=row.get("block_m", BLOCK_SIZE_M),
            is_ep=p.is_ep,
            has_stage2_bias="bias2" in p.features,
        )
        if supported:
            return None
        return RowRejection(
            f"[fused_moe] Opus stage2 config unsupported ({reason}); "
            "using default heuristics"
        )

    def row_params(self, p, row):
        route = _opus_a8w4.route_bucket_metadata(row)
        return {"sort_block_m": _row_block_m(row), "route": tuple(route.items())}

    def build(self, p, params):
        launch = _opus_a8w4.parse_stage2_config(
            {"kernelName2": params.kernel}, params.sort_block_m
        )
        return functools.partial(
            _opus_a8w4.opus_a8w4_stage2_wrapper,
            kernelName=params.kernel,
            inter_dim_pad=p.layout.intermediate_pad,
            model_dim_pad=p.layout.hidden_pad,
            launch=launch,
        )

    def metadata_fields(self, p, params):
        return dict(params.route)

    def param_space(self, p):
        return opus_param_space(self, p)
