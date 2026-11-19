# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w4`: mxfp8 activations into MXFP4 weights (1x32 groups, e8m0 scales).

Claims `fused_moe_` calls with `QuantType.per_1x32` and g1u1 fp4x2 weights on gfx950
whose activations are fp8 (the `policy.py` choice or `quant_dtype_a`), Silu, Swiglu or
SiTUv2, separated or interleaved gate/up, outside AITER_ONLINE_TUNE, without
`quant_dtype_a2`, global scales, scatter, HERD routing, deterministic mode or registered
presort. Today these resolve to the FlyDSL heuristic (gate/up interleaved, or separated
with w1 tagged `gate_up_interleave=False`), CK-Tile (Swiglu; split-K on preshuffled
weights), CK 2-stage (Silu and SiTUv2 otherwise, which ignores the padding), or a tuned
FlyDSL, MXMOE or Opus row (FlyDSL, layout-v2, Opus or CK-Tile stage 2), or a tuned row
with a `cktile_*` kernelName1, which legacy runs as ASM stage 1 under that name (Opus or
CK-Tile stage 2) when no default MXFP4 branch takes it; legacy discards Opus rows for
weights not both preshuffled, with expert_mask or a stage 2 bias, and MXMOE rows under
EP that are BM16, not SiTUv2, or without a layout GEMM2. Interleaved keys outside the
FlyDSL heuristic and the CK-Tile interleaved Swiglu branch, the padded CK default, Opus
rows with padding, `cktile_*` rows with bias or a FlyDSL stage 2, and any other tuned row
are declined back to the legacy code. Interleaved calls that legacy's grouped gfx1250
branch takes (a gfx1250 build, or AITER_USE_GROUPED_GEMM with gfx1250 named by the build
variables; both weight scales set, and not expert_mask with `doweight_stage1`; todo
5.26) are not claimed.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import os

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

from .. import paths
from ..segments.asm import AsmStage1Cktile
from ..segments.ck import CkGemm1, CkGemm2
from ..segments.ck_tile import (
    CktileGemm1,
    CktileGemm2,
    cktile_interleaved_swiglu,
    flydsl_mxfp4,
)
from ..segments.mxfp4 import (
    FlydslGemm1,
    FlydslGemm1Layout,
    FlydslGemm1Quant,
    FlydslGemm2,
    FlydslLayoutGemm2,
    FlydslLayoutGemm2Scatter,
    MxmoeGemm1,
    MxmoeGemm2,
    mxfp4,
    mxfp8,
)
from ..segments.opus import OpusGemm1, OpusGemm1Layout, OpusGemm2
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort, SortAux
from . import local

NAME = "a8w4"
ARCHS = ("gfx950",)
SEGMENTS = (
    Sort(),
    SortAux(),
    QuantA1(),
    AsmStage1Cktile(),
    CkGemm1(),
    CktileGemm1(),
    FlydslGemm1(),
    FlydslGemm1Quant(),
    FlydslGemm1Layout(),
    MxmoeGemm1(),
    OpusGemm1(),
    OpusGemm1Layout(),
    QuantInter(),
    CkGemm2(),
    CktileGemm2(),
    FlydslGemm2(),
    FlydslLayoutGemm2(),
    FlydslLayoutGemm2Scatter(),
    MxmoeGemm2(),
    OpusGemm2(),
)

ACTIVATIONS = (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)
GATE_MODES = (GateMode.SEPARATED, GateMode.INTERLEAVE)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    reason = local.why_not(
        call,
        (QuantType.per_1x32,),
        ARCHS,
        (dtypes.fp4x2,),
        act_dtypes=(dtypes.fp8,),
        act_dtype_arg=True,
        activations=ACTIVATIONS,
        gate_modes=GATE_MODES,
        bias=True,
        act_params=True,
        pads=True,
    )
    if reason is None and _legacy_grouped_gfx1250(call):
        reason = "interleaved gate/up on legacy's grouped gfx1250 path"
    return reason


def _legacy_grouped_gfx1250(call) -> bool:
    """`grouped_gemm_gfx1250_a8w4` runs the call instead of returning None (its checks
    before the launch, for a call `why_not` already accepted)."""
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    if GateMode(call["gate_mode"]) != GateMode.INTERLEAVE:
        return False
    if not grouped._use_grouped_gemm_enabled():
        return False
    if os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1":
        return False
    if call["w1_scale"] is None or call["w2_scale"] is None:
        return False
    if call["expert_mask"] is not None and call["doweight_stage1"]:
        return False
    gfx_env = ";".join(
        str(os.environ.get(k, "")).lower()
        for k in ("GPU_ARCHS", "TARGET_ARCH", "AITER_GPU_ARCHS")
    )
    return (
        get_gfx() in ("gfx1250", "gfx1250-strict")
        or "gfx1250" in gfx_env
        or os.environ.get("AITER_FORCE_GFX1250", "0") in grouped._TRUTHY_ENV
    )


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(
            problem,
            activations=tuple(a.name.lower() for a in ACTIVATIONS),
            gate_modes=GATE_MODES,
            act_params=True,
        )
        and problem.quant_type == QuantType.per_1x32
        and problem.w1 == mxfp4()
        and problem.act_in == mxfp8()
        and (
            problem.gate_mode == GateMode.SEPARATED
            or flydsl_mxfp4(problem)
            or cktile_interleaved_swiglu(problem)
        )
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a8w4 until tests/moe_dispatch/gpu_differential.py passes on an arch.
PATH = paths.Path(
    NAME,
    claims,
    LOCAL.run,
    why_not=why_not,
    dispatch_record=LOCAL.dispatch_record,
    entry="op",
    segments=SEGMENTS,
    supported_archs=frozenset(ARCHS),
    takes=takes,
)
paths.register(PATH)
