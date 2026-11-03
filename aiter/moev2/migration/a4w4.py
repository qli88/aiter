# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a4w4`: MXFP4 activations and weights (fp4 in 1x32 groups, e8m0 scales).

Claims `fused_moe_` calls with `QuantType.per_1x32` and g1u1 fp4x2 weights on gfx950
whose activations are fp4 (the `policy.py` choice or `quant_dtype_a`), Silu, Gelu,
Swiglu or SiTUv2, separated gate/up, outside AITER_ONLINE_TUNE, without `quant_dtype_a2`
or scatter. Today these resolve to the FlyDSL heuristic (default on preshuffled
weights), the MXMOE heuristic (SiTUv2), CK-Tile (split-K on preshuffled weights; Swiglu
without FlyDSL), CK 2-stage (Silu/Gelu default otherwise), or a tuned FlyDSL, FlyDSL
layout-v2, MXMOE or CK row (with a CK-Tile stage 2 or not). The SiTUv2 default without
FlyDSL or MXMOE, tuned CK rows on preshuffled weights (which the legacy code
overrides with its defaults), bias or padding on kernels that take none, and any other
tuned row are declined back to the legacy code. Tuned 1-stage rows (all FLAT) run
1-stage ASM `fmoe`.
"""

from aiter import ActivationType, QuantType, dtypes

from .. import paths
from ..segments.asm import AsmFmoe1Stage
from ..segments.ck import CkGemm1, CkGemm2
from ..segments.ck_tile import CktileGemm1, CktileGemm2
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
)
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort, SortAux, SortFlat
from . import local

NAME = "a4w4"
ARCHS = ("gfx950",)
SEGMENTS = (
    Sort(),
    SortAux(),
    SortFlat(),
    QuantA1(),
    CkGemm1(),
    CktileGemm1(),
    FlydslGemm1(),
    FlydslGemm1Quant(),
    FlydslGemm1Layout(),
    MxmoeGemm1(),
    QuantInter(),
    CkGemm2(),
    CktileGemm2(),
    FlydslGemm2(),
    FlydslLayoutGemm2(),
    FlydslLayoutGemm2Scatter(),
    MxmoeGemm2(),
    AsmFmoe1Stage(),
)


ACTIVATIONS = (
    ActivationType.Silu,
    ActivationType.Gelu,
    ActivationType.Swiglu,
    ActivationType.Situv2,
)


def claims(call) -> bool:
    return local.claims(
        call,
        (QuantType.per_1x32,),
        ARCHS,
        (dtypes.fp4x2,),
        act_dtypes=(dtypes.fp4x2,),
        act_dtype_arg=True,
        activations=ACTIVATIONS,
        bias=True,
        act_params=True,
        pads=True,
    )


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(
            problem,
            ARCHS,
            activations=tuple(a.name.lower() for a in ACTIVATIONS),
            act_params=True,
        )
        and problem.quant_type == QuantType.per_1x32
        and problem.w1 == mxfp4()
        and problem.act_in == mxfp4()
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a4w4 until tests/moe_dispatch/gpu_differential.py passes on an arch.
PATH = paths.Path(
    NAME,
    claims,
    LOCAL.run,
    dispatch_record=LOCAL.dispatch_record,
    entry="op",
    resolve=LOCAL.resolve_call,
    segments=SEGMENTS,
    takes=takes,
)
paths.register(PATH)
