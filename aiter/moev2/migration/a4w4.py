# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a4w4`: MXFP4 activations and weights (fp4 in 1x32 groups, e8m0 scales).

Claims `fused_moe_` calls with `QuantType.per_1x32` and g1u1 fp4x2 weights on gfx950,
Silu or Gelu, separated gate/up, no bias, padding, scatter, caller-chosen activation
dtypes, clamping swiglu_limit (None, 0 and inf do not clamp), beta or linear_beta, HERD
routing, deterministic mode or registered presort, outside AITER_ONLINE_TUNE. Calls with
num_local_tokens, a1_scale or activation scales find no plan and are declined. Today
these resolve to the FlyDSL heuristic (default on preshuffled weights), CK 2-stage
(default otherwise), or a tuned FlyDSL, FlyDSL layout-v2 or MXMOE row; the CK-Tile
split-K default, 1-stage tuned rows and tuned CK rows on preshuffled weights (which the
legacy code overrides with the FlyDSL heuristic) are declined back to the legacy code,
as is any other tuned row.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import QuantType, dtypes

from .. import paths
from ..segments.ck import CkGemm1, CkGemm2
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
from ..segments.sort import Sort, SortAux
from . import local

NAME = "a4w4"
ARCHS = ("gfx950",)
SEGMENTS = (
    Sort(),
    SortAux(),
    QuantA1(),
    CkGemm1(),
    FlydslGemm1(),
    FlydslGemm1Quant(),
    FlydslGemm1Layout(),
    MxmoeGemm1(),
    QuantInter(),
    CkGemm2(),
    FlydslGemm2(),
    FlydslLayoutGemm2(),
    FlydslLayoutGemm2Scatter(),
    MxmoeGemm2(),
)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    return local.why_not(call, (QuantType.per_1x32,), ARCHS, (dtypes.fp4x2,))


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem)
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
    why_not=why_not,
    dispatch_record=LOCAL.dispatch_record,
    entry="op",
    segments=SEGMENTS,
    supported_archs=frozenset(ARCHS),
    takes=takes,
)
paths.register(PATH)
