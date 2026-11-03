# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mxfp8`: mxfp8 activations into mxfp8 weights (1x32 groups, e8m0 scales).

Claims `fused_moe_` calls with `QuantType.per_1x32` and g1u1 fp8 weights on gfx950
whose activations are fp8 (the `policy.py` choice or `quant_dtype_a`), Silu or Swiglu,
separated or interleaved gate/up, outside AITER_ONLINE_TUNE, without `quant_dtype_a2`
or scatter. Today these resolve to the FlyDSL heuristic (gate/up interleaved, or
separated with w1 tagged `gate_up_interleave=False`), CK 2-stage otherwise (which
ignores the padding), or a tuned FlyDSL row (FlyDSL or layout-v2 stage 2). Interleaved
keys outside the FlyDSL heuristic, the padded CK default and any other tuned row are
declined back to the legacy code.
"""

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .. import paths
from ..segments.ck import CkGemm1, CkGemm2
from ..segments.ck_tile import flydsl_mxfp4
from ..segments.mxfp4 import (
    FlydslGemm1,
    FlydslGemm1Layout,
    FlydslGemm1Quant,
    FlydslGemm2,
    FlydslLayoutGemm2,
    FlydslLayoutGemm2Scatter,
    mxfp8,
)
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort, SortAux
from . import local

NAME = "mxfp8"
ARCHS = ("gfx950",)
SEGMENTS = (
    Sort(),
    SortAux(),
    QuantA1(),
    CkGemm1(),
    FlydslGemm1(),
    FlydslGemm1Quant(),
    FlydslGemm1Layout(),
    QuantInter(),
    CkGemm2(),
    FlydslGemm2(),
    FlydslLayoutGemm2(),
    FlydslLayoutGemm2Scatter(),
)

ACTIVATIONS = (ActivationType.Silu, ActivationType.Swiglu)
GATE_MODES = (GateMode.SEPARATED, GateMode.INTERLEAVE)


def claims(call) -> bool:
    return local.claims(
        call,
        (QuantType.per_1x32,),
        ARCHS,
        (dtypes.fp8,),
        act_dtypes=(dtypes.fp8,),
        act_dtype_arg=True,
        activations=ACTIVATIONS,
        gate_modes=GATE_MODES,
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
            gate_modes=GATE_MODES,
            act_params=True,
        )
        and problem.quant_type == QuantType.per_1x32
        and problem.w1 == mxfp8()
        and problem.act_in == mxfp8()
        and (problem.gate_mode == GateMode.SEPARATED or flydsl_mxfp4(problem))
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=mxfp8 until tests/moe_dispatch/gpu_differential.py passes on an arch.
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
