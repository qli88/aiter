# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_ptok`: fp8 activations and weights, per-token scales.

Claims `fused_moe_` calls with g1u1 per-token fp8 weights (the arch's fp8) on
gfx942/gfx950, Silu or Gelu, separated gate/up, no bias, padding, scatter, caller-chosen
activation dtypes, clamping swiglu_limit (None, 0 and inf do not clamp), beta or
linear_beta, HERD routing, deterministic mode or registered presort, outside
AITER_ONLINE_TUNE. Today these resolve to CK 2-stage (default and most tuned rows), ASM
stage 1 + CK stage 2, or the FlyDSL gfx942 whole graph; the default 1-stage ASM `fmoe`
region and 1-stage tuned rows are declined back to the legacy code, as is any other tuned
row and a call whose second lookup after the sort raises or is declined. int8 and int4
weights stay on the legacy code.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import QuantType, dtypes

from .. import paths
from ..segments.asm import AsmStage1
from ..segments.ck import CkGemm1, CkGemm2, per_token_fp8
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from ..segments.whole_graph import WholeGraph
from . import local

NAME = "a8w8_ptok"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (
    Sort(),
    QuantA1(),
    CkGemm1(),
    AsmStage1(),
    QuantInter(),
    CkGemm2(),
    WholeGraph(),
)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    return local.why_not(call, (QuantType.per_Token,), ARCHS, (dtypes.fp8,))


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem)
        and problem.quant_type == QuantType.per_Token
        and problem.w1 == per_token_fp8()
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a8w8_ptok until tests/moe_dispatch/gpu_differential.py passes on an arch.
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
