# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_ptok`: 8-bit activations, per-token scales.

Claims `fused_moe_` calls with per-token weights on gfx942/gfx950: fp8 (the arch's fp8,
g1u1), int8 (g1u1 or g1u0, without doweight_stage1, which no ASM stage 1 runs), or int4
packed in `uint32` (g1u1, with fp8 activations); Silu or Gelu, separated gate/up, no
bias, padding, scatter, caller-chosen activation dtypes, clamping swiglu_limit (None, 0
and inf do not clamp), beta or linear_beta, HERD routing, deterministic mode or
registered presort, outside AITER_ONLINE_TUNE. Today fp8 resolves to 1-stage ASM `fmoe`
or CK 2-stage (default; most tuned rows), ASM stage 1 + CK stage 2, or the FlyDSL gfx942
whole graph; g1u1 int8 to 1-stage ASM `fmoe` or ASM stage 1 + CK stage 2 (default) or CK
2-stage (tuned); int4 to CK 2-stage; g1u0 int8 to 1-stage ASM `fmoe_int8_g1u0` (gfx942,
more than 32 tokens). Tuned 1-stage rows run 1-stage ASM `fmoe` (FLAT too). Other g1u0
int8 calls (legacy's ASM stage 1 + CK stage 2, which has g1u1 kernels only, todo 5.3),
any other tuned row and a 2-stage call whose second lookup after the sort raises or is
declined are declined back to the legacy code.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import torch

from aiter import QuantType, dtypes

from .. import paths
from ..segments.asm import AsmFmoe1Stage, AsmStage1
from ..segments.ck import (
    CkGemm1,
    CkGemm2,
    per_token_fp8,
    per_token_int4,
    per_token_int8,
)
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort, SortFlat
from ..segments.whole_graph import WholeGraph
from . import local

NAME = "a8w8_ptok"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (
    Sort(),
    SortFlat(),
    QuantA1(),
    CkGemm1(),
    AsmStage1(),
    QuantInter(),
    CkGemm2(),
    AsmFmoe1Stage(),
    WholeGraph(),
)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    return local.why_not(
        call,
        (QuantType.per_Token,),
        ARCHS,
        (dtypes.fp8, dtypes.i8, torch.uint32),
        g1u0=call["w1"].dtype == dtypes.i8,
    )


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem, g1u0=problem.w1 == per_token_int8())
        and problem.quant_type == QuantType.per_Token
        and problem.w1 in (per_token_fp8(), per_token_int8(), per_token_int4())
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
