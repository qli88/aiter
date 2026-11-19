# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_pt`: 8-bit activations and weights, per-tensor scales.

Claims `fused_moe_` calls with g1u1 per-tensor fp8 (the arch's fp8) or int8 weights (int8
without doweight_stage1, which no ASM stage 1 runs) on gfx942/gfx950, Silu or Gelu,
separated gate/up, no bias, clamping swiglu_limit (None, 0 and inf do not clamp), beta
or linear_beta, padding, scatter, caller-chosen activation dtypes, HERD routing,
deterministic mode or registered presort, outside AITER_ONLINE_TUNE. Today fp8 resolves
to CK 2-stage (default and most tuned rows), ASM stage 1 + CK stage 2, or the FlyDSL
gfx942 whole graph; int8 to ASM stage 1 + CK stage 2 (default and tuned) or CK 2-stage
(tuned). Any other tuned row is declined back to the legacy code, and so is a call whose
second lookup after the sort raises or is declined.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import QuantType, dtypes

from .. import paths
from ..segments.asm import AsmStage1
from ..segments.ck import CkGemm1, CkGemm2, per_tensor_fp8, per_tensor_int8
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from ..segments.whole_graph import WholeGraph
from . import local

NAME = "a8w8_pt"
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
    return local.why_not(call, (QuantType.per_Tensor,), ARCHS, (dtypes.fp8, dtypes.i8))


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem)
        and problem.quant_type == QuantType.per_Tensor
        and problem.w1 in (per_tensor_fp8(), per_tensor_int8())
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)


# Not graduated: runs only when enabled with AITER_MOEV2=a8w8_pt (or `on`).
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
