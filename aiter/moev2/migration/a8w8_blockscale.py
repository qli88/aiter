# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_blockscale`: fp8 activations in 1x128 groups, fp8 weights in 128x128 blocks.

Claims `fused_moe_` calls with `QuantType.per_1x128` or `per_128x128` (which the legacy
code remaps to `per_1x128`) and g1u1 fp8 weights (the arch's fp8) on gfx942/gfx950, Silu
or Gelu, separated gate/up, no bias, padding, scatter, caller-chosen activation dtypes,
clamping swiglu_limit (None, 0 and inf do not clamp), beta or linear_beta, HERD routing,
deterministic mode or registered presort, outside AITER_ONLINE_TUNE. Today these resolve
to 1-stage ASM `fmoe_fp8_blockscale_g1u1` (default, in its region), CK 2-stage (default
elsewhere, with the 1x128 block_m and split-K; most tuned rows), ASM stage 1 + CK stage
2, or a tuned 1-stage row (FLAT too); any other tuned row is declined back to the legacy
code, as is a 2-stage call whose second lookup after the sort raises or is declined.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import QuantType, dtypes

from .. import paths
from ..segments.asm import AsmFmoe1Stage, AsmStage1Blockscale
from ..segments.ck import CkGemm1, CkGemm2, blockscale_fp8
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort, SortFlat
from . import local

NAME = "a8w8_blockscale"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (
    Sort(),
    SortFlat(),
    QuantA1(),
    CkGemm1(),
    AsmStage1Blockscale(),
    QuantInter(),
    CkGemm2(),
    AsmFmoe1Stage(),
)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    return local.why_not(
        call, (QuantType.per_1x128, QuantType.per_128x128), ARCHS, (dtypes.fp8,)
    )


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem)
        and problem.quant_type == QuantType.per_1x128
        and problem.w1 == blockscale_fp8()
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a8w8_blockscale until tests/moe_dispatch/gpu_differential.py passes on an
# arch.
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
