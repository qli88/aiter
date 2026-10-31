# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_blockscale`: fp8 activations in 1x128 groups, fp8 weights in 128x128 blocks.

Claims `fused_moe_` calls with `QuantType.per_1x128` or `per_128x128` (which the legacy
code remaps to `per_1x128`) and g1u1 fp8 weights (the arch's fp8) on gfx942/gfx950, Silu
or Gelu, separated gate/up, no bias, padding, scatter or caller-chosen activation dtypes,
outside AITER_ONLINE_TUNE. Today these resolve to CK 2-stage (default, with the 1x128
block_m and split-K, and most tuned rows) or ASM stage 1 + CK stage 2; the default 1-stage
ASM `fmoe_fp8_blockscale_g1u1` region and 1-stage tuned rows are declined back to the
legacy code, as is any other tuned row.
"""

from aiter import QuantType, dtypes

from .. import paths
from ..segments.asm import AsmStage1Blockscale
from ..segments.ck import CkGemm1, CkGemm2, blockscale_fp8
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from . import local

NAME = "a8w8_blockscale"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (
    Sort(),
    QuantA1(),
    CkGemm1(),
    AsmStage1Blockscale(),
    QuantInter(),
    CkGemm2(),
)


def claims(call) -> bool:
    return local.claims(
        call, (QuantType.per_1x128, QuantType.per_128x128), ARCHS, (dtypes.fp8,)
    )


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
    dispatch_record=LOCAL.dispatch_record,
    entry="op",
    segments=SEGMENTS,
    takes=takes,
)
paths.register(PATH)
