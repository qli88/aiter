# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_ptok`: 8-bit activations, per-token scales.

Claims `fused_moe_` calls with g1u1 per-token weights on gfx942/gfx950: fp8 (the arch's
fp8), int8, or int4 packed in `uint32` (with fp8 activations); Silu or Gelu, separated
gate/up, no bias, padding, scatter or caller-chosen activation dtypes, outside
AITER_ONLINE_TUNE. Today fp8 resolves to CK 2-stage (default and most tuned rows), ASM
stage 1 + CK stage 2, or the FlyDSL gfx942 whole graph; int8 to ASM stage 1 + CK stage 2
(default) or CK 2-stage (tuned); int4 to CK 2-stage. The default 1-stage ASM `fmoe`
regions and 1-stage tuned rows are declined back to the legacy code, as is any other
tuned row.
"""

import torch

from aiter import QuantType, dtypes

from .. import paths
from ..segments.asm import AsmStage1
from ..segments.ck import (
    CkGemm1,
    CkGemm2,
    per_token_fp8,
    per_token_int4,
    per_token_int8,
)
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


def claims(call) -> bool:
    return local.claims(
        call, (QuantType.per_Token,), ARCHS, (dtypes.fp8, dtypes.i8, torch.uint32)
    )


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem)
        and problem.quant_type == QuantType.per_Token
        and problem.w1 in (per_token_fp8(), per_token_int8(), per_token_int4())
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a8w8_ptok until tests/moe_dispatch/gpu_differential.py passes on an arch.
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
