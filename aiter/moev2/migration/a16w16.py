# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a16w16`: bf16/fp16 activations and weights, unquantized (`QuantType.No`).

Claims `fused_moe_` calls with g1u1 bf16/fp16 weights on gfx942/gfx950, Silu or Gelu,
separated gate/up, no bias, padding, scatter or caller-chosen activation dtypes, outside
AITER_ONLINE_TUNE. Today these resolve to CK 2-stage (default and tuned rows) or the
FlyDSL gfx942 whole graph (tuned rows); any other tuned row is declined back to the
legacy code. g1u0 calls are not claimed: legacy runs them on the 1-stage ASM `fmoe`
(gfx942, Silu, fewer than 256 tokens) or on CK 2-stage, which is wrong for g1u0.
"""

from aiter import QuantType, dtypes

from .. import paths
from ..segments.ck import CkGemm1, CkGemm2
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from ..segments.whole_graph import WholeGraph
from . import local

NAME = "a16w16"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (Sort(), QuantA1(), CkGemm1(), QuantInter(), CkGemm2(), WholeGraph())


def claims(call) -> bool:
    return local.claims(call, QuantType.No, ARCHS, (dtypes.bf16, dtypes.fp16))


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem)
        and problem.quant_type == QuantType.No
        and problem.w1.dtype in (dtypes.bf16, dtypes.fp16)
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a16w16 until tests/moe_dispatch/gpu_differential.py passes on an arch.
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
