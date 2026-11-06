# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_pt`: 8-bit activations and weights, per-tensor scales.

Claims `fused_moe_` calls with g1u1 per-tensor fp8 (the arch's fp8) or int8 weights on
gfx942/gfx950, Silu, Gelu, Swiglu or GeluTanh, separated gate/up, no bias, padding,
scatter or caller-chosen activation dtypes, outside AITER_ONLINE_TUNE. Today fp8
resolves to CK 2-stage (default and most tuned rows), ASM stage 1 + CK stage 2, or the
FlyDSL gfx942 whole graph; int8 to ASM stage 1 + CK stage 2 (default and tuned) or CK
2-stage (tuned). Any other tuned row is declined back to the legacy code.
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


def claims(call) -> bool:
    return local.claims(call, (QuantType.per_Tensor,), ARCHS, (dtypes.fp8, dtypes.i8))


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(problem, ARCHS)
        and problem.quant_type == QuantType.per_Tensor
        and problem.w1 in (per_tensor_fp8(), per_tensor_int8())
        and problem.act_in == problem.w1
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)


# Not graduated: runs only when enabled with AITER_MOEV2=a8w8_pt (or `on`).
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
