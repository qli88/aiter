# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a16wi4`: bf16 activations into int4 weights (`torch.int4` in 1x32 groups, bf16
scales).

Claims `fused_moe_` calls with `QuantType.per_1x32` and `dtypes.i4x2` weights on gfx942
and gfx950 whose activations stay bf16: Silu and SiTUv2, g1u1, separated gate/up,
outside AITER_ONLINE_TUNE, without `quant_dtype_a2`, global scales, scatter, HERD
routing, deterministic mode or registered presort. These resolve to the legacy untuned
a16wi4 FlyDSL fallback or a tuned FlyDSL `abf16_wint4` row; neither stage quantizes.
Calls with `doweight_stage1` (no a16wi4 segment applies the routing weights in stage
1), 1-stage tuned rows legacy keeps (the ones it discards, v2 discards too) and any
other tuned row are declined back to the legacy code. Interleaved gate/up is not
claimed (legacy runs the same kernels, todo 5.8).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .. import paths
from ..segments.mxfp4 import FlydslGemm1, FlydslGemm2, a16w4, wi4
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from . import local

NAME = "a16wi4"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (Sort(), QuantA1(), FlydslGemm1(), QuantInter(), FlydslGemm2())

ACTIVATIONS = (ActivationType.Silu, ActivationType.Situv2)
GATE_MODES = (GateMode.SEPARATED,)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    return local.why_not(
        call,
        (QuantType.per_1x32,),
        ARCHS,
        (dtypes.i4x2,),
        act_dtypes=(dtypes.bf16,),
        act_dtype_arg=True,
        activations=ACTIVATIONS,
        gate_modes=GATE_MODES,
        bias=True,
        act_params=True,
        pads=True,
    )


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    return (
        local.key_args_supported(
            problem,
            ARCHS,
            activations=tuple(a.name.lower() for a in ACTIVATIONS),
            gate_modes=GATE_MODES,
            act_params=True,
            bias=True,
            pads=True,
        )
        and problem.quant_type == QuantType.per_1x32
        and problem.w1 == wi4()
        and problem.act_in == a16w4()
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a16wi4 until tests/moe_dispatch/gpu_differential.py passes on an arch.
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
