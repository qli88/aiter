# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a16w4`: bf16 activations into MXFP4 weights (fp4 in 1x32 groups, e8m0 scales).

Claims `fused_moe_` calls with `QuantType.per_1x32` and fp4x2 weights on gfx942 and
gfx950 whose activations stay bf16 (the `policy.py` choice or `quant_dtype_a`): Swiglu
and SiTUv2 (g1u1) and gate-only Relu2, separated gate/up (Swiglu also interleaved),
outside AITER_ONLINE_TUNE, without `quant_dtype_a2`, global scales, scatter, HERD
routing, deterministic mode or registered presort. Today these resolve to CK-Tile
(Swiglu: interleaved, split-K, or without a kernelName1; Relu2; split-K for SiTUv2 keys
with `doweight_stage1`, whose calls the a16w4 SiTUv2 guard of the executor rejects as
legacy does), the FlyDSL a16w4 heuristic (SiTUv2; Swiglu where CK-Tile is unsafe), or a
tuned FlyDSL a16w4 row or `swiglu_mxfp4_bf16_cktile` row. The CK branch (gfx942 Relu2
and SiTUv2 outside FlyDSL and CK-Tile), interleaved calls outside the CK-Tile
interleaved branch (legacy runs FlyDSL kernels for separated weights there, todo 5.8),
Relu2 with g1u1, bias or `doweight_stage1` (legacy raises), the alias rows outside the
CK-Tile branches (ASM stage 1 + CK stage 2) and any other tuned row are declined back to
the legacy code. Tuned Swiglu 1-stage rows (all FLAT) run 1-stage ASM `fmoe`; the SiTUv2
ones never run: legacy discards them on preshuffled weights, as v2 does, and raises
before the dispatch on unshuffled ones, which v2 declines.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .. import paths
from ..segments.asm import AsmFmoe1Stage
from ..segments.ck_tile import CktileGemm1, CktileGemm2, cktile_interleaved_swiglu
from ..segments.mxfp4 import FlydslGemm1, FlydslGemm2, a16w4, mxfp4
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort, SortFlat
from ..shapes import get_inter_dim
from . import local

NAME = "a16w4"
ARCHS = ("gfx942", "gfx950")
SEGMENTS = (
    Sort(),
    SortFlat(),
    QuantA1(),
    CktileGemm1(),
    FlydslGemm1(),
    QuantInter(),
    CktileGemm2(),
    FlydslGemm2(),
    AsmFmoe1Stage(),
)

ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Situv2, ActivationType.Relu2)
GATE_MODES = (GateMode.SEPARATED, GateMode.INTERLEAVE)


def why_not(call):
    """Why this path does not claim the `fused_moe_` call, or None if it does."""
    reason = local.why_not(
        call,
        (QuantType.per_1x32,),
        ARCHS,
        (dtypes.fp4x2,),
        act_dtypes=(dtypes.bf16,),
        act_dtype_arg=True,
        activations=ACTIVATIONS,
        gate_modes=GATE_MODES,
        bias=True,
        act_params=True,
        pads=True,
        g1u0=True,
    )
    if reason is not None:
        return reason
    # Relu2 is gate-only, the others g1u1; legacy raises for Relu2 with g1u1, bias or
    # doweight_stage1.
    relu2 = ActivationType(call["activation"]) == ActivationType.Relu2
    w1, w2 = call["w1"], call["w2"]
    if (w1.shape[1] == 2 * get_inter_dim(w1.shape, w2.shape)[2]) == relu2:
        if relu2:
            return f"Relu2 with g1u1 w1 {tuple(w1.shape)}"
        return f"w1 {tuple(w1.shape)} is not g1u1"
    if relu2 and (
        call["doweight_stage1"] or any(call[n] is not None for n in ("bias1", "bias2"))
    ):
        return "Relu2 with bias or doweight_stage1"
    return None


def claims(call) -> bool:
    return why_not(call) is None


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims`."""
    relu2 = problem.activation.activation == ActivationType.Relu2
    return (
        local.key_args_supported(
            problem,
            activations=tuple(a.name.lower() for a in ACTIVATIONS),
            gate_modes=GATE_MODES,
            act_params=True,
            g1u0=True,
        )
        and problem.quant_type == QuantType.per_1x32
        and problem.w1 == mxfp4()
        and problem.act_in == a16w4()
        # Relu2 is gate-only; legacy raises for g1u1, bias or doweight_stage1.
        and problem.layout.g1u1 != relu2
        and not (relu2 and {"bias1", "bias2", "doweight_stage1"} & problem.features)
        and (
            problem.gate_mode == GateMode.SEPARATED
            or cktile_interleaved_swiglu(problem)
        )
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes)

# AITER_MOEV2=a16w4 until tests/moe_dispatch/gpu_differential.py passes on an arch.
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
