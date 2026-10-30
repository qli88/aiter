# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `a8w8_pt`: fp8 activations and weights, per-tensor scales.

Claims `fused_moe_` calls with per-tensor fp8 weights (the arch's fp8) on gfx942/gfx950,
Silu or Gelu, separated gate/up, no bias, padding, scatter or caller-chosen activation
dtypes, outside AITER_ONLINE_TUNE. Today these resolve to CK 2-stage (default and most
tuned rows), ASM stage 1 + CK stage 2, or the FlyDSL gfx942 whole graph; any other tuned
row is declined back to the legacy code. int8 per-tensor stays on the legacy code.
"""

import functools
from dataclasses import replace

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import overrides, resolver, tuned
from ..execute import executor
from ..policy import quant_remap
from ..segments.asm import AsmStage1
from ..segments.ck import CkGemm1, CkGemm2
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from ..segments.whole_graph import WholeGraph
from ..shapes import get_inter_dim, get_padded_M

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
# fused_moe_2stages looks the key up again without these, after sorting, for its log
# lines only.
_RELOOKUP_DROPS = frozenset({"act_scales", "num_local_tokens", "scatter"})


def claims(call) -> bool:
    """Cheap checks on the `fused_moe_` arguments; `run` may still decline."""
    if overrides.online_tune():
        return False
    if QuantType(call["quant_type"]) != QuantType.per_Tensor:
        return False
    quant_type_a = call["quant_type_a"]
    if quant_type_a is not None and QuantType(quant_type_a) != QuantType.per_Tensor:
        return False
    if any(
        call[name] is not None
        for name in (
            "quant_dtype_a",
            "quant_dtype_a2",
            "ep_source_token_map",
            "bias1",
            "bias2",
            "swiglu_limit",
            "beta",
            "linear_beta",
        )
    ):
        return False
    if call["hidden_pad"] or call["intermediate_pad"]:
        return False
    if GateMode(call["gate_mode"]) != GateMode.SEPARATED:
        return False
    if ActivationType(call["activation"]) not in (
        ActivationType.Silu,
        ActivationType.Gelu,
    ):
        return False
    if get_gfx() not in ARCHS or get_gfx_runtime() not in ARCHS:
        return False
    w1, w2 = call["w1"], call["w2"]
    if (
        w1.dtype != dtypes.fp8
        or w2.dtype != dtypes.fp8
        or w1.dim() != 3
        or w2.dim() != 3
    ):
        return False
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if w1.shape[1] not in (inter_dim, 2 * inter_dim):
        return False
    hidden_states = call["hidden_states"]
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    return dtype in (dtypes.bf16, dtypes.fp16) and call["topk_ids"].dim() == 2


@tuned.cached(maxsize=2048)
def _resolve(problem, relookup):
    resolution = resolver.resolve(problem, SEGMENTS)
    resolution.emit()
    return resolution


@tuned.cached(maxsize=2048)
def _relookup(problem, token):
    """The second lookup, keyed on the `hidden_states` rows rather than `topk_ids`."""
    shape = replace(problem.shape, m=token)
    return _resolve(
        replace(problem.with_features(remove=_RELOOKUP_DROPS), inter=None, shape=shape),
        True,
    )


@tuned.cached(maxsize=2048)
def _dispatch(args, kwargs_items, arch, cu_num):
    """`(resolution, problem)` for a key, or None when declined.

    `problem` is None when no second lookup follows (whole-graph plans).
    """
    problem = api.problem_from_key(args, dict(kwargs_items), arch, cu_num)
    primary = _resolve(problem, False)
    if primary.plan is None:
        return None
    if primary.metadata.full_impl is not None:
        return primary, None
    return primary, problem


def run(call):
    c = executor.prologue(call, quant_remap)
    args, kwargs, q_dtype_a = api.dispatch_key(c)
    dispatched = _dispatch(args, tuple(kwargs.items()), get_gfx_runtime(), get_cu_num())
    if dispatched is None:
        return paths.DECLINED
    resolution, problem = dispatched
    after_sort = None
    if problem is not None:
        token = get_padded_M(c["hidden_states"].shape[0])
        after_sort = functools.partial(_relookup, problem, token)
    return executor.run_two_stage(
        c, resolution.metadata, resolution.plan, q_dtype_a, after_sort=after_sort
    )


def dispatch_record(args, kwargs):
    """The v2 `MOEMetadata` for a legacy key (logs emitted), or None if not taken."""
    problem = api.problem_from_key(args, kwargs)
    if problem.activation.swiglu_limit is not None or problem.inter is not None:
        return None
    if (problem.activation.situ_beta, problem.activation.situ_linear_beta) != (
        1.0,
        1.0,
    ):
        return None
    resolution = resolver.resolve(problem, SEGMENTS)
    resolution.emit()
    return None if resolution.plan is None else resolution.metadata


# Not graduated: runs only when enabled with AITER_MOEV2=a8w8_pt (or `on`).
PATH = paths.Path(
    NAME,
    claims,
    run,
    dispatch_record=dispatch_record,
    entry="op",
    segments=SEGMENTS,
)
paths.register(PATH)
