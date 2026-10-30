# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch and execution shared by the local paths (one GPU, the `fused_moe_` entry).

A local path supplies its segments and `takes(problem)`, the problems it runs (its
segments may accept more: a CK segment serves several quant formats, each in its own
path). `LocalDispatch` resolves each legacy dispatch key once, as the cached legacy
`get_2stage_cfgs` does, and runs the plan with `executor.run_two_stage`.
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
from ..shapes import get_inter_dim, get_padded_M

# fused_moe_2stages looks the key up again without these, after sorting, for its log
# lines only.
_RELOOKUP_DROPS = frozenset({"act_scales", "num_local_tokens", "scatter"})

# `fused_moe_` arguments no local path supports yet.
_UNSUPPORTED_ARGS = (
    "quant_dtype_a",
    "quant_dtype_a2",
    "ep_source_token_map",
    "bias1",
    "bias2",
    "swiglu_limit",
    "beta",
    "linear_beta",
)


def claims(call, quant_type, archs, weight_dtypes) -> bool:
    """Cheap checks on the `fused_moe_` arguments; `run` may still decline.

    `quant_type` (and a `quant_type_a`, if given), Silu or Gelu, separated gate/up, no
    bias, padding or caller-chosen activation dtypes, outside AITER_ONLINE_TUNE, on
    `archs`, with 3D `w1`/`w2` of `weight_dtypes` and `w1` holding gate and up (g1u1).
    """
    if overrides.online_tune():
        return False
    if QuantType(call["quant_type"]) != quant_type:
        return False
    quant_type_a = call["quant_type_a"]
    if quant_type_a is not None and QuantType(quant_type_a) != quant_type:
        return False
    if any(call[name] is not None for name in _UNSUPPORTED_ARGS):
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
    if get_gfx() not in archs or get_gfx_runtime() not in archs:
        return False
    w1, w2 = call["w1"], call["w2"]
    if (
        w1.dtype not in weight_dtypes
        or w2.dtype not in weight_dtypes
        or w1.dim() != 3
        or w2.dim() != 3
    ):
        return False
    # g1u0 stays on legacy: CK stage 1 always reads w1 as gate and up, and the 1-stage
    # ASM `fmoe` that legacy runs for some g1u0 keys is not ported.
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if w1.shape[1] != 2 * inter_dim:
        return False
    hidden_states = call["hidden_states"]
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    return dtype in (dtypes.bf16, dtypes.fp16) and call["topk_ids"].dim() == 2


def key_args_supported(problem) -> bool:
    """False for keys with arguments or a layout no local path supports yet (see `claims`)."""
    activation = problem.activation
    return (
        problem.layout.g1u1
        and activation.swiglu_limit is None
        and problem.inter is None
        and (activation.situ_beta, activation.situ_linear_beta) == (1.0, 1.0)
    )


class LocalDispatch:
    def __init__(self, segments, takes):
        self.segments = tuple(segments)
        self.takes = takes
        self.resolve = tuned.cached(maxsize=2048)(self._resolve)
        self.relookup = tuned.cached(maxsize=2048)(self._relookup)
        self.dispatch = tuned.cached(maxsize=2048)(self._dispatch)

    def _resolve(self, problem, relookup):
        resolution = resolver.resolve(problem, self.segments)
        resolution.emit()
        return resolution

    def _relookup(self, problem, token):
        """The second lookup, keyed on the `hidden_states` rows rather than `topk_ids`."""
        shape = replace(problem.shape, m=token)
        problem = problem.with_features(remove=_RELOOKUP_DROPS)
        return self.resolve(replace(problem, inter=None, shape=shape), True)

    def _dispatch(self, args, kwargs_items, arch, cu_num):
        """`(resolution, problem)` for a key, or None when not taken or declined.

        `problem` is None when no second lookup follows (whole-graph plans).
        """
        problem = api.problem_from_key(args, dict(kwargs_items), arch, cu_num)
        if not self.takes(problem):
            return None
        primary = self.resolve(problem, False)
        if primary.plan is None:
            return None
        if primary.metadata.full_impl is not None:
            return primary, None
        return primary, problem

    def run(self, call):
        c = executor.prologue(call, quant_remap)
        args, kwargs, q_dtype_a = api.dispatch_key(c)
        dispatched = self.dispatch(
            args, tuple(kwargs.items()), get_gfx_runtime(), get_cu_num()
        )
        if dispatched is None:
            return paths.DECLINED
        resolution, problem = dispatched
        after_sort = None
        if problem is not None:
            token = get_padded_M(c["hidden_states"].shape[0])
            after_sort = functools.partial(self.relookup, problem, token)
        return executor.run_two_stage(
            c, resolution.metadata, resolution.plan, q_dtype_a, after_sort=after_sort
        )

    def dispatch_record(self, args, kwargs):
        """The v2 `MOEMetadata` for a legacy key (logs emitted), or None if not taken."""
        problem = api.problem_from_key(args, kwargs)
        if not self.takes(problem):
            return None
        resolution = resolver.resolve(problem, self.segments)
        resolution.emit()
        return None if resolution.plan is None else resolution.metadata
