# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch and execution shared by the local paths (one GPU, the `fused_moe_` entry).

A local path supplies its segments and `takes(problem)`, the problems it runs (its
segments may accept more: a CK segment serves several quant formats, each in its own
path). `LocalDispatch` resolves each legacy dispatch key once, as the cached legacy
`get_2stage_cfgs` does, and runs the plan with `executor.run_two_stage`.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import functools
import math
from dataclasses import replace

from aiter import ActivationType, QuantType, dtypes, presorted_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import overrides, resolver, tuned
from ..execute import executor
from ..policy import quant_remap
from ..shapes import get_inter_dim, get_padded_M

# After sorting, fused_moe_2stages looks the key up again without these, on the
# hidden_states rows. _fused_moe_impl's _metadata_transform then replaces the result
# with the first lookup's, so the second lookup only logs, or raises.
_RELOOKUP_DROPS = frozenset({"act_scales", "num_local_tokens", "scatter"})
_UNRESOLVED = object()

# `fused_moe_` arguments no local path supports yet.
_UNSUPPORTED_ARGS = (
    "quant_dtype_a",
    "quant_dtype_a2",
    "ep_source_token_map",
    "bias1",
    "bias2",
    "beta",
    "linear_beta",
    "w1_global_scale",
    "w2_global_scale",
)


def clamps(swiglu_limit) -> bool:
    """None, 0 (vLLM's spelling) and inf are no clamp, as `fused_moe_` reads them for
    Silu. Legacy ignores a finite limit with Gelu; the local paths decline such a call.
    """
    return swiglu_limit is not None and swiglu_limit != 0 and swiglu_limit != math.inf


def why_not(call, quant_type, archs, weight_dtypes):
    """Why a local path does not claim the `fused_moe_` call, or None if it does.

    Cheap checks on the arguments; `run` may still decline a claimed call. Claimed:
    `quant_type` (and a `quant_type_a`, if given), Silu or Gelu, separated gate/up, no
    bias, clamping swiglu_limit, beta or linear_beta, global scales, padding, scatter,
    caller-chosen activation dtypes, HERD routing, deterministic mode or registered
    presort, outside AITER_ONLINE_TUNE, on `archs`, with 3D `w1`/`w2` of
    `weight_dtypes` and `w1` holding gate and up (g1u1).
    """
    if overrides.online_tune():
        return "AITER_ONLINE_TUNE=1"
    if call["herd_topk"]:
        return "HERD routing (herd_topk)"
    if call["deterministic"]:
        return "deterministic=True"
    # The legacy code consumes a registered presort on every call; the local paths do not.
    if getattr(presorted_moe._PRESORTED_MOE, "aux", None) is not None:
        return "a registered presort"
    if QuantType(call["quant_type"]) != quant_type:
        return f"quant_type {QuantType(call['quant_type'])}"
    quant_type_a = call["quant_type_a"]
    if quant_type_a is not None and QuantType(quant_type_a) != quant_type:
        return f"quant_type_a {QuantType(quant_type_a)}"
    for name in _UNSUPPORTED_ARGS:
        if call[name] is not None:
            return f"{name} is set"
    if clamps(call["swiglu_limit"]):
        return f"swiglu_limit={call['swiglu_limit']} clamps"
    if call["hidden_pad"] or call["intermediate_pad"]:
        return "hidden_pad / intermediate_pad"
    if GateMode(call["gate_mode"]) != GateMode.SEPARATED:
        return f"gate_mode {GateMode(call['gate_mode']).value}"
    activation = ActivationType(call["activation"])
    if activation not in (ActivationType.Silu, ActivationType.Gelu):
        return f"activation {activation}"
    arch = get_gfx_runtime()
    if arch not in archs:
        return f"arch {arch}"
    w1, w2 = call["w1"], call["w2"]
    if w1.dtype not in weight_dtypes or w2.dtype not in weight_dtypes:
        expected = " or ".join(map(str, weight_dtypes))
        return f"weights {w1.dtype} / {w2.dtype}, not {expected}"
    if w1.dim() != 3 or w2.dim() != 3:
        return f"weights of {w1.dim()} / {w2.dim()} dimensions"
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if w1.shape[1] not in (inter_dim, 2 * inter_dim):
        return f"w1 {tuple(w1.shape)} does not match w2 {tuple(w2.shape)}"
    # g1u0 stays on legacy: CK stage 1 always reads w1 as gate and up, and the 1-stage
    # ASM `fmoe` that legacy runs for some g1u0 keys is not ported.
    if w1.shape[1] == inter_dim:
        return f"w1 {tuple(w1.shape)} is g1u0 (no separate up projection)"
    hidden_states = call["hidden_states"]
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    if dtype not in (dtypes.bf16, dtypes.fp16):
        return f"dtype {dtype}"
    if call["topk_ids"].dim() != 2:
        return f"topk_ids of {call['topk_ids'].dim()} dimensions"
    return None


def key_args_supported(problem) -> bool:
    """False for keys with arguments or a layout no local path supports yet (see
    `why_not`)."""
    activation = problem.activation
    return (
        problem.layout.g1u1
        and not clamps(activation.swiglu_limit)
        and problem.inter is None
        and (activation.situ_beta, activation.situ_linear_beta) == (1.0, 1.0)
    )


def _relookup_problem(problem, token):
    shape = replace(problem.shape, m=token)
    return replace(
        problem.with_features(remove=_RELOOKUP_DROPS), inter=None, shape=shape
    )


class LocalDispatch:
    def __init__(self, segments, takes):
        self.segments = tuple(segments)
        self.takes = takes
        self.peek = tuned.cached(maxsize=2048)(self._peek)
        self.resolve = tuned.cached(maxsize=2048)(self._resolve)
        self.relookup = tuned.cached(maxsize=2048)(self._relookup)
        self.dispatch = tuned.cached(maxsize=2048)(self._dispatch)

    def _peek(self, problem):
        """The resolution, with nothing emitted; None where resolving raises."""
        try:
            return resolver.resolve(problem, self.segments, emit_on_raise=False)
        except Exception:  # noqa: BLE001 - callers decline, or resolve again to raise
            return None

    def _resolve(self, problem, relookup):
        resolution = self.peek(problem)
        if resolution is None:
            # Raises, after emitting the lines legacy logs before raising.
            resolution = resolver.resolve(problem, self.segments)
        resolution.emit()
        return resolution

    def _relookup(self, problem, token):
        """The second lookup, keyed on the `hidden_states` rows rather than `topk_ids`."""
        return self.resolve(_relookup_problem(problem, token), True)

    def relookup_decline(self, problem, token):
        """Why the second lookup sends the call back to legacy, or None.

        Legacy raises there after the sort; v2 cannot emit the lines of a lookup it
        declines.
        """
        first = self.peek(problem)
        if first is None or first.plan is None or first.metadata.full_impl is not None:
            return None
        second = self.peek(_relookup_problem(problem, token))
        if second is None:
            return "the lookup after the sort raises"
        if second.plan is None:
            return f"the lookup after the sort declines ({second.declined})"
        return None

    def _dispatch(self, args, kwargs_items, arch, cu_num, token, emit=True):
        """`(resolution, problem)` for a key, or None when not taken or declined.

        `token` is the bucketed `hidden_states` row count. `problem` is None when no
        second lookup follows (whole-graph plans). With `emit` False nothing is logged,
        and a key whose resolution raises gives `_UNRESOLVED`.
        """
        problem = api.problem_from_key(args, dict(kwargs_items), arch, cu_num)
        if not self.takes(problem):
            return None
        reason = self.relookup_decline(problem, token)
        if reason is not None:
            if emit and overrides.explain():
                line = f"declined: {reason}; the legacy code runs this call"
                resolver.Resolution(None, declined=reason, explain=(line,)).emit()
            return None
        primary = self.resolve(problem, False) if emit else self.peek(problem)
        if primary is None:
            return _UNRESOLVED
        if primary.plan is None:
            return None
        if primary.metadata.full_impl is not None:
            return primary, None
        return primary, problem

    def run(self, call):
        # A declined call goes back to legacy before the prologue copies anything.
        preview = executor.prologue(call, quant_remap, prepare=False)
        args, kwargs, _ = api.dispatch_key(preview)
        token = get_padded_M(preview["hidden_states"].shape[0])
        where = (args, tuple(kwargs.items()), get_gfx_runtime(), get_cu_num(), token)
        if self.dispatch(*where, emit=False) is None:
            self.dispatch(*where, emit=True)
            return paths.DECLINED

        c = executor.prologue(call, quant_remap)
        args, kwargs, q_dtype_a = api.dispatch_key(c)
        dispatched = self.dispatch(
            args,
            tuple(kwargs.items()),
            get_gfx_runtime(),
            get_cu_num(),
            token,
            emit=True,
        )
        if dispatched is None:
            return paths.DECLINED
        resolution, problem = dispatched
        after_sort = None
        if problem is not None:
            after_sort = functools.partial(self.relookup, problem, token)
        return executor.run_two_stage(
            c, resolution.metadata, resolution.plan, q_dtype_a, after_sort=after_sort
        )

    def dispatch_record(self, args, kwargs):
        """The v2 `MOEMetadata` for a legacy key (logs emitted), or None if not taken.

        Declines a key whose second lookup after the sort would raise or be declined,
        as `run` does.
        """
        problem = api.problem_from_key(args, kwargs)
        if not self.takes(problem):
            return None
        if self.relookup_decline(problem, problem.shape.m) is not None:
            return None
        resolution = resolver.resolve(problem, self.segments)
        resolution.emit()
        return None if resolution.plan is None else resolution.metadata
