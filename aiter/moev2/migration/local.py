# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch and execution shared by the local paths (one GPU, the `fused_moe_` entry).

A local path supplies its segments and `takes(problem)`, the problems it runs (its
segments may accept more: a CK segment serves several quant formats, each in its own
path). `LocalDispatch` resolves each legacy dispatch key once, as the cached legacy
`get_2stage_cfgs` does, and runs the plan with `executor.run_two_stage` (`run_mxfp4` for
`per_1x32`).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import functools
import math
from dataclasses import replace

from aiter import ActivationType, QuantType, dtypes, logger, presorted_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import overrides, resolver, tuned
from ..execute import executor
from ..policy import quant_remap, resolve_activation_dtype
from ..problem import FrozenMap
from ..segments.mxmoe import (
    _is_mxfp4_inline_sort,
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_inline_sort_unsupported,
)
from ..shapes import get_inter_dim, get_padded_M

# After sorting, fused_moe_2stages looks the key up again without these, on the
# hidden_states rows. _fused_moe_impl's _metadata_transform then replaces the result
# with the first lookup's, so the second lookup only logs, or raises.
_RELOOKUP_DROPS = frozenset({"act_scales", "num_local_tokens", "scatter"})
_RELOOKUP_DROPS_EXTRAS = frozenset({"disable_inline_sort", "exclude_mxmoe"})
_UNRESOLVED = object()

# `fused_moe_` arguments no local path supports yet.
_UNSUPPORTED_ARGS = (
    "quant_dtype_a2",
    "ep_source_token_map",
    "w1_global_scale",
    "w2_global_scale",
)
_BIAS_ARGS = ("bias1", "bias2")
_ACT_PARAM_ARGS = ("beta", "linear_beta")
SILU_GELU = (ActivationType.Silu, ActivationType.Gelu)
SEPARATED = (GateMode.SEPARATED,)


def clamps(swiglu_limit) -> bool:
    """None, 0 (vLLM's spelling) and inf are no clamp, as `fused_moe_` reads them for
    Silu. Legacy ignores a finite limit with Gelu; paths without `act_params` decline
    such a call.
    """
    return swiglu_limit is not None and swiglu_limit != 0 and swiglu_limit != math.inf


def activation_dtype(call):
    """`q_dtype_a` as `_fused_moe_impl` picks it for the call: `quant_dtype_a`, else the
    `policy.py` choice."""
    if call["quant_dtype_a"] is not None:
        return call["quant_dtype_a"]
    quant_type = QuantType(call["quant_type"])
    w1 = call["w1"]
    return resolve_activation_dtype(
        quant_remap.get(quant_type, quant_type),
        w1.dtype,
        activation=ActivationType(call["activation"]),
        gate_mode=GateMode(call["gate_mode"]),
        M=call["topk_ids"].shape[0],
        hidden_dtype=call["hidden_states"].dtype,
        has_a1_scale=call["a1_scale"] is not None,
        gfx=get_gfx_runtime(),
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
    )


def why_not(
    call,
    quant_types,
    archs,
    weight_dtypes,
    *,
    act_dtypes=None,
    act_dtype_arg=False,
    activations=SILU_GELU,
    gate_modes=SEPARATED,
    bias=False,
    act_params=False,
    pads=False,
    g1u0=False,
):
    """Why a local path does not claim the `fused_moe_` call, or None if it does.

    Cheap checks on the arguments; `run` may still decline a claimed call. Claimed: a
    quant type in `quant_types` (and the same `quant_type_a`, if given), an activation
    in `activations` and a gate mode in `gate_modes`, no `quant_dtype_a2`, global
    scales, scatter, HERD routing, deterministic mode or registered presort, outside
    AITER_ONLINE_TUNE, on `archs`, with 3D `w1`/`w2` of one of `weight_dtypes` and,
    unless `g1u0`, `w1` holding gate and up (g1u1); int8 weights only without
    doweight_stage1. A caller-chosen `quant_dtype_a`, bias tensors, a clamping
    `swiglu_limit`, `beta`/`linear_beta` and padding only where allowed. With
    `act_dtypes`, the activation format of the call (`activation_dtype`) must be one of
    them (paths that share a weight format split it by activation format).
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
    quant_type = QuantType(call["quant_type"])
    if quant_type not in quant_types:
        return f"quant_type {quant_type}"
    quant_type_a = call["quant_type_a"]
    if quant_type_a is not None and QuantType(quant_type_a) != quant_type:
        return f"quant_type_a {QuantType(quant_type_a)}"
    unsupported = _UNSUPPORTED_ARGS
    if not act_dtype_arg:
        unsupported += ("quant_dtype_a",)
    if not bias:
        unsupported += _BIAS_ARGS
    if not act_params:
        unsupported += _ACT_PARAM_ARGS
    for name in unsupported:
        if call[name] is not None:
            return f"{name} is set"
    if not act_params and clamps(call["swiglu_limit"]):
        return f"swiglu_limit={call['swiglu_limit']} clamps"
    if not pads and (call["hidden_pad"] or call["intermediate_pad"]):
        return "hidden_pad / intermediate_pad"
    gate_mode = GateMode(call["gate_mode"])
    if gate_mode not in gate_modes:
        return f"gate_mode {gate_mode.value}"
    activation = ActivationType(call["activation"])
    if activation not in activations:
        return f"activation {activation}"
    arch = get_gfx_runtime()
    if arch not in archs:
        return f"arch {arch}"
    w1, w2 = call["w1"], call["w2"]
    if w1.dtype not in weight_dtypes:
        expected = " or ".join(map(str, weight_dtypes))
        return f"weights {w1.dtype} / {w2.dtype}, not {expected}"
    if w2.dtype != w1.dtype:
        return f"w2 dtype {w2.dtype} differs from w1 dtype {w1.dtype}"
    if w1.dim() != 3 or w2.dim() != 3:
        return f"weights of {w1.dim()} / {w2.dim()} dimensions"
    # No ASM int8 stage 1 applies the routing weights; legacy raises in its kernel lookup.
    if w1.dtype == dtypes.i8 and call["doweight_stage1"]:
        return "int8 weights with doweight_stage1"
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if w1.shape[1] not in (inter_dim, 2 * inter_dim):
        return f"w1 {tuple(w1.shape)} does not match w2 {tuple(w2.shape)}"
    # g1u0 stays on legacy unless the path runs it: CK stage 1 always reads w1 as gate
    # and up, and the 1-stage ASM `fmoe` that legacy runs for some g1u0 keys is not
    # ported.
    if not g1u0 and w1.shape[1] == inter_dim:
        return f"w1 {tuple(w1.shape)} is g1u0 (no separate up projection)"
    hidden_states = call["hidden_states"]
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    if dtype not in (dtypes.bf16, dtypes.fp16):
        return f"dtype {dtype}"
    if call["topk_ids"].dim() != 2:
        return f"topk_ids of {call['topk_ids'].dim()} dimensions"
    if act_dtypes is not None:
        q_dtype_a = activation_dtype(call)
        if q_dtype_a not in act_dtypes:
            expected = " or ".join(map(str, act_dtypes))
            return f"activations {q_dtype_a}, not {expected}"
    return None


def key_args_supported(
    problem,
    *,
    activations=("silu", "gelu"),
    gate_modes=SEPARATED,
    act_params=False,
    g1u0=False,
) -> bool:
    """False for keys with arguments or a layout the path does not support (see
    `why_not`; bias and padding are left to the segments' `Cap`s)."""
    activation = problem.activation
    return (
        (g1u0 or problem.layout.g1u1)
        and activation.kind in activations
        and problem.gate_mode in gate_modes
        and problem.inter is None
        and (
            act_params
            or (
                not clamps(activation.swiglu_limit)
                and (activation.situ_beta, activation.situ_linear_beta) == (1.0, 1.0)
            )
        )
        and not (
            problem.w1.dtype == dtypes.i8 and "doweight_stage1" in problem.features
        )
    )


def _relookup_problem(problem, token):
    shape = replace(problem.shape, m=token)
    problem = problem.with_features(remove=_RELOOKUP_DROPS)
    extras = FrozenMap(
        {k: v for k, v in problem.extras.items() if k not in _RELOOKUP_DROPS_EXTRAS}
    )
    return replace(problem, inter=None, shape=shape, extras=extras)


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
        args, kwargs, q_dtype_a = api.dispatch_key(preview)
        if executor.a16w4_unsupported(preview, q_dtype_a) is not None:
            # Legacy raises NotImplementedError before its dispatch.
            return paths.DECLINED
        token = get_padded_M(preview["hidden_states"].shape[0])
        arch, cu_num = get_gfx_runtime(), get_cu_num()
        where = (args, tuple(kwargs.items()), arch, cu_num, token)
        if self.dispatch(*where, emit=False) is None:
            self.dispatch(*where, emit=True)
            return paths.DECLINED
        fallbacks = ()
        if preview.quant_type == QuantType.per_1x32:
            fallbacks = self._mxfp4_fallbacks(
                preview, args, kwargs, arch, cu_num, token
            )
            if fallbacks is None:
                return paths.DECLINED

        c = executor.prologue(call, quant_remap)
        args, kwargs, q_dtype_a = api.dispatch_key(c)
        dispatched = self.dispatch(
            args, tuple(kwargs.items()), arch, cu_num, token, emit=True
        )
        if dispatched is None:
            return paths.DECLINED
        for fallback_kwargs, warning in fallbacks:
            message = warning(c)
            if not message:
                continue
            logger.warning(message)
            dispatched = self.dispatch(
                args, tuple(fallback_kwargs.items()), arch, cu_num, token
            )
            if "_exclude_mxmoe" not in fallback_kwargs:
                continue
            block_m = dispatched[0].metadata.block_m
            if int(block_m) != int(c.block_size_M):
                raise NotImplementedError(
                    f"No fallback backend supports block_size_M={c.block_size_M}; "
                    f"resolved backend requires block_m={block_m}"
                )
        resolution, problem = dispatched
        after_sort = None
        if problem is not None:
            after_sort = functools.partial(self.relookup, problem, token)
        run = (
            executor.run_mxfp4
            if c.quant_type == QuantType.per_1x32
            else executor.run_two_stage
        )
        return run(
            c, resolution.metadata, resolution.plan, q_dtype_a, after_sort=after_sort
        )

    def _mxfp4_fallbacks(self, preview, args, kwargs, arch, cu_num, token):
        """`_fused_moe_impl`'s second resolutions, in its order, as `(kwargs, warning)`
        pairs: an MXMOE row whose block size the caller overrides, then a tuned
        inline-sort row this call cannot run, are discarded and the key resolved again.
        `warning(c)` gives the line to log for the prepared call, or "" where the row
        runs after all; `run` calls it after the first lookup has logged, as legacy
        checks then.

        None where v2 declines a key resolved again; decided before anything is logged.
        """
        fallbacks = []
        first = self._peek_key(args, kwargs, arch, cu_num, token)
        if first is None or first is paths.DECLINED:
            return fallbacks
        metadata = first.metadata
        block_size_M = preview.block_size_M
        if (
            getattr(metadata.stage1, "func", metadata.stage1) is _mxfp4_a4w4_stage1_fw
            and not _is_mxfp4_inline_sort(metadata)
            and block_size_M is not None
            and int(block_size_M) != int(metadata.block_m)
        ):
            reason = (
                f"block_size_M={block_size_M} differs from configured "
                f"BM={metadata.block_m}"
            )
            fallback_kwargs = {**kwargs, "_exclude_mxmoe": True}
            first = self._peek_key(args, fallback_kwargs, arch, cu_num, token)
            if first is paths.DECLINED:
                return None
            message = (
                f"[fused_moe] MXMOE pair is unsupported ({reason}); "
                "using default heuristics"
            )
            fallbacks.append((fallback_kwargs, lambda c: message))
            # `run` raises where resolving raises or the block size still differs.
            if first is None or int(block_size_M) != int(first.metadata.block_m):
                return fallbacks
            metadata = first.metadata
        if not _is_mxfp4_inline_sort(metadata):
            return fallbacks
        fallback_kwargs = {**kwargs, "_disable_inline_sort": True}
        # Declined even if the row could run: the check probes a JIT module, which
        # legacy loads only after the first lookup has logged.
        if self._peek_key(args, fallback_kwargs, arch, cu_num, token) is paths.DECLINED:
            return None

        def inline_sort_warning(c):
            reason = _mxfp4_inline_sort_unsupported(
                metadata,
                c["hidden_states"],
                c["w1"],
                c["w2"],
                c["w1_scale"],
                c["w2_scale"],
                c["topk_ids"],
                gate_mode=c.gate_mode,
                expert_mask=c["expert_mask"],
                bias1=c["bias1"],
                bias2=c["bias2"],
                a1_scale=c["a1_scale"],
                a2_scale=c["a2_scale"],
                stage2_scatter=None,
                block_size_M=block_size_M,
                hidden_pad=c["hidden_pad"],
                intermediate_pad=c["intermediate_pad"],
            )
            if not reason:
                return ""
            return (
                f"[fused_moe] inline-sort config is unsupported ({reason}); "
                "using default heuristics"
            )

        fallbacks.append((fallback_kwargs, inline_sort_warning))
        return fallbacks

    def _peek_key(self, args, kwargs, arch, cu_num, token):
        """A key's first resolution, with nothing emitted: DECLINED where `dispatch`
        declines the key, None where it raises."""
        problem = api.problem_from_key(args, kwargs, arch, cu_num)
        if not self.takes(problem) or self.relookup_decline(problem, token):
            return paths.DECLINED
        first = self.peek(problem)
        if first is not None and first.plan is None:
            return paths.DECLINED
        return first

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
