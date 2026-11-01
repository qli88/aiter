# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch and execution shared by the local paths (one GPU, the `fused_moe_` entry).

A local path supplies its segments and `takes(problem)`, the problems it runs (its
segments may accept more: a CK segment serves several quant formats, each in its own
path). `LocalDispatch` resolves each legacy dispatch key once, as the cached legacy
`get_2stage_cfgs` does, and runs the plan with `executor.run_two_stage` (`run_mxfp4` for
MXFP4 weights, `run_one_stage` for a 1-stage plan, which has no second lookup).
"""

import functools
from dataclasses import replace

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import overrides, resolver, tuned
from ..execute import executor
from ..policy import quant_remap, resolve_activation_dtype
from ..problem import FrozenMap
from ..segments.mxmoe import _is_mxfp4_inline_sort, _mxfp4_inline_sort_unsupported
from ..shapes import get_inter_dim, get_padded_M

# fused_moe_2stages looks the key up again without these, after sorting, for its log
# lines only.
_RELOOKUP_DROPS = frozenset({"act_scales", "num_local_tokens", "scatter"})
_RELOOKUP_DROPS_EXTRAS = frozenset({"disable_inline_sort"})

# `fused_moe_` arguments no local path supports yet.
_UNSUPPORTED_ARGS = ("quant_dtype_a2", "ep_source_token_map")
_BIAS_ARGS = ("bias1", "bias2")
_ACT_PARAM_ARGS = ("swiglu_limit", "beta", "linear_beta")
SILU_GELU = (ActivationType.Silu, ActivationType.Gelu)
SEPARATED = (GateMode.SEPARATED,)


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


def claims(
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
) -> bool:
    """Cheap checks on the `fused_moe_` arguments; `run` may still decline.

    A quant type in `quant_types` (and the same `quant_type_a`, if given), an activation
    in `activations` and a gate mode in `gate_modes`, no `quant_dtype_a2` or scatter,
    outside AITER_ONLINE_TUNE, on `archs`, with 3D `w1`/`w2` of one of `weight_dtypes`
    and, unless `g1u0`, `w1` holding gate and up (g1u1). A caller-chosen
    `quant_dtype_a`, bias tensors, `swiglu_limit`/`beta`/`linear_beta` and padding only
    where allowed. With `act_dtypes`, the activation format of the call
    (`activation_dtype`) must be one of them (paths that share a weight format split it
    by activation format).
    """
    if overrides.online_tune():
        return False
    quant_type = QuantType(call["quant_type"])
    if quant_type not in quant_types:
        return False
    quant_type_a = call["quant_type_a"]
    if quant_type_a is not None and QuantType(quant_type_a) != quant_type:
        return False
    unsupported = _UNSUPPORTED_ARGS
    if not act_dtype_arg:
        unsupported += ("quant_dtype_a",)
    if not bias:
        unsupported += _BIAS_ARGS
    if not act_params:
        unsupported += _ACT_PARAM_ARGS
    if any(call[name] is not None for name in unsupported):
        return False
    if not pads and (call["hidden_pad"] or call["intermediate_pad"]):
        return False
    if GateMode(call["gate_mode"]) not in gate_modes:
        return False
    if ActivationType(call["activation"]) not in activations:
        return False
    if get_gfx() not in archs or get_gfx_runtime() not in archs:
        return False
    w1, w2 = call["w1"], call["w2"]
    if (
        w1.dtype not in weight_dtypes
        or w2.dtype != w1.dtype
        or w1.dim() != 3
        or w2.dim() != 3
    ):
        return False
    # g1u0 stays on legacy unless the path runs it: CK and ASM stage 1 always read w1
    # as gate and up, so only the 1-stage ASM `fmoe` keys have a plan.
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if not g1u0 and w1.shape[1] != 2 * inter_dim:
        return False
    hidden_states = call["hidden_states"]
    dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
    if dtype not in (dtypes.bf16, dtypes.fp16) or call["topk_ids"].dim() != 2:
        return False
    return act_dtypes is None or activation_dtype(call) in act_dtypes


def key_args_supported(
    problem,
    *,
    activations=("silu", "gelu"),
    gate_modes=SEPARATED,
    act_params=False,
    g1u0=False,
) -> bool:
    """False for keys with arguments or a layout the path does not support (see
    `claims`; bias and padding are left to the segments' `Cap`s)."""
    activation = problem.activation
    return (
        (g1u0 or problem.layout.g1u1)
        and activation.kind in activations
        and problem.gate_mode in gate_modes
        and problem.inter is None
        and (
            act_params
            or (
                activation.swiglu_limit is None
                and (activation.situ_beta, activation.situ_linear_beta) == (1.0, 1.0)
            )
        )
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
        extras = FrozenMap(
            {k: v for k, v in problem.extras.items() if k not in _RELOOKUP_DROPS_EXTRAS}
        )
        return self.resolve(
            replace(problem, inter=None, shape=shape, extras=extras), True
        )

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
        if executor.a16w4_unsupported(c, q_dtype_a) is not None:
            # Legacy raises NotImplementedError before its dispatch.
            return paths.DECLINED
        dispatched = self.dispatch(
            args, tuple(kwargs.items()), get_gfx_runtime(), get_cu_num()
        )
        if dispatched is None:
            return paths.DECLINED
        if c.quant_type == QuantType.per_1x32:
            dispatched = self._inline_sort_fallback(c, args, kwargs, dispatched)
            if dispatched is None:
                return paths.DECLINED
        resolution, problem = dispatched
        if resolution.metadata.run_1stage:
            return executor.run_one_stage(
                c, resolution.metadata, resolution.plan, q_dtype_a
            )
        after_sort = None
        if problem is not None:
            token = get_padded_M(c["hidden_states"].shape[0])
            after_sort = functools.partial(self.relookup, problem, token)
        run = (
            executor.run_mxfp4
            if c.quant_type == QuantType.per_1x32
            else executor.run_two_stage
        )
        return run(
            c, resolution.metadata, resolution.plan, q_dtype_a, after_sort=after_sort
        )

    def _inline_sort_fallback(self, c, args, kwargs, dispatched):
        """`_fused_moe_impl`: a tuned inline-sort row this call cannot run is discarded,
        and the key resolved again without inline-sort rows. None: that key declines."""
        metadata = dispatched[0].metadata
        if not _is_mxfp4_inline_sort(metadata):
            return dispatched
        reason = _mxfp4_inline_sort_unsupported(
            metadata,
            c["hidden_states"],
            c["w1"],
            c["w2"],
            c["w1_scale"],
            c["w2_scale"],
            c["topk_ids"],
            gate_mode=c.gate_mode,
            num_local_tokens=c["num_local_tokens"],
            expert_mask=c["expert_mask"],
            bias1=c["bias1"],
            bias2=c["bias2"],
            a1_scale=c["a1_scale"],
            a2_scale=c["a2_scale"],
            stage2_scatter=None,
            block_size_M=c.block_size_M,
            hidden_pad=c["hidden_pad"],
            intermediate_pad=c["intermediate_pad"],
        )
        if not reason:
            return dispatched
        kwargs = {**kwargs, "_disable_inline_sort": True}
        fallback = self.dispatch(
            args, tuple(kwargs.items()), get_gfx_runtime(), get_cu_num()
        )
        if fallback is not None:
            logger.warning(
                f"[fused_moe] inline-sort config is unsupported ({reason}); "
                "using default heuristics"
            )
        return fallback

    def dispatch_record(self, args, kwargs):
        """The v2 `MOEMetadata` for a legacy key (logs emitted), or None if not taken."""
        problem = api.problem_from_key(args, kwargs)
        if not self.takes(problem):
            return None
        resolution = resolver.resolve(problem, self.segments)
        resolution.emit()
        return None if resolution.plan is None else resolution.metadata
