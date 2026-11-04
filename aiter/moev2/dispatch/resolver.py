# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Resolve a `MoeProblem` to a plan.

The order and every message are the legacy `get_2stage_cfgs` ones for the kernel families
the given segments cover: tuned lookup (with its fallbacks), checks that discard a tuned
row (logged as today), then the tuned or the default parameters. A tuned row naming a
family no given segment owns is not resolved here: the result is a decline, with nothing
logged, and the caller hands the call back to the legacy code.

Log lines are collected, not emitted, so a declined resolution prints nothing; the caller
emits them (`Resolution.emit`) once per problem, where the legacy cache would have logged.
"""

import logging
import math
from dataclasses import dataclass

from aiter import logger

from ..execute.metadata import MOEMetadata
from ..segments.asm import stage2_block_m
from ..segments.base import NO_OPTIONS, Params
from ..segments.defaults import (
    BLOCK_SIZE_M,
    default_block_m,
    default_ksplit,
    use_nt,
)
from ..segments.whole_graph import FusedMoeImplResolutionError, resolve_fused_moe_impl
from . import kernel_names, overrides, profile, ranking, tuned
from .buf import CALL_INPUT
from .ops import Op, chain
from .plan import MoePlan, SegmentChoice

INFO = logging.INFO
WARNING = logging.WARNING
EXPLAIN_PREFIX = "[aiter.moe] "
# Activations the 1-stage ASM kernels have (Silu/Gelu, gfx950 MXFP4 SiTUv2 FLAT).
ONE_STAGE_ACTIVATIONS = ("silu", "gelu", "situv2")


@dataclass(frozen=True)
class Resolution:
    plan: MoePlan | None  # None when declined
    metadata: MOEMetadata | None = (
        None  # the plan lowered to the legacy stage callables
    )
    logs: tuple = ()  # (level, message), in legacy order
    declined: str | None = None
    explain: tuple = ()

    def emit(self):
        for line in self.explain:
            logger.info(EXPLAIN_PREFIX + line)
        for level, message in self.logs:
            if level == WARNING:
                logger.warning(message)
            else:
                logger.info(message)


def feasible_plans(problem, segments):
    """`(plans, rejected)`: segment tuples covering the op chain, and `{name: reason}`.

    In a plan, every `Buf` a segment reads matches the last one of that kind produced
    before it. No kernel is chosen yet: a format the kernel fixes is None (any) here,
    and `plan_mismatch` checks it once the kernels are chosen. A segment that fits the chain but in no plan, because of its boundaries,
    is in `rejected` with the first mismatch found.
    """
    ops = chain(problem)
    rejected = {}
    usable = []
    for segment in segments:
        reason = segment.reject_reason(problem)
        if reason is None:
            usable.append(segment)
        else:
            rejected[segment.name] = reason
    reads = {s.name: s.reads(problem) for s in usable}
    produced = {s.name: s.produces(problem, Params()) for s in usable}
    mismatches = {}
    placed = set()

    def walk(i, last):
        if i == len(ops):
            yield ()
            return
        for segment in usable:
            start, end = segment.covers
            if start != ops[i] or end not in ops:
                continue
            reason = next(
                filter(
                    None, (b.mismatch(last.get(b.kind)) for b in reads[segment.name])
                ),
                None,
            )
            if reason is not None:
                mismatches.setdefault(segment.name, reason)
                continue
            out = produced[segment.name]
            for rest in walk(ops.index(end) + 1, {**last, out.kind: out}):
                placed.add(segment.name)
                yield (segment,) + rest

    plans = list(walk(0, {CALL_INPUT.kind: CALL_INPUT}))
    for name, reason in mismatches.items():
        if name not in placed:
            rejected[name] = reason
    return plans, rejected


def _plan_name(structure):
    return " | ".join(s.name for s in structure)


def _structure_value(name, value):
    # A tuned table's block_m may be a numpy or float scalar (legacy int()s it).
    return int(value) if name == "block_m" and value is not None else value


def _boundary_params(shared, choice):
    """The params a chosen segment's boundaries depend on: its own, plus the shared
    value of each structure parameter it declares and does not set."""
    filled = {
        k: shared[k]
        for k in choice.segment.structure_params
        if k in shared and k not in choice.params
    }
    return choice.params.replace(**filled)


def _plan(problem, choices, source, params):
    return MoePlan(
        choices,
        source,
        params,
        tuple(
            c.segment.produces(
                problem,
                _boundary_params(
                    {k: _structure_value(k, v) for k, v in params.items()}, c
                ),
            )
            for c in choices[:-1]
        ),
    )


def plan_mismatch(problem, plan) -> str | None:
    """Why the chosen kernels of `plan` disagree, or None: a structure parameter a
    segment sets to another value than the plan (or, when the plan has none, the first
    segment setting it), or a `Buf` it reads with its kernel (`reads_with`) that the
    last one of that kind before it (`produces` with the chosen params) does not
    match. Only the plan's `block_m` is checked: a caller's `block_size_M` overrides
    it at run time, as in legacy."""
    shared = {k: _structure_value(k, v) for k, v in plan.params.items()}
    owners = dict.fromkeys(shared, "the plan")
    for c in plan.segments:
        for k in sorted(c.segment.structure_params & c.params.keys()):
            value = _structure_value(k, c.params[k])
            if k not in shared:
                shared[k], owners[k] = value, c.segment.name
            elif value != shared[k]:
                return f"{c.segment.name} has {k}={value!r}, {owners[k]} {shared[k]!r}"
    last = {CALL_INPUT.kind: CALL_INPUT}
    for c in plan.segments:
        params = _boundary_params(shared, c)
        for buf in c.segment.reads_with(problem, params):
            reason = buf.mismatch(last.get(buf.kind))
            if reason is not None:
                return f"{c.segment.name} {reason}"
        out = c.segment.produces(problem, params)
        last[out.kind] = out
    return None


def _owned_kernel(raw):
    """The parsed kernel name of a tuned cell, or a decline reason (str)."""
    if not isinstance(raw, str):
        return f"kernel name {raw!r} is not a string"
    if raw != raw.strip():
        return f"kernel name {raw!r} has surrounding blanks"
    kernel = kernel_names.try_parse(raw)
    if raw and kernel is None:
        return f"unknown kernel name {raw!r}"
    return kernel


def _no_stage2(raw) -> bool:
    """A 1-stage row's kernelName2: empty, the literal "Null", or a blank cell (NaN)."""
    return raw in ("", "Null") or (isinstance(raw, float) and math.isnan(raw))


def _plain_1stage_row(row) -> bool:
    """A 1-stage row with an ASM kernelName1 and no stage 2: none of legacy's discards
    before the 1-stage activation check look at it."""
    k1 = _owned_kernel(row.get("kernelName1", ""))
    return (
        bool(row.get("run_1stage", False))
        and not isinstance(k1, str)
        and k1 is not None
        and k1.family == kernel_names.ASM
        and _no_stage2(row.get("kernelName2", ""))
    )


def match_row(plans, row):
    """The plan structure that runs tuned `row`, or a decline reason (str)."""
    k1 = _owned_kernel(row.get("kernelName1", ""))
    if isinstance(k1, str):
        return k1
    if row.get("run_1stage", False):
        if not _plain_1stage_row(row):
            return (
                f"1-stage row with kernelName1={k1}, "
                f"kernelName2={row.get('kernelName2', '')!r}"
            )
        flat = bool(int(row["flat"])) if "flat" in row else False
        for structure in plans:
            one_stage = _one_stage(structure)
            if (
                one_stage is not None
                and one_stage.owns(k1)
                and getattr(structure[0], "flat", False) == flat
            ):
                return structure
        return f"no segment runs 1-stage {k1}{' (FLAT)' if flat else ''}"
    if k1 is not None and k1.family == kernel_names.IMPL:
        for structure in plans:
            if len(structure) == 1 and structure[0].owns(k1):
                return structure
        return f"no segment runs {k1}"
    k2 = _owned_kernel(row.get("kernelName2", ""))
    if isinstance(k2, str):
        return k2
    for structure in plans:
        gemm1 = [s for s in structure if s.covers[0] == Op.GEMM1]
        gemm2 = [s for s in structure if s.covers[0] == Op.GEMM2]
        if gemm1 and gemm2 and gemm1[0].owns(k1) and gemm2[0].owns(k2):
            return structure
    return f"no segments run kernelName1={k1}, kernelName2={k2}"


def _one_stage(structure):
    """The segment of `structure` that runs both GEMMs in one launch, or None."""
    if len(structure) == 1:
        return None
    return next(
        (s for s in structure if s.covers[0] <= Op.GEMM1 and s.covers[1] >= Op.GEMM2),
        None,
    )


def _one_stage_plan(
    problem,
    structure,
    source,
    kernelName1,
    block_m,
    ksplit,
    xbf16,
    run_1stage=True,
    flat=0,
):
    """The plan and the legacy `fused_moe_1stage` metadata of a 1-stage structure,
    or the reason (str) its chosen kernels disagree (`plan_mismatch`).
    `run_1stage`: the tuned row's value as legacy passes it on."""
    one_stage = _one_stage(structure)
    params = Params(kernel=kernelName1, xbf16=xbf16, flat=flat)
    choices = tuple(
        SegmentChoice(segment, params if segment is one_stage else Params())
        for segment in structure
    )
    plan = _plan(problem, choices, source, Params(block_m=block_m, ksplit=ksplit))
    reason = plan_mismatch(problem, plan)
    if reason is not None:
        return reason
    metadata = MOEMetadata(
        one_stage.build(problem, params),
        None,
        block_m,
        ksplit,
        run_1stage,
        flat=flat,
    )
    return plan, metadata


def _two_stage_plan(
    problem,
    structure,
    source,
    kernelName1,
    kernelName2,
    block_m,
    ksplit,
    nt,
    run_1stage,
    options,
    row,
):
    """The plan and its metadata, or the reason (str) its chosen kernels disagree
    (`plan_mismatch`). `row`: the tuned row the plan runs, None for a default."""
    gemm1 = next(s for s in structure if s.covers[0] == Op.GEMM1)
    is_asm = kernel_names.ASM in gemm1.prefixes
    if is_asm:
        block_m = stage2_block_m(block_m)
    choices = []
    for segment in structure:
        if segment.covers[0] == Op.GEMM1:
            params = Params(
                options, kernel=kernelName1, kernel2=kernelName2, ksplit=ksplit, nt=nt
            )
        elif segment.covers[0] == Op.GEMM2:
            params = Params(kernel=kernelName2, kernel1=kernelName1, nt=nt)
        else:
            params = Params()
        if row is not None:
            params = params.replace(**segment.row_params(problem, row))
        choices.append(SegmentChoice(segment, params))
    fields = {}
    for choice in choices:
        fields.update(choice.segment.metadata_fields(problem, choice.params))
    block_m = fields.pop("block_m", block_m)
    ksplit = fields.pop("ksplit", ksplit)
    plan = _plan(
        problem, tuple(choices), source, Params(block_m=block_m, ksplit=ksplit)
    )
    reason = plan_mismatch(problem, plan)
    if reason is not None:
        return reason
    stage1 = plan.starting(Op.GEMM1)
    stage2 = plan.starting(Op.GEMM2)
    metadata = MOEMetadata(
        stage1.segment.build(problem, stage1.params),
        stage2.segment.build(problem, stage2.params),
        block_m,
        ksplit if is_asm else int(ksplit),
        run_1stage,
        **fields,
    )
    return plan, metadata


def resolve(problem, segments) -> Resolution:
    explain = [] if overrides.explain() else None

    def note(line):
        if explain is not None:
            explain.append(line)

    note(f"problem: {problem}")
    plans, rejected = feasible_plans(problem, segments)
    note(f"feasible structures ({len(plans)}):")
    for i, structure in enumerate(plans, 1):
        note(f"   S{i} {_plan_name(structure)}")
    if rejected:
        note("rejected: " + "; ".join(f"{k} ({v})" for k, v in rejected.items()))

    def declined(reason):
        note(f"declined: {reason}; the legacy code runs this call")
        return Resolution(None, declined=reason, explain=tuple(explain or ()))

    if not plans:
        return declined("no feasible plan")
    keys = problem.log_keys()
    cfg = tuned.lookup(problem)
    if cfg is None:
        note(f"tuned lookup: miss (tier {problem.shape.m})")
        matched = None
    else:
        note(
            f"tuned lookup: kernelName1={cfg.get('kernelName1')!r} "
            f"kernelName2={cfg.get('kernelName2')!r}"
        )
        matched = None
    logs = []
    if (
        cfg is not None
        and _plain_1stage_row(cfg)
        and problem.activation.kind not in ONE_STAGE_ACTIVATIONS
    ):
        # Legacy's check after its other discards, none of which applies to the row.
        message = (
            f"[fused_moe] discarding 1-stage tuned config for unsupported "
            f"activation {problem.activation.activation}; using default heuristics"
        )
        logs.append((WARNING, message))
        note(f"tuned row discarded: {message}")
        cfg = None
    if cfg is not None:
        matched = match_row(plans, cfg)
        if isinstance(matched, str):
            return declined(matched)
    if matched is not None and len(matched) > 1:
        reason = next(
            filter(None, (s.declines_row(problem, cfg) for s in matched)), None
        )
        if reason is not None:
            return declined(reason)
        rejection = next(
            filter(None, (s.check_row(problem, cfg) for s in matched)), None
        )
        if rejection is not None:
            logs.append((WARNING, rejection.warning))
            note(f"tuned row discarded: {rejection.warning}")
            cfg = None
            matched = None
    bypass_tuned_config = overrides.bypass_tuned()
    kernel_name1 = (
        str(cfg.get("kernelName1", "") or "").strip() if cfg is not None else ""
    )
    try:
        full_impl = (
            None if bypass_tuned_config else resolve_fused_moe_impl(kernel_name1)
        )
    except FusedMoeImplResolutionError as error:
        logs.append((WARNING, f"[fused_moe] {error}; using default heuristics."))
        cfg = None
        full_impl = None
    if full_impl is not None:
        rejection = matched[0].check_row(problem, cfg, kernel_name1)
        if rejection is not None:
            if rejection.warning is not None:
                logs.append((WARNING, rejection.warning))
            note(f"tuned row discarded: {rejection.warning or 'expert_mask'}")
            cfg = None
            full_impl = None
    if full_impl is not None:
        block_m = int(cfg.get("block_m", BLOCK_SIZE_M))
        ksplit = int(cfg.get("ksplit", 0))
        logs.append((INFO, f"[fused_moe] using {kernel_name1!r} for {keys}"))
        whole = matched[0]
        plan = MoePlan(
            (SegmentChoice(whole, Params(kernel=kernel_name1, impl=full_impl)),),
            "tuned",
            Params(block_m=block_m, ksplit=ksplit),
        )
        note(f"selected {plan.describe()}")
        metadata = MOEMetadata(None, None, block_m, ksplit, full_impl=full_impl)
        return Resolution(plan, metadata, tuple(logs), explain=tuple(explain or ()))
    for _row in profile.exact(problem):
        pass  # no profile files are read yet

    if problem.extras.get("tuned_table") is not None and (
        cfg is None or bypass_tuned_config
    ):
        # Legacy raises: a dedicated table (FHMoE) runs its exact rows only.
        return declined("no exact row in the dedicated table")
    s = problem.shape
    q_type = problem.quant_type
    if cfg is None or bypass_tuned_config:
        best = ranking.best(problem, ranking.rankable(plans), tuned_row=cfg is not None)
        if best is None:
            return declined("no default ranking entry applies")
        region, structure, legacy = best
        if legacy is not None:
            return declined(f"ranking {region}: {legacy} is not ported")
        if structure is None:
            return declined(f"ranking {region}: no listed structure is feasible")
        kernelName1 = ""
        kernelName2 = ""
        one_stage = _one_stage(structure)
        run_1stage = one_stage is not None
        run_1stage_xbf16 = run_1stage and one_stage.default_xbf16(problem)
        flat = 0
        dims = (q_type, s.m, s.topk, s.experts, s.inter_dim, s.model_dim)
        block_m = BLOCK_SIZE_M if run_1stage else default_block_m(*dims)
        ksplit = 0 if run_1stage else default_ksplit(*dims)
        use_non_temporal_load = use_nt(s.m, s.topk, s.experts)
        logs.append(
            (
                INFO,
                f"run_1stage = {run_1stage}, xbf16 = {run_1stage_xbf16}, ksplit = {ksplit} q_type = {q_type} block_m = {block_m} use_nt = {use_non_temporal_load}, estimated_m_per_expert = {s.m * s.topk // s.experts}",
            )
        )
        source = "override" if cfg is not None else "default"
        note(f"ranking: {_plan_name(structure)} ({region})")
    else:
        structure = matched
        use_non_temporal_load = False
        block_m = cfg["block_m"]
        nt_override = overrides.nt_override()
        if nt_override != -1:
            use_non_temporal_load = bool(nt_override)
        elif "nt" in cfg:
            try:
                use_non_temporal_load = bool(int(float(cfg["nt"])))
            except (TypeError, ValueError):
                # blank or malformed column: keep the pre-column behaviour
                use_non_temporal_load = False
        ksplit = 0 if overrides.ksplit_disabled() else cfg["ksplit"]
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
        run_1stage = cfg.get("run_1stage", False)
        if "xbf16" in cfg:
            run_1stage_xbf16 = run_1stage and bool(int(cfg["xbf16"]))
        else:
            run_1stage_xbf16 = run_1stage and "blockscaleBf16" in str(kernelName1)
        if "flat" in cfg:
            flat = int(cfg["flat"]) if run_1stage else 0
        else:
            flat = 0
        if not problem.any_shuffled and not run_1stage:
            message = (
                f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
                "Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
                "Running with preshuffle_off may produce incorrect results."
            )
            logs.append((WARNING, message))
        source = "tuned"
    if run_1stage:
        tag = f"({kernelName1=}, {kernelName2=})"
        xbf16 = " xbf16" if run_1stage_xbf16 else ""
        logs.append(
            (
                INFO,
                f"[fused_moe] using 1stage{xbf16} {'default' if cfg is None else tag} for {keys} ",
            )
        )
        built = _one_stage_plan(
            problem,
            structure,
            source,
            kernelName1,
            block_m,
            ksplit,
            run_1stage_xbf16,
            run_1stage,
            flat,
        )
        if isinstance(built, str):
            return declined(f"the chosen kernels disagree: {built}")
        plan, metadata = built
        note(f"selected {plan.describe()}")
        return Resolution(plan, metadata, tuple(logs), explain=tuple(explain or ()))
    gemm1 = next(s for s in structure if s.covers[0] == Op.GEMM1)
    gemm2 = next(s for s in structure if s.covers[0] == Op.GEMM2)
    kernelName2 = gemm2.stage2_kernel(problem, kernelName1, kernelName2)
    tag = f"({kernelName1=}, {kernelName2=})"
    logs.append(
        (
            INFO,
            f"[fused_moe] using 2stage {'default' if cfg is None else tag} for {keys} ",
        )
    )
    default = gemm1.default_kernels(problem) if source != "tuned" else None
    options = NO_OPTIONS
    if default is not None:
        kernelName1, kernelName2 = default.kernel1, default.kernel2
        block_m, ksplit = default.block_m, default.ksplit
        options = default.options
        if default.warning is not None:
            logs.append((WARNING, default.warning))
    built = _two_stage_plan(
        problem,
        structure,
        source,
        kernelName1,
        kernelName2,
        block_m,
        ksplit,
        use_non_temporal_load,
        run_1stage,
        options,
        cfg if source == "tuned" else None,
    )
    if isinstance(built, str):
        return declined(f"the chosen kernels disagree: {built}")
    plan, metadata = built
    note(f"selected {plan.describe()}")
    return Resolution(plan, metadata, tuple(logs), explain=tuple(explain or ()))
