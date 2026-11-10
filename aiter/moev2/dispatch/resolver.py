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

import csv
import logging
import math
from dataclasses import dataclass

from aiter import logger

from ..execute.metadata import MOEMetadata
from ..problem import GROUP_1X32
from ..segments.asm import stage2_block_m
from ..segments.base import NO_OPTIONS, Params
from ..segments.defaults import (
    BLOCK_SIZE_M,
    default_block_m,
    default_ksplit,
    use_nt,
)
from ..segments.mxfp4 import mx_token
from ..segments.sort import AUX_SORT_OPUS
from ..segments.whole_graph import FusedMoeImplResolutionError, resolve_fused_moe_impl
from . import kernel_names, overrides, profile, ranking, tuned, tuned_plans
from .buf import CALL_INPUT
from .ops import Op, chain
from .plan import MoePlan, SegmentChoice
from .tuned_plans import STRUCTURE_SEPARATOR

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
    and `plan_mismatch` checks it once the kernels are chosen. A segment that fits the
    chain but in no plan, because of its boundaries, is in `rejected` with the first
    mismatch found. Raises if a usable segment's `produces` is not `Buf.concrete`.
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
    for name, out in produced.items():
        if not out.concrete:
            raise ValueError(f"{name} produces {out}: one value per field")
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
    return STRUCTURE_SEPARATOR.join(_names(structure))


def _names(structure) -> tuple:
    return tuple(s.name for s in structure)


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


def boundary_flags(plan) -> dict:
    """The legacy `MOEMetadata` flags that keep a 2-stage plan's stages consistent,
    from its boundaries: `output_aux` the sort's auxiliary outputs (rows
    `aux_sorted`), `fuse_quant` the per-1x32 dtype ("fp4", "fp8") of an intermediate
    stage 1 writes quantized itself, `skip_inter_quant` GEMM2 reading the v2 layout."""
    flags = {}
    for choice, buf in zip(plan.segments, plan.boundaries):
        if buf.kind == "sorted" and buf.rows == "aux_sorted":
            flags["output_aux"] = AUX_SORT_OPUS
        if buf.kind != "inter":
            continue
        flags["skip_inter_quant"] = buf.scales == "v2"
        first, last = choice.segment.covers
        if (
            first <= Op.GEMM1
            and last == Op.QUANT_INTER
            and buf.fmt is not None
            and buf.fmt.group == GROUP_1X32
        ):
            flags["fuse_quant"] = mx_token(buf.fmt.dtype)
    return flags


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
        gemm1 = [s for s in structure if _covers(s, Op.GEMM1)]
        gemm2 = [s for s in structure if s.covers[0] == Op.GEMM2]
        if gemm1 and gemm2 and gemm1[0].owns(k1) and gemm2[0].owns(k2):
            return structure
    return f"no segments run kernelName1={k1}, kernelName2={k2}"


def _row_rejection(problem, structure, row):
    """For a structure of two or more segments: `(declined, reason)` if one of them
    declines tuned `row` (the call goes back to the legacy code) or discards it (it
    runs untuned, `reason` the warning logged), else None."""
    reason = next(filter(None, (s.declines_row(problem, row) for s in structure)), None)
    if reason is not None:
        return True, reason
    rejection = next(filter(None, (s.check_row(problem, row) for s in structure)), None)
    if rejection is not None:
        return False, rejection.warning
    return None


def _row_values(row):
    """`(nt, run_1stage, xbf16, flat)` of tuned `row`, read as legacy reads them."""
    nt = False
    if "nt" in row:
        try:
            nt = bool(int(float(row["nt"])))
        except (TypeError, ValueError):
            pass  # blank or malformed column: keep the pre-column behaviour
    run_1stage = row.get("run_1stage", False)
    if "xbf16" in row:
        xbf16 = run_1stage and bool(int(row["xbf16"]))
    else:
        xbf16 = run_1stage and "blockscaleBf16" in str(row["kernelName1"])
    flat = int(row["flat"]) if "flat" in row and run_1stage else 0
    return nt, run_1stage, xbf16, flat


def _covers(segment, op) -> bool:
    return segment.covers[0] <= op <= segment.covers[1]


def _gemm1(structure):
    """The stage 1 of a 2-stage structure: it may start before GEMM1
    (`mxmoe.quant_gemm1` quantizes its own input)."""
    return next(s for s in structure if _covers(s, Op.GEMM1))


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
    gemm1 = _gemm1(structure)
    is_asm = kernel_names.ASM in gemm1.prefixes
    if is_asm:
        block_m = stage2_block_m(block_m)
    choices = []
    for segment in structure:
        if segment is gemm1:
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
    derived = fields.keys() & {"fuse_quant", "skip_inter_quant", "output_aux"}
    assert not derived, f"metadata_fields sets {sorted(derived)}, which boundaries set"
    fields.update(boundary_flags(plan))
    stage1 = plan.covering(Op.GEMM1)
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


# The keys of a stage 1's default params the resolver places itself; the rest are the
# stage 1's options.
_PLAN_DEFAULTS = frozenset({"kernel", "block_m", "ksplit", "nt"})


def _default_params(problem, structure) -> dict:
    """`{segment: default_params}` of a ranked structure; raises where one has none."""
    defaults = {segment: segment.default_params(problem) for segment in structure}
    missing = [segment.name for segment, params in defaults.items() if params is None]
    if missing:
        raise ValueError(f"{', '.join(missing)}: no default params for {problem}")
    return defaults


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

    file_logs = []

    def declined(reason):
        note(f"declined: {reason}; the legacy code runs this call")
        return Resolution(
            None, logs=tuple(file_logs), declined=reason, explain=tuple(explain or ())
        )

    if not plans:
        return declined("no feasible plan")
    keys = problem.log_keys()
    try:
        chosen = () if overrides.bypass_tuned() else tuned_plans.lookup(problem)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        chosen = ()
        file_logs.append(
            (WARNING, f"[fused_moe] moev2 tuned plan file not read: {error}")
        )
    for names, candidates in chosen:
        structure = next((s for s in plans if _names(s) == names), None)
        built = (
            build(problem, structure, candidates)
            if structure is not None
            else Resolution(None, declined="the structure is not feasible here")
        )
        if built.plan is not None:
            note(f"tuned plan file: selected {built.plan.describe()}")
            message = f"[fused_moe] moev2 tuned plan {_plan_name(structure)} for {keys}"
            return Resolution(
                built.plan,
                built.metadata,
                (*file_logs, (INFO, message)),
                explain=tuple(explain or ()),
            )
        name = STRUCTURE_SEPARATOR.join(names)
        message = f"[fused_moe] moev2 tuned plan {name} for {keys} not used"
        file_logs.append((WARNING, f"{message}: {built.declined}"))
    for _, message in file_logs:
        note(message)
    logs = list(file_logs)
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
        rejection = _row_rejection(problem, matched, cfg)
        if rejection is not None:
            is_decline, reason = rejection
            if is_decline:
                return declined(reason)
            logs.append((WARNING, reason))
            note(f"tuned row discarded: {reason}")
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
        defaults = _default_params(problem, structure)
        kernelName2 = ""
        one_stage = _one_stage(structure)
        run_1stage = one_stage is not None
        if run_1stage:
            first = defaults[one_stage]
            kernelName1, block_m, ksplit = first.kernel, first.block_m, first.ksplit
            run_1stage_xbf16, flat = first.xbf16, first.flat
        else:
            # Legacy logs its generic 2-stage heuristic whatever branch runs; the
            # stages' `default_params` replace it below.
            kernelName1 = ""
            run_1stage_xbf16, flat = False, 0
            dims = (q_type, s.m, s.topk, s.experts, s.inter_dim, s.model_dim)
            block_m, ksplit = default_block_m(*dims), default_ksplit(*dims)
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
        block_m = cfg["block_m"]
        use_non_temporal_load, run_1stage, run_1stage_xbf16, flat = _row_values(cfg)
        nt_override = overrides.nt_override()
        if nt_override != -1:
            use_non_temporal_load = bool(nt_override)
        ksplit = 0 if overrides.ksplit_disabled() else cfg["ksplit"]
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
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
    gemm1 = _gemm1(structure)
    gemm2 = next(s for s in structure if s.covers[0] == Op.GEMM2)
    kernelName2 = gemm2.stage2_kernel(problem, kernelName1, kernelName2)
    tag = f"({kernelName1=}, {kernelName2=})"
    logs.append(
        (
            INFO,
            f"[fused_moe] using 2stage {'default' if cfg is None else tag} for {keys} ",
        )
    )
    options = NO_OPTIONS
    if source != "tuned":
        first, second = defaults[gemm1], defaults[gemm2]
        kernelName1, kernelName2 = first.kernel, second.kernel
        block_m, ksplit, use_non_temporal_load = first.block_m, first.ksplit, first.nt
        options = Params({k: v for k, v in first.items() if k not in _PLAN_DEFAULTS})
        warning = gemm1.default_warning(problem)
        if warning is not None:
            logs.append((WARNING, warning))
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


# The `param_space` fields a candidate may set; each is a tuned-row column.
_CANDIDATE_FIELDS = frozenset({"kernel", "block_m", "ksplit", "nt", "xbf16", "flat"})


def tunable(segment) -> bool:
    """Whether `segment` runs a named kernel, chosen from its `param_space`."""
    return bool(segment.prefixes)


def candidate_row(structure, candidates) -> dict:
    """The tuned row naming `candidates`, `{segment name: Params}` with one
    `param_space` entry per tunable segment of `structure`: the columns dispatch reads.
    Raises where the candidates disagree on a shared column or set a field no column
    holds."""
    chosen = {s.name: candidates.get(s.name) for s in structure if tunable(s)}
    missing = sorted(name for name, c in chosen.items() if c is None)
    extra = sorted(candidates.keys() - chosen.keys())
    if missing or extra:
        raise ValueError(f"candidates missing for {missing}, not tunable: {extra}")
    unknown = sorted({f for c in chosen.values() for f in c} - _CANDIDATE_FIELDS)
    if unknown:
        raise ValueError(f"candidate fields with no tuned-row column: {unknown}")

    def shared(field, default):
        values = {c[field] for c in chosen.values() if field in c}
        if len(values) > 1:
            raise ValueError(f"the candidates disagree on {field}: {sorted(values)}")
        return next(iter(values), default)

    one_stage = len(structure) == 1 or _one_stage(structure) is not None
    if one_stage:
        (only,) = chosen.values()
        kernel1, kernel2 = only.kernel, ""
    else:
        kernel1 = chosen[_gemm1(structure).name].kernel
        gemm2 = next(s for s in structure if s.covers[0] == Op.GEMM2)
        kernel2 = chosen[gemm2.name].kernel
    return {
        "kernelName1": kernel1,
        "kernelName2": kernel2,
        "block_m": int(shared("block_m", BLOCK_SIZE_M)),
        "ksplit": int(shared("ksplit", 0)),
        "nt": int(bool(shared("nt", False))),
        "run_1stage": int(_one_stage(structure) is not None),
        "xbf16": int(bool(shared("xbf16", False))),
        "flat": int(shared("flat", 0)),
    }


def build(problem, structure, candidates) -> Resolution:
    """What dispatch runs for the tuned row naming `candidates` (`candidate_row`) on
    `structure`, one of `feasible_plans(problem, ...)`: the plan and its metadata, or
    declined with the reason dispatch would not run that row as tuned. The env
    overrides of a tuned row (`overrides`) are not applied. Declined too: a row
    `match_row` does not put on `structure` (a 1-stage kernel on the sort of the other
    `flat`), and a plan whose tunable segments do not run their candidate."""
    row = candidate_row(structure, candidates)

    def declined(reason):
        return Resolution(None, declined=reason)

    if _plain_1stage_row(row) and problem.activation.kind not in ONE_STAGE_ACTIVATIONS:
        return declined(f"no 1-stage kernel for {problem.activation.activation}")
    matched = match_row((structure,), row)
    if matched is not structure:
        return declined(f"dispatch would not put the row on this structure: {matched}")
    kernel1, block_m, ksplit = row["kernelName1"], row["block_m"], row["ksplit"]
    if len(structure) == 1:
        (whole,) = structure
        try:
            full_impl = resolve_fused_moe_impl(kernel1)
        except FusedMoeImplResolutionError as error:
            return declined(str(error))
        if full_impl is None:
            return declined(f"{kernel1!r} names no fused-MoE implementation")
        rejection = whole.check_row(problem, row, kernel1)
        if rejection is not None:
            return declined(f"discarded: {rejection.warning or 'expert_mask'}")
        plan = MoePlan(
            (SegmentChoice(whole, Params(kernel=kernel1, impl=full_impl)),),
            "tuned",
            Params(block_m=block_m, ksplit=ksplit),
        )
        return Resolution(
            plan, MOEMetadata(None, None, block_m, ksplit, full_impl=full_impl)
        )
    rejection = _row_rejection(problem, structure, row)
    if rejection is not None:
        is_decline, reason = rejection
        return declined(reason if is_decline else f"discarded: {reason}")
    nt, run_1stage, xbf16, flat = _row_values(row)
    if run_1stage:
        built = _one_stage_plan(
            problem,
            structure,
            "tuned",
            kernel1,
            block_m,
            ksplit,
            xbf16,
            run_1stage,
            flat,
        )
    else:
        gemm2 = next(s for s in structure if s.covers[0] == Op.GEMM2)
        kernel2 = gemm2.stage2_kernel(problem, kernel1, row["kernelName2"])
        built = _two_stage_plan(
            problem,
            structure,
            "tuned",
            kernel1,
            kernel2,
            block_m,
            ksplit,
            nt,
            run_1stage,
            NO_OPTIONS,
            row,
        )
    if isinstance(built, str):
        return declined(f"the chosen kernels disagree: {built}")
    plan, metadata = built
    for choice in plan.segments:
        if not tunable(choice.segment):
            continue
        params = Params(plan.params, **choice.params)
        reason = choice.segment.params_reason(problem, params)
        if reason is not None:
            return declined(f"{choice.segment.name} runs no candidate: {reason}")
    return Resolution(plan, metadata)
