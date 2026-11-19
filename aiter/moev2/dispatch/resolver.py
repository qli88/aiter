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

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import logging
import math
from dataclasses import dataclass

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.moe_limits import normalize_swiglu_limit

from ..execute.metadata import AsmMoeMode, MOEMetadata
from ..segments.asm import _retarget_asm_activation, stage2_block_m
from ..segments.base import NO_OPTIONS, Params
from ..segments.ck import _require_shuffled_gfx950_fp8
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
    before it. A segment that fits the chain but in no plan, because of its boundaries,
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


def _situv2_asm_row(problem, row) -> bool:
    """`row` is a 1-stage SiTUv2 ASM row on MXFP4 weights, which legacy
    `get_2stage_cfgs` discards on preshuffled weights (and raises on for a mixed
    shuffle, a gate/up layout other than separated, or runtime betas unshuffled)."""
    return bool(
        row.get("run_1stage", False)
        and str(row.get("kernelName1", "") or "").strip().startswith("_ZN5aiter")
        and problem.activation.activation == ActivationType.Situv2
        and problem.quant_type == QuantType.per_1x32
        and problem.w1.dtype == dtypes.fp4x2
    )


# The activations of the 1-stage ASM kernels; legacy discards other 1-stage rows.
_ASM_1STAGE_ACTIVATIONS = (
    ActivationType.Silu,
    ActivationType.Gelu,
    ActivationType.Situv2,
    ActivationType.Swiglu,
)


def _plain_asm_1stage_row(row) -> bool:
    """`row` names only an ASM 1-stage kernel (or none). Legacy checks any other
    1-stage row (Opus, FlyDSL, CK-Tile or whole-graph names) before its 1-stage
    activation and SwiGLU OAI discards, so those discards apply to plain rows only."""

    def family(cell):
        name = cell.strip() if isinstance(cell, str) else ""
        if not name:
            return None
        parsed = kernel_names.try_parse(name)
        return "unknown" if parsed is None else parsed.family

    return family(row.get("kernelName1", "")) in (None, kernel_names.ASM) and family(
        row.get("kernelName2", "")
    ) in (None, kernel_names.NULL)


def _swiglu_asm_row_unsupported(problem, row) -> str:
    """Why legacy `get_2stage_cfgs` discards 1-stage Swiglu `row` (the SwiGLU OAI FLAT
    ASM kernels: an explicit name, flat=1, no bias, limit 7.0, separated GGUU weights),
    or ""."""
    if not row.get("run_1stage", False) or problem.activation.activation != (
        ActivationType.Swiglu
    ):
        return ""
    kernel1 = row.get("kernelName1", "")
    if not isinstance(kernel1, str):  # a blank cell reads as NaN
        kernel1 = ""
    try:
        flat = int(row.get("flat", AsmMoeMode.SORTED))
    except (TypeError, ValueError):
        flat = AsmMoeMode.SORTED
    limit = problem.activation.swiglu_limit
    if not kernel1.strip():
        return "empty kernelName1 (FLAT kernels need an explicit name)"
    if flat != AsmMoeMode.NO_SORT:
        return f"flat={row.get('flat')} (FLAT kernels need flat=1)"
    if "bias1" in problem.features or "bias2" in problem.features:
        return "per-expert bias"
    if limit is not None and float(limit) != 7.0:
        return f"swiglu_limit={limit} (kernel uses 7.0)"
    if problem.gate_mode != GateMode.SEPARATED:
        return f"gate mode {problem.gate_mode.value!r}"
    if problem.layout.gate_up_interleave:
        return "gate/up-interleaved (GUGU) weights"
    return ""


def _discards_post_reduction(problem, row) -> bool:
    """Legacy `get_2stage_cfgs` discards post-reduction `row` for this layout (not
    separated gate/up on fully shuffled weights). False for rows an earlier legacy
    check may discard or raise on instead (Opus or FlyDSL names)."""
    if row.get("flat", AsmMoeMode.SORTED) != AsmMoeMode.POST_REDUCE:
        return False
    names = (str(row.get(k, "") or "").strip() for k in ("kernelName1", "kernelName2"))
    if any(n.startswith(("opus_", "flydsl_")) for n in names):
        return False
    return (
        problem.gate_mode != GateMode.SEPARATED
        or problem.layout.gate_up_interleave is True
        or not problem.fully_shuffled
    )


def _silu_limit_1stage(problem) -> bool:
    """Legacy `_fused_moe_impl` raises for a 1-stage plan with a finite SiLU limit."""
    limit = problem.activation.swiglu_limit
    return (
        problem.activation.activation == ActivationType.Silu
        and limit not in (None, 0, math.inf)
        and normalize_swiglu_limit(limit) is not None
    )


def match_row(plans, row):
    """The plan structure that runs tuned `row`, or a decline reason (str)."""
    if row.get("flat", AsmMoeMode.SORTED) == AsmMoeMode.POST_REDUCE:
        return "post-reduction tuned row"
    k1 = _owned_kernel(row.get("kernelName1", ""))
    if isinstance(k1, str):
        return k1
    if row.get("run_1stage", False):
        if k1 is None or not _plain_asm_1stage_row(row):
            return (
                f"1-stage row with kernelName1={k1}, "
                f"kernelName2={row.get('kernelName2', '')!r}"
            )
        try:
            mode = AsmMoeMode(int(row.get("flat", AsmMoeMode.SORTED)))
        except (TypeError, ValueError):  # legacy raises on the cell
            return f"1-stage row with flat={row.get('flat')!r}"
        try:
            int(row.get("xbf16", 0))
        except (TypeError, ValueError):  # legacy raises on the cell
            return f"1-stage row with xbf16={row.get('xbf16')!r}"
        flat = not mode.requires_host_sort
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
    flat=AsmMoeMode.SORTED,
):
    """The plan and the legacy `fused_moe_1stage` metadata of a 1-stage structure.
    `run_1stage`: the tuned row's value as legacy passes it on."""
    one_stage = _one_stage(structure)
    params = Params(kernel=kernelName1, xbf16=xbf16, flat=flat)
    choices = tuple(
        SegmentChoice(segment, params if segment is one_stage else Params())
        for segment in structure
    )
    plan = MoePlan(
        choices,
        source,
        Params(block_m=block_m, ksplit=ksplit),
        tuple(c.segment.produces(problem, c.params) for c in choices[:-1]),
    )
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
    """`row`: the tuned row the plan runs, None for a default."""
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
    plan = MoePlan(
        tuple(choices),
        source,
        Params(block_m=block_m, ksplit=ksplit),
        tuple(c.segment.produces(problem, c.params) for c in choices[:-1]),
    )
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


def _require_shuffled_ck_fp8(problem, logs, explain, emit):
    """Raise as the legacy code does, after the lines it has logged by then."""
    try:
        _require_shuffled_gfx950_fp8(
            problem.arch, problem.w1.dtype, problem.quant_type, problem.fully_shuffled
        )
    except RuntimeError:
        if emit:
            Resolution(None, logs=tuple(logs), explain=tuple(explain or ())).emit()
        raise


def resolve(problem, segments, emit_on_raise=True) -> Resolution:
    """The plan for `problem`, with the legacy log lines to emit.

    Where legacy raises, this emits the lines legacy logged before raising (unless
    `emit_on_raise` is False) and raises the same error.
    """
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
    logs = []
    cfg = tuned.lookup(problem)
    if cfg is None:
        note(f"tuned lookup: miss (tier {problem.shape.m})")
        matched = None
    else:
        note(
            f"tuned lookup: kernelName1={cfg.get('kernelName1')!r} "
            f"kernelName2={cfg.get('kernelName2')!r}"
        )
        situv2_asm = _situv2_asm_row(problem, cfg)
        if situv2_asm and problem.any_shuffled:
            if (
                not problem.fully_shuffled
                or problem.gate_mode != GateMode.SEPARATED
                or problem.layout.gate_up_interleave is True
            ):
                return declined("1-stage SiTUv2 ASM row legacy raises on")
            if problem.extras.get("tuned_table") is not None:
                # Legacy raises for a dedicated table without a usable row.
                return declined(
                    "1-stage SiTUv2 ASM row discarded for a dedicated table"
                )
            warning = (
                "[fused_moe] discarding 1-stage SiTUv2 ASM config for preshuffled "
                "MXFP4 weights; using the two-stage layout-compatible path"
            )
            logs.append((WARNING, warning))
            note(f"tuned row discarded: {warning}")
            cfg = None
        elif situv2_asm and (
            problem.activation.situ_beta != DEFAULT_SITUV2_BETA
            or problem.activation.situ_linear_beta != DEFAULT_SITUV2_LINEAR_BETA
        ):
            return declined("1-stage SiTUv2 ASM row with runtime betas (legacy raises)")
        elif _discards_post_reduction(problem, cfg):
            if problem.extras.get("tuned_table") is not None:
                # Legacy raises for a dedicated table.
                return declined("post-reduction row for an incompatible layout")
            warning = (
                "[fused_moe] discarding post-reduction config for incompatible "
                "weight layout; using default heuristics"
            )
            logs.append((WARNING, warning))
            note(f"tuned row discarded: {warning}")
            cfg = None
        activation = problem.activation.activation
        plain_1stage = (
            cfg is not None
            and cfg.get("run_1stage", False)
            and _plain_asm_1stage_row(cfg)
        )
        # A dedicated table is the FHMoE path (not ported), where legacy discards
        # these rows too and runs the default heuristics.
        if plain_1stage and activation not in _ASM_1STAGE_ACTIVATIONS:
            if problem.extras.get("tuned_table") is not None:
                return declined("1-stage row discarded for a dedicated table")
            warning = (
                f"[fused_moe] discarding 1-stage tuned config for unsupported "
                f"activation {activation}; using default heuristics"
            )
            logs.append((WARNING, warning))
            note(f"tuned row discarded: {warning}")
            cfg = None
        swiglu_unsupported = (
            plain_1stage
            and cfg is not None
            and _swiglu_asm_row_unsupported(problem, cfg)
        )
        if swiglu_unsupported:
            if problem.extras.get("tuned_table") is not None:
                return declined("1-stage row discarded for a dedicated table")
            warning = (
                f"[fused_moe] discarding SwiGLU OAI 1-stage asm config for {keys}: "
                f"unsupported {swiglu_unsupported}; using default heuristics"
            )
            logs.append((WARNING, warning))
            note(f"tuned row discarded: {warning}")
            cfg = None
        matched = None
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
        flat = AsmMoeMode.SORTED
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
        tuned_names = (
            str(cfg.get("kernelName1", "") or ""),
            str(cfg.get("kernelName2", "") or ""),
        )
        if not cfg.get("run_1stage", False) and any(
            "moe_ck2stages" in name for name in tuned_names
        ):
            _require_shuffled_ck_fp8(problem, logs, explain, emit_on_raise)
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
        if run_1stage:
            activation = problem.activation.activation
            retargeted = _retarget_asm_activation(kernelName1, activation)
            if retargeted != kernelName1:
                message = (
                    f"[fused_moe] 1-stage kernel {kernelName1!r} is for a "
                    f"different activation; using {retargeted!r} for {activation}"
                )
                logs.append((INFO, message))
                kernelName1 = retargeted
        if "xbf16" in cfg:
            run_1stage_xbf16 = run_1stage and bool(int(cfg["xbf16"]))
        else:
            run_1stage_xbf16 = run_1stage and "blockscaleBf16" in str(kernelName1)
        if "flat" in cfg:
            flat = AsmMoeMode(int(cfg["flat"])) if run_1stage else AsmMoeMode.SORTED
        else:
            flat = AsmMoeMode.SORTED
        if not problem.any_shuffled and not run_1stage:
            message = (
                f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
                "Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
                "Running with preshuffle_off may produce incorrect results."
            )
            logs.append((WARNING, message))
        source = "tuned"
    if run_1stage:
        if _silu_limit_1stage(problem):
            return declined("1-stage plan with a SiLU limit (legacy raises)")
        tag = f"({kernelName1=}, {kernelName2=})"
        xbf16 = " xbf16" if run_1stage_xbf16 else ""
        logs.append(
            (
                INFO,
                f"[fused_moe] using 1stage{xbf16} {'default' if cfg is None else tag} for {keys} ",
            )
        )
        plan, metadata = _one_stage_plan(
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
    names = [s.name for s in structure]
    if "ck.gemm1" in names and "ck.gemm2" in names:
        _require_shuffled_ck_fp8(problem, logs, explain, emit_on_raise)
    default = gemm1.default_kernels(problem) if source != "tuned" else None
    options = NO_OPTIONS
    if default is not None:
        kernelName1, kernelName2 = default.kernel1, default.kernel2
        block_m, ksplit = default.block_m, default.ksplit
        options = default.options
        if default.warning is not None:
            logs.append((WARNING, default.warning))
    plan, metadata = _two_stage_plan(
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
    note(f"selected {plan.describe()}")
    return Resolution(plan, metadata, tuple(logs), explain=tuple(explain or ()))
