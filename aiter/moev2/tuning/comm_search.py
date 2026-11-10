# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The comm_fused search (plan §4.6) for one `collective.CommProblem`: `tune`.

What it compares, end to end on every rank of a TP group (`collective`): the local plan
of the key with its stage 2 replaced by a `comm_fused.stage2_rs` runner, against the
ordinary call: the plan dispatch resolves for the key today, then the TP collective
(`collective.ordinary_runner`). The runner and the local plan are chosen together.

Local plans: level 1 of the local key (`search.level1`, single-GPU tasks); its
combinations split into classes by structure, `block_m` and the intermediate layout
the stage 2 kernel writes (its name has `_moe2_layout_`, which the runners read as
`sorted_input`). A class keeps its first `local_finalists` combinations by summed
stage time with distinct stage-1 candidates (each the fastest stage 2 it builds with),
the plan dispatch resolves today first in its class. Structures without a stage 2
segment to replace take no runner.

Runners (`runner_candidates`): the families of the comm mode ("ar": mega, window;
"rs": direct, window), from `seeds` (the shipped winners of the shape, any GPU model,
at token counts within `seed_span` times M either way, retargeted to M) and the
defaults of their families (window has none; every family's when no seed is that
close), each with `sort_block_m` set to the class's `block_m`, the `axes` given for its
family applied on top (`{family: {field: values}}`, every combination: the legacy
tuner's `--axis`). A config that does not construct is dropped. A seed far from M can
take minutes a call there (a small-M megakernel config at M=4096), each a group timeout.

Level 1: the ordinary call, and per class each runner with the class's first local
plan (collective, end to end); the class keeps its `top_k` fastest passing runners and
any within `eps` of its fastest. Level 2: the class's other local plans with each kept
runner. The passing runs within max(`tie_rel` of the fastest, `tie_abs_us`) of the
fastest (at most `max_rivals`, the current call first, and at most
`collective.MAX_RUNNER_SETS` runner configs) are timed again in turn
(`collective.CommRivals`) and rank by that; among those still within the margin, the
ordinary call, then lower CoV, goes first: a runner must beat it beyond the noise.

The current call is the ordinary one, unless a seed is this GPU model's winner at M
(the fastest such seed, as the runtime takes the fastest of duplicate rows) and the
plan dispatch resolves is searched and heads its class (then that runner with that
plan, as `CommFusedMoeRuntime` runs it today). Outcomes: one per local structure and
runner family (the ordinary call its own), its best run, ranked; failed ones after.
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass

from .. import paths
from ..dispatch import resolver
from ..dispatch.ops import Op
from ..segments.base import Params
from .candidates import NO_FILTERS
from .collective import (
    MAX_RUNNER_SETS,
    CommRivals,
    CommTask,
    runner_config,
    runner_spec,
)
from .search import (
    CURRENT_NOTE,
    Level1,
    Outcome,
    Settings,
    _failed,
    _names,
    current,
    keep,
    level1,
)

RUNNER = "comm_fused.stage2_rs"  # the runner's entry in a row's candidates
ORDINARY_NOTE = "ordinary: the local plan, then the TP collective"
FAMILIES = {"ar": ("mega", "window"), "rs": ("direct", "window")}
DEFAULTLESS = frozenset({"window"})


@dataclass(frozen=True)
class CommSettings(Settings):
    local_finalists: int = 4  # local plans per class
    seed_span: int = 4  # seeds from token counts within this factor of M (0: any)


COMM_DEFAULTS = CommSettings()


@dataclass(frozen=True)
class Seed:
    """A runner to start from: `spec` (`collective.runner_spec`), and the winner row
    it comes from (None for a family default)."""

    spec: tuple
    token: int | None = None
    gpu_model: str | None = None
    us: float | None = None


def runner_params(spec) -> Params:
    """`spec` as a row's candidate: the family and the config's non-default params."""
    family, params = spec
    return Params(dict(params), family=family)


def retarget(problem, family, values, block_m):
    """The spec of a runner of `family` with `values` for `problem` and a local plan
    of `block_m` (`sort_block_m` set, `sorted_input` left to the runtime), or None
    when its config does not construct."""
    values = {k: v for k, v in dict(values).items() if k != "sorted_input"}
    values["sort_block_m"] = block_m
    try:
        config = runner_config(problem, (family, tuple(sorted(values.items()))))
    except (TypeError, ValueError, AssertionError):
        return None
    return runner_spec(config)


def runner_candidates(
    problem, block_m, seeds, axes=None, span=COMM_DEFAULTS.seed_span
) -> list:
    """The runner specs of `problem` for a local plan of `block_m` (see the module
    doc), in seed order, without repeats."""
    families = FAMILIES[problem.comm]
    m = problem.local.shape.m
    close = [
        s
        for s in seeds
        if s.spec[0] in families
        and (s.token is None or not span or max(s.token, m) <= span * min(s.token, m))
    ]
    defaults = {s.spec[0] for s in close} if close else set(families)
    starts = [s.spec for s in close]
    starts += [(f, ()) for f in families if f in defaults and f not in DEFAULTLESS]
    found = {}
    for family, params in starts:
        grid = [dict(params)]
        for field, values in (axes or {}).get(family, {}).items():
            grid = [{**g, field: v} for g in grid for v in values]
        for values in grid:
            spec = retarget(problem, family, values, block_m)
            if spec is not None:
                found.setdefault(spec, None)
    return list(found)


def stage2_replaceable(structure) -> bool:
    return any(s.covers[0] == Op.GEMM2 for s in structure)


def layout(problem, structure, combo):
    """`(block_m, sorted intermediate)` of a combination, or None when it does not
    build."""
    built = resolver.build(problem, structure, combo)
    if built.plan is None or built.metadata.run_1stage:
        return None
    keywords = getattr(built.metadata.stage2, "keywords", None) or {}
    name = keywords.get("kernelName") or keywords.get("kernelName2") or ""
    return int(built.metadata.block_m), "_moe2_layout_" in str(name)


def local_classes(problem, state: Level1, today, settings):
    """`{(structure index, block_m, sorted): [combo, ...]}`: the local plans per class
    (`today`: the current plan as `(index, combo)`, first in its class)."""
    classes = defaultdict(list)
    stage1 = defaultdict(list)  # class -> the stage-1 candidates taken

    def add(i, combo, forced=False):
        found = layout(problem, state.found[i].structure, combo)
        if found is None:
            return
        key = (i, *found)
        first = {
            name: c
            for name, c in combo.items()
            if not any(
                s.name == name and s.covers[0] == Op.GEMM2
                for s in state.found[i].structure
            )
        }
        if combo in classes[key]:
            return
        if not forced and (
            first in stage1[key] or len(classes[key]) >= settings.local_finalists
        ):
            return
        classes[key].append(combo)
        stage1[key].append(first)

    if today is not None and stage2_replaceable(state.found[today[0]].structure):
        add(*today, forced=True)
    for i, queue in state.queues.items():
        if i in state.single or not stage2_replaceable(state.found[i].structure):
            continue
        for _, combo in queue:
            add(i, combo)
    return dict(classes)


@dataclass
class _Run:
    timing: object  # measure.Timing
    index: int | None  # structure index; None: the ordinary call's plan unsearched
    combo: dict | None
    spec: tuple | None  # None: the ordinary call

    def group(self, names):
        family = None if self.spec is None else self.spec[0]
        return names, family


def _today_runner(problem, seeds):
    m = problem.local.shape.m
    return min(
        (s for s in seeds if s.token == m and s.gpu_model == problem.gpu_model),
        key=lambda s: float("inf") if s.us is None else s.us,
        default=None,
    )


def tune(
    problem,
    path,
    evaluate_local,
    evaluate_group,
    seeds=(),
    axes=None,
    settings=COMM_DEFAULTS,
    filters=NO_FILTERS,
    log=None,
):
    """The `Outcome` of each local structure and runner family of `problem` (a
    `CommProblem`; `path` the local plans' registered path), winner first, then the
    other ranked ones, then the failed ones. `evaluate_local` runs `task.Task`s (a
    `Pool`), `evaluate_group` `collective` tasks (a `Group`)."""
    log = log or (lambda line: None)
    local = problem.local
    state = level1(local, path, evaluate_local, settings, filters, log)
    if not isinstance(state, Level1):
        return state
    found = state.found
    failed = list(state.failed)
    why = defaultdict(list)  # group -> reasons
    today, not_today = current(local, path, found)
    resolved = resolver.resolve(local, paths.registered()[path].segments)
    if resolved.plan is None:
        return failed + [
            Outcome((), reason=f"the key's plan is declined: {resolved.declined}")
        ]
    today_names = tuple(c.segment.name for c in resolved.plan.segments)

    classes = local_classes(local, state, today, settings)
    for i, sc in enumerate(found):
        if i not in state.queues:
            continue
        if not stage2_replaceable(sc.structure):
            why[_names(sc.structure), "*"].append("no stage 2 to replace")
        elif i in state.single:
            why[_names(sc.structure), "*"].append(
                "one tunable segment: level 1 does not rank its plans"
            )

    def comm_task(i, combo, spec):
        structure = None if i is None else _names(found[i].structure)
        candidates = None if combo is None or i is None else tuple(combo.items())
        return CommTask(
            problem,
            path,
            structure,
            candidates,
            spec,
            iters=settings.iters,
            warmup=settings.warmup,
            seed=settings.seed,
        )

    def names_of(i):
        return today_names if i is None else _names(found[i].structure)

    runs = []

    def collect(index, tasks):
        for (i, combo, spec), result in zip(index, evaluate_group(tasks)):
            group = (names_of(i), None if spec is None else spec[0])
            if result.ok:
                runs.append(_Run(result.e2e, i, combo, spec))
            else:
                why[group].append(result.reason)

    # The ordinary call: today's plan (searched or not), then the collective.
    ordinary = (None, None, None) if today is None else (today[0], today[1], None)
    today_seed = _today_runner(problem, seeds)
    today_fused = None

    # Level 1: every runner with each class's first local plan.
    tasks, index = [comm_task(*ordinary)], [ordinary]
    candidates = {}
    for key, combos in classes.items():
        i, block_m, _ = key
        candidates[key] = runner_candidates(
            problem, block_m, seeds, axes, settings.seed_span
        )
        if not candidates[key]:
            why[names_of(i), "*"].append(f"no runner for block_m={block_m}")
        for spec in candidates[key]:
            tasks.append(comm_task(i, combos[0], spec))
            index.append((i, combos[0], spec))
        if (
            today is not None
            and today_seed is not None
            and combos[0] == today[1]
            and i == today[0]
        ):
            spec = retarget(problem, *today_seed.spec, block_m)
            if spec in candidates[key]:
                today_fused = (i, combos[0], spec)
    log(f"level 1: {len(tasks)} collective runs ({len(classes)} local classes)")
    collect(index, tasks)

    # Level 2: the classes' other local plans with their kept runners.
    tasks, index = [], []
    for key, combos in classes.items():
        i = key[0]
        timed = [
            (r.timing.median_us, r.spec)
            for r in runs
            if r.index == i and r.combo == combos[0] and r.spec in candidates[key]
        ]
        for combo, (_, spec) in itertools.product(combos[1:], keep(timed, settings)):
            tasks.append(comm_task(i, combo, spec))
            index.append((i, combo, spec))
    if tasks:
        log(f"level 2: {len(tasks)} collective runs")
        collect(index, tasks)

    current_run = today_fused or ordinary

    def is_current(r):
        return (r.index, r.combo, r.spec) == current_run

    runs.sort(key=lambda r: r.timing.median_us)
    notes = defaultdict(list)
    if len(runs) > 1:
        fastest = runs[0].timing.median_us
        margin = max(settings.tie_rel * fastest, settings.tie_abs_us)
        near = [r for r in runs if r.timing.median_us <= fastest + margin]
        tied, specs = [], set()
        for r in [r for r in near if is_current(r)] + [
            r for r in near if not is_current(r)
        ]:
            if len(tied) == settings.max_rivals:
                break
            if r.spec is not None and r.spec not in specs:
                if len(specs) == MAX_RUNNER_SETS:
                    continue
                specs.add(r.spec)
            tied.append(r)
        tied.sort(key=lambda r: r.timing.median_us)
        if len(tied) > 1:
            log(f"near-tie of {len(tied)} runs: timed again in turn")
            rivals = CommRivals(
                problem,
                path,
                tuple(
                    (
                        None if r.index is None else _names(found[r.index].structure),
                        (
                            None
                            if r.combo is None or r.index is None
                            else tuple(r.combo.items())
                        ),
                        r.spec,
                    )
                    for r in tied
                ),
                rounds=settings.rounds,
                iters=settings.iters,
                warmup=settings.warmup,
                seed=settings.seed,
            )
            (results,) = evaluate_group([rivals])
            rest = [r for r in runs if all(r is not t for t in tied)]
            whole = None
            if not isinstance(results, list):
                whole = results.reason
            elif len(results) != len(tied):
                whole = f"{len(results)} results for {len(tied)} plans"
            elif not any(r.ok for r in results):
                whole = results[0].reason
            if whole is not None:
                log(f"timing the near-tie again failed: {whole}")
                for r in tied:
                    notes[r.group(names_of(r.index))].append(
                        f"timing again in turn failed: {whole}"
                    )
                kept = list(tied)
            else:
                kept = []
                for r, result in zip(tied, results):
                    group = r.group(names_of(r.index))
                    if result.ok:
                        kept.append(_Run(result.e2e, r.index, r.combo, r.spec))
                        notes[group].append(
                            f"timed again in turn with {len(tied) - 1} more"
                        )
                    else:
                        why[group].append(f"timed again: {result.reason}")
                kept.sort(key=lambda r: r.timing.median_us)
            if kept:
                fastest = kept[0].timing.median_us
                margin = max(settings.tie_rel * fastest, settings.tie_abs_us)
                close = [r for r in kept if r.timing.median_us <= fastest + margin]
                close.sort(key=lambda r: (r.spec is not None, r.timing.cov))
                runs = close + kept[len(close) :] + rest
            else:
                runs = rest

    current_us = next((r.timing.median_us for r in runs if is_current(r)), None)
    current_group = (
        names_of(current_run[0]),
        None if current_run[2] is None else current_run[2][0],
    )
    shipped = next((s for s in seeds if s.token == local.shape.m), None)
    best, order = {}, []
    for r in runs:
        group = r.group(names_of(r.index))
        if group not in best:
            best[group] = r
            order.append(group)
    outcomes = []
    for rank, group in enumerate(order, 1):
        r = best[group]
        combo = dict(r.combo) if r.combo is not None else None
        if combo is not None and r.spec is not None:
            combo[RUNNER] = runner_params(r.spec)
        group_notes = []
        if r.spec is None:
            group_notes.append(ORDINARY_NOTE)
            if r.combo is None:
                group_notes.append(f"its plan is not searched ({not_today})")
        if is_current(r):
            group_notes.append(CURRENT_NOTE)
        elif rank == 1 and current_us is not None:
            group_notes.append(f"{CURRENT_NOTE}: {current_us:.1f} us")
        elif rank == 1:
            reasons = why.get(current_group, [])
            group_notes.append(_failed(reasons, f"{CURRENT_NOTE}: failed"))
        if rank == 1 and shipped is not None and shipped is not today_seed:
            group_notes.append(
                f"shipped {shipped.gpu_model} winner at M={shipped.token}: "
                f"{shipped.spec[0]}, {shipped.us} us (its stage 2 and collective)"
            )
        group_notes += notes[group]
        outcomes.append(
            Outcome(
                group[0],
                combo,
                r.timing,
                None,
                rank,
                notes="; ".join(dict.fromkeys(group_notes)),
            )
        )
    for i, sc in enumerate(found):
        if i not in state.queues and i not in state.refused:
            reasons = state.why[i] or [f"declined: {d}" for d in state.declined[i]]
            failed.append(
                Outcome(
                    _names(sc.structure), reason=_failed(reasons, "no level 1 result")
                )
            )
    for (names, family), reasons in why.items():
        if (names, family) in best:
            continue
        if family == "*" and any(g[0] == names for g in best):
            continue
        label = "ordinary" if family is None else family
        failed.append(
            Outcome(names, reason=_failed(reasons, f"{label}: no passing run"))
        )
    return outcomes + failed
