# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The two-level search (plan §4.2) for one problem on one path: `tune`.

Level 1, for a structure with a stage-1 and a stage-2 tunable segment: per `block_m`
and `nt` its candidates share (a group), a base combination that builds and passes (the
first of `base_tries`, in `base_order`), then every other candidate of each
segment with the other segment at the base, timing that segment's stage launch in
context (`measure` with `segments`, no end-to-end run; the output is still checked).
A segment keeps its `top_k` fastest passing candidates and any within `eps` of its
fastest. The kept candidates' combinations rank by summed stage time, across the
groups of the structure; the fastest go to level 2 (`finalists_graph` when the
deployment mode is graph, else `finalists_eager`). Level 1 runs per structure: a
segment in several structures is measured in each.

A structure with one tunable segment (a 1-stage kernel, a whole-graph implementation)
sends every candidate to level 2.

The plan dispatch resolves for the problem today (a tuned row, or the default plan)
goes to level 2 as well, besides the finalists, when its structure and candidates are
searched and build what dispatch runs. The structure's result notes it when it is that
plan; else the winner's notes give its time, or why it did not run or failed.

Level 2: each finalist end to end in the deployment mode, its output checked. A
structure whose finalists fail takes its next combinations, at most as many more
tries. The passing level-2 runs of every structure within max(`tie_rel` of the
fastest, `tie_abs_us`) of the fastest (at most `max_rivals`: the current plan, then
each structure's fastest before any structure's second, and so on) are timed again
in turn (`task.Rivals`) and rank by that, ahead of the others (which keep their
level-2 times); among those still within the margin of the fastest, fewer segments,
then lower CoV, go first. A run failing that is left out (noted on its structure's
result), and a structure left with no run is dropped with the reason; when the run
fails for all of them, they keep their level-2 times, with a note. A structure's
first run in this order is its result, and the structures rank by it. A structure
without level 1 whose only run is the current plan says so.

A combination `resolver.build` declines is never measured; a structure with nothing
measured gives the declines as its reason. `evaluate(tasks)` runs
`tuning.task` tasks and returns their results in order (a `Pool.map` in the CLI).
"""

import dataclasses
import functools
import itertools
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from .. import paths
from ..dispatch import resolver
from ..dispatch.ops import Op
from ..dispatch.tuned_plans import STRUCTURE_SEPARATOR, structure_names
from ..segments.base import Params
from ..segments.mxmoe import _is_mxfp4_inline_sort
from .candidates import NO_FILTERS, plan_candidates, structures
from .inputs import unsupported
from .measure import deploy_mode
from .task import Rivals, Task


@dataclass(frozen=True)
class Settings:
    top_k: int = 3
    eps: float = 0.03
    # Level 2 per structure; graph mode is where stages timed alone mislead most.
    finalists_graph: int = 12
    finalists_eager: int = 4
    tie_rel: float = 0.02
    tie_abs_us: float = 0.5
    base_tries: int = 4
    rounds: int = 4  # of the interleaved re-measurement
    iters: int = 50
    warmup: int = 3
    seed: int = 0
    max_rivals: int = 8  # level-2 runs timed again in turn, at most


@dataclass
class Outcome:
    """A structure of the problem: its best plan, or the reason it has none."""

    structure: tuple  # segment names; () when the problem is not tuned at all
    candidates: dict | None = None  # {segment name: Params}
    e2e: object = None  # measure.Timing
    sum_us: float | None = None  # level 1: the summed stage times
    rank: int | None = None  # 1: the winner
    reason: str | None = None
    notes: str = ""


# Combinations of a group tried for a base that `build` accepts (about 4 ms each).
BASE_BUILD_LIMIT = 1000

INLINE_SORT_NOTE = (
    "inline sort: a call it cannot run (expert mask, bias, pads, caller scales) runs "
    "the key's default plan (todo 2.35)"
)


def stage(segment) -> str | None:
    """The stage launch (`hooks.record_kernel` name) of a tunable segment covering
    one GEMM, else None."""
    first, last = segment.covers
    gemm1, gemm2 = first <= Op.GEMM1 <= last, first <= Op.GEMM2 <= last
    if gemm1 != gemm2:
        return "stage1" if gemm1 else "stage2"
    return None


def groups(candidates):
    """`((block_m, nt), {segment name: candidates})` for each `block_m` and `nt` every
    segment of `candidates` has a candidate for (a field a candidate leaves out agrees
    with any value)."""
    values = defaultdict(set)
    for cs in candidates.values():
        for c in cs:
            for field in ("block_m", "nt"):
                if field in c:
                    values[field].add(c[field])
    for bm, nt in itertools.product(
        sorted(values["block_m"]) or [None], sorted(values["nt"]) or [None]
    ):
        lists = {
            name: [
                c for c in cs if c.get("block_m", bm) == bm and c.get("nt", nt) == nt
            ]
            for name, cs in candidates.items()
        }
        if all(lists.values()):
            yield (bm, nt), lists


def base_order(lists):
    """Index tuples into `lists`' values, the order bases are tried in: first the k-th
    candidate of every segment (wrapping around), so each try changes every segment,
    then the other combinations."""
    sizes = [len(cs) for cs in lists.values()]
    diagonal = [tuple(k % n for n in sizes) for k in range(max(sizes))]
    yield from diagonal
    seen = set(diagonal)
    for indices in itertools.product(*(range(n) for n in sizes)):
        if indices not in seen:
            yield indices


def keep(timed, settings) -> list:
    """The `(us, candidate)` of `timed` level 1 keeps, fastest first."""
    ranked = sorted(timed, key=lambda t: t[0])
    if not ranked:
        return []
    best = ranked[0][0]
    return [
        t
        for i, t in enumerate(ranked)
        if i < settings.top_k or t[0] <= best * (1 + settings.eps)
    ]


def _names(structure) -> tuple:
    return tuple(s.name for s in structure)


def _failed(reasons, what) -> str:
    if not reasons:
        return what
    return f"{what}: {reasons[0]}" + (
        f" (and {len(reasons) - 1} more)" if len(reasons) > 1 else ""
    )


DEFAULTS = Settings()

CURRENT_NOTE = "the current plan"


def current(problem, path, found):
    """`((structure index, combo), None)`: the plan dispatch resolves for `problem`
    today, as one of `found` (`StructureCandidates`) and its candidates; else
    `(None, why not)`."""
    resolved = resolver.resolve(problem, paths.registered()[path].segments)
    plan = resolved.plan
    if plan is None:
        return None, f"declined: {resolved.declined}"
    names = tuple(c.segment.name for c in plan.segments)
    i = next((i for i, sc in enumerate(found) if _names(sc.structure) == names), None)
    if i is None:
        return None, f"{STRUCTURE_SEPARATOR.join(names)} is not searched"
    combo = plan_candidates(problem, plan)
    if isinstance(combo, str):
        return None, f"{combo} runs no listed candidate"
    for name, c in combo.items():
        if c not in found[i].candidates[name]:
            return None, f"{name} {c.get('kernel')} is filtered out"
    built = resolver.build(problem, found[i].structure, combo)
    if built.plan is None:
        return None, f"declined: {built.declined}"
    if _runs(problem, built) != _runs(problem, resolved):
        return None, "its candidates build another plan than dispatch runs"
    return (i, combo), None


def _plain(value, names):
    if isinstance(value, functools.partial):
        func = getattr(value.func, "__qualname__", repr(value.func))
        return (func, _plain(value.args, names), _plain(value.keywords, names))
    if isinstance(value, Mapping):
        return tuple(sorted((k, _plain(v, names)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_plain(v, names) for v in value)
    if callable(value):
        return "<callable>"
    if isinstance(value, str):
        return names.get(value, value)
    return value


def _runs(problem, resolution):
    """What `resolution` runs: its segments with their effective kernels, boundaries
    and metadata (the executor reads no params) as plain values, kernel names replaced
    by the kernels that run."""
    plan, names = resolution.plan, {}
    for c in plan.segments:
        if resolver.tunable(c.segment) and "kernel" in c.params:
            full = Params(plan.params, **c.params)
            names[c.params["kernel"]] = c.segment.effective_kernel(problem, full)
    metadata = resolution.metadata
    fields = {}
    if metadata is not None:
        fields = {
            f.name: getattr(metadata, f.name) for f in dataclasses.fields(metadata)
        }
        # The staged launches only test ksplit > 1; a whole graph gets it as is.
        if metadata.full_impl is None and fields["ksplit"] <= 1:
            fields["ksplit"] = 0
    return (
        tuple(
            (c.segment.name, names.get(c.params.get("kernel"))) for c in plan.segments
        ),
        plan.boundaries,
        tuple((k, _plain(v, names)) for k, v in fields.items()),
    )


def tune(problem, path, evaluate, settings=DEFAULTS, filters=NO_FILTERS, log=None):
    """The `Outcome` of each structure of `problem` on `path` (a registered path name),
    winner first, then the other ranked ones, then the failed ones."""
    log = log or (lambda line: None)
    reason = unsupported(problem)
    if reason is not None:
        return [Outcome((), reason=f"unsupported: {reason}")]
    found, skipped = structures(problem, paths.registered()[path].segments, filters)
    failed = [
        Outcome(structure_names(name), reason=f"skipped: {why}")
        for name, why in skipped.items()
    ]

    def task(structure, combo, **kwargs):
        return Task(
            problem,
            path,
            _names(structure),
            tuple(combo.items()),
            iters=settings.iters,
            warmup=settings.warmup,
            seed=settings.seed,
            **kwargs,
        )

    def level1_task(structure, combo, stages):
        return task(structure, combo, segments=stages, e2e=False)

    declined = defaultdict(list)

    def builds(i, combo) -> bool:
        built = resolver.build(problem, found[i].structure, combo)
        if built.plan is None and built.declined not in declined[i]:
            declined[i].append(built.declined)
        return built.plan is not None

    queues, single, refused, why = {}, set(), set(), defaultdict(list)
    level1 = []  # (structure index, group lists, {segment name: stage})
    for i, sc in enumerate(found):
        tunables = [s for s in sc.structure if resolver.tunable(s)]
        stages = {s.name: stage(s) for s in tunables}
        if len(tunables) <= 1:
            single.add(i)
            queues[i] = [(None, combo) for combo in sc.combinations()]
        elif sorted(map(str, stages.values())) == ["stage1", "stage2"]:
            found_groups = [(i, lists, stages) for _, lists in groups(sc.candidates)]
            if not found_groups:
                why[i].append("no block_m and nt every tunable segment has")
            level1 += found_groups
        else:
            refused.add(i)
            failed.append(
                Outcome(
                    _names(sc.structure),
                    reason=f"no level 1 for tunable segments {stages}",
                )
            )

    # Level 1: a passing base per group (one try per group at a time), then each
    # candidate against it.
    def built_combinations(i, lists):
        for indices in itertools.islice(base_order(lists), BASE_BUILD_LIMIT):
            combo = {name: lists[name][k] for name, k in zip(lists, indices)}
            if builds(i, combo):
                yield combo

    bases = {}
    trials = {j: built_combinations(i, lists) for j, (i, lists, _) in enumerate(level1)}
    for _ in range(settings.base_tries):
        tasks, index = [], []
        for j, (i, _, stages) in enumerate(level1):
            combo = next(trials[j], None) if j not in bases else None
            if combo is not None:
                tasks.append(
                    level1_task(found[i].structure, combo, tuple(stages.values()))
                )
                index.append((j, combo))
        if not tasks:
            break
        log(f"level 1: {len(tasks)} base runs")
        for (j, combo), result in zip(index, evaluate(tasks)):
            if result.ok:
                bases[j] = (combo, result)
            else:
                why[level1[j][0]].append(f"base: {result.reason}")
    timed = defaultdict(list)  # (group, segment name) -> [(us, candidate)]
    tasks, index = [], []
    for j, (combo, result) in bases.items():
        i, lists, stages = level1[j]
        for name, cs in lists.items():
            for c in cs:
                if c == combo[name]:
                    timed[j, name].append((result.stages[stages[name]].median_us, c))
                    continue
                trial = {**combo, name: c}
                if builds(i, trial):
                    tasks.append(
                        level1_task(found[i].structure, trial, (stages[name],))
                    )
                    index.append((j, name, c, stages[name]))
    log(f"level 1: {len(tasks)} segment runs")
    for (j, name, c, stage_), result in zip(index, evaluate(tasks)):
        if result.ok:
            timed[j, name].append((result.stages[stage_].median_us, c))
    sums = defaultdict(list)
    for j in bases:
        i, lists, _ = level1[j]
        kept = [keep(timed[j, name], settings) for name in lists]
        for chosen in itertools.product(*kept):
            combo = dict(zip(lists, (c for _, c in chosen)))
            sums[i].append((sum(us for us, _ in chosen), combo))
    for i, ranked in sums.items():
        queues[i] = sorted(ranked, key=lambda t: t[0])

    # Level 2: finalists end to end, refilled from the queue while some fail, and
    # the current plan.
    finalists = (
        settings.finalists_graph
        if deploy_mode(problem) == "graph"
        else settings.finalists_eager
    )
    today, not_today = current(problem, path, found)
    if today is not None and today[0] in refused:
        today, not_today = None, "its structure has no level 1"
    passed = defaultdict(list)  # structure index -> [(Timing, combo, sum)]
    ran = defaultdict(list)  # structure index -> combos run end to end
    position, tries, extra = defaultdict(int), defaultdict(int), defaultdict(int)
    pending = [today] if today is not None else []
    while True:
        tasks, index = [], []
        for i, combo in pending:
            built = resolver.build(problem, found[i].structure, combo)
            if built.plan is None:
                today, not_today = None, f"declined: {built.declined}"
                continue
            total = next((t for t, c in queues.get(i, ()) if c == combo), None)
            ran[i].append(combo)
            tasks.append(task(found[i].structure, combo))
            index.append((i, total, combo, True))
        pending = []
        for i, queue in queues.items():
            limit = len(queue) if i in single else 2 * finalists
            need = limit if i in single else finalists - len(passed[i]) + extra[i]
            while need > 0 and position[i] < len(queue) and tries[i] < limit:
                total, combo = queue[position[i]]
                position[i] += 1
                if combo in ran[i] or not builds(i, combo):
                    continue
                tries[i] += 1
                need -= 1
                ran[i].append(combo)
                tasks.append(task(found[i].structure, combo))
                index.append((i, total, combo, False))
        if not tasks:
            break
        log(f"level 2: {len(tasks)} end-to-end runs")
        for (i, total, combo, is_today), result in zip(index, evaluate(tasks)):
            if result.ok:
                passed[i].append((result.e2e, combo, total))
                extra[i] += is_today
            else:
                why[i].append(result.reason)
                if is_today:
                    today, not_today = None, f"failed: {result.reason}"

    # (structure index, Timing, combo, sum), fastest first.
    runs = sorted(
        ((i, *run) for i, rs in passed.items() for run in rs),
        key=lambda r: r[1].median_us,
    )
    notes, retimed, failures, today_failed = defaultdict(list), {}, {}, None
    if len(runs) > 1:
        fastest = runs[0][1].median_us
        margin = max(settings.tie_rel * fastest, settings.tie_abs_us)
        near = defaultdict(list)  # structure index -> its runs near the fastest
        for r in runs:
            if r[1].median_us <= fastest + margin:
                near[r[0]].append(r)
        # The current plan first, then each structure's k-th fastest before any
        # structure's (k+1)-th.
        tied = [r for rs in near.values() for r in rs if (r[0], r[2]) == today]
        tied += [
            rs[k]
            for k in range(len(runs))
            for rs in near.values()
            if k < len(rs) and (rs[k][0], rs[k][2]) != today
        ]
        tied = sorted(tied[: settings.max_rivals], key=lambda r: r[1].median_us)
        if len(tied) > 1:
            log(f"near-tie of {len(tied)} level-2 runs: timed again in turn")
            rivals = Rivals(
                problem,
                path,
                tuple(
                    (_names(found[i].structure), tuple(combo.items()))
                    for i, _, combo, _ in tied
                ),
                rounds=settings.rounds,
                iters=settings.iters,
                warmup=settings.warmup,
                seed=settings.seed,
            )
            (results,) = evaluate([rivals])
            chosen = {id(r) for r in tied}
            rest = [r for r in runs if id(r) not in chosen]
            if not isinstance(results, list):
                whole = results.reason
            elif len(results) != len(tied):
                whole = f"{len(results)} results for {len(tied)} plans"
            elif not any(r.ok for r in results):
                whole = results[0].reason
            else:
                whole = None
            if whole is not None:
                log(f"timing the near-tie again failed: {whole}")
                for i in {r[0] for r in tied}:
                    notes[i].append(f"timing again in turn failed: {whole}")
                kept = list(tied)
            else:
                kept = []
                for (i, timing, combo, total), result in zip(tied, results):
                    if result.ok:
                        kept.append((i, result.e2e, combo, total))
                        retimed[id(kept[-1])] = len(tied) - 1
                        continue
                    failures.setdefault(i, ((timing, combo, total), result.reason))
                    if today == (i, combo):
                        today, today_failed = None, today
                        not_today = f"failed timed again: {result.reason}"
                kept.sort(key=lambda r: r[1].median_us)
            fastest = kept[0][1].median_us
            margin = max(settings.tie_rel * fastest, settings.tie_abs_us)
            close = [r for r in kept if r[1].median_us <= fastest + margin]
            close.sort(key=lambda r: (len(found[r[0]].structure), r[1].cov))
            runs = close + kept[len(close) :] + rest

    best, order = {}, []
    for run in runs:
        if run[0] not in best:
            best[run[0]] = run[1:]
            order.append(run[0])
            if id(run) in retimed:
                notes[run[0]].append(
                    f"timed again in turn with {retimed[id(run)]} more"
                )
    dropped = {
        i: (failure, f"timed again: {reason}")
        for i, (failure, reason) in failures.items()
        if i not in best
    }
    for i, (_, reason) in failures.items():
        if i in best:
            notes[i].append(f"another plan failed timed again: {reason}")
    today_us = next(
        (r[1].median_us for r in runs if today is not None and (r[0], r[2]) == today),
        None,
    )

    outcomes = []
    for rank, i in enumerate(order, 1):
        timing, combo, total = best[i]
        built = resolver.build(problem, found[i].structure, combo)
        if built.metadata is not None and _is_mxfp4_inline_sort(built.metadata):
            notes[i].append(INLINE_SORT_NOTE)
        if i not in queues and i not in single:
            reasons = why[i] or [f"declined: {d}" for d in declined[i]]
            notes[i].append(_failed(reasons, "only the current plan ran, no level 1"))
        if today is not None and today == (i, combo):
            notes[i].append(CURRENT_NOTE)
        elif rank == 1 and not_today is not None:
            notes[i].append(f"{CURRENT_NOTE}: {not_today}")
        elif rank == 1 and today_us is not None:
            notes[i].append(f"{CURRENT_NOTE}: {today_us:.1f} us")
        outcomes.append(
            Outcome(
                _names(found[i].structure),
                combo,
                timing,
                total,
                rank,
                notes="; ".join(notes[i]),
            )
        )
    for i, ((timing, combo, total), reason) in dropped.items():
        note = CURRENT_NOTE if today_failed == (i, combo) else ""
        failed.append(
            Outcome(
                _names(found[i].structure),
                combo,
                timing,
                total,
                reason=reason,
                notes=note,
            )
        )
    for i, sc in enumerate(found):
        if i in best or i in dropped:
            continue
        if any(o.structure == _names(sc.structure) for o in failed):
            continue
        what = "no passing plan" if queues.get(i) else "no level 1 result"
        reasons = why[i] or [f"declined: {d}" for d in declined[i]]
        failed.append(Outcome(_names(sc.structure), reason=_failed(reasons, what)))
    return outcomes + failed
