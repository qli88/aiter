# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`python -m aiter.moev2.tuning.refit_ranking`: the ranking refit report (plan §3.6).

Informational only: it changes nothing, reads no stored tuning data, and fails on no
regret; a user applies a hint to `ranking_table.py` (plan 7.5).

Per key, on one device: the search's winner over every structure (`search.tune`) and
each rankable structure at its default params (`resolver.default_plan`), end to end in
the key's deployment mode. The regret of the default ranking at the key is

    regret = the time of the structure the ranking picks, at its default params
             / the fastest time measured for the key (the winner or any default)

The report gives per (entry, mode) the geomean and the worst regret and the keys left
out (`left_out`: not measured, no entry, a `legacy=` region, no listed structure
feasible, the pick's default plan not running; a key on two paths counts once per
path); per numeric cut of an entry (a bound of its `when` conditions on a
scalar; every condition of the entry at that bound moves with it, `moved`) the bound
over the sample's values with the lowest geomean (`search_cut`); per region the
structure that would lower its geomean if listed first, by more than `HINT_MARGIN`
(`region_hints`); and the structures whose default plan did not run, with a reason.

Keys come from CSVs in the tuner's input formats (`tuning.cli`: untuned rows, or tuned
tables with `--keys-from`; FHMoE keys with `--fhmoe-keys`), each on every registered
path taking it; `--per-class N` keeps N keys per (entry, mode), spread over M
(`sample`). Keys run in parallel, one device each (`--devices`). `--times FILE` keeps
each measured key (JSON lines) and reuses it on the next run, so a long run can be
stopped and resumed.
"""

import argparse
import copy
import csv
import dataclasses
import itertools
import json
import math
import os
import queue
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import paths
from ..dispatch import ranking, resolver
from ..dispatch.tuned_plans import STRUCTURE_SEPARATOR
from ..problem import FrozenMap
from .measure import deploy_mode
from .search import Settings, tune
from .task import Task

# A region's hint names another structure only when it lowers the geomean by more.
HINT_MARGIN = 0.01
# Set when a run stops (SIGTERM, an exception): no further batch starts, nothing more
# is saved to --times.
STOPPING = threading.Event()
REASON_CHARS = 200


@dataclass(frozen=True)
class KeyTimes:
    """A key's measurements: per rankable structure its default plan's time (or why
    it did not run), and the search's winner."""

    label: str
    problem: object
    path: str
    plans: tuple  # the rankable feasible structures (segment tuples)
    default_us: Mapping  # {structure names: median us}
    default_why: Mapping  # {structure names: why its default plan did not run}
    tuned: tuple | None = None  # the winner's structure names
    tuned_us: float | None = None
    tuned_why: str | None = None
    error: str | None = None  # the measurement raised: nothing measured

    @property
    def fastest_us(self) -> float | None:
        times = list(self.default_us.values())
        if self.tuned_us is not None:
            times.append(self.tuned_us)
        return min(times, default=None)

    @property
    def mode(self) -> str:
        return deploy_mode(self.problem)


def _names(structure) -> tuple:
    return tuple(s.name for s in structure)


def _show(names) -> str:
    return STRUCTURE_SEPARATOR.join(names)


def pick(key, table=None):
    """`(region label, structure names, why none)` the ranking `table` gives `key`."""
    if key.error is not None:
        return None, None, "not measured"
    best = ranking.best(key.problem, key.plans, table)
    if best is None:
        return None, None, "no entry"
    label, plan, legacy = best
    if legacy is not None:
        return label, None, "legacy region"
    if plan is None:
        return label, None, "no listed structure feasible"
    return label, _names(plan), None


def regret(key, names):
    """`(regret, why none)` of running `names` at its default params at `key`."""
    if names not in key.default_us:
        return None, "default plan not run"
    fastest = key.fastest_us
    return key.default_us[names] / fastest, None


def geomean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return math.exp(sum(map(math.log, values)) / len(values))


@dataclass
class Summary:
    regrets: list = dataclasses.field(default_factory=list)  # (regret, label)
    left_out: Counter = dataclasses.field(default_factory=Counter)

    @property
    def geomean(self):
        return geomean(r for r, _ in self.regrets)

    @property
    def worst(self):
        return max(self.regrets, default=(None, None))


def summarize(keys, table=None) -> dict:
    """`{(entry, mode): Summary}`, and `(None, mode)` over every entry."""
    out = defaultdict(Summary)
    for key in keys:
        entry = ranking.entry_for(key.problem, table)
        group = (entry.name if entry else "(no entry)", key.mode)
        _, names, why = pick(key, table)
        value = None
        if names is not None:
            value, why = regret(key, names)
        for g in (group, (None, key.mode)):
            if value is None:
                out[g].left_out[why] += 1
            else:
                out[g].regrets.append((value, key.label))
    return dict(out)


def cuts(entry) -> list:
    """`(scalar, bound)` of each numeric cut of `entry`: a `lo` or `hi` of a region's
    condition on a scalar (`x < bound` against `x >= bound`)."""
    found = set()
    for region in entry.regions:
        for name, domain in region.when.domains.items():
            if name not in ranking.SCALARS:
                continue
            found.update((name, b) for b in (domain.lo, domain.hi) if b is not None)
    return sorted(found)


def _with(obj, **fields):
    """A copy of frozen `obj` with `fields` set, without its `__post_init__` (which
    parses structure texts again)."""
    out = copy.copy(obj)
    for name, value in fields.items():
        object.__setattr__(out, name, value)
    return out


def moved(table, entry_name, scalar, bound, new):
    """`table` with every bound `bound` of `entry_name`'s conditions on `scalar` at
    `new`, or None where a condition would be empty or two regions would overlap."""
    entries = []
    for entry in table:
        if entry.name != entry_name:
            entries.append(entry)
            continue
        regions = []
        for region in entry.regions:
            domains = dict(region.when.domains)
            d = domains.get(scalar)
            if d is not None and bound in (d.lo, d.hi):
                d = dataclasses.replace(
                    d,
                    lo=new if d.lo == bound else d.lo,
                    hi=new if d.hi == bound else d.hi,
                )
                if d.empty:
                    return None
                domains[scalar] = d
            w = dataclasses.replace(region.when, domains=FrozenMap(domains))
            regions.append(_with(region, when=w))
        if not all(
            a.when.disjoint(b.when) for a, b in itertools.combinations(regions, 2)
        ):
            return None
        entries.append(_with(entry, regions=tuple(regions)))
    return tuple(entries)


def _scored(keys, table):
    """`(geomean, keys without a regret)` of `keys` under `table`."""
    values, unserved = [], 0
    for key in keys:
        _, names, _ = pick(key, table)
        value = None if names is None else regret(key, names)[0]
        if value is None:
            unserved += 1
        else:
            values.append(value)
    return geomean(values), unserved


@dataclass(frozen=True)
class CutHint:
    entry: str
    scalar: str
    bound: int
    best: int
    before: float | None
    after: float | None
    keys: int


def search_cut(keys, table, entry, scalar, bound) -> CutHint:
    """The bound of the cut with the lowest geomean of `entry`'s keys and no more keys
    left without a regret; on a tie the bound nearest `bound` (`bound` itself kept).
    Each split of the values the keys take is tried at both of its tightest bounds
    (`v` and `v + 1` for each value `v`), so one of them may stay clear of another
    region."""
    mine = [k for k in keys if ranking.entry_for(k.problem, table) is entry]
    before, unserved = _scored(mine, table)
    best, after = bound, before
    values = {ranking.SCALARS[scalar](k.problem) for k in mine}
    candidates = values | {v + 1 for v in values}
    for new in sorted(candidates, key=lambda b: (abs(b - bound), b)):
        changed = moved(table, entry.name, scalar, bound, new)
        if new == bound or changed is None:
            continue
        score, left = _scored(mine, changed)
        if score is not None and left <= unserved and (after is None or score < after):
            best, after = new, score
    return CutHint(entry.name, scalar, bound, best, before, after, len(mine))


@dataclass(frozen=True)
class RegionHint:
    label: str
    current: tuple  # the first structure of the region's list
    best: tuple  # the structure that would lower the geomean most if listed first
    before: float
    after: float
    keys: int
    # structure names -> keys it would take whose default plan did not run
    fails: Counter


def region_hints(keys, table=None) -> list:
    """Per region with measured keys: the structure that, listed first, gives its keys
    the lowest geomean, if lower than today's by more than `HINT_MARGIN` (else the
    first listed); one whose default plan did not run at one of those keys (the
    region's keys where today's pick ran) is not proposed."""
    by_region, firsts = defaultdict(list), {}
    for key in keys:
        label, names, _ = pick(key, table)
        if names is None or regret(key, names)[0] is None:
            continue
        by_region[label].append((key, names))
        prefer = ranking.entry_for(key.problem, table).region_for(key.problem)[1]
        firsts[label] = prefer[0]
    hints = []
    for label, items in sorted(by_region.items()):
        before = geomean(regret(k, names)[0] for k, names in items)
        current = firsts[label]
        options = {n for k, _ in items for n in (*k.default_us, *k.default_why)}
        options -= {current}
        fails, lowest = Counter(), None
        for option in sorted(options):
            values, failed = [], 0
            for key, names in items:
                feasible = option in {_names(p) for p in key.plans}
                if feasible and option not in key.default_us:
                    failed += 1
                    continue
                values.append(regret(key, option if feasible else names)[0])
            if failed:
                fails[option] = failed
                continue
            score = geomean(values)
            if lowest is None or score < lowest[0]:
                lowest = score, option
        best, after = current, before
        if lowest is not None and lowest[0] < before * (1 - HINT_MARGIN):
            after, best = lowest
        hints.append(RegionHint(label, current, best, before, after, len(items), fails))
    return hints


def _num(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def report(keys, table=None) -> list:
    """The report's lines."""
    table = ranking._default_table() if table is None else table
    lines = [f"ranking refit report: {len(keys)} keys (informational; nothing applied)"]
    summaries = summarize(keys, table)
    for (entry, mode), s in sorted(
        summaries.items(), key=lambda kv: (kv[0][0] is None, kv[0][0] or "", kv[0][1])
    ):
        worst, where = s.worst
        name = "all entries" if entry is None else entry
        line = (
            f"{name} [{mode}]: {len(s.regrets)} keys, geomean regret "
            f"{_num(s.geomean)}, worst {_num(worst)}"
        )
        if where is not None:
            line += f" @ {where}"
        if s.left_out:
            line += "; left out: " + ", ".join(
                f"{n} {why}" for why, n in sorted(s.left_out.items())
            )
        lines.append(line)
    lines.append(
        "thresholds (every condition of the entry at the bound moves with it):"
    )
    for entry in table:
        for scalar, bound in cuts(entry):
            h = search_cut(keys, table, entry, scalar, bound)
            if not h.keys:
                continue
            cut = f"{scalar} < {h.bound}"
            if h.best == h.bound:
                lines.append(
                    f"  {entry.name}: {cut} (unchanged) geomean {_num(h.before)}"
                    f" ({h.keys} keys)"
                )
            else:
                lines.append(
                    f"  {entry.name}: {cut} -> {scalar} < {h.best} geomean "
                    f"{_num(h.before)} -> {_num(h.after)} ({h.keys} keys)"
                )
    lines.append("regions (the structure listed first):")
    for h in region_hints(keys, table):
        text = f"  {h.label}: "
        if h.best == h.current:
            text += f"{_show(h.current)} (unchanged) geomean {_num(h.before)}"
        else:
            text += (
                f"{_show(h.current)} -> {_show(h.best)} first, geomean "
                f"{_num(h.before)} -> {_num(h.after)}"
            )
        text += f" ({h.keys} keys)"
        if h.fails:
            text += "; default plan not run: " + ", ".join(
                f"{_show(n)} at {c}" for n, c in sorted(h.fails.items())
            )
        lines.append(text)
    failures = defaultdict(list)
    for key in keys:
        found = [(_show(n), why) for n, why in key.default_why.items()]
        if key.error is not None:
            found.append(("not measured", key.error))
        for what, why in found:
            failures[what].append((key.label, " ".join(str(why).split())))
    if failures:
        lines.append("did not run (structure, or not measured): keys, first reason")
    for what, seen in sorted(failures.items()):
        label, reason = seen[0]
        lines.append(f"  {what}: {len(seen)} keys; {label}: {reason[:REASON_CHARS]}")
    return lines


def _rankable(problem, path) -> tuple:
    segments = paths.registered()[path].segments
    return tuple(ranking.rankable(resolver.feasible_plans(problem, segments)[0]))


def measure_key(label, problem, path, evaluate, settings, filters, log=None):
    """`KeyTimes` of `problem` on `path`; `evaluate` runs tasks on one device."""
    plans = _rankable(problem, path)
    names = [_names(p) for p in plans]
    tasks = [
        Task(
            problem,
            path,
            n,
            None,
            iters=settings.iters,
            warmup=settings.warmup,
            seed=settings.seed,
        )
        for n in names
    ]
    default_us, default_why = {}, {}
    for n, result in zip(names, evaluate(tasks)):
        if result.ok:
            default_us[n] = result.e2e.median_us
        else:
            default_why[n] = result.reason
    outcomes = tune(problem, path, evaluate, settings, filters, log)
    win = next((o for o in outcomes if o.rank == 1), None)
    if win is None:
        why = outcomes[0].reason if outcomes else "nothing measured"
        return KeyTimes(
            label, problem, path, plans, default_us, default_why, None, None, why
        )
    return KeyTimes(
        label,
        problem,
        path,
        plans,
        default_us,
        default_why,
        tuple(win.structure),
        win.e2e.median_us,
    )


def _key_id(problem, path) -> str:
    # A frozenset's repr order changes with the process's string hash seed.
    text = repr(problem).replace(repr(problem.features), repr(sorted(problem.features)))
    return f"{path}|{text}"


def saved_line(key) -> str:
    """A `--times` line of a measured `key`."""
    return json.dumps(
        {
            "key": _key_id(key.problem, key.path),
            "default_us": [[list(n), us] for n, us in key.default_us.items()],
            "default_why": [[list(n), why] for n, why in key.default_why.items()],
            "tuned": list(key.tuned) if key.tuned else None,
            "tuned_us": key.tuned_us,
            "tuned_why": key.tuned_why,
        }
    )


SAVED_FIELDS = ("key", "default_us", "default_why", "tuned", "tuned_us", "tuned_why")


def read_saved(path, log=None) -> dict:
    """`{key id: saved fields}` of a `--times` file (the last line of a key wins);
    lines that do not parse (a run killed while writing) are skipped."""
    saved, skipped = {}, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                fields = json.loads(line)
            except json.JSONDecodeError:
                fields = None
            if not isinstance(fields, dict) or not all(
                f in fields for f in SAVED_FIELDS
            ):
                skipped += 1
                continue
            saved[fields["key"]] = fields
    if skipped and log is not None:
        log(f"{path}: skipped {skipped} unreadable lines")
    return saved


def from_saved(fields, label, problem, path) -> KeyTimes:
    """The `KeyTimes` of `fields` (`read_saved`) for `label`."""
    return KeyTimes(
        label,
        problem,
        path,
        _rankable(problem, path),
        {tuple(n): us for n, us in fields["default_us"]},
        {tuple(n): why for n, why in fields["default_why"]},
        tuple(fields["tuned"]) if fields["tuned"] else None,
        fields["tuned_us"],
        fields["tuned_why"],
    )


def group_of(problem, table=None) -> tuple:
    """`(entry name, mode)` of `problem` ("" without an entry)."""
    entry = ranking.entry_for(problem, table)
    return entry.name if entry else "", deploy_mode(problem)


def sample(keys, per_class, table=None) -> list:
    """`keys` (`(label, MoeProblem)`) with at most `per_class` per (entry, mode), spread
    evenly over M (then label) within each; all of them when `per_class` is 0."""
    if not per_class:
        return list(keys)
    groups = defaultdict(list)
    for label, problem in keys:
        groups[group_of(problem, table)].append((label, problem))
    kept = set()
    for members in groups.values():
        members.sort(key=lambda lp: (lp[1].shape.m, lp[0]))
        n = len(members)
        if n <= per_class:
            kept.update(lp[0] for lp in members)
            continue
        step = (n - 1) / (per_class - 1) if per_class > 1 else 0
        kept.update(members[round(i * step)][0] for i in range(per_class))
    return [(label, problem) for label, problem in keys if label in kept]


CSV_COLUMNS = (
    "label",
    "path",
    "mode",
    "region",
    "pick",
    "pick_us",
    "fastest_us",
    "regret",
    "tuned",
    "tuned_us",
    "why",
    "pick_why",
    "tuned_why",
    "error",
)


def key_rows(keys, table=None):
    """A CSV row per key: what the ranking picks and its regret; `why`: why no
    regret, `pick_why`: why the pick's default plan did not run."""
    for key in keys:
        region, names, why = pick(key, table)
        value = None
        if names is not None:
            value, why = regret(key, names)
        yield {
            "label": key.label,
            "path": key.path,
            "mode": key.mode,
            "region": region or "",
            "pick": _show(names) if names else "",
            "pick_us": key.default_us.get(names, "") if names else "",
            "fastest_us": key.fastest_us or "",
            "regret": "" if value is None else f"{value:.4f}",
            "tuned": _show(key.tuned) if key.tuned else "",
            "tuned_us": key.tuned_us or "",
            "why": why or "",
            "pick_why": key.default_why.get(names, "") if names else "",
            "tuned_why": key.tuned_why or "",
            "error": key.error or "",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m aiter.moev2.tuning.refit_ranking",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("inputs", nargs="*", help="keys, legacy untuned CSV format")
    parser.add_argument("--keys-from", nargs="*", default=[], help="tuned CSVs")
    parser.add_argument("--fhmoe-keys", nargs="*", default=[], help="FHMoE key CSVs")
    parser.add_argument("--fhmoe-table", help="default: the shipped tuned_fhmoe.csv")
    parser.add_argument("--per-class", type=int, default=0, help="0: every key")
    parser.add_argument("--paths", nargs="*", help="only these paths")
    parser.add_argument("--devices", help="HIP device ids, comma-separated")
    parser.add_argument("--timeout", type=float, default=600.0, help="per task, s")
    parser.add_argument("--csv", help="also write a row per key here")
    parser.add_argument(
        "--times",
        help="JSON lines: each measured key is appended here, and keys found here "
        "(same path and problem) are not measured again, whatever the tuner "
        "settings or code they were measured with",
    )
    args = parser.parse_args(argv)
    if not (args.inputs or args.keys_from or args.fhmoe_keys):
        parser.error("no keys: give input CSVs, --keys-from or --fhmoe-keys")

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from ..dispatch.convert import paths_taking
    from ..migration import fhmoe
    from .candidates import Filters
    from .cli import default_devices, read_keys
    from .pool import Pool
    from .task import run

    def log(line):
        print(f"[moev2 refit] {line}", file=sys.stderr, flush=True)

    arch, cu_num = get_gfx(), get_cu_num()
    keys = read_keys(args.inputs + args.keys_from, arch, cu_num, None, log)
    if args.fhmoe_keys:
        table = args.fhmoe_table or fhmoe._fhmoe_config_file()
        keys += read_keys(args.fhmoe_keys, arch, cu_num, table, log)
    keys = sample(keys, args.per_class)
    work = [
        (label, problem, path)
        for label, problem in keys
        for path in paths_taking(problem, args.paths)
    ]
    listed = args.devices.split(",") if args.devices else default_devices()
    devices = list(dict.fromkeys(str(d).strip() for d in listed if str(d).strip()))
    if not devices:
        parser.error("no devices: give --devices")
    filters, settings = Filters.from_env(), Settings()
    saved = {}
    if args.times and os.path.exists(args.times):
        saved = read_saved(args.times, log)
    done = {}
    for i, (label, problem, path) in enumerate(work):
        fields = saved.get(_key_id(problem, path))
        if fields is not None:
            try:
                done[i] = from_saved(fields, label, problem, path)
            except Exception as error:  # noqa: BLE001 - measured again instead
                log(f"{label} {path}: saved times not used: {error!r}")
    log(
        f"{arch} cu{cu_num}: {len(work)} key/path pairs ({len(done)} from --times), "
        f"devices {devices}"
    )
    pools = queue.Queue()
    lock = threading.Lock()
    STOPPING.clear()

    def one(item):
        label, problem, path = item
        pool = pools.get()
        try:
            start = time.time()

            def evaluate(tasks):
                if STOPPING.is_set():
                    raise RuntimeError("the run is stopping")
                return [r for _, r in sorted(pool.map(tasks), key=lambda t: t[0])]

            try:
                times = measure_key(
                    label,
                    problem,
                    path,
                    evaluate,
                    settings,
                    filters,
                    lambda line: log(f"{label} {path}: {line}"),
                )
            except Exception as error:  # noqa: BLE001 - reported as not measured
                why = f"{type(error).__name__}: {str(error)[:300]}"
                with lock:
                    log(f"{label} {path}: not measured: {why}")
                return KeyTimes(label, problem, path, (), {}, {}, error=why)
            with lock:
                log(f"{label} {path}: measured ({time.time() - start:.0f} s)")
                if args.times and STOPPING.is_set():
                    # Its last runs may have failed because the workers were killed.
                    log(f"{label} {path}: not saved: the run is stopping")
                elif args.times:
                    try:
                        with open(args.times, "a") as f:
                            f.write(saved_line(times) + "\n")
                    except OSError as error:
                        log(f"{label} {path}: not saved to {args.times}: {error}")
            return times
        finally:
            pools.put(pool)

    todo = [item for i, item in enumerate(work) if i not in done]
    executor = ThreadPoolExecutor(len(devices)) if todo else None
    try:
        if executor is not None:
            for device in devices:
                pools.put(Pool(run, devices=[device], timeout=args.timeout))
        fresh = iter(list(executor.map(one, todo)) if executor else [])
        measured = [done[i] if i in done else next(fresh) for i in range(len(work))]
    except BaseException:
        # Keys in flight stop at their next batch of runs; queued ones never start.
        STOPPING.set()
        raise
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
        while not pools.empty():
            pools.get().close()
    print(f"{arch} cu{cu_num}")
    for line in report(measured):
        print(line)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(key_rows(measured))
    return 0


def exit_on_sigterm():
    """Killed plainly, the run's workers would outlive it."""
    import signal

    def stop(signum, frame):
        STOPPING.set()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)


if __name__ == "__main__":
    exit_on_sigterm()
    sys.exit(main())
