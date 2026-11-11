# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU gate of the tuner's search (aiter.moev2.tuning, step 5.2c): the plan it picks
against the shipped tuned row's, on this GPU's arch.

For a sample of the keys each single-GPU path serves from a shipped tuned row (the
`tuning_gpu_checks` keys whose resolved plan is "tuned"), the search runs
(`search.tune`, in `tuning.pool` workers) over a sample of each tunable segment's
candidates (`--per-group` per block_m and nt, plus the shipped row's own; `--full` for
all), then its winner and the shipped plan are timed again in turn (`task.Rivals`;
not when the winner is the shipped plan). `AITER_MOEV2_TUNED_TABLE` is ignored.
Verdicts: "same plan"; "faster" or "within noise" (`--tie-rel`, `--tie-abs-us`);
"slower, outside the candidates" when the shipped plan is not one the search can pick
(listed, not a problem). A winner slower than a shipped plan the search could pick, a
key with no winner, or a shipped plan failing is a problem. `--record` writes every
verdict to a CSV.

    HIP_VISIBLE_DEVICES=0 python tests/moe_dispatch/tuning_search_checks.py
    python tests/moe_dispatch/tuning_search_checks.py --paths a8w8_ptok --full
"""

import argparse
import csv
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from moe_dispatch.build_checks import plan_candidates
from moe_dispatch.tuning_gpu_checks import sample


def _key(params) -> str:
    return repr(sorted(params.items()))


@dataclass(frozen=True)
class Chosen:
    """`candidates.Filters`-like: only these candidates, `{segment name: {_key}}`."""

    allowed: dict

    def keeps(self, segment, params) -> bool:
        return _key(params) in self.allowed.get(segment.name, ())


def shipped_choice(problem, path):
    """`(structure names, {segment name: candidate})` of the plan dispatch resolves at
    `problem` from a tuned row (the candidates None when a segment runs none of its
    `param_space`), or None for any other plan."""
    from aiter.moev2 import paths
    from aiter.moev2.dispatch import resolver

    resolved = resolver.resolve(problem, paths.registered()[path].segments)
    plan = resolved.plan
    if plan is None or plan.source != "tuned":
        return None
    chosen = plan_candidates(problem, plan)
    names = tuple(c.segment.name for c in plan.segments)
    return names, chosen if isinstance(chosen, dict) else None


def sampled(problem, path, shipped, per_group, base):
    """`(filters, in_space)`: up to `per_group` candidates per (segment, block_m, nt),
    by a fixed hash order, plus the shipped row's; `in_space`: the search can pick the
    shipped plan (its structure is listed, with the row's candidates)."""
    from aiter.moev2 import paths
    from aiter.moev2.tuning.candidates import structures

    structure, params = shipped
    found, _ = structures(problem, paths.registered()[path].segments, base)
    allowed, in_space = defaultdict(set), False
    for sc in found:
        for name, cs in sc.candidates.items():
            groups = defaultdict(list)
            for c in cs:
                groups[c.get("block_m"), c.get("nt")].append(c)
            for group in groups.values():
                group.sort(key=lambda c: zlib.crc32(_key(c).encode()))
                allowed[name].update(_key(c) for c in group[:per_group])
        if params is not None and tuple(s.name for s in sc.structure) == structure:
            listed = {n: {_key(c) for c in cs} for n, cs in sc.candidates.items()}
            in_space = all(_key(c) in listed.get(n, ()) for n, c in params.items())
    if in_space:
        for name, c in params.items():
            allowed[name].add(_key(c))
    return Chosen({k: frozenset(v) for k, v in allowed.items()}), in_space


def verdict(winner_us, shipped_us, same, in_space, settings):
    margin = max(settings.tie_rel * shipped_us, settings.tie_abs_us)
    if same:
        return "same plan", False
    if winner_us < shipped_us - margin:
        return "faster", False
    if winner_us <= shipped_us + margin:
        return "within noise", False
    if not in_space:
        return "slower, outside the candidates", False
    return "slower", True


def main(argv=None):
    from aiter.moev2.tuning.search import Settings

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-path", type=int, default=2)
    parser.add_argument("--paths", nargs="*", help="only these paths")
    parser.add_argument("--largest", action="store_true", help="the largest M per kind")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="every candidate")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--record", help="CSV of every verdict")
    defaults = Settings()
    for field, value in vars(defaults).items():
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=type(value), default=value
        )
    args = parser.parse_args(argv)
    os.environ.pop("AITER_MOEV2_TUNED_TABLE", None)
    os.environ.pop("AITER_MOEV2_TUNED_TABLE_MODE", None)
    settings = Settings(**{f: getattr(args, f) for f in vars(defaults)})

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.moev2 import api
    from aiter.moev2.tuning import task as tuning_task
    from aiter.moev2.tuning.candidates import Filters
    from aiter.moev2.tuning.pool import Pool
    from aiter.moev2.tuning.search import tune

    arch, cu_num = get_gfx(), get_cu_num()
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "keys.pkl"
        subprocess.run(
            [
                sys.executable,
                str(_HERE / "tuning_gpu_checks.py"),
                "--collect",
                str(out),
                "--arch",
                arch,
            ],
            check=True,
        )
        keys = pickle.loads(out.read_bytes())
    base = Filters.from_env()
    jobs = []
    for name in sorted(keys):
        if args.paths and name not in args.paths:
            continue
        shipped = {}
        for a, kw, fhmoe, _ in keys[name]:
            problem = api.problem_from_key(a, kw, arch, cu_num, fhmoe)
            choice = shipped_choice(problem, name)
            if choice is not None:
                shipped[problem] = choice
        for p in sample(list(shipped), args.per_path, args.largest):
            jobs.append((name, p, shipped[p]))
    if not jobs:
        print(f"no keys served from tuned rows on {arch}")
        return 0
    print(f"{arch}: {len(jobs)} keys over {len({j[0] for j in jobs})} paths")

    problems, rows, start = [], [], time.time()
    with Pool(tuning_task.run, timeout=args.timeout) as pool:

        def evaluate(tasks):
            return [r for _, r in sorted(pool.map(tasks), key=lambda t: t[0])]

        for path, problem, shipped in jobs:
            s = problem.shape
            label = f"{path} M={s.m} H={s.model_dim} I={s.inter_dim} E={s.experts} "
            label += f"k={s.topk} {problem.activation.kind}"
            filters, in_space = sampled(
                problem, path, shipped, 10**9 if args.full else args.per_group, base
            )
            began = time.time()
            outcomes = tune(problem, path, evaluate, settings, filters)
            winner = next((o for o in outcomes if o.rank == 1), None)
            row = {
                "path": path,
                "key": problem.log_keys(),
                "shipped": " | ".join(shipped[0]),
                "shipped_candidates": {
                    k: dict(v) for k, v in (shipped[1] or {}).items()
                },
                "in_space": in_space,
                "search_s": f"{time.time() - began:.0f}",
            }
            if winner is None:
                problems.append(f"{label}: no winner ({outcomes[0].reason})")
                print(f"  PROBLEM {problems[-1]}", flush=True)
                rows.append({**row, "verdict": "no winner"})
                continue
            same = (
                winner.structure == shipped[0]
                and shipped[1] is not None
                and {n: _key(c) for n, c in winner.candidates.items()}
                == {n: _key(c) for n, c in shipped[1].items()}
            )
            if same:
                old = new = winner
            else:
                rivals = tuning_task.Rivals(
                    problem,
                    path,
                    (
                        (None, None),
                        (winner.structure, tuple(winner.candidates.items())),
                    ),
                    rounds=settings.rounds,
                    iters=settings.iters,
                    warmup=settings.warmup,
                )
                (results,) = evaluate([rivals])
                if not isinstance(results, list):
                    results = [results, results]
                old, new = results
                if not old.ok or not new.ok:
                    reason = old.reason if not old.ok else new.reason
                    who = "shipped" if not old.ok else "winner"
                    problems.append(f"{label}: {who} failed timed again: {reason}")
                    print(f"  PROBLEM {problems[-1]}", flush=True)
                    rows.append({**row, "verdict": f"{who} failed: {reason}"})
                    continue
            old_us, new_us = old.e2e.median_us, new.e2e.median_us
            what, bad = verdict(new_us, old_us, same, in_space, settings)
            line = (
                f"{label}: {what}: winner {' | '.join(winner.structure)} "
                f"{new_us:.1f} us, shipped {row['shipped']} {old_us:.1f} us "
                f"({old.e2e.mode}, {row['search_s']} s)"
            )
            rows.append(
                {
                    **row,
                    "verdict": what,
                    "winner": " | ".join(winner.structure),
                    "winner_candidates": {
                        k: dict(v) for k, v in winner.candidates.items()
                    },
                    "winner_us": f"{new_us:.3f}",
                    "shipped_us": f"{old_us:.3f}",
                    "mode": old.e2e.mode,
                }
            )
            if bad:
                problems.append(line)
                print(f"  PROBLEM {line}", flush=True)
            else:
                print(f"  {line}", flush=True)
    if args.record:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.record, "w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(problems)} problems ({time.time() - start:.0f} s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
