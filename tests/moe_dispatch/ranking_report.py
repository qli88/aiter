# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ranking refit report (`aiter/moev2/tuning/refit_ranking.py`) over a committed key
sample, on this GPU. Informational only: it prints the report and exits 0 whatever the
regret; only users change the ranking table.

    python tests/moe_dispatch/ranking_report.py [--devices 4,5,6,7] [--csv out.csv]
        [--times times.jsonl]   # a long run: keep measured keys, resume where stopped
    python tests/moe_dispatch/ranking_report.py --update [--per-class N]

The sample (`golden/ranking_report_keys.csv`), per simulated arch: up to N keys per
(ranking entry, mode) of the shipped tuned tables, spread over M
(`refit_ranking.sample`), filled up to N from the dispatch key grid (its base,
unpadded keys) where the tables have fewer. A key is kept where a registered path takes
it, the default plan of the ranking's pick builds (`resolver.default_plan`: not a
`legacy=` region, not a dedicated table) and the tuner builds inputs for it
(`inputs.unsupported`); `--update` prints the groups it drops and why. No FHMoE key
qualifies: the shipped ones are all dedicated-table keys, which have no default ranking
(`--fhmoe-keys` of the CLI takes others). `--update` rebuilds the sample on the CPU
from the shipped tables; nothing checks it.
"""

import argparse
import csv
import glob
import pathlib
import sys
from collections import Counter, defaultdict

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
SAMPLE = _HERE / "golden" / "ranking_report_keys.csv"
PER_CLASS = 4
NO_INPUTS = "no tuner inputs (todo 2.42, tuning/inputs.py)"


def _columns():
    from aiter.moev2.dispatch.convert import KEY_FIELDS

    return ("gfx", "cu_num", *KEY_FIELDS, "hidden_pad", "intermediate_pad")


def _shipped_rows(files, gfx, cu_num):
    """`(label, row)` of `files`' rows for `gfx` and `cu_num`."""
    for name in files:
        with open(name, newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                if row.get("gfx") == gfx and int(row.get("cu_num") or 0) == cu_num:
                    yield f"{pathlib.Path(name).name}:{n}", row


def _grid_rows(gfx, cu_num):
    """`(label, row)` of the grid's base keys without padding."""
    from aiter.utility import dtypes
    from moe_dispatch import keys

    for key in keys.grid_keys(dtypes.fp8, keys.tuned_table()):
        a = key.args
        if key.variant != "base" or a[12] or a[13]:
            continue
        yield key.label(), {
            "gfx": gfx,
            "cu_num": cu_num,
            "token": a[0],
            "model_dim": a[1],
            "inter_dim": a[2],
            "expert": a[3],
            "topk": a[4],
            "act_type": str(a[10]),
            "dtype": str(a[5]),
            "q_dtype_a": str(a[6]),
            "q_dtype_w": str(a[7]),
            "q_type": str(a[8]),
            "use_g1u1": int(a[9]),
            "doweight_stage1": int(a[11]),
        }


def why_not(problem) -> str | None:
    """Why `problem` is not sampled, or None."""
    from aiter.moev2 import paths
    from aiter.moev2.dispatch import ranking, resolver
    from aiter.moev2.dispatch.convert import paths_taking
    from aiter.moev2.tuning import inputs

    taking = paths_taking(problem)
    if not taking:
        return "no path takes it"
    why = "no structure from the ranking (no entry, legacy=, nothing feasible)"
    for name in taking:
        segments = paths.registered()[name].segments
        plans = ranking.rankable(resolver.feasible_plans(problem, segments)[0])
        best = ranking.best(problem, plans)
        if best is None or best[1] is None:
            continue
        names = [s.name for s in best[1]]
        if resolver.default_plan(problem, segments, names).plan is not None:
            return None if inputs.unsupported(problem) is None else NO_INPUTS
        why = "the pick's default plan does not build"
    return why


def _keyed(rows, gfx, cu_num, dropped):
    """`{label: (row, problem)}` of the rows sampled from, in order; rows that do not
    key are skipped, the others counted in `dropped` by (entry, mode, why)."""
    from aiter.moev2.dispatch.convert import problem_of
    from aiter.moev2.tuning import refit_ranking as rr

    out = {}
    for label, row in rows:
        text = {k: str(v) for k, v in row.items()}
        try:
            problem = problem_of(text, gfx, cu_num)
        except Exception:  # noqa: BLE001, S112 - key_checks.py reports these
            continue
        if label in out:
            continue
        why = why_not(problem)
        if why is None:
            out[label] = (row, problem)
        elif why != "no path takes it":
            dropped[(*rr.group_of(problem), why)] += 1
    return out


def chosen(shipped, grid, per_class):
    """The labels of `shipped` (`{label: (row, problem)}`) sampled, then of `grid`
    filling each (entry, mode) to `per_class`."""
    from aiter.moev2.tuning import refit_ranking as rr

    picked = rr.sample([(lb, p) for lb, (_, p) in shipped.items()], per_class)
    counts = Counter(rr.group_of(p) for _, p in picked)
    by_group = defaultdict(list)
    for label, (_, problem) in grid.items():
        by_group[rr.group_of(problem)].append((label, problem))
    for group, members in sorted(by_group.items()):
        need = per_class - counts[group]
        if need > 0:
            picked += rr.sample(members, need)
    return [label for label, _ in picked]


def update(per_class):
    from moe_dispatch import snapshot

    snapshot._setup(frozen=False)
    from moe_dispatch import arch_sim
    from moe_dispatch.differential import captured_logs

    configs = _REPO / "aiter" / "configs"
    files = [str(configs / "tuned_fmoe.csv")]
    files += sorted(glob.glob(str(configs / "model_configs" / "*tuned_fmoe*.csv")))
    columns = _columns()
    out = []
    for name, (gfx, cu_num) in arch_sim.ARCHS.items():
        dropped = Counter()
        with arch_sim.simulated_arch(name), captured_logs():
            shipped = _keyed(_shipped_rows(files, gfx, cu_num), gfx, cu_num, dropped)
            grid = _keyed(_grid_rows(gfx, cu_num), gfx, cu_num, dropped)
            labels = chosen(shipped, grid, per_class)
        for label in labels:
            row = (shipped.get(label) or grid[label])[0]
            out.append({c: row.get(c, "") for c in columns})
        print(f"{name}: {len(labels)} keys")
        by_why = defaultdict(list)
        for (entry, mode, why), n in sorted(dropped.items()):
            by_why[why].append(f"{entry} [{mode}] {n}")
        for why, groups in sorted(by_why.items()):
            print(f"  not sampled, {why}: {', '.join(groups)}")
    with open(SAMPLE, "w", newline="") as f:
        writer = csv.DictWriter(f, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {SAMPLE.relative_to(_REPO)} ({len(out)} rows)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="rebuild the sample")
    parser.add_argument(
        "--per-class", type=int, default=PER_CLASS, help="with --update"
    )
    parser.add_argument("--devices", help="HIP device ids, comma-separated")
    parser.add_argument("--csv", help="also write a row per key here")
    parser.add_argument("--times", help="keep and reuse measured keys here")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(_HERE.parent))
    if args.update:
        return update(args.per_class)
    from aiter.moev2.tuning import refit_ranking

    refit_ranking.exit_on_sigterm()
    forwarded = [str(SAMPLE)]
    for flag, value in (
        ("--devices", args.devices),
        ("--csv", args.csv),
        ("--times", args.times),
    ):
        if value:
            forwarded += [flag, value]
    refit_ranking.main(forwarded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
