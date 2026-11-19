# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Coverage of the v2 paths over the snapshot key grid (`aiter.moev2.dispatch.coverage`).

On every simulated arch, for every snapshot key and for the key as `fhmoe_` dispatches
it, for each path that takes it:

- prints the capability matrix: per path, the keys with a plan, handed to a named legacy
  branch by the ranking, declined because of their tuned row, and the gaps (no plan from
  the path's segments), each split by whether legacy runs the key or raises;
- fails on a raise, or on a gap `known_declines.py` does not list, where legacy runs
  the key;
- reports the tuned rows that do not run (discarded or declined, by reason), one key per
  row (the `tuned` and `fhmoe` keys of the `base` variant);
- fails on a segment feasible for some key but in no ranking entry (unless
  `tuned_only`), and on a `tuned_only` segment in a ranking entry.

    python tests/moe_dispatch/coverage_checks.py --check [--arch gfx950] [-v]

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import pathlib
import sys
from collections import Counter, defaultdict

_HERE = pathlib.Path(__file__).resolve().parent

# The tuned rows: the key of each row (its own shape and combination, `base` variant).
_ROW_SECTIONS = {"tuned": False, "fhmoe": True}


def _legacy_runs(golden, key, fhmoe):
    from moe_dispatch import snapshot
    from moe_dispatch.differential import legacy_fhmoe

    if fhmoe:
        expected, _ = legacy_fhmoe(key)
    else:
        expected = snapshot.resolve_same_as_base(golden, key.label())
    return "error" not in expected


def check_arch(arch, served, verbose=False):
    """`(errors, report lines, problems)` for one simulated arch; `problems` are the
    `(label, problem)` pairs for `coverage.segment_errors`."""
    from aiter.moev2 import api
    from aiter.moev2.dispatch import coverage, tuned
    from moe_dispatch import arch_sim, known_declines, snapshot
    from moe_dispatch.differential import captured_logs

    golden = snapshot.read(arch)
    errors, lines, problems = [], [], []
    matrix = defaultdict(Counter)
    known = Counter()
    rows, bad_rows = Counter(), Counter()
    seen = set()
    with arch_sim.simulated_arch(arch) as (gfx, cu_num), captured_logs():
        tuned._default_table()
        for key in snapshot.dispatch_keys(gfx, cu_num):
            if key.label() in seen:  # a row repeated in the tuned table
                continue
            seen.add(key.label())
            try:
                base = api.problem_from_key(key.args, key.call_kwargs())
            except Exception:  # noqa: BLE001, S112 - key_checks.py reports these
                continue
            for fhmoe, problem in ((False, base), (True, api.fhmoe_problem(base))):
                label = f"{arch} {key.label()}" + ("|fhmoe" if fhmoe else "")
                problems.append((label, problem))
                is_row = (
                    key.variant == "base" and _ROW_SECTIONS.get(key.section) is fhmoe
                )
                taken = False
                for path in served:
                    if not path.takes(problem):
                        continue
                    taken = True
                    kind, resolution = coverage.classify(problem, path.segments)
                    runs = kind == "plan" or _legacy_runs(golden, key, fhmoe)
                    matrix[path.name][kind, runs] += 1
                    status = coverage.row_status(problem, resolution)
                    if is_row and status is not None:
                        rows[status != "used"] += 1
                        if status != "used":
                            bad_rows[f"{path.name}: {status}"] += 1
                            if verbose:
                                lines.append(f"  row {label} [{path.name}]: {status}")
                    if kind == "raises" and runs:
                        errors.append(f"{label} [{path.name}]: raises; legacy runs it")
                    if kind != "gap" or not runs:
                        continue
                    entry = known_declines.find(path.name, problem, resolution.declined)
                    if entry is not None:
                        known[entry.name] += 1
                        continue
                    errors.append(
                        f"{label} [{path.name}]: gap where legacy runs the key "
                        f"({resolution.declined})"
                    )
                if is_row and not taken and tuned.lookup(problem) is not None:
                    rows["no path"] += 1
    for name in sorted(matrix):
        counts = matrix[name]
        parts = []
        for kind in coverage.KINDS:
            runs, raises = counts[kind, True], counts[kind, False]
            if runs or raises:
                parts.append(
                    f"{kind} {runs + raises}"
                    + (
                        f" ({runs} legacy runs, {raises} raises)"
                        if kind != "plan"
                        else ""
                    )
                )
        lines.append(f"  {name}: " + ", ".join(parts))
    lines.append(
        f"  tuned rows: {rows[False]} run, {rows[True]} do not, "
        f"{rows['no path']} on keys no path takes"
    )
    for reason, n in bad_rows.most_common():
        lines.append(f"    {n:5d} {reason}")
    for name, n in sorted(known.items()):
        lines.append(f"  known gaps: {name}: {n}")
    return errors, lines, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="list the rows")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup()
    from aiter.moev2.dispatch import coverage
    from moe_dispatch import arch_sim
    from moe_dispatch.ranking_checks import ranked_paths

    served = ranked_paths()
    errors, segment_errors = [], {}
    for arch in args.arch or list(arch_sim.ARCHS):
        arch_errors, lines, problems = check_arch(arch, served, args.verbose)
        print(f"{arch}: {len(arch_errors)} problems")
        for line in lines:
            print(line)
        errors += arch_errors
        # `takes` and the caps see the arch's dtypes (fp8) only inside the simulation.
        with arch_sim.simulated_arch(arch):
            for error in coverage.segment_errors(served, problems):
                segment_errors.setdefault(error.split(":")[0], error)
    segment_errors = list(segment_errors.values())
    tuned_only = sorted({s.name for p in served for s in p.segments if s.tuned_only})
    print(
        f"segments: {len(segment_errors)} problems; tuned_only: {', '.join(tuned_only)}"
    )
    errors += segment_errors
    for line in errors[:60]:
        print("  " + line)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
