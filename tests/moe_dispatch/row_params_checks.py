# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The shipped tuned rows against the tuner's candidates (`coverage.param_reasons`).

On every simulated arch, for each tuned row v2 runs (the `tuned` and `fhmoe` keys of the
`base` variant whose plan comes from the row; the tables are the snapshot's frozen copy
in `golden/inputs`, so a table change shows here once the snapshot is refrozen), each
segment running a named kernel is checked against its `param_space`, comparing what runs
(`Segment.effective_kernel`) and the candidate's other fields. The choices outside it
are counted per path, segment and reason (a row counts once per such segment);
`--check` fails on any difference from the counts in `golden/row_params.json`, in
either direction, and `--update` rewrites them. A change that adds such a choice, or
covers one, updates the file with it.

The groups (todos in docs/fused_moe_revamp_todos.md): legacy tuners `param_space` does
not mirror yet (MXMOE, FHMoE, the gfx942 whole-graph configs: 5.58), rows no legacy
tuner produces (per-1x32 1-stage ASM, the a16w4 CK-Tile Swiglu branch, per-1x32 FlyDSL
on gfx942), the CK blockscale split-K instances (5.57), the gfx942 `a16_` 1-stage
kernels run with `xbf16` (5.60), and `asm.stage1_cktile` (5.59).

    python tests/moe_dispatch/row_params_checks.py --check [--arch gfx950] [-v]
    python tests/moe_dispatch/row_params_checks.py --update [--arch gfx950]
"""

import argparse
import json
import pathlib
import sys
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent
BASELINE = _HERE / "golden" / "row_params.json"


def check_arch(arch, served, verbose=False):
    """`(counts, ok, lines)` for one simulated arch: `counts` of the choices outside
    `param_space` by "path | segment | reason", `ok` the number inside."""
    from aiter.moev2 import api
    from aiter.moev2.dispatch import coverage, tuned
    from aiter.moev2.dispatch.resolver import resolve
    from moe_dispatch import arch_sim, snapshot
    from moe_dispatch.coverage_checks import _ROW_SECTIONS
    from moe_dispatch.differential import captured_logs

    counts, ok, lines, seen = Counter(), 0, [], set()
    with arch_sim.simulated_arch(arch) as (gfx, cu_num), captured_logs():
        tuned._default_table()
        for key in snapshot.dispatch_keys(gfx, cu_num):
            fhmoe = _ROW_SECTIONS.get(key.section)
            if key.variant != "base" or fhmoe is None or key.label() in seen:
                continue
            seen.add(key.label())
            try:
                problem = api.problem_from_key(key.args, key.call_kwargs())
            except Exception:  # noqa: BLE001, S112 - key_checks.py reports these
                continue
            if fhmoe:
                problem = api.fhmoe_problem(problem)
            for path in served:
                if not path.takes(problem):
                    continue
                plan = resolve(problem, path.segments).plan
                if plan is None or plan.source != "tuned":
                    continue
                for segment, reason in coverage.param_reasons(problem, plan):
                    if reason is None:
                        ok += 1
                        continue
                    counts[f"{path.name} | {segment.name} | {reason}"] += 1
                    if verbose:
                        lines.append(f"  {key.label()} [{segment.name}]: {reason}")
    return counts, ok, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true", help=f"rewrite {BASELINE.name}")
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="list the rows")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup()
    from moe_dispatch import arch_sim
    from moe_dispatch.ranking_checks import ranked_paths

    served = ranked_paths()
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    problems = 0
    if not args.arch:
        stale = sorted(baseline.keys() - set(arch_sim.ARCHS))
        if args.update:
            baseline = {}
        elif stale:
            problems += len(stale)
            print(f"baseline archs no longer simulated: {', '.join(stale)}")
    for arch in args.arch or list(arch_sim.ARCHS):
        counts, ok, lines = check_arch(arch, served, args.verbose)
        expected = baseline.get(arch, {})
        changed = sorted(
            g
            for g in counts.keys() | expected.keys()
            if counts[g] != expected.get(g, 0)
        )
        print(
            f"{arch}: {ok} choices are candidates, {sum(counts.values())} are not; "
            f"{0 if args.update else len(changed)} problems"
        )
        for group, n in sorted(counts.items()):
            print(f"  {n:5d} {group}")
        for line in lines:
            print(line)
        if args.update:
            baseline[arch] = dict(sorted(counts.items()))
            continue
        problems += len(changed)
        for group in changed:
            print(f"  changed: {group}: {expected.get(group, 0)} -> {counts[group]}")
    if args.update:
        BASELINE.write_text(json.dumps(dict(sorted(baseline.items())), indent=1) + "\n")
        print(f"wrote {BASELINE}")
        return 0
    if problems:
        print(
            "the counts differ from the baseline; if the change is intended, run "
            "python tests/moe_dispatch/row_params_checks.py --update and commit "
            f"{BASELINE.relative_to(_HERE.parents[1])}"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
