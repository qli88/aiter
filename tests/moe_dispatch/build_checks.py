# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`resolver.build` against dispatch, on the shipped tuned rows.

On every simulated arch, for each tuned row v2 runs (the `tuned` and `fhmoe` keys of the
`base` variant whose plan comes from the row, as `row_params_checks.py`), the candidates
its plan runs (per tunable segment, the `param_space` entry with its effective kernel
and fields) are built on its structure. The build must equal what `resolve` gives when
the tuned table holds `candidate_row` of those candidates, and run what the shipped row
runs: the same plan and metadata once each kernel name is replaced by the kernel that
runs (`Segment.effective_kernel`). Rows with a segment outside `param_space` are counted
and skipped (`row_params_checks.py` reports them). An arch with tuned rows on keys v2
serves but none built is a problem.

On the first `KEYS` keys of every structure the tuner lists for a path
(`tuning.candidates.structures`), its first `COMBINATIONS` combinations are built too: a built plan must equal what
`resolve` gives for their `candidate_row`, and a declined one must not be what dispatch
runs as tuned on that structure.

    python tests/moe_dispatch/build_checks.py --check [--arch gfx950] [-v]
"""

import argparse
import dataclasses
import functools
import itertools
import pathlib
import sys
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent
COMBINATIONS = 4
KEYS = 3


def _norm(value, names):
    if isinstance(value, functools.partial):
        func = getattr(value.func, "__qualname__", repr(value.func))
        return (func, _norm(value.args, names), _norm(value.keywords, names))
    if isinstance(value, dict):
        return tuple(sorted((k, _norm(v, names)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_norm(v, names) for v in value)
    if callable(value):
        return "<callable>"
    if isinstance(value, str):
        return names.get(value, value)
    return value


def fingerprint(problem, resolution, effective=False):
    """The plan and metadata of `resolution` as plain values (callables but partials
    reduced to a marker); with `effective`, kernel names replaced by what runs."""
    from aiter.moev2.dispatch.resolver import tunable
    from aiter.moev2.segments.base import Params

    plan = resolution.plan
    if plan is None:
        return ("declined", resolution.declined)
    names = {}
    if effective:
        for c in plan.segments:
            if tunable(c.segment) and "kernel" in c.params:
                full = Params(plan.params, **c.params)
                names[c.params["kernel"]] = c.segment.effective_kernel(problem, full)
    metadata = resolution.metadata
    return (
        tuple((c.segment.name, _norm(dict(c.params), names)) for c in plan.segments),
        _norm(dict(plan.params), names),
        plan.boundaries,
        plan.source,
        tuple(
            (f.name, _norm(getattr(metadata, f.name), names))
            for f in dataclasses.fields(metadata)
        ),
    )


def plan_candidates(problem, plan):
    from aiter.moev2.tuning.candidates import plan_candidates as candidates

    return candidates(problem, plan)


def _resolve_row(problem, path, row):
    """`resolve` with tuned `row` as the key's row."""
    from aiter.moev2.dispatch import resolver, tuned

    lookup = tuned.lookup
    tuned.lookup = lambda _: dict(row)
    try:
        return resolver.resolve(problem, path.segments)
    finally:
        tuned.lookup = lookup


def _check_combinations(problem, path, counts):
    """The errors of the first `COMBINATIONS` combinations of each structure the
    tuner lists for `problem` on `path`, for a structure's first `KEYS` keys
    (`counts["keys", path, structure]`)."""
    from aiter.moev2.dispatch import resolver
    from aiter.moev2.tuning.candidates import structures

    plans, _ = resolver.feasible_plans(problem, path.segments)
    names = [" | ".join(s.name for s in structure) for structure in plans]
    for name in names:
        counts["keys", path.name, name] += 1
    if all(counts["keys", path.name, name] > KEYS for name in names):
        return []
    found, _ = structures(problem, path.segments)
    errors = []
    for entry in found:
        if counts["keys", path.name, entry.name] > KEYS:
            continue
        for combination in itertools.islice(entry.combinations(), COMBINATIONS):
            built = resolver.build(problem, entry.structure, combination)
            row = resolver.candidate_row(entry.structure, combination)
            resolved = _resolve_row(problem, path, row)
            if built.plan is not None:
                if fingerprint(problem, built) == fingerprint(problem, resolved):
                    counts["combination built as resolved"] += 1
                    continue
                group, detail = "combination built, resolving its row differs", ""
            elif (
                resolved.plan is not None
                and resolved.plan.source == "tuned"
                and tuple(c.segment.name for c in resolved.plan.segments)
                == tuple(s.name for s in entry.structure)
            ):
                group = "combination declined, dispatch runs it as tuned"
                detail = f" ({built.declined})"
            else:
                counts["combination declined"] += 1
                continue
            errors.append((group, f"{entry.name}: {group}{detail}", built, resolved))
    return errors


def check_arch(arch, served, verbose=False):
    """`(counts, problems, lines)` for one simulated arch."""
    from aiter.moev2 import api
    from aiter.moev2.dispatch import resolver, tuned
    from moe_dispatch import arch_sim, snapshot
    from moe_dispatch.coverage_checks import _ROW_SECTIONS
    from moe_dispatch.differential import captured_logs

    counts, problems, lines, seen = Counter(), 0, [], set()
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
            paths = [path for path in served if path.takes(problem)]
            if paths and tuned.lookup(problem) is not None:
                counts["served keys with a tuned row"] += 1
            for path in paths:
                errors = _check_combinations(problem, path, counts)
                shipped = resolver.resolve(problem, path.segments)
                if shipped.plan is None or shipped.plan.source != "tuned":
                    pass
                elif isinstance(
                    candidates := plan_candidates(problem, shipped.plan), str
                ):
                    counts[f"skipped: {candidates} runs no candidate"] += 1
                else:
                    structure = tuple(c.segment for c in shipped.plan.segments)
                    built = resolver.build(problem, structure, candidates)
                    row = resolver.candidate_row(structure, candidates)
                    resolved = _resolve_row(problem, path, row)
                    if fingerprint(problem, built) != fingerprint(problem, resolved):
                        error = "differs from resolving its candidate row"
                    elif fingerprint(problem, built, True) != fingerprint(
                        problem, shipped, True
                    ):
                        error = "runs other kernels than the shipped row"
                    elif fingerprint(problem, built) != fingerprint(problem, shipped):
                        counts["same launches, other kernel names"] += 1
                        error = None
                    else:
                        counts["same plan"] += 1
                        error = None
                    if error is not None:
                        errors.append((f"build {error}", error, built, resolved))
                for group, error, built, resolved in errors:
                    problems += 1
                    counts[group] += 1
                    if verbose or problems <= 20:
                        lines.append(f"  {key.label()} [{path.name}]: {error}")
                        lines.append(f"    built:    {fingerprint(problem, built)}")
                        lines.append(f"    resolved: {fingerprint(problem, resolved)}")
    if counts["served keys with a tuned row"] and not counts["same plan"]:
        problems += 1
        lines.append("  no shipped row built")
    return counts, problems, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup()
    from moe_dispatch import arch_sim
    from moe_dispatch.ranking_checks import ranked_paths

    served = ranked_paths()
    problems = 0
    for arch in args.arch or list(arch_sim.ARCHS):
        counts, n, lines = check_arch(arch, served, args.verbose)
        problems += n
        print(f"{arch}: {n} problems")
        for group, count in sorted(
            (g, c) for g, c in counts.items() if isinstance(g, str)
        ):
            print(f"  {count:5d} {group}")
        for line in lines:
            print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
