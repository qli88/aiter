# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Coverage of the v2 dispatch: what the paths take against what their segments run.

Over a set of problems (in CI, the snapshot key grid on every arch:
tests/moe_dispatch/coverage_checks.py):

- `classify`: how the resolver serves a problem a path takes: a plan, a gap (no plan
  from the path's segments), a ranking region that hands it to a named legacy branch,
  or a decline caused by its tuned row.
- `row_status`: what the resolver does with the problem's exact tuned row: used, or
  why not (legacy's discard warning, or the decline).
- `param_reasons`: for a plan a tuned row gives, whether each kernel segment's params
  are a tuner candidate (`Segment.params_reason`; in CI over the shipped rows:
  tests/moe_dispatch/row_params_checks.py).
- `segment_errors`: segments whose `reach` does not hold: "ranked" in no ranking entry,
  "tuned" or "hibernating" in one, "tuned" run by no shipped tuned row, "hibernating"
  run by one or without a GPU test.
"""

from ..segments.base import REACHES, Params
from . import tuned_table
from .ranking import _default_table, best
from .resolver import WARNING, resolve

KINDS = ("plan", "gap", "legacy", "row")


def ranked_segments(table=None) -> frozenset:
    """The names of the segments in some structure of `table`."""
    table = _default_table() if table is None else table
    return frozenset(
        name
        for entry in table
        for _, structures, _ in entry.lists()
        for structure in structures
        for name in structure
    )


def classify(problem, segments):
    """`(kind, resolution)` of `problem` resolved with `segments`, `kind` in `KINDS`.

    Without a feasible plan the resolver declines before the ranking; a problem without
    a tuned row whose region names a legacy branch is "legacy" all the same.
    """
    resolution = resolve(problem, segments)
    reason = resolution.declined
    if reason is None:
        return "plan", resolution
    if reason.startswith("ranking ") and reason.endswith(" is not ported"):
        return "legacy", resolution
    if reason == "no feasible plan" and tuned_table.lookup(problem).cells is None:
        hit = best(problem, [])
        if hit is not None and hit[2] is not None:
            return "legacy", resolution
    if reason in ("no feasible plan", "no default ranking entry applies") or (
        reason.startswith("ranking ")
        and reason.endswith(": no listed structure is feasible")
    ):
        return "gap", resolution
    return "row", resolution


def row_status(problem, resolution) -> str | None:
    """None without an exact tuned row; "used"; else why the row does not run."""
    if tuned_table.lookup(problem).cells is None:
        return None
    if resolution.declined is not None:
        return f"declined: {resolution.declined}"
    if resolution.plan.source == "tuned":
        return "used"
    if resolution.plan.source == "override":
        return "bypassed (AITER_BYPASS_TUNE_CONFIG)"
    warning = next((m for level, m in resolution.logs if level == WARNING), None)
    return f"discarded: {warning}" if warning else "discarded"


def param_reasons(problem, plan) -> list:
    """`(segment, reason)` for each segment of `plan` that runs a named kernel: its
    `params_reason` for its params over the plan's (None: a tuner candidate)."""
    return [
        (c.segment, c.segment.params_reason(problem, Params(plan.params, **c.params)))
        for c in plan.segments
        if c.segment.prefixes and "kernel" in c.params
    ]


def segment_errors(paths, table=None, used=None, tested=frozenset()) -> list:
    """Errors in the `reach` of the paths' segments.

    `used`: the names of the segments in the plan of some shipped tuned row that runs,
    over every arch; None skips the checks that need it. `tested`: the hibernating
    segments with a GPU test.
    """
    table = _default_table() if table is None else table
    ranked = ranked_segments(table)
    segments = {s.name: s for path in paths for s in path.segments}
    errors = []
    for name, s in sorted(segments.items()):
        if s.reach not in REACHES:
            errors.append(f"{name}: unknown reach {s.reach!r}; known: {REACHES}")
        elif s.reach == "ranked":
            if name not in ranked:
                errors.append(
                    f"{name}: in no ranking entry; rank it, or set reach to "
                    '"tuned" (only tuned rows run it) or "hibernating" (nothing does)'
                )
        elif name in ranked:
            errors.append(f'{name}: reach "{s.reach}", but in a ranking entry')
        elif s.reach == "tuned":
            if used is not None and name not in used:
                errors.append(
                    f'{name}: reach "tuned", but no shipped tuned row runs it; '
                    'set reach to "hibernating" or remove it'
                )
        else:
            if used is not None and name in used:
                errors.append(
                    f'{name}: reach "hibernating", but a shipped tuned row runs it; '
                    'set reach to "tuned"'
                )
            if name not in tested:
                errors.append(
                    f'{name}: reach "hibernating" without a GPU test; list one in '
                    "tests/moe_dispatch/hibernating.py"
                )
    return errors
