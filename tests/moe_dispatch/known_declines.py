# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keys a v2 path takes and declines although the legacy code runs them, with reasons.

The coverage report fails on any other gap (no plan from the path's segments) where
legacy runs the key, and prints the count per entry. An entry names the paths, a regular
expression searched in the resolver's decline reason, the problems it covers and why
(with the todo it is tracked under).

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Known:
    name: str
    paths: frozenset
    reason: str  # searched (`re.search`) in the resolver's decline reason
    holds: Callable  # (problem) -> bool
    why: str


_NO_PLAN = r"(^no feasible plan|: no listed structure is feasible)$"
_MXFP4_PATHS = frozenset({"a16w4", "a16wi4", "a4w4", "a8w4", "mxfp8"})

KNOWN = (
    Known(
        "per_1x32 calls with activation scales or num_local_tokens (5.103)",
        _MXFP4_PATHS,
        _NO_PLAN,
        lambda p: bool(p.features & {"act_scales", "num_local_tokens"}),
        "the MXFP4 segments' caps leave out caller activation scales (a1_scale, "
        "a2_scale, also with fp8 hidden states) and num_local_tokens; not ported",
    ),
    Known(
        "a16wi4 with doweight_stage1 (5.103)",
        frozenset({"a16wi4"}),
        _NO_PLAN,
        lambda p: "doweight_stage1" in p.features,
        "no a16wi4 segment applies the routing weights in stage 1; not ported",
    ),
)


def find(path_name, problem, reason):
    """The entry covering a decline of `problem` by path `path_name`, or None."""
    for known in KNOWN:
        if (
            path_name in known.paths
            and re.search(known.reason, reason)
            and known.holds(problem)
        ):
            return known
    return None
