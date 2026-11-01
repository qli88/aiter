# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keys a v2 path takes and declines although the legacy code runs them, with reasons.

The coverage report (gaps) and the differential (declines) fail on any other such key.
An entry names the paths, the start of the resolver's decline reason, the problems it
covers and why; the counts per entry are printed with the reports.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Known:
    name: str
    paths: frozenset
    reason: str  # how the resolver's decline reason starts
    holds: Callable  # (problem) -> bool
    why: str


KNOWN = ()


def find(path_name, problem, reason):
    """The entry covering a decline of `problem` by path `path_name`, or None."""
    for known in KNOWN:
        if (
            path_name in known.paths
            and reason.startswith(known.reason)
            and known.holds(problem)
        ):
            return known
    return None
