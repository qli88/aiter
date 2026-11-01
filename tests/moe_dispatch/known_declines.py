# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keys a v2 path takes and declines although the legacy code runs them, with reasons.

The coverage report (gaps) and the differential (every decline) fail on any other such
key. An entry names the paths, a part of the resolver's decline reason, the problems it
covers and why (with its todo in docs/fused_moe_revamp_todos.md); the reports print the
count per entry. Declined keys keep running on the legacy code, so a path that graduates
with them still runs them there.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Known:
    name: str
    paths: frozenset
    reason: str  # in the resolver's decline reason
    holds: Callable  # (problem) -> bool
    why: str


def _padded(problem):
    return bool(problem.layout.hidden_pad or problem.layout.intermediate_pad)


KNOWN = (
    Known(
        "g1u0 outside the 1-stage fmoe keys (5.3)",
        frozenset({"a16w16", "a8w8_ptok"}),
        "",
        lambda p: not p.layout.g1u1,
        "legacy runs CK 2-stage (bf16) or ASM stage 1 + CK stage 2 (int8), which read a "
        "g1u0 w1 as half-size gate/up: wrong output, a legacy bug no segment reproduces",
    ),
    Known(
        "CK 2-stage a16w4 (1f.2)",
        frozenset({"a16w4"}),
        "CK 2-stage a16w4 is not ported",
        lambda p: True,
        "not ported: needs a16w4 caps on ck.gemm1/2 (bf16 x fp4 in their module key)",
    ),
    Known(
        "CK 2-stage, padding ignored (1e.2, 5.10)",
        frozenset({"a4w4", "a8w4", "mxfp8"}),
        "CK 2-stage, padding ignored is not ported",
        _padded,
        "legacy runs CK, which takes no padding, on padded weights (a legacy bug)",
    ),
    Known(
        "SiTUv2 on CK-Tile split-K or CK (1e.2)",
        frozenset({"a4w4"}),
        "CK-Tile split-K MXFP4 or CK 2-stage SiTUv2 is not ported",
        lambda p: True,
        "not ported: SiTUv2 outside FlyDSL and MXMOE (reached under "
        "AITER_BYPASS_TUNE_CONFIG); CK has no SiTUv2 instance",
    ),
    Known(
        "a bypassed tuned row turns the MXMOE fallback off (1e.2)",
        frozenset({"a4w4"}),
        "the next branch (a tuned row is bypassed) is not ported",
        lambda p: True,
        "under AITER_BYPASS_TUNE_CONFIG legacy's MXMOE SiTUv2 fallback needs no tuned "
        "row, so a kept row sends the call to the next branch; not ported",
    ),
    Known(
        "padded keys on rows of kernels without padding (5.10)",
        frozenset({"a4w4"}),
        "no segments run ",
        _padded,
        "the row's MXMOE, CK or layout-v2 GEMM2 kernels take no padding; legacy runs "
        "them ignoring it (a legacy bug), the segments decline padding",
    ),
    Known(
        "CK-Tile alias rows outside the CK-Tile branches (5.12)",
        frozenset({"a16w4"}),
        "alias tuned row outside the CK-Tile branches",
        lambda p: True,
        "legacy runs ASM stage 1 under the swiglu_mxfp4_bf16_cktile alias name on "
        "weights not preshuffled (it reads the alias by position)",
    ),
    Known(
        "FHMoE on rows of kernels it has no segment for (5.28)",
        frozenset({"fhmoe"}),
        "no segment",
        lambda p: p.shared is not None,
        "legacy discards the row (MXMOE rows the aux kernels do not cover) or, under "
        "AITER_BYPASS_TUNE_CONFIG, ignores it after its discard checks, and runs the "
        "FlyDSL heuristic; path fhmoe has no segment of the row's family to apply them",
    ),
)


def find(path_name, problem, reason):
    """The entry covering a decline of `problem` by path `path_name`, or None."""
    for known in KNOWN:
        if path_name in known.paths and known.reason in reason and known.holds(problem):
            return known
    return None
