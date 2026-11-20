# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keys a v2 path takes and declines although the legacy code runs them, with reasons.

The coverage report (gaps) and the differential (every decline) fail on any other such
key. An entry names the paths, a regular expression searched in the resolver's decline
reason, the problems it covers and why (with the todo it is tracked under); the reports
print the count per entry. Declined keys keep running on the legacy code, so a path that
graduates with them still runs them there.

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


def _padded(problem):
    return bool(problem.layout.hidden_pad or problem.layout.intermediate_pad)


def _g1u0_outside_fmoe_1stage(problem):
    from aiter.moev2.dispatch.ranking import _fmoe_1stage

    return not problem.layout.g1u1 and not _fmoe_1stage(problem)


def _bypass_tuned():
    from aiter.moev2.dispatch.overrides import bypass_tuned

    return bypass_tuned() != 0


def _no_1stage_kernel_for(problem):
    return bool(
        problem.features & {"bias1", "bias2"}
        or _padded(problem)
        or problem.extras.get("exclude_mxmoe")
        or problem.activation.kind == "gelu"
    )


def _interleaved(problem):
    from aiter.ops.flydsl.moe_common import GateMode

    return problem.gate_mode == GateMode.INTERLEAVE


_NO_PLAN = r"(^no feasible plan|: no listed structure is feasible)$"
_NO_PLAN_OR_ROW = (
    r"(^no feasible plan$|: no listed structure is feasible$|^no segments run )"
)
# The rows whose kernels take no padding: MXMOE stage 1, layout-v2 FlyDSL or CK stage 2.
_UNPADDED_ROW = (
    r"^no segments run kernelName1=(flydsl_mxmoe_|\S+, "
    r"kernelName2=(flydsl_moe2_layout_|moe_ck2stages_))"
)
_MXFP4_PATHS = frozenset({"a16w4", "a16wi4", "a4w4", "a8w4", "mxfp8"})

KNOWN = (
    Known(
        "g1u0 on the CK or ASM 2-stage branch (5.3)",
        frozenset({"a16w16", "a8w8_ptok"}),
        r"g1u0 is not ported$",
        lambda p: not p.layout.g1u1,
        "legacy runs CK 2-stage (bf16) or ASM stage 1 + CK stage 2 (int8), which read a "
        "g1u0 w1 as half-size gate/up: wrong output, a legacy bug no segment reproduces",
    ),
    Known(
        "g1u0 keys no segment runs, outside the 1-stage fmoe keys (5.3)",
        frozenset({"a16w16", "a8w8_ptok"}),
        r"^no feasible plan$",
        _g1u0_outside_fmoe_1stage,
        "as above, on an arch without a g1u0 segment (int8 on gfx950): the resolver "
        "stops before the ranking region that names the legacy branch",
    ),
    Known(
        "CK 2-stage a16w4 (1f.2)",
        frozenset({"a16w4"}),
        r"CK 2-stage a16w4 is not ported$",
        lambda p: True,
        "not ported: needs a16w4 caps on ck.gemm1/2 (bf16 x fp4 in their module key)",
    ),
    Known(
        "CK 2-stage, padding ignored (1e.2, 5.10)",
        frozenset({"a4w4", "a8w4", "mxfp8"}),
        r"CK 2-stage, padding ignored is not ported$",
        _padded,
        "legacy runs CK, which takes no padding, on padded weights (a legacy bug)",
    ),
    Known(
        "SiTUv2 on CK-Tile split-K or CK (1e.2)",
        frozenset({"a4w4"}),
        r"CK-Tile split-K MXFP4 or CK 2-stage SiTUv2 is not ported$",
        lambda p: True,
        "not ported: SiTUv2 outside FlyDSL and MXMOE (reached without the MXMOE "
        "fallback: AITER_BYPASS_TUNE_CONFIG, AITER_MXMOE_FALLBACK=0 or a shape it "
        "rejects); CK has no SiTUv2 instance",
    ),
    Known(
        "a bypassed tuned row turns the MXMOE fallback off (1e.2)",
        frozenset({"a4w4"}),
        r"the next branch \(a tuned row is bypassed\) is not ported$",
        lambda p: True,
        "under AITER_BYPASS_TUNE_CONFIG legacy's MXMOE SiTUv2 fallback needs no tuned "
        "row, so a kept row sends the call to the next branch; not ported",
    ),
    Known(
        "padded keys on rows of kernels without padding (5.10)",
        frozenset({"a4w4", "a8w4", "mxfp8"}),
        _UNPADDED_ROW,
        _padded,
        "the row's MXMOE, CK or layout-v2 GEMM2 kernels take no padding; legacy runs "
        "them ignoring it (a legacy bug), the segments decline padding",
    ),
    Known(
        "CK-Tile alias rows outside the CK-Tile branches (5.12)",
        frozenset({"a16w4", "a4w4"}),
        r"^alias tuned row outside the CK-Tile branches$",
        lambda p: True,
        "legacy runs ASM stage 1 under the swiglu_mxfp4_bf16_cktile alias name on "
        "weights not preshuffled (it reads the alias by position)",
    ),
    Known(
        "FHMoE on MXMOE rows (5.28)",
        frozenset({"fhmoe"}),
        r"^no segments run kernelName1=flydsl_mxmoe_",
        lambda p: p.shared is not None,
        "legacy discards the row (MXMOE rows the aux kernels do not cover) and runs the "
        "FlyDSL heuristic; path fhmoe has no MXMOE segment to apply the discard checks",
    ),
    Known(
        "FHMoE under AITER_BYPASS_TUNE_CONFIG (5.28)",
        frozenset({"fhmoe"}),
        r"^no segments? runs? ",
        lambda p: p.shared is not None and _bypass_tuned(),
        "legacy ignores the row after its discard checks and runs the FlyDSL "
        "heuristic; path fhmoe has no segment of the row's family to apply them",
    ),
    Known(
        "padded a8w4 keys on Opus rows (1h.2)",
        frozenset({"a8w4"}),
        r"^no segments run kernelName1=\S+, kernelName2=opus_moe2_",
        _padded,
        "the Opus stages declare a8w4 caps without padding, which legacy passes to "
        "the Opus kernels; not ported",
    ),
    Known(
        "MXMOE a8w4 stage 1 with an Opus stage 2 (5.103)",
        frozenset({"a8w4"}),
        r"^no segments run kernelName1=flydsl_mxmoe_g1_a8w4_\S+, kernelName2=opus_moe2_",
        lambda p: True,
        "upstream dsv4 rows pair the two; no v2 plan joins them; not ported",
    ),
    Known(
        "1-stage rows for calls the 1-stage kernels cannot serve (5.101)",
        frozenset({"a4w4", "a16w4", "a8w8_ptok", "a8w8_blockscale"}),
        r"^no segment runs 1-stage ",
        _no_1stage_kernel_for,
        "legacy runs the row without the call's bias or padding, or retargets it to a "
        "Gelu kernel that may not exist (a legacy bug); under AITER_BYPASS_TUNE_CONFIG "
        "legacy runs 2-stage, and v2 still declines the row first",
    ),
    Known(
        "post-reduction tuned rows (5.100)",
        frozenset({"a4w4", "fhmoe"}),
        r"^post-reduction tuned row$",
        lambda p: True,
        "kept gap: v2 has no host sort into an empty buffer and reduce into the output",
    ),
    Known(
        "1-stage plan with a finite SiLU limit (5.100)",
        _MXFP4_PATHS,
        r"^1-stage plan with a SiLU limit \(legacy raises\)$",
        lambda p: True,
        "legacy `_fused_moe_impl` raises when it runs the plan, after the dispatch the "
        "differential records",
    ),
    Known(
        "a16w4 interleaved gate/up on FlyDSL rows (5.8)",
        frozenset({"a16w4"}),
        r"^no segments run kernelName1=flydsl_",
        _interleaved,
        "legacy runs separated-layout fp4 FlyDSL kernels on interleaved weights (a "
        "legacy bug); the FlyDSL segments decline the interleaved gate mode",
    ),
    Known(
        "per_1x32 calls with activation scales or num_local_tokens (5.103)",
        _MXFP4_PATHS,
        _NO_PLAN_OR_ROW,
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


# `LocalDispatch.relookup_decline`'s wording for a decline of the lookup after the sort.
_AFTER_SORT = re.compile(r"^the lookup after the sort declines \((.*)\)$")


def find(path_name, problem, reason):
    """The entry covering a decline of `problem` by path `path_name`, or None. A decline
    of the lookup after the sort is matched by that lookup's reason."""
    after_sort = _AFTER_SORT.match(reason)
    if after_sort is not None:
        reason = after_sort.group(1)
    for known in KNOWN:
        if (
            path_name in known.paths
            and re.search(known.reason, reason)
            and known.holds(problem)
        ):
            return known
    return None
