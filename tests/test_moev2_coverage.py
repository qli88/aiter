# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2.dispatch.coverage: gaps, tuned rows that do not run, segment reach."""

import pathlib
import subprocess
import sys
from types import SimpleNamespace

from aiter import ActivationType, QuantType, dtypes
from aiter.moev2.dispatch import coverage
from aiter.moev2.dispatch.ranking import Entry, match
from aiter.moev2.problem import MoeProblem

_HERE = pathlib.Path(__file__).resolve().parent


def _problem(quant=QuantType.per_Tensor, w=dtypes.fp8, g1u1=True, token=64):
    args = (token, 7168, 256, 256, 8, dtypes.bf16, w, w, quant, g1u1)
    args += (ActivationType.Silu, False, 0, 0)
    return MoeProblem.from_legacy_key(*args, arch="gfx950", cu_num=256)


def test_the_coverage_report_passes_over_the_key_grid():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "coverage_checks.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


def test_classify_tells_plans_gaps_and_legacy_branches_apart():
    from aiter.moev2.migration import a8w8_pt, a16w16

    kind, resolution = coverage.classify(_problem(), a8w8_pt.SEGMENTS)
    assert kind == "plan" and resolution.plan is not None
    assert coverage.row_status(_problem(), resolution) is None
    # No a8w8_pt segment runs int8 per_1x32, and no ranking entry names a branch.
    kind, resolution = coverage.classify(
        _problem(QuantType.per_1x32, dtypes.i8), a8w8_pt.SEGMENTS
    )
    assert (kind, resolution.declined) == ("gap", "no feasible plan")
    # bf16 g1u0 on gfx950: no segment runs it; the ranking names legacy's CK branch.
    g1u0 = _problem(QuantType.No, dtypes.bf16, g1u1=False)
    assert coverage.classify(g1u0, a16w16.SEGMENTS)[0] == "legacy"


def _reach_errors(reach, *, ranked=False, used=None, tested=()):
    segment = SimpleNamespace(name="x.seg", reach=reach)
    table = (
        (Entry("x", match(quant_type=QuantType.per_1x128), otherwise=["x.seg"]),)
        if ranked
        else ()
    )
    return coverage.segment_errors(
        [SimpleNamespace(segments=[segment])], table=table, used=used, tested=tested
    )


def test_each_reach_is_checked():
    assert _reach_errors("ranked", ranked=True) == []
    assert "in no ranking entry" in _reach_errors("ranked")[0]
    assert _reach_errors("tuned", used={"x.seg"}) == []
    assert _reach_errors("tuned") == []  # usage not known: not checked
    assert "no shipped tuned row runs it" in _reach_errors("tuned", used=set())[0]
    assert "but in a ranking entry" in _reach_errors("tuned", ranked=True)[0]
    assert _reach_errors("hibernating", used=set(), tested={"x.seg"}) == []
    assert "a shipped tuned row runs it" in "".join(
        _reach_errors("hibernating", used={"x.seg"}, tested={"x.seg"})
    )
    assert "without a GPU test" in _reach_errors("hibernating", used=set())[0]
    assert "but in a ranking entry" in "".join(
        _reach_errors("hibernating", ranked=True, tested={"x.seg"})
    )
    assert "unknown reach" in _reach_errors("tuned_only")[0]


def test_the_registered_segments_hold_their_reach():
    from aiter.moev2 import paths

    registered = paths.registered().values()
    assert coverage.segment_errors(registered) == []
    ranked = coverage.ranked_segments()
    segments = {s.name: s for p in registered for s in p.segments}
    assert {"sort", "ck.gemm1", "flydsl_fhmoe.gemm2"} <= ranked
    assert {"asm.stage1_1x128", "impl.whole_graph", "sort.flat"} <= {
        name for name, s in segments.items() if s.reach == "tuned"
    }


def test_every_hibernating_segment_lists_its_test():
    sys.path.insert(0, str(_HERE))
    from moe_dispatch import hibernating

    from aiter.moev2 import paths

    segments = {s for p in paths.registered().values() for s in p.segments}
    assert set(hibernating.TESTS) == {
        s.name for s in segments if s.reach == "hibernating"
    }


def test_known_declines_match_path_reason_and_problem():
    sys.path.insert(0, str(_HERE))
    from moe_dispatch import known_declines

    from aiter.moev2 import paths

    registered = set(paths.registered())
    for known in known_declines.KNOWN:
        assert known.paths <= registered and known.why, known.name
    g1u0 = _problem(QuantType.No, dtypes.bf16, g1u1=False)
    reason = "ranking a16w16_g1u0[not fmoe_1stage]: CK 2-stage g1u0 is not ported"
    found = known_declines.find("a16w16", g1u0, reason)
    assert found is not None and "5.3" in found.name
    assert known_declines.find("a8w8_pt", g1u0, reason) is None
    assert (
        known_declines.find("a16w16", _problem(QuantType.No, dtypes.bf16), "x") is None
    )
    padding = "ranking a4w4[...]: CK 2-stage, padding ignored is not ported"
    assert known_declines.find("a4w4", _problem(), padding) is None
    # A g1u0 key declined for another reason is not a g1u0 decline.
    assert known_declines.find("a16w16", g1u0, "x") is None
    assert known_declines.find("a16w16", g1u0, "no feasible plan") is not None
    one_stage = MoeProblem.from_legacy_key(
        64, 7168, 256, 256, 8, dtypes.bf16, dtypes.bf16, dtypes.bf16, QuantType.No,
        False, ActivationType.Silu, False, 0, 0, arch="gfx942", cu_num=304,
    )  # fmt: skip
    assert known_declines.find("a16w16", one_stage, "no feasible plan") is None


def test_known_declines_name_the_row_families():
    sys.path.insert(0, str(_HERE))
    from moe_dispatch import known_declines

    padded = MoeProblem.from_legacy_key(
        64, 7168, 256, 256, 8, dtypes.bf16, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, True, ActivationType.Silu, False, 128, 0,
        arch="gfx950", cu_num=256,
    )  # fmt: skip
    layout = (
        "no segments run kernelName1=flydsl_moe1_x, kernelName2=flydsl_moe2_layout_y"
    )
    assert known_declines.find("a4w4", padded, layout) is not None
    mxmoe = "no segments run kernelName1=flydsl_mxmoe_g1_x, kernelName2=flydsl_moe2_y"
    assert known_declines.find("a4w4", padded, mxmoe) is not None
    plain = "no segments run kernelName1=flydsl_moe1_x, kernelName2=flydsl_moe2_y"
    assert known_declines.find("a4w4", padded, plain) is None
