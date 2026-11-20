# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2.dispatch.coverage: gaps, tuned rows that do not run, unranked segments.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import pathlib
import subprocess
import sys
from dataclasses import replace

from aiter import ActivationType, QuantType, dtypes
from aiter.moev2.dispatch import coverage
from aiter.moev2.dispatch.ranking import Entry, match
from aiter.moev2.problem import MoeProblem

_HERE = pathlib.Path(__file__).resolve().parent


def _problem(
    quant=QuantType.per_Tensor, w=dtypes.fp8, g1u1=True, token=64, hidden_pad=0, **kw
):
    args = (token, 7168, 256, 256, 8, dtypes.bf16, w, w, quant, g1u1)
    args += (ActivationType.Silu, False, hidden_pad, 0)
    return MoeProblem.from_legacy_key(*args, arch="gfx950", cu_num=256, **kw)


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
    # Unshuffled fp8 CK weights on gfx950: the resolver raises, as legacy does.
    unshuffled = _problem(is_shuffled=False)
    assert coverage.classify(unshuffled, a8w8_pt.SEGMENTS) == ("raises", None)
    assert coverage.row_status(unshuffled, None) is None


def test_takes_rejects_the_keys_claims_rejects():
    from aiter.moev2 import api
    from aiter.moev2.migration import a4w4, a8w8_pt, fhmoe

    takes = a8w8_pt.PATH.takes
    assert takes(_problem())
    assert not takes(_problem(has_stage1_bias=True))
    assert not takes(_problem(hidden_pad=128))
    assert not takes(_problem(has_stage2_scatter=True))
    assert not takes(_problem(_exclude_mxmoe=True))
    takes, fp4 = a4w4.PATH.takes, dtypes.fp4x2
    assert takes(_problem(QuantType.per_1x32, fp4, has_stage1_bias=True))
    assert takes(_problem(QuantType.per_1x32, fp4, _exclude_mxmoe=True))
    assert not takes(_problem(QuantType.per_1x32, fp4, has_stage2_scatter=True))
    takes, base = fhmoe.PATH.takes, api.fhmoe_problem(_problem(QuantType.per_1x32, fp4))
    assert takes(base)
    for feature in ("act_scales", "num_local_tokens", "doweight_stage1"):
        assert not takes(replace(base, features=base.features | {feature}))


def test_a_segment_in_no_ranking_entry_fails_unless_tuned_only():
    from aiter.moev2.migration import a8w8_blockscale, a8w8_pt

    errors = coverage.segment_errors([a8w8_pt.PATH], [("p", _problem())], table=())
    assert "ck.gemm1: feasible for p but in no ranking entry; " in "".join(errors)
    assert not any(
        error.startswith(s.name + ":")
        for error in errors
        for s in a8w8_pt.SEGMENTS
        if s.tuned_only
    )
    path = a8w8_blockscale.PATH
    table = (
        Entry(
            "x",
            match(quant_type=QuantType.per_1x128),
            otherwise=["sort | quant.a1 | asm.stage1_1x128 | ck.gemm2"],
        ),
    )
    assert coverage.segment_errors([path], [], table=table) == [
        "asm.stage1_1x128: tuned_only, but in a ranking entry"
    ]


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


def test_the_ranking_names_no_tuned_only_segment():
    from aiter.moev2 import paths

    ranked = coverage.ranked_segments()
    segments = {s.name: s for p in paths.registered().values() for s in p.segments}
    assert not {name for name, s in segments.items() if s.tuned_only} & ranked
    assert {"sort", "ck.gemm1", "flydsl_fhmoe.gemm2"} <= ranked
