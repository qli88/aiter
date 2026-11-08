# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 tuning pieces on the CPU: candidates per structure and `resolver.build`.

`build` against dispatch over the shipped tuned rows runs in a subprocess
(tests/moe_dispatch/build_checks.py); the rest are unit tests.
"""

import itertools
import pathlib
import subprocess
import sys

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.moev2.dispatch import resolver
from aiter.moev2.problem import MoeProblem
from aiter.moev2.segments.base import Params
from aiter.moev2.tuning import candidates as tc

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))


@pytest.fixture(autouse=True)
def _gfx950():
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx950"):
        yield


def _mx(q_a, activation=ActivationType.Silu, **kwargs):
    args = (64, 7168, 256, 256, 8, dtypes.bf16, q_a, dtypes.fp4x2, QuantType.per_1x32)
    args += (True, activation, False, 0, 0)
    kwargs.setdefault("is_shuffled", True)
    return MoeProblem.from_legacy_key(*args, **kwargs, arch="gfx950", cu_num=256)


def _a4w4_structures(filters=tc.NO_FILTERS):
    from aiter.moev2.migration import a4w4

    found, skipped = tc.structures(_mx(dtypes.fp4x2), a4w4.SEGMENTS, filters)
    return {s.name.replace("sort | quant.a1 | ", ""): s for s in found}, skipped


def test_build_runs_what_dispatch_runs_for_every_shipped_row():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "build_checks.py"), "--check"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


def test_structures_list_every_feasible_structure_with_candidates():
    from aiter.moev2.migration import a4w4

    p = _mx(dtypes.fp4x2)
    found, skipped = tc.structures(p, a4w4.SEGMENTS)
    plans, _ = resolver.feasible_plans(p, a4w4.SEGMENTS)
    assert {s.name for s in found} | skipped.keys() == {
        " | ".join(x.name for x in plan) for plan in plans
    }
    assert set(skipped.values()) == {
        "cktile.gemm1: no candidates",
        "cktile.gemm2: no candidates",
    }
    for s in found:
        tunables = [x for x in s.structure if resolver.tunable(x)]
        assert list(s.candidates) == [x.name for x in tunables]
        for x in tunables:
            assert s.candidates[x.name] == x.param_space(p)


def test_filters_keep_segments_families_and_kernel_names():
    ck = tc.Filters(only=frozenset({"ck"}))
    found, skipped = _a4w4_structures(ck)
    assert list(found) == ["ck.gemm1 | quant.inter | ck.gemm2"]
    assert skipped["sort | quant.a1 | ck.gemm1 | quant.inter | flydsl.gemm2"] == (
        "flydsl.gemm2: filtered out"
    )
    named = tc.Filters(only=frozenset({"flydsl.gemm1_q", "flydsl.gemm2"}))
    assert list(_a4w4_structures(named)[0]) == ["flydsl.gemm1_q | flydsl.gemm2"]
    # As legacy's, the exclude list filters the 2-stage ASM stage 1 and CK stage 2 only.
    found, skipped = _a4w4_structures(tc.Filters(exclude=("_",)))
    assert skipped["sort | quant.a1 | ck.gemm1 | quant.inter | ck.gemm2"] == (
        "ck.gemm2: filtered out"
    )
    assert found["flydsl.gemm1_q | flydsl.gemm2"] == (
        _a4w4_structures()[0]["flydsl.gemm1_q | flydsl.gemm2"]
    )
    # As legacy's, the regex applies to every stage's names.
    regex = tc.Filters(regex=r"^flydsl_moe1_|_t32x\d+x256_reduce$")
    found, _ = _a4w4_structures(regex)
    name = "flydsl.gemm1 | quant.inter | flydsl.gemm2"
    kernels = [c.kernel for c in found[name].candidates["flydsl.gemm2"]]
    assert kernels and all(k.endswith("_reduce") and "_t32x" in k for k in kernels)
    stage1 = _a4w4_structures()[0][name].candidates["flydsl.gemm1"]
    assert found[name].candidates["flydsl.gemm1"] == stage1
    assert "ck.gemm1 | quant.inter | ck.gemm2" not in found


def test_filters_from_the_legacy_tuner_env():
    env = {
        "TUNE_ONLY": "cktile,opus",
        "AITER_FMOE_TUNE_EXCLUDE_KERNELS": " _bnt2, ,_kw4",
        "TUNE_MOE_KERNEL_REGEX": "t32",
    }
    filters = tc.Filters.from_env(env)
    assert filters.only == frozenset(
        tc.TUNE_ONLY_SEGMENTS["cktile"] + tc.TUNE_ONLY_SEGMENTS["opus"]
    )
    assert filters.exclude == ("_bnt2", "_kw4")
    assert filters.regex == "t32"
    assert tc.Filters.from_env({}) == tc.NO_FILTERS
    assert tc.Filters.from_env({"TUNE_ONLY": ""}) == tc.NO_FILTERS
    opus = frozenset(tc.TUNE_ONLY_SEGMENTS["opus"])
    assert tc.Filters.from_env({"OPUS_ONLY": "1"}).only == opus
    assert tc.Filters.from_env({"OPUS_ONLY": "1", "TUNE_ONLY": "asm"}).only == (
        frozenset(tc.TUNE_ONLY_SEGMENTS["asm"])
    )
    skip = tc.Filters.from_env({"OPUS_SKIP_CKTILE": "1"})
    assert skip.drop == frozenset(tc.TUNE_ONLY_SEGMENTS["cktile"])
    assert "ck.gemm1 | quant.inter | ck.gemm2" not in _a4w4_structures(skip)[0]
    with pytest.raises(ValueError, match="TUNE_ONLY='flydslv3'"):
        tc.Filters.from_env({"TUNE_ONLY": "flydslv3"})


def test_combinations_agree_on_block_m_and_nt():
    a = (Params(block_m=32, nt=False), Params(block_m=64, nt=True), Params(block_m=128))
    b = (Params(block_m=32, nt=True), Params(block_m=32), Params(block_m=64))
    s = tc.StructureCandidates((), {"a": a, "b": b, "c": (Params(kernel="any"),)})
    assert s.block_ms() == (32, 64)
    combos = [(c["a"], c["b"]) for c in s.combinations()]
    assert combos == [(a[0], b[1]), (a[1], b[2])]
    assert [c["c"] for c in s.combinations()] == [Params(kernel="any")] * 2
    assert [(c["a"], c["b"]) for c in s.combinations(64)] == [(a[1], b[2])]


def test_candidate_row_holds_the_columns_dispatch_reads():
    found, _ = _a4w4_structures()
    s = found["ck.gemm1 | quant.inter | flydsl.gemm2"]
    combo = next(s.combinations(64))
    row = resolver.candidate_row(s.structure, combo)
    assert row == {
        "kernelName1": combo["ck.gemm1"].kernel,
        "kernelName2": combo["flydsl.gemm2"].kernel,
        "block_m": 64,
        "ksplit": 0,
        "nt": int(combo["ck.gemm1"].nt),
        "run_1stage": 0,
        "xbf16": 0,
        "flat": 0,
    }
    s = found["sort.flat | asm.fmoe_1stage"]
    combo = next(s.combinations())
    row = resolver.candidate_row(s.structure, combo)
    assert (row["kernelName2"], row["run_1stage"], row["flat"]) == ("", 1, 1)
    with pytest.raises(ValueError, match=r"missing for \['flydsl.gemm2'\]"):
        resolver.candidate_row(
            found["ck.gemm1 | quant.inter | flydsl.gemm2"].structure,
            {"ck.gemm1": combo["asm.fmoe_1stage"]},
        )
    with pytest.raises(ValueError, match=r"not tunable: \['sort'\]"):
        resolver.candidate_row(s.structure, {**combo, "sort": Params()})
    with pytest.raises(ValueError, match="disagree on block_m"):
        resolver.candidate_row(
            found["ck.gemm1 | quant.inter | flydsl.gemm2"].structure,
            {
                "ck.gemm1": Params(kernel="a", block_m=32),
                "flydsl.gemm2": Params(kernel="b", block_m=64),
            },
        )
    with pytest.raises(ValueError, match=r"no tuned-row column: \['tile'\]"):
        resolver.candidate_row(
            s.structure, {"asm.fmoe_1stage": Params(kernel="a", block_m=32, tile=1)}
        )


def test_build_declines_what_dispatch_declines():
    from aiter.moev2.migration import a8w4

    found, _ = _a4w4_structures()
    s = found["ck.gemm1 | quant.inter | ck.gemm2"]
    res = resolver.build(_mx(dtypes.fp4x2), s.structure, next(s.combinations()))
    assert res.plan is None
    assert res.declined == "tuned CK row for shuffled MXFP4 weights"
    s = found["flydsl.gemm1 | quant.inter | flydsl.gemm2"]
    combo = next(s.combinations())
    res = resolver.build(_mx(dtypes.fp4x2), s.structure, combo)
    assert res.plan is not None and res.plan.source == "tuned"
    assert res.metadata.stage1.keywords["kernelName"] == combo["flydsl.gemm1"].kernel
    stage2 = combo["flydsl.gemm2"]
    other = {**combo, "flydsl.gemm2": stage2.replace(kernel=stage2.kernel + "_x")}
    res = resolver.build(_mx(dtypes.fp4x2), s.structure, other)
    assert res.declined == "flydsl.gemm2 runs no candidate: not a candidate"
    # Only the fused-quant stage 1 writes the v2 intermediate some Opus GEMM2s read.
    p = _mx(dtypes.fp8)
    found, _ = tc.structures(p, a8w4.SEGMENTS)
    (s,) = [
        s for s in found if s.name.endswith("flydsl.gemm1 | quant.inter | opus.gemm2")
    ]
    reasons = {resolver.build(p, s.structure, c).declined for c in s.combinations(32)}
    assert None in reasons
    assert any(
        r and r.startswith("the chosen kernels disagree: opus.gemm2 reads")
        for r in reasons
    )


def test_build_declines_a_1stage_kernel_on_the_other_sort():
    # `param_space` lists the FLAT and non-FLAT kernels on both sorts; dispatch puts
    # a row on the sort matching its `flat`.
    from aiter.moev2.migration import a8w8_blockscale

    args = (1, 6144, 256, 257, 9, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_1x128, True, ActivationType.Silu, False, 0, 0)
    p = MoeProblem.from_legacy_key(*args, arch="gfx950", cu_num=256)
    found, _ = tc.structures(p, a8w8_blockscale.SEGMENTS)
    one_stage = [s for s in found if list(s.candidates) == ["asm.fmoe_1stage"]]
    assert {s.structure[0].name for s in one_stage} == {"sort", "sort.flat"}
    for s in one_stage:
        flat = s.structure[0].name == "sort.flat"
        flats = set()
        for combo in s.combinations():
            res = resolver.build(p, s.structure, combo)
            flats.add(bool(combo["asm.fmoe_1stage"]["flat"]))
            if bool(combo["asm.fmoe_1stage"]["flat"]) == flat:
                assert res.plan is not None, res.declined
            else:
                assert res.declined.startswith(
                    "dispatch would not put the row on this structure: "
                )
        assert flats == {False, True}


def test_build_carries_nt_and_ksplit_to_the_launches():
    from aiter.moev2.migration import a8w8_blockscale

    args = (64, 7168, 256, 256, 8, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_1x128, True, ActivationType.Silu, False, 0, 0)
    p = MoeProblem.from_legacy_key(*args, arch="gfx950", cu_num=256)
    found, _ = tc.structures(p, a8w8_blockscale.SEGMENTS)
    (s,) = [s for s in found if list(s.candidates) == ["ck.gemm1", "ck.gemm2"]]
    for nt, ksplit in ((False, 0), (False, 2), (True, 0), (True, 2)):
        combo = next(
            c
            for c in s.combinations()
            if c["ck.gemm1"]["nt"] is nt and c["ck.gemm1"]["ksplit"] == ksplit
        )
        res = resolver.build(p, s.structure, combo)
        stage1, stage2 = res.metadata.stage1.keywords, res.metadata.stage2.keywords
        assert stage1["use_non_temporal_load"] is stage2["use_non_temporal_load"] is nt
        assert stage1["splitk"] == res.metadata.ksplit == ksplit
        assert res.plan.params["ksplit"] == ksplit


def test_build_declines_the_layout_stages_of_an_fp8_gemm2_input():
    # Todo 2.34: dispatch runs the fp8 twins of the key's fp4 layout names.
    from aiter.moev2.migration import a4w4

    p = _mx(dtypes.fp4x2, q_dtype_a2=dtypes.fp8)
    found, _ = tc.structures(p, a4w4.SEGMENTS)
    for s in found:
        combos = itertools.islice(s.combinations(), 50)
        built = [resolver.build(p, s.structure, c) for c in combos]
        if "flydsl.gemm1_layout" in s.candidates:
            assert {res.declined for res in built} == {
                "flydsl.gemm1_layout runs no candidate: not a candidate"
            }
        elif "ck.gemm2" not in s.candidates or "ck.gemm1" not in s.candidates:
            assert any(res.plan is not None for res in built), s.name


def test_build_resolves_a_whole_graph_config():
    from aiter.moev2.migration import a16w16

    args = (16, 2048, 256, 64, 8, dtypes.bf16, dtypes.bf16, dtypes.bf16, QuantType.No)
    p = MoeProblem.from_legacy_key(
        *args, True, ActivationType.Silu, False, 0, 0, arch="gfx950", cu_num=256
    )
    found, _ = tc.structures(p, a16w16.SEGMENTS)
    (s,) = [s for s in found if s.name == "impl.whole_graph"]
    combo = next(s.combinations())
    res = resolver.build(p, s.structure, combo)
    assert res.plan.segments[0].params["kernel"] == combo["impl.whole_graph"].kernel
    assert res.metadata.full_impl is not None
    assert (res.metadata.stage1, res.metadata.stage2) == (None, None)


def test_check_output_passes_on_the_error_ratio_or_the_cosine_distance():
    import torch

    from aiter.moev2.tuning import measure as tm

    ref = torch.linspace(1.0, 2.0, 1000)
    assert tm.check_output(ref.clone(), ref) is None
    off = ref.clone()
    off[:50] += 1.0  # 5% of the elements wrong, cosine distance above 0.01
    assert tm.errors(off, ref)[1] > tm.COS_TOL
    assert tm.check_output(off, ref) is None
    scaled = ref * 1.05  # every element wrong, cosine distance about 1.2e-3
    assert tm.errors(scaled, ref)[0] == 1.0
    assert tm.check_output(scaled, ref) is None
    assert tm.check_output(-ref, ref).startswith("mismatch: error ratio 1 > 0.1")
    nan = ref.clone()
    nan[3] = float("nan")
    assert tm.check_output(nan, ref) == "non-finite output"
    assert tm.check_output(ref, nan) == "non-finite reference"
    assert tm.check_output(torch.zeros_like(ref), ref) == "all-zero output"
    assert tm.check_output(torch.zeros(4), torch.zeros(4)) is None


def test_stats_reports_the_median_and_the_spread():
    from aiter.moev2.tuning import measure as tm

    t = tm.stats([10.0, 12.0, 8.0, 30.0], "graph")
    assert (t.median_us, t.samples, t.mode) == (11.0, 4, "graph")
    assert t.cov == pytest.approx(77**0.5 / 15)
    assert tm.stats([5.0], "eager").cov == 0.0


def test_rotating_copies_exceed_the_cache_and_share_weights_that_do_not_fit():
    import torch

    from aiter.moev2.tuning import measure as tm

    w1 = torch.zeros(64 << 20, dtype=torch.uint8)
    w1.is_shuffled = True
    x = torch.zeros(64 << 10, dtype=torch.uint8)
    call = {"hidden_states": x, "w1": w1, "w2": w1, "topk": 8}
    assert tm.copy_count(call, free_bytes=1 << 34) == (4, True)
    assert tm.copy_count(call, free_bytes=1 << 20) == (tm.MAX_COPIES, False)
    calls = tm.rotations(call, 3, with_weights=False)
    assert calls[0] is call and len(calls) == 3
    assert all(c["w1"] is w1 and c["topk"] == 8 for c in calls)
    assert len({c["hidden_states"].data_ptr() for c in calls}) == 3
    copied = tm.rotations(call, 2, with_weights=True)[1]["w1"]
    assert copied.data_ptr() != w1.data_ptr() and copied.is_shuffled


def test_problems_and_params_pickle_into_another_process():
    import pickle

    from aiter.moev2.problem import FrozenMap

    p = Params(kernel="k", block_m=32)
    q = pickle.loads(pickle.dumps(p))
    assert type(q) is Params and q == p and q.kernel == "k"
    m = FrozenMap({"a": "x"})
    object.__setattr__(m, "_hash", 0)  # as hashed by a process with another seed
    assert hash(pickle.loads(pickle.dumps(m))) == hash(FrozenMap({"a": "x"}))
    problem = _mx(dtypes.fp4x2)
    assert pickle.loads(pickle.dumps(problem)) == problem


def test_a_task_measures_the_plan_it_names():
    from aiter.moev2.tuning import task as tt

    found, _ = _a4w4_structures()
    s = found["flydsl.gemm1 | quant.inter | flydsl.gemm2"]
    combo = next(s.combinations())
    names = tuple(x.name for x in s.structure)
    p = _mx(dtypes.fp4x2)
    res = tt.resolution(tt.Task(p, "a4w4", names, tuple(combo.items())))
    assert res.plan == resolver.build(p, s.structure, combo).plan
    assert res.plan.source == "tuned"
    resolved = tt.resolution(tt.Task(p, "a4w4"))
    assert resolved.plan == resolver.resolve(p, _path_segments("a4w4")).plan
    res = tt.resolution(tt.Task(p, "a4w4", ("sort", "nothing"), ()))
    assert res.plan is None and res.declined.startswith("no feasible structure")


def _path_segments(name):
    from aiter.moev2 import paths

    return paths.registered()[name].segments


def _toy(payload):
    import os
    import time

    from aiter.moev2.tuning.pool import Restart

    kind, value = payload
    if kind == "crash":
        os._exit(7)
    if kind == "hang":
        time.sleep(60)
    if kind == "raise":
        raise ValueError(value)
    if kind == "restart":
        raise Restart(value)
    return value, os.environ.get("HIP_VISIBLE_DEVICES"), os.getpid()


def test_the_pool_survives_crashes_hangs_and_restarts():
    from aiter.moev2.tuning.pool import Failed, Pool

    payloads = [("ok", 0), ("crash", 1), ("ok", 2), ("hang", 3), ("raise", "bad")]
    payloads += [("restart", 5), ("ok", 6), ("ok", 7)]
    with Pool(_toy, devices=("3", "4"), timeout=3) as pool:
        results = list(pool.map(payloads))
    out = dict(results)
    assert sorted(i for i, _ in results) == list(range(len(payloads)))
    assert out[1] == Failed("worker died (exit code 7)")
    assert out[3] == Failed("timed out after 3 s")
    assert out[4] == Failed("ValueError: bad")
    assert out[5] == 5
    ok = [out[i] for i in (0, 2, 6, 7)]
    assert [v for v, _, _ in ok] == [0, 2, 6, 7]
    assert {device for _, device, _ in ok} <= {"3", "4"}


class _DiesOnLoad:
    def __reduce__(self):
        import os

        return os._exit, (3,)


def test_the_pool_gives_up_on_dead_starts_and_drops_closed_maps():
    from aiter.moev2.tuning.pool import Pool

    dead = Pool(_DiesOnLoad(), start_timeout=2)
    with dead, pytest.raises(RuntimeError, match="no worker started in 2 s"):
        list(dead.map([("ok", 0)]))
    with Pool(_DiesOnLoad()) as dead:
        slot = dead.slots[0]
        slot.process.join()
        since = slot.started
        slot.restart()
        assert slot.started == since
    with Pool(_toy, devices=(None, None), timeout=60) as pool:
        first = pool.map([("hang", 0), ("ok", 1)])
        assert next(first)[0] == 1
        first.close()
        assert [slot.task for slot in pool.slots] == [None, None]
        second = pool.map([("ok", 2)])
        index, (value, _, _) = next(second)
        assert (index, value) == (0, 2)
        assert list(second) == []
        ready = next(slot for slot in pool.slots if slot.ready)
        since = ready.started
        ready.restart()
        assert ready.started > since


def test_inputs_decline_what_they_cannot_build():
    import dataclasses

    from aiter.moev2 import api
    from aiter.moev2.tuning import inputs as tin

    p = _mx(dtypes.fp4x2)
    big = dataclasses.replace(p, shape=dataclasses.replace(p.shape, m=65536))
    with pytest.raises(tin.Unsupported, match="M=65536 is not a dispatch bucket"):
        tin.build(big, device="cpu")
    with pytest.raises(
        tin.Unsupported, match="format .*float4_e2m1fn_x2 x .*, separated"
    ):
        tin._fhmoe_format(api.fhmoe_problem(p))
    with pytest.raises(tin.Unsupported, match="format .* separated"):
        tin._fhmoe_format(api.fhmoe_problem(_mx(dtypes.fp8)))
    args = (64, 7168, 384, 385, 7, dtypes.bf16, dtypes.fp8, dtypes.fp4x2)
    args += (QuantType.per_1x32, True, ActivationType.Silu, False, 0, 0)
    dsv4 = MoeProblem.from_legacy_key(
        *args, True, "interleave", arch="gfx950", cu_num=256
    )
    assert tin._fhmoe_format(api.fhmoe_problem(dsv4)) == "fp4"
    fp4_act = MoeProblem.from_legacy_key(
        *args[:6],
        dtypes.fp4x2,
        *args[7:],
        True,
        "interleave",
        arch="gfx950",
        cu_num=256
    )
    with pytest.raises(tin.Unsupported, match="format .*, interleave gates"):
        tin._fhmoe_format(api.fhmoe_problem(fp4_act))
    with pytest.raises(tin.Unsupported, match="no input builder for a shared expert"):
        tin._format(api.fhmoe_problem(dsv4))


def test_a_task_restarts_its_worker_when_the_device_is_gone(monkeypatch):
    from aiter.moev2.tuning import inputs as tin
    from aiter.moev2.tuning import measure as tm
    from aiter.moev2.tuning import task as tt
    from aiter.moev2.tuning.pool import Restart

    def no_builder(problem, seed):
        raise tin.Unsupported("no builder")

    task = tt.Task(_mx(dtypes.fp4x2), "a4w4")
    monkeypatch.setattr(tt, "_inputs", no_builder)
    assert tt.run(task).reason == "unsupported: no builder"
    monkeypatch.setattr(tt, "_inputs", lambda problem, seed: None)
    monkeypatch.setattr(tm, "measure", lambda *args: tm.Result("HIP error"))
    monkeypatch.setattr(tt, "_device_usable", lambda: True)
    assert tt.run(task).reason == "HIP error"
    monkeypatch.setattr(tt, "_device_usable", lambda: False)
    with pytest.raises(Restart) as raised:
        tt.run(task)
    assert raised.value.result.reason == "HIP error (device unusable)"

    def build_fault(problem, seed):
        raise RuntimeError("HIP fault")

    monkeypatch.setattr(tt, "_inputs", build_fault)
    with pytest.raises(Restart) as raised:
        tt.run(task)
    assert raised.value.result.reason == "RuntimeError: HIP fault (device unusable)"
