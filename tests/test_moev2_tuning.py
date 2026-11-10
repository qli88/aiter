# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 tuning on the CPU: candidates per structure, `resolver.build`, the
measuring pieces, the search over a fake measure, the plan file and the CLI.

`build` against dispatch over the shipped tuned rows runs in a subprocess
(tests/moe_dispatch/build_checks.py); the rest are unit tests.
"""

import dataclasses
import itertools
import pathlib
import subprocess
import sys
from collections import defaultdict

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.moev2.dispatch import convert, resolver
from aiter.moev2.problem import MoeProblem, ParallelSpec, ScaleSpec, TailSpec
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


def test_pool_workers_start_with_their_device(monkeypatch):
    import multiprocessing.context
    import os

    from aiter.moev2.tuning.pool import Pool

    started = []
    start = multiprocessing.context.SpawnProcess.start

    def recording(process):
        started.append(os.environ.get("HIP_VISIBLE_DEVICES"))
        start(process)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", recording)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    with Pool(_toy, devices=("3", "4", None), timeout=60):
        assert started == ["3", "4", "0,1,2,3"]
        assert os.environ["HIP_VISIBLE_DEVICES"] == "0,1,2,3"
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    with Pool(_toy, devices=("5",), timeout=60):
        assert started[-1] == "5" and "HIP_VISIBLE_DEVICES" not in os.environ


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
        cu_num=256,
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
    monkeypatch.setattr(tm, "measure", lambda *args, **kw: tm.Result("HIP error"))
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


def test_interleaved_alternates_the_order_and_keeps_failures_apart(monkeypatch):
    from types import SimpleNamespace

    from aiter.moev2.tuning import measure as tm

    runs = []

    def samples(fns, mode, iters, warmup, check):
        plan = fns[0][0]
        runs.append((plan, warmup))
        if plan == "raises":
            raise RuntimeError("HIP")
        if plan == "wrong":
            return [1.0], "copy 0, graph: mismatch"
        return [float(len(runs))], None

    monkeypatch.setattr(tm, "_calls", lambda inputs, free_bytes: ["c0", "c1"])
    monkeypatch.setattr(tm, "_verified", lambda inputs, r, err_ratio: r.declined)
    monkeypatch.setattr(tm, "plan_runner", lambda call, r: (r.plan, call))
    monkeypatch.setattr(tm, "_samples", samples)
    plans = ["a", "declined", "raises", "b", "wrong"]
    resolutions = [
        SimpleNamespace(plan=p, declined="declined: x" if p == "declined" else None)
        for p in plans
    ]
    results = tm.interleaved(None, resolutions, "graph", rounds=3, iters=5, warmup=7)
    assert runs == [
        ("a", 7),
        ("raises", 7),
        ("b", 7),
        ("wrong", 7),
        ("b", 0),
        ("a", 0),
        ("a", 0),
        ("b", 0),
    ]
    assert [r.reason for r in results] == [
        None,
        "declined: x",
        "RuntimeError: HIP",
        None,
        "copy 0, graph: mismatch",
    ]
    assert results[0].e2e.median_us == 6.0 and results[0].e2e.samples == 3
    assert results[3].e2e.median_us == 5.0

    def no_memory(inputs, free_bytes):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(tm, "_calls", no_memory)
    results = tm.interleaved(None, resolutions[:2], "graph")
    assert [r.reason for r in results] == ["RuntimeError: out of memory"] * 2


def test_rivals_give_a_result_per_plan_in_order(monkeypatch):
    from aiter.moev2.tuning import inputs as tin
    from aiter.moev2.tuning import measure as tm
    from aiter.moev2.tuning import task as tt
    from aiter.moev2.tuning.pool import Restart

    problem = _ptok()
    segments = _path_segments("a8w8_ptok")
    found, _ = tc.structures(problem, segments)
    plans = tuple(
        (tuple(s.name for s in sc.structure), tuple(next(sc.combinations()).items()))
        for sc in found[:2]
    )
    rivals = tt.Rivals(problem, "a8w8_ptok", ((None, None), *plans), rounds=2)
    seen = {}

    def interleaved(inputs, resolutions, mode, rounds, iters, warmup):
        seen.update(plans=[r.plan for r in resolutions], mode=mode, rounds=rounds)
        return [tm.Result(f"plan {i}") for i in range(len(resolutions))]

    monkeypatch.setattr(tt, "_inputs", lambda problem, seed: None)
    monkeypatch.setattr(tm, "interleaved", interleaved)
    monkeypatch.setattr(tt, "_device_usable", lambda: True)
    results = tt.run(rivals)
    assert [r.reason for r in results] == ["plan 0", "plan 1", "plan 2"]
    expected = [resolver.resolve(problem, segments).plan] + [
        tt.plan_of(problem, "a8w8_ptok", *p).plan for p in plans
    ]
    assert seen == {"plans": expected, "mode": "graph", "rounds": 2}
    monkeypatch.setattr(tt, "_device_usable", lambda: False)
    with pytest.raises(Restart) as raised:
        tt.run(rivals)
    assert [r.reason for r in raised.value.result] == [
        f"plan {i} (device unusable)" for i in range(3)
    ]

    def no_builder(problem, seed):
        raise tin.Unsupported("no builder")

    monkeypatch.setattr(tt, "_inputs", no_builder)
    assert [r.reason for r in tt.run(rivals)] == ["unsupported: no builder"] * 3


def _ptok(m=64):
    args = (m, 4096, 1024, 32, 4, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_Token, True, ActivationType.Silu, False, 0, 0)
    return MoeProblem.from_legacy_key(
        *args, input_dtype=dtypes.bf16, arch="gfx950", cu_num=256
    )


def _us(params):
    import zlib

    return 10.0 + zlib.crc32(params.kernel.encode()) % 997 / 10.0


class _FakeMeasure:
    """`evaluate` for `search.tune`: a kernel's stage time is `_us` of it; a plan's end
    to end time the sum of its kernels' plus 3 us per segment (or `e2e(plan)` if
    given; a stage's `stage_us(candidate)` if given). Kernels in `fail` fail any run,
    in `fail_e2e` only end-to-end runs, in `fail_level1` only level-1 runs;
    `rivals(task)`, if given, answers `Rivals` tasks."""

    def __init__(
        self, fail=(), fail_e2e=(), e2e=None, rivals=None, stage_us=_us, fail_level1=()
    ):
        from aiter.moev2.migration import a8w8_ptok
        from aiter.moev2.tuning import search

        self.stage = {s.name: search.stage(s) for s in a8w8_ptok.SEGMENTS}
        self.fail, self.fail_e2e, self.e2e = set(fail), set(fail_e2e), e2e
        self.rivals, self.stage_us = rivals, stage_us
        self.fail_level1 = set(fail_level1)
        self.tasks = []

    def plan_us(self, structure, combo):
        if self.e2e is not None:
            return self.e2e(structure, combo)
        return sum(_us(c) for c in combo.values()) + 3.0 * len(structure)

    def result(self, task, structure, combo):
        from aiter.moev2.tuning.measure import Result, Timing

        timed_e2e = getattr(task, "e2e", True)
        segments = getattr(task, "segments", ())
        kernels = {c.kernel for c in combo.values()}
        if kernels & self.fail:
            return Result("mismatch: fake")
        if timed_e2e and kernels & self.fail_e2e:
            return Result("mismatch: fake end to end")
        if not timed_e2e and kernels & self.fail_level1:
            return Result("mismatch: fake level 1")
        mode = "graph"
        stages = {
            self.stage[name]: Timing(self.stage_us(c), 0.01, 10, mode)
            for name, c in combo.items()
            if segments and self.stage[name] in segments
        }
        e2e = Timing(self.plan_us(structure, combo), 0.01, 10, mode)
        return Result(e2e=e2e if timed_e2e else None, stages=stages)

    def __call__(self, tasks):
        from aiter.moev2.tuning.task import Rivals

        out = []
        for task in tasks:
            self.tasks.append(task)
            if isinstance(task, Rivals) and self.rivals is not None:
                out.append(self.rivals(task))
            elif isinstance(task, Rivals):
                out.append([self.result(task, s, dict(c)) for s, c in task.plans])
            else:
                out.append(self.result(task, task.structure, dict(task.candidates)))
        return out


def _best_by_brute_force(problem, fake):
    from aiter.moev2.migration import a8w8_ptok

    found, _ = tc.structures(problem, a8w8_ptok.SEGMENTS)
    best = {}
    for sc in found:
        times = [
            fake.plan_us(tuple(s.name for s in sc.structure), combo)
            for combo in sc.combinations()
            if not {c.kernel for c in combo.values()} & (fake.fail | fake.fail_e2e)
            and resolver.build(problem, sc.structure, combo).plan is not None
        ]
        if times:
            best[tuple(s.name for s in sc.structure)] = min(times)
    return best


def test_the_search_finds_the_fastest_plan_of_every_structure():
    from aiter.moev2.tuning import search
    from aiter.moev2.tuning.task import Task

    problem, fake = _ptok(), _FakeMeasure()
    settings = search.Settings(finalists_graph=3)
    outcomes = search.tune(problem, "a8w8_ptok", fake, settings)
    expected = _best_by_brute_force(problem, fake)
    ranked = [o for o in outcomes if o.rank is not None]
    assert len(expected) == 3 and len(ranked) == 3
    assert {o.structure: o.e2e.median_us for o in ranked} == pytest.approx(expected)
    assert [o.rank for o in ranked] == [1, 2, 3]
    assert ranked[0].e2e.median_us == min(expected.values())
    found, _ = tc.structures(problem, _path_segments("a8w8_ptok"))
    by_names = {tuple(s.name for s in sc.structure): sc.structure for sc in found}
    e2e_runs = {}
    for task in fake.tasks:
        built = resolver.build(problem, by_names[task.structure], dict(task.candidates))
        assert built.plan is not None
        assert isinstance(task, Task) and bool(task.segments) != task.e2e
        if task.e2e:
            e2e_runs[task.structure] = e2e_runs.get(task.structure, 0) + 1
    two_stage = [s for s in by_names if len(s) > 2]
    assert all(e2e_runs[s] <= settings.finalists_graph + 1 for s in two_stage)
    sums = {o.structure: o.sum_us for o in ranked if len(o.candidates) == 2}
    assert sums and sums == pytest.approx(
        {
            o.structure: sum(_us(c) for c in o.candidates.values())
            for o in ranked
            if len(o.candidates) == 2
        }
    )


def test_the_search_retries_bases_and_finalists_that_fail():
    from aiter.moev2.tuning import search

    problem, settings = _ptok(), search.Settings(finalists_graph=3)
    first = _FakeMeasure()
    search.tune(problem, "a8w8_ptok", first, settings)
    bases = [t for t in first.tasks if t.segments and len(t.segments) == 2]
    failing_base = {bases[0].candidates[0][1].kernel}
    fake = _FakeMeasure(fail=failing_base)
    outcomes = search.tune(problem, "a8w8_ptok", fake, settings)
    retried = [t for t in fake.tasks if t.segments and len(t.segments) == 2]
    failed = [t for t in retried if {c.kernel for _, c in t.candidates} & failing_base]
    assert len(failed) == 1 and len(retried) == len(bases) + 1
    assert (
        retried.index(failed[0]) < len(bases) <= retried.index(failed[0]) + len(bases)
    )
    expected = _best_by_brute_force(problem, fake)
    ranked = {o.structure: o.e2e.median_us for o in outcomes if o.rank is not None}
    assert ranked == pytest.approx(expected)

    two_stage = [o for o in outcomes if o.rank is not None and len(o.candidates) == 2]
    winner = min(two_stage, key=lambda o: o.rank)
    kernels = {c.kernel for c in winner.candidates.values()}
    fake = _FakeMeasure(fail_e2e=kernels)
    outcomes = search.tune(problem, "a8w8_ptok", fake, settings)
    again = next(o for o in outcomes if o.structure == winner.structure)
    assert (
        again.rank is not None
        and not {c.kernel for c in again.candidates.values()} & kernels
    )
    runs = [t for t in fake.tasks if t.e2e and t.structure == winner.structure]
    assert settings.finalists_graph < len(runs) <= 2 * settings.finalists_graph + 1

    clean = search.tune(problem, "a8w8_ptok", _FakeMeasure())
    unranked = {o.structure: o.reason for o in clean if o.rank is None}
    assert unranked[("sort.flat", "asm.fmoe_1stage")].startswith(
        "no passing plan: declined: dispatch would not put the row on this structure"
    )
    assert all(unranked.values())
    every = _FakeMeasure(fail={c.kernel for t in first.tasks for _, c in t.candidates})
    outcomes = search.tune(problem, "a8w8_ptok", every)
    assert all(o.rank is None for o in outcomes)
    assert {o.structure: o.reason for o in outcomes if o.structure in unranked} == (
        unranked
    )
    ranked_before = [o for o in outcomes if o.structure not in unranked]
    assert {o.reason.split(" (and")[0] for o in ranked_before} == {
        "no passing plan: mismatch: fake",
        "no level 1 result: base: mismatch: fake",
    }


def test_the_search_times_near_ties_again_and_prefers_fewer_segments():
    from aiter.moev2.tuning import search
    from aiter.moev2.tuning.measure import Result, Timing
    from aiter.moev2.tuning.task import Rivals

    problem, every = _ptok(), search.Settings(max_rivals=1000)

    def even(structure, combo):
        return 100.0

    fake = _FakeMeasure(e2e=even)
    outcomes = search.tune(problem, "a8w8_ptok", fake, every)
    rivals = [t for t in fake.tasks if isinstance(t, Rivals)]
    runs = [t for t in fake.tasks if getattr(t, "e2e", False)]
    assert len(rivals) == 1 and len(rivals[0].plans) == len(runs)
    ranked = [o for o in outcomes if o.rank is not None]
    assert len(ranked) == 3
    assert [len(o.structure) for o in ranked] == sorted(
        len(o.structure) for o in ranked
    )
    assert all(
        f"timed again in turn with {len(runs) - 1} more" in o.notes for o in ranked
    )

    capped = _FakeMeasure(e2e=even)
    search.tune(problem, "a8w8_ptok", capped, search.Settings(max_rivals=2))
    (rivals,) = [t for t in capped.tasks if isinstance(t, Rivals)]
    assert len(rivals.plans) == 2 and len({s for s, _ in rivals.plans}) == 2

    fake = _FakeMeasure(e2e=lambda structure, combo: 100.0 * len(structure))
    spread = search.tune(problem, "a8w8_ptok", fake, every)
    (rivals,) = [t for t in fake.tasks if isinstance(t, Rivals)]
    assert {s for s, _ in rivals.plans} == {spread[0].structure}
    assert all("timed again" not in o.notes for o in spread[1:])

    crashed = _FakeMeasure(e2e=even, rivals=lambda task: Result("worker died"))
    again = search.tune(problem, "a8w8_ptok", crashed, every)
    assert [o.structure for o in again] == [o.structure for o in outcomes]
    ranked = [o for o in again if o.rank is not None]
    assert len(ranked) == 3
    assert all("timing again in turn failed: worker died" in o.notes for o in ranked)
    for answer, note in (
        (lambda task: [Result("no copies")] * len(task.plans), "no copies"),
        (lambda task: [], "0 results for"),
    ):
        again = search.tune(problem, "a8w8_ptok", _FakeMeasure(e2e=even, rivals=answer))
        ranked = [o for o in again if o.rank is not None]
        assert len(ranked) == 3
        assert all(f"timing again in turn failed: {note}" in o.notes for o in ranked)

    def first_fails(task):
        timing = Timing(100.0, 0.01, 10, "graph")
        return [Result("mismatch: once"), *(Result(e2e=timing) for _ in task.plans[1:])]

    once = _FakeMeasure(e2e=even, rivals=first_fails)
    again = search.tune(problem, "a8w8_ptok", once, every)
    (rivals,) = [t for t in once.tasks if isinstance(t, Rivals)]
    kept = next(o for o in again if o.structure == rivals.plans[0][0])
    assert kept.rank is not None and kept.candidates != dict(rivals.plans[0][1])
    assert "another plan failed timed again: mismatch: once" in kept.notes

    def by_cov(low):
        def answer(task):
            return [
                Result(e2e=Timing(100.0, 0.01 if low in s else 0.05, 10, "graph"))
                for s, _ in task.plans
            ]

        return answer

    for low, high in (("asm.stage1", "ck.gemm1"), ("ck.gemm1", "asm.stage1")):
        again = search.tune(
            problem, "a8w8_ptok", _FakeMeasure(e2e=even, rivals=by_cov(low)), every
        )
        five = [
            o.structure for o in again if o.rank is not None and len(o.structure) == 5
        ]
        assert low in five[0] and high in five[1]

    def first_structure_fails(task):
        timing = Timing(100.0, 0.01, 10, "graph")
        return [
            Result("mismatch: fake") if s == task.plans[0][0] else Result(e2e=timing)
            for s, _ in task.plans
        ]

    one_failing = _FakeMeasure(e2e=even, rivals=first_structure_fails)
    again = search.tune(problem, "a8w8_ptok", one_failing, every)
    (rivals,) = [t for t in one_failing.tasks if isinstance(t, Rivals)]
    dropped = next(o for o in again if o.structure == rivals.plans[0][0])
    assert dropped.rank is None and dropped.reason == "timed again: mismatch: fake"
    assert dropped.candidates == dict(rivals.plans[0][1]) and dropped.notes == ""
    assert dropped.e2e.median_us == 100.0
    assert [o.rank for o in again if o.rank is not None] == [1, 2]


def test_the_search_times_near_ties_of_one_structure_again():
    from aiter.moev2.tuning import search
    from aiter.moev2.tuning.measure import Result, Timing
    from aiter.moev2.tuning.task import Rivals

    problem = _ptok()

    def level2(structure, combo):
        return 100.0 + sum(_us(c) for c in combo.values()) / 1000

    def reversed_(task):
        return [
            Result(e2e=Timing(200.0 - level2(s, dict(c)), 0.01, 10, "graph"))
            for s, c in task.plans
        ]

    fake = _FakeMeasure(e2e=level2, rivals=reversed_)
    outcomes = search.tune(problem, "a8w8_ptok", fake, search.Settings(max_rivals=1000))
    (rivals,) = [t for t in fake.tasks if isinstance(t, Rivals)]
    runs = defaultdict(list)
    for t in fake.tasks:
        if getattr(t, "e2e", False):
            runs[t.structure].append(level2(t.structure, dict(t.candidates)))
    assert len(rivals.plans) == sum(map(len, runs.values()))
    several = [s for s, us in runs.items() if len(us) > 1]
    assert several
    for o in outcomes:
        if o.rank is not None and o.structure in several:
            assert level2(o.structure, o.candidates) == max(runs[o.structure])
            assert o.e2e.median_us == pytest.approx(200.0 - max(runs[o.structure]))


def test_the_search_runs_the_current_plan_besides_the_finalists(monkeypatch):
    import re

    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned
    from aiter.moev2.tuning import search
    from aiter.moev2.tuning.measure import Result, Timing

    monkeypatch.delenv(envs.AITER_MOEV2_TUNED_PLANS, raising=False)
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG", raising=False)
    tuned.clear_cache()
    # A key with a shipped tuned row on a two-stage CK plan.
    args = (1, 5120, 1536, 40, 8, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_Token, True, ActivationType.Silu, False, 0, 0)
    problem = MoeProblem.from_legacy_key(
        *args, input_dtype=dtypes.bf16, arch="gfx950", cu_num=256
    )
    found, _ = tc.structures(problem, _path_segments("a8w8_ptok"))
    (i, today), why = search.current(problem, "a8w8_ptok", found)
    names = tuple(s.name for s in found[i].structure)
    assert why is None and len(names) == 5
    kernels = {c.kernel for c in today.values()}
    settings = search.Settings(finalists_graph=3)

    def fast_today(structure, combo):
        return 50.0 if combo == today else 100.0

    def slow_alone(c):
        return 1000.0 if c.kernel in kernels else _us(c)

    fake = _FakeMeasure(e2e=fast_today, stage_us=slow_alone)
    outcomes = search.tune(problem, "a8w8_ptok", fake, settings)
    winner = outcomes[0]
    assert winner.structure == names and winner.candidates == today
    assert search.CURRENT_NOTE in winner.notes.split("; ")
    runs = [dict(t.candidates) for t in fake.tasks if t.e2e and t.structure == names]
    assert len(runs) == settings.finalists_graph + 1 and runs.count(today) == 1

    lead = next(run for run in runs if run != today)
    broken = next(c.kernel for c in lead.values() if c.kernel not in kernels)
    refill = _FakeMeasure(e2e=fast_today, stage_us=slow_alone, fail_e2e={broken})
    search.tune(problem, "a8w8_ptok", refill, settings)
    passing = [
        dict(t.candidates)
        for t in refill.tasks
        if t.e2e
        and t.structure == names
        and broken not in {c.kernel for _, c in t.candidates}
    ]
    assert len(passing) == settings.finalists_graph + 1 and today in passing

    plan = resolver.resolve(problem, _path_segments("a8w8_ptok")).plan
    assert tc.plan_candidates(problem, plan) == today
    choice = next(c for c in plan.segments if c.segment.name == "ck.gemm1")
    other = 64 if today["ck.gemm1"]["block_m"] != 64 else 32
    moved = dataclasses.replace(choice, params=Params(choice.params, block_m=other))
    segments = tuple(moved if c is choice else c for c in plan.segments)
    assert (
        tc.plan_candidates(problem, dataclasses.replace(plan, segments=segments))
        == "ck.gemm1"
    )

    def fast_alone(c):
        return 1.0 if c.kernel in kernels else _us(c)

    first = _FakeMeasure(stage_us=fast_alone)
    search.tune(problem, "a8w8_ptok", first, settings)
    runs = [dict(t.candidates) for t in first.tasks if t.e2e and t.structure == names]
    assert len(runs) == settings.finalists_graph + 1 and runs.count(today) == 1

    gemm1 = today["ck.gemm1"].kernel
    others = tc.Filters(regex=f"^(?!{re.escape(gemm1)}$)")
    outcomes = search.tune(problem, "a8w8_ptok", _FakeMeasure(), settings, others)
    assert f"{search.CURRENT_NOTE}: ck.gemm1 {gemm1} is filtered out" in (
        outcomes[0].notes.split("; ")
    )

    failing = _FakeMeasure(fail_e2e={gemm1})
    outcomes = search.tune(problem, "a8w8_ptok", failing, settings)
    assert f"{search.CURRENT_NOTE}: failed: mismatch: fake end to end" in outcomes[
        0
    ].notes.split("; ")

    def slow_today(structure, combo):
        return 100.4 if combo == today else 100.0

    outcomes = search.tune(problem, "a8w8_ptok", _FakeMeasure(e2e=slow_today), settings)
    assert outcomes[0].candidates != today
    assert f"{search.CURRENT_NOTE}: 100.4 us" in outcomes[0].notes.split("; ")

    def today_fails(task):
        timing = Timing(100.0, 0.01, 10, "graph")
        return [
            Result("mismatch: fake again") if dict(c) == today else Result(e2e=timing)
            for _, c in task.plans
        ]

    again = _FakeMeasure(e2e=slow_today, rivals=today_fails)
    outcomes = search.tune(problem, "a8w8_ptok", again, settings)
    assert f"{search.CURRENT_NOTE}: failed timed again: mismatch: fake again" in (
        outcomes[0].notes.split("; ")
    )

    alone = _FakeMeasure(
        fail_level1={c.kernel for cs in found[i].candidates.values() for c in cs}
    )
    outcomes = search.tune(problem, "a8w8_ptok", alone, settings)
    only = next(o for o in outcomes if o.structure == names)
    assert only.candidates == today and search.CURRENT_NOTE in only.notes.split("; ")
    assert any(
        n.startswith("only the current plan ran, no level 1")
        for n in only.notes.split("; ")
    )
    alone = _FakeMeasure(
        fail_level1=alone.fail_level1, e2e=slow_today, rivals=today_fails
    )
    outcomes = search.tune(problem, "a8w8_ptok", alone, settings)
    dropped = next(o for o in outcomes if o.structure == names)
    assert dropped.rank is None and dropped.candidates == today
    assert dropped.reason == "timed again: mismatch: fake again"
    assert dropped.notes == search.CURRENT_NOTE

    real = resolver.resolve

    def changed(plan_params=None, gemm1_params=None, ksplit=None):
        def resolve(problem, segments):
            resolution = real(problem, segments)
            plan, metadata = resolution.plan, resolution.metadata
            if plan_params is not None:
                params = Params(plan.params, **plan_params)
                plan = dataclasses.replace(plan, params=params)
            if gemm1_params is not None:
                segments = tuple(
                    (
                        dataclasses.replace(c, params=Params(c.params, **gemm1_params))
                        if c.segment.name == "ck.gemm1"
                        else c
                    )
                    for c in plan.segments
                )
                plan = dataclasses.replace(plan, segments=segments)
            if ksplit is not None:
                metadata = dataclasses.replace(metadata, ksplit=ksplit)
            return dataclasses.replace(resolution, plan=plan, metadata=metadata)

        return resolve

    assert real(problem, _path_segments("a8w8_ptok")).metadata.ksplit <= 1
    for same in (
        changed(plan_params={"ksplit": 7, "tuned": True}),
        changed(gemm1_params={"tuned": True}),
        changed(ksplit=-1),
        changed(ksplit=1),
    ):
        monkeypatch.setattr(resolver, "resolve", same)
        assert search.current(problem, "a8w8_ptok", found) == ((i, today), None)
    monkeypatch.setattr(resolver, "resolve", changed(ksplit=7))
    assert search.current(problem, "a8w8_ptok", found) == (
        None,
        "its candidates build another plan than dispatch runs",
    )


def test_the_search_declines_what_it_cannot_measure():
    from aiter.moev2.tuning import search

    fake = _FakeMeasure()
    (outcome,) = search.tune(_ptok(m=65), "a8w8_ptok", fake)
    assert outcome.structure == () and outcome.rank is None
    assert outcome.reason.startswith("unsupported: M=65 is not a dispatch bucket")
    assert not fake.tasks


def test_level_one_groups_and_base_order():
    from aiter.moev2.tuning import search

    a = Params(block_m=32, nt=False, kernel="a")
    b = Params(block_m=32, nt=True, kernel="b")
    c = Params(block_m=64, nt=False, kernel="c")
    any_bm = Params(kernel="d")
    groups = dict(search.groups({"s1": (a, b, c), "s2": (any_bm,)}))
    assert groups == {
        (32, False): {"s1": [a], "s2": [any_bm]},
        (32, True): {"s1": [b], "s2": [any_bm]},
        (64, False): {"s1": [c], "s2": [any_bm]},
    }
    order = list(search.base_order({"s1": [a, b, c], "s2": [a, b]}))
    assert order[:3] == [(0, 0), (1, 1), (2, 0)]
    assert sorted(order) == sorted(itertools.product(range(3), range(2)))


def test_level_one_keeps_the_top_k_and_those_within_eps():
    from aiter.moev2.tuning import search

    timed = [(10.3, "d"), (10.0, "a"), (11.0, "e"), (10.25, "c"), (10.2, "b")]
    kept = search.keep(timed, search.Settings(top_k=1, eps=0.03))
    assert [c for _, c in kept] == ["a", "b", "c", "d"]
    kept = search.keep(timed, search.Settings(top_k=2, eps=0.0))
    assert [c for _, c in kept] == ["a", "b"]
    assert search.keep([], search.Settings()) == []


def test_the_search_notes_an_inline_sort_winner(monkeypatch):
    from aiter.moev2.tuning import search

    outcomes = search.tune(_ptok(), "a8w8_ptok", _FakeMeasure())
    assert not any(search.INLINE_SORT_NOTE in o.notes for o in outcomes)
    monkeypatch.setattr(search, "_is_mxfp4_inline_sort", lambda metadata: True)
    outcomes = search.tune(_ptok(), "a8w8_ptok", _FakeMeasure())
    ranked = [o for o in outcomes if o.rank is not None]
    assert ranked and all(search.INLINE_SORT_NOTE in o.notes for o in ranked)


def test_the_cli_writes_winners_logs_every_structure_and_resumes(tmp_path, monkeypatch):
    import csv

    from aiter.jit.utils import chip_info
    from aiter.moev2.dispatch import tuned_plans
    from aiter.moev2.tuning import cli, search

    class NoPool:
        def __init__(self, handler, devices, timeout):
            assert devices == ["3", "4"] and timeout == 60.0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    tuned_keys = []

    def tune(problem, path, evaluate, settings, filters, log):
        tuned_keys.append((problem, path, settings.top_k))
        return search.tune(problem, path, _FakeMeasure(), settings, filters)

    monkeypatch.setattr(cli, "Pool", NoPool)
    monkeypatch.setattr(cli, "tune", tune)
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    keys = tmp_path / "keys.csv"
    keys.write_text(
        ",".join(convert.KEY_FIELDS) + "\n"
        "64,4096,1024,32,4,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,"
        "torch.float8_e4m3fn,QuantType.per_Token,1,0\n"
    )
    out = tmp_path / "plans" / "out.csv"
    argv = [str(keys), "-o", str(out), "--devices", "3,4", "--timeout", "60"]
    assert cli.main([*argv, "--top-k", "5"]) == 0
    assert tuned_keys == [(_ptok(), "a8w8_ptok", 5)]

    def read(name):
        with open(name, newline="") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)

    expected = search.tune(_ptok(), "a8w8_ptok", _FakeMeasure(), search.Settings(5))
    winner = expected[0]
    assert winner.rank == 1 and any(o.reason for o in expected)
    columns, rows = read(out)
    assert columns == list(tuned_plans.COLUMNS)
    assert [r["structure"] for r in rows] == [" | ".join(winner.structure)]
    assert tuned_plans.decode_candidates(rows[0]["candidates"]) == winner.candidates
    assert rows[0]["e2e_us"] == f"{winner.e2e.median_us:.3f}"
    assert rows[0]["mode"] == winner.e2e.mode
    log_file = tmp_path / "plans" / "out.log.csv"
    columns, logged = read(log_file)
    assert columns == list(cli.LOG_COLUMNS)
    assert [r["structure"] for r in logged] == [
        " | ".join(o.structure) for o in expected
    ]
    assert [r["rank"] for r in logged] == [str(o.rank or "") for o in expected]
    assert [r["reason"] for r in logged] == [o.reason or "" for o in expected]
    assert {tuple(r[c] for c in tuned_plans.KEY_COLUMNS) for r in rows + logged} == {
        tuned_plans.key_values(_ptok())
    }
    written, log_text = out.read_text(), log_file.read_text()
    assert cli.main([*argv[:4], " 3, 4,", *argv[5:]]) == 0
    assert len(tuned_keys) == 1 and out.read_text() == written
    assert log_file.read_text() == log_text

    def unsupported(problem, path, evaluate, settings, filters, log):
        tuned_keys.append((problem, path, settings.top_k))
        return [search.Outcome((), reason="unsupported: no")]

    monkeypatch.setattr(cli, "tune", unsupported)
    nothing = tmp_path / "nothing.csv"
    custom = tmp_path / "custom.csv"
    custom.write_text("a,b\n1,2\n")
    again = [str(keys), "-o", str(nothing), *argv[3:], "--log-file", str(custom)]
    assert cli.main(again) == 0 and cli.main(again) == 0
    assert len(tuned_keys) == 3  # no winner, no row: tuned again
    assert read(nothing) == (list(tuned_plans.COLUMNS), [])
    assert custom.read_text() == "a,b\n1,2\n"
    columns, logged = read(tmp_path / "custom.csv.1")
    assert columns == list(cli.LOG_COLUMNS)
    assert [(r["structure"], r["reason"]) for r in logged] == [
        ("", "unsupported: no")
    ] * 2
    assert not (tmp_path / "nothing.log.csv").exists()
    empty_log = tmp_path / "empty.log.csv"
    empty_log.touch()
    assert cli.main([*again[:-1], str(empty_log)]) == 0
    assert read(empty_log)[0] == list(cli.LOG_COLUMNS)
    assert not (tmp_path / "empty.log.csv.1").exists()
    with pytest.raises(SystemExit):
        cli.main([*again, "--log-file", str(nothing)])
    assert read(nothing) == (list(tuned_plans.COLUMNS), [])
    other = tmp_path / "other.csv"
    other.write_text("token,path\n")
    with pytest.raises(ValueError, match="other columns than a v2 plan file"):
        cli.main([str(keys), "-o", str(other), "--devices", "0"])


def test_the_plan_file_digest_is_of_what_the_process_read(tmp_path, monkeypatch):
    import csv
    import hashlib

    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned, tuned_plans
    from aiter.moev2.tuning import cli, search

    def named(path):
        if path is None:
            monkeypatch.delenv(envs.AITER_MOEV2_TUNED_PLANS, raising=False)
        else:
            monkeypatch.setenv(envs.AITER_MOEV2_TUNED_PLANS, str(path))
        tuned.clear_cache()

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    named(None)
    assert tuned_plans.digest() is None
    problem = _ptok()
    winner = search.tune(problem, "a8w8_ptok", _FakeMeasure())[0]
    plan_file = tmp_path / "plans.csv"
    with open(plan_file, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS)
        writer.writeheader()
        writer.writerow(
            cli._plan_row(tuned_plans.key_values(problem), "a8w8_ptok", winner)
        )
    first = plan_file.read_bytes()
    named(plan_file)
    rows = tuned_plans.lookup(problem)
    assert rows == ((winner.structure, winner.candidates),)
    plan_file.write_text(",".join(tuned_plans.COLUMNS) + "\n")
    # The engine digests after resolving: the digest is of the read the rows came from.
    assert tuned_plans.digest() == sha(first)
    assert tuned_plans.lookup(problem) == rows
    tuned.clear_cache()
    assert tuned_plans.digest() == sha(plan_file.read_bytes())
    assert tuned_plans.lookup(problem) == ()
    plan_file.write_bytes(b"\xff\xfe")
    tuned.clear_cache()
    assert tuned_plans.digest() == sha(b"\xff\xfe")
    with pytest.raises(ValueError):
        tuned_plans.lookup(problem)
    named(tmp_path / "missing.csv")
    assert tuned_plans.digest() == "missing"
    (tmp_path / "missing.csv").write_bytes(first)
    assert tuned_plans.digest() == "missing"
    with pytest.raises(FileNotFoundError):
        tuned_plans.lookup(problem)
    tuned.clear_cache()
    assert tuned_plans.digest() == sha(first)
    named(None)


def test_the_plan_file_round_trips_and_resolve_reads_it_only_when_named(
    tmp_path, monkeypatch
):
    import csv
    import logging

    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned, tuned_plans
    from aiter.moev2.tuning import cli, search

    problem = _ptok()
    outcomes = search.tune(problem, "a8w8_ptok", _FakeMeasure())
    plan_file = tmp_path / "plans.csv"
    with open(plan_file, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS)
        writer.writeheader()
        keys = tuned_plans.key_values(problem)
        writer.writerow(cli._plan_row(keys, "a8w8_ptok", outcomes[0]))
    winner = outcomes[0]
    assert winner.rank == 1
    segments = _path_segments("a8w8_ptok")
    found, _ = tc.structures(problem, segments)
    structure = next(
        sc.structure
        for sc in found
        if tuple(s.name for s in sc.structure) == winner.structure
    )
    expected = resolver.build(problem, structure, winner.candidates).plan

    def resolved(path=None):
        if path is None:
            monkeypatch.delenv(envs.AITER_MOEV2_TUNED_PLANS, raising=False)
        else:
            monkeypatch.setenv(envs.AITER_MOEV2_TUNED_PLANS, str(path))
        tuned.clear_cache()
        return resolver.resolve(problem, segments)

    plain = resolved()
    named = resolved(plan_file)
    assert named.plan == expected
    name = " | ".join(winner.structure)
    message = f"[fused_moe] moev2 tuned plan {name} for {problem.log_keys()}"
    assert named.logs == ((logging.INFO, message),)
    assert tuned_plans.lookup(problem) == ((winner.structure, winner.candidates),)
    assert tuned_plans.lookup(_ptok(m=128)) == ()
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    assert resolved(plan_file).plan == resolved().plan
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG")

    def rewritten(rows):
        with open(plan_file, "w", newline="") as f:
            writer = csv.DictWriter(f, tuned_plans.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return resolved(plan_file)

    with open(plan_file, newline="") as f:
        rows = list(csv.DictReader(f))
    other_path = {**rows[0], "path": "other", "structure": "sort | no.such_segment"}
    both = rewritten([other_path, *rows])
    assert both.plan == expected
    assert both.logs[0][0] == logging.WARNING and both.logs[1] == named.logs[0]
    assert "not used: the structure is not feasible here" in both.logs[0][1]
    fallback = rewritten([other_path])
    assert fallback.plan == plain.plan
    assert fallback.logs[0] == both.logs[0]

    ep = ParallelSpec("ep", 2, "a2a", "flydsl", 64, 16)
    tail = TailSpec("add_rmsnorm_quant", "rms", 1e-6, ScaleSpec(dtypes.bf16))
    assert tuned_plans.lookup(problem)
    assert tuned_plans.lookup(dataclasses.replace(problem, parallel=ep)) == ()
    assert tuned_plans.lookup(dataclasses.replace(problem, tail=tail)) == ()
    twice = rewritten([rows[0], rows[0]])
    assert twice.plan == plain.plan
    assert "file not read: " in twice.logs[0][1] and "two rows" in twice.logs[0][1]
    with pytest.raises(ValueError, match="two rows for"):
        tuned_plans.lookup(problem)
    with open(plan_file, "w", newline="") as f:
        writer = csv.DictWriter(f, cli.LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(cli._log_row(keys, "a8w8_ptok", o) for o in outcomes)
    tuned.clear_cache()
    log_read = resolver.resolve(problem, segments)
    assert log_read.plan == plain.plan
    assert "file not read: " in log_read.logs[0][1]
    assert "columns a v2 plan file does not (rank, cov" in log_read.logs[0][1]
    missing = resolved(tmp_path / "missing.csv")
    assert missing.plan == plain.plan and missing.logs[0][0] == logging.WARNING
    resolved()


def test_the_cli_reads_keys_as_the_legacy_tuner_writes_them(tmp_path):
    from aiter.moev2.tuning import cli

    untuned = tmp_path / "untuned.csv"
    untuned.write_text(
        "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
        "q_type,use_g1u1,doweight_stage1\n"
        "64,4096,1024,32,4,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,"
        "torch.float8_e4m3fn,QuantType.per_Token,1,0\n"
        "64,4096,1024,32,4,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,"
        "torch.float8_e4m3fn,QuantType.per_Token,1,0\n"
    )
    tuned_csv = tmp_path / "tuned.csv"
    tuned_csv.write_text(
        "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
        "q_dtype_w,q_type,use_g1u1,doweight_stage1,kernelName1\n"
        "gfx950,256,128,4096,1024,32,4,ActivationType.Silu,torch.bfloat16,"
        "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_Token,1,0,x\n"
        "gfx942,304,128,4096,1024,32,4,ActivationType.Silu,torch.bfloat16,"
        "torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,1,0,x\n"
    )
    lines = []
    keys = cli.read_keys([untuned, tuned_csv], "gfx950", 256, log=lines.append)
    assert [p for _, p in keys] == [_ptok(), _ptok(m=128)]
    assert lines == ["skipped 1 rows of another gfx or CU count"]
    assert cli.paths_taking(_ptok()) == ["a8w8_ptok"]

    values = ("64", "7168", "384", "385", "7", "ActivationType.Silu")
    values += ("torch.bfloat16", "torch.float8_e4m3fn", "torch.float4_e2m1fn_x2")
    values += ("QuantType.per_1x32", "1", "0")
    row = dict(zip(convert.KEY_FIELDS, values))
    dsv4 = cli.problem_of(row, "gfx950", 256, fhmoe_table="t.csv")
    assert dsv4.gate_mode.value == "interleave"
    assert dsv4.shared is not None and dsv4.extras["tuned_table"] == "t.csv"
    assert cli.paths_taking(dsv4) == ["fhmoe"]
    fp8_table = tmp_path / "fhmoe.csv"
    fp8_table.write_text(
        "model_dim,inter_dim,expert,topk,shared_expert_id,act_type,dtype,q_dtype_a,"
        "q_dtype_w,q_type,use_g1u1,doweight_stage1,hidden_pad,intermediate_pad,"
        "gate_mode\n"
        "6144,256,257,9,256,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,"
        "torch.float8_e4m3fn,QuantType.per_1x32,1,0,0,0,GateMode.INTERLEAVE\n"
    )
    mxfp8 = dict(row, q_dtype_w="torch.float8_e4m3fn")
    main_table = cli.problem_of(mxfp8, "gfx950", 256, fhmoe_table=str(fp8_table))
    assert main_table.shared is not None and "tuned_table" not in main_table.extras
    assert cli.paths_taking(main_table) == ["fhmoe"]
    declared = dict(mxfp8, model_dim="6144", inter_dim="256", expert="257", topk="9")
    dedicated = cli.problem_of(declared, "gfx950", 256, fhmoe_table=str(fp8_table))
    assert dedicated.extras["tuned_table"] == str(fp8_table)
    with pytest.raises(ValueError, match="partially matches"):
        cli.problem_of(
            dict(declared, gate_mode="GateMode.SEPARATED"),
            "gfx950",
            256,
            fhmoe_table=str(fp8_table),
        )

    blank = tmp_path / "blank.csv"
    blank.write_text(
        "gfx,cu_num," + ",".join(convert.KEY_FIELDS) + "\n"
        ",,64,4096,1024,32,4,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,"
        "torch.float8_e4m3fn,QuantType.per_Token,1,0\n"
    )
    assert [p for _, p in cli.read_keys([blank], "gfx950", 256)] == [_ptok()]
