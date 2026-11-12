# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ranking refit report (aiter/moev2/tuning/refit_ranking.py) on the CPU, over
fake measurements, and `resolver.default_plan`.

`default_plan` against `resolve` over the whole key grid runs in
tests/moe_dispatch/ranking_checks.py.
"""

import csv
import dataclasses
import math
import pathlib
import sys
import threading

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.moev2.dispatch import ranking, ranking_table, resolver, tuned_table
from aiter.moev2.dispatch.ranking import Entry, Region, match, when
from aiter.moev2.problem import MoeProblem
from aiter.moev2.tuning import refit_ranking as rr
from aiter.moev2.tuning import task as tuning_task
from aiter.moev2.tuning.measure import Result, Timing
from aiter.moev2.tuning.search import Outcome

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

CK = tuple(ranking_table.CK_2STAGE.split(" | "))
ASM1 = tuple(ranking_table.ASM_1STAGE.split(" | "))
ASM2 = ("sort", "quant.a1", "asm.stage1", "quant.inter", "ck.gemm2")
FLAT = ("sort.flat", "asm.fmoe_1stage")


@pytest.fixture(autouse=True)
def _gfx950(monkeypatch):
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx950"):
        monkeypatch.setattr(tuned_table, "lookup", lambda p: tuned_table.Found())
        yield


def _ptok(m, inter_dim=256):
    args = (m, 7168, inter_dim, 256, 8, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_Token, True, ActivationType.Silu, False, 0, 0)
    return MoeProblem.from_legacy_key(
        *args, arch="gfx950", cu_num=256, is_shuffled=True
    )


def _segments():
    from aiter.moev2.migration import a8w8_ptok

    return a8w8_ptok.SEGMENTS


def _key(m, times, tuned_us=None, inter_dim=256, label=None):
    """`KeyTimes` of fp8 per-token at `m`: `times` {structure: us or a reason}."""
    p = _ptok(m, inter_dim)
    plans = tuple(ranking.rankable(resolver.feasible_plans(p, _segments())[0]))
    us = {n: t for n, t in times.items() if not isinstance(t, str)}
    why = {n: t for n, t in times.items() if isinstance(t, str)}
    tuned = None if tuned_us is None else ASM2
    return rr.KeyTimes(
        label or f"m{m}", p, "a8w8_ptok", plans, us, why, tuned, tuned_us
    )


def test_default_plan_is_resolve_s_untuned_branch_for_the_pick_and_any_structure():
    p = _ptok(32)
    resolved = resolver.resolve(p, _segments())
    picked = resolver.default_plan(p, _segments(), ASM1)
    assert resolved.plan.describe() == picked.plan.describe()
    assert resolved.plan.source == picked.plan.source == "default"
    other = resolver.default_plan(p, _segments(), CK)
    assert [c.segment.name for c in other.plan.segments] == list(CK)
    assert other.plan.source == "default"
    assert resolver.default_plan(p, _segments(), FLAT).declined.startswith(
        "sort.flat: no default params"
    )
    assert resolver.default_plan(p, _segments(), ("sort", "x")).declined == (
        "the structure is not feasible here"
    )


def test_default_plan_declines_a_dedicated_table_key_as_resolve_does():
    from aiter.moev2 import paths
    from aiter.moev2.migration import fhmoe
    from aiter.moev2.tuning import cli

    table = fhmoe._fhmoe_config_file()
    (_, p), *_ = cli.read_keys([table], "gfx950", 256, table)
    assert p.extras["tuned_table"] == table
    segments = paths.registered()["fhmoe"].segments
    plans = ranking.rankable(resolver.feasible_plans(p, segments)[0])
    declined = "no exact row in the dedicated table"
    assert plans and resolver.resolve(p, segments).declined == declined
    for structure in plans:
        names = [s.name for s in structure]
        assert resolver.default_plan(p, segments, names).declined == declined


def test_default_plan_ignores_a_tuned_row(monkeypatch):
    p = _ptok(32)
    row = {"kernelName1": "x", "kernelName2": "y", "block_m": 32}
    monkeypatch.setattr(tuned_table, "lookup", lambda q: tuned_table.Found(cells=row))
    assert resolver.default_plan(p, _segments(), CK).plan.source == "default"


def test_a_task_without_candidates_measures_the_default_plan():
    p = _ptok(32)
    plan = tuning_task.plan_of(p, "a8w8_ptok", CK)
    assert (
        plan.plan.describe()
        == resolver.default_plan(p, _segments(), CK).plan.describe()
    )
    assert tuning_task.plan_of(p, "a8w8_ptok").plan.describe() == (
        resolver.resolve(p, _segments()).plan.describe()
    )


def test_regret_is_the_pick_s_default_time_over_the_fastest_time():
    key = _key(32, {CK: 60.0, ASM1: 50.0, ASM2: 55.0}, tuned_us=40.0)
    assert rr.pick(key) == ("fp8_per_token[fmoe_1stage, m>16]", ASM1, None)
    assert rr.regret(key, ASM1) == (50.0 / 40.0, None)
    fast_default = _key(32, {CK: 30.0, ASM1: 50.0}, tuned_us=40.0)
    assert rr.regret(fast_default, ASM1) == (50.0 / 30.0, None)
    failing = _key(32, {CK: 60.0, ASM1: "RuntimeError: x"}, tuned_us=40.0)
    assert rr.regret(failing, ASM1) == (None, "default plan not run")


def test_summaries_per_entry_and_mode_with_the_keys_left_out():
    keys = [
        _key(4, {CK: 20.0, ASM1: 10.0}, tuned_us=10.0),
        _key(32, {CK: 30.0, ASM1: 30.0}, tuned_us=24.0),
        _key(64, {CK: 30.0, ASM1: "declined: x"}, tuned_us=24.0),
        _key(1024, {CK: 100.0, ASM1: 110.0}),
        rr.KeyTimes("broken", _ptok(8), "a8w8_ptok", (), {}, {}, error="E: x"),
    ]
    s = rr.summarize(keys)
    graph = s[("fp8_per_token", "graph")]
    assert sorted(graph.regrets) == [(1.25, "m32"), (2.0, "m4")]
    assert graph.geomean == pytest.approx(math.sqrt(2.5))
    assert graph.worst == (2.0, "m4")
    assert graph.left_out == {"default plan not run": 1, "not measured": 1}
    eager = s[("fp8_per_token", "eager")]
    assert eager.regrets == [(1.1, "m1024")]
    assert s[(None, "graph")].regrets == graph.regrets


def test_a_legacy_region_s_keys_are_left_out():
    legacy = (
        Entry(
            "ptok",
            match(quant_type=QuantType.per_Token),
            regions=[Region(when(m__le=16), legacy="the legacy branch")],
            otherwise=[ranking_table.CK_2STAGE],
        ),
    )
    keys = [_key(4, {CK: 20.0}), _key(32, {CK: 20.0})]
    assert rr.pick(keys[0], legacy) == ("ptok[m<=16]", None, "legacy region")
    s = rr.summarize(keys, legacy)[("ptok", "graph")]
    assert s.left_out == {"legacy region": 1} and s.regrets == [(1.0, "m32")]


def test_cuts_and_moving_one():
    entry = next(e for e in ranking_table.RANKING if e.name == "fp8_per_token")
    assert rr.cuts(entry) == [("m", 17)]
    table = rr.moved(ranking_table.RANKING, "fp8_per_token", "m", 17, 9)
    moved = next(e for e in table if e.name == "fp8_per_token")
    assert [r.when.domains["m"] for r in moved.regions] == [
        ranking.Domain(lo=9),
        ranking.Domain(hi=9),
    ]
    assert [e for e in table if e.name != "fp8_per_token"] == [
        e for e in ranking_table.RANKING if e.name != "fp8_per_token"
    ]
    assert rr.pick(_key(8, {}), table)[1] == CK
    assert rr.pick(_key(9, {}), table)[1] == ASM1
    band = (
        Entry(
            "b",
            match(),
            regions=[
                Region(when(m__gt=16, m__lt=256), prefer=[ranking_table.CK_2STAGE])
            ],
            otherwise=[ranking_table.CK_2STAGE],
        ),
    )
    assert rr.cuts(band[0]) == [("m", 17), ("m", 256)]
    assert rr.moved(band, "b", "m", 17, 256) is None
    assert rr.moved(_two_ends(), "e", "m", 17, 257) is None
    assert rr.moved(_two_ends(), "e", "m", 17, 256) is not None


def _two_ends():
    return (
        Entry(
            "e",
            match(quant_type=QuantType.per_Token),
            regions=[
                Region(when(m__le=16), prefer=[ranking_table.ASM_1STAGE]),
                Region(when(m__gt=255), prefer=[ranking_table.ASM_1STAGE]),
            ],
            otherwise=[ranking_table.CK_2STAGE],
        ),
    )


def test_the_cut_search_moves_no_bound_into_another_region():
    # ASM 1-stage wins everywhere: the low region would grow up to the high one.
    keys = [_key(m, {CK: 20.0, ASM1: 10.0}) for m in (4, 64, 300)]
    table = _two_ends()
    hint = rr.search_cut(keys, table, table[0], "m", 17)
    assert (hint.best, hint.after) == (65, pytest.approx(1.0))


def test_the_cut_search_finds_the_bound_with_the_lowest_geomean():
    # ASM 1-stage is faster from M=8 on; the table switches above 16.
    keys = [
        _key(m, {CK: 10.0 * (1 if m < 8 else 2), ASM1: 10.0 * (2 if m < 8 else 1)})
        for m in (1, 4, 8, 16, 32, 64)
    ]
    entry = next(e for e in ranking_table.RANKING if e.name == "fp8_per_token")
    hint = rr.search_cut(keys, ranking_table.RANKING, entry, "m", 17)
    assert (hint.best, hint.keys) == (8, 6)
    assert hint.before == pytest.approx(2 ** (2 / 6))
    assert hint.after == pytest.approx(1.0)
    flat = [_key(m, {CK: 10.0, ASM1: 10.0}) for m in (1, 32)]
    kept = rr.search_cut(flat, ranking_table.RANKING, entry, "m", 17)
    assert (kept.best, kept.before, kept.after) == (17, 1.0, 1.0)


def test_the_cut_search_keeps_no_bound_leaving_more_keys_out():
    keys = [_key(4, {CK: 20.0, ASM1: "declined: x"}), _key(32, {CK: 10.0, ASM1: 20.0})]
    entry = next(e for e in ranking_table.RANKING if e.name == "fp8_per_token")
    hint = rr.search_cut(keys, ranking_table.RANKING, entry, "m", 17)
    assert hint.best == 33 and hint.after == pytest.approx(1.0)
    worse = [_key(4, {CK: 20.0, ASM1: "declined: x"}), _key(32, {CK: 30.0, ASM1: 20.0})]
    assert rr.search_cut(worse, ranking_table.RANKING, entry, "m", 17).best == 17
    # Any bound above 32 hands M=32 to CK, which did not run there; that would lower
    # the geomean of the keys left (M=64 to 1.0) only by dropping M=32.
    lost = [
        _key(32, {CK: "declined: x", ASM1: 20.0}, tuned_us=10.0),
        _key(64, {CK: 10.0, ASM1: 20.0}),
    ]
    hint = rr.search_cut(lost, ranking_table.RANKING, entry, "m", 17)
    assert (hint.best, hint.after) == (17, pytest.approx(2.0))


def test_region_hints_name_the_structure_to_list_first():
    keys = [
        _key(4, {CK: 20.0, ASM1: 10.0, ASM2: 15.0, FLAT: "declined: no default"}),
        _key(8, {CK: 20.0, ASM1: 12.0, ASM2: 15.0, FLAT: "declined: no default"}),
        _key(32, {CK: 20.0, ASM1: 20.0, ASM2: 30.0}),
    ]
    hints = {h.label: h for h in rr.region_hints(keys)}
    other = hints["fp8_per_token[otherwise]"]
    assert (other.current, other.best, other.keys) == (CK, ASM1, 2)
    assert other.before == pytest.approx(math.sqrt(2.0 * 20 / 12))
    assert other.after == pytest.approx(math.sqrt(1.0 * 1.0))
    assert other.fails == {FLAT: 2}
    above = hints["fp8_per_token[fmoe_1stage, m>16]"]
    assert (above.current, above.best, above.after) == (ASM1, ASM1, 1.0)


def test_a_region_hint_never_proposes_a_structure_that_did_not_run():
    # ASM1 would score 1.0 if M=8 kept CK's time; it did not run there.
    keys = [
        _key(4, {CK: 20.0, ASM1: 10.0, ASM2: 15.0}),
        _key(8, {CK: 12.0, ASM1: "declined: x", ASM2: 15.0}),
    ]
    (hint,) = rr.region_hints(keys)
    assert (hint.current, hint.best) == (CK, ASM2)
    assert hint.after == pytest.approx(math.sqrt(1.5 * 1.25))
    assert hint.fails == {ASM1: 1}


def test_a_region_hint_needs_more_than_the_margin():
    keys = [_key(4, {CK: 10.0, ASM1: 10.0 * (1 - rr.HINT_MARGIN / 2)})]
    (hint,) = rr.region_hints(keys)
    assert hint.best == hint.current == CK


def test_a_region_hint_is_the_lowest_option_whatever_the_order():
    # ASM1 sorts before ASM2 and is better than CK, but ASM2 is lower still,
    # by less than the margin.
    keys = [_key(4, {CK: 12.6, ASM1: 10.08, ASM2: 10.0})]
    (hint,) = rr.region_hints(keys)
    assert (hint.best, hint.after) == (ASM2, pytest.approx(1.0))


def test_the_report_lines():
    keys = [
        _key(m, {CK: 10.0 * (1 if m < 8 else 2), ASM1: 10.0 * (2 if m < 8 else 1)})
        for m in (1, 4, 8, 16, 32, 64)
    ]
    lines = rr.report(keys)
    assert lines[0].startswith("ranking refit report: 6 keys (informational")
    assert any(
        line.startswith(
            "fp8_per_token [graph]: 6 keys, geomean regret 1.260, worst 2.000"
        )
        for line in lines
    )
    assert "  fp8_per_token: m < 17 -> m < 8 geomean 1.260 -> 1.000 (6 keys)" in lines
    # Listing ASM 1-stage first helps M=8 and 16 as much as it hurts M=1 and 4.
    assert (
        f"  fp8_per_token[otherwise]: {' | '.join(CK)} (unchanged) geomean 1.414"
        " (4 keys)"
    ) in lines
    assert not any(line.startswith("did not run") for line in lines)
    keys += [_key(64, {CK: "declined: no\n  default", ASM1: 10.0}, label="d1")]
    keys += [_key(128, {CK: "declined: no default", ASM1: 10.0}, label="d2")]
    lines = rr.report(keys)
    keys += [_key(256, {CK: "declined: other", ASM1: 10.0}, label="d3")]
    lines = rr.report(keys)
    assert lines[-2:] == [
        "did not run (structure, or not measured): keys, first reason",
        f"  {' | '.join(CK)}: 3 keys; d1: declined: no default",
    ]


def test_the_sample_keeps_n_per_class_spread_over_m():
    keys = [(f"k{m}", _ptok(m)) for m in (1, 2, 4, 8, 16, 32, 64, 128)]
    keys += [(f"e{m}", _ptok(m)) for m in (1024, 4096)]
    kept = [label for label, _ in rr.sample(keys, 3)]
    assert kept == ["k1", "k16", "k128", "e1024", "e4096"]
    assert rr.sample(keys, 0) == keys
    assert [label for label, _ in rr.sample(keys, 1)] == ["k1", "e1024"]


def test_saved_times_round_trip_and_unreadable_lines_are_skipped(tmp_path):
    key = _key(32, {CK: 20.0, FLAT: "declined: x"}, tuned_us=10.0, label="a.csv:2")
    path = tmp_path / "times.jsonl"
    line = rr.saved_line(key)
    path.write_text(line + "\n" + '{"key": "x"}\n' + line[: len(line) // 2])
    logged = []
    saved = rr.read_saved(path, logged.append)
    assert logged == [f"{path}: skipped 2 unreadable lines"]
    (fields,) = saved.values()
    back = rr.from_saved(fields, "b.csv:9", key.problem, key.path)
    assert back == dataclasses.replace(key, label="b.csv:9")


def test_the_sample_fills_thin_groups_from_the_grid():
    from moe_dispatch import ranking_report

    shipped = {f"s{m}": (None, _ptok(m)) for m in (1, 2, 1024)}
    grid = {f"g{m}": (None, _ptok(m)) for m in (4, 8, 16, 32, 2048, 4096)}
    got = ranking_report.chosen(shipped, grid, 3)
    assert got == ["s1", "s2", "s1024", "g2048", "g4096", "g4"]


def test_measure_key_times_each_default_plan_and_takes_the_winner(monkeypatch):
    p = _ptok(32)
    seen = []

    def evaluate(tasks):
        seen.extend(tasks)
        return [
            (
                Result(e2e=Timing(10.0 + i, 0.0, 1, "graph"))
                if t.structure != FLAT
                else Result("declined: no default")
            )
            for i, t in enumerate(tasks)
        ]

    win = Outcome(ASM2, {}, Timing(7.0, 0.0, 1, "graph"), rank=1)
    monkeypatch.setattr(rr, "tune", lambda *a, **k: [win, Outcome(CK, rank=2)])
    key = rr.measure_key("k", p, "a8w8_ptok", evaluate, rr.Settings(), None)
    assert [t.structure for t in seen] == [CK, ASM2, ASM1, FLAT]
    assert all(t.candidates is None and t.path == "a8w8_ptok" for t in seen)
    assert key.default_us == {CK: 10.0, ASM2: 11.0, ASM1: 12.0}
    assert key.default_why == {FLAT: "declined: no default"}
    assert (key.tuned, key.tuned_us, key.fastest_us) == (ASM2, 7.0, 7.0)
    monkeypatch.setattr(rr, "tune", lambda *a, **k: [Outcome((), reason="declined")])
    none = rr.measure_key("k", p, "a8w8_ptok", evaluate, rr.Settings(), None)
    assert (none.tuned, none.tuned_why, none.fastest_us) == (None, "declined", 10.0)


def test_the_cli_reports_every_key_on_its_own_device(tmp_path, monkeypatch, capsys):
    rows = tmp_path / "keys.csv"
    columns = "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
    fp8 = str(dtypes.fp8)
    lines = [columns] + [
        f"{m},7168,256,256,8,ActivationType.Silu,torch.bfloat16,{fp8},{fp8},QuantType.per_Token,1,0"
        for m in (4, 32)
    ]
    rows.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx", lambda: "gfx950")
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_cu_num", lambda: 256)
    opened = []

    class FakePool:
        def __init__(self, handler, devices, timeout):
            opened.append(self)
            self.devices, self.closed, self.served = devices, False, []

        def map(self, tasks):
            self.served.extend(tasks)
            return [(i, f"on {self.devices[0]}") for i, _ in enumerate(tasks)]

        def close(self):
            self.closed = True

    monkeypatch.setattr("aiter.moev2.tuning.pool.Pool", FakePool)
    devices_used = {}
    # First run: M=4 measures once, then M=32 twice, then M=4 again, so a key that
    # took a device per batch would see both.
    first_done, other_done = threading.Event(), threading.Event()

    def measure(label, problem, path, evaluate, settings, filters, log):
        ordered = not other_done.is_set()
        if ordered and problem.shape.m == 32:
            assert first_done.wait(10)
        used = set(evaluate([label]))
        if ordered and problem.shape.m == 4:
            first_done.set()
            assert other_done.wait(10)
        devices_used[label] = used | set(evaluate([label]))
        if ordered and problem.shape.m == 32:
            other_done.set()
        if problem.shape.m == 4:
            raise RuntimeError("boom")
        key = _key(32, {CK: 20.0, ASM1: 10.0}, tuned_us=10.0, label=label)
        return dataclasses.replace(key, problem=problem, path=path)

    monkeypatch.setattr(rr, "measure_key", measure)
    out_csv, times = tmp_path / "out.csv", tmp_path / "times.jsonl"
    argv = [str(rows), "--devices", "3,4", "--paths", "a8w8_ptok"]
    code = rr.main(argv + ["--csv", str(out_csv), "--times", str(times)])
    assert code == 0
    assert [p.devices for p in opened] == [["3"], ["4"]]
    assert all(p.closed for p in opened)
    assert sorted(map(len, devices_used.values())) == [1, 1]
    assert sorted(t for p in opened for t in p.served) == sorted(2 * list(devices_used))
    printed = capsys.readouterr().out
    assert "fp8_per_token [graph]: 1 keys, geomean regret 1.000" in printed
    assert "left out: 1 not measured" in printed
    got = list(csv.DictReader(out_csv.open()))
    assert [r["why"] for r in got] == ["not measured", ""]
    assert got[0]["error"] == "RuntimeError: boom"
    assert got[1]["regret"] == "1.0000" and got[1]["pick"] == " | ".join(ASM1)
    assert "  not measured: 1 keys; " in printed
    assert printed.rstrip().endswith(": RuntimeError: boom")
    # Resumed: the measured key comes from the file, the failed one runs again.
    assert len(times.read_text().splitlines()) == 1
    devices_used.clear()
    assert rr.main(argv + ["--csv", str(out_csv), "--times", str(times)]) == 0
    assert list(devices_used) == [got[0]["label"]]
    assert list(csv.DictReader(out_csv.open())) == got
    assert len(times.read_text().splitlines()) == 1


def test_a_key_measured_while_the_run_stops_is_not_saved(tmp_path, monkeypatch):
    rows = tmp_path / "keys.csv"
    fp8 = str(dtypes.fp8)
    rows.write_text(
        "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
        "q_type,use_g1u1,doweight_stage1\n"
        f"32,7168,256,256,8,ActivationType.Silu,torch.bfloat16,{fp8},{fp8},"
        "QuantType.per_Token,1,0\n"
    )
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx", lambda: "gfx950")
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_cu_num", lambda: 256)

    class FakePool:
        def __init__(self, handler, devices, timeout):
            pass

        def close(self):
            pass

    monkeypatch.setattr("aiter.moev2.tuning.pool.Pool", FakePool)

    def measure(label, problem, path, evaluate, settings, filters, log):
        rr.STOPPING.set()
        key = _key(32, {CK: 20.0, ASM1: 10.0}, tuned_us=10.0, label=label)
        return dataclasses.replace(key, problem=problem, path=path)

    monkeypatch.setattr(rr, "measure_key", measure)
    times = tmp_path / "times.jsonl"
    argv = [str(rows), "--devices", "0", "--paths", "a8w8_ptok", "--times", str(times)]
    try:
        assert rr.main(argv) == 0
    finally:
        rr.STOPPING.clear()
    assert not times.exists()
