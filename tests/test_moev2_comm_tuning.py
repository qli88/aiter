# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 comm_fused tuning on the CPU: the rank group, runner candidates, the
search over fake measurements, the plan file's TP rows and `CommFusedTemplate`
reading them.

The GPU gate (tests/moe_dispatch/tuning_comm_checks.py) measures on a TP8 group.
"""

import csv
import dataclasses
import json
import pathlib
import re
import sys
import zlib

import pytest

from aiter.moev2.segments.base import Params

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
_TABLE = _REPO / "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"


@pytest.fixture(autouse=True)
def _gfx950():
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx950"):
        yield


def _rows():
    with open(_TABLE, newline="") as f:
        return {
            int(r["token"]): r
            for r in csv.DictReader(f)
            if r.get("comm_fused_configs", "").strip()
        }


def _problem(m=64, comm="rs", gpu_model="mi350x"):
    from aiter.moev2.tuning import cli, collective

    local = cli.problem_of(_rows()[m], "gfx950", 256)
    return collective.CommProblem(local, 8, comm, True, gpu_model)


# The rank group.


def _rank_toy(payload):
    import os
    import time

    from aiter.moev2.tuning.measure import Result
    from aiter.moev2.tuning.pool import Restart

    kind, value = payload
    rank = int(os.environ["RANK"])
    if kind == "raise" and rank == value:
        raise ValueError("bad")
    if kind == "fail" and rank == value:
        return Result("mismatch")
    if kind in ("hang", "fail and hang") and rank == value:
        time.sleep(60)
    if kind == "slow" and rank == value:
        time.sleep(7)
    if kind == "fail and hang":
        return Result("its own reason")
    if kind == "crash" and rank == value:
        os._exit(7)
    if kind == "restart" and rank == 0:
        raise Restart(("restarted", rank))
    if kind == "list":
        return [Result(None) if i != rank else Result(f"plan {i}") for i in range(2)]
    env = ("HIP_VISIBLE_DEVICES", "WORLD_SIZE", "MASTER_ADDR", "LOCAL_RANK")
    return (value, rank, *(os.environ[k] for k in env), os.getpid())


def _bad_setup():
    raise RuntimeError("no group")


def _setup_failing_once(marker):
    import os

    try:
        os.remove(marker)
    except FileNotFoundError:
        return
    raise RuntimeError("devices still busy")


def _teardown_marking(directory):
    import os

    open(os.path.join(directory, os.environ["RANK"]), "w").close()


def test_combine_takes_rank_0_unless_a_rank_failed():
    from aiter.moev2.tuning.group import combine
    from aiter.moev2.tuning.measure import Result
    from aiter.moev2.tuning.pool import Failed

    assert combine([1, 2]) == 1
    assert combine([Result(None), Result("bad")]).reason == "rank 1: bad"
    assert combine([Result("first"), Result("bad")]).reason == "first"
    assert combine([Failed("gone"), 1]) == Failed("gone")
    assert combine([Result(None), Result("rank 1: x")]).reason == "rank 1: x"
    columns = combine([[Result(None), Result("a")], [Result("b"), Result(None)]])
    assert [r.reason for r in columns] == ["rank 1: b", "a"]
    assert combine([[1], [1, 2]]) == Failed("rank 0 returned 1 results, others differ")


def test_the_group_runs_every_task_on_every_rank_and_restarts_when_one_is_lost():
    from aiter.moev2.tuning.group import Group
    from aiter.moev2.tuning.pool import Failed

    payloads = [("ok", 0), ("raise", 1), ("hang", 1), ("ok", 3), ("crash", 0)]
    payloads += [("fail", 1), ("restart", None), ("list", None), ("ok", 8)]
    payloads += [("fail and hang", 1)]
    with Group(_rank_toy, ("3", "4"), timeout=5) as group:
        assert group.starts == 1
        out = dict(group.map(payloads))
        assert group.starts == 5  # after the hangs, the crash and the restart
        assert group.run(10, ("slow", 1)) == Failed("timed out after 5 s (ranks [1])")
        assert group.run(11, ("slow", 1), timeout=30)[:2] == (1, 0)
        assert group.starts == 6
    value, rank, devices, world, addr, local_rank, pid = out[0]
    assert (value, rank, devices, world, addr, local_rank) == (
        0,
        0,
        "3,4",
        "2",
        "127.0.0.1",
        "0",
    )
    assert out[1] == Failed("rank 1: ValueError: bad")
    assert out[2] == Failed("timed out after 5 s (ranks [1])")
    assert out[3][:2] == (3, 0) and out[3][-1] != pid
    assert out[4] == Failed("rank 0 died (exit code 7)")
    assert out[5].reason == "rank 1: mismatch"
    assert out[6] == ("restarted", 0)
    assert [r.reason for r in out[7]] == ["plan 0", "rank 1: plan 1"]
    assert out[8][:2] == (8, 0)
    assert out[9] == Failed("timed out after 5 s (ranks [1]); rank 0: its own reason")


def test_ranks_start_with_the_group_environment(monkeypatch):
    import multiprocessing.context
    import os

    from aiter.moev2.tuning.group import Group

    names = ("HIP_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT")
    started = []
    start = multiprocessing.context.SpawnProcess.start

    def recording(process):
        started.append(tuple(os.environ.get(n) for n in names))
        start(process)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", recording)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.delenv("RANK", raising=False)
    with Group(_rank_toy, ("4", "5"), timeout=60):
        assert [s[:4] for s in started] == [
            ("4,5", "0", "0", "2"),
            ("4,5", "1", "1", "2"),
        ]
        assert started[0][4] == started[1][4] is not None
        assert os.environ["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert "RANK" not in os.environ


def test_a_group_whose_setup_fails_does_not_start():
    from aiter.moev2.tuning.group import Group

    with pytest.raises(RuntimeError, match=r"did not start: rank \d: RuntimeError"):
        Group(_rank_toy, ("0", "1"), setup=_bad_setup)


def test_a_group_that_cannot_start_a_rank_stops_the_others(monkeypatch):
    from aiter.moev2.tuning import group

    started = []
    start = group.start_spawned

    def failing(target, args, env):
        if env["RANK"] == "1":
            raise OSError("no process")
        started.append(start(target, args, env))
        return started[-1]

    monkeypatch.setattr(group, "start_spawned", failing)
    with pytest.raises(OSError, match="no process"):
        group.Group(_rank_toy, ("0", "1"))
    assert len(started) == 1 and not started[0].is_alive()


def test_a_restart_that_does_not_start_is_tried_again_and_close_tears_down(tmp_path):
    import functools

    from aiter.moev2.tuning.group import Group

    marker = tmp_path / "fail"
    with Group(
        _rank_toy,
        ("0", "1"),
        timeout=30,
        setup=functools.partial(_setup_failing_once, str(marker)),
        teardown=functools.partial(_teardown_marking, str(tmp_path)),
        restart_wait=0.1,
    ) as group:
        marker.touch()
        assert group.run(0, ("crash", 1)).reason == "rank 1 died (exit code 7)"
        assert group.starts == 3 and not marker.exists()
        assert group.run(1, ("ok", 1))[:2] == (1, 0)
        assert not (tmp_path / "0").exists()
    assert (tmp_path / "0").exists() and (tmp_path / "1").exists()


# Per-rank inputs.


def test_a_weight_seed_changes_the_weights_only():
    import torch

    from aiter import ActivationType, QuantType, dtypes
    from aiter.moev2.problem import MoeProblem
    from aiter.moev2.tuning import inputs as tin

    args = (16, 256, 128, 8, 2, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_Token, True, ActivationType.Silu, False, 0, 0)
    p = MoeProblem.from_legacy_key(
        *args, input_dtype=dtypes.bf16, arch="gfx950", cu_num=256
    )
    plain = tin.build(p, 3, "cpu")
    ranks = [tin.build(p, 3, "cpu", weight_seed=s) for s in (7, 8)]
    other = tin.build(p, 4, "cpu", weight_seed=7)

    def bits(inputs, name):
        return inputs.call[name].view(torch.uint8)

    assert torch.equal(bits(plain, "w1"), bits(tin.build(p, 3, "cpu"), "w1"))
    for name in ("hidden_states", "topk_ids", "topk_weight"):
        assert all(torch.equal(plain.call[name], r.call[name]) for r in ranks)
        assert not torch.equal(plain.call[name], other.call[name])
    assert not torch.equal(bits(ranks[0], "w1"), bits(ranks[1], "w1"))
    assert not torch.equal(bits(ranks[0], "w1"), bits(plain, "w1"))
    assert torch.equal(bits(ranks[0], "w2"), bits(other, "w2"))
    assert not torch.equal(ranks[0].reference, ranks[1].reference)


# Runner candidates.


def test_runner_candidates_retarget_the_close_seeds_and_add_their_defaults():
    from aiter.moev2.tuning import collective, comm_search

    rs, ar = _problem(64, "rs"), _problem(4096, "ar")
    seeds = [
        comm_search.Seed(("direct", (("grid_cap", 32), ("sort_block_m", 32))), 256),
        comm_search.Seed(("mega", (("tile_m", 32),)), 1),
        comm_search.Seed(("window", (("sorted_input", True), ("tile_m", 99))), 9),
    ]
    found = comm_search.runner_candidates(rs, 64, seeds)
    assert found == [
        ("direct", (("grid_cap", 32), ("sort_block_m", 64))),
        ("direct", (("sort_block_m", 64),)),
    ]
    far = [comm_search.Seed(("direct", (("grid_cap", 32),)), 257)]
    assert comm_search.runner_candidates(rs, 64, far) == [
        ("direct", (("sort_block_m", 64),))
    ]
    assert comm_search.runner_candidates(rs, 64, far, span=0) == found
    axes = {"direct": {"grid_cap": (16, 32)}, "mega": {"tile_m": (16, 32)}}
    assert comm_search.runner_candidates(rs, 32, seeds, axes) == [
        ("direct", (("grid_cap", 16), ("sort_block_m", 32))),
        ("direct", (("grid_cap", 32), ("sort_block_m", 32))),
    ]
    mega = comm_search.runner_candidates(ar, 32, seeds, axes)
    assert mega == [("mega", ()), ("mega", (("tile_m", 32),))]
    for spec in mega:
        config = collective.runner_config(ar, spec)
        assert (config.m, config.sort_block_m) == (4096, 32)
    shipped = json.loads(_rows()[4096]["comm_fused_configs"])["ar"]
    window = comm_search.Seed(collective.spec_of_record(shipped), 4096)
    spec = collective.spec_of_record(shipped)
    assert comm_search.runner_candidates(ar, 64, [window, *seeds]) == [spec]
    assert comm_search.runner_candidates(ar, 64, [window, *seeds], span=0) == [
        spec,
        ("mega", (("sort_block_m", 64), ("tile_m", 32))),
        ("mega", (("sort_block_m", 64),)),
    ]
    assert comm_search.retarget(ar, "window", {"tile_m": 64}, 64) is None
    params = comm_search.runner_params(spec)
    assert params == Params(dict(spec[1]), family="window")


def test_the_cli_reads_tp_keys_and_seeds_from_the_comm_fused_rows(tmp_path):
    from aiter.moev2.tuning import cli

    keys = cli.read_comm_keys([str(_TABLE)], "gfx950", 256, 8, "mi350x")
    expected = [
        (m, comm)
        for m, row in _rows().items()
        for comm in sorted(json.loads(row["comm_fused_configs"]))
    ]
    assert [(p.local.shape.m, p.comm) for _, p in keys] == expected
    assert all(p.add_shared and p.gpu_model == "mi350x" and p.tp == 8 for _, p in keys)
    assert keys[0][0].endswith(" ar") and str(_TABLE) in keys[0][0]
    with open(_TABLE, newline="") as f:
        every = {cli.problem_of(r, "gfx950", 256) for r in csv.DictReader(f)}
    both = cli.read_comm_keys([str(_TABLE)], "gfx950", 256, 8, "x", ["ar", "rs"], False)
    assert len(both) == 2 * len(every) and not any(p.add_shared for _, p in both)
    assert cli.read_comm_keys([str(_TABLE)], "gfx942", 304, 8, "x") == []
    seeds = cli.read_seeds([str(_TABLE)], _problem(64, "rs"), "gfx950", 256)
    assert [(s.spec[0], s.token, s.gpu_model) for s in seeds] == [
        ("direct", m, "mi355x") for m in (64, 128, 256, 512, 1024)
    ] + [("window", m, "mi355x") for m in (4096, 8192, 16384, 32768, 131072)]
    assert seeds[0].us == 104.1663
    other = dataclasses.replace(_problem(64, "rs"), tp=4)
    assert cli.read_seeds([str(_TABLE)], other, "gfx950", 256) == []
    row = _rows()[64]
    mixed = tmp_path / "mixed.csv"
    with open(mixed, "w", newline="") as f:
        writer = csv.DictWriter(f, list(row))
        writer.writeheader()
        writer.writerow({**row, "inter_dim": "512"})
        writer.writerow({**row, "expert": "256"})
        writer.writerow({**row, "token": "2048"})
    found = cli.read_seeds([str(mixed)], _problem(64, "rs"), "gfx950", 256)
    assert [s.token for s in found] == [2048]
    entry = json.loads(row["comm_fused_configs"])["rs"]
    tp4 = tmp_path / "tp4.csv"
    with open(tp4, "w", newline="") as f:
        writer = csv.DictWriter(f, list(row))
        writer.writeheader()
        writer.writerow(
            {
                **row,
                "comm_fused_configs": json.dumps(
                    {"rs": {**entry, "tp": 4, "add_shared": False}}
                ),
            }
        )
    assert cli.read_comm_keys([str(tp4)], "gfx950", 256, 8, "x") == []
    ((_, forced),) = cli.read_comm_keys([str(tp4)], "gfx950", 256, 8, "x", ["rs"])
    assert forced.add_shared  # not the TP4 entry's
    assert cli.parse_axes(["window:tile_n=128,256", "direct:x=true"]) == {
        "window": {"tile_n": (128, 256)},
        "direct": {"x": (True,)},
    }
    with pytest.raises(ValueError, match="FAMILY:FIELD"):
        cli.parse_axes(["tile_n=128"])


def test_the_cli_tunes_each_tp_key_once_into_the_plan_file(tmp_path, monkeypatch):
    from aiter.jit.utils import chip_info
    from aiter.moev2.dispatch import tuned_plans
    from aiter.moev2.tuning import cli, comm_search, search
    from aiter.moev2.tuning.measure import Timing
    from aiter.ops.flydsl import comm_fused_moe_host as host

    made, tuned = [], []

    class Fake:
        def __init__(self, handler, devices=None, **kw):
            made.append(
                (
                    type(self).__name__,
                    list(devices),
                    kw.get("setup"),
                    kw.get("teardown"),
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    class FakePool(Fake):
        pass

    class FakeGroup(Fake):
        pass

    def tune(problem, path, local, group, seeds, axes, settings, filters, log):
        tuned.append((problem.local.shape.m, problem.comm, path, len(seeds), axes))
        win = {"sort": Params(kernel="k"), comm_search.RUNNER: Params(family="x")}
        timing = Timing(10.0, 0.01, 10, "graph")
        failed = search.Outcome(("sort", "x"), reason="no passing run")
        return [search.Outcome(("sort", "g"), win, timing, rank=1), failed]

    monkeypatch.setattr(cli, "Pool", FakePool)
    monkeypatch.setattr(cli, "Group", FakeGroup)
    monkeypatch.setattr(cli.comm_search, "tune", tune)
    monkeypatch.setattr(host, "get_gpu_model", lambda: "mi350x")
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    with open(_TABLE, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["token"] in ("64", "4096")]
        columns = reader.fieldnames
    table = tmp_path / "table.csv"
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "plans.csv"
    argv = ["--keys-from", str(table), "-o", str(out), "--comm-fused"]
    argv += ["--devices", "0,1,2,3,4,5,6,7", "--runner-axis", "direct:grid_cap=16"]
    assert cli.main(argv) == 0
    assert [n for n, *_ in made] == ["FakePool", "FakeGroup"]
    assert made[1][1] == [str(d) for d in range(8)] and made[1][2] is not None
    axes = {"direct": {"grid_cap": (16,)}}
    assert tuned == [
        (64, "rs", "a8w4", 2, axes),
        (4096, "ar", "a8w4", 1, axes),
        (4096, "rs", "a8w4", 2, axes),
    ]
    with open(out, newline="") as f:
        written = list(csv.DictReader(f))
    assert [(r["token"], r["comm"], r["path"]) for r in written] == [
        ("64", "rs", "comm_fused"),
        ("4096", "ar", "comm_fused"),
        ("4096", "rs", "comm_fused"),
    ]
    assert all(r["tp"] == "8" and r["gpu_model"] == "mi350x" for r in written)
    assert tuned_plans.decode_candidates(written[0]["candidates"])[
        comm_search.RUNNER
    ] == Params(family="x")
    with open(tmp_path / "plans.log.csv", newline="") as f:
        logged = list(csv.DictReader(f))
    assert [(r["token"], r["comm"], r["rank"], r["reason"]) for r in logged] == [
        (m, comm, rank, reason)
        for m, comm in (("64", "rs"), ("4096", "ar"), ("4096", "rs"))
        for rank, reason in (("1", ""), ("", "no passing run"))
    ]
    text = out.read_text()
    assert cli.main(argv) == 0 and len(tuned) == 3 and out.read_text() == text
    assert len(made) == 2  # every key done: no pool or group started
    with pytest.raises(SystemExit, match="--tp 8 needs 8 devices"):
        cli.main([*argv[:5], "--devices", "4,5,6,7"])
    seeds = ["--seeds-from", str(_TABLE)]
    tp4 = [*seeds, "--devices", "4,5,6,7", "--tp", "4"]
    assert cli.main([*argv[:5], *tp4]) == 0 and len(tuned) == 3  # TP8 entries
    table4 = tmp_path / "table4.csv"
    with open(table4, "w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for r in rows:
            entries = json.loads(r["comm_fused_configs"] or "{}")
            for entry in entries.values():
                entry["tp"] = 4
            writer.writerow({**r, "comm_fused_configs": json.dumps(entries)})
    assert cli.main(["--keys-from", str(table4), *argv[2:5], *tp4]) == 0
    assert made[-1][1] == ["4", "5", "6", "7"]
    assert made[-1][2] is not None and made[-1][3] is not None
    assert [t[:4] for t in tuned[3:]] == [
        (64, "rs", "a8w4", 2),
        (4096, "ar", "a8w4", 1),
        (4096, "rs", "a8w4", 2),
    ]
    assert cli.main([*argv, *seeds, "-o", str(tmp_path / "again.csv")]) == 0
    assert [t[3] for t in tuned[6:]] == [10, 9, 10]
    empty = tmp_path / "empty.csv"
    empty.touch()
    assert cli.main([*argv[:3], str(empty), *argv[4:]]) == 0
    with open(empty, newline="") as f:
        assert [r["token"] for r in csv.DictReader(f)] == ["64", "4096", "4096"]

    def no_winner(problem, *args):
        tuned.append(problem.local.shape.m)
        return [search.Outcome(("sort", "x"), reason="no passing run")]

    monkeypatch.setattr(cli.comm_search, "tune", no_winner)
    none = tmp_path / "none.csv"
    count = len(tuned)
    assert cli.main([*argv[:3], str(none), *argv[4:]]) == 0
    assert cli.main([*argv[:3], str(none), *argv[4:]]) == 0
    assert len(tuned) == count + 6  # no winner, no row: tuned again
    with open(none, newline="") as f:
        assert list(csv.DictReader(f)) == []


@pytest.mark.parametrize(
    "signame, taker",
    [("SIGTERM", "main"), ("SIGTERM", "other thread"), ("SIGINT", "main")],
)
def test_a_key_s_rows_are_written_whole_when_a_stop_signal_arrives(
    tmp_path, signame, taker
):
    import os
    import signal
    import threading
    import time

    from aiter.moev2.tuning import cli

    signum = getattr(signal, signame)

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    idle = threading.Event()
    other = threading.Thread(target=idle.wait, daemon=True)
    other.start()

    class Out:
        def __init__(self, f):
            self.f = f

        def write(self, text):
            if taker == "main":
                os.kill(os.getpid(), signum)
            else:
                signal.pthread_kill(other.ident, signum)
            time.sleep(0.1)
            for line in text.splitlines(keepends=True):
                self.f.write(line)

        def flush(self):
            self.f.flush()

    rows = [{"token": str(t), "path": "a8w4"} for t in (1, 2, 3)]
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with open(tmp_path / "rows.csv", "w", newline="") as f, open(
            tmp_path / "log.csv", "w", newline=""
        ) as log, pytest.raises(SystemExit, match=str(128 + signum)):
            cli._append(
                [
                    (Out(log), cli.LOG_COLUMNS, rows),
                    (Out(f), cli.tuned_plans.COLUMNS, rows[:1]),
                ]
            )
        restored = [signal.getsignal(s) for s in previous]
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        idle.set()
    assert restored == [stop, stop]
    with open(tmp_path / "rows.csv", newline="") as f:
        written = list(csv.DictReader(f, cli.tuned_plans.COLUMNS))
    assert [(r["token"], r["path"]) for r in written] == [("1", "a8w4")]
    with open(tmp_path / "log.csv", newline="") as f:
        logged = list(csv.DictReader(f, cli.LOG_COLUMNS))
    assert [(r["token"], r["path"]) for r in logged] == [
        ("1", "a8w4"),
        ("2", "a8w4"),
        ("3", "a8w4"),
    ]


def test_sigterm_stops_the_tuner_through_its_shutdown(tmp_path):
    import signal
    import subprocess

    closed = tmp_path / "closed"
    script = f"""
import runpy, sys, time
sys.path.insert(0, {str(_REPO)!r})
from aiter.moev2.tuning import cli

def main():
    try:
        print("ready", flush=True)
        time.sleep(120)
    finally:
        open({str(closed)!r}, "w").write("closed")

cli.main = main
runpy.run_module("aiter.moev2.tuning", run_name="__main__")
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGTERM)
        assert process.wait(60) == 128 + signal.SIGTERM
    finally:
        process.kill()
        process.stdout.close()
    assert closed.read_text() == "closed"


# The search over fake measurements.


def _us(params):
    text = params.get("kernel") or repr(sorted(params.items()))
    return 10.0 + zlib.crc32(str(text).encode()) % 997 / 10.0


def _gemm2(name):
    from aiter.moev2.dispatch.ops import Op
    from aiter.moev2.migration import a8w4

    return next(s for s in a8w4.SEGMENTS if s.name == name).covers[0] == Op.GEMM2


class _LocalFake:
    """`evaluate_local`: a stage's time is `_us` of its candidate."""

    def __init__(self):
        from aiter.moev2.migration import a8w4
        from aiter.moev2.tuning import search

        self.stage = {s.name: search.stage(s) for s in a8w4.SEGMENTS}
        self.tasks = 0

    def __call__(self, tasks):
        from aiter.moev2.tuning.measure import Result, Timing

        out = []
        for task in tasks:
            self.tasks += 1
            combo = dict(task.candidates)
            stages = {
                self.stage[n]: Timing(_us(c), 0.01, 10, "graph")
                for n, c in combo.items()
                if task.segments and self.stage[n] in task.segments
            }
            total = Timing(sum(_us(c) for c in combo.values()), 0.01, 10, "graph")
            out.append(Result(e2e=total if task.e2e else None, stages=stages))
        return out


class _GroupFake:
    """`evaluate_group`: the ordinary call takes `ordinary_us`; a runner its stage-1
    candidates' `_us` plus `runner_us(spec)` (None: it fails)."""

    def __init__(self, ordinary_us, runner_us, rivals=None):
        self.ordinary_us, self.runner_us, self.rivals = ordinary_us, runner_us, rivals
        self.tasks = []

    def time(self, candidates, runner):
        if runner is None:
            return self.ordinary_us
        cost = self.runner_us(runner)
        if cost is None:
            return None
        local = sum(_us(c) for n, c in candidates if not _gemm2(n))
        return local + cost

    def result(self, candidates, runner, scale=1.0):
        from aiter.moev2.tuning.measure import Result, Timing

        us = self.time(candidates, runner)
        if us is None:
            return Result("mismatch: fake")
        return Result(e2e=Timing(us * scale, 0.01, 10, "graph"))

    def __call__(self, tasks):
        from aiter.moev2.tuning.collective import CommRivals

        out = []
        for task in tasks:
            self.tasks.append(task)
            if isinstance(task, CommRivals):
                if self.rivals is not None:
                    out.append(self.rivals(self, task))
                else:
                    out.append([self.result(c, r) for _, c, r in task.plans])
            else:
                out.append(self.result(task.candidates, task.runner))
        return out


def _direct_us(spec):
    params = dict(spec[1])
    return 5.0 + params.get("grid_cap", 80) / 10.0


def _tune(problem, group, seeds=(), axes=None, **settings):
    from aiter.moev2.tuning import comm_search

    return comm_search.tune(
        problem,
        "a8w4",
        _LocalFake(),
        group,
        seeds,
        axes,
        comm_search.CommSettings(**settings),
    )


def _brute_force(problem, group, seeds, axes, settings):
    """The fastest fused run over every searched local plan and runner."""
    from aiter.moev2.tuning import comm_search, search

    state = search.level1(problem.local, "a8w4", _LocalFake(), settings)
    today, _ = search.current(problem.local, "a8w4", state.found)
    classes = comm_search.local_classes(problem.local, state, today, settings)
    best = None
    for (_, block_m, _), combos in classes.items():
        for spec in comm_search.runner_candidates(problem, block_m, seeds, axes):
            for combo in combos:
                us = group.time(tuple(combo.items()), spec)
                if us is not None and (best is None or us < best[0]):
                    best = (us, combo, spec)
    return best


def test_the_search_picks_a_runner_that_beats_the_ordinary_call():
    from aiter.moev2.tuning import comm_search

    problem = _problem(64, "rs")
    axes = {"direct": {"grid_cap": (16, 32, 48, 80)}}
    group = _GroupFake(1000.0, _direct_us)
    outcomes = _tune(problem, group, axes=axes)
    win = outcomes[0]
    assert win.rank == 1 and comm_search.RUNNER in win.candidates
    runner = win.candidates[comm_search.RUNNER]
    assert runner["family"] == "direct" and runner["grid_cap"] == 16
    built = comm_search.layout(
        problem.local,
        next(
            sc.structure
            for sc in comm_search.level1(problem.local, "a8w4", _LocalFake()).found
            if tuple(s.name for s in sc.structure) == win.structure
        ),
        {k: v for k, v in win.candidates.items() if k != comm_search.RUNNER},
    )
    assert runner["sort_block_m"] == built[0]
    best = _brute_force(problem, group, (), axes, comm_search.COMM_DEFAULTS)
    assert win.e2e.median_us == best[0]
    ordinary = next(o for o in outcomes if comm_search.ORDINARY_NOTE in o.notes)
    assert ordinary.e2e.median_us == 1000.0 and comm_search.RUNNER not in (
        ordinary.candidates or {}
    )
    assert "the current plan" in ordinary.notes
    assert "the current plan: 1000.0 us" in win.notes
    ranked = [o for o in outcomes if o.rank is not None]
    assert [o.rank for o in ranked] == list(range(1, len(ranked) + 1))
    families = {
        (o.structure, (o.candidates or {}).get(comm_search.RUNNER, {}).get("family"))
        for o in ranked
    }
    assert len(families) == len(ranked)
    runners = [t for t in group.tasks if getattr(t, "runner", None) is not None]
    assert {dict(t.runner[1]).get("grid_cap", 80) for t in runners} == {16, 32, 48, 80}


def test_the_ordinary_call_wins_when_no_runner_beats_it():
    from aiter.moev2.tuning import comm_search

    outcomes = _tune(_problem(64, "rs"), _GroupFake(1.0, _direct_us))
    win = outcomes[0]
    assert comm_search.RUNNER not in win.candidates
    assert comm_search.ORDINARY_NOTE in win.notes and "the current plan" in win.notes
    assert all(o.rank is None or o.rank > 1 for o in outcomes[1:])


def test_a_near_tie_goes_to_the_ordinary_call_unless_a_runner_wins_again():
    from aiter.moev2.tuning import collective, comm_search

    problem = _problem(64, "rs")
    probe = _GroupFake(1000.0, _direct_us)
    best = _brute_force(problem, probe, (), None, comm_search.COMM_DEFAULTS)
    tied = _GroupFake(best[0] + 0.2, _direct_us)
    outcomes = _tune(problem, tied)
    rivals = [t for t in tied.tasks if isinstance(t, collective.CommRivals)]
    assert len(rivals) == 1 and len(rivals[0].plans) > 1
    assert [p[2] for p in rivals[0].plans].count(None) == 1
    assert comm_search.RUNNER not in outcomes[0].candidates
    assert "timed again in turn" in outcomes[0].notes

    def faster_runner(fake, task):
        return [fake.result(c, r, 0.5 if r else 1.0) for _, c, r in task.plans]

    again = _GroupFake(best[0] + 0.2, _direct_us, rivals=faster_runner)
    outcomes = _tune(problem, again)
    assert comm_search.RUNNER in outcomes[0].candidates


def test_a_near_tie_times_at_most_the_kept_runner_sets_again():
    from aiter.moev2.tuning import collective, comm_search

    problem = _problem(64, "rs")
    axes = {"direct": {"grid_cap": (8, 16, 24, 32, 40, 48, 56, 64)}}
    probe = _GroupFake(1000.0, lambda spec: 5.0)
    best = _brute_force(problem, probe, (), axes, comm_search.COMM_DEFAULTS)
    tied = _GroupFake(best[0] + 0.1, lambda spec: 5.0)
    _tune(problem, tied, axes=axes)
    (rivals,) = [t for t in tied.tasks if isinstance(t, collective.CommRivals)]
    specs = {spec for _, _, spec in rivals.plans if spec is not None}
    assert len(specs) == collective.MAX_RUNNER_SETS
    assert len(rivals.plans) <= comm_search.COMM_DEFAULTS.max_rivals
    assert any(spec is None for _, _, spec in rivals.plans)


def test_measure_plans_refuses_more_runner_configs_than_it_keeps(monkeypatch):
    from aiter.moev2.tuning import collective

    monkeypatch.setattr(collective, "_tp", lambda: None)
    plans = [(None, ("direct", (("grid_cap", g),))) for g in range(7)]
    plans.append((None, None))
    results = collective.measure_plans(None, None, plans, "graph")
    assert [r.reason for r in results] == ["7 runner configs, more than the 6 kept"] * 8


@pytest.mark.parametrize("kept", ["the same", "others"])
def test_runner_sets_a_failure_leaves_apart_restart_the_group(monkeypatch, kept):
    from types import SimpleNamespace

    from aiter.moev2.tuning import collective
    from aiter.moev2.tuning.measure import Result
    from aiter.moev2.tuning.pool import Restart

    tp = SimpleNamespace(world_size=2, cpu_group="cpu")
    agreed = iter([None, "rank 1: ValueError: no arena"])

    def gather(out, mine, group):
        out[:] = [mine, mine if kept == "the same" else ["another set"]]

    monkeypatch.setattr(collective, "_copies", lambda inputs, tp: [("call", None)])
    monkeypatch.setattr(collective, "_check", lambda inputs: None)
    monkeypatch.setattr(collective, "agree", lambda reason, tp=None: next(agreed))
    monkeypatch.setattr(collective, "_Plan", lambda *args: "plan")
    monkeypatch.setattr(collective.dist, "all_gather_object", gather)
    plans = [(None, ("direct", ()))]
    if kept == "the same":
        prepared, _ = collective._prepared(None, None, plans, tp)
        assert prepared == ["rank 1: ValueError: no arena"]
        return
    with pytest.raises(collective._Diverged, match="no arena"):
        collective._prepared(None, None, plans, tp)

    def diverged(*args):
        raise collective._Diverged("rank 1: x (the ranks then kept other runner sets)")

    monkeypatch.setattr(collective, "_inputs", lambda problem, seed: None)
    monkeypatch.setattr(collective, "plan_of", lambda *args: None)
    monkeypatch.setattr(collective, "agree", lambda reason, tp=None: reason)
    monkeypatch.setattr(collective, "measure_plans", diverged)
    task = collective.CommTask(_problem(64, "rs"), "a8w4", runner=("direct", ()))
    with pytest.raises(Restart) as restart:
        collective.run(task)
    assert restart.value.result == Result(
        "rank 1: x (the ranks then kept other runner sets)"
    )


def test_this_gpu_models_shipped_runner_is_the_current_call():
    from aiter.moev2.tuning import cli, comm_search

    problem = _problem(64, "rs", gpu_model="mi355x")
    seeds = cli.read_seeds([str(_TABLE)], problem, "gfx950", 256)
    shipped = dict(next(s for s in seeds if s.token == 64).spec[1])

    def shipped_us(spec):
        params = dict(spec[1])
        same = spec[0] == "direct" and all(
            params.get(k) == v for k, v in shipped.items() if k != "sort_block_m"
        )
        return 100.0 if same else 500.0

    group = _GroupFake(1.0, shipped_us)
    outcomes = _tune(problem, group, seeds)
    win = outcomes[0]
    assert comm_search.RUNNER not in win.candidates
    assert "the current plan" not in win.notes.split("; ")
    us = float(re.search(r"the current plan: ([\d.]+) us", win.notes).group(1))
    today = [
        t
        for t in group.tasks
        if t.runner is not None and abs(group.time(t.candidates, t.runner) - us) < 0.06
    ]
    assert today and all(shipped_us(t.runner) == 100.0 for t in today)
    other = _problem(64, "rs", gpu_model="mi350x")
    outcomes = _tune(other, _GroupFake(1.0, lambda spec: 100.0), seeds)
    assert "shipped mi355x winner at M=64: direct, 104.1663 us" in outcomes[0].notes


def test_failing_runners_are_reported_per_family():
    from aiter.moev2.tuning import comm_search

    outcomes = _tune(_problem(64, "rs"), _GroupFake(50.0, lambda spec: None))
    assert comm_search.ORDINARY_NOTE in outcomes[0].notes
    failed = [o for o in outcomes if o.reason and o.reason.startswith("direct")]
    assert failed and all("mismatch: fake" in o.reason for o in failed)


# The plan file and the template.


def _comm_rows(problem, outcomes):
    from aiter.moev2.dispatch import tuned_plans
    from aiter.moev2.tuning import cli

    keys = tuned_plans.comm_key_values(
        problem.local, problem.tp, problem.comm, problem.add_shared, problem.gpu_model
    )
    path = tuned_plans.COMM_FUSED_PATH
    return [cli._plan_row(keys, path, o) for o in outcomes if o.rank == 1]


def _write(path, rows):
    from aiter.moev2.dispatch import tuned_plans

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_the_plan_file_keeps_tp_rows_apart_from_local_ones(tmp_path, monkeypatch):
    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned, tuned_plans, tuned_table

    problem = _problem(64, "rs")
    fused = _tune(problem, _GroupFake(1000.0, _direct_us))
    ordinary = _tune(_problem(128, "rs"), _GroupFake(1.0, _direct_us))
    plan_file = tmp_path / "plans.csv"
    rows = _comm_rows(problem, fused) + _comm_rows(_problem(128, "rs"), ordinary)
    other_dtype = _comm_rows(_problem(256, "rs"), ordinary)
    for row in other_dtype:
        row["q_dtype_a"] = "other" + row["q_dtype_a"]
    _write(plan_file, rows + other_dtype)
    assert rows[0]["tp"] == "8" and rows[0]["comm"] == "rs"
    assert rows[0]["add_shared"] == "True" and rows[0]["gpu_model"] == "mi350x"
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(plan_file))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
    tuned.clear_cache()
    key = (8, "rs", True, "mi350x")
    names, candidates = tuned_table.comm_lookup(problem.local, *key)
    assert names == fused[0].structure and candidates == fused[0].candidates
    names, candidates = tuned_table.comm_lookup(_problem(128).local, *key)
    assert tuned_plans.RUNNER not in candidates
    assert tuned_table.comm_lookup(problem.local, 8, "ar", True, "mi350x") is None
    assert tuned_table.comm_lookup(problem.local, 8, "rs", True, "mi355x") is None
    assert tuned_table.lookup(problem.local).tuner == ()
    assert tuned_table.comm_tokens(_problem(1).local, *key) == (64, 128, 256)
    old = tmp_path / "old.csv"
    local_columns = [
        c for c in tuned_plans.COLUMNS if c not in tuned_plans.PARALLEL_COLUMNS
    ]
    with open(old, "w", newline="") as f:
        writer = csv.DictWriter(f, local_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "path": "a8w4"} for r in rows[:1])
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(old))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
    tuned.clear_cache()
    assert tuned_table.lookup(problem.local).tuner == (
        (fused[0].structure, fused[0].candidates),
    )
    assert tuned_table.comm_lookup(problem.local, *key) is None
    monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE, raising=False)
    monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, raising=False)
    tuned.clear_cache()
    assert tuned_table.comm_lookup(problem.local, *key) is None


def _template(monkeypatch, comm):
    import torch

    from aiter import ActivationType, QuantType, dtypes
    from aiter.moev2 import engine as engine_mod
    from aiter.moev2.migration import comm_fused
    from aiter.ops.flydsl import comm_fused_moe_host as host

    shape = host.comm_fused_shape(
        tp_size=8, model_dim=7168, inter_dim=384, experts=384, topk=6, comm=comm
    )
    shipped = {}
    for m, row in _rows().items():
        entry = json.loads(row["comm_fused_configs"]).get(comm)
        if entry is not None:
            shipped[m] = host._config(entry, shape.kernel_shape(), m, comm)
    monkeypatch.setattr(host, "comm_fused_configs", lambda shape: (shipped, False))
    monkeypatch.setattr(host, "get_gpu_model", lambda: "mi350x")
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")  # as comm_fused runs
    weights = engine_mod.MoeWeights(
        torch.empty(384, 768, 3584, dtype=dtypes.fp4x2, device="meta"),
        torch.empty(384, 7168, 192, dtype=dtypes.fp4x2, device="meta"),
        torch.empty(384, 768, 224, dtype=dtypes.fp8_e8m0, device="meta"),
        torch.empty(384, 7168, 12, dtype=dtypes.fp8_e8m0, device="meta"),
    )
    for w in (weights.w1, weights.w2):
        w.is_shuffled = True
    template = comm_fused.CommFusedTemplate(
        weights,
        topk=6,
        options={
            "quant_type": QuantType.per_1x32,
            "activation": ActivationType.Silu,
            "gate_mode": "interleave",
        },
        side_inputs=frozenset({"shared_partial"}),
        tp_size=8,
        comm=comm,
    )
    return template, shipped


def _host_shape(comm):
    from aiter.ops.flydsl import comm_fused_moe_host as host

    return host.comm_fused_shape(
        tp_size=8, model_dim=7168, inter_dim=384, experts=384, topk=6, comm=comm
    )


def test_the_template_serves_the_plan_files_rows(tmp_path, monkeypatch, caplog):
    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned, tuned_plans
    from aiter.moev2.segments import comm_fused as stage2_rs
    from aiter.moev2.tuning import comm_search

    plain, shipped = _template(monkeypatch, "rs")
    assert plain._stage2().configs == shipped and not plain._stage2().tuned
    axes = {"direct": {"grid_cap": (16, 80)}}
    fused = _tune(_problem(64, "rs"), _GroupFake(1000.0, _direct_us), axes=axes)
    ordinary = _tune(_problem(128, "rs"), _GroupFake(1.0, _direct_us))
    unbuilt = _tune(_problem(256, "rs"), _GroupFake(1000.0, _direct_us))
    unbuilt[0] = dataclasses.replace(unbuilt[0], structure=("sort", "no.such"))
    broken = _tune(_problem(512, "rs"), _GroupFake(1000.0, _direct_us), axes=axes)
    candidates = dict(broken[0].candidates)
    candidates[comm_search.RUNNER] = Params(family="direct", grid_cap=-1)
    broken[0] = dataclasses.replace(broken[0], candidates=candidates)
    other_block = _tune(_problem(1024, "rs"), _GroupFake(1000.0, _direct_us))
    candidates = dict(other_block[0].candidates)
    runner = candidates[comm_search.RUNNER]
    shape = _host_shape("rs")
    block_m = stage2_rs.runner_config(runner, shape, 1024).sort_block_m
    other_sbm = 64 if block_m == 32 else 32
    candidates[comm_search.RUNNER] = Params(dict(runner), sort_block_m=other_sbm)
    other_block[0] = dataclasses.replace(other_block[0], candidates=candidates)
    rows = _comm_rows(_problem(64, "rs"), fused)
    rows += _comm_rows(_problem(128, "rs"), ordinary)
    rows += _comm_rows(_problem(256, "rs"), unbuilt)
    rows += _comm_rows(_problem(512, "rs"), broken)
    rows += _comm_rows(_problem(1024, "rs"), other_block)
    rows += [{**r, "token": "100"} for r in _comm_rows(_problem(64, "rs"), fused)]
    plan_file = tmp_path / "plans.csv"
    _write(plan_file, rows)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(plan_file))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
    tuned.clear_cache()
    template, _ = _template(monkeypatch, "rs")
    with caplog.at_level("WARNING", logger="aiter"):
        segment = template._stage2()
    assert "comm_fused row for M=256 not used: ValueError: its plan" in caplog.text
    assert "comm_fused row for M=512 not used" in caplog.text
    assert f"M=1024 not used: ValueError: the runner's sort_block_m={other_sbm}" in (
        caplog.text
    )
    assert "M=100 not used: calls of 100 tokens key as M=128" in caplog.text
    assert segment.tuned and set(segment.configs) == set(shipped) - {128}
    assert [m for m, *_ in segment.rows] == [64, 128, 256, 512, 1024]
    assert plain._stage2().rows == ()
    runner = fused[0].candidates[comm_search.RUNNER]
    assert segment.configs[64].grid_cap == runner["grid_cap"] == 16
    for m in (256, 512, 1024):
        assert segment.configs[m] == shipped[m]
    assert set(template._tuned) == {64}
    win = template._tuned[64]
    assert tuple(c.segment.name for c in win.plan.segments) == fused[0].structure
    one = template.resolve(64, False)
    assert one.description.startswith("comm_fused M=64: ") and one.supports(64)
    assert "direct_sbm" in one.description
    assert template.resolve(128, False).description.startswith("not served")
    assert not template.resolve(128, False).supports(128)
    assert tuned_plans.RUNNER == comm_search.RUNNER


def test_rows_naming_the_table_s_runners_keep_its_runner_set(tmp_path, monkeypatch):
    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned
    from aiter.moev2.tuning import comm_search

    fused = _tune(_problem(64, "rs"), _GroupFake(1000.0, _direct_us))
    assert fused[0].candidates[comm_search.RUNNER] == Params(
        family="direct", sort_block_m=32
    )
    plan_file = tmp_path / "plans.csv"
    _write(plan_file, _comm_rows(_problem(64, "rs"), fused))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(plan_file))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
    tuned.clear_cache()
    template, shipped = _template(monkeypatch, "rs")
    segment = template._stage2()
    assert not segment.tuned and segment.configs == shipped
    assert set(template._tuned) == {64} and [m for m, *_ in segment.rows] == [64]
    win = template._tuned[64]
    assert tuple(c.segment.name for c in win.plan.segments) == fused[0].structure


def test_tuned_window_runners_follow_the_mxfp8_fallback(monkeypatch):
    from aiter.moev2.segments import comm_fused as stage2_rs
    from aiter.moev2.tuning import collective

    entry = json.loads(_rows()[4096]["comm_fused_configs"])["ar"]
    shape = _host_shape("ar")
    params = Params(entry["params"], family=entry["family"])
    spec = (entry["family"], tuple(sorted(entry["params"].items())))
    problem = _problem(4096, "ar", gpu_model="mi355x")
    monkeypatch.setenv("AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK", "0")
    raw = stage2_rs.runner_config(params, shape, 4096)
    assert (raw.partial_payload_bits, raw.reduced_payload_bits) != (8, 8)
    assert collective.runner_config(problem, spec) == raw
    monkeypatch.setenv("AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK", "1")
    rewritten = dataclasses.replace(raw, partial_payload_bits=8, reduced_payload_bits=8)
    assert stage2_rs.runner_config(params, shape, 4096) == rewritten
    assert collective.runner_config(problem, spec) == rewritten


def test_tuned_rows_are_dropped_when_their_runner_set_does_not_fit(
    tmp_path, monkeypatch, caplog
):
    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned
    from aiter.ops.flydsl import comm_fused_moe_host as host

    axes = {"direct": {"grid_cap": (16, 80)}}
    fused = _tune(_problem(64, "rs"), _GroupFake(1000.0, _direct_us), axes=axes)
    plan_file = tmp_path / "plans.csv"
    _write(plan_file, _comm_rows(_problem(64, "rs"), fused))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(plan_file))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
    tuned.clear_cache()
    template, shipped = _template(monkeypatch, "rs")
    monkeypatch.setattr(host, "FLAT_VA_RANK_STRIDE", 1)
    with caplog.at_level("WARNING", logger="aiter"):
        segment = template._stage2()
    assert "tuned comm_fused rows not used: their runner set needs" in caplog.text
    assert not segment.tuned and segment.configs == shipped and not template._tuned
    assert [m for m, *_ in segment.rows] == [64]


def test_a_runner_set_needs_the_same_plan_file_rows_on_every_rank(monkeypatch):
    from types import SimpleNamespace

    import torch.distributed as dist

    from aiter.moev2.segments import comm_fused as stage2_rs
    from aiter.ops.flydsl import comm_fused_moe_host as host

    others = {"rows": ((64, ("sort",), "{}"),)}

    def gather(out, mine, group):
        out[:] = [mine, others["rows"]]

    made = []

    class Runners:
        def __init__(self, comm, shape, configs):
            made.append(configs)
            self.configs = configs

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(host, "_LazyRunners", Runners)
    monkeypatch.setattr(stage2_rs, "_TUNED_RUNNERS", {})
    comm = SimpleNamespace(cpu_group="cpu", world_size=2)
    ctx = SimpleNamespace(comm=comm, world_size=2)
    shape = dataclasses.replace(_host_shape("rs"), tp=2)
    configs = {64: "config"}
    rows = others["rows"]
    first = stage2_rs.CommFusedStage2(configs, shape=shape, tuned=True, rows=rows)
    first.prepare(ctx)
    second = stage2_rs.CommFusedStage2(configs, shape=shape, tuned=True, rows=rows)
    second.prepare(ctx)
    assert first.runners is second.runners and made == [configs]
    other = stage2_rs.CommFusedStage2(configs, shape=shape, tuned=True, rows=())
    with pytest.raises(
        RuntimeError, match=r"different tuned comm_fused rows \(ranks \[1\]"
    ):
        other.prepare(ctx)
    plain = stage2_rs.CommFusedStage2(configs, shape=shape)
    with pytest.raises(RuntimeError, match="AITER_MOEV2_TUNED_TABLE differs"):
        plain.prepare(ctx)  # rank 1 read a file, this rank none
