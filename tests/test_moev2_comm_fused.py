# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the `comm_fused` path and `comm_fused.stage2_rs` (stub runners).

The legacy `CommFusedMoeRuntime.run` runs with `_fused_moe_impl` replaced by a stub that
calls its `_stage2_override` on fixed stage 2 arguments; v2's `prepare_call`, `launch`
and `finish` run on the same ones. Both must pad the same call, hand the runner the same
arguments in the same order, return the same rows and raise the same errors. The kernels
and the TP8 runs are in tests/moe_dispatch/comm_fused_differential.py.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from moe_dispatch.differential import moev2_mode

import aiter.fused_moe as legacy_fused_moe
from aiter.moev2 import engine as engine_mod
from aiter.moev2.dispatch.ops import Op
from aiter.moev2.dispatch.plan import MoePlan, SegmentChoice
from aiter.moev2.migration import comm_fused, local
from aiter.moev2.segments import comm_fused as stage2_rs
from aiter.moev2.segments.base import MemoryNeed, Params, Segment
from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime

MODEL_DIM, TOPK, TP = 8, 2, 4


@dataclasses.dataclass(frozen=True)
class _Shape:
    tp_size: int = TP
    add_shared: bool = True


@dataclasses.dataclass(frozen=True)
class _Config:
    m: int
    output_rows: int
    requires_exact_m: bool = False
    shape: _Shape = _Shape()


class _Runner:
    """Records what the runtime hands it; returns a fresh output each call."""

    def __init__(self, config, log, *, rank=1, direct=False, ragged=False):
        self.config, self.rank, self.log = config, rank, log
        self.supports_ragged_m = ragged
        self.supports_external_reuse_sync = direct
        self.requires_reuse_wait_before_stage2_write = direct
        self._destination = (
            torch.zeros(config.m, MODEL_DIM, dtype=torch.bfloat16) if direct else None
        )
        self.output = torch.zeros(config.output_rows, MODEL_DIM, dtype=torch.bfloat16)

    @property
    def stage2_destination(self):
        return self._destination

    def wait_for_stage2_destination_reuse(self):
        self.log.append(("reuse_wait",))

    def prepare_padded_shared_partial(self, shared, rows):
        self.log.append(("pad_shared", tuple(shared.shape), rows))
        padded = shared.new_zeros((self.config.m, MODEL_DIM))
        padded[:rows].copy_(shared)
        return padded

    def prepare_shared_partial(self, shared):
        self.log.append(("shared", tuple(shared.shape)))
        return shared

    def __call__(self, **kwargs):
        shared = kwargs.pop("shared_partial")
        join = kwargs.pop("before_shared_add")
        if join is not None:
            join()
        self.log.append(
            (
                "runner",
                None if shared is None else shared.float().sum().item(),
                sorted(kwargs),
                {k: v for k, v in kwargs.items() if isinstance(v, (int, bool))},
                kwargs["stage2_args"],
                kwargs["stage2_kwargs"],
            )
        )
        return self.output + 1


class _Runners:
    def __init__(self, configs, log):
        self.configs = configs
        self.instances = {m: _Runner(c, log) for m, c in configs.items()}

    def __getitem__(self, m):
        return self.instances[m]

    def __contains__(self, m):
        return m in self.configs

    def output_rows_for(self, bucket, rows):
        return stage2_rs.output_rows_for(self.configs, bucket, rows)


_CONFIGS = {
    1: _Config(1, 1),
    2: _Config(2, 2),
    4: _Config(4, 4),
    64: _Config(64, 64 // TP, requires_exact_m=True),
    128: _Config(128, 128 // TP),
}


def _runners(log, bucket=None, **runner_kw):
    """The stub runner set; the runner of `bucket` built with `runner_kw`."""
    runners = _Runners(_CONFIGS, log)
    if bucket is not None:
        runners.instances[bucket] = _Runner(_CONFIGS[bucket], log, **runner_kw)
    return runners


def _moe_args(rows, **extra):
    g = torch.Generator().manual_seed(rows)
    return {
        "hidden_states": torch.randn(rows, MODEL_DIM, generator=g).bfloat16(),
        "w1": None,
        "w2": None,
        "topk_weight": torch.rand(rows, TOPK, generator=g),
        "topk_ids": torch.randint(0, 8, (rows, TOPK), generator=g, dtype=torch.int32),
        **extra,
    }


_STAGE2_ARGS = ("a2", None, "w2", "ids", "experts", "num_valid", "moe_out", TOPK)
_STAGE2_KWARGS = {"block_m": 32}


def _ordinary_stage2(*args, **kwargs):
    raise AssertionError("the runner replaces the ordinary stage 2")


def _legacy(monkeypatch, runners, moe_args, **run_kw):
    """`CommFusedMoeRuntime.run` with a stub `_fused_moe_impl`: what that was called
    with, and the result or the exception."""
    seen = {}

    def impl(**kwargs):
        override = kwargs.pop("_stage2_override")
        seen.update(kwargs)
        return override(
            ordinary_stage2=_ordinary_stage2,
            stage2_args=_STAGE2_ARGS,
            stage2_kwargs=_STAGE2_KWARGS,
        )

    monkeypatch.setattr(legacy_fused_moe, "_fused_moe_impl", impl)
    run_kw.setdefault("shared_partial", None)
    try:
        out = CommFusedMoeRuntime(runners=runners).run(**run_kw, **moe_args)
    except Exception as exc:  # noqa: BLE001
        return seen, exc
    return seen, out


def _v2(runners, moe_args, **run_kw):
    run_kw.setdefault("shared_partial", None)
    try:
        padded, fused = stage2_rs.prepare_call(runners, moe_args, **run_kw)
        stage2_rs.wait_for_reuse(fused)
        out = stage2_rs.launch(
            fused,
            ordinary_stage2=_ordinary_stage2,
            stage2_args=_STAGE2_ARGS,
            stage2_kwargs=_STAGE2_KWARGS,
        )
    except Exception as exc:  # noqa: BLE001
        return {}, exc
    return padded, stage2_rs.finish(fused, out)


def _shared(rows, value=1.0):
    return torch.full((rows, MODEL_DIM), value, dtype=torch.bfloat16)


_JOINS: list = []


def _join():
    _JOINS.append(1)


# name: (rows, extra fused_moe arguments, run arguments, the runner set's override)
_CASES = {
    "exact bucket, shared": (4, {}, lambda: {"shared_partial": _shared(4)}, {}),
    "padded 3 to 4, shared for rows": (
        3,
        {},
        lambda: {"before_stage2_for_rows": lambda rows: _shared(3, rows)},
        {},
    ),
    "padded, full-bucket shared": (3, {}, lambda: {"shared_partial": _shared(4)}, {}),
    "padded, producer joined once": (
        3,
        {},
        lambda: {"before_stage2": lambda: _shared(3), "before_shared_add": _join},
        {},
    ),
    "no shared": (2, {}, dict, {}),
    "expert mask": (
        2,
        {"expert_mask": torch.ones(8)},
        lambda: {"shared_partial": _shared(2)},
        {},
    ),
    "routes may be missing": (
        2,
        {},
        lambda: {"shared_partial": _shared(2), "routes_may_be_missing": True},
        {},
    ),
    "both producers": (
        2,
        {},
        lambda: {
            "before_stage2": lambda: None,
            "before_stage2_for_rows": lambda rows: None,
        },
        {},
    ),
    "no runner for the bucket": (200, {}, lambda: {"shared_partial": _shared(200)}, {}),
    "exact-M runner, padded call": (
        40,
        {},
        lambda: {"shared_partial": _shared(10)},
        {"bucket": 64, "direct": True},
    ),
    "exact-M runner": (
        64,
        {},
        lambda: {"shared_partial": _shared(16)},
        {"bucket": 64, "direct": True},
    ),
    "ragged": (
        30,
        {},
        lambda: {"shared_partial": _shared(20), "reduce_scatter_sizes": (7, 20, 2, 1)},
        {"bucket": 128, "ragged": True},
    ),
    "ragged on a runner without it": (
        30,
        {},
        lambda: {"shared_partial": _shared(20), "reduce_scatter_sizes": (7, 20, 2, 1)},
        {},
    ),
    "ragged, sizes do not add up": (
        30,
        {},
        lambda: {"shared_partial": _shared(20), "reduce_scatter_sizes": (7, 20, 2, 2)},
        {"bucket": 128, "ragged": True},
    ),
    "ragged, wrong rank count": (
        80,
        {},
        lambda: {"shared_partial": _shared(40), "reduce_scatter_sizes": (40, 40)},
        {"bucket": 128, "ragged": True},
    ),
    "ragged, above capacity": (
        40,
        {},
        lambda: {"shared_partial": _shared(33), "reduce_scatter_sizes": (1, 33, 3, 3)},
        {"bucket": 128, "ragged": True},
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_v2_runs_the_runtime_contract_as_legacy(monkeypatch, name):
    rows, extra, run_kw, runner_kw = _CASES[name]
    legacy_log, v2_log = [], []
    _JOINS.clear()
    seen, legacy_out = _legacy(
        monkeypatch,
        _runners(legacy_log, **runner_kw),
        _moe_args(rows, **extra),
        **run_kw(),
    )
    legacy_joins = len(_JOINS)
    _JOINS.clear()
    padded, v2_out = _v2(
        _runners(v2_log, **runner_kw), _moe_args(rows, **extra), **run_kw()
    )
    if isinstance(legacy_out, Exception):
        assert type(v2_out) is type(legacy_out), v2_out
        assert str(v2_out) == str(legacy_out)
        return
    assert not isinstance(v2_out, Exception), v2_out
    assert legacy_log == v2_log and legacy_joins == len(_JOINS)
    assert set(seen) == set(padded)
    for key, value in seen.items():
        other = padded[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    assert torch.equal(legacy_out, v2_out)


def test_direct_runner_refuses_output(monkeypatch):
    out = torch.empty(64, MODEL_DIM, dtype=torch.bfloat16)
    runner_kw = {"bucket": 64, "direct": True}
    _, legacy_err = _legacy(
        monkeypatch,
        _runners([], **runner_kw),
        _moe_args(64, output=out),
        shared_partial=_shared(16),
    )
    _, v2_err = _v2(
        _runners([], **runner_kw), _moe_args(64, output=out), shared_partial=_shared(16)
    )
    assert isinstance(legacy_err, RuntimeError) and str(legacy_err) == str(v2_err)


def test_reuse_synchronized_skips_the_barrier(monkeypatch):
    runner_kw = {"bucket": 64, "direct": True}
    legacy_log, v2_log = [], []
    kw = {"shared_partial": _shared(16), "reuse_is_synchronized": True}
    _legacy(monkeypatch, _runners(legacy_log, **runner_kw), _moe_args(64), **kw)
    _v2(_runners(v2_log, **runner_kw), _moe_args(64), **kw)
    assert legacy_log == v2_log and ("reuse_wait",) not in v2_log


def test_supports_and_bucket_match_the_runtime():
    runners = _runners([])
    runtime = CommFusedMoeRuntime(runners=runners)
    for tokens in (1, 2, 3, 4, 5, 32, 63, 64, 65, 100, 128, 129, 40000, 200000):
        configs = runners.configs
        assert stage2_rs.bucket_for(configs, tokens) == runtime.bucket_for(tokens)
        assert stage2_rs.supports(configs, tokens) == runtime.supports(tokens)


# The path and the legacy hook.


def _runtime_call(runtime, moe_args, **run_kw):
    call = {name: None for name in comm_fused.RUN_ARGS}
    call.update(reuse_is_synchronized=False, routes_may_be_missing=False)
    call.update(run_kw, runtime=runtime, moe_args=moe_args)
    return call


_CELLS = {"kernelName1": "the comm_fused row's"}


def _comm_rows_match(monkeypatch, cells=_CELLS):
    """v2's comm_fused row at each claimed bucket: `cells` (None: no row with the
    runtime's runner); the `(bucket, padded rows)` asked about."""
    asked = []

    def runtime_cells(runners, bucket, op):
        asked.append((bucket, op["topk_ids"].shape[0]))
        return cells

    monkeypatch.setattr(comm_fused, "_runtime_cells", runtime_cells)
    return asked


def test_claims_test_the_a8w4_claim_at_the_padded_count(monkeypatch):
    seen = []

    def a8w4_claims(op):
        seen.append(tuple(op["topk_ids"].shape))
        return True

    monkeypatch.setattr(comm_fused.a8w4, "claims", a8w4_claims)
    asked = _comm_rows_match(monkeypatch)
    runtime = CommFusedMoeRuntime(runners=_runners([]))
    assert comm_fused.claims(_runtime_call(runtime, _moe_args(3)))
    assert seen == [(4, TOPK)]
    ragged = _runtime_call(runtime, _moe_args(30), reduce_scatter_sizes=(7, 20, 2, 1))
    assert comm_fused.claims(ragged) and seen[-1] == (128, TOPK)
    # The runtime raises for these before its dispatch: legacy raises it.
    bad = _runtime_call(runtime, _moe_args(30), reduce_scatter_sizes=(1, 2))
    assert not comm_fused.claims(bad)
    # Arguments `fused_moe` does not take (the private `_fused_moe_impl` ones).
    private = _runtime_call(runtime, _moe_args(4, _q_dtype_a=torch.float8_e4m3fn))
    assert not comm_fused.claims(private)
    # `fused_moe` arguments `_fused_moe_impl` raises TypeError for, and block_size_M=0
    # (`fused_moe` reads it as -1, `_fused_moe_impl` does not).
    for extra in ({"splitk": 2}, {"clamp_shared": True}, {"block_size_M": 0}):
        assert not comm_fused.claims(_runtime_call(runtime, _moe_args(4, **extra)))
    assert comm_fused.claims(_runtime_call(runtime, _moe_args(4, block_size_M=-1)))
    assert len(seen) == 3
    assert asked == [(4, 4), (128, 128), (4, 4)]


def test_claims_need_a_comm_fused_row_with_the_runtime_s_runner(monkeypatch):
    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: True)
    asked = _comm_rows_match(monkeypatch, cells=None)
    runtime = CommFusedMoeRuntime(runners=_runners([]))
    assert not comm_fused.claims(_runtime_call(runtime, _moe_args(3)))
    assert asked == [(4, 4)]
    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: False)
    assert not comm_fused.claims(_runtime_call(runtime, _moe_args(3)))
    assert asked == [(4, 4)]  # the a8w4 claim first


def test_a_call_without_its_comm_fused_row_runs_on_legacy(monkeypatch):
    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: True)
    fake = _FakeLocal()
    monkeypatch.setattr(comm_fused.a8w4, "LOCAL", fake)
    _comm_rows_match(monkeypatch, cells=None)
    kw = {"shared_partial": _shared(4)}
    _, expected = _legacy(monkeypatch, _runners([]), _moe_args(3), **kw)
    with moev2_mode("comm_fused"):
        _, out = _legacy(monkeypatch, _runners([]), _moe_args(3), **kw)
    assert torch.equal(out, expected) and fake.ran == []


def test_the_hook_hands_calls_back_while_legacys_variables_are_set(monkeypatch, caplog):
    import logging

    from aiter.moev2 import envs, paths

    def reread():
        envs.tuned_table.cache_clear()
        envs.legacy_tuned_tables.cache_clear()

    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: True)
    asked = _comm_rows_match(monkeypatch)
    monkeypatch.setattr(paths, "active", lambda: (comm_fused.PATH,))
    monkeypatch.setattr(paths, "_warned_legacy_tables", False)
    monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE, raising=False)
    monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, raising=False)
    call = _runtime_call(CommFusedMoeRuntime(runners=_runners([])), _moe_args(3))
    try:
        monkeypatch.setenv("AITER_CONFIG_FMOE", "/t.csv")
        reread()
        with caplog.at_level(logging.WARNING, logger="aiter"):
            assert paths.route_to_v2(call, "comm_fused") is None
        assert "AITER_CONFIG_FMOE set" in caplog.text and asked == []
        monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, "/v2.csv")
        monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
        reread()
        with pytest.raises(ValueError, match="are both set"):
            paths.route_to_v2(call, "comm_fused")
        for name in ("AITER_CONFIG_FMOE", envs.AITER_MOEV2_TUNED_TABLE):
            monkeypatch.delenv(name)
        monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE_MODE)
        reread()
        assert paths.route_to_v2(call, "comm_fused") is comm_fused.PATH
        assert asked == [(4, 4)]
    finally:
        monkeypatch.undo()
        reread()


def test_a_runner_set_without_a_shape_is_not_claimed(monkeypatch):
    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: True)
    monkeypatch.setattr(
        comm_fused, "_local_problem", lambda op: pytest.fail("not looked up")
    )
    runtime = CommFusedMoeRuntime(runners=_runners([]))
    assert not hasattr(runtime.runners, "shape")
    assert not comm_fused.claims(_runtime_call(runtime, _moe_args(3)))


def test_run_declines_a_call_without_its_comm_fused_row(monkeypatch):
    _comm_rows_match(monkeypatch, cells=None)
    log = []
    call = _runtime_call(CommFusedMoeRuntime(runners=_runners(log)), _moe_args(3))
    call["shared_partial"] = _shared(4)
    assert comm_fused.run(call) is comm_fused.paths.DECLINED
    assert not [entry for entry in log if entry[0] in ("runner", "reuse_wait")]


def test_not_impl_args_are_the_signature_difference():
    import inspect

    fused_moe = inspect.signature(legacy_fused_moe.fused_moe).parameters
    impl = inspect.signature(legacy_fused_moe._fused_moe_impl).parameters
    assert comm_fused.NOT_IMPL_ARGS == set(fused_moe) - set(impl)


class _Call:
    def __init__(self, args):
        self.args = args

    def __getitem__(self, name):
        return self.args[name]


class _FakeLocal:
    """`a8w4.LOCAL`: resolves unless told not to; `run` launches the replaced stage 2
    as the executor does."""

    def __init__(self, resolves=True):
        self.resolves = resolves
        self.ran, self.resolved = [], []

    def resolve_call(self, op, cells=None):
        self.resolved.append(cells)
        return object() if self.resolves else None

    def run(self, op, stage2=None, cells=None):
        self.ran.append((tuple(op["topk_ids"].shape), str(stage2), cells))
        launch = stage2.segment.stage2_launch(_Call(op))
        return launch(
            ordinary_stage2=_ordinary_stage2,
            stage2_args=_STAGE2_ARGS,
            stage2_kwargs=_STAGE2_KWARGS,
        )


def test_the_hook_routes_claimed_calls_to_v2(monkeypatch):
    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: True)
    fake = _FakeLocal()
    monkeypatch.setattr(comm_fused.a8w4, "LOCAL", fake)
    monkeypatch.setattr(comm_fused, "_ADOPTED", {})
    _comm_rows_match(monkeypatch)
    log = []
    runtime = CommFusedMoeRuntime(runners=_runners(log))

    def legacy_impl(**kwargs):
        raise AssertionError("legacy must not run a claimed call")

    monkeypatch.setattr(legacy_fused_moe, "_fused_moe_impl", legacy_impl)
    with moev2_mode("comm_fused"):
        out = runtime.run(shared_partial=_shared(4), **_moe_args(3))
        runtime.run(shared_partial=_shared(4), **_moe_args(4))
    assert out.shape == (3, MODEL_DIM)
    assert [shape for shape, *_ in fake.ran] == [(4, TOPK), (4, TOPK)]
    assert fake.ran[0][1].startswith("comm_fused.stage2_rs(runner=_Config(m=4,")
    assert [cells for *_, cells in fake.ran] == [_CELLS] * 2
    assert fake.resolved == [_CELLS] * 2
    assert [entry[0] for entry in log] == ["shared", "runner"] * 2
    assert len(comm_fused._ADOPTED) == 1  # one segment per runner set


def test_a_declined_call_runs_on_legacy_before_any_barrier(monkeypatch):
    monkeypatch.setattr(comm_fused.a8w4, "claims", lambda op: True)
    monkeypatch.setattr(comm_fused.a8w4, "LOCAL", _FakeLocal(resolves=False))
    _comm_rows_match(monkeypatch)
    runner_kw = {"bucket": 64, "direct": True}
    kw = {"shared_partial": _shared(16)}
    _, expected = _legacy(monkeypatch, _runners([], **runner_kw), _moe_args(64), **kw)
    log = []
    with moev2_mode("comm_fused"):
        _, out = _legacy(monkeypatch, _runners(log, **runner_kw), _moe_args(64), **kw)
    # One barrier, legacy's: v2 declined before its own.
    assert log.count(("reuse_wait",)) == 1 and torch.equal(out, expected)


class _Named(Segment):
    def __init__(self, name, covers):
        self.name, self.covers = name, covers


def _local_plan():
    return MoePlan(
        (
            SegmentChoice(_Named("sort", (Op.SORT, Op.SORT))),
            SegmentChoice(_Named("g1", (Op.QUANT_A1, Op.QUANT_INTER))),
            SegmentChoice(_Named("g2", (Op.GEMM2, Op.REDUCE))),
        ),
        "tuned",
    )


def test_with_stage2_replaces_the_stage_2_only():
    choice = SegmentChoice(_Named("comm", (Op.GEMM2, Op.REDUCE)), Params(runner="r"))
    replaced = local.with_stage2(_local_plan(), choice)
    assert [str(c) for c in replaced.segments] == ["sort", "g1", "comm(runner=r)"]
    with pytest.raises(ValueError, match="does not cover"):
        local.with_stage2(
            _local_plan(), SegmentChoice(_Named("x", (Op.GEMM2, Op.GEMM2)))
        )


@dataclasses.dataclass(frozen=True)
class _Geometry:
    tp: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    comm: str
    add_shared: bool


def test_segment_lifecycle(monkeypatch):
    from aiter.ops.flydsl import comm_fused_moe_host as host

    configs = {1: _Config(1, 1)}
    adopted = stage2_rs.CommFusedStage2.adopt(SimpleNamespace(configs=configs))
    assert adopted.workspace_bytes() == MemoryNeed()
    adopted.prepare(None)  # allocated by its creator

    monkeypatch.setattr(host, "runner_arena_bytes", lambda c: 4096 * len(c))
    monkeypatch.setattr(stage2_rs, "_RUNNERS", {})
    shape = _Geometry(
        tp=2, model_dim=8, inter_dim=4, experts=8, topk=2, comm="ar", add_shared=True
    )
    created = stage2_rs.CommFusedStage2(configs, shape=shape)
    assert created.workspace_bytes() == MemoryNeed(symmetric=4096)
    ctx = engine_mod.EngineContext(SimpleNamespace(cpu_group=1), 0, 2, None, None)
    made, gathered = [], []

    def gather(out, mine, group):
        gathered.append((mine, group))
        out[:] = [mine] * len(out)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(
        host,
        "_LazyRunners",
        lambda *args: made.append(args) or SimpleNamespace(configs=configs),
    )
    created.prepare(ctx)
    assert gathered == [(None, 1)]  # the ranks' plan file rows: none
    assert made == [(ctx.comm, shape, configs)]
    assert created.workspace_bytes() == MemoryNeed()
    torch_group = engine_mod.EngineContext(object(), 0, 2, None, None)
    with pytest.raises(ValueError, match="GroupCoordinator"):
        stage2_rs.CommFusedStage2(configs, shape=shape).prepare(torch_group)
    wrong_tp = engine_mod.EngineContext(SimpleNamespace(cpu_group=1), 0, 4, None, None)
    with pytest.raises(ValueError, match="TP2"):
        stage2_rs.CommFusedStage2(configs, shape=shape).prepare(wrong_tp)


# The engine template.


def _template(monkeypatch, configs, side_inputs=("shared_partial",)):
    from aiter import QuantType, dtypes
    from aiter.moev2.dispatch import tuned_table

    resolved = []

    class _Local:
        def resolve_call(self, op, cells=None):
            m = op["topk_ids"].shape[0]
            assert cells == {"row": m}
            resolved.append(m)
            return SimpleNamespace(metadata={"m": m}, plan=_local_plan())

    monkeypatch.setattr(comm_fused.a8w4, "LOCAL", _Local())
    segment = stage2_rs.CommFusedStage2.adopt(_Runners(configs, []))
    made = []
    monkeypatch.setattr(
        stage2_rs,
        "CommFusedStage2",
        lambda configs, **kw: made.append((configs, kw)) or segment,
    )
    rows = {m: tuned_table.CommCells(Params(m=m), {"row": m}) for m in configs}
    monkeypatch.setattr(
        comm_fused.CommFusedTemplate, "_file_rows", lambda self, shape: rows
    )
    monkeypatch.setattr(stage2_rs, "runner_config", lambda p, s, m: configs[p["m"]])
    monkeypatch.setattr(
        "aiter.ops.flydsl.comm_fused_moe_host.comm_fused_shape", lambda **kw: None
    )
    weights = engine_mod.MoeWeights(
        torch.empty(16, 64, 128, dtype=dtypes.fp4x2, device="meta"),
        torch.empty(16, 256, 16, dtype=dtypes.fp4x2, device="meta"),
    )
    template = comm_fused.CommFusedTemplate(
        weights,
        topk=TOPK,
        options={"quant_type": QuantType.per_1x32},
        side_inputs=frozenset(side_inputs),
        tp_size=TP,
    )
    return template, segment, resolved


def test_template_plans_name_each_runner_bucket(monkeypatch):
    configs = {1: _Config(1, 1), 2: _Config(2, 2), 32768: _Config(32768, 32768)}
    template, segment, resolved = _template(monkeypatch, configs)
    one = template.resolve(1, False)
    assert one.stateful == (segment,) and one.side_inputs == {"shared_partial"}
    assert one.description.startswith(
        "comm_fused M=1: sort | g1 | comm_fused.stage2_rs(runner=_Config(m=1,"
    )
    assert template.resolve(4, False).description == "not served (no comm_fused runner)"
    top = template.resolve(32768, False)
    assert top.description.startswith("16385-32768 tokens: comm_fused M=32768: ")
    assert top.description.endswith(
        "; 32769-131071 tokens: not served (no comm_fused runner)"
    )
    assert top.supports(32768) and not top.supports(32769)
    assert resolved == [1, 2, 32768, 1, 32768]  # each row's cells, then the plans


def test_template_plans_split_the_top_tier_at_each_runner(monkeypatch):
    configs = {m: _Config(m, m) for m in (32768, 65536, 131072)}
    template, _, _ = _template(monkeypatch, configs)
    parts = template.resolve(32768, False).description.split("; ")
    assert [p.split(": ")[:2] for p in parts] == [
        ["16385-32768 tokens", "comm_fused M=32768"],
        ["32769-65536 tokens", "comm_fused M=65536"],
        ["65537-131071 tokens", "comm_fused M=131072"],
    ]
    top = template.resolve(131072, False).description
    assert top.startswith("131072 tokens: comm_fused M=131072: ")
    assert top.endswith("; 131073+ tokens: not served (no comm_fused runner)")


def test_template_side_inputs_are_checked(monkeypatch):
    with pytest.raises(ValueError, match="not per-call inputs"):
        _template(monkeypatch, {1: _Config(1, 1)}, side_inputs=("stage2_stream",))


def test_engine_supports_reports_unserved_counts(monkeypatch):
    configs = {1: _Config(1, 1), 2: _Config(2, 2), 4: _Config(4, 4)}
    template, _, _ = _template(monkeypatch, configs)
    eng = engine_mod.MoeEngine.create(template, max_tokens=8)
    with pytest.raises(engine_mod.EngineStateError):
        eng.supports(1)
    eng.prepare()
    assert eng.memory() == MemoryNeed()  # an adopted runner set: nothing to allocate
    supported = [eng.supports(t) for t in (1, 2, 3, 4, 5, 8, 9)]
    assert supported == [True, True, True, True, False, False, False]
    assert "not served" in eng.explain()


def test_padded_and_tuned_runs_decline_before_the_reuse_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(comm_fused.api, "op_call", lambda padded: {})
    monkeypatch.setattr(
        comm_fused.a8w4.LOCAL, "resolve_call", lambda op, cells=None: None
    )
    monkeypatch.setattr(stage2_rs, "wait_for_reuse", waits.append)
    assert comm_fused.run_padded(None, {}, "fused", {}) is comm_fused.paths.DECLINED
    assert comm_fused.run_tuned(None, {}, "fused", None) is comm_fused.paths.DECLINED
    assert waits == []
