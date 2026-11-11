# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`aiter.moev2.engine.MoeEngine`: lifecycle, buckets, per-call checks, the cross-rank
plan agreement, and local MoE through the migration paths (CPU with meta weights; on a
GPU, the same launches and output as `aiter.moev2.api.fused_moe`)."""

import dataclasses
import pathlib
import sys
import types

import pytest
import torch

from aiter.moev2 import api, engine, paths
from aiter.moev2.engine import (
    BucketPlan,
    EngineState,
    EngineStateError,
    LocalTemplate,
    MemoryBudgetExceeded,
    MoeEngine,
    MoeWeights,
    PlanDisagreement,
    PlanSource,
    StepInfo,
)
from aiter.moev2.problem import ParallelSpec
from aiter.moev2.segments.base import MemoryNeed, Segment

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_EP = types.SimpleNamespace(world_size=1, input_replication=1)


class _Stateful(Segment):
    name = "test.stateful"
    stateful = True

    def __init__(self, device=0, symmetric=0):
        self.calls = []
        self.failure = None
        self.need = MemoryNeed(device, symmetric)

    def workspace_bytes(self):
        return self.need

    def prepare(self, ctx):
        self.calls.append(("prepare", ctx.world_size))

    def health(self):
        return self.failure

    def reset(self):
        self.calls.append("reset")

    def release(self):
        self.calls.append("release")

    def abort(self):
        self.calls.append("abort")


class _Source(PlanSource):
    serves_parallel = True

    def __init__(self, tag="plan", **plan_fields):
        self.tag = tag
        self.plan_fields = plan_fields
        self.runs = []
        self.warmed = []
        self.weights_checked = 0

    def resolve(self, bucket, graph_capture):
        if self.tag == "raise":
            raise NotImplementedError(f"no plan for bucket {bucket}")

        def run(x, topk_ids, topk_weights, weights, side, hints, out):
            self.runs.append((bucket, dict(side), dict(hints), out))
            return x * 2

        return BucketPlan(
            bucket,
            f"{self.tag} {bucket}",
            run,
            warmup=lambda: self.warmed.append(bucket),
            **self.plan_fields,
        )

    def digests(self):
        return {"table": "abc"}

    def check_weights(self, weights):
        self.weights_checked += 1


def _forward(e, tokens=3, **kwargs):
    x = torch.ones(tokens, 4)
    ids = torch.zeros(tokens, 2, dtype=torch.int32)
    return e.forward(x, ids, torch.ones(tokens, 2), weights=None, **kwargs)


def _ep_engine(source, max_tokens=256):
    ctx = engine.EngineContext(None, 0, 1, None, None)
    return MoeEngine(source, ctx, _EP, max_tokens, 0)


def test_buckets_are_the_padded_m_tiers():
    assert engine.buckets(1) == (1,)
    assert engine.buckets(300) == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert engine.buckets(20000)[-2:] == (16384, 32768)
    assert engine.buckets(200000)[-3:] == (16384, 32768, 131072)
    assert engine.bucket_of(0) == 1
    assert engine.bucket_of(129) == 256
    with pytest.raises(ValueError):
        engine.buckets(0)
    for m in (1, 3, 255, 256, 257, 16385, 32767, 32768, 100000, 131072, 200000):
        assert engine.bucket_of(m) in engine.buckets(200000)


def test_bucket_ranges_cover_the_counts_of_each_bucket():
    assert engine.bucket_range(1) == (1, 1)
    assert engine.bucket_range(2) == (2, 2)
    assert engine.bucket_range(256) == (129, 256)
    assert engine.bucket_range(32768) == (16385, 131071)
    assert engine.bucket_range(131072) == (131072, None)
    for b in engine.buckets(200000):
        lowest, highest = engine.bucket_range(b)
        assert engine.bucket_of(lowest) == b
        assert highest is None or engine.bucket_of(highest) == b
        assert engine.bucket_of(lowest - 1) != b or lowest == 1


def test_lifecycle_without_capture():
    source = _Source()
    e = MoeEngine.create(source, max_tokens=100)
    assert e.state == EngineState.CREATED
    with pytest.raises(EngineStateError, match="prepare\\(\\) first"):
        _forward(e)
    e.prepare()
    assert e.state == EngineState.READY
    assert source.warmed == []
    out = _forward(e, tokens=3)
    assert torch.equal(out, torch.full((3, 4), 2.0))
    assert e.state == EngineState.ACTIVE
    assert source.runs[-1][0] == 4 and source.weights_checked == 1
    assert e.explain().splitlines()[0] == "bucket 1: plan 1"
    e.release()
    assert e.state == EngineState.RELEASED
    with pytest.raises(EngineStateError):
        _forward(e)
    with pytest.raises(EngineStateError):
        e.prepare()


def test_capture_warms_up_the_captured_buckets_only():
    source = _Source()
    e = MoeEngine.create(source, max_tokens=100, capture_max_tokens=4)
    with pytest.raises(EngineStateError):
        e.capture()
    e.prepare()
    assert e.state == EngineState.PREPARED
    with pytest.raises(EngineStateError, match="capture\\(\\) first"):
        _forward(e)
    e.capture()
    assert source.warmed == [1, 2, 4]
    assert e.state == EngineState.READY
    assert "bucket 4 (captured)" in e.explain() and "bucket 8 (captured)" not in (
        e.explain()
    )


def test_capture_covers_the_bucket_of_capture_max_tokens():
    source = _Source()
    e = MoeEngine.create(source, max_tokens=100, capture_max_tokens=5)
    assert e.captured == {1, 2, 4, 8}  # 5 tokens run in bucket 8
    e.prepare()
    e.capture()
    assert source.warmed == [1, 2, 4, 8]


def test_a_captured_bucket_needs_a_capturable_plan():
    e = MoeEngine.create(
        _Source(graph_capturable=False), max_tokens=100, capture_max_tokens=1
    )
    with pytest.raises(NotImplementedError, match="graph-capturable"):
        e.prepare()


def test_tokens_above_max_tokens_are_refused():
    e = MoeEngine.create(_Source(), max_tokens=10)
    e.prepare()
    with pytest.raises(ValueError, match="max_tokens=10"):
        _forward(e, tokens=11)


def test_ep_needs_a_group_and_a_step():
    with pytest.raises(ValueError, match="comm="):
        MoeEngine.create(_Source(), max_tokens=10, parallel=_EP)
    source = _Source()
    e = _ep_engine(source)
    e.prepare()
    with pytest.raises(ValueError, match="StepInfo"):
        _forward(e)
    with pytest.raises(ValueError, match="below"):
        _forward(e, tokens=3, step=StepInfo(2))
    with pytest.raises(ValueError, match="max_tokens"):
        _forward(e, tokens=3, step=StepInfo(300))
    _forward(e, tokens=3, step=StepInfo(100))
    assert source.runs[-1][0] == 128
    _forward(e, tokens=0, step=StepInfo(100))
    assert source.runs[-1][0] == 128


def test_ep_refuses_plans_that_cannot_run_without_tokens():
    with pytest.raises(NotImplementedError, match="no tokens"):
        _ep_engine(_Source(zero_tokens_ok=False)).prepare()
    MoeEngine.create(_Source(zero_tokens_ok=False), max_tokens=4).prepare()


def test_side_inputs_must_match_and_hints_are_filtered():
    source = _Source(side_inputs=frozenset({"a1_scale"}), hints=frozenset({"next"}))
    e = MoeEngine.create(source, max_tokens=8)
    e.prepare()
    with pytest.raises(ValueError, match="missing \\['a1_scale'\\]"):
        _forward(e)
    with pytest.raises(ValueError, match="not read \\['a2_scale'\\]"):
        _forward(e, side_inputs={"a1_scale": 1, "a2_scale": 2})
    _forward(e, side_inputs={"a1_scale": 1}, hints={"next": 5, "other": 6})
    assert source.runs[-1][1:3] == ({"a1_scale": 1}, {"next": 5})


def test_optional_side_inputs_and_what_supports_them():
    source = _Source(
        side_inputs=frozenset({"a1_scale"}),
        optional_side_inputs=frozenset({"res", "norm"}),
        supports=lambda tokens: tokens != 5,
        supports_optional=lambda tokens, names: tokens > 2 and names == {"res"},
    )
    e = MoeEngine.create(source, max_tokens=8)
    e.prepare()
    _forward(e, side_inputs={"a1_scale": 1})
    _forward(e, side_inputs={"a1_scale": 1, "res": 2})
    assert source.runs[-1][1] == {"a1_scale": 1, "res": 2}
    with pytest.raises(ValueError, match="optional \\['norm', 'res'\\]"):
        _forward(e, side_inputs={"res": 2})
    with pytest.raises(ValueError, match="not read \\['other'\\]"):
        _forward(e, side_inputs={"a1_scale": 1, "other": 2})
    assert e.supports(3) and e.supports(3, side_inputs=("a1_scale",))
    assert e.supports(3, side_inputs=("res",)) and not e.supports(2, ("res",))
    assert e.supports(3, side_inputs="res") and not e.supports(2, "res")
    assert e.supports(2, side_inputs=None)
    assert not e.supports(3, side_inputs=("res", "norm"))
    assert not e.supports(3, side_inputs=("other",))
    assert not e.supports(5) and not e.supports(5, side_inputs=("res",))
    assert not e.supports(9, side_inputs=("res",))


def test_out_receives_the_result():
    e = MoeEngine.create(_Source(output_lifetime="until_next_forward"), max_tokens=8)
    e.prepare()
    out = torch.zeros(3, 4)
    assert _forward(e, out=out) is out
    assert torch.equal(out, torch.full((3, 4), 2.0))


class _TupleSource(_Source):
    def __init__(self, write_out):
        super().__init__()
        self.write_out = write_out

    def resolve(self, bucket, graph_capture):
        def run(x, topk_ids, topk_weights, weights, side, hints, out):
            y = x * 2
            if out is not None and self.write_out:
                y = out.copy_(y)
            return y, x + 1

        return BucketPlan(bucket, f"tuple {bucket}", run)


@pytest.mark.parametrize("write_out", [True, False])
def test_out_receives_the_first_element_of_a_tuple_result(write_out):
    e = MoeEngine.create(_TupleSource(write_out), max_tokens=8)
    e.prepare()
    out = torch.zeros(3, 4)
    y, extra = _forward(e, out=out)
    assert y is out and torch.equal(out, torch.full((3, 4), 2.0))
    assert torch.equal(extra, torch.full((3, 4), 2.0))
    y, extra = _forward(e)
    assert torch.equal(y, torch.full((3, 4), 2.0))


def test_stateful_segments_follow_the_engine():
    segment = _Stateful()
    e = MoeEngine.create(_Source(stateful=(segment,)), max_tokens=8)
    e.prepare()
    assert segment.calls == [("prepare", 1)]  # one instance, prepared once
    e.reset()
    assert e.health().ok
    e.release()
    assert segment.calls[1:] == ["reset", "release"]


def test_a_failed_health_check_only_allows_abort():
    segment = _Stateful()
    e = MoeEngine.create(_Source(stateful=(segment,)), max_tokens=8)
    e.prepare()
    _forward(e)
    segment.failure = "peer 3 timed out"
    health = e.health()
    assert not health.ok and health.reasons == ("peer 3 timed out",)
    assert e.state == EngineState.FAILED
    segment.failure = None
    assert not e.health().ok  # stays failed
    for call in (_forward, MoeEngine.reset, MoeEngine.release):
        with pytest.raises(EngineStateError, match="only abort"):
            call(e)
    e.abort()
    e.abort()
    assert e.state == EngineState.RELEASED
    assert segment.calls[-1] == "abort" and segment.calls.count("abort") == 1


def test_plan_hash_follows_the_plans():
    def prepared(tag):
        e = MoeEngine.create(_Source(tag), max_tokens=8)
        e.prepare()
        return e.plan_hash

    assert prepared("a") == prepared("a") != prepared("b")


class _TwoSegmentSource(_Source):
    """Buckets up to 4 use `small`, larger buckets `large` (one instance each)."""

    def __init__(self, small, large):
        super().__init__()
        self.small, self.large = small, large

    def resolve(self, bucket, graph_capture):
        plan = super().resolve(bucket, graph_capture)
        segment = self.small if bucket <= 4 else self.large
        return dataclasses.replace(plan, stateful=(segment,))


def test_memory_counts_each_stateful_instance_once():
    small, large = _Stateful(device=10, symmetric=100), _Stateful(device=1, symmetric=7)
    e = MoeEngine.create(_TwoSegmentSource(small, large), max_tokens=64)
    with pytest.raises(EngineStateError):
        e.memory()
    e.prepare()
    assert e.memory() == MemoryNeed(device=11, symmetric=107)
    assert small.calls == large.calls == [("prepare", 1)]

    def plan_hash(symmetric):
        other = MoeEngine.create(
            _TwoSegmentSource(_Stateful(10, symmetric), _Stateful(1, 7)), max_tokens=64
        )
        other.prepare()
        return other.plan_hash

    assert plan_hash(100) == e.plan_hash != plan_hash(200)


def test_memory_budget_is_checked_before_any_segment_allocates():
    small, large = _Stateful(device=10, symmetric=100), _Stateful(device=1, symmetric=7)
    e = MoeEngine.create(
        _TwoSegmentSource(small, large), max_tokens=64, memory_budget=117
    )
    with pytest.raises(MemoryBudgetExceeded, match="need 118 bytes per rank"):
        e.prepare()
    assert small.calls == large.calls == [] and e.state == EngineState.CREATED
    e = MoeEngine.create(
        _TwoSegmentSource(small, large), max_tokens=64, memory_budget=118
    )
    e.prepare()
    assert e.state == EngineState.READY


def _coordinator_worker(rank, init_file, budgets, results):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        # What aiter's GroupCoordinator offers the engine.
        coordinator = types.SimpleNamespace(
            cpu_group=dist.new_group([0, 1], backend="gloo"),
            rank_in_group=rank,
            world_size=2,
            device=torch.device("cpu"),
        )
        segment = _Stateful(device=5, symmetric=50)
        e = MoeEngine.create(
            _Source(stateful=(segment,)),
            max_tokens=8,
            parallel=types.SimpleNamespace(world_size=2, input_replication=1),
            comm=coordinator,
            memory_budget=budgets[rank],
        )
        assert e.ctx.comm is coordinator and e.ctx.rank == rank
        assert e.ctx.device == torch.device("cpu")
        try:
            e.prepare()
            results[rank] = ("ok", e.plan_hash, segment.calls)
        except PlanDisagreement as exc:
            results[rank] = ("disagree", str(exc), segment.calls)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("budgets", [(None, None), (100, 55)])
def test_ranks_agree_over_a_group_coordinator(tmp_path, budgets):
    import torch.multiprocessing as mp

    results = mp.get_context("spawn").Manager().dict()
    mp.spawn(
        _coordinator_worker, args=(str(tmp_path / "init"), budgets, results), nprocs=2
    )
    if budgets[0] == budgets[1]:
        assert results[0] == results[1] and results[0][0] == "ok"
        assert results[0][2] == [("prepare", 2)]
    else:
        # Different budgets could let one rank refuse while its peer enters a
        # collective prepare, so the ranks must agree on the budget too.
        expected = "ranks 0 and 1 resolved different MoE plans: memory_budget 100 vs 55"
        assert results[0] == results[1] == ("disagree", expected, [])


def _agreement_worker(rank, init_file, tags, results):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        e = MoeEngine.create(
            _Source(tags[rank]),
            max_tokens=8,
            parallel=types.SimpleNamespace(world_size=2, input_replication=1),
            comm=dist.group.WORLD,
        )
        try:
            e.prepare()
            results[rank] = ("ok", e.plan_hash)
        except PlanDisagreement as exc:
            results[rank] = ("disagree", str(exc))
        except NotImplementedError as exc:
            results[rank] = ("raised", str(exc))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tags", [("a", "a"), ("a", "b"), ("a", "raise")])
def test_ranks_agree_on_their_plans(tmp_path, tags):
    import torch.multiprocessing as mp

    results = mp.get_context("spawn").Manager().dict()
    mp.spawn(_agreement_worker, args=(str(tmp_path / "init"), tags, results), nprocs=2)
    if tags[0] == tags[1]:
        assert results[0] == results[1] and results[0][0] == "ok"
    elif tags[1] == "raise":
        # The failing rank raises its error; its peer learns it instead of hanging.
        assert results[1] == ("raised", "no plan for bucket 1")
        assert results[0] == (
            "disagree",
            "rank 1 could not prepare: NotImplementedError: no plan for bucket 1",
        )
    else:
        assert results[0] == results[1]  # both ranks raise the same message
        assert results[0] == (
            "disagree",
            "ranks 0 and 1 resolved different MoE plans: bucket 1: a 1 vs b 1",
        )


class _FailsOnRank(_Stateful):
    def __init__(self, rank):
        super().__init__()
        self.fail_on = rank

    def prepare(self, ctx):
        super().prepare(ctx)
        if ctx.rank == self.fail_on:
            raise ValueError("no arena")


def _prepare_failure_worker(rank, init_file, results):
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        before, failing, after = _Stateful(), _FailsOnRank(1), _Stateful()
        e = MoeEngine.create(
            _Source(stateful=(before, failing, after)),
            max_tokens=8,
            parallel=types.SimpleNamespace(world_size=2, input_replication=1),
            comm=dist.group.WORLD,
        )
        try:
            e.prepare()
            outcome = ("ok",)
        except Exception as exc:  # noqa: BLE001
            outcome = (type(exc).__name__, str(exc))
        results[rank] = (
            *outcome,
            e.state.value,
            [s.calls for s in (before, failing, after)],
        )
    finally:
        dist.destroy_process_group()


def test_a_segment_prepare_failing_on_one_rank_raises_on_every_rank(tmp_path):
    import torch.multiprocessing as mp

    results = mp.get_context("spawn").Manager().dict()
    mp.spawn(_prepare_failure_worker, args=(str(tmp_path / "init"), results), nprocs=2)
    prepared = ("prepare", 2)
    assert results[1] == (
        "ValueError",
        "no arena",
        "created",
        [[prepared, "abort"], [prepared], []],
    )
    assert results[0] == (
        "PeerPrepareError",
        "rank 1 could not prepare test.stateful: ValueError: no arena",
        "created",
        [[prepared, "abort"], [prepared, "abort"], []],
    )


# Local MoE through the migration paths.


def _meta_weights(dtype=torch.bfloat16, experts=8, model_dim=512, inter_dim=256):
    meta = torch.device("meta")
    return MoeWeights(
        torch.empty(experts, 2 * inter_dim, model_dim, dtype=dtype, device=meta),
        torch.empty(experts, model_dim, inter_dim, dtype=dtype, device=meta),
    )


@pytest.fixture
def gfx950():
    from moe_dispatch.arch_sim import simulated_arch

    with simulated_arch("gfx950"):
        yield


def test_local_template_resolves_every_bucket(gfx950):
    e = MoeEngine.create(LocalTemplate(_meta_weights(), topk=2), max_tokens=300)
    e.prepare()
    lines = e.explain().splitlines()
    assert len(lines) == len(engine.buckets(300)) == 10
    assert lines[0].startswith("bucket 1: a16w16: sort | quant.a1 | ck.gemm1")
    layout = e.weight_layout()
    assert (layout.shuffle, layout.gate_mode) == ("none", "separated")


def test_local_template_resolves_as_the_path_dispatches(gfx950):
    template = LocalTemplate(_meta_weights(), topk=2)
    plan = template.resolve(64, graph_capture=False)
    call = template._meta_call(64)
    path, path_call = next(api.claiming_paths(call))
    resolution = path.resolve(path_call)
    assert plan.description == f"{path.name}: {resolution.plan.describe()}"
    assert plan.identity == engine._canonical(resolution.metadata)
    assert "0x" not in plan.identity  # no object addresses: the same on every rank


def test_the_local_template_agrees_on_the_tuned_table(gfx950, tmp_path, monkeypatch):
    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned, tuned_plans

    template = LocalTemplate(_meta_weights(), topk=2)

    def digests(name, text=None):
        if name is None:
            monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE, raising=False)
            monkeypatch.delenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, raising=False)
        else:
            (tmp_path / name).write_text(text)
            monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(tmp_path / name))
            monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "override")
        tuned.clear_cache()
        return template.digests()

    empty = ",".join(tuned_plans.COLUMNS) + "\n"
    other = ",".join(tuned_plans.COLUMNS + tuned_plans.INFO_COLUMNS) + "\n"
    unset = digests(None)
    assert set(unset) == {"tuned_table"}
    a = digests("a.csv", empty)
    assert a != unset
    template.resolve(1, graph_capture=False)
    (tmp_path / "a.csv").write_text(other)
    assert template.digests() == a  # of the read the plans came from
    assert digests("same.csv", empty) == a  # the content, not the path
    b = digests("b.csv", other)
    record = {"aiter": "v", "buckets": [], "memory": [0, 0], "memory_budget": None}
    for theirs in (b, unset):
        with pytest.raises(PlanDisagreement, match="tuned_table differ"):
            engine._check_agreement(
                [{**record, "digests": a}, {**record, "digests": theirs}]
            )
    engine._check_agreement([{**record, "digests": a}, {**record, "digests": dict(a)}])
    digests(None)


def test_local_template_checks_the_weights(gfx950):
    template = LocalTemplate(_meta_weights(), topk=2)
    template.check_weights(_meta_weights())
    with pytest.raises(ValueError, match="weights.w1"):
        template.check_weights(_meta_weights(dtype=torch.float16))
    with pytest.raises(ValueError, match="weights.w1"):
        template.check_weights(_meta_weights(model_dim=1024))


def test_local_template_serves_no_parallel_spec():
    with pytest.raises(ValueError, match="without a ParallelSpec"):
        MoeEngine.create(
            LocalTemplate(_meta_weights(), topk=2), max_tokens=8, parallel=_EP
        )


def test_replicated_input_is_refused():
    parallel = ParallelSpec("ep", 2, "a2a", "mega_moe_v2", 64, 4, input_replication=2)
    with pytest.raises(ValueError, match="input_replication=2: no plan"):
        MoeEngine.create(_Source(), max_tokens=8, parallel=parallel)


def test_local_template_splits_a_bucket_at_an_activation_bound(monkeypatch):
    from aiter.moev2 import policy

    monkeypatch.setattr(policy, "activation_dtype_bounds", lambda: (200, 1000))
    template = LocalTemplate(_meta_weights(), topk=2)
    resolved = []

    def resolve_count(tokens):
        resolved.append(tokens)
        return ("bf16 plan", "x") if tokens < 200 else ("fp8 plan", "y")

    monkeypatch.setattr(template, "_resolve_count", resolve_count)
    plan = template.resolve(256, graph_capture=False)
    assert resolved == [129, 199, 200, 256]
    assert plan.description == "129-199 tokens: bf16 plan; 200-256 tokens: fp8 plan"
    assert plan.identity == "129-199 tokens: x; 200-256 tokens: y"
    assert plan.warmup.args == ((129, 200),)
    resolved.clear()
    assert template.resolve(512, False).description == "fp8 plan"
    assert resolved == [257, 512]
    assert template.resolve(131072, False).description == "fp8 plan"


def test_local_template_options():
    weights = _meta_weights()
    with pytest.raises(ValueError, match="output"):
        LocalTemplate(weights, topk=2, options={"output": None})
    with pytest.raises(ValueError, match="a1_scale"):
        LocalTemplate(weights, topk=2, options={"a1_scale": None})
    with pytest.raises(ValueError, match="no_such"):
        LocalTemplate(weights, topk=2, options={"no_such": 1})
    with pytest.raises(NotImplementedError, match="shared_w1"):
        LocalTemplate(weights, topk=2, options={"shared_w1": None})
    with pytest.raises(ValueError, match="w1_scale"):
        LocalTemplate(weights, topk=2, side_inputs={"w1_scale"})


@pytest.fixture
def no_paths():
    saved = paths.registered()
    paths._REGISTRY.clear()
    yield
    paths._REGISTRY.clear()
    paths._REGISTRY.update(saved)
    paths.reload()


def test_local_template_without_a_path(no_paths):
    with pytest.raises(NotImplementedError, match="no aiter.moev2 path"):
        LocalTemplate(_meta_weights(), topk=2).resolve(1, False)


def test_local_template_refuses_paths_without_resolve(no_paths):
    paths.register(paths.Path("a16w16", lambda call: True, lambda call: None, False))
    with pytest.raises(NotImplementedError, match="path a16w16 .* does not run it"):
        LocalTemplate(_meta_weights(), topk=2).resolve(1, False)


def test_local_template_skips_paths_that_decline_the_key(no_paths):
    def resolve(call):
        return None

    paths.register(
        paths.Path(
            "a16w16", lambda call: True, None, False, entry="op", resolve=resolve
        )
    )
    with pytest.raises(NotImplementedError, match="no aiter.moev2 path"):
        LocalTemplate(_meta_weights(), topk=2).resolve(1, False)


# GPU: the engine launches what `aiter.moev2.fused_moe` launches.

# (path, gpu_differential case, buckets `capture()` warms up): legacy fails below 512
# tokens on the a8w8_pt case (todo 5.32) and from 4 on the blockscale one (5.33).
_GPU_CASES = [
    ("a16w16", (16, 7168, 256, 256, 8, "Silu", False, "bf16", False), 16),
    ("a16w16", (300, 4096, 512, 64, 4, "Silu", True, "bf16", False), 16),
    ("a8w8_pt", (512, 4096, 384, 128, 8, "Gelu", False, "bf16", False), 0),
    ("a8w8_blockscale", (256, 4096, 384, 128, 8, "Gelu", False, "bf16", False), 2),
    ("a4w4", (64, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4"), 16),
]


def _template_of(call):
    weights = MoeWeights(
        *(call.get(k) for k in ("w1", "w2", "w1_scale", "w2_scale", "bias1", "bias2"))
    )
    defaults = engine._fused_moe_defaults()
    side = {k for k in engine.LOCAL_SIDE_INPUTS if call.get(k) is not None}
    options = {
        k: v
        for k, v in call.items()
        if k in defaults
        and k not in engine._PER_CALL
        and k not in engine.LOCAL_SIDE_INPUTS
        and v is not defaults[k]
    }
    template = LocalTemplate(
        weights,
        topk=call["topk_ids"].shape[1],
        hidden_dtype=call["hidden_states"].dtype,
        options=options,
        side_inputs=side,
    )
    return template, weights, {k: call[k] for k in side}


def _matches_fused_moe(e, call, weights, side):
    """`e.forward` launches what `aiter.moev2.fused_moe(**call)` launches, with an
    output within the noise of two `fused_moe` runs."""
    from moe_dispatch import differential

    from aiter.moev2 import hooks

    def launch(fn):
        with hooks.set_kernel_bench_callable() as launches:
            out = fn()
        torch.cuda.synchronize()
        return out, launches

    ref, ref_launches = launch(lambda: api.fused_moe(**call))
    ref2, _ = launch(lambda: api.fused_moe(**call))
    args = (call["hidden_states"], call["topk_ids"], call["topk_weight"])
    out_buf = torch.empty_like(ref)
    out, launches = launch(
        lambda: e.forward(*args, weights=weights, side_inputs=side, out=out_buf)
    )
    assert out is out_buf
    problems = differential.compare_launches(ref_launches, launches, call, ref, out)
    assert not problems, problems
    differential._check_output("engine", ref, ref2, out)


def _plan_of_call(template, call, weights, side):
    """`(description, identity)` of the plan the real call resolves to."""
    full = template._call(
        call["hidden_states"],
        call["topk_ids"],
        call["topk_weight"],
        weights,
        side,
        None,
    )
    for path, path_call in api.claiming_paths(full):
        resolution = path.resolve(path_call)
        if resolution is not None and resolution.plan is not None:
            return (
                f"{path.name}: {resolution.plan.describe()}",
                engine._canonical(resolution.metadata),
            )
    raise AssertionError("no path resolves the call")


@pytest.mark.parametrize("path_name,case,capture_max_tokens", _GPU_CASES)
def test_engine_launches_what_fused_moe_launches(path_name, case, capture_max_tokens):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from moe_dispatch.gpu_differential import _make_call

    call = _make_call(path_name, case)
    template, weights, side = _template_of(call)
    e = MoeEngine.create(
        template, max_tokens=512, capture_max_tokens=capture_max_tokens
    )
    e.prepare()
    # The plan prepare() recorded is the one of the call (its per-call checks see
    # tensors on the weights' device, not meta ones).
    plan = e._plans[engine.bucket_of(case[0])]
    assert (plan.description, plan.identity) == _plan_of_call(
        template, call, weights, side
    )
    if capture_max_tokens:
        e.capture()  # warm-up with the real weights
    _matches_fused_moe(e, call, weights, side)
    e.release()


def test_an_inline_sort_row_is_prepared_as_it_runs():
    """The DeepSeek-V3 a4w4 row of 32 tokens is an MXMOE inline-sort row: its per-call
    check (`_mxfp4_inline_sort_unsupported`) compares the devices of the call's tensors,
    so prepare() resolves it with tensors on the weights' device. Launches are not
    compared: the inline sort's slot order differs between two `fused_moe` runs."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from moe_dispatch import differential
    from moe_dispatch.gpu_differential import _make_call

    call = _make_call(
        "a4w4", (32, 7168, 256, 256, 8, "Silu", False, "bf16", False, "fp4")
    )
    template, weights, side = _template_of(call)
    e = MoeEngine.create(template, max_tokens=32, capture_max_tokens=32)
    e.prepare()
    description, identity = _plan_of_call(template, call, weights, side)
    if "sort.aux" not in description:
        pytest.skip(f"no inline-sort row here: {description}")
    assert (e._plans[32].description, e._plans[32].identity) == (description, identity)
    e.capture()
    ref, ref2 = api.fused_moe(**call), api.fused_moe(**call)
    args = (call["hidden_states"], call["topk_ids"], call["topk_weight"])
    out = e.forward(*args, weights=weights, side_inputs=side)
    torch.cuda.synchronize()
    differential._check_output("engine", ref, ref2, out)
    e.release()


def test_a_bucket_with_two_activation_formats():
    """gfx950 MXFP4 Swiglu: bf16 activations (a16w4) below 256 tokens, fp4 (a4w4) from
    256, both in bucket 256."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from moe_dispatch.gpu_differential import _make_call

    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("gfx950 MXFP4")
    call = _make_call(
        "a4w4", (256, 3072, 512, 128, 4, "Swiglu", False, "bf16", False, "fp4")
    )
    template, weights, side = _template_of(call)
    e = MoeEngine.create(template, max_tokens=256, capture_max_tokens=256)
    e.prepare()
    line = e.explain().splitlines()[-1]
    assert line.startswith("bucket 256 (captured): 129-255 tokens: a16w4: "), line
    assert "; 256 tokens: a4w4: " in line, line
    e.capture()  # runs 129 and 256 tokens
    for m in (200, 256):
        sliced = {
            **call,
            **{k: call[k][:m] for k in ("hidden_states", "topk_ids", "topk_weight")},
        }
        _matches_fused_moe(e, sliced, weights, side)
    e.release()
