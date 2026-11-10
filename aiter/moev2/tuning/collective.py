# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Measuring a TP MoE call with its collective, on the ranks of a `tuning.group`.

A `CommProblem` is one rank's local a8w4 key (M: the tokens every rank gets) with the
TP collective after it (`comm`: "ar" all-reduce, "rs" reduce-scatter; the shared
partial added or not). Each rank builds the call of the local key with `tuning.inputs`
(the same activations and routing on every rank, its own weights) and a random shared
partial of the output rows (M, or M / tp for "rs"); the reference is the ranks' torch
references summed, as the collective sums them (all-reduced, or reduce-scattered),
plus the shared partial.

A task runs a local plan (`task.plan_of`: a structure with candidates, or what
dispatch resolves today) either ordinarily, its output then all-reduced (shared
partial added first) or reduce-scattered (shared partial added after) by the TP
`GroupCoordinator` as the legacy comm-fused tuner's baseline does, or with its stage 2
replaced by `comm_fused.stage2_rs` over a runner (a `runner` spec: the family and
non-default params of a runner config, `config_record`'s), as `CommFusedMoeRuntime.run`
runs it (`prepare_call`, the plan, `finish`). Runner sets are built as the runtime's
(`_LazyRunners`: one packed arena and communicator), one config each, a few kept.

Timing (`measure.deploy_mode`): graph mode captures one run per rotating copy under
the group's `graph_capture` and replays it back to back, each replay timed by its own
events; eager mode times each run from the end of an all-reduce of a flag, so every
rank starts together on the device (the host does not wait for it, so the launches
overlap it); a sample is the slowest rank's. Every step a rank can fail in outside a
collective (its inputs, the rotating copies, each plan's build and first run) is
agreed across ranks before the next collective, so a rank failing alone fails the task
on every rank instead of leaving the others in a collective; a failure inside one (a
runner set's creation or release) is the group's timeout. A runner set that failed on
some ranks only after its collectives leaves the ranks keeping different sets: that
restarts the group. The rotating copies are as many on every rank (the least free
memory of any).
"""

import collections
import os
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from .. import api
from ..dispatch.plan import SegmentChoice
from ..execute.executor import prologue
from ..migration.local import execute, with_stage2
from ..policy import quant_remap
from ..segments import comm_fused as stage2_rs
from ..segments.base import Params
from . import inputs as tuning_inputs
from . import measure
from .pool import Restart
from .task import _device_usable, plan_of

COMMS = ("ar", "rs")
# Runner sets kept per rank (each holds a communicator and its arena).
MAX_RUNNER_SETS = 6


@dataclass(frozen=True)
class CommProblem:
    local: object  # MoeProblem: one rank's call, every rank's tokens
    tp: int
    comm: str  # "ar" or "rs"
    add_shared: bool
    gpu_model: str  # the runner rows' `gpu_model`

    def __post_init__(self):
        if self.comm not in COMMS:
            raise ValueError(f"comm must be one of {COMMS}, got {self.comm!r}")

    @property
    def output_rows(self) -> int:
        m = self.local.shape.m
        return m if self.comm == "ar" else m // self.tp

    def shape_key(self):
        """The runner set's `ShapeKey`."""
        from aiter.ops.flydsl.comm_fused_moe_host import ShapeKey

        s = self.local.shape
        return ShapeKey(
            self.local.arch,
            s.model_dim,
            s.inter_dim,
            s.experts,
            s.topk,
            self.tp,
            self.local.cu_num,
            add_shared=self.add_shared,
            comm=self.comm,
            gpu_model=self.gpu_model,
        )

    def __str__(self):
        shared = "+shared" if self.add_shared else ""
        return f"TP{self.tp} {self.comm}{shared} {self.local}"


def runner_spec(config) -> tuple:
    """`(family, ((param, value), ...))`: a runner config without its shape and M."""
    from aiter.ops.flydsl.comm_fused_moe_host import config_record

    record = config_record(config)
    return record["family"], tuple(sorted(record["params"].items()))


def runner_config(problem, spec):
    """The runner config of `spec` for `problem` (M: the local key's), as the runtime
    runs a plan file's runner (`comm_fused.runner_config`)."""
    family, params = spec
    return stage2_rs.runner_config(
        Params(dict(params), family=family), problem.shape_key(), problem.local.shape.m
    )


def spec_of_record(record) -> tuple:
    return record["family"], tuple(sorted(record["params"].items()))


@dataclass(frozen=True)
class CommTask:
    problem: CommProblem
    path: str  # the local plan's path (`paths.registered()` name)
    structure: tuple | None = None  # as `task.Task`; None: what dispatch resolves
    candidates: tuple | None = None
    runner: tuple | None = None  # `runner_spec`; None: the ordinary stage 2
    mode: str | None = None
    iters: int = 50
    warmup: int = 3
    seed: int = 0


@dataclass(frozen=True)
class CommRivals:
    """Plans of one problem timed in turn."""

    problem: CommProblem
    path: str
    plans: tuple  # ((structure, candidates, runner), ...) as in `CommTask`
    mode: str | None = None
    rounds: int = 4
    iters: int = 50
    warmup: int = 3
    seed: int = 0


def setup():
    """The rank's distributed init (a `Group` setup): the TP group over every rank."""
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    ensure_model_parallel_initialized(world, 1)
    group = get_tp_group()
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group.device_group)
    torch.cuda.synchronize()


def teardown():
    """The rank's release at the group's close (a `Group` teardown): its runner sets,
    the model-parallel groups (the shared-memory broadcast queue among them) and the
    process group."""
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    _runner_sets.clear()
    _cached.clear()
    torch.cuda.synchronize()
    destroy_model_parallel()
    destroy_distributed_environment()


def _tp():
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group()


def agree(reason, tp=None):
    """The first rank's failure reason (prefixed by its rank, but rank 0's), or None
    when every rank passes."""
    tp = tp or _tp()
    reasons = [None] * tp.world_size
    dist.all_gather_object(reasons, reason, group=tp.cpu_group)
    for rank, r in enumerate(reasons):
        if r is not None:
            return r if rank == 0 else f"rank {rank}: {r}"
    return None


def slowest(samples, tp=None) -> list:
    """Each sample's maximum across the ranks."""
    tp = tp or _tp()
    local = torch.tensor(samples, dtype=torch.float64, device="cuda")
    every = torch.empty(
        (tp.world_size, local.numel()), dtype=torch.float64, device="cuda"
    )
    dist.all_gather_into_tensor(every, local, group=tp.device_group)
    return every.max(dim=0).values.tolist()


@dataclass
class CommInputs:
    call: dict  # a rank's `api.fused_moe` arguments
    shared: torch.Tensor | None  # its shared partial (output rows)
    reference: torch.Tensor  # its output rows after the collective


def build_inputs(problem, seed=0) -> CommInputs:
    """A rank's call, shared partial and reference (collective)."""
    tp = _tp()
    rank = tp.rank_in_group
    if problem.tp != tp.world_size:
        raise tuning_inputs.Unsupported(
            f"the problem is TP{problem.tp}; the group has {tp.world_size} ranks"
        )
    if problem.local.shape.m % problem.tp and problem.comm == "rs":
        raise tuning_inputs.Unsupported(
            f"M={problem.local.shape.m} does not split over {problem.tp} ranks"
        )
    s = problem.local.shape
    try:
        local = tuning_inputs.build(
            problem.local, seed, weight_seed=(seed + 1) * 1000 + rank
        )
        shared = None
        if problem.add_shared:
            generator = torch.Generator(device="cuda").manual_seed(
                seed * 1000 + 999 - rank
            )
            shared = torch.randn(
                (problem.output_rows, s.model_dim),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
        total = local.reference.float()
        shared_f32 = None if shared is None else shared.float()
        if problem.comm == "ar" and shared_f32 is not None:
            total += shared_f32
        rows = None
        if problem.comm == "rs":
            rows = torch.empty(
                (problem.output_rows, s.model_dim), dtype=torch.float32, device="cuda"
            )
    except Exception as error:
        agree(_reason(error), tp)
        raise
    reason = agree(None, tp)
    if reason is not None:
        raise RuntimeError(f"the inputs failed on another rank: {reason}")
    if problem.comm == "ar":
        dist.all_reduce(total, group=tp.device_group)
    else:
        dist.reduce_scatter_tensor(rows, total, group=tp.device_group)
        total = rows if shared_f32 is None else rows.add_(shared_f32)
    reference = total.to(local.reference.dtype)
    del local.reference
    return CommInputs(local.call, shared, reference)


class RunnerSets:
    """Runner sets of one config each (`_LazyRunners`), the last `limit` kept."""

    def __init__(self, limit=MAX_RUNNER_SETS):
        self.limit = limit
        self.sets = collections.OrderedDict()

    def get(self, problem, spec):
        """`(runner set, segment)` of `spec` (collective when created)."""
        from aiter.ops.flydsl.comm_fused_moe_host import _LazyRunners

        key = (problem.shape_key(), problem.local.shape.m, spec)
        if key in self.sets:
            self.sets.move_to_end(key)
            return self.sets[key]
        while len(self.sets) >= self.limit:
            self._release(self.sets.popitem(last=False)[1][0])
        config = runner_config(problem, spec)
        runners = _LazyRunners(_tp(), problem.shape_key(), {config.m: config})
        self.sets[key] = (runners, stage2_rs.CommFusedStage2.adopt(runners))
        return self.sets[key]

    def _release(self, runners):
        torch.cuda.synchronize()
        _tp().barrier()
        communicator = getattr(runners, "communicator", None)
        if communicator is not None and hasattr(communicator, "destroy"):
            communicator.destroy()

    def clear(self):
        while self.sets:
            self._release(self.sets.popitem(last=False)[1][0])


_runner_sets = RunnerSets()
_cached = {}
_current = {}  # "problem": the problem of the runner sets kept


def _inputs(problem, seed):
    key = (problem, seed)
    if _current.get("problem") != problem:
        _runner_sets.clear()
        _current["problem"] = problem
    if key not in _cached:
        _cached.clear()
        torch.cuda.empty_cache()
        _cached[key] = build_inputs(problem, seed)
    return _cached[key]


def ordinary_runner(call, resolution, shared, comm):
    """The local plan on `call`, then the TP collective (the shared partial added)."""
    run_local = measure.plan_runner(call, resolution)
    tp = _tp()

    def run():
        out = run_local()
        if comm == "ar":
            if shared is not None:
                out.add_(shared)
            return tp.all_reduce(out, ca_fp8_quant=False)
        out = tp.reduce_scatter_tensor(out)
        if shared is not None:
            out.add_(shared)
        return out

    return run


def fused_runner(call, resolution, runners, segment, shared):
    """`CommFusedMoeRuntime.run` of `call` with the local plan `resolution`: its stage 2
    replaced by the runner of `runners` (one bucket)."""
    _, q_dtype_a = tuning_inputs.prologued(call)
    (config,) = runners.configs.values()
    choice = SegmentChoice(segment, Params(runner=stage2_rs.runner_label(config)))
    fused_resolution = replace(resolution, plan=with_stage2(resolution.plan, choice))

    def run():
        padded, fused = stage2_rs.prepare_call(runners, call, shared_partial=shared)
        op = api.op_call(padded)
        stage2_rs.wait_for_reuse(fused)
        op["comm_fused"] = fused
        out = execute(prologue(op, quant_remap), fused_resolution, q_dtype_a)
        return stage2_rs.finish(fused, out)

    return run


def _copies(inputs, tp):
    """`(call, shared)` rotating copies of `inputs`, as many on every rank (sized by
    the least free memory of any)."""
    free = torch.tensor(
        [torch.cuda.mem_get_info()[0]], dtype=torch.int64, device="cuda"
    )
    dist.all_reduce(free, op=dist.ReduceOp.MIN, group=tp.device_group)
    calls = measure._calls(inputs, int(free.item()))
    shared = [inputs.shared] + [
        None if inputs.shared is None else inputs.shared.clone() for _ in calls[1:]
    ]
    return list(zip(calls, shared))


def _samples(fns, mode, iters, warmup, check, tp):
    """`(slowest rank's samples, reason)`, as `measure._samples` but across ranks."""
    n = len(fns)
    for i in range(max(warmup, n)):
        fns[i % n]()
    torch.cuda.synchronize()
    tp.barrier()
    outputs, local = {}, []
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with tp.graph_capture() as capture, torch.cuda.graph(
            graph, stream=capture.stream
        ):
            outputs = {i: fn() for i, fn in enumerate(fns)}
        graph.replay()
        torch.cuda.synchronize()
        tp.barrier()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iters)
        ]
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        local = [start.elapsed_time(end) * 1000 / n for start, end in events]
        del graph
    else:
        flag = torch.zeros(1, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(iters):
            torch.cuda.synchronize()
            dist.all_reduce(flag, group=tp.device_group)
            start.record()
            outputs[i % n] = fns[i % n]()
            end.record()
            end.synchronize()
            local.append(start.elapsed_time(end) * 1000)
    reason = None
    if check is not None:
        for i, out in sorted(outputs.items()):
            reason = check(out)
            if reason is not None:
                reason = f"copy {i}, {mode}: {reason}"
                break
    del outputs
    return local, reason


def _failure(error) -> str:
    return f"{type(error).__name__}: {str(error)[:500]}"


def _reason(error) -> str:
    if isinstance(error, tuning_inputs.Unsupported):
        return f"unsupported: {error}"
    return _failure(error)


class _Plan:
    """A plan's per-copy callables (built on every rank alike)."""

    def __init__(self, problem, resolution, runner, copies):
        if resolution.plan is None:
            raise ValueError(f"declined: {resolution.declined}")
        if runner is None:
            self.fns = [
                ordinary_runner(call, resolution, shared, problem.comm)
                for call, shared in copies
            ]
        else:
            runners, segment = _runner_sets.get(problem, runner)
            self.fns = [
                fused_runner(call, resolution, runners, segment, shared)
                for call, shared in copies
            ]


def comm_errors(out, ref) -> tuple:
    """`(max abs, relative L2)` of `out` against `ref`: the legacy comm-fused tuner's
    measures."""
    diff = out.float() - ref.float()
    return (
        diff.abs().max().item(),
        (diff.norm() / ref.float().norm().clamp_min(1.0e-12)).item(),
    )


def _check(inputs):
    def check(out):
        if tuple(out.shape) != tuple(inputs.reference.shape):
            return f"output shape {tuple(out.shape)} != {tuple(inputs.reference.shape)}"
        reason = measure.check_output(out, inputs.reference)
        if reason is not None:
            max_abs, rel_l2 = comm_errors(out, inputs.reference)
            reason += f" (max abs {max_abs:.4g}, rel L2 {rel_l2:.4g})"
        return reason

    return check


class _Diverged(Exception):
    """The ranks' runner sets differ: the next set created or released on some ranks
    only would leave the others in its collectives."""


def _runner_sets_agree(tp) -> bool:
    keys = [None] * tp.world_size
    mine = [repr(key) for key in _runner_sets.sets]
    dist.all_gather_object(keys, mine, group=tp.cpu_group)
    return all(k == keys[0] for k in keys)


def _prepared(problem, inputs, plans, tp):
    """`[_Plan or reason]` for `plans` (`(resolution, runner)`), each built and run
    once on the first copy, its output checked (agreed across ranks)."""
    reason = None
    try:
        copies = _copies(inputs, tp)
        check = _check(inputs)
    except Exception as error:  # noqa: BLE001 - every plan fails on this rank
        reason = _failure(error)
    reason = agree(reason, tp)
    if reason is not None:
        return [reason] * len(plans), None
    prepared = []
    for resolution, runner in plans:
        reason = None
        try:
            plan = _Plan(problem, resolution, runner, copies)
        except Exception as error:  # noqa: BLE001 - this plan fails on this rank
            reason = _failure(error)
        reason = agree(reason, tp)
        if reason is not None and runner is not None and not _runner_sets_agree(tp):
            raise _Diverged(f"{reason} (the ranks then kept other runner sets)")
        if reason is None:
            try:
                out = plan.fns[0]()
                torch.cuda.synchronize()
                reason = check(out)
            except Exception as error:  # noqa: BLE001 - this plan fails
                reason = _failure(error)
            reason = agree(reason, tp)
        prepared.append(plan if reason is None else reason)
    return prepared, check


def _timed(plan, mode, iters, warmup, check, tp):
    """`(samples, reason)` of one timing pass of `plan` (agreed across ranks)."""
    reason, local = None, []
    try:
        local, reason = _samples(plan.fns, mode, iters, warmup, check, tp)
    except Exception as error:  # noqa: BLE001 - this plan fails
        reason = _failure(error)
    reason = agree(reason, tp)
    if reason is not None:
        return None, reason
    return slowest(local, tp), None


def measure_plans(problem, inputs, plans, mode, rounds=1, iters=50, warmup=3):
    """A `measure.Result` per plan of `plans` (`(resolution, runner)`): timed end to
    end, in turn over `rounds` rounds (each in the reverse order of the one before),
    every round's samples together."""
    tp = _tp()
    specs = {runner for _, runner in plans if runner is not None}
    if len(specs) > _runner_sets.limit:
        reason = f"{len(specs)} runner configs, more than the {_runner_sets.limit} kept"
        return [measure.Result(reason)] * len(plans)
    prepared, check = _prepared(problem, inputs, plans, tp)
    results = [measure.Result(p) if isinstance(p, str) else None for p in prepared]
    samples = {i: [] for i, r in enumerate(results) if r is None}
    order = list(samples)
    for round_ in range(rounds):
        for i in order if round_ % 2 == 0 else order[::-1]:
            if results[i] is not None:
                continue
            taken, reason = _timed(
                prepared[i], mode, iters, warmup if round_ == 0 else 0, check, tp
            )
            if reason is not None:
                results[i] = measure.Result(reason)
                continue
            samples[i] += taken
    for i in order:
        if results[i] is None:
            results[i] = measure.Result(e2e=measure.stats(samples[i], mode))
    return results


def run(task):
    """A worker's `CommTask` (a `measure.Result`) or `CommRivals` (one per plan)."""
    rivals = isinstance(task, CommRivals)
    plans_of = (
        task.plans if rivals else ((task.structure, task.candidates, task.runner),)
    )
    reason = None
    try:
        inputs = _inputs(task.problem, task.seed)
        plans = [
            (plan_of(task.problem.local, task.path, structure, candidates), runner)
            for structure, candidates, runner in plans_of
        ]
    except Exception as error:  # noqa: BLE001 - reported as the task's failure
        reason = _reason(error)
    reason = agree(reason)
    if reason is not None:
        _cached.clear()
        results = [measure.Result(reason)] * len(plans_of)
    else:
        mode = task.mode or measure.deploy_mode(task.problem.local)
        try:
            results = measure_plans(
                task.problem,
                inputs,
                plans,
                mode,
                task.rounds if rivals else 1,
                task.iters,
                task.warmup,
            )
        except _Diverged as error:
            results = [measure.Result(str(error))] * len(plans_of)
            raise Restart(results if rivals else results[0]) from error
    if not all(r.ok for r in results) and not _device_usable():
        results = [
            r if r.ok else measure.Result(f"{r.reason} (device unusable)")
            for r in results
        ]
        raise Restart(results if rivals else results[0])
    return results if rivals else results[0]
