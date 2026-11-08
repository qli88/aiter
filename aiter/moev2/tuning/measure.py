# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Measuring a plan on real inputs (plan §4.2): end to end, or one segment in context,
and checking its output against the torch reference.

A plan runs through `migration.local.execute`, the code the local paths run, on the
call of `tuning.inputs`. Timing: CUDA events around each run, after warm-up; eager runs
start on an idle device (a synchronize before each), so their time includes the host
launch overhead, as deployed; graph mode captures one run per rotating copy in a graph
and times its replays. The rotating copies of the call (`rotations`) keep each run from
reading the previous one's data from the caches. A segment in context: the plan runs
once per copy with `hooks.set_kernel_bench_callable`, which collects each stage launch
with its real inputs (sorted rows, quantized activations); those launches are timed
alone. Failures in-process (an exception, a capture error, a wrong output) come back as
a reason; a crash or a hang is the worker pool's (`tuning.pool`).
"""

import math
import statistics
from dataclasses import dataclass, field

import torch

from .. import hooks
from ..execute.executor import prologue
from ..migration.local import execute
from ..policy import quant_remap
from .inputs import entry, prologued

# Legacy's end-to-end acceptance: the error ratio (elements outside atol/rtol 1e-2)
# at most `ERR_RATIO`, or the cosine distance at most `COS_TOL`.
ERR_RATIO = 0.1
COS_TOL = 0.01
ATOL = RTOL = 1e-2
# Graph replay up to this token count (vLLM's default largest captured batch).
GRAPH_MAX_M = 512
# The rotating copies together exceed this (above L2 and the 256 MiB MALL).
ROTATE_BYTES = 512 << 20
MAX_COPIES = 16
# Tensor attributes the dispatch key reads, kept on copies.
_TENSOR_TAGS = ("is_shuffled", "gate_up_interleave", "aiter_original_k")
_WEIGHTS = ("w1", "w2", "w1_scale", "w2_scale")


def deploy_mode(problem) -> str:
    return "graph" if problem.shape.m <= GRAPH_MAX_M else "eager"


@dataclass(frozen=True)
class Timing:
    median_us: float
    cov: float  # standard deviation / mean of the samples
    samples: int
    mode: str  # "graph" or "eager"


def stats(samples_us, mode) -> Timing:
    mean = statistics.fmean(samples_us)
    sd = statistics.pstdev(samples_us) if len(samples_us) > 1 else 0.0
    return Timing(
        statistics.median(samples_us), sd / mean if mean else 0.0, len(samples_us), mode
    )


def errors(out, ref) -> tuple:
    """`(error ratio, cosine distance)` of `out` against `ref`, as legacy computes
    them."""
    close = torch.isclose(out, ref, rtol=RTOL, atol=ATOL)
    ratio = 1.0 - close.sum().item() / close.numel()
    x, y = out.float(), ref.float()
    den = (x * x + y * y).sum().item()
    cos = 1.0 - 2.0 * (x * y).sum().item() / den if den else 0.0
    return ratio, cos


def check_output(out, ref, err_ratio=ERR_RATIO, cos_tol=COS_TOL) -> str | None:
    """None if `out` passes against `ref`, else the reason."""
    if not torch.isfinite(out).all():
        return "non-finite output"
    if not torch.isfinite(ref).all():
        return "non-finite reference"
    if out.count_nonzero() == 0 and ref.count_nonzero() > 0:
        return "all-zero output"
    ratio, cos = errors(out, ref)
    if ratio <= err_ratio or cos <= cos_tol:
        return None
    return f"mismatch: error ratio {ratio:.4g} > {err_ratio}, cosine distance {cos:.4g} > {cos_tol}"


def _clone(tensor):
    copy = tensor.clone()
    for tag in _TENSOR_TAGS:
        if hasattr(tensor, tag):
            setattr(copy, tag, getattr(tensor, tag))
    return copy


def _nbytes(call, names=None):
    return sum(
        v.nbytes
        for k, v in call.items()
        if isinstance(v, torch.Tensor) and (names is None or k in names)
    )


def copy_count(call, free_bytes, rotate_bytes=ROTATE_BYTES, max_copies=MAX_COPIES):
    """`(copies, with_weights)`: how many copies of `call` rotate, and whether the
    weights are copied (only when every copy fits in half of `free_bytes`)."""
    whole = max(_nbytes(call), 1)
    copies = max(1, min(max_copies, math.ceil(rotate_bytes / whole)))
    if (copies - 1) * whole <= free_bytes // 2:
        return copies, True
    rest = max(whole - _nbytes(call, _WEIGHTS), 1)
    return max(1, min(max_copies, math.ceil(rotate_bytes / rest))), False


def rotations(call, copies, with_weights):
    """`copies` calls: `call`, then copies of its tensors (the weights shared unless
    `with_weights`)."""
    out = [call]
    for _ in range(copies - 1):
        out.append(
            {
                k: (
                    _clone(v)
                    if isinstance(v, torch.Tensor)
                    and (with_weights or k not in _WEIGHTS)
                    else v
                )
                for k, v in call.items()
            }
        )
    return out


def plan_runner(call, resolution):
    """A zero-argument callable running `resolution` on `call` as the local paths do
    (the prologue included)."""
    op = entry(call)[0]
    _, q_dtype_a = prologued(call)

    def run():
        return execute(prologue(op, quant_remap), resolution, q_dtype_a)

    return run


def time_calls(fns, mode, iters=50, warmup=3, check=None):
    """`(Timing, reason)`: time the zero-argument callables `fns` (one per rotating
    copy) run in turn; `reason` is the first `check(output)` failing on what a copy
    returned in its timed runs (the graph's outputs after its replays), else None."""
    for i in range(max(warmup, len(fns))):
        fns[i % len(fns)]()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    outputs = {}
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = {i: fn() for i, fn in enumerate(fns)}
        graph.replay()
        torch.cuda.synchronize()
        for _ in range(iters):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / len(fns))
    else:
        for i in range(iters):
            fn = fns[i % len(fns)]
            torch.cuda.synchronize()
            start.record()
            outputs[i % len(fns)] = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
    reason = None
    if check is not None:
        for i, out in sorted(outputs.items()):
            reason = check(out)
            if reason is not None:
                reason = f"copy {i}, {mode}: {reason}"
                break
    del outputs
    return stats(samples, mode), reason


@dataclass
class Result:
    reason: str | None = None  # None: measured and verified
    e2e: Timing | None = None
    stages: dict = field(default_factory=dict)  # stage name -> Timing

    @property
    def ok(self) -> bool:
        return self.reason is None


def measure(
    inputs,
    resolution,
    mode,
    iters=50,
    warmup=3,
    segments=False,
    err_ratio=ERR_RATIO,
    free_bytes=None,
) -> Result:
    """Verify `resolution` (a plan) on `inputs` (`tuning.inputs.build`), then time it end
    to end in `mode`, verifying each copy's output from the timed runs again; with
    `segments`, also each stage launch alone, in context."""
    if resolution.plan is None:
        return Result(f"declined: {resolution.declined}")
    try:
        out = plan_runner(inputs.call, resolution)()
        torch.cuda.synchronize()
        reason = check_output(out, inputs.reference, err_ratio)
        if reason is not None:
            return Result(reason)
        if free_bytes is None:
            free_bytes = torch.cuda.mem_get_info()[0]
        copies, with_weights = copy_count(inputs.call, free_bytes)
        calls = rotations(inputs.call, copies, with_weights)
        runners = [plan_runner(c, resolution) for c in calls]
        e2e, reason = time_calls(
            runners,
            mode,
            iters,
            warmup,
            lambda out: check_output(out, inputs.reference, err_ratio),
        )
        if reason is not None:
            return Result(reason)
        result = Result(e2e=e2e)
        if segments:
            launches = {}
            with hooks.set_kernel_bench_callable() as recorded:
                for run in runners:
                    run()
            for stage, launch in recorded:
                launches.setdefault(stage, []).append(launch)
            result.stages = {
                stage: time_calls(fns, mode, iters, warmup)[0]
                for stage, fns in launches.items()
            }
        return result
    except Exception as error:  # noqa: BLE001 - a failing candidate, with its reason
        return Result(f"{type(error).__name__}: {str(error)[:500]}")
