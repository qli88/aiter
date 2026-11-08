# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A tuning task, as a worker of `tuning.pool` runs it: `run(Task)` -> `measure.Result`.

A task names one problem, the registered path serving it, and the plan to measure: a
structure (its segment names, one of `resolver.feasible_plans`) and the candidate of
each tunable segment (`resolver.build`), or neither for the plan dispatch resolves
today. The worker builds the inputs once per problem and keeps the last ones. A failure
that leaves the device unusable restarts the worker (`pool.Restart`).
"""

from dataclasses import dataclass

import torch

from .. import paths
from ..dispatch import resolver
from . import inputs as tuning_inputs
from . import measure
from .pool import Restart


@dataclass(frozen=True)
class Task:
    problem: object  # MoeProblem
    path: str  # `paths.registered()` name
    structure: tuple | None = None  # segment names; None: what dispatch resolves
    candidates: tuple | None = None  # ((segment name, Params), ...) with `structure`
    mode: str | None = None  # "graph" or "eager"; None: `measure.deploy_mode`
    iters: int = 50
    warmup: int = 3
    segments: bool = False
    seed: int = 0


_cached = {}


def _inputs(problem, seed):
    key = (problem, seed)
    if key not in _cached:
        _cached.clear()
        torch.cuda.empty_cache()
        _cached[key] = tuning_inputs.build(problem, seed)
    return _cached[key]


def resolution(task):
    """The plan `task` measures (a `Resolution`, possibly declined)."""
    segments = paths.registered()[task.path].segments
    if task.structure is None:
        return resolver.resolve(task.problem, segments)
    plans, _ = resolver.feasible_plans(task.problem, segments)
    names = tuple(task.structure)
    structure = next((s for s in plans if tuple(x.name for x in s) == names), None)
    if structure is None:
        return resolver.Resolution(None, declined=f"no feasible structure {names}")
    return resolver.build(task.problem, structure, dict(task.candidates))


def _device_usable() -> bool:
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda").add_(1).item()
        return True
    except Exception:  # noqa: BLE001 - any error here means the context is gone
        return False


def run(task) -> measure.Result:
    try:
        inputs = _inputs(task.problem, task.seed)
        plan = resolution(task)
    except tuning_inputs.Unsupported as error:
        return measure.Result(f"unsupported: {error}")
    except Exception as error:  # noqa: BLE001 - reported as the task's failure
        _cached.clear()
        result = measure.Result(f"{type(error).__name__}: {str(error)[:500]}")
    else:
        result = measure.measure(
            inputs,
            plan,
            task.mode or measure.deploy_mode(task.problem),
            task.iters,
            task.warmup,
            task.segments,
        )
    if not result.ok and not _device_usable():
        raise Restart(measure.Result(f"{result.reason} (device unusable)"))
    return result
