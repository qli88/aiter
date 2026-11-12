# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A tuning task, as a worker of `tuning.pool` runs it: `run(Task)` -> `measure.Result`,
`run(Rivals)` -> a `measure.Result` per plan.

A task names one problem, the registered path serving it, and the plan to measure: a
structure (its segment names, one of `resolver.feasible_plans`) and the candidate of
each tunable segment (`resolver.build`) or none (its default params,
`resolver.default_plan`), or neither for the plan dispatch resolves today. The worker
builds the inputs once per problem and keeps the last ones. A failure that leaves the
device unusable restarts the worker (`pool.Restart`).
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
    # ((segment name, Params), ...) with `structure`; None: its default params
    candidates: tuple | None = None
    mode: str | None = None  # "graph" or "eager"; None: `measure.deploy_mode`
    iters: int = 50
    warmup: int = 3
    segments: bool | tuple = False  # stage launches timed alone: all, or these names
    seed: int = 0
    e2e: bool = True  # time the whole plan


@dataclass(frozen=True)
class Rivals:
    """Plans of one problem timed in turn (`measure.interleaved`)."""

    problem: object
    path: str
    plans: tuple  # ((structure, candidates), ...) as in `Task`
    mode: str | None = None
    rounds: int = 4
    iters: int = 50
    warmup: int = 3
    seed: int = 0


_cached = {}


def _inputs(problem, seed):
    key = (problem, seed)
    if key not in _cached:
        _cached.clear()
        torch.cuda.empty_cache()
        _cached[key] = tuning_inputs.build(problem, seed)
    return _cached[key]


def plan_of(problem, path, structure=None, candidates=None):
    """The plan to measure (a `Resolution`, possibly declined): `structure` (segment
    names) built with `candidates` (at its default params when None), or what dispatch
    resolves when None."""
    segments = paths.registered()[path].segments
    if structure is None:
        return resolver.resolve(problem, segments)
    if candidates is None:
        return resolver.default_plan(problem, segments, structure)
    plans, _ = resolver.feasible_plans(problem, segments)
    names = tuple(structure)
    found = next((s for s in plans if tuple(x.name for x in s) == names), None)
    if found is None:
        return resolver.Resolution(None, declined=f"no feasible structure {names}")
    return resolver.build(problem, found, dict(candidates))


def resolution(task):
    """The plan `task` measures."""
    return plan_of(task.problem, task.path, task.structure, task.candidates)


def _device_usable() -> bool:
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda").add_(1).item()
        return True
    except Exception:  # noqa: BLE001 - any error here means the context is gone
        return False


def run(task):
    try:
        inputs = _inputs(task.problem, task.seed)
        if isinstance(task, Rivals):
            plans = [plan_of(task.problem, task.path, *p) for p in task.plans]
        else:
            plan = resolution(task)
    except tuning_inputs.Unsupported as error:
        result = measure.Result(f"unsupported: {error}")
        return [result] * len(task.plans) if isinstance(task, Rivals) else result
    except Exception as error:  # noqa: BLE001 - reported as the task's failure
        _cached.clear()
        failed = measure.Result(f"{type(error).__name__}: {str(error)[:500]}")
        results = [failed] * len(task.plans) if isinstance(task, Rivals) else [failed]
    else:
        mode = task.mode or measure.deploy_mode(task.problem)
        if isinstance(task, Rivals):
            results = measure.interleaved(
                inputs, plans, mode, task.rounds, task.iters, task.warmup
            )
        else:
            results = [
                measure.measure(
                    inputs,
                    plan,
                    mode,
                    task.iters,
                    task.warmup,
                    task.segments,
                    e2e=task.e2e,
                )
            ]
    if not all(r.ok for r in results) and not _device_usable():
        results = [
            r if r.ok else measure.Result(f"{r.reason} (device unusable)")
            for r in results
        ]
        raise Restart(results if isinstance(task, Rivals) else results[0])
    return results if isinstance(task, Rivals) else results[0]
