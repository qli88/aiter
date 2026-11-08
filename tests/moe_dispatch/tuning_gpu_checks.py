# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU check of the tuner's measuring (aiter.moev2.tuning, step 5.2b), on this GPU's arch.

For a sample of the dispatch keys each single-GPU path serves (the shipped tuned tables
plus the snapshot grid), the tuner builds its inputs (`tuning.inputs.build`: the call
must key on the problem), then measures the plan dispatch resolves (`tuning.task.run`
in a `tuning.pool` worker): its output must pass the acceptance rule against the torch
reference. A key whose format has no builder, a call keying elsewhere, or a failing
shipped plan is a problem. Listed apart: keys with no v2 plan (declined), table rows at
a token count no call looks up (not a dispatch bucket), and plans the shared dispatch
code refuses at the key (NotImplementedError, as legacy raises it).

    HIP_VISIBLE_DEVICES=0 python tests/moe_dispatch/tuning_gpu_checks.py
    python tests/moe_dispatch/tuning_gpu_checks.py --per-path 6 --paths a4w4 fhmoe

The keys are collected in a subprocess: the arch simulation that lists them hides the
GPU.
"""

import argparse
import itertools
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
_SKIPPED = ("declined: ", "unsupported: M=")


def collect(arch, out):
    """Pickle `{path: [(args, kwargs, fhmoe, label)]}` for the keys each path takes."""
    from collections import defaultdict

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup(frozen=False)
    from aiter.moev2 import api
    from moe_dispatch import arch_sim
    from moe_dispatch.coverage_checks import _ROW_SECTIONS
    from moe_dispatch.differential import captured_logs
    from moe_dispatch.ranking_checks import ranked_paths

    keys = defaultdict(list)
    with arch_sim.simulated_arch(arch) as (gfx, cu_num), captured_logs():
        for key in snapshot.dispatch_keys(gfx, cu_num):
            fhmoe = _ROW_SECTIONS.get(key.section)
            if key.variant != "base" or fhmoe is None:
                continue
            kwargs = key.call_kwargs()
            problem = api.problem_from_key(key.args, kwargs, fhmoe=fhmoe)
            for path in ranked_paths():
                if path.takes(problem):
                    keys[path.name].append((key.args, kwargs, fhmoe, key.label()))
    pathlib.Path(out).write_bytes(pickle.dumps(dict(keys)))


def sample(problems, per_path, largest=False):
    """Up to `per_path` problems, one per (graph or eager, activation, doweight) with
    the smallest M (`largest`: the largest), graph and eager in turn."""
    from aiter.moev2.tuning.measure import deploy_mode

    first = {}
    for p in sorted(problems, key=lambda p: p.shape.m, reverse=largest):
        sig = (
            deploy_mode(p) == "eager",
            p.activation.kind,
            "doweight_stage1" in p.features,
        )
        first.setdefault(sig, p)
    graph = [p for sig, p in first.items() if not sig[0]]
    eager = [p for sig, p in first.items() if sig[0]]
    turns = itertools.zip_longest(graph, eager)
    return [p for pair in turns for p in pair if p is not None][:per_path]


def check_timed_outputs():
    """Problems of `time_calls` not reporting a copy whose timed output is wrong."""
    import torch

    from aiter.moev2.tuning.measure import time_calls

    fns = [lambda: torch.ones(4, device="cuda"), lambda: torch.zeros(4, device="cuda")]
    problems = []
    for mode in ("graph", "eager"):
        _, reason = time_calls(
            fns, mode, 4, 2, lambda out: None if bool(out.all()) else "zeros"
        )
        if reason != f"copy 1, {mode}: zeros":
            problems.append(f"time_calls {mode} reported {reason!r}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-path", type=int, default=3)
    parser.add_argument("--paths", nargs="*", help="only these paths")
    parser.add_argument("--largest", action="store_true", help="the largest M per kind")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--collect", help=argparse.SUPPRESS)
    parser.add_argument("--arch", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.collect:
        collect(args.arch, args.collect)
        return 0

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.moev2 import api
    from aiter.moev2.tuning import task as tuning_task
    from aiter.moev2.tuning.pool import Pool

    arch, cu_num = get_gfx(), get_cu_num()
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "keys.pkl"
        subprocess.run(
            [sys.executable, __file__, "--collect", str(out), "--arch", arch],
            check=True,
        )
        keys = pickle.loads(out.read_bytes())
    tasks, problems = [], check_timed_outputs()
    for problem in problems:
        print(f"  PROBLEM {problem}", flush=True)
    for name in sorted(keys):
        if args.paths and name not in args.paths:
            continue
        served = [
            api.problem_from_key(a, kw, arch, cu_num, fhmoe)
            for a, kw, fhmoe, _ in keys[name]
        ]
        for p in sample(served, args.per_path, args.largest):
            tasks.append(tuning_task.Task(p, name, iters=args.iters, warmup=2))
    if not tasks:
        print(f"no keys served on {arch}")
        return 0
    print(f"{arch}: {len(tasks)} keys over {len({t.path for t in tasks})} paths")
    start = time.time()
    with Pool(tuning_task.run, timeout=args.timeout) as pool:
        for i, result in pool.map(tasks):
            t = tasks[i]
            shape = t.problem.shape
            label = f"{t.path} M={shape.m} H={shape.model_dim} I={shape.inter_dim} "
            label += f"E={shape.experts} k={shape.topk} {t.problem.activation.kind}"
            if result.ok:
                e2e = result.e2e
                print(f"  ok {label}: {e2e.median_us:.1f} us {e2e.mode}", flush=True)
            elif result.reason.startswith(_SKIPPED):
                print(f"  skipped {label}: {result.reason}", flush=True)
            elif result.reason.startswith("NotImplementedError: "):
                print(f"  legacy raises {label}: {result.reason}", flush=True)
            else:
                problems.append(f"{label}: {result.reason}")
                print(f"  PROBLEM {problems[-1]}", flush=True)
    print(f"{len(problems)} problems ({time.time() - start:.0f} s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
