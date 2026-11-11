# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 differential of the `comm_fused` path against `CommFusedMoeRuntime.run`.

    python tests/moe_dispatch/comm_fused_differential.py   # runs torchrun on 8 GPUs
    _MOEV2_CF_CASES="ar M=1|rs M=64 direct" python tests/moe_dispatch/...  # some cases

Each case runs the DeepSeek-V4 TP8 shape of op_tests/multigpu_tests/test_comm_fused_moe.py
(its fixtures and runner sets) with AITER_MOEV2=off twice and AITER_MOEV2=comm_fused
once, on every rank: v2 must run the call (or decline it where legacy's a8w4 path
would not run either), launch the same kernels on bitwise-equal inputs (the stage 2
compared as the ordinary launch the runner replaces), use the runner of the same bucket,
return the same output up to the noise between the two legacy runs, and log the same
dispatch lines. The engine cases run `CommFusedTemplate` through `MoeEngine` against
the legacy runtime. Errors must match.

The shipped comm-fused rows are tuned on MI355X; `get_gpu_model` is patched to
"mi355x" in these processes so an MI350X runs them too (same gfx950 kernels).
"""

import dataclasses
import functools
import os
import pathlib
import subprocess
import sys
from unittest import mock

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
WORLD = 8

# name: (comm, tokens, variant, options). Variants: how the shared partial arrives
# ("shared", "for_rows", "producer"), "stream" (a separate stage 2 stream), "graph"
# (captured, then replayed), "engine" (`CommFusedTemplate`), "tuned" (the engine with
# a plan file giving the call's bucket a runner other than the shipped one, the plan
# dispatch resolves; it must match legacy within the same noise). Options: "route",
# "sizes" (ragged reduce-scatter), "bound" (AITER_BF16_FP8_MOE_BOUND; default 0, as
# the production test runs).
CASES = {
    "ar M=1": ("ar", 1, "shared", {}),
    "ar M=3 padded to 4, shared for rows": ("ar", 3, "for_rows", {}),
    "ar M=3, producer and join": ("ar", 3, "producer", {}),
    "ar M=13 padded to 16": ("ar", 13, "shared", {}),
    "ar M=16 uniform": ("ar", 16, "shared", {"route": "uniform"}),
    "ar M=3 separate stream": ("ar", 3, "stream", {}),
    "ar M=3 graph": ("ar", 3, "graph", {}),
    "ar M=40 (no runner)": ("ar", 40, "shared", {}),
    "ar M=3000 window, padded": ("ar", 3000, "shared", {}),
    "ar M=4096 window": ("ar", 4096, "shared", {}),
    "ar M=3, default bf16 bound (a16w4)": ("ar", 3, "shared", {"bound": None}),
    "rs M=64 direct": ("rs", 64, "shared", {}),
    "rs M=64 direct, separate stream": ("rs", 64, "stream", {}),
    "rs M=64 direct graph": ("rs", 64, "graph", {}),
    "rs M=1024 direct, shared for rows": ("rs", 1024, "for_rows", {}),
    "rs M=40 (exact-M runner)": ("rs", 40, "shared", {}),
    "rs M=4096 window": ("rs", 4096, "shared", {}),
    "rs M=3984 window, ragged": (
        "rs",
        3984,
        "shared",
        {"sizes": (512,) * 7 + (400,)},
    ),
    "engine ar M=3": ("ar", 3, "engine", {}),
    "engine rs M=64": ("rs", 64, "engine", {}),
    "engine rs M=3984, ragged": ("rs", 3984, "engine", {"sizes": (512,) * 7 + (400,)}),
    "engine rs M=64, plan file runner": ("rs", 64, "tuned", {}),
}


def _ordinary_stage2_launches(bench):
    """The recorded launches with the stage 2 as the ordinary launch the runner replaces
    (its output buffer, which the runner does not write, by dtype and shape), and the
    `CommFusedCall` of v2's stage 2 (None for legacy's)."""
    import torch

    from aiter.moev2.segments import comm_fused as stage2_rs

    launches, fused = [], None
    for name, launch in bench:
        if name == "stage2":
            kw = launch.keywords
            if launch.func is stage2_rs.launch:
                fused = launch.args[0]
            args = list(kw["stage2_args"])
            out = args[6]
            if isinstance(out, torch.Tensor):
                args[6] = f"{out.dtype}{tuple(out.shape)}"
            launch = functools.partial(
                kw["ordinary_stage2"], *args, **kw["stage2_kwargs"]
            )
        launches.append((name, launch))
    return launches, fused


class _Session:
    def __init__(self, t, rank, device, group):
        from aiter.dist.parallel_state import get_tp_group
        from aiter.ops.flydsl.comm_fused_moe_host import (
            create_flydsl_comm_fused_runners,
        )

        self.t, self.rank, self.device, self.group = t, rank, device, group
        self.tp = get_tp_group()
        self.runners = {
            comm: create_flydsl_comm_fused_runners(
                tp_group=self.tp,
                model_dim=t.MODEL_DIM,
                inter_dim=t.INTER_DIM,
                experts=t.EXPERTS,
                topk=t.TOPK,
                comm=comm,
                add_shared=True,
            )
            for comm in ("ar", "rs")
        }
        self.weights = t._make_stage2_weights(rank, device)
        # One W1 for every case: the 4096-token skew routes reach every expert, and
        # building W1 (one expert at a time) costs minutes per case.
        self.w1_case = t._make_full_moe_case(4096, "skew", rank, device)
        self.stream = None

    def options(self):
        from aiter import ActivationType, QuantType

        return {
            "activation": ActivationType.Silu,
            "quant_type": QuantType.per_1x32,
            "doweight_stage1": False,
            "hidden_pad": 0,
            "intermediate_pad": 0,
            "gate_mode": "interleave",
        }


def _case_inputs(session, comm, tokens, variant, opts):
    import torch

    t = session.t
    generator = torch.Generator(device=session.device).manual_seed(
        20260903 + 1000 * tokens + session.rank
    )
    hidden_states = torch.randn(
        (tokens, t.MODEL_DIM),
        dtype=torch.bfloat16,
        device=session.device,
        generator=generator,
    ).mul_(t.MODEL_DIM**-0.25)
    topk_ids, topk_weights = t._make_routes(
        tokens, opts.get("route", "skew"), session.device
    )
    case = dataclasses.replace(
        session.w1_case,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    sizes = opts.get("sizes")
    if comm == "ar":
        rows = tokens
    elif sizes is not None:
        rows = sizes[session.rank]
    else:
        rows = tokens // WORLD
    shared = t._shared_partial(rows, session.rank, session.device)
    moe_args = {
        "hidden_states": case.hidden_states,
        "w1": case.w1,
        "w2": session.weights.kernel,
        "topk_weight": case.topk_weights,
        "topk_ids": case.topk_ids,
        "w1_scale": case.w1_scale,
        "w2_scale": session.weights.kernel_scale,
        **session.options(),
    }
    run_kw = {"shared_partial": shared, "reduce_scatter_sizes": sizes}
    if variant == "for_rows":
        run_kw.update(shared_partial=None, before_stage2_for_rows=lambda _: shared)
    elif variant == "producer":
        run_kw.update(
            shared_partial=None,
            before_stage2=lambda: shared,
            before_shared_add=lambda: None,
        )
    elif variant == "stream":
        if session.stream is None:
            session.stream = torch.cuda.Stream(device=session.device)
        run_kw.update(
            shared_partial=None,
            before_stage2_for_rows=lambda _: shared,
            stage2_stream=session.stream,
        )
    return case, moe_args, run_kw


def _run(session, mode, fn, graph=False):
    """`(output clone or error string, launches, log lines)` of `fn` under
    AITER_MOEV2=`mode`, dispatch caches cleared first. With `graph`, `fn` is warmed up,
    captured, and its output poisoned and replayed."""
    import torch

    import aiter.fused_moe as fm
    from moe_dispatch.differential import captured_logs, moev2_mode
    from moe_dispatch.gpu_differential import _clear_dispatch_caches

    bench = []
    with moev2_mode(mode):
        try:
            if graph:
                fn()
                torch.cuda.synchronize()
            session.t._barrier(session.group)
            _clear_dispatch_caches()
            fm.kernel_bench_callable = bench
            with captured_logs() as logs:
                if graph:
                    g = torch.cuda.CUDAGraph()
                    with session.tp.graph_capture() as capture, torch.cuda.graph(
                        g, stream=capture.stream
                    ):
                        out = fn()
                    out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    g.replay()
                else:
                    out = fn()
            torch.cuda.synchronize()
            out = out.clone()
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            out = f"{type(exc).__name__}: {exc}"
        finally:
            fm.kernel_bench_callable = None
    session.t._barrier(session.group)
    return out, bench, logs


TUNED_RUNNER = {"family": "direct", "grid_cap": 16}


def _write_plan_file(session, template, comm, bucket, path):
    """A plan file whose only row gives `bucket` the plan dispatch resolves for it,
    with `TUNED_RUNNER` (written by rank 0)."""
    import torch.distributed as dist

    error = None
    if session.rank == 0:
        try:
            _write_row(template, comm, bucket, path)
        except Exception as exc:  # noqa: BLE001 - every rank gives up on the case
            error = f"{type(exc).__name__}: {exc}"
    errors = [None] * WORLD
    dist.all_gather_object(errors, error, group=session.tp.cpu_group)
    if errors[0] is not None:
        raise RuntimeError(f"the plan file was not written: {errors[0]}")


def _write_row(template, comm, bucket, path):
    import csv

    from aiter.moev2 import api
    from aiter.moev2.dispatch import tuned_plans
    from aiter.moev2.migration import a8w4
    from aiter.moev2.segments.base import Params
    from aiter.moev2.tuning import cli, search
    from aiter.moev2.tuning.candidates import plan_candidates
    from aiter.moev2.tuning.measure import Timing
    from aiter.ops.flydsl.comm_fused_moe_host import get_gpu_model

    problem = template._local_problem(bucket)
    resolution = a8w4.LOCAL.resolve_call(api.op_call(template._meta_call(bucket)))
    candidates = plan_candidates(problem, resolution.plan)
    candidates[tuned_plans.RUNNER] = Params(
        TUNED_RUNNER, sort_block_m=int(resolution.metadata.block_m)
    )
    outcome = search.Outcome(
        tuple(c.segment.name for c in resolution.plan.segments),
        candidates,
        Timing(1.0, 0.0, 1, "graph"),
        rank=1,
    )
    keys = tuned_plans.comm_key_values(problem, WORLD, comm, True, get_gpu_model())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS)
        writer.writeheader()
        writer.writerow(cli._plan_row(keys, tuned_plans.COMM_FUSED_PATH, outcome))


def _engine_fn(session, comm, tokens, moe_args, run_kw, tuned=False):
    """`MoeEngine.forward` of a `CommFusedTemplate` engine over the TP group; with
    `tuned`, built while `AITER_MOEV2_TUNED_TABLE` names `_write_plan_file`'s file."""
    from aiter.moev2 import engine as eng
    from aiter.moev2 import envs
    from aiter.moev2.dispatch import tuned as tuned_cache
    from aiter.moev2.migration.comm_fused import CommFusedTemplate
    from aiter.moev2.problem import ParallelSpec
    from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime

    t = session.t
    weights = eng.MoeWeights(
        moe_args["w1"], moe_args["w2"], moe_args["w1_scale"], moe_args["w2_scale"]
    )
    side = {"shared_partial": run_kw["shared_partial"]}
    if run_kw["reduce_scatter_sizes"] is not None:
        side["reduce_scatter_sizes"] = run_kw["reduce_scatter_sizes"]
    template = CommFusedTemplate(
        weights,
        topk=t.TOPK,
        options=session.options(),
        side_inputs=frozenset(side),
        tp_size=WORLD,
        comm=comm,
    )
    parallel = ParallelSpec(
        mode="tp",
        world_size=WORLD,
        comm=comm,
        backend="comm_fused",
        max_tokens_per_rank=tokens,
        experts_per_rank=t.EXPERTS,
    )
    sizes = run_kw["reduce_scatter_sizes"]
    # The count the runtime buckets by (ragged: max(sizes) * ranks).
    rows = tokens if sizes is None else max(sizes) * len(sizes)
    env = {}
    if tuned:
        path = f"/tmp/moev2_cf_differential_plans_{os.getppid()}.csv"
        _write_plan_file(session, template, comm, rows, path)
        env = {
            envs.AITER_MOEV2_TUNED_TABLE: path,
            envs.AITER_MOEV2_TUNED_TABLE_MODE: "override",
        }
    with mock.patch.dict(os.environ, env):
        tuned_cache.clear_cache()
        try:
            engine = eng.MoeEngine.create(
                template, max_tokens=max(rows, 64), parallel=parallel, comm=session.tp
            )
            engine.prepare()  # no captured buckets: READY
        finally:
            tuned_cache.clear_cache()
    if tuned:
        session.t._barrier(session.group)  # every rank has read the file
        if session.rank == 0:
            os.remove(path)
    for count in sorted({tokens, rows}):
        legacy = CommFusedMoeRuntime(runners=session.runners[comm]).supports(count)
        if engine.supports(count) != legacy:
            raise AssertionError(
                f"engine.supports({count}) is {not legacy}, legacy says {legacy}"
            )
    if not engine.supports(rows):
        raise AssertionError(f"the engine does not serve {rows} tokens")
    runners = engine._stateful[0].runners
    if tuned:
        config = runners.configs[rows]
        if (
            runners is session.runners[comm]
            or config.grid_cap != TUNED_RUNNER["grid_cap"]
        ):
            raise AssertionError(f"the engine does not run the file's runner: {config}")
        if session.runners[comm].configs[rows] == config:
            raise AssertionError("the file's runner is the shipped one")
    elif runners is not session.runners[comm]:
        raise AssertionError("the engine made a runner set of its own")

    def forward():
        return engine.forward(
            moe_args["hidden_states"],
            moe_args["topk_ids"],
            moe_args["topk_weight"],
            weights=weights,
            step=eng.StepInfo(max_tokens_across_ranks=tokens),
            side_inputs=side,
        )

    return forward


def run_case(session, name):
    """A problem string, or None when legacy and v2 agree."""
    import torch

    from aiter.moev2 import paths
    from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime
    from moe_dispatch.differential import _check_output, compare_launches
    from moe_dispatch.gpu_differential import _without_build_output

    comm, tokens, variant, opts = CASES[name]
    bound = opts.get("bound", 0)
    env = {} if bound is None else {"AITER_BF16_FP8_MOE_BOUND": str(bound)}
    with mock.patch.dict(os.environ, env):
        if bound is None:
            os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
        case, moe_args, run_kw = _case_inputs(session, comm, tokens, variant, opts)
        runtime = CommFusedMoeRuntime(runners=session.runners[comm])

        def legacy():
            return runtime.run(**run_kw, **moe_args)

        graph = variant == "graph"
        _run(session, "off", legacy)  # warm-up: JIT and runner compiles log here
        ref, legacy_bench, legacy_logs = _run(session, "off", legacy, graph)
        ref2, _, _ = _run(session, "off", legacy, graph)
        if variant in ("engine", "tuned"):
            try:
                v2 = _engine_fn(
                    session, comm, tokens, moe_args, run_kw, variant == "tuned"
                )
            except Exception as exc:  # noqa: BLE001
                return f"{name}: engine: {type(exc).__name__}: {exc}"
            out, _, _ = _run(session, "off", v2)
            if isinstance(out, str) or isinstance(ref, str):
                return f"{name}: legacy {ref if isinstance(ref, str) else 'ok'}; engine {out if isinstance(out, str) else 'ok'}"
            try:
                _check_output(name, ref, ref2, out)
            except AssertionError as exc:
                return str(exc)
            return None

        path = paths.registered()["comm_fused"]
        results = []

        def counted(call):
            result = path.run(call)
            results.append(result)
            return result

        paths._REGISTRY["comm_fused"] = dataclasses.replace(path, run=counted)
        try:
            out, v2_bench, v2_logs = _run(session, "comm_fused", legacy, graph)
        finally:
            paths._REGISTRY["comm_fused"] = path
            paths.reload()
    if _without_build_output(legacy_logs) != _without_build_output(v2_logs):
        return f"{name}: log lines differ\n    legacy: {legacy_logs}\n    v2:     {v2_logs}"
    if isinstance(ref, str) or isinstance(out, str):
        if ref != out:
            return f"{name}: legacy {ref}; v2 {out}"
        return None if session.rank else f"both raise {ref[:200]}"
    ran = [r for r in results if r is not paths.DECLINED]
    expect_v2 = bound is not None
    expected_runs = (2 if graph else 1) if expect_v2 else 0  # graph: warm-up, capture
    if len(ran) != expected_runs:
        return f"{name}: v2 ran {len(ran)} times ({len(results)} offered), expected {expected_runs}"
    legacy_launches, _ = _ordinary_stage2_launches(legacy_bench)
    v2_launches, fused = _ordinary_stage2_launches(v2_bench)
    if expect_v2:
        bucket = runtime.bucket_for(
            tokens
            if opts.get("sizes") is None
            else max(opts["sizes"]) * len(opts["sizes"])
        )
        if fused is None or fused.runner is not runtime.runners[bucket]:
            return f"{name}: v2 stage 2 is not the M={bucket} runner"
        topk_ids = torch.zeros(
            (bucket, case.topk_ids.shape[1]),
            dtype=case.topk_ids.dtype,
            device=case.topk_ids.device,
        )
        topk_ids[:tokens] = case.topk_ids
        call = {**moe_args, "topk_ids": topk_ids}
        problems = compare_launches(legacy_launches, v2_launches, call, ref, out)
        if problems:
            return f"{name}: launches differ: " + "; ".join(list(problems.values())[:6])
    try:
        _check_output(name, ref, ref2, out)
    except AssertionError as exc:
        return str(exc)
    return None


def _rank_main():
    sys.path.insert(0, str(_HERE.parent))
    sys.path.insert(0, str(_REPO))
    import torch
    import torch.distributed as dist

    from aiter.ops.flydsl import comm_fused_moe_host

    comm_fused_moe_host.get_gpu_model = lambda: "mi355x"
    import op_tests.multigpu_tests.test_comm_fused_moe as t

    skip = t._runtime_skip_reason()
    if skip is not None:
        print(f"comm_fused differential skipped: {skip}")
        return 0
    rank, world, device, group = t._setup_distributed(WORLD)
    if rank != 0:
        t.aiter.logger.setLevel("WARNING")
    session = _Session(t, rank, device, group)
    cpu_group = session.tp.cpu_group
    # The per-expert weight builds can keep some ranks past NCCL's 10 min watchdog
    # while the others wait in the first case's barrier; gloo waits 30 min.
    dist.barrier(group=cpu_group)
    problems = []
    selected = os.environ.get("_MOEV2_CF_CASES")
    names = selected.split("|") if selected else list(CASES)
    try:
        for name in names:
            try:
                problem = run_case(session, name)
            except torch.OutOfMemoryError:
                problem = f"{name}: out of GPU memory (not compared)"
            torch.cuda.empty_cache()
            gathered = [None] * world
            dist.all_gather_object(gathered, problem, group=cpu_group)
            failed = [
                f"rank {r}: {p}"
                for r, p in enumerate(gathered)
                if p and not p.startswith("both raise")
            ]
            if rank == 0:
                note = gathered[0] if gathered[0] and not failed else ""
                print(f"{'FAIL' if failed else 'ok  '} {name} {note}", flush=True)
                for line in failed:
                    print("  " + line[:3000], flush=True)
            problems += failed
    finally:
        t._cleanup_distributed(rank)
    if rank == 0:
        print(f"{len(problems)} problems", flush=True)
    return 1 if problems else 0


def main():
    if "LOCAL_RANK" in os.environ:
        return _rank_main()
    import torch

    if torch.cuda.device_count() < WORLD:
        print(
            f"comm_fused differential needs {WORLD} GPUs, has {torch.cuda.device_count()}"
        )
        return 0
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={WORLD}",
        str(pathlib.Path(__file__).resolve()),
    ]
    return subprocess.run(cmd, cwd=_REPO, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
