# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP differential of the `mega_moe_tp` path against `MegaMoeTP`.

    HIP_VISIBLE_DEVICES=4,5,6,7 python tests/moe_dispatch/mega_moe_tp_differential.py
    _MOEV2_MTP_CASES="ag_rs M=7|ar M=32" python tests/moe_dispatch/...  # some cases
    _MOEV2_MTP_WORLD=8 python tests/moe_dispatch/...  # TP8

The M3 geometry of op_tests/multigpu_tests/test_mega_moe_tp.py (its fixtures, Swiglu,
MXFP4): one ag_rs and one ar instance (512 local tokens, the small-batch engine; big-M
is not reached, todo 6.4) over two layers. Hook cases run each forward with
AITER_MOEV2=off twice and AITER_MOEV2=mega_moe_tp once on every rank: v2 must run the
call, make the same kernel launches (the launcher and every argument: pointers, offsets,
counts, the stream) and return the same output (bitwise, or within the noise between
the two legacy runs, reported), fused tail included. Errors must match. Engine cases run
`MegaMoeTPTemplate` over the same instances through `MoeEngine` (captured buckets, two
layers, the optional tail where `supports` says legacy fuses it) against legacy's use
of the instance (`set_weights`, then `forward`): the same launches and outputs.
"""

import dataclasses
import functools
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
WORLD = int(os.environ.get("_MOEV2_MTP_WORLD", "4"))
SEED = 123
MAX_LOCAL = 512

# name: (instance, rows (forward's: ag_rs, this rank's; ar, the replicated batch),
# variant, options). Variants: "forward", "graph" (captured, poisoned, replayed),
# "engine" (rows: the counts, run on both layers). Options: "out" (out=), "tail"
# (the fused tail into separate res_out, "bf16": its bf16 rows too), "dtypes" (int64
# ids, bf16 weights), "error" ("dtype": fp32 x), "graph" (engine: captured as above);
# engines: "tail" / "tail_bf16" the variants prepared, each run where it fuses.
# "memory": a new instance that reaches big-M; what constructing it and its big-M
# kernel allocates must be the segments' `workspace_bytes()`.
CASES = {
    "ag_rs M=1": ("ag_rs", 1, "forward", {}),
    "ag_rs M=7": ("ag_rs", 7, "forward", {}),
    "ag_rs M=64": ("ag_rs", 64, "forward", {}),
    "ag_rs M=200": ("ag_rs", 200, "forward", {}),
    "ag_rs M=512": ("ag_rs", 512, "forward", {}),
    "ag_rs M=0": ("ag_rs", 0, "forward", {}),
    "ag_rs M=100 out": ("ag_rs", 100, "forward", {"out": True}),
    "ag_rs M=33 int64 ids, bf16 weights": ("ag_rs", 33, "forward", {"dtypes": True}),
    "ag_rs M=128 tail": ("ag_rs", 128, "forward", {"tail": True}),
    "ag_rs M=128 tail bf16": ("ag_rs", 128, "forward", {"tail": True, "bf16": True}),
    "ag_rs M=128 tail out": ("ag_rs", 128, "forward", {"tail": True, "out": True}),
    "ag_rs M=8 tail (does not fuse)": ("ag_rs", 8, "forward", {"tail": True}),
    "ag_rs M=513 too many tokens": ("ag_rs", 513, "forward", {}),
    "ag_rs M=8 fp32 x": ("ag_rs", 8, "forward", {"error": "dtype"}),
    "ag_rs M=16 graph": ("ag_rs", 16, "graph", {}),
    "ag_rs M=128 graph": ("ag_rs", 128, "graph", {}),
    "ag_rs M=128 tail graph": ("ag_rs", 128, "graph", {"tail": True}),
    "ar M=4": ("ar", 4, "forward", {}),
    "ar M=32": ("ar", 32, "forward", {}),
    "ar M=256": ("ar", 256, "forward", {}),
    "ar M=1024": ("ar", 1024, "forward", {}),
    "ar M=64 out": ("ar", 64, "forward", {"out": True}),
    "ar M=256 out": ("ar", 256, "forward", {"out": True}),
    "ar M=6 not a multiple of tp": ("ar", 6, "forward", {}),
    "ar M=256 tail (none in ar)": ("ar", 256, "forward", {"tail": True}),
    "ar M=64 graph": ("ar", 64, "graph", {}),
    "engine ag_rs M=1, 7, 64, 200, 512": ("ag_rs", (1, 7, 64, 200, 512), "engine", {}),
    "engine ag_rs tail M=8, 64, 128, 300": (
        "ag_rs",
        (8, 64, 128, 300),
        "engine",
        {"tail": True},
    ),
    "engine ag_rs both tails M=8, 64, 128": (
        "ag_rs",
        (8, 64, 128),
        "engine",
        {"tail": True, "tail_bf16": True},
    ),
    "engine ar M=4, 256, 1024": ("ar", (4, 256, 1024), "engine", {}),
    "engine ag_rs graph M=16, 128": ("ag_rs", (16, 128), "engine", {"graph": True}),
    "engine ag_rs tail graph M=64, 128": (
        "ag_rs",
        (64, 128),
        "engine",
        {"tail": True, "graph": True},
    ),
    "engine ar graph M=8, 512": ("ar", (8, 512), "engine", {"graph": True}),
    "memory ag_rs": ("ag_rs", None, "memory", {}),
    "memory ar": ("ar", None, "memory", {}),
}


def _canon(value):
    """A launch argument as a comparable value."""
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canon(v)) for k, v in value.items()))
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if hasattr(value, "cuda_stream"):
        return ("stream", value.cuda_stream)
    return (type(value).__name__, id(value))


class _Recorder:
    def __init__(self):
        self.launches = None

    def wrap(self, name, fn):
        def recorded(*args, **kwargs):
            if self.launches is not None:
                self.launches.append((name, _canon(args), _canon(kwargs)))
            return fn(*args, **kwargs)

        return recorded


@dataclasses.dataclass
class _Session:
    t: object
    ctx: object
    recorder: _Recorder
    layers: dict
    instances: dict
    inputs: dict = dataclasses.field(default_factory=dict)
    written: list = dataclasses.field(default_factory=list)
    tails: dict = dataclasses.field(default_factory=dict)
    engines: dict = dataclasses.field(default_factory=dict)
    capture_stream: object = None
    graph_pool: object = None
    pool_holder: object = None

    @property
    def shape(self):
        return self.t.MODELS["m3"]


def _session(t):
    import torch

    ctx = t.Ctx()
    from aiter.ops.flydsl.kernels.mega_moe_tp import mega_moe_tp as kernels

    recorder = _Recorder()
    kernels._run_compiled = recorder.wrap("launch", kernels._run_compiled)
    shape = t.MODELS["m3"]
    layers = {
        "a": t.build_weights(shape, ctx, SEED),
        "b": t.build_weights(shape, ctx, SEED + 7),
    }
    instances = {
        mode: t.new_layer(shape, layers["a"], ctx, mode, MAX_LOCAL) for mode in t.MODES
    }
    for mtp in instances.values():
        if mtp._big(MAX_LOCAL):
            raise SystemExit(
                "the instances reach big-M; it does not load here (todo 6.4)"
            )
    session = _Session(t, ctx, recorder, layers, instances)
    session.capture_stream = torch.cuda.Stream()
    session.graph_pool = torch.cuda.graph_pool_handle()
    session.pool_holder = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        session.pool_holder, pool=session.graph_pool, stream=session.capture_stream
    ):
        torch.zeros(1, device=ctx.device)
    return session


def _inputs(session, mode, rows, seed=SEED, dtypes=False):
    """This rank's rows (ag_rs) or the replicated batch (ar, the same on every rank),
    kept per case so legacy and v2 read the same tensors."""
    key = (mode, rows, seed, dtypes)
    if key not in session.inputs:
        session.inputs[key] = _new_inputs(session, mode, rows, seed, dtypes)
    return session.inputs[key]


def _new_inputs(session, mode, rows, seed, dtypes):
    import torch

    t, ctx, shape = session.t, session.ctx, session.shape
    rank_seed = seed + 7919 * ctx.rank if mode == "ag_rs" else seed
    gen = torch.Generator(device=ctx.device).manual_seed(rank_seed)
    x = torch.randn(
        (max(rows, 1), shape.model_dim),
        dtype=torch.bfloat16,
        device=ctx.device,
        generator=gen,
    )
    w, ids = t.route(max(rows, 1), shape, "random", gen, ctx.device)
    x, w, ids = x[:rows].contiguous(), w[:rows].contiguous(), ids[:rows].contiguous()
    if dtypes:
        w, ids = w.to(torch.bfloat16), ids.to(torch.int64)
    return x, w, ids


def _tail(session, local):
    """`(res_in, res_out, norm_w)`, kept per count so every run sees the same pointers."""
    import torch

    key = local
    if key not in session.tails:
        ctx, H = session.ctx, session.shape.model_dim
        gen = torch.Generator(device=ctx.device).manual_seed(77 + ctx.rank + local)
        res_in = torch.randn((local, H), generator=gen, device=ctx.device)
        norm_w = 0.1 * torch.randn((H,), generator=gen, device=ctx.device)
        session.tails[key] = (
            res_in.to(torch.bfloat16),
            torch.zeros((local, H), dtype=torch.bfloat16, device=ctx.device),
            norm_w.to(torch.bfloat16),
        )
    return session.tails[key]


def _host(result, mtp, local, tail):
    """The result on the host: y, and with the tail res_out and this rank's rows of
    q / scale (/ bf16 rows: the others are not written, `tn_local`)."""
    if not isinstance(result, tuple):
        return (result.cpu(),)
    rank = mtp.cfg.rank
    own = slice(rank * local, (rank + 1) * local)
    return (result[0].cpu(), tail[1].cpu(), *(r[own].cpu() for r in result[1:]))


def _poison(buffers):
    """Fill the buffers with 0xFF bytes, on every rank before any rank runs on (peers
    write into the symmetric ones)."""
    import torch

    for r in buffers:
        r.view(torch.uint8).fill_(255)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def _attempt(step):
    """`(value, None)` or `(None, "Type: message")`; no host collective inside, so a
    rank whose step raises still joins every host collective of the case (its peers
    may first wait in a fused kernel until the kernel's timeout)."""
    import torch

    stream = torch.cuda.current_stream()
    try:
        return step(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        torch.cuda.set_stream(stream)  # a failed capture_end skips the stream exit


def _run(session, mode, fn, graph=False):
    """`(host outputs or error string, launches, log lines)` of `fn` under
    AITER_MOEV2=`mode`. The buffers the previous run wrote are poisoned first. With
    `graph`, `fn` is warmed up, captured, and its output poisoned and replayed (the
    launches recorded are the capture's). Out of GPU memory on any rank raises
    `OutOfMemoryError` on every rank."""
    import torch

    from moe_dispatch.differential import captured_logs, moev2_mode

    recorder = session.recorder
    launches = []
    _poison(session.written)
    with moev2_mode(mode):
        error = None
        if graph:
            _, error = _attempt(fn)
            torch.cuda.synchronize()
        torch.distributed.barrier()
        recorder.launches = launches
        try:
            with captured_logs() as logs:
                if error is None and graph:
                    g = torch.cuda.CUDAGraph()

                    def capture():
                        with torch.cuda.graph(
                            g, pool=session.graph_pool, stream=session.capture_stream
                        ):
                            return fn()

                    out, error = _attempt(capture)
                elif error is None:
                    out, error = _attempt(fn)
        finally:
            recorder.launches = None
        if graph:
            _poison(out if error is None else [])
            if error is None:
                _, error = _attempt(g.replay)
        torch.cuda.synchronize()
        if error is None:
            session.written = list(out)
            out, error = _attempt(out)
    oom = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(
        oom, bool(error) and error.startswith("OutOfMemoryError")
    )
    if any(oom):
        raise torch.OutOfMemoryError(f"rank {oom.index(True)} ran out of GPU memory")
    return (error if error is not None else out), launches, logs


def _compare(name, ref, ref2, out):
    """None (bitwise), a note (within the legacy noise) or a problem (AssertionError)."""
    import torch

    from moe_dispatch.differential import _check_output

    notes = []
    for i, (a, a2, b) in enumerate(zip(ref, ref2, out)):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{name}: output {i} is {b.dtype} {tuple(b.shape)}")
        for who, t in (("legacy", a), ("v2", b)):
            rows = t.reshape(t.shape[0], -1).view(torch.uint8) if t.numel() else None
            if rows is not None and (rows == 255).all(dim=-1).any():
                raise AssertionError(f"{name}: {who} left rows of output {i} unwritten")
        if torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            continue
        _check_output(f"{name} output {i}", a.float(), a2.float(), b.float())
        notes.append(f"output {i} within legacy noise")
    if len(ref) != len(out):
        raise AssertionError(f"{name}: {len(ref)} legacy outputs, {len(out)} v2")
    return "; ".join(notes) or None


def _forward_fn(session, mtp, mode, rows, opts, seed=SEED):
    """A forward of the instance that returns a host-copy thunk (so graph replays copy
    after the replay). The buffers are allocated at the first call (inside `_attempt`,
    so an allocation failing on one rank does not desync the ranks), then reused."""
    import torch

    local = mtp._local(rows)
    state = {}

    def setup():
        x, w, ids = _inputs(session, mode, rows, seed, dtypes=opts.get("dtypes", False))
        if opts.get("error") == "dtype":
            x = x.float()
        kwargs = {}
        if opts.get("tail"):
            kwargs["tail"] = _tail(session, max(local, 1))
            kwargs["bf16"] = bool(opts.get("bf16"))
        if opts.get("out"):
            kwargs["out"] = torch.zeros(
                (rows, session.shape.model_dim), dtype=torch.bfloat16, device=x.device
            )
        state.update(args=(x, w, ids), kwargs=kwargs)

    def fn():
        if not state:
            setup()
        result = mtp.forward(*state["args"], **state["kwargs"])
        return _Pending(result, mtp, local, state["kwargs"].get("tail"))

    return fn


class _Pending:
    """A forward's result, copied to the host when called (after a graph replay)."""

    def __init__(self, result, mtp, local, tail):
        self.result, self.mtp, self.local, self.tail = result, mtp, local, tail

    def __iter__(self):
        """The device buffers the call writes (poisoned before a graph replay)."""
        r = self.result
        written = list(r) if isinstance(r, tuple) else [r]
        return iter(written + ([self.tail[1]] if self.tail else []))

    def __call__(self):
        return _host(self.result, self.mtp, self.local, self.tail)


def _hook_case(session, name, instance, rows, variant, opts):
    from aiter.moev2 import paths

    mtp = session.instances[instance]
    mtp.set_weights(**_layer_kwargs(session, "a"))
    fn = _forward_fn(session, mtp, instance, rows, opts)
    graph = variant == "graph"
    _run(session, "off", fn)  # warm-up: compiles happen here
    ref, legacy_launches, legacy_logs = _run(session, "off", fn, graph)
    ref2, _, _ = _run(session, "off", fn, graph)
    path = paths.registered()["mega_moe_tp"]
    calls = []

    def counted(call):
        calls.append(call)
        return path.run(call)

    paths._REGISTRY["mega_moe_tp"] = dataclasses.replace(path, run=counted)
    try:
        out, v2_launches, v2_logs = _run(session, "mega_moe_tp", fn, graph)
    finally:
        paths._REGISTRY["mega_moe_tp"] = path
        paths.reload()
    if legacy_logs != v2_logs:
        return f"{name}: log lines differ\n    legacy: {legacy_logs}\n    v2:     {v2_logs}"
    if isinstance(ref, str) or isinstance(out, str):
        if not calls:
            return f"{name}: v2 never ran (legacy {ref}; v2 {out})"
        if ref != out:
            return f"{name}: legacy {ref}; v2 {out}"
        return f"note: both raise {ref[:200]}"
    expected = 2 if graph else 1  # graph: warm-up and capture
    if len(calls) != expected:
        return f"{name}: v2 ran {len(calls)} times, expected {expected}"
    problem = _launches_differ(name, legacy_launches, v2_launches, rows)
    if problem:
        return problem
    try:
        note = _compare(name, ref, ref2, out)
    except AssertionError as exc:
        return str(exc)
    return f"note: {note}" if note else None


def _launches_differ(name, legacy, v2, rows):
    if not legacy and rows:
        return f"{name}: no launch recorded (the recorder misses legacy's launcher)"
    if legacy == v2:
        return None
    for i, (a, b) in enumerate(zip(legacy, v2)):
        if a != b:
            return f"{name}: launch {i} differs\n    legacy: {a}\n    v2:     {b}"
    return f"{name}: {len(legacy)} legacy launches, {len(v2)} v2"


def _layer_kwargs(session, layer):
    wt = session.layers[layer]
    return {
        "w1": wt.w1,
        "w1_scale": wt.w1_scale,
        "w2": wt.w2,
        "w2_scale": wt.w2_scale,
    }


def _engine(session, instance, tail, tail_bf16):
    import torch.distributed as dist

    from aiter.moev2 import engine as eng
    from aiter.moev2.migration.mega_moe_tp import MegaMoeTPTemplate
    from aiter.moev2.problem import ParallelSpec

    key = (instance, tail, tail_bf16)
    if key in session.engines:
        return session.engines[key]
    mtp = session.instances[instance]
    mtp.set_weights(**_layer_kwargs(session, "a"))
    max_tokens = MAX_LOCAL * (WORLD if instance == "ar" else 1)
    parallel = ParallelSpec(
        mode="tp",
        world_size=WORLD,
        comm=instance,
        backend="mega_moe_tp",
        max_tokens_per_rank=MAX_LOCAL,
        experts_per_rank=session.shape.experts,
    )
    engine = eng.MoeEngine.create(
        MegaMoeTPTemplate(mtp, tail=tail, tail_bf16=tail_bf16),
        max_tokens=max_tokens,
        capture_max_tokens=128 * (WORLD if instance == "ar" else 1),
        parallel=parallel,
        comm=dist.group.WORLD,
    )
    try:
        engine.prepare()
        engine.capture()
    except Exception:
        engine.abort()
        raise
    session.engines[key] = engine
    return engine


def _engine_case(session, name, instance, counts, opts):
    import torch.distributed as dist

    from aiter.moev2 import engine as eng

    key = (instance, bool(opts.get("tail")), bool(opts.get("tail_bf16")))
    engine, error = _attempt(lambda: _engine(session, *key))
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        if engine is not None:
            engine.abort()
            del session.engines[key]
        session.instances[instance].reset()  # peers' kernels may have timed out
        return f"{name}: engine: " + "; ".join(
            f"rank {r} {e}" for r, e in enumerate(errors) if e
        )
    mtp = session.instances[instance]
    TAIL = ("res_in", "res_out", "norm_w")
    kinds = {
        kind: names
        for kind, names, prepared in (
            ("tail", TAIL, key[1]),
            ("tail bf16", (*TAIL, "tail_bf16"), key[2]),
        )
        if prepared
    }
    for rows in range(MAX_LOCAL + 1):
        for kind, names in kinds.items():
            if engine.supports(rows, side_inputs=names) != mtp.tail_ok(rows):
                return (
                    f"{name}: engine.supports({rows}, {kind}) is not legacy's tail_ok"
                )
    notes = []
    if kinds:
        fused_at = [rows for rows in counts if mtp.tail_ok(rows)]
        if not fused_at:
            return f"{name}: the tail fuses at none of {counts}"
        notes.append(f"the tail fuses at M={fused_at}")
    problems = []  # every rank runs every step: a rank that stopped would hang the rest
    runs = [
        (i, rows, kind)
        for i, rows in enumerate(counts)
        for kind in ([k for k in kinds if mtp.tail_ok(rows)] or [None])
    ]
    for i, rows, kind in runs:
        fused = kind is not None
        case_opts = {"tail": True, "bf16": kind == "tail bf16"} if fused else {}
        for layer in ("a", "b"):
            label = f"{name} layer {layer} M={rows}{f' {kind}' if fused else ''}"
            legacy_fn = _forward_fn(session, mtp, instance, rows, case_opts, SEED + i)
            weights = _layer_kwargs(session, layer)

            def legacy(fn=legacy_fn, weights=weights):
                mtp.set_weights(**weights)
                return fn()

            v2 = functools.partial(
                _engine_forward,
                session,
                engine,
                eng,
                instance,
                rows,
                layer,
                kind,
                SEED + i,
            )
            graph = bool(opts.get("graph"))
            _run(session, "off", legacy)
            ref, legacy_launches, _ = _run(session, "off", legacy, graph)
            ref2, _, _ = _run(session, "off", legacy, graph)
            mtp.set_weights(**_layer_kwargs(session, "b" if layer == "a" else "a"))
            out, v2_launches, _ = _run(session, "off", v2, graph)
            if isinstance(ref, str) or isinstance(out, str):
                if ref != out:
                    problems.append(f"{label}: legacy {ref}; engine {out}")
                else:
                    notes.append(f"both raise {ref[:120]}")
                continue
            problem = _launches_differ(label, legacy_launches, v2_launches, rows)
            if problem:
                problems.append(problem)
                continue
            try:
                note = _compare(label, ref, ref2, out)
            except AssertionError as exc:
                problems.append(str(exc))
                continue
            if note:
                notes.append(f"{label} {note}")
    health, error = _attempt(engine.health)
    if error is not None:
        problems.append(f"{name}: engine health: {error}")
    elif not health.ok:
        problems.append(f"{name}: engine health {health.reasons}")
    healthy = [None] * dist.get_world_size()
    dist.all_gather_object(healthy, error is None and health.ok)
    if not all(healthy):  # a FAILED engine would fail every later case too
        engine.abort()
        del session.engines[key]
        mtp.reset()
    if problems:
        return "\n  ".join(problems)
    return f"note: {'; '.join(notes)}" if notes else None


def _engine_forward(session, engine, eng, instance, rows, layer, kind, seed):
    mtp = session.instances[instance]
    x, w, ids = _inputs(session, instance, rows, seed)
    wt = session.layers[layer]
    side = None
    tail = None
    if kind is not None:
        tail = _tail(session, mtp._local(rows))
        side = dict(zip(("res_in", "res_out", "norm_w"), tail))
        if kind == "tail bf16":
            side["tail_bf16"] = True
    result = engine.forward(
        x,
        ids,
        w,
        weights=eng.MoeWeights(wt.w1, wt.w2, wt.w1_scale, wt.w2_scale),
        step=eng.StepInfo(max_tokens_across_ranks=rows),
        side_inputs=side,
    )
    return _Pending(result, mtp, mtp._local(rows), tail)


def _memory_case(session, name, mode):
    import torch
    import torch.distributed as dist

    from aiter.moev2.segments.mega_moe_tp import MegaMoeTPBigM, MegaMoeTPSmall
    from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_config import PAD

    def requested():
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["requested_bytes.all.current"]

    t = session.t
    built = {}

    def build():
        before = requested()
        built["mtp"] = t.new_layer(
            session.shape,
            session.layers["a"],
            session.ctx,
            mode,
            MAX_LOCAL,
            bigm_min_tokens=PAD * WORLD,
        )
        return requested() - before

    def bigm():
        before = requested()
        built["mtp"].bigm()
        return requested() - before

    small, error = _attempt(build)
    big = None
    if error is None:
        big, error = _attempt(bigm)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        return f"{name}: " + "; ".join(
            f"rank {r} {e}" for r, e in enumerate(errors) if e
        )
    mtp = built["mtp"]
    problems = []
    for segment, measured, arena in (
        (MegaMoeTPSmall.adopt(mtp), small, mtp.engine.arena.storage),
        (MegaMoeTPBigM.adopt(mtp), big, mtp._bigm.sd.storage),
    ):
        need = segment.workspace_bytes()
        if need.total != measured or need.symmetric != arena.nbytes:
            problems.append(
                f"{name}: {segment.name} reports {need}, {measured} bytes allocated "
                f"({arena.nbytes} in the arena)"
            )
    torch.cuda.synchronize()
    dist.barrier()
    mtp.close()
    if problems:
        return "\n  ".join(problems)
    return f"note: small {small} bytes, big-M {big} bytes"


def run_case(session, name):
    """None, a note (starting with "note:") or a problem."""
    instance, rows, variant, opts = CASES[name]
    if variant == "memory":
        return _memory_case(session, name, instance)
    if variant == "engine":
        return _engine_case(session, name, instance, rows, opts)
    return _hook_case(session, name, instance, rows, variant, opts)


def _is_note(problem):
    return problem.startswith("note:")


def _rank_main():
    sys.path.insert(0, str(_HERE.parent))
    sys.path.insert(0, str(_REPO))
    import torch
    import torch.distributed as dist

    import op_tests.multigpu_tests.test_mega_moe_tp as t

    problems = []
    selected = os.environ.get("_MOEV2_MTP_CASES")
    names = selected.split("|") if selected else list(CASES)
    session = _session(t)
    rank, world = session.ctx.rank, session.ctx.world
    if world != WORLD:
        raise SystemExit(f"needs {WORLD} ranks, got {world}")
    if rank != 0:
        t.aiter.logger.setLevel("WARNING")
    try:
        for name in names:
            try:
                problem = run_case(session, name)
            except torch.OutOfMemoryError:
                problem = f"{name}: out of GPU memory (not compared)"
            gathered = [None] * world
            dist.all_gather_object(gathered, problem)
            failed = [
                f"rank {r}: {p}"
                for r, p in enumerate(gathered)
                if p and not _is_note(p)
            ]
            notes = sorted({p for p in gathered if p and _is_note(p)})
            if rank == 0:
                note = (
                    f" ({len(notes)} distinct notes: {notes[0][:300]})"
                    if notes and not failed
                    else ""
                )
                print(f"{'FAIL' if failed else 'ok  '} {name}{note}", flush=True)
                for line in failed:
                    print("  " + line[:3000], flush=True)
            problems += failed
        for engine in session.engines.values():
            engine.release()
        torch.cuda.synchronize()
        dist.barrier()
        for mtp in session.instances.values():
            mtp.close()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print(f"{len(problems)} problems", flush=True)
    return 1 if problems else 0


def main():
    if "LOCAL_RANK" in os.environ:
        return _rank_main()
    import torch

    if torch.cuda.device_count() < WORLD:
        print(
            f"mega_moe_tp differential needs {WORLD} GPUs, has {torch.cuda.device_count()}"
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
