# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP8 differential of the `mega_moe_v2` path against `MegaMoEV2`.

    python tests/moe_dispatch/mega_moe_v2_differential.py   # runs torchrun on 8 GPUs
    _MOEV2_MV2_CASES="fixed M=7|compact M=200" python tests/moe_dispatch/...  # some cases
    _MOEV2_MV2_WORLD=4 HIP_VISIBLE_DEVICES=4,5,6,7 python tests/moe_dispatch/...  # EP4

The DeepSeek-V4 EP8 geometry of op_tests/multigpu_tests/test_mega_moe_v2.py (its
fixtures): a fixed-slot instance (capacity 128), a compact one (1024) and a compact one
with a fused shared expert (49 local experts). Hook cases run each forward with
AITER_MOEV2=off twice and AITER_MOEV2=mega_moe_v2 once on every rank: v2 must run the
call, make the same kernel launches (every argument: tensors by address, shape, dtype
and strides) and return the same output (bitwise, or within the noise between the two
legacy runs, reported). Errors must match. Engine cases run `MegaMoEV2Template` through
`MoeEngine` for two layers against vLLM's use of the class (`select_runtime` on the
agreed count, the shared route, the private weight rebind): the same instance
capacities and the same outputs. Fixed-slot needs EP8 (and 48 experts per rank): at
another world size the fixed cases are skipped and the engine has one instance.
"""

import contextlib
import dataclasses
import functools
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
WORLD = int(os.environ.get("_MOEV2_MV2_WORLD", "8"))
SEED = 123

# name: (instance, tokens (one count, or one per rank), variant, options). Instances:
# "fixed" (capacity 128), "compact" (1024), "shared" (1024, fused shared expert).
# Variants: "forward", "prequant" (`forward_prequant` of the instance's own quant),
# "graph" (captured, poisoned, replayed), "engine". Options: "mask" (the last
# `mask` rows of each rank routed to -1, mask_invalid_slots=True), "full"
# (slice_output=False), "error" ("tokens": above the capacity; "dtype": fp32 tokens).
# "memory": what `engine.prepare()` allocated building the template's instances (device
# tensors, MORI symmetric tensors) must be `engine.memory()`; option "quant" (a8w4).
# Engine option "graph": each forward captured and replayed, legacy's too (vLLM's
# padded decode). "disagree": rank 1's template differs, so every rank's `prepare()`
# raises `PlanDisagreement` before any instance is built. "lifetime": a result written
# to `out=` survives the next forward; one without `out` is the view the next overwrites.
UNEVEN = (0, 1, 5, 128, 17, 64, 3, 100)
UNEVEN_COMPACT = (0, 1, 300, 1024, 17, 640, 3, 999)
# vLLM's EP + DP steps on one engine: padded decode (fixed-slot), uneven prefill
# (compact), idle ranks' dummy batches (0 or 1 token) beside a busy rank or alone.
DP_STEPS = (
    64,
    (1000, 300, 0, 300, 280, 0, 310, 1),
    (100, 1, 0, 1, 0, 1, 0, 1),
    (1, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1024),
    128,
)
CASES = {
    "fixed M=1": ("fixed", 1, "forward", {}),
    "fixed M=7": ("fixed", 7, "forward", {}),
    "fixed M=64": ("fixed", 64, "forward", {}),
    "fixed M=128": ("fixed", 128, "forward", {}),
    "fixed uneven, a zero-token rank": ("fixed", UNEVEN, "forward", {}),
    "fixed M=0 everywhere": ("fixed", 0, "forward", {}),
    "fixed M=33 masked tail": ("fixed", 33, "forward", {"mask": 5}),
    "fixed M=33 full output": ("fixed", 33, "forward", {"full": True}),
    "fixed M=33 prequant": ("fixed", 33, "prequant", {}),
    "fixed M=16 graph": ("fixed", 16, "graph", {}),
    "fixed M=129 too many tokens": ("fixed", 129, "forward", {"error": "tokens"}),
    "fixed M=8 fp32 tokens": ("fixed", 8, "forward", {"error": "dtype"}),
    "compact M=1": ("compact", 1, "forward", {}),
    "compact M=200": ("compact", 200, "forward", {}),
    "compact M=1024": ("compact", 1024, "forward", {}),
    "compact uneven, a zero-token rank": ("compact", UNEVEN_COMPACT, "forward", {}),
    "compact M=300 masked tail": ("compact", 300, "forward", {"mask": 7}),
    "compact M=300 full output": ("compact", 300, "forward", {"full": True}),
    "compact M=300 prequant": ("compact", 300, "prequant", {}),
    "compact M=256 graph": ("compact", 256, "graph", {}),
    "shared M=100": ("shared", 100, "forward", {}),
    "engine M=1, 100, 128": ("compact", (1, 100, 128), "engine", {}),
    "engine M=129, 1000, 1024": ("compact", (129, 1000, 1024), "engine", {}),
    "engine uneven": ("compact", (UNEVEN, UNEVEN_COMPACT), "engine", {}),
    "engine shared M=1, 100, 1000": ("shared", (1, 100, 1000), "engine", {}),
    "engine M=50 masked tail": ("compact", (50,), "engine", {"mask": 3}),
    "engine DP steps": ("compact", DP_STEPS, "engine", {}),
    "engine graph M=1, 16, 128": ("compact", (1, 16, 128), "engine", {"graph": True}),
    "engine graph M=256, 1024": ("compact", (256, 1024), "engine", {"graph": True}),
    "engine shared graph M=64": ("shared", (64,), "engine", {"graph": True}),
    "engine plan disagreement": ("compact", None, "disagree", {}),
    "engine output lifetime": ("compact", 100, "lifetime", {}),
    "memory routed": ("compact", None, "memory", {}),
    "memory shared": ("shared", None, "memory", {}),
    "memory routed a4w4": ("compact", None, "memory", {"quant": "a4w4"}),
}


def _canon(value):
    """A launch argument as a comparable value (tensors by address and layout)."""
    import torch

    if isinstance(value, torch.Tensor):
        return (
            "T",
            value.data_ptr(),
            tuple(value.shape),
            str(value.dtype),
            value.stride(),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canon(v)) for k, v in value.items()))
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if hasattr(value, "value"):
        return (type(value).__name__, _canon(value.value))
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


_LAUNCHERS = ("_s1_mega", "_s1_mega_bundle", "_s1_prepare", "_g2_run", "_g2_pair_run")


def _instrument(recorder, mega):
    for attr in _LAUNCHERS:
        setattr(mega, attr, recorder.wrap(attr, getattr(mega, attr)))
    op = mega.comb_op
    op.combine_no_stage1 = recorder.wrap("combine", op.combine_no_stage1)


@dataclasses.dataclass
class _Session:
    t: object
    rank: int
    device: object
    recorder: _Recorder
    layers: dict
    instances: dict
    engines: dict = dataclasses.field(default_factory=dict)
    allocated: dict = dataclasses.field(default_factory=dict)
    capture_stream: object = None  # one for every capture: launches name their stream
    graph_pool: object = None  # and one pool: launches name what a capture allocates
    pool_holder: object = None  # a graph that keeps the pool's memory between captures

    @property
    def net(self):
        return self.t.NETWORKS["v4_pro"]


def _weights(t, net, local_experts, rank, seed, device):
    return t._quantize_weights(
        net["model_dim"], net["inter_dim"], local_experts, rank, seed, device
    )[:4]


def _fixed_slot_allowed(net) -> bool:
    """vLLM's `_uses_fixed_slot` for the 128-token instance."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
        build_mega_moe_bundle_plan,
    )

    return build_mega_moe_bundle_plan(
        128,
        experts_per_rank=net["experts"] // WORLD,
        model_dim=net["model_dim"],
        inter_dim=net["inter_dim"],
        world_size=WORLD,
    ).fixed_slot_dispatch


def _session(t, rank, device):
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_v2

    net = t.NETWORKS["v4_pro"]
    epr = net["experts"] // WORLD
    recorder = _Recorder()
    mega_moe_v2.per_1x32_mx_quant = recorder.wrap(
        "quant", mega_moe_v2.per_1x32_mx_quant
    )
    layers = {
        "a": _weights(t, net, epr, rank, SEED, device),
        "b": _weights(t, net, epr, rank, SEED + 7, device),
        "shared_a": _weights(t, net, epr + 1, rank, SEED + 11, device),
        "shared_b": _weights(t, net, epr + 1, rank, SEED + 13, device),
    }

    def build(capacity, shared):
        w1, w1_scale, w2, w2_scale = layers["shared_a" if shared else "a"]
        args = {**net}
        if shared:
            args["experts"] = net["experts"] + WORLD
            args["topk"] = net["topk"] + 1
        mega = mega_moe_v2.MegaMoEV2(
            rank=rank,
            world_size=WORLD,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=capacity,
            **args,
        )
        _instrument(recorder, mega)
        return mega

    instances = {"compact": build(1024, False), "shared": build(1024, True)}
    if _fixed_slot_allowed(net):
        instances["fixed"] = build(128, False)
        assert instances["fixed"]._s1_fixed_slot
    assert not instances["compact"]._s1_fixed_slot
    import torch

    session = _Session(t, rank, device, recorder, layers, instances)
    session.capture_stream = torch.cuda.Stream()
    session.graph_pool = torch.cuda.graph_pool_handle()
    session.pool_holder = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        session.pool_holder, pool=session.graph_pool, stream=session.capture_stream
    ):
        torch.zeros(1, device=device)
    return session


def _inputs(session, tokens, topk, experts, seed=SEED):
    t = session.t
    x, weights, ids = t._make_inputs(
        max(tokens, 1),
        session.net["model_dim"],
        experts,
        topk,
        session.rank,
        seed,
        session.device,
        force_fanout_boundary=False,
        inject_invalid_route=False,
        force_padding_boundary=False,
    )
    return (
        x[:tokens].contiguous(),
        weights[:tokens].contiguous(),
        ids[:tokens].contiguous(),
    )


def _local(tokens, rank):
    """This rank's count (per-rank tuples list 8 ranks; a smaller world uses the
    first ones)."""
    return tokens[rank] if isinstance(tokens, tuple) else tokens


def _shared_route_reference(weights, ids, rank, routed_per_rank):
    """vLLM's `append_shared_expert`, written out."""
    import torch

    shifted = torch.where(ids >= 0, ids + ids // routed_per_rank, ids)
    shared = rank * (routed_per_rank + 1) + routed_per_rank
    column = torch.full((ids.shape[0], 1), shared, dtype=ids.dtype, device=ids.device)
    ones = torch.ones((ids.shape[0], 1), dtype=torch.float32, device=ids.device)
    return (
        torch.cat([weights.float(), ones], dim=1).contiguous(),
        torch.cat([shifted, column], dim=1).contiguous(),
    )


def _run(session, mode, fn, graph=False):
    """`(output clone or error string, launches, log lines)` of `fn` under
    AITER_MOEV2=`mode`. With `graph`, `fn` is warmed up, captured, and its output
    poisoned and replayed (the launches recorded are the capture's)."""
    import torch

    from moe_dispatch.differential import captured_logs, moev2_mode

    t = session.t
    recorder = session.recorder
    launches, logs = [], []
    with moev2_mode(mode):
        try:
            if graph:
                fn()
                t._barrier()
            t._barrier()
            recorder.launches = launches
            with captured_logs() as logs:
                if graph:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        g, pool=session.graph_pool, stream=session.capture_stream
                    ):
                        out = fn()
                    recorder.launches = None
                    out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    g.replay()
                else:
                    out = fn()
            torch.cuda.synchronize()
            # The copy goes to the host: a device copy kept alive would move the next
            # run's fresh allocations (the fixed-slot quant) and so its launch pointers.
            out = out.cpu()
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            out = f"{type(exc).__name__}: {exc}"
        finally:
            recorder.launches = None
    t._barrier()
    return out, launches, logs


def _compare(name, ref, ref2, out):
    """None (bitwise), a note (within the legacy noise) or a problem (AssertionError)."""
    import torch

    from moe_dispatch.differential import _check_output

    if torch.equal(ref.view(torch.uint8), out.view(torch.uint8)):
        return None
    _check_output(name, ref, ref2, out)
    return "within legacy noise"


def _hook_case(session, name, instance, tokens, variant, opts):
    import torch

    from aiter.moev2 import paths

    mega = session.instances[instance]
    shared = instance == "shared"
    topk = session.net["topk"]
    experts = session.net["experts"]
    local = _local(tokens, session.rank)
    x, weights, ids = _inputs(session, local, topk, experts)
    if shared:
        weights, ids = _shared_route_reference(
            weights, ids, session.rank, experts // WORLD
        )
    if opts.get("mask"):
        ids = ids.clone()
        ids[max(local - opts["mask"], 0) :] = -1
    if opts.get("error") == "tokens":
        x, weights, ids = _inputs(session, mega.mtpr + 1, topk, experts)
    if opts.get("error") == "dtype":
        x = x.float()
    kwargs = {}
    if opts.get("mask"):
        kwargs["mask_invalid_slots"] = True
    if opts.get("full"):
        kwargs["slice_output"] = False
    if variant == "prequant":
        x_q, scales = mega.quantize(x)
        torch.cuda.synchronize()

        def fn():
            return mega.forward_prequant(x_q, scales, weights, ids, **kwargs)

    else:

        def fn():
            return mega.forward(x, weights, ids, **kwargs)

    graph = variant == "graph"
    _run(session, "off", fn)  # warm-up: compiles happen here
    ref, legacy_launches, legacy_logs = _run(session, "off", fn, graph)
    ref2, _, _ = _run(session, "off", fn, graph)
    path = paths.registered()["mega_moe_v2"]
    calls = []

    def counted(call):
        calls.append(call)
        return path.run(call)

    paths._REGISTRY["mega_moe_v2"] = dataclasses.replace(path, run=counted)
    try:
        out, v2_launches, v2_logs = _run(session, "mega_moe_v2", fn, graph)
    finally:
        paths._REGISTRY["mega_moe_v2"] = path
        paths.reload()
    if legacy_logs != v2_logs:
        return f"{name}: log lines differ\n    legacy: {legacy_logs}\n    v2:     {v2_logs}"
    expected = 2 if graph else 1  # graph: warm-up and capture
    if len(calls) != expected:
        return f"{name}: v2 ran {len(calls)} times, expected {expected}"
    if isinstance(ref, str) or isinstance(out, str):
        if ref != out:
            return f"{name}: legacy {ref}; v2 {out}"
        return f"note: both raise {ref[:200]}"
    if not legacy_launches:
        return f"{name}: no launch recorded (the recorder misses legacy's launchers)"
    if legacy_launches != v2_launches:
        for i, (a, b) in enumerate(zip(legacy_launches, v2_launches)):
            if a != b:
                return f"{name}: launch {i} differs\n    legacy: {a}\n    v2:     {b}"
        return (
            f"{name}: {len(legacy_launches)} legacy launches, {len(v2_launches)} v2: "
            f"{[n for n, _, _ in legacy_launches]} vs {[n for n, _, _ in v2_launches]}"
        )
    try:
        note = _compare(name, ref, ref2, out)
    except AssertionError as exc:
        return str(exc)
    return f"note: {note}" if note else None


def _engine(session, shared, mask=None, quant="a8w4"):
    """The engine of this kind, created, prepared and captured once."""
    key = ("shared" if shared else "routed", mask, quant)
    if key in session.engines:
        return session.engines[key]
    engine = _new_engine(session, shared, mask, quant)
    with _allocations() as allocated:
        engine.prepare()
    session.allocated[key] = allocated
    engine.capture()
    session.engines[key] = engine
    return engine


def _new_engine(session, shared, mask, quant="a8w4"):
    import torch.distributed as dist

    from aiter.moev2 import engine as eng
    from aiter.moev2.migration.mega_moe_v2 import MegaMoEV2Template
    from aiter.moev2.problem import ParallelSpec

    net = session.net
    w1, w1_scale, w2, w2_scale = session.layers["shared_a" if shared else "a"]
    template = MegaMoEV2Template(
        eng.MoeWeights(w1, w2, w1_scale, w2_scale),
        world_size=WORLD,
        model_dim=net["model_dim"],
        inter_dim=net["inter_dim"],
        experts=net["experts"],
        topk=net["topk"],
        max_tokens_per_rank=1024,
        quant=quant,
        swiglu_limit=net["swiglu_limit"],
        fuse_shared_expert=shared,
        mask_invalid_slots=mask,
    )
    parallel = ParallelSpec(
        mode="ep",
        world_size=WORLD,
        comm="a2a",
        backend="mega_moe_v2",
        max_tokens_per_rank=1024,
        experts_per_rank=net["experts"] // WORLD + int(shared),
    )
    return eng.MoeEngine.create(
        template,
        max_tokens=1024,
        capture_max_tokens=1024,
        parallel=parallel,
        comm=dist.group.WORLD,
    )


@contextlib.contextmanager
def _allocations():
    """`MemoryNeed` of what the block allocated and still holds: device bytes (torch's
    requested bytes) and MORI symmetric bytes (`mori_shmem_create_tensor`)."""
    import torch

    from aiter.moev2.segments.base import MemoryNeed
    from aiter.ops.flydsl.kernels import flydsl_dispatch_combine_intranode_op as op_mod

    def requested():
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["requested_bytes.all.current"]

    create = op_mod.mori_shmem_create_tensor
    symmetric = []

    def counted(shape, dtype):
        tensor = create(shape, dtype)
        symmetric.append(tensor.nbytes)
        return tensor

    result = {}
    before = requested()
    op_mod.mori_shmem_create_tensor = counted
    try:
        yield result
    finally:
        op_mod.mori_shmem_create_tensor = create
        result["need"] = MemoryNeed(
            device=requested() - before, symmetric=sum(symmetric)
        )


def _disagree_case(session, name):
    import torch.distributed as dist

    from aiter.moev2 import engine as eng

    with _allocations() as allocated:
        try:
            engine = _new_engine(session, False, True if session.rank == 1 else None)
            engine.prepare()
            outcome = "prepared"
        except eng.PlanDisagreement as exc:
            outcome = f"PlanDisagreement: {exc}"
        except Exception as exc:  # noqa: BLE001
            outcome = f"{type(exc).__name__}: {exc}"
    outcomes = [None] * dist.get_world_size()
    dist.all_gather_object(outcomes, outcome)
    wrong = [
        f"rank {r}: {o[:200]}"
        for r, o in enumerate(outcomes)
        if not o.startswith("PlanDisagreement")
    ]
    if wrong:
        return f"{name}: " + "; ".join(wrong)
    if allocated["need"].symmetric:
        return f"{name}: {allocated['need'].symmetric} symmetric bytes allocated"
    return f"note: {outcome[:160]}"


def _lifetime_case(session, name, tokens):
    import torch

    from aiter.moev2 import engine as eng

    try:
        engine = _engine(session, False)
    except Exception as exc:  # noqa: BLE001
        return f"{name}: engine: {type(exc).__name__}: {exc}"
    net = session.net
    topk, experts = net["topk"], net["experts"]
    first = _inputs(session, tokens, topk, experts, seed=SEED)
    second = _inputs(session, tokens, topk, experts, seed=SEED + 1)
    w1, w1_scale, w2, w2_scale = session.layers["a"]
    weights = eng.MoeWeights(w1, w2, w1_scale, w2_scale)
    step = eng.StepInfo(max_tokens_across_ranks=tokens)

    def forward(route, out=None):
        x, w, ids = route
        return engine.forward(x, ids, w, weights=weights, step=step, out=out)

    def fresh():
        return torch.empty(
            tokens, net["model_dim"], dtype=torch.bfloat16, device=session.device
        )

    out = fresh()
    kept = forward(first, out=out)
    torch.cuda.synchronize()
    want = out.clone()
    view = forward(second)
    torch.cuda.synchronize()
    second_result = view.clone()
    again = forward(first)
    torch.cuda.synchronize()
    if kept is not out or not torch.equal(out, want):
        return f"{name}: the out= result changed at the next forward"
    if again.data_ptr() != view.data_ptr():
        return f"{name}: a result without out= is not the view the next forward writes"
    if torch.equal(second_result, want):
        return f"{name}: other inputs gave the same output"
    rewritten = view.cpu()
    noise = forward(first, out=fresh()).cpu()
    try:
        note = _compare(
            f"{name} (the view after the next forward)", want.cpu(), noise, rewritten
        )
    except AssertionError as exc:
        return str(exc)
    return f"note: {note}" if note else None


def _memory_case(session, name, instance, quant):
    shared = instance == "shared"
    try:
        engine = _engine(session, shared, quant=quant)
    except Exception as exc:  # noqa: BLE001
        return f"{name}: engine: {type(exc).__name__}: {exc}"
    allocated = session.allocated[("shared" if shared else "routed", None, quant)]
    allocated = allocated["need"]
    if engine.memory() != allocated:
        return f"{name}: engine.memory() {engine.memory()}, allocated {allocated}"
    return f"note: {engine.memory()} for capacities {engine.template.capacities}"


def _engine_case(session, name, instance, counts, opts):
    import torch

    from aiter.moev2 import engine as eng
    from aiter.moev2.routing import shared_route

    shared = instance == "shared"
    net = session.net
    topk, experts = net["topk"], net["experts"]
    epr = experts // WORLD
    try:
        engine = _engine(session, shared, True if opts.get("mask") else None)
    except Exception as exc:  # noqa: BLE001
        return f"{name}: engine: {type(exc).__name__}: {exc}"
    capacities = engine.template.capacities
    if shared:
        legacy_runtimes = [(1024, session.instances["shared"])]
    else:
        legacy_runtimes = [(1024, session.instances["compact"])]
        if "fixed" in session.instances:
            legacy_runtimes.insert(0, (128, session.instances["fixed"]))
    if capacities != tuple(c for c, _ in legacy_runtimes):
        return f"{name}: engine capacities {capacities}, vLLM's {[c for c, _ in legacy_runtimes]}"
    notes = []
    for i, count in enumerate(counts):
        local = _local(count, session.rank)
        agreed = max(count[:WORLD]) if isinstance(count, tuple) else count
        x, weights, ids = _inputs(session, local, topk, experts, seed=SEED + i)
        if opts.get("mask"):
            ids = ids.clone()
            ids[max(local - opts["mask"], 0) :] = -1
        if shared:
            got = shared_route.append_shared_expert(
                weights,
                ids,
                shared_expert_id=shared_route.shared_expert_id(session.rank, epr),
                routed_per_rank=epr,
            )
            want = _shared_route_reference(weights, ids, session.rank, epr)
            if not (torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])):
                return f"{name}: the shared route differs from vLLM's at M={local}"
        for layer in ("a", "b"):
            key = f"shared_{layer}" if shared else layer
            w1, w1_scale, w2, w2_scale = session.layers[key]
            runtime = next(r for c, r in legacy_runtimes if agreed <= c)
            route = (x, weights, ids)
            layer_weights = (w1, w1_scale, w2, w2_scale)
            legacy = functools.partial(
                _vllm_forward,
                runtime,
                layer_weights,
                route,
                (session.rank, epr) if shared else None,
                bool(opts.get("mask")) or None,
            )
            v2 = functools.partial(
                _engine_forward, engine, eng, layer_weights, route, agreed
            )
            graph = bool(opts.get("graph"))
            _run(session, "off", legacy)
            ref, _, _ = _run(session, "off", legacy, graph)
            ref2, _, _ = _run(session, "off", legacy, graph)
            out, _, _ = _run(session, "off", v2, graph)
            label = f"{name} layer {layer} M={local} (agreed {agreed})"
            if isinstance(ref, str) or isinstance(out, str):
                if ref != out:
                    return f"{label}: legacy {ref}; engine {out}"
                notes.append(f"both raise {ref[:120]}")
                continue
            try:
                note = _compare(label, ref, ref2, out)
            except AssertionError as exc:
                return str(exc)
            if note:
                notes.append(f"{label} {note}")
    return f"note: {'; '.join(notes)}" if notes else None


def _weights_of(mega):
    """The weights a legacy instance launches with (vLLM rebinds them per layer)."""
    return mega._s1_w1, mega._s1_w1_scale, mega.w2, mega.w2_scale


def _vllm_forward(runtime, layer_weights, route, shared, mask):
    """vLLM's use of the class: the shared route (`shared`: rank and routed experts per
    rank, or None), the private weight rebind (its parameters are uint8), then the
    forward."""
    import torch

    x, w, d = route
    if shared is not None:
        w, d = _shared_route_reference(w, d, *shared)
    runtime._s1_w1, runtime._s1_w1_scale, runtime.w2, runtime.w2_scale = (
        w.view(torch.uint8) for w in layer_weights
    )
    return runtime.forward(x.contiguous(), w, d, mask_invalid_slots=mask)


def _engine_forward(engine, eng, layer_weights, route, agreed):
    x, weights, ids = route
    w1, w1_scale, w2, w2_scale = layer_weights
    return engine.forward(
        x,
        ids,
        weights,
        weights=eng.MoeWeights(w1, w2, w1_scale, w2_scale),
        step=eng.StepInfo(max_tokens_across_ranks=agreed),
    )


def run_case(session, name):
    """None, a note (starting with "note:") or a problem."""
    instance, tokens, variant, opts = CASES[name]
    if instance not in session.instances:
        return f"note: skipped, no {instance} instance at EP{WORLD}"
    if variant == "memory":
        return _memory_case(session, name, instance, opts.get("quant", "a8w4"))
    if variant == "disagree":
        return _disagree_case(session, name)
    if variant == "lifetime":
        return _lifetime_case(session, name, tokens)
    if variant == "engine":
        bound = {k: _weights_of(m) for k, m in session.instances.items()}
        try:
            return _engine_case(session, name, instance, tokens, opts)
        finally:
            for k, m in session.instances.items():
                m._s1_w1, m._s1_w1_scale, m.w2, m.w2_scale = bound[k]
    return _hook_case(session, name, instance, tokens, variant, opts)


def _is_note(problem):
    return problem.startswith("note:")


def _rank_main():
    sys.path.insert(0, str(_HERE.parent))
    sys.path.insert(0, str(_REPO))
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
    import torch
    import torch.distributed as dist

    import op_tests.multigpu_tests.test_mega_moe_v2 as t

    rank, world, device = t._setup_dist()
    if world != WORLD:
        raise SystemExit(f"needs {WORLD} ranks, got {world}")
    if rank != 0:
        t.aiter.logger.setLevel("WARNING")
    problems = []
    selected = os.environ.get("_MOEV2_MV2_CASES")
    names = selected.split("|") if selected else list(CASES)
    try:
        session = _session(t, rank, device)
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
    finally:
        t._cleanup()
    if rank == 0:
        print(f"{len(problems)} problems", flush=True)
    return 1 if problems else 0


def main():
    if "LOCAL_RANK" in os.environ:
        return _rank_main()
    import torch

    if torch.cuda.device_count() < WORLD:
        print(
            f"mega_moe_v2 differential needs {WORLD} GPUs, has {torch.cuda.device_count()}"
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
