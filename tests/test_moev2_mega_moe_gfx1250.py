# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mega_moe_gfx1250` against legacy `MegaMoEGfx1250.forward`, on CPU.

Each case builds two pipelines alike, as gfx1250 (`arch_sim`), on a fake cco
communicator whose arena is a CPU buffer, with every kernel stubbed: the FlyDSL TDM or
(fake) MORI dispatch, the compact plan, the dispatch-wire quant, the grouped kernel
entry (`_grouped_a8w4_tdm_moe`) and the combine sync and reduce. The stubs record each
launch with its arguments (a pointer as the allocation it falls in and its offset, a
tensor with its values) and move bytes, so the output depends on every stage. The same
forward sequence runs on one legacy and on the other with `AITER_MOEV2=mega_moe_gfx1250`:
the same launches, stream and event operations, outputs, exceptions and log lines, and
v2 serves exactly the forwards whose nested call legacy's grouped branch serves.
"""

import contextlib
import dataclasses
import enum
import sys
import types
from math import prod

import pytest
import torch
from moe_dispatch.arch_sim import simulated_arch
from moe_dispatch.differential import captured_logs, moev2_mode

import aiter.fused_moe as fm
import aiter.ops.quant as quant_ops
from aiter import ActivationType, dtypes
from aiter.moev2 import engine as engine_mod
from aiter.moev2 import paths
from aiter.moev2.migration import mega_moe_gfx1250 as path
from aiter.moev2.problem import ParallelSpec
from aiter.moev2.segments.base import MemoryNeed
from aiter.moev2.segments.comm_gfx1250 import CommDispatchGfx1250
from aiter.moev2.segments.grouped_gfx1250 import GROUPED_P2P
from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import compact_plan, mega_moe
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext

WORLD, EXPERTS, TOPK, MODEL, INTER, MAX_TOK = 2, 8, 2, 256, 128, 8
_ENV = (
    "AITER_USE_GROUPED_GEMM",
    "AITER_DISABLE_GROUPED_A8W4",
    "AITER_FORCE_A8W4",
    "AITER_TDM_DIRECT_EP_MASK",
    "AITER_TDM_COMPACT_BLOCKS",
    "AITER_TDM_COMPACT_WARPS",
    "MEGA_DISPATCH",
    "MEGA_DISPATCH_WIRE",
    "MEGA_COMBINE_QUANT",
    "MORI_EP_TOKOFF_EXT",
    "GPU_ARCHS",
    "AITER_MOEV2",
)


class _FellThrough(Exception):
    """Legacy's nested call reached `get_2stage_cfgs`: the grouped branch returned
    None."""


class _Memory:
    """The device memory of one pipeline: its allocations and the call's inputs, by
    name, so a pointer reads the same in two pipelines built alike."""

    def __init__(self):
        self.blocks = []  # (base, nbytes, name, uint8 view)
        self.inputs = {}

    def alloc(self, nbytes, name=None):
        raw = torch.zeros(nbytes + 256, dtype=torch.uint8)
        pad = -raw.data_ptr() % 256
        buf = raw[pad : pad + nbytes]
        self.blocks.append(
            (buf.data_ptr(), nbytes, name or f"t{len(self.blocks)}", buf)
        )
        return buf

    def tensor(self, shape, dtype, fill):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        dtype = dtype or torch.float32
        nbytes = prod(shape) * dtype.itemsize
        t = self.alloc(max(nbytes, 1))[:nbytes].view(dtype).view(shape)
        if fill is not None:
            t.fill_(fill)
        return t

    def _ranges(self):
        yield from self.blocks
        for name, t in self.inputs.items():
            if isinstance(t, torch.Tensor) and t.numel() and t.is_contiguous():
                n = t.numel() * t.element_size()
                yield t.data_ptr(), n, name, t.reshape(-1).view(torch.uint8)

    def find(self, ptr):
        for base, nbytes, name, buf in self._ranges():
            if base <= ptr < base + nbytes:
                return name, ptr - base, buf
        return None

    def name(self, ptr):
        hit = self.find(ptr)
        return ptr if hit is None else f"{hit[0]}+{hit[1]}"

    def bytes(self, ptr, nbytes):
        _, offset, buf = self.find(ptr)
        return buf[offset : offset + nbytes]

    def copy(self, dst, src, nbytes):
        if nbytes:
            self.bytes(dst, nbytes).copy_(self.bytes(src, nbytes))


class _Stream:
    def __init__(self, h, name):
        self.h, self.name = h, name
        self.cuda_stream = name

    def wait_stream(self, other):
        self.h.log("wait_stream", self.name, other.name)

    def wait_event(self, event):
        self.h.log("wait_event", self.name, event.name)


class _Event:
    def __init__(self, h, name):
        self.h, self.name = h, name

    def record(self, stream):
        self.h.log("event.record", self.name, stream.name)


class _Cuda:
    """`torch.cuda` as `mega_moe` uses it."""

    def __init__(self, h):
        self.h = h

    def Stream(self):
        self.h.streams += 1
        return _Stream(self.h, f"side{self.h.streams}")

    def Event(self):
        self.h.events += 1
        return _Event(self.h, f"event{self.h.events}")

    def current_stream(self):
        return self.h.stream

    @contextlib.contextmanager
    def stream(self, stream):
        saved, self.h.stream = self.h.stream, stream
        try:
            yield
        finally:
            self.h.stream = saved

    def is_current_stream_capturing(self):
        return False

    def current_device(self):
        return 0


class _Torch:
    """`torch` for `mega_moe`: CPU devices, allocations in the pipeline's memory."""

    def __init__(self, h):
        self.h = h
        self.cuda = _Cuda(h)

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, *args, **kwargs):
        return torch.device("cpu")

    def zeros(self, *shape, dtype=None, device=None):
        return self.h.mem.tensor(shape, dtype, 0)

    def ones(self, *shape, dtype=None, device=None):
        return self.h.mem.tensor(shape, dtype, 1)

    def empty(self, *shape, dtype=None, device=None):
        return self.h.mem.tensor(shape, dtype, None)

    def full(self, shape, value, dtype=None, device=None):
        return self.h.mem.tensor((shape,), dtype, value)


class _Comm:
    """A cco communicator: the arena is a CPU buffer, its handle its address."""

    def __init__(self, h):
        self.h = h

    def alloc_mem(self, nbytes):
        buf = self.h.mem.alloc(nbytes, "arena")
        self.h.arena_bytes.append(nbytes)
        return types.SimpleNamespace(
            ptr=buf.data_ptr(), close=lambda: self.h.log("arena.close")
        )

    def register_window(self, ptr, nbytes):
        return types.SimpleNamespace(
            handle=ptr, local_ptr=ptr, close=lambda: self.h.log("window.close")
        )


class _Harness:
    def __init__(self):
        self.mem = _Memory()
        self.launches = []
        self.builds = []
        self.arena_bytes = []
        self.streams = self.events = 0
        self.stream = _Stream(self, "main")
        self.served = []  # grouped_gemm_gfx1250_a8w4 returned a result
        self.nested = 0  # legacy's nested fused_moe calls
        self.v2_launches = 0

    def fresh(self):
        """A new pipeline's memory and stream names."""
        self.mem = _Memory()
        self.streams = self.events = 0
        self.builds = []

    def log(self, *what):
        self.launches.append(tuple(self.describe(w) for w in what))

    def describe(self, x):
        if isinstance(x, torch.Tensor):
            where = "meta" if x.is_meta else self.mem.name(x.data_ptr())
            data = (
                x.contiguous().reshape(-1).view(torch.uint8).tolist()
                if x.numel() and not x.is_meta
                else []
            )
            return ("tensor", where, tuple(x.shape), x.stride(), str(x.dtype), data)
        if isinstance(x, bool) or x is None or isinstance(x, (float, str)):
            return x
        if isinstance(x, enum.Enum):
            return str(x)
        if isinstance(x, int):
            return self.mem.name(x) if x >= 1 << 20 else x
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return (type(x).__name__,) + tuple(
                (f.name, self.describe(getattr(x, f.name)))
                for f in dataclasses.fields(x)
            )
        if isinstance(x, (tuple, list)):
            return tuple(self.describe(v) for v in x)
        if isinstance(x, dict):
            return tuple(sorted((k, self.describe(v)) for k, v in x.items()))
        if isinstance(x, (torch.dtype, _Stream, _Event)):
            return str(getattr(x, "name", x))
        return type(x).__name__


def _flattened(ctx):
    """What legacy's `fused_moe_` keeps of a scatter context across the custom op."""
    return Stage2ScatterContext(
        arena_handle=ctx.arena_handle,
        combine_input_offset=ctx.combine_input_offset,
        slot_stride_bytes=ctx.slot_stride_bytes,
        max_tokens_per_rank=ctx.max_tokens_per_rank,
        world_size=ctx.world_size,
        source_token_map=ctx.source_token_map,
        combine_quant_bits=int(ctx.combine_quant_bits),
    )


@pytest.fixture
def h(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    h = _Harness()
    monkeypatch.setattr(mega_moe, "torch", _Torch(h))
    monkeypatch.setattr(
        mega_moe, "fx", types.SimpleNamespace(Stream=lambda s: f"fx.Stream({s.name})")
    )

    def from_ptr(ptr, shape, dtype):
        nbytes = prod(shape) * dtype.itemsize
        return h.mem.bytes(ptr, nbytes).view(dtype).view(tuple(shape))

    monkeypatch.setattr(mega_moe, "_from_gpu_ptr", from_ptr)
    monkeypatch.setattr(
        grouped,
        "route_counter_buffer",
        lambda E, device: h.mem.tensor((E,), torch.int32, 0),
    )

    def make_dispatch_tdm(**kw):
        h.builds.append(("make_dispatch_tdm", h.describe(kw)))

        def kern(handle, tok, idx, wts, tok_map, dest, bar, total, *rest):
            h.log("dispatch_tdm", kw["block_num"], handle, tok, idx, wts, tok_map, dest)
            h.log("dispatch_tdm", bar, total, *rest)
            sent_scales, ntok = rest[4], rest[7]
            h.mem.copy(
                handle + kw["off_out_tok"],
                tok,
                ntok * kw["hidden_dim"] * kw["hidden_elem_size"],
            )
            h.mem.copy(handle + kw["off_out_idx"], idx, ntok * TOPK * 4)
            h.mem.copy(handle + kw["off_out_wts"], wts, ntok * TOPK * 4)
            if sent_scales and kw["off_out_scales"]:
                h.mem.copy(
                    handle + kw["off_out_scales"], sent_scales, ntok * kw["scale_bytes"]
                )
            h.mem.bytes(total, 4).view(torch.int32)[0] = ntok * TOPK

        return kern

    monkeypatch.setattr(mega_moe, "_make_dispatch_tdm", make_dispatch_tdm)

    def make_reduce(**kw):
        h.builds.append(("make_combine_reduce", h.describe(kw)))

        def kern(comb_inp, out, count, stream):
            h.log("combine_reduce", kw["quant_bits"], comb_inp, out, count, stream)
            h.mem.copy(out, comb_inp, count * MODEL * 2)

        return kern

    def make_sync(**kw):
        h.builds.append(("make_combine_sync", h.describe(kw)))
        return lambda *args: h.log("combine_sync", *args)

    monkeypatch.setattr(mega_moe, "_make_combine_fused_reduce", make_reduce)
    monkeypatch.setattr(mega_moe, "_make_combine_fused_sync", make_sync)

    def compile_plan(**kw):
        h.builds.append(("compile_compact_plan", h.describe(kw)))

        def kern(handle, ids, tok_map, hist, send, masked_m, psum, barrier, *rest):
            h.log("compact_plan", kw["tile_m"], handle, ids, tok_map, hist, send)
            h.log("compact_plan", masked_m, psum, barrier, *rest)
            ntok = rest[1]
            h.mem.copy(tok_map, ids, ntok * TOPK * 4)
            h.mem.bytes(masked_m, 4).view(torch.int32)[0] = ntok
            h.mem.bytes(psum, 4).view(torch.int32)[0] = ntok * TOPK

        return kern

    monkeypatch.setattr(compact_plan, "compile_tdm_compact_plan", compile_plan)

    def quant(payload, hidden, scales, group, shuffle_scale):
        h.log("dispatch_quant", payload, hidden, scales, group, shuffle_scale)
        rows = hidden.view(torch.uint8)
        payload_bytes = payload.view(torch.uint8)
        payload_bytes.copy_(rows[:, : payload_bytes.shape[1]])
        scales.view(torch.uint8).fill_(127)

    monkeypatch.setattr(quant_ops, "dynamic_per_group_scaled_quant", quant)

    def tdm(*args, **kwargs):
        context = kwargs["dispatch_context"]
        scatter = kwargs["stage2_scatter"]
        if context is not None and context.compact_layout:
            scatter = context
        elif scatter is not None:
            scatter = _flattened(scatter)
        h.log("grouped_tdm", args, {**kwargs, "stage2_scatter": scatter})
        x = args[0]
        src = x.contiguous().reshape(-1).view(torch.uint8)
        dst = h.mem.bytes(
            scatter.arena_handle + scatter.combine_input_offset, MAX_TOK * TOPK * 1024
        )
        n = min(src.numel(), dst.numel())
        dst[:n].copy_(src[:n])
        return torch.zeros(x.shape[0], kwargs["model_dim"], dtype=kwargs["dtype"])

    monkeypatch.setattr(grouped, "_grouped_a8w4_tdm_moe", tdm)
    real_entry = grouped.grouped_gemm_gfx1250_a8w4

    def entry(*args, **kwargs):
        out = real_entry(*args, **kwargs)
        h.served.append(out is not None)
        return out

    monkeypatch.setattr(grouped, "grouped_gemm_gfx1250_a8w4", entry)

    def two_stage(*args, **kwargs):
        raise _FellThrough

    monkeypatch.setattr(fm, "get_2stage_cfgs", two_stage)
    real_fused_moe = mega_moe.fused_moe

    def nested(*args, **kwargs):
        h.nested += 1
        return real_fused_moe(*args, **kwargs)

    monkeypatch.setattr(mega_moe, "fused_moe", nested)
    real_launch = GROUPED_P2P.launch

    def launch(*args, **kwargs):
        h.v2_launches += 1
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(GROUPED_P2P, "launch", launch)
    _fake_mori(monkeypatch, h)
    grouped._find_grouped_config.cache_clear()
    with captured_logs():
        grouped._load_grouped_config_rows()  # logs the table merge on first use
    return h


def _fake_mori(monkeypatch, h):
    """The mori dispatch the pipeline drives: a plan per spec, the tokoff-ext
    allocator and the scale-row stride."""

    class EpDispatchPlan:
        def __init__(self, **kw):
            self.kw = kw
            h.builds.append(("mori_plan", h.describe(kw)))

        def bind(self, rank):
            h.builds.append(("mori_bind", rank))

        def launch(self, **kw):
            h.log("mori_dispatch", self.kw["block_num"], kw)
            arena, ntok = self.kw["arena"], kw["num_tokens"]
            row = self.kw["hidden_dim"] * self.kw["dtype"].itemsize
            h.mem.copy(arena.local_ptr("disp_out"), kw["inp_token_buf"], ntok * row)
            h.mem.copy(arena.local_ptr("out_idx"), kw["token_indices"], ntok * TOPK * 4)
            h.mem.copy(arena.local_ptr("out_wts"), kw["weights_buf"], ntok * TOPK * 4)
            if kw["scales_buf"]:
                scale_bytes = self.kw["scale_bytes"]
                h.mem.copy(
                    arena.local_ptr("disp_out_scales"),
                    kw["scales_buf"],
                    ntok * scale_bytes,
                )
            total = h.mem.bytes(kw["total_recv_token_num"], 4).view(torch.int32)
            total[0] += ntok * TOPK

    class TokOffExt:
        def __init__(self, rank, world_size, device):
            self.peers = 4096
            h.builds.append(("tokoff_ext", rank, world_size))

        def close(self):
            h.log("tokoff_ext.close")

    names = (
        "mori",
        "mori.ops",
        "mori.ops.dispatch_combine_v2",
        "mori.ops.dispatch_combine_v2.ep_plans",
        "mori.ops.dispatch_combine_v2.hip_backend",
        "mori.ops.dispatch_combine_v2.hip_tuning_configs",
    )
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["mori.ops.dispatch_combine_v2.hip_tuning_configs"].lookup = (
        lambda *args, **kw: {"schedule": [(4, 16, 4, 0, 0), (None, 32, 8, 0, 0)]}
    )
    sys.modules["mori.ops.dispatch_combine_v2.ep_plans"].EpDispatchPlan = EpDispatchPlan
    backend = sys.modules["mori.ops.dispatch_combine_v2.hip_backend"]
    backend.TokOffExt = TokOffExt
    backend.scale_stride_bytes = lambda n: -(-n // 128) * 128


def _pipeline(h, rank=0, world_size=WORLD, **ctor):
    h.fresh()
    mega = mega_moe.MegaMoEGfx1250(
        communicator=_Comm(h),
        rank=rank,
        world_size=world_size,
        model_dim=MODEL,
        inter_dim=INTER,
        experts=EXPERTS,
        topk=TOPK,
        max_tokens_per_rank=MAX_TOK,
        **ctor,
    )
    return mega, h.mem, list(h.builds)


def _fp4(*shape):
    return torch.zeros(*shape, dtype=torch.uint8).view(dtypes.fp4x2)


def _weights(local=EXPERTS // WORLD, dtype="fp4x2"):
    gen = torch.Generator().manual_seed(1)

    def packed(*shape):
        t = torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
        return t if dtype == "uint8" else t.view(dtypes.fp4x2)

    return {
        "w1": packed(local, 2 * INTER, MODEL // 2),
        "w2": packed(local, MODEL, INTER // 2),
        "w1_scale": torch.full((local, 2 * INTER, MODEL // 32), 127, dtype=torch.uint8),
        "w2_scale": torch.full((local, MODEL, INTER // 32), 127, dtype=torch.uint8),
    }


def _forward_args(step, seed, world_size=WORLD):
    """`forward`'s arguments for one step (`step`: overrides, some by name)."""
    step = dict(step)
    m = step.pop("m", 5)
    local = EXPERTS // world_size
    gen = torch.Generator().manual_seed(seed)
    call = {
        "hidden_states": torch.randn(m, MODEL, generator=gen).to(torch.bfloat16),
        "topk_weights": torch.rand(m, TOPK, generator=gen),
        "topk_ids": torch.randint(0, EXPERTS, (m, TOPK), generator=gen).to(torch.int32),
        **_weights(local, dtype=step.pop("weights", "fp4x2")),
    }
    named = {
        "bias1": lambda: torch.rand(local, 2 * INTER, generator=gen),
        "bias2": lambda: torch.rand(local, MODEL, generator=gen),
        "a1_scale": lambda: torch.full((m, MODEL // 32), 127, dtype=torch.uint8),
        "a2_scale": lambda: torch.ones(1),
        "fp16 hidden": lambda: call["hidden_states"].to(torch.float16),
        "strided ids": lambda: call["topk_ids"].t().contiguous().t(),
        "bad w1": lambda: _fp4(EXPERTS // WORLD, INTER, MODEL // 2),
        "bad weights": lambda: torch.rand(m, TOPK + 1, generator=gen),
    }
    for name, value in step.items():
        if name == "next_topk_ids":
            value = torch.randint(0, EXPERTS, (value, TOPK), generator=gen).to(
                torch.int32
            )
        elif isinstance(value, str) and value in named:
            value = named[value]()
        call[name] = value
    return call


def _run(h, mega, mem, steps, mode, seed=0):
    """Each step's (result, launches, logs, served, nested calls, v2 launches)."""
    results = []
    for i, step in enumerate(steps):
        call = _forward_args(step, seed + i, mega._config.world_size)
        h.mem = mem
        mem.inputs = {k: v for k, v in call.items() if isinstance(v, torch.Tensor)}
        h.launches, h.served = [], []
        h.nested = h.v2_launches = 0
        with moev2_mode(mode), captured_logs() as logs:
            try:
                out = mega.forward(**call)
                result = ("out", h.describe(out))
            except Exception as e:  # noqa: BLE001
                result = ("raised", type(e).__name__, str(e))
        results.append(
            (result, h.launches, list(logs), h.served, h.nested, h.v2_launches)
        )
    return results


# name: (constructor arguments, environment, forward steps, legacy's grouped branch per
# step: True served, False fell through, None raised before).
_STEP = {}
CASES = {
    "flydsl bf16": ({}, {}, [_STEP, _STEP], [True, True]),
    "flydsl bf16 recv_token_bound": (
        {},
        {},
        [{"recv_token_bound": 6}, {"recv_token_bound": 16}],
        [True, True],
    ),
    "recv_token_bound too large": ({}, {}, [{"recv_token_bound": 17}], [None]),
    "recv_token_bound 0": ({}, {}, [{"recv_token_bound": 0}], [None]),
    "flydsl fp8 wire": ({"dispatch_wire": "fp8"}, {}, [_STEP], [True]),
    "flydsl fp8 wire bound": (
        {"dispatch_wire": "fp8"},
        {},
        [{"recv_token_bound": 7}],
        [True],
    ),
    "flydsl fp4 wire": ({"dispatch_wire": "fp4"}, {}, [_STEP], [True]),
    "fp8 wire with a1_scale": (
        {"dispatch_wire": "fp8"},
        {},
        [{"a1_scale": "a1_scale"}],
        [None],
    ),
    "bf16 wire with a1_scale": ({}, {}, [{"a1_scale": "a1_scale"}], [True]),
    "MEGA_DISPATCH_WIRE=fp8": ({}, {"MEGA_DISPATCH_WIRE": "fp8"}, [_STEP], [True]),
    "compact fp8": (
        {"stage1_fused": True, "dispatch_wire": "fp8"},
        {},
        [_STEP, {"recv_token_bound": 4}],
        [True, True],
    ),
    "compact fp4": (
        {"stage1_fused": True, "dispatch_wire": "fp4"},
        {},
        [_STEP],
        [True],
    ),
    "compact next_topk_ids": (
        {"stage1_fused": True, "dispatch_wire": "fp8"},
        {},
        [
            {"next_topk_ids": 4, "next_recv_token_bound": 6},
            {"m": 4, "next_topk_ids": 3},
            {"m": 3, "next_topk_ids": 0},
            {"m": 2},
        ],
        [True, True, True, True],
    ),
    "compact blocks env": (
        {"stage1_fused": True, "dispatch_wire": "fp8"},
        {"AITER_TDM_COMPACT_BLOCKS": "32"},
        [_STEP],
        [True],
    ),
    "next_topk_ids off compact": ({}, {}, [{"next_topk_ids": 3}], [True]),
    "mori bf16": ({"dispatch_backend": "mori"}, {}, [_STEP, _STEP], [True, True]),
    "mori fp8 wire": (
        {"dispatch_backend": "mori", "dispatch_wire": "fp8"},
        {},
        [_STEP, {"recv_token_bound": 9}],
        [True, True],
    ),
    "mori no tokoff-ext": (
        {"dispatch_backend": "mori"},
        {"MORI_EP_TOKOFF_EXT": "0"},
        [_STEP],
        [True],
    ),
    "MEGA_DISPATCH=mori": ({}, {"MEGA_DISPATCH": "mori"}, [_STEP], [True]),
    "mxfp8 combine": (
        {"combine_quant": "mxfp8"},
        {},
        [{"combine_quant": "mxfp8"}, _STEP, {"combine_quant": "none"}],
        [True, True, True],
    ),
    "mxfp4 combine": (
        {"combine_quant": "mxfp4"},
        {},
        [{"combine_quant": "mxfp4"}],
        [True],
    ),
    "combine format not built": ({}, {}, [{"combine_quant": "mxfp8"}], [None]),
    "unknown combine format": ({}, {}, [{"combine_quant": "fp8"}], [None]),
    "MEGA_COMBINE_QUANT=mxfp8": (
        {},
        {"MEGA_COMBINE_QUANT": "mxfp8"},
        [{"combine_quant": "mxfp8"}],
        [True],
    ),
    "swiglu limit": (
        {"activation": ActivationType.Swiglu, "swiglu_limit": 7.0},
        {},
        [_STEP],
        [True],
    ),
    "situv2 betas": (
        {
            "activation": ActivationType.Situv2,
            "situ_beta": torch.tensor(2.0),
            "situ_linear_beta": torch.tensor(3.0),
        },
        {},
        [_STEP],
        [True],
    ),
    "pads": ({"hidden_pad": 64, "intermediate_pad": 32}, {}, [_STEP], [True]),
    "bias and a2_scale": (
        {},
        {},
        [{"bias1": "bias1", "bias2": "bias2", "a2_scale": "a2_scale"}],
        [True],
    ),
    "a8w4 (AITER_FORCE_A8W4)": ({}, {"AITER_FORCE_A8W4": "1"}, [_STEP], [True]),
    "uint8 weights": ({}, {}, [{"weights": "uint8"}], [True]),
    "zero tokens": ({}, {}, [{"m": 0}, _STEP], [True, True]),
    "zero tokens compact": (
        {"stage1_fused": True, "dispatch_wire": "fp8"},
        {},
        [{"m": 0}],
        [True],
    ),
    # Raised by the checks, before the dispatch.
    "fp16 hidden_states": ({}, {}, [{"hidden_states": "fp16 hidden"}], [None]),
    "strided topk_ids": ({}, {}, [{"topk_ids": "strided ids"}], [None]),
    "grouped disabled": ({}, {"AITER_DISABLE_GROUPED_A8W4": "1"}, [_STEP], [None]),
    "no w1_scale": ({}, {}, [{"w1_scale": None}], [None]),
    "w1 shape": ({}, {}, [{"w1": "bad w1"}], [None]),
    "too many tokens": ({}, {}, [{"m": MAX_TOK + 1}], [None]),
    "topk_weights shape": ({}, {}, [{"topk_weights": "bad weights"}], [None]),
    "bf16 weights": (
        {},
        {},
        [
            {
                "w1": torch.zeros(
                    EXPERTS // WORLD, 2 * INTER, MODEL // 2, dtype=torch.bfloat16
                ),
                "w2": torch.zeros(
                    EXPERTS // WORLD, MODEL, INTER // 2, dtype=torch.bfloat16
                ),
            }
        ],
        [None],
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_forward_matches_legacy(h, monkeypatch, name):
    ctor, env, steps, served = CASES[name]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with simulated_arch("gfx1250"):
        legacy_mega, legacy_mem, legacy_builds = _pipeline(h, **ctor)
        legacy = _run(h, legacy_mega, legacy_mem, steps, "off")
        v2_mega, v2_mem, v2_builds = _pipeline(h, **ctor)
        v2 = _run(h, v2_mega, v2_mem, steps, path.NAME)
    assert v2_builds == legacy_builds
    for i, (mine, theirs) in enumerate(zip(v2, legacy)):
        result, launches, logs, legacy_served, nested, launched = theirs
        assert launched == 0
        if served[i] is None:
            assert result[0] == "raised" and legacy_served == [], (i, result)
        else:
            assert legacy_served == [served[i]], (i, legacy_served, result)
        assert mine[0] == result, (i, mine[0], result)
        assert mine[1] == launches, f"step {i}: launches differ"
        assert mine[2] == logs, (i, mine[2], logs)
        assert mine[3] == legacy_served
        # v2 serves the calls the grouped branch serves, without the nested call.
        assert mine[5] == int(bool(served[i])), (i, mine[5])
        assert mine[4] == (0 if served[i] else nested), (i, mine[4], nested)


def test_off_gfx1250_stays_legacy(h):
    """On another device the nested call falls through to the 2-stage dispatch;
    v2 does not claim it."""
    with simulated_arch("gfx1250"):
        legacy_mega, legacy_mem, _ = _pipeline(h)
        v2_mega, v2_mem, _ = _pipeline(h)
    with simulated_arch("gfx950"):
        legacy = _run(h, legacy_mega, legacy_mem, [_STEP], "off")
        v2 = _run(h, v2_mega, v2_mem, [_STEP], path.NAME)
    assert legacy[0][0][:2] == ("raised", "_FellThrough")
    assert legacy[0][3] == [False] and legacy[0][4] == 1
    assert v2 == legacy


def test_claims_and_decline_reasons(h):
    with simulated_arch("gfx1250"):
        mega, _, _ = _pipeline(h)
        call = {"mega": mega, **_forward_args(_STEP, 0)}
        call.update(
            {k: None for k in path.FORWARD_ARGS if k not in call}, combine_quant=None
        )
        assert set(call) == {"mega", *path.FORWARD_ARGS}
        assert path.claims(call)
        assert path.grouped_unsupported(mega, call) is None
        assert not path.claims({**call, "w1": None})
        assert not path.claims({**call, "w1": _fp4(1, 2, 3)})
        bf16 = {
            k: v.view(torch.bfloat16)
            for k, v in _weights(dtype="uint8").items()
            if k in ("w1", "w2")
        }
        assert path.grouped_unsupported(mega, {**call, **bf16}) is not None
    with simulated_arch("gfx950"):
        assert not path.claims(call)


def test_hook_passes_every_forward_argument(h, monkeypatch):
    seen = []

    def run(call):
        seen.append(call)
        return "v2 result"

    def route_to_v2(call, entry):
        assert entry == "mega_gfx1250"
        return types.SimpleNamespace(run=run)

    monkeypatch.setattr(paths, "route_to_v2", route_to_v2)
    with simulated_arch("gfx1250"):
        mega, mem, _ = _pipeline(h)
        call = _forward_args(_STEP, 0)
        mem.inputs = dict(call)
        with moev2_mode(path.NAME):
            assert mega.forward(**call) == "v2 result"
    assert set(seen[0]) == {"mega", *path.FORWARD_ARGS}
    assert seen[0]["mega"] is mega and seen[0]["recv_token_bound"] is None


# The engine template.


def _template(h, **kw):
    return path.MegaMoEGfx1250Template(
        communicator=_Comm(h),
        world_size=1,
        model_dim=MODEL,
        inter_dim=INTER,
        experts=EXPERTS,
        topk=TOPK,
        max_tokens_per_rank=MAX_TOK,
        **kw,
    )


def _engine(template, capture_max_tokens=0):
    group = types.SimpleNamespace(
        cpu_group=None, rank_in_group=0, world_size=1, device=None
    )
    spec = ParallelSpec("ep", 1, "a2a", "flydsl", MAX_TOK, EXPERTS)
    return engine_mod.MoeEngine.create(
        template,
        max_tokens=MAX_TOK,
        parallel=spec,
        comm=group,
        capture_max_tokens=capture_max_tokens,
    )


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"dispatch_wire": "fp8", "combine_quant": {4: "mxfp8"}},
        {"stage1_fused": True, "dispatch_wire": "fp4", "recv_token_bound": {8: 6}},
        {"dispatch_backend": "mori", "combine_quant": "mxfp4"},
    ],
)
def test_engine_matches_legacy(h, kw):
    """The engine's pipeline is built and runs as a legacy instance with the same
    arguments: the same kernels, arena, launches and outputs per step."""
    template = _template(h, **kw)
    ctor = {
        k: v for k, v in kw.items() if k not in ("combine_quant", "recv_token_bound")
    }
    combine = kw.get("combine_quant", "none")
    built = combine if isinstance(combine, str) else next(iter(combine.values()))
    bounds = kw.get("recv_token_bound", {})
    with simulated_arch("gfx1250"):
        h.fresh()
        eng = _engine(template)
        need = template.resolve(1, False).stateful[0].workspace_bytes()
        eng.prepare()
        engine_builds, engine_mem = list(h.builds), h.mem
        assert eng.memory() == need == MemoryNeed(symmetric=h.arena_bytes[-1])
        mega = template._dispatch.mega
        legacy_mega, legacy_mem, legacy_builds = _pipeline(
            h, world_size=1, combine_quant=built, **ctor
        )
        assert engine_builds == legacy_builds
        assert "comm.dispatch_gfx1250" in eng.explain()
        for i, m in enumerate((1, 4, 7, 0)):
            bucket = engine_mod.bucket_of(max(m, 1))
            quant = combine if isinstance(combine, str) else combine.get(bucket, "none")
            step = {
                "m": m,
                "combine_quant": quant,
                "recv_token_bound": bounds.get(bucket),
            }
            call = _forward_args(step, i, world_size=1)
            (theirs,) = _run(h, legacy_mega, legacy_mem, [step], "off", seed=i)
            h.mem = engine_mem
            engine_mem.inputs = {
                k: v for k, v in call.items() if isinstance(v, torch.Tensor)
            }
            h.launches = []
            weights = engine_mod.MoeWeights(
                call["w1"], call["w2"], call["w1_scale"], call["w2_scale"]
            )
            out = eng.forward(
                call["hidden_states"],
                call["topk_ids"],
                call["topk_weights"],
                weights=weights,
                step=engine_mod.StepInfo(max(m, 1)),
            )
            assert ("out", h.describe(out)) == theirs[0], i
            assert h.launches == theirs[1], f"step {i}: launches differ"
        assert template._dispatch.mega is mega
        h.launches = []
        eng.release()
        assert ("arena.close",) in h.launches or ("window.close",) in h.launches


def test_engine_raises_for_a_call_the_grouped_branch_does_not_serve(h):
    template = _template(h)
    with simulated_arch("gfx1250"):
        h.fresh()
        eng = _engine(template)
        eng.prepare()
    call = _forward_args(_STEP, 0, world_size=1)
    h.launches = []
    with simulated_arch("gfx950"), pytest.raises(
        NotImplementedError, match="does not serve"
    ):
        eng.forward(
            call["hidden_states"],
            call["topk_ids"],
            call["topk_weights"],
            weights=engine_mod.MoeWeights(
                call["w1"], call["w2"], call["w1_scale"], call["w2_scale"]
            ),
            step=engine_mod.StepInfo(5),
        )
    assert h.launches == []  # before the dispatch


def test_engine_capture_warms_the_compact_plans(h):
    template = _template(
        h, stage1_fused=True, dispatch_wire="fp8", recv_token_bound={2: 2}
    )
    with simulated_arch("gfx1250"):
        h.fresh()
        eng = _engine(template, capture_max_tokens=4)
        eng.prepare()
        assert eng.state == engine_mod.EngineState.PREPARED
        mega = template._dispatch.mega
        warmed = []
        real = mega.warmup_compact_plan

        def warmup(**kw):
            warmed.append(kw)
            return real(**kw)

        mega.warmup_compact_plan = warmup
        eng.capture()
    assert eng.state == engine_mod.EngineState.READY
    # Bucket 1: count 1; bucket 2: its bound; bucket 4: counts 3 and 4.
    assert warmed == [
        {"token_count": 1},
        {"recv_token_bound": 2},
        {"token_count": 3},
        {"token_count": 4},
    ]
    assert _template(h).resolve(1, True).warmup is None  # token-major: no plan


def test_template_plans(h):
    template = _template(h, dispatch_wire="fp8", combine_quant={4: "mxfp8"})
    plan = template.resolve(4, False)
    assert plan.description.startswith(
        "comm.dispatch_gfx1250(dispatch_backend=flydsl, dispatch_wire=fp8, "
        "layout=token_major) | grouped_gfx1250(combine_quant=mxfp8) | "
        "comm.combine_gfx1250(combine_quant=mxfp8)"
    )
    assert "combine_quant=none" in template.resolve(1, False).description
    assert plan.hints == {"next_topk_ids"}
    assert plan.output_lifetime == "until_next_forward"
    assert plan.warmup is None and plan.side_inputs == frozenset()
    assert template.resolve(1, False).identity == plan.identity
    other = _template(h, dispatch_wire="fp4").resolve(4, False)
    assert other.identity != plan.identity
    bound = _template(
        h, dispatch_wire="fp8", combine_quant={4: "mxfp8"}, recv_token_bound={4: 6}
    )
    assert "grouped_gfx1250(combine_quant=mxfp8, recv_token_bound=6)" in (
        bound.resolve(4, False).description
    )
    assert bound.resolve(1, False).description == template.resolve(1, False).description
    with pytest.raises(ValueError, match="one quantized combine format"):
        _template(h, combine_quant={1: "mxfp8", 2: "mxfp4"})
    with pytest.raises(ValueError, match="not in"):
        _template(h, combine_quant="fp8")
    with pytest.raises(ValueError, match="not per-call inputs"):
        _template(h, side_inputs={"bias1"})
    with pytest.raises(ValueError, match="divisible"):
        path.MegaMoEGfx1250Template(
            communicator=None,
            world_size=3,
            model_dim=MODEL,
            inter_dim=INTER,
            experts=EXPERTS,
            topk=TOPK,
            max_tokens_per_rank=MAX_TOK,
        )


def test_template_dispatch_defaults_read_the_environment(h, monkeypatch):
    monkeypatch.setenv("MEGA_DISPATCH", "mori")
    monkeypatch.setenv("MEGA_DISPATCH_WIRE", "fp8")
    monkeypatch.setenv("MEGA_COMBINE_QUANT", "mxfp4")
    spec = _template(h).spec()
    assert (spec.dispatch_backend, spec.dispatch_wire, spec.combine_quant) == (
        "mori",
        "fp8",
        "mxfp4",
    )
    assert _template(h, combine_quant="mxfp8").spec().combine_quant == "mxfp8"


def test_prepare_checks_the_group(h):
    segment = CommDispatchGfx1250.for_spec(_template(h).spec())
    ctx = engine_mod.EngineContext(None, 0, 2, None, None)
    with pytest.raises(ValueError, match="1 ranks"):
        segment.prepare(ctx)
    adopted = CommDispatchGfx1250.adopt(object())
    assert adopted.workspace_bytes() == MemoryNeed()


def test_path_registration():
    assert path.PATH.entry == "mega_gfx1250" and not path.PATH.graduated
    assert paths.registered()[path.NAME] is path.PATH
    assert GROUPED_P2P.covers[1].name == "GEMM2" and GROUPED_P2P.p2p
