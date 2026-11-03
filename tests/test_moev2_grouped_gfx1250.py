# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `grouped_gfx1250` against legacy's grouped gfx1250 branch, call by call.

The branch runs before `get_2stage_cfgs`, so the dispatch snapshot cannot see it. On
CPU, as gfx1250 (`arch_sim`), each call runs through legacy `fused_moe` and through v2
with the TDM kernel entry (`_grouped_a8w4_tdm_moe`) stubbed and `get_2stage_cfgs`
raising: the same outcome (output, exception, or the fall-through to the 2-stage
dispatch), the same kernel-entry arguments and log lines, and v2 runs exactly the
calls legacy's `grouped_gemm_gfx1250_a8w4` serves (except the ones it leaves to
legacy: `stage2_scatter`, a FlyDSL dispatch context, a device other than gfx1250).
"""

import dataclasses

import pytest
import torch
from moe_dispatch.arch_sim import simulated_arch
from moe_dispatch.differential import captured_logs, moev2_mode

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.moev2 import api, paths
from aiter.moev2.migration import grouped_gfx1250 as path
from aiter.moev2.segments.grouped_gfx1250 import GROUPED
from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext

_ENV = (
    "AITER_USE_GROUPED_GEMM",
    "AITER_DISABLE_GROUPED_A8W4",
    "AITER_FORCE_GFX1250",
    "AITER_FORCE_A8W4",
    "AITER_SITUV2_A8W4",
    "AITER_SITUV2_A4W4",
    "GPU_ARCHS",
    "TARGET_ARCH",
    "AITER_GPU_ARCHS",
    "AITER_MOEV2",
)
E, TOPK, MODEL, INTER = 8, 2, 256, 128


class _FellThrough(Exception):
    """Legacy reached `get_2stage_cfgs`: the grouped branch returned None."""


class _Recorder:
    def __init__(self):
        self.tdm = []  # (args, kwargs) of each kernel-entry call
        self.served = []  # grouped_gemm_gfx1250_a8w4 returned a tensor
        self.v2_launches = 0

    def clear(self):
        self.tdm.clear()
        self.served.clear()
        self.v2_launches = 0


@pytest.fixture
def rec(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    r = _Recorder()

    def tdm(*args, **kwargs):
        r.tdm.append((args, kwargs))
        return torch.full(
            (args[0].shape[0], kwargs["model_dim"]), 0.5, dtype=kwargs["dtype"]
        )

    def two_stage(*args, **kwargs):
        raise _FellThrough

    real = grouped.grouped_gemm_gfx1250_a8w4

    def grouped_entry(*args, **kwargs):
        out = real(*args, **kwargs)
        r.served.append(out is not None)
        return out

    real_launch = GROUPED.launch

    def launch(*args, **kwargs):
        r.v2_launches += 1
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(grouped, "_grouped_a8w4_tdm_moe", tdm)
    monkeypatch.setattr(grouped, "grouped_gemm_gfx1250_a8w4", grouped_entry)
    monkeypatch.setattr(fm, "get_2stage_cfgs", two_stage)
    monkeypatch.setattr(GROUPED, "launch", launch)
    grouped._find_grouped_config.cache_clear()
    with captured_logs():
        grouped._load_grouped_config_rows()  # logs the table merge on first use
    return r


def _fp4(*shape):
    return torch.zeros(*shape, dtype=torch.uint8).view(dtypes.fp4x2)


def _call(m=5, g1u1=True, weights="fp4x2", scales=True, **kw):
    gen = torch.Generator().manual_seed(m)
    rows = 2 * INTER if g1u1 else INTER
    w1, w2 = _fp4(E, rows, MODEL // 2), _fp4(E, MODEL, INTER // 2)
    if weights == "uint8":
        w1, w2 = w1.view(torch.uint8), w2.view(torch.uint8)
    call = {
        "hidden_states": torch.randn(m, MODEL, generator=gen).to(torch.bfloat16),
        "w1": w1,
        "w2": w2,
        "topk_weight": torch.rand(m, TOPK, generator=gen),
        "topk_ids": torch.randint(0, E, (m, TOPK), generator=gen, dtype=torch.int32),
        "activation": ActivationType.Swiglu,
        "quant_type": QuantType.per_1x32,
        "gate_mode": "interleave",
    }
    if scales:
        call["w1_scale"] = torch.full((E, rows, MODEL // 32), 127, dtype=torch.uint8)
        call["w2_scale"] = torch.full((E, MODEL, INTER // 32), 127, dtype=torch.uint8)
    call.update(kw)
    return call


def _scatter(call):
    m = call["topk_ids"].shape[0]
    return Stage2ScatterContext(
        arena_handle=0,
        combine_input_offset=0,
        slot_stride_bytes=1024,
        max_tokens_per_rank=max(m, 1),
        world_size=1,
        source_token_map=torch.zeros(m, TOPK, dtype=torch.int32),
    )


def _mask():
    mask = torch.zeros(2 * E, dtype=torch.int32)
    mask[:E] = 1
    return mask


# name: (call arguments, environment, served by legacy's grouped branch)
CASES = {
    "a4w4 swiglu": ({}, {}, True),
    "a8w4 (AITER_FORCE_A8W4)": ({}, {"AITER_FORCE_A8W4": "1"}, True),
    "quant_dtype_a fp8": ({"quant_dtype_a": dtypes.fp8}, {}, True),
    "quant_dtype_a bf16": ({"quant_dtype_a": dtypes.bf16}, {}, False),
    "silu": ({"activation": ActivationType.Silu}, {}, True),
    "situv2 betas": (
        {"activation": ActivationType.Situv2, "beta": 2.0, "linear_beta": 3.0},
        {},
        True,
    ),
    "gelu": ({"activation": ActivationType.Gelu}, {}, False),
    "uint8 weights": ({"weights": "uint8"}, {}, True),
    "no weight scales": ({"scales": False}, {}, False),
    "g1u0": ({"g1u1": False}, {}, False),
    "per_Token": ({"quant_type": QuantType.per_Token}, {}, False),
    "grouped disabled": ({}, {"AITER_DISABLE_GROUPED_A8W4": "1"}, False),
    "expert_mask": ({"expert_mask": "mask"}, {}, True),
    "expert_mask + doweight_stage1": (
        {"expert_mask": "mask", "doweight_stage1": True},
        {},
        False,
    ),
    "doweight_stage1": ({"doweight_stage1": True}, {}, True),
    "bias": ({"bias1": "bias1", "bias2": "bias2"}, {}, True),
    "pads": ({"hidden_pad": 64, "intermediate_pad": 32}, {}, True),
    "swiglu_limit": ({"swiglu_limit": 7.0}, {}, True),
    "num_local_tokens": ({"num_local_tokens": "nlt"}, {}, True),
    "a1_scale": ({"a1_scale": "a1_scale"}, {}, True),
    "block_size_M": ({"block_size_M": 32}, {}, True),
    "zero tokens": ({"m": 0}, {}, True),
    "zero tokens, expert_mask + doweight_stage1": (
        {"m": 0, "expert_mask": "mask", "doweight_stage1": True},
        {},
        True,
    ),
    "output buffer": ({"output": "output"}, {}, True),
    "fp16 out": ({"dtype": dtypes.fp16}, {}, True),
    "quant_type_a mismatch": ({"quant_type_a": QuantType.per_Token}, {}, None),
    "separated": ({"gate_mode": "separated"}, {}, None),
    "bad output buffer": ({"output": "bad output"}, {}, None),
    # Left to legacy.
    "stage2_scatter": ({"stage2_scatter": "scatter"}, {}, True),
    "dispatch context": ({"context": True}, {}, True),
    "gfx950, GPU_ARCHS names gfx1250": (
        {"arch": "gfx950", "quant_dtype_a": dtypes.fp8},
        {"AITER_USE_GROUPED_GEMM": "1", "GPU_ARCHS": "gfx950;gfx1250"},
        True,
    ),
}
LEGACY_ONLY = {"stage2_scatter", "dispatch context", "gfx950, GPU_ARCHS names gfx1250"}


def _materialize(kw):
    kw = dict(kw)
    call = _call(
        **{k: kw.pop(k) for k in ("m", "g1u1", "weights", "scales") if k in kw}
    )
    m = call["topk_ids"].shape[0]
    tensors = {
        "mask": _mask(),
        "bias1": torch.zeros(E, 2 * INTER),
        "bias2": torch.zeros(E, MODEL),
        "nlt": torch.tensor([m], dtype=torch.int32),
        "a1_scale": torch.full((m, MODEL // 32), 127, dtype=torch.uint8),
        "output": torch.empty(m, MODEL, dtype=torch.bfloat16),
        "bad output": torch.empty(m, MODEL + 1, dtype=torch.bfloat16),
        "scatter": _scatter(call),
    }
    for name, value in kw.items():
        call[name] = tensors.get(value, value) if isinstance(value, str) else value
    return call


def _problem(kw):
    """The `MoeProblem` of the `fused_moe` call `_materialize(kw)`."""
    import inspect

    defaults = {
        name: p.default
        for name, p in inspect.signature(api.fused_moe).parameters.items()
    }
    return api.build_problem(api.op_call({**defaults, **_materialize(kw)}))


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and (
                a is b
                or (
                    a.shape == b.shape
                    and a.dtype == b.dtype
                    and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                )
            )
        )
    if isinstance(a, (tuple, list)):
        return (
            isinstance(b, (tuple, list))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    return type(a) is type(b) and a == b


def _outcome(rec, call, context):
    rec.clear()
    ctx = _scatter(call) if context else None
    with captured_logs() as logs:
        grouped.set_flydsl_dispatch_context(ctx)
        try:
            out = fm.fused_moe(**call)
            result = ("out", out, out is call.get("output"))
        except _FellThrough:
            result = ("fell through",)
        except Exception as e:  # noqa: BLE001
            result = ("raised", type(e).__name__, str(e))
        finally:
            grouped.set_flydsl_dispatch_context(None)
    return result, list(rec.tdm), list(rec.served), rec.v2_launches, list(logs)


@pytest.mark.parametrize("name", CASES)
def test_call_matches_legacy(rec, monkeypatch, name):
    kw, env, served = CASES[name]
    kw = dict(kw)
    arch = kw.pop("arch", "gfx1250")
    context = kw.pop("context", False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    call = _materialize(kw)
    with simulated_arch(arch):
        legacy = _outcome(rec, call, context)
        with moev2_mode(path.NAME):
            v2 = _outcome(rec, call, context)
    result, tdm, legacy_served, legacy_launches, logs = legacy
    assert legacy_launches == 0
    if served is None:
        assert result[0] == "raised", result
    else:
        assert legacy_served == [served], (legacy_served, result)
    assert _same(v2[0], result), (v2[0], result)
    assert _same(v2[1], tdm), "kernel-entry arguments differ"
    assert v2[2] == legacy_served
    assert v2[4] == logs
    assert v2[3] == int(bool(served) and name not in LEGACY_ONLY)


def test_v2_api_runs_a_served_call(rec):
    call = _materialize({})
    with simulated_arch("gfx1250"):
        out = api.fused_moe(**call)
        assert rec.v2_launches == 1 and rec.served == [True]
        assert torch.equal(out, torch.full_like(out, 0.5))
        with pytest.raises(NotImplementedError):
            api.fused_moe(**_materialize({"activation": ActivationType.Gelu}))


@pytest.mark.parametrize(
    "name", [n for n, (_, _, s) in CASES.items() if s and n not in LEGACY_ONLY]
)
def test_segment_caps_accept_served_calls(monkeypatch, name):
    kw, env, _ = CASES[name]
    if kw.get("weights") == "uint8" or kw.get("m") == 0:
        pytest.skip("no MoeProblem: uint8 weights / zero tokens")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with simulated_arch("gfx1250"):
        problem = _problem(kw)
        assert GROUPED.reject_reason(problem) is None, problem


@pytest.mark.parametrize(
    "kw",
    [
        {"activation": ActivationType.Gelu},
        {"gate_mode": "separated"},
        {"g1u1": False},
        {"quant_dtype_a": dtypes.bf16},
    ],
)
def test_segment_caps_reject(kw):
    with simulated_arch("gfx1250"):
        problem = _problem(kw)
        assert GROUPED.reject_reason(problem) is not None


GPU_CASES = {
    "a4w4 swiglu": ({}, {}),
    "a8w4 (AITER_FORCE_A8W4)": ({}, {"AITER_FORCE_A8W4": "1"}),
    "silu": ({"activation": ActivationType.Silu}, {}),
    "situv2 betas": (
        {"activation": ActivationType.Situv2, "beta": 2.0, "linear_beta": 3.0},
        {},
    ),
    "expert_mask": ({"expert_mask": "mask"}, {}),
    "bias": ({"bias1": "bias1", "bias2": "bias2"}, {}),
    "output buffer": ({"output": "output"}, {}),
    "zero tokens": ({"m": 0}, {}),
    "64 tokens": ({"m": 64}, {}),
}


def _on_gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx1250"


@pytest.mark.skipif(not _on_gfx1250(), reason="needs a gfx1250 GPU")
@pytest.mark.parametrize("name", GPU_CASES)
def test_gpu_outputs_match_legacy(monkeypatch, name):
    """The GPU gate: the real kernels, legacy and v2 outputs bitwise equal."""
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    kw, env = GPU_CASES[name]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    gen = torch.Generator().manual_seed(7)
    call = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in _materialize(kw).items()
    }
    call["topk_weight"] = call["topk_weight"].to(torch.bfloat16)
    # As op_tests/flydsl_tests/test_flydsl_grouped_gemm.py prepares them: GUGU w1,
    # scales folded to the gfx1250 layout.
    for k, interleave in (("w1", True), ("w2", False)):
        gate_up = {"is_guinterleave": True, "gate_up": True} if interleave else {}
        raw = torch.randint(0, 256, call[k].shape, generator=gen, dtype=torch.uint8)
        weight = moe_shuffle_weight(raw.cuda(), experts_cnt=E, **gate_up)
        call[k] = weight.view(call[k].dtype)
        scale = torch.randint(
            118, 127, call[f"{k}_scale"].shape, generator=gen, dtype=torch.uint8
        )
        call[f"{k}_scale"] = moe_shuffle_scale(
            scale.cuda().contiguous(), experts_cnt=E, **gate_up
        )
    launches = []
    real_launch = GROUPED.launch

    def launch(*args, **kwargs):
        launches.append(1)
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(GROUPED, "launch", launch)
    outs, logs = [], []
    for mode in ("off", "off", path.NAME):
        if "output" in kw:
            call["output"] = torch.full_like(call["output"], float("nan"))
        with moev2_mode(mode), captured_logs() as records:
            out = fm.fused_moe(**call)
            torch.cuda.synchronize()
        outs.append(out.clone())
        logs.append(list(records))
    assert launches == [1]
    assert torch.equal(
        outs[0].view(torch.int16), outs[1].view(torch.int16)
    ), "legacy disagrees with itself"
    assert torch.equal(outs[2].view(torch.int16), outs[0].view(torch.int16))
    assert logs[2] == logs[1]


def test_path_registration():
    assert path.PATH.entry == "op" and not path.PATH.graduated
    assert paths.registered()[path.NAME] is path.PATH
    assert GROUPED.covers[0].name == "SORT" and GROUPED.covers[1].name == "REDUCE"
