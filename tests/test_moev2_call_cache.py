# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The aiter.moev2 call cache (`aiter/moev2/dispatch/call_cache.py`) on the CPU.

The signature classifies every entry parameter, reads every environment variable and
state the decision reads per call, and calls that share a signature get one decision;
a cache hit skips the decision only, and AITER_MOEV2_CALL_CACHE=check catches a
signature that misses an input.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import contextlib
import dataclasses
import inspect
import itertools
import logging
import os
import pathlib
import sys

import pytest
import torch

import aiter.fused_moe as legacy
from aiter import ActivationType, QuantType, dtypes
from aiter.fhmoe import fhmoe_
from aiter.moev2 import envs, paths
from aiter.moev2.dispatch import call_cache, tuned
from aiter.moev2.execute import executor
from aiter.ops.flydsl.moe_common import GateMode
from aiter.presorted_moe import register_presorted_moe

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MODE = call_cache.AITER_MOEV2_CALL_CACHE
TOKENS = (1, 31, 32, 33, 200, 255, 256, 257, 1024, 5000)


@pytest.fixture(autouse=True)
def _all_paths_on_gfx950(monkeypatch):
    from moe_dispatch import arch_sim

    monkeypatch.setenv(envs.AITER_MOEV2, "on")
    monkeypatch.setenv(MODE, "check")
    for name in ("AITER_ONLINE_TUNE", "AITER_MOE_EXPLAIN", "AITER_USE_GROUPED_GEMM"):
        monkeypatch.delenv(name, raising=False)
    with arch_sim.simulated_arch("gfx950"):
        tuned.clear_cache()
        yield
    monkeypatch.undo()
    tuned.clear_cache()


def _with_defaults(function, **arguments):
    call = {
        name: param.default
        for name, param in inspect.signature(function).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    call.update(arguments)
    return call


def _weights(w_dtype, experts, model_dim, inter_dim, shuffled=True):
    packed = {dtypes.fp4x2: 2, dtypes.i4x2: 2, torch.uint32: 8}.get(w_dtype, 1)
    w1 = torch.empty(experts, 2 * inter_dim, model_dim // packed, dtype=w_dtype)
    w2 = torch.empty(experts, model_dim, inter_dim // packed, dtype=w_dtype)
    w1.is_shuffled = w2.is_shuffled = shuffled
    return w1, w2


def _op_call(
    quant_type,
    w_dtype,
    tokens=64,
    activation=ActivationType.Silu,
    gate_mode=GateMode.SEPARATED,
    shape=(256, 128, 8, 2),
    **changes,
):
    model_dim, inter_dim, experts, topk = shape
    w1, w2 = _weights(w_dtype, experts, model_dim, inter_dim)
    scaled = quant_type != QuantType.No
    call = _with_defaults(
        legacy.fused_moe_,
        hidden_states=torch.empty(tokens, model_dim, dtype=dtypes.bf16),
        w1=w1,
        w2=w2,
        topk_weight=torch.empty(tokens, topk),
        topk_ids=torch.zeros(tokens, topk, dtype=torch.int32),
        w1_scale=torch.empty(1) if scaled else None,
        w2_scale=torch.empty(1) if scaled else None,
        activation=activation.value,
        quant_type=quant_type.value,
        gate_mode=gate_mode.value,
    )
    call.update(changes)
    return call


def _fhmoe_call(tokens=64, gate_mode=GateMode.INTERLEAVE, w_dtype=None):
    experts, model_dim, inter_dim, topk = 9, 256, 128, 3
    w1, w2 = _weights(w_dtype or dtypes.fp4x2, experts, model_dim, inter_dim)
    return _with_defaults(
        fhmoe_,
        hidden_states=torch.empty(tokens, model_dim, dtype=dtypes.bf16),
        w1=w1,
        w2=w2,
        topk_weight=torch.empty(tokens, topk),
        topk_ids=torch.zeros(tokens, topk, dtype=torch.int32),
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        activation=ActivationType.Silu.value,
        quant_type=QuantType.per_1x32.value,
        gate_mode=gate_mode.value,
        shared_expert_id=experts - 1,
    )


def _calls(tokens=64):
    """`(entry, call)`: calls each local path and FHMoE claims, and some no path does."""
    fp4, fp8, i8 = dtypes.fp4x2, dtypes.fp8, dtypes.i8
    swiglu = ActivationType.Swiglu
    op = [
        _op_call(QuantType.per_Tensor, fp8, tokens),
        _op_call(QuantType.per_Tensor, i8, tokens),
        _op_call(QuantType.No, dtypes.bf16, tokens),
        _op_call(QuantType.per_Token, fp8, tokens),
        _op_call(QuantType.per_Token, i8, tokens),
        _op_call(QuantType.per_Token, torch.uint32, tokens),
        _op_call(QuantType.per_1x128, fp8, tokens, shape=(256, 256, 8, 2)),
        _op_call(QuantType.per_1x32, fp4, tokens),
        _op_call(QuantType.per_1x32, fp4, tokens, activation=swiglu),
        _op_call(QuantType.per_1x32, fp4, tokens, activation=ActivationType.Situv2),
        _op_call(QuantType.per_1x32, fp4, tokens, quant_dtype_a=fp8),
        _op_call(
            QuantType.per_1x32,
            fp4,
            tokens,
            swiglu,
            GateMode.INTERLEAVE,
            block_size_M=32,
        ),
        _op_call(QuantType.per_1x32, dtypes.i4x2, tokens),
        _op_call(QuantType.per_1x32, fp8, tokens),
        _op_call(QuantType.per_1x32, fp8, tokens, activation=swiglu, swiglu_limit=7.0),
        _op_call(QuantType.per_Tensor, fp8, tokens, bias1=torch.empty(1)),
        _op_call(QuantType.per_Tensor, fp8, tokens, herd_topk=4),
    ]
    fhmoe = [
        _fhmoe_call(tokens),
        _fhmoe_call(tokens, GateMode.SEPARATED),
        _fhmoe_call(tokens, w_dtype=dtypes.fp8),
    ]
    return [("op", c) for c in op] + [("fhmoe", c) for c in fhmoe]


def _forced(path, call):
    if path.name != "fhmoe":
        return (None, None)
    from aiter.moev2.migration import fhmoe

    gate_mode = GateMode(call["gate_mode"])
    q_dtype_a = dtypes.fp8 if gate_mode == GateMode.INTERLEAVE else dtypes.fp4x2
    return q_dtype_a, fhmoe.config_file(call, gate_mode)


def _local(path):
    import importlib

    return importlib.import_module(f"aiter.moev2.migration.{path.name}").LOCAL


def _decide(entry, call):
    """`(path, decision)` as `route_to_v2` and `LocalDispatch.run` decide afresh."""
    path = paths._route(call, entry)
    if path is None:
        return None, None
    if path.name == "fhmoe":
        from aiter.moev2.migration import fhmoe

        call = {**call, **fhmoe._OP_DEFAULTS}
    try:
        decision = _local(path)._decide(call, _forced(path, call))
    except Exception as error:  # noqa: BLE001 - raising is a decision too
        decision = (type(error), str(error))
    return path, decision


def test_every_entry_parameter_has_one_class():
    assert set(call_cache.ENTRY_FUNCTIONS) == set(paths.ENTRIES)
    assert set(call_cache.PARAMS.values()) <= set(call_cache.KINDS)
    seen = set()
    for entry in paths.ENTRIES:
        params = call_cache.entry_params(entry)
        assert not set(params) - set(call_cache.PARAMS), entry
        names = call_cache._projection(entry).names
        assert sorted(n for kind in names.values() for n in kind) == sorted(params)
        seen.update(params)
    assert not set(call_cache.PARAMS) - seen, "PARAMS names no entry parameter"


def _changed(name, value, kind):
    if kind == call_cache.EXEC:
        return object()
    if kind == call_cache.PRESENT:
        return torch.empty(1) if value is None else None
    if kind == call_cache.WEIGHT:
        flipped = torch.empty_like(value)
        flipped.is_shuffled = not value.is_shuffled
        return flipped
    if kind == call_cache.ROWS:
        return torch.empty(value.shape[0], value.shape[1] + 1, dtype=value.dtype)
    if isinstance(value, bool):
        return not value
    if isinstance(value, torch.dtype):
        return dtypes.fp16 if value != dtypes.fp16 else dtypes.bf16
    if name == "gate_mode":
        interleave = GateMode(value) == GateMode.INTERLEAVE
        return (GateMode.SEPARATED if interleave else GateMode.INTERLEAVE).value
    return 1 if value is None else value + 1


@pytest.mark.parametrize("entry", ["op", "fhmoe"])
def test_the_signature_keys_each_parameter_by_its_class(entry):
    call = dict(_calls()[0 if entry == "op" else -1][1])
    key = call_cache.signature(call, entry)
    assert key is not None
    for name in call_cache.entry_params(entry):
        kind = call_cache.PARAMS[name]
        changed = {**call, name: _changed(name, call[name], kind)}
        assert (call_cache.signature(changed, entry) == key) == (
            kind == call_cache.EXEC
        ), (name, kind)
    rows = {"hidden_states", "topk_ids"}
    for tokens, same in ((33, True), (65, False)):
        resized = {
            **call,
            **{
                n: torch.empty(tokens, *call[n].shape[1:], dtype=call[n].dtype)
                for n in rows
            },
        }
        assert (call_cache.signature(resized, entry) == key) == same, tokens


def test_calls_with_one_signature_get_one_decision():
    """What check mode verifies per hit, over the token buckets, the activation-dtype
    bounds and execution-only arguments of every path."""
    exec_changes = {
        "topk_weight": torch.ones(1),
        "moe_sorting_dispatch_policy": 1,
        "splitk": 4,
        "herd_routed_scaling_factor": 2.0,
        "output": torch.empty(1),
        "clamp_shared": False,
        "shared_expert_id": 0,
    }
    decisions = {}
    shared = 0
    for tokens in TOKENS:
        for entry, call in _calls(tokens):
            params = call_cache.entry_params(entry)
            changes = {k: v for k, v in exec_changes.items() if k in params}
            for variant in (call, {**call, **changes}):
                key = call_cache.signature(variant, entry)
                assert key is not None
                decision = _decide(entry, variant)
                if key in decisions:
                    shared += 1
                    assert decisions[key] == decision, call_cache.describe(key)
                decisions[key] = decision
    assert shared > len(decisions)
    routed = {d[0].name for d in decisions.values() if d[0] is not None}
    assert routed >= {"a8w8_pt", "a8w8_ptok", "a4w4", "a16w4", "a8w4", "fhmoe"}


@contextlib.contextmanager
def _env_reads():
    reads = set()
    getitem = os._Environ.__getitem__

    def recording(self, key):
        reads.add(key)
        return getitem(self, key)

    os._Environ.__getitem__ = recording
    try:
        yield reads
    finally:
        os._Environ.__getitem__ = getitem


@pytest.mark.parametrize("tokens", [32, 300])
def test_the_signature_reads_each_variable_the_decision_reads_per_call(
    monkeypatch, tokens
):
    """A variable the dispatch caches read on a miss is frozen until `clear_cache()`
    anyway; one read on every decision must be read for the signature too."""
    monkeypatch.setenv("AITER_USE_GROUPED_GEMM", "0")
    for entry, call in _calls(tokens):
        _decide(entry, call)
        with _env_reads() as decided:
            _decide(entry, call)
        call_cache.signature(call, entry)
        with _env_reads() as keyed:
            call_cache.signature(call, entry)
        assert decided <= keyed, (entry, sorted(decided - keyed))


def test_the_activation_memo_keys_each_variable_the_policy_reads():
    from aiter.moev2.policy import resolve_activation_dtype

    weights = (dtypes.fp4x2, dtypes.fp8, dtypes.i4x2, dtypes.bf16, torch.uint32)
    with _env_reads() as reads:
        for quant_type, w, activation, gate_mode, gfx, M in itertools.product(
            QuantType.__members__.values(),
            weights,
            ActivationType.__members__.values(),
            GateMode,
            ("gfx942", "gfx950", "gfx1250"),
            (None, 1, 1 << 16),
        ):
            resolve_activation_dtype(
                quant_type, w, activation=activation, gate_mode=gate_mode, M=M, gfx=gfx
            )
    assert reads == set(call_cache.ACTIVATION_ENV), sorted(
        reads ^ set(call_cache.ACTIVATION_ENV)
    )


def test_the_activation_memo_follows_its_variables(monkeypatch):
    call = _op_call(
        QuantType.per_1x32,
        dtypes.fp4x2,
        300,
        ActivationType.Swiglu,
        GateMode.INTERLEAVE,
    )
    monkeypatch.setattr(call_cache, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "256")
    assert call_cache._activation_dtype(call) == dtypes.fp8
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "1024")
    assert call_cache._activation_dtype(call) == dtypes.bf16


def test_state_the_decision_reads_is_in_the_signature():
    from moe_dispatch import arch_sim

    entry, call = _calls()[0]
    key = call_cache.signature(call, entry)
    register_presorted_moe({"M": 2})
    try:
        assert call_cache.signature(call, entry) != key
        assert paths.route_to_v2(dict(call), entry) is None
    finally:
        register_presorted_moe(None)
    assert paths.route_to_v2(dict(call), entry).name == "a8w8_pt"
    with arch_sim.simulated_arch("gfx942"):
        assert call_cache.signature(call, entry) != key


def _stub_path(claims):
    return paths.Path(name="a16w16", claims=claims, run=lambda call: "ran", entry="op")


def test_check_mode_raises_where_the_signature_misses_an_input(monkeypatch):
    stub = _stub_path(lambda call: call["splitk"] == 0)
    monkeypatch.setattr(paths, "active", lambda: (stub,))
    call = _calls()[0][1]
    assert paths.route_to_v2(dict(call), "op") is stub
    with pytest.raises(
        call_cache.CallCacheMismatch, match=r"route decision differs .*'entry': 'op'"
    ):
        paths.route_to_v2({**call, "splitk": 1}, "op")
    # What check mode guards against: the hit the default mode returns.
    monkeypatch.setenv(MODE, "1")
    assert paths.route_to_v2({**call, "splitk": 1}, "op") is stub
    monkeypatch.setenv(MODE, "0")
    assert paths.route_to_v2({**call, "splitk": 1}, "op") is None


def _stub_execution(monkeypatch):
    for name in ("run_two_stage", "run_one_stage", "run_mxfp4"):
        monkeypatch.setattr(executor, name, lambda *a, **k: "ran")


def test_check_mode_compares_the_dispatch_decision(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    _stub_execution(monkeypatch)
    entry, call = _calls()[0]
    path = paths.route_to_v2(call, entry)
    assert path is a8w8_pt.PATH and path.run(call) == "ran"
    decide = a8w8_pt.LOCAL._decide
    monkeypatch.setattr(
        a8w8_pt.LOCAL,
        "_decide",
        lambda c, f: dataclasses.replace(decide(c, f), q_dtype_a=dtypes.bf16),
    )
    again = dict(_calls()[0][1])
    assert paths.route_to_v2(again, entry) is path
    with pytest.raises(call_cache.CallCacheMismatch, match=r"q_dtype_a: cached"):
        path.run(again)


@pytest.mark.parametrize("mode", ["1", "check", "0"])
def test_a_hit_skips_only_the_decision(monkeypatch, mode):
    from aiter.moev2.migration import a8w8_pt, local

    monkeypatch.setenv(MODE, mode)
    _stub_execution(monkeypatch)
    counts = dict.fromkeys(("why_not", "decide", "prepared", "emitted"), 0)

    def counting(name, function):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(local, "why_not", counting("why_not", local.why_not))
    monkeypatch.setattr(
        a8w8_pt.LOCAL, "_decide", counting("decide", a8w8_pt.LOCAL._decide)
    )
    prologue = executor.prologue

    def counting_prologue(call, quant_remap, prepare=True):
        counts["prepared"] += prepare
        return prologue(call, quant_remap, prepare=prepare)

    monkeypatch.setattr(executor, "prologue", counting_prologue)
    dispatch = a8w8_pt.LOCAL.dispatch

    def counting_dispatch(*args, emit=True):
        counts["emitted"] += emit
        return dispatch(*args, emit=emit)

    monkeypatch.setattr(a8w8_pt.LOCAL, "dispatch", counting_dispatch)
    for _ in range(3):
        call = dict(_calls()[0][1])
        assert paths.route_to_v2(call, "op").run(call) == "ran"
    decided = {"1": 1, "check": 3, "0": 3}[mode]
    assert counts["decide"] == decided
    assert counts["why_not"] == (1 if mode == "1" else 3)
    assert counts["prepared"] == 3
    assert counts["emitted"] == (1 if mode != "0" else 3)


def test_declines_are_cached_and_clear_cache_forgets(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setenv(MODE, "1")
    monkeypatch.setattr(a8w8_pt.LOCAL, "_decide", lambda c, f: paths.DECLINED)
    for _ in range(2):
        call = dict(_calls()[0][1])
        assert paths.route_to_v2(call, "op").run(call) is paths.DECLINED
    [entry] = paths._ROUTES.values()
    assert entry.path is a8w8_pt.PATH
    assert entry.decisions == {(a8w8_pt.LOCAL, (None, None)): paths.DECLINED}
    tuned.clear_cache()
    assert not paths._ROUTES.values()
    paths.route_to_v2(dict(_calls()[0][1]), "op")
    paths.reload()
    assert not paths._ROUTES.values()


def test_calls_the_signature_cannot_key_are_decided_afresh():
    entry, call = _calls()[0]
    unhashable = {**call, "block_size_M": [32]}
    assert call_cache.signature(unhashable, entry) is None
    paths.route_to_v2(unhashable, entry)
    assert unhashable[call_cache.SIGNATURE] == ("check", None, None)
    assert not paths._ROUTES.values()
    del call["dtype"]
    assert call_cache.signature(call, entry) is None


def test_an_unclassified_parameter_turns_the_cache_off_at_its_entry(
    monkeypatch, caplog
):
    params = {k: v for k, v in call_cache.PARAMS.items() if k != "splitk"}
    monkeypatch.setattr(call_cache, "PARAMS", params)
    monkeypatch.setattr(call_cache, "_projections", {})
    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="aiter"):
        assert call_cache.signature(_calls()[0][1], "op") is None
    assert "['splitk'] are not in call_cache.PARAMS" in caplog.text
