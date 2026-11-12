# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER_MOEV2 switch and the legacy fused_moe hook."""

import inspect
import logging
import pathlib
import sys

import pytest

import aiter.fused_moe as legacy
from aiter.moev2 import api, envs, paths

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Each test starts with no path registered (the built-in ones are restored after)."""
    monkeypatch.delenv(envs.AITER_MOEV2, raising=False)
    monkeypatch.delenv(envs.AITER_MOEV2_TRACE, raising=False)
    saved = paths.registered()
    paths._REGISTRY.clear()
    envs.reload()
    paths.reload()
    yield
    paths._REGISTRY.clear()
    paths._REGISTRY.update(saved)
    envs.reload()
    paths.reload()


def _set(monkeypatch, value):
    monkeypatch.setenv(envs.AITER_MOEV2, value)
    envs.reload()


def _fake_path(name, graduated=False, claims=lambda call: True, entry="api", out=None):
    calls = []

    def run(call):
        calls.append(call)
        return ("v2", name) if out is None else out(call)

    paths.register(paths.Path(name, claims, run, graduated, entry=entry))
    return calls


def _cpu_moe_args(tokens=2, model_dim=8, inter_dim=4, experts=3, topk=2):
    import torch

    hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16)
    w2 = torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16)
    topk_weight = torch.rand(tokens, topk, dtype=torch.float32)
    topk_ids = torch.randint(0, experts, (tokens, topk), dtype=torch.int32)
    return hidden, w1, w2, topk_weight, topk_ids


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", "default"),
        ("off", "off"),
        ("0", "off"),
        ("ON", "on"),
        ("all", "on"),
        ("a4w4, fhmoe", frozenset({"a4w4", "fhmoe"})),
    ],
)
def test_selection_parsing(monkeypatch, value, expected):
    _set(monkeypatch, value)
    assert envs.moev2_selection() == expected


def test_nothing_registered_routes_nothing(monkeypatch):
    for value in ("", "off", "on"):
        _set(monkeypatch, value)
        assert not paths.any_active()
        assert paths.route_to_v2({}) is None


def test_unknown_path_name_is_an_error(monkeypatch):
    _set(monkeypatch, "a4w4,a4w44")
    with pytest.raises(ValueError, match="a4w44"):
        paths.any_active()


def test_planned_but_unimplemented_path_warns(monkeypatch, caplog):
    _set(monkeypatch, "a4w4")
    with caplog.at_level(logging.WARNING, logger="aiter"):
        assert not paths.any_active()
    assert "not implemented yet" in caplog.text


def test_register_rejects_unplanned_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown"):
        _fake_path("not_a_path")
    _fake_path("a4w4")
    with pytest.raises(ValueError, match="already"):
        _fake_path("a4w4")


def test_default_runs_only_graduated_paths(monkeypatch):
    _fake_path("a4w4", graduated=False)
    _fake_path("fhmoe", graduated=True)
    assert [p.name for p in paths.active()] == ["fhmoe"]
    _set(monkeypatch, "on")
    assert [p.name for p in paths.active()] == ["a4w4", "fhmoe"]
    _set(monkeypatch, "a4w4")
    assert [p.name for p in paths.active()] == ["a4w4"]
    _set(monkeypatch, "off")
    assert paths.active() == ()


def test_legacy_fused_moe_routes_claimed_calls(monkeypatch):
    calls = _fake_path("a16w16", claims=lambda call: call["hidden_states"] == "claimed")
    _set(monkeypatch, "a16w16")
    out = legacy.fused_moe("claimed", "w1", "w2", "tw", "ti", splitk=3)
    assert out == ("v2", "a16w16")
    (call,) = calls
    assert call["splitk"] == 3 and call["w1"] == "w1"
    assert set(call) == set(inspect.signature(legacy.fused_moe).parameters)


def test_op_entry_sees_the_custom_op_arguments(monkeypatch):
    import torch

    from aiter import ActivationType, QuantType

    marker = torch.full((2, 8), 7.0, dtype=torch.bfloat16)
    calls = _fake_path("a16w16", entry="op", out=lambda call: marker.clone())
    _set(monkeypatch, "a16w16")
    out = legacy.fused_moe(*_cpu_moe_args(), activation=ActivationType.Gelu)
    assert torch.equal(out, marker)
    (call,) = calls
    assert set(call) == set(inspect.signature(legacy.fused_moe_fake).parameters)
    assert call["activation"] == ActivationType.Gelu.value
    assert call["quant_type"] == QuantType.No.value
    assert call["block_size_M"] == -1 and call["ep_source_token_map"] is None


def test_declined_calls_fall_back_to_legacy(monkeypatch):
    import torch

    _fake_path("a16w16", out=lambda call: paths.DECLINED)
    op_calls = _fake_path("a8w8_pt", entry="op", out=lambda call: paths.DECLINED)
    _set(monkeypatch, "on")
    legacy_out = torch.zeros(2, 8, dtype=torch.bfloat16)
    legacy_calls = []

    def fake_impl(*args, **kwargs):
        legacy_calls.append(args)
        return legacy_out

    monkeypatch.setattr(legacy, "_fused_moe_impl", fake_impl)
    assert legacy.fused_moe(*_cpu_moe_args()) is legacy_out
    assert len(op_calls) == 1 and len(legacy_calls) == 1


def _legacy_impl(monkeypatch):
    import torch

    monkeypatch.setattr(
        legacy,
        "_fused_moe_impl",
        lambda *args, **kwargs: torch.zeros(2, 8, dtype=torch.bfloat16),
    )


def test_outcomes_count_how_each_routed_call_ran(monkeypatch):
    import torch

    _legacy_impl(monkeypatch)
    served = torch.zeros(2, 8, dtype=torch.bfloat16)
    _fake_path(
        "a16w16",
        entry="op",
        claims=lambda call: call["hidden_states"].shape[0] != 3,
        out=lambda call: (
            paths.DECLINED if call["hidden_states"].shape[0] == 4 else served
        ),
    )
    paths.reset_outcomes()
    legacy.fused_moe(*_cpu_moe_args())
    assert paths.outcomes() == {}, "nothing is counted without an active path"
    _set(monkeypatch, "a16w16")
    for tokens in (2, 2, 3, 4):
        legacy.fused_moe(*_cpu_moe_args(tokens=tokens))
    assert paths.outcomes() == {
        ("op", "a16w16", "served"): 2,
        ("op", None, "unclaimed"): 1,
        ("op", "a16w16", "declined"): 1,
    }
    monkeypatch.setenv("AITER_CONFIG_FMOE", "legacy.csv")
    envs.reload()
    paths.reset_outcomes()
    legacy.fused_moe(*_cpu_moe_args())
    assert paths.outcomes() == {("op", "a16w16", "legacy_tables"): 1}


def test_trace_file_has_a_line_per_distinct_outcome(monkeypatch, tmp_path):
    import json
    import os

    _legacy_impl(monkeypatch)
    _fake_path("a16w16", entry="op", out=lambda call: paths.DECLINED)
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv(envs.AITER_MOEV2_TRACE, str(trace))
    _set(monkeypatch, "a16w16")
    paths.reset_outcomes()
    for tokens in (2, 2, 5):
        legacy.fused_moe(*_cpu_moe_args(tokens=tokens))
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [line["call"]["hidden_states"] for line in lines] == [
        "bfloat16[2, 8]",
        "bfloat16[5, 8]",
    ]
    first = lines[0]
    assert (first["entry"], first["path"], first["outcome"]) == (
        "op",
        "a16w16",
        "declined",
    )
    assert first["pid"] == os.getpid()
    assert first["call"]["quant_type"] == 0 and first["call"]["output"] is None
    assert paths.outcomes() == {("op", "a16w16", "declined"): 3}


def test_fhmoe_entry_counts_served_and_declined_calls(monkeypatch):
    import torch

    from aiter import fhmoe

    _legacy_impl(monkeypatch)
    monkeypatch.setattr(fhmoe, "_validate_fhmoe_contract", lambda *a, **k: None)
    served = torch.ones(2, 8, dtype=torch.bfloat16)
    _fake_path(
        "fhmoe",
        entry="fhmoe",
        out=lambda call: (
            served if call["hidden_states"].shape[0] == 2 else paths.DECLINED
        ),
    )
    _set(monkeypatch, "fhmoe")
    paths.reset_outcomes()
    assert torch.equal(fhmoe.fhmoe_(*_cpu_moe_args(tokens=2)), served)
    assert not fhmoe.fhmoe_(*_cpu_moe_args(tokens=3)).any()
    assert paths.outcomes() == {
        ("fhmoe", "fhmoe", "served"): 1,
        ("fhmoe", "fhmoe", "declined"): 1,
    }


def test_no_trace_without_the_variable(monkeypatch):
    _legacy_impl(monkeypatch)
    _fake_path("a16w16", entry="op", out=lambda call: paths.DECLINED)
    _set(monkeypatch, "a16w16")
    paths.reset_outcomes()
    legacy.fused_moe(*_cpu_moe_args())
    assert paths.outcomes() and not paths._traced


def test_an_unwritable_trace_warns_once_and_the_call_goes_on(
    monkeypatch, tmp_path, caplog
):
    _legacy_impl(monkeypatch)
    _fake_path("a16w16", entry="op", out=lambda call: paths.DECLINED)
    monkeypatch.setenv(envs.AITER_MOEV2_TRACE, str(tmp_path / "missing" / "t.jsonl"))
    _set(monkeypatch, "a16w16")
    paths.reset_outcomes()
    with caplog.at_level(logging.WARNING, logger="aiter"):
        for tokens in (2, 5):
            legacy.fused_moe(*_cpu_moe_args(tokens=tokens))
    assert caplog.text.count("not traced") == 1
    assert paths.outcomes() == {("op", "a16w16", "declined"): 2}


def test_trace_describes_arguments_without_their_contents():
    import enum
    import json

    import torch

    from aiter import ActivationType
    from aiter.ops.flydsl.moe_common import GateMode

    class Opaque:
        pass

    class Color(enum.Enum):
        RED = 1

    call = {
        "t": torch.zeros(3, 4, dtype=torch.bfloat16),
        "act": ActivationType.Silu,
        "gate": GateMode.INTERLEAVE,
        "color": Color.RED,
        "dtype": torch.float8_e4m3fn,
        "pair": (1, torch.zeros(2, dtype=torch.int32)),
        "nested": {"deeper": {"x": 1}},
        "obj": Opaque(),
    }
    described = {k: paths._describe(v) for k, v in call.items()}
    assert described == {
        "t": "bfloat16[3, 4]",
        "act": str(ActivationType.Silu),
        "gate": GateMode.INTERLEAVE.value,
        "color": "Color.RED",
        "dtype": "torch.float8_e4m3fn",
        "pair": [1, "int32[2]"],
        "nested": {"deeper": "dict"},
        "obj": "Opaque",
    }
    json.dumps(described)


def test_op_entry_path_records_into_the_legacy_kernel_bench(monkeypatch):
    import torch

    from aiter.moev2 import hooks

    def run(call):
        hooks.record_kernel("stage1", run)
        return torch.zeros(2, 8, dtype=torch.bfloat16)

    paths.register(paths.Path("a8w8_pt", lambda call: True, run, entry="op"))
    _set(monkeypatch, "a8w8_pt")
    bench = []
    monkeypatch.setattr(legacy, "kernel_bench_callable", bench)
    legacy.fused_moe(*_cpu_moe_args())
    assert bench == [("stage1", run)]


def test_op_entry_path_sees_the_legacy_fused_decode_sort_quant_switch(monkeypatch):
    import torch

    from aiter.moev2 import hooks

    seen = []

    def run(call):
        seen.append(hooks.fused_decode_sort_quant_disabled())
        return torch.zeros(2, 8, dtype=torch.bfloat16)

    paths.register(paths.Path("a4w4", lambda call: True, run, entry="op"))
    _set(monkeypatch, "a4w4")
    monkeypatch.setattr(legacy, "_fused_decode_sort_quant_disabled", True)
    legacy.fused_moe(*_cpu_moe_args())
    assert seen == [True]
    assert not hooks.fused_decode_sort_quant_disabled()


def test_torch_compile_traces_the_hooks_with_a_path_active(monkeypatch):
    import torch

    marker = torch.full((2, 8), 7.0, dtype=torch.bfloat16)
    _fake_path("a16w16", entry="op", out=lambda call: marker.clone())
    _set(monkeypatch, "a16w16")
    # The first call also imports the built-in paths, which dynamo cannot trace.
    monkeypatch.setattr(paths, "_builtin_loaded", False)
    paths.reload()
    compiled = torch.compile(legacy.fused_moe, backend="eager", fullgraph=True)
    assert torch.equal(compiled(*_cpu_moe_args()), marker)


def test_graduation_can_be_scoped_to_archs(monkeypatch):
    _fake_path("a8w8_pt", graduated=frozenset({"gfx950"}))
    for arch, expected in (("gfx950", ["a8w8_pt"]), ("gfx942", [])):
        monkeypatch.setattr(paths, "get_gfx_runtime", lambda arch=arch: arch)
        paths.reload()
        assert [p.name for p in paths.active()] == expected

    def no_gpu():
        raise RuntimeError("no GPU")

    monkeypatch.setattr(paths, "get_gfx_runtime", no_gpu)
    paths.reload()
    assert paths.active() == ()


def test_unknown_entry_is_an_error():
    with pytest.raises(ValueError, match="entry"):
        paths.Path("a4w4", lambda call: True, lambda call: None, entry="fused_moe")


def test_builtin_paths():
    from aiter.moev2.migration import (
        a4w4,
        a8w4,
        a8w8_blockscale,
        a8w8_pt,
        a8w8_ptok,
        a16w4,
        a16w16,
        a16wi4,
        comm_fused,
        fhmoe,
        grouped_gfx1250,
        mega_moe_gfx1250,
        mega_moe_tp,
        mega_moe_v2,
        mxfp8,
    )

    local = (
        a8w8_pt,
        a16w16,
        a8w8_ptok,
        a8w8_blockscale,
        a4w4,
        a16w4,
        a16wi4,
        a8w4,
        mxfp8,
        fhmoe,
    )
    modules = local + (
        comm_fused,
        grouped_gfx1250,
        mega_moe_gfx1250,
        mega_moe_v2,
        mega_moe_tp,
    )
    assert paths._BUILTIN_MODULES == tuple(m.__name__ for m in modules)
    for module in local:
        entry = "fhmoe" if module is fhmoe else "op"
        assert module.PATH.name == module.NAME and module.PATH.entry == entry
        assert module.PATH.dispatch_record is not None
        assert module.PATH.segments and module.PATH.takes is not None
    # The comm-fused stage 1 is the a8w4 path's; the runner set is not resolved.
    assert comm_fused.PATH.entry == "comm_fused"
    assert comm_fused.PATH.dispatch_record is None and not comm_fused.PATH.segments
    # The grouped gfx1250 branch runs before the 2-stage lookup: no key, no resolver.
    assert grouped_gfx1250.PATH.entry == "op"
    assert grouped_gfx1250.PATH.dispatch_record is None
    assert not grouped_gfx1250.PATH.segments
    # The mega pipeline owns its instance; the hook passes it, there is no resolver.
    assert mega_moe_gfx1250.PATH.entry == "mega_gfx1250"
    assert mega_moe_gfx1250.PATH.dispatch_record is None
    assert not mega_moe_gfx1250.PATH.segments
    assert mega_moe_v2.PATH.entry == "mega_v2"
    assert mega_moe_v2.PATH.dispatch_record is None
    assert not mega_moe_v2.PATH.segments
    assert mega_moe_tp.PATH.entry == "mega_tp"
    assert mega_moe_tp.PATH.dispatch_record is None
    assert not mega_moe_tp.PATH.segments
    # No path graduates: each runs only when enabled with AITER_MOEV2.
    assert not any(m.PATH.graduated for m in modules)


def test_gpu_suites_cover_exactly_the_planned_paths():
    from moe_dispatch import gpu_suites

    assert set(gpu_suites.PATHS) == set(paths.PLANNED)


def test_local_paths_need_the_build_arch_to_be_the_device(monkeypatch):
    from aiter.moev2.migration import local

    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(local, "get_gfx", lambda: "gfx942")
    assert local.same_arch(("gfx942", "gfx950"))
    # GPU_ARCHS="gfx942;gfx950" on a gfx942: legacy picks 1-stage keys for gfx950.
    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    assert not local.same_arch(("gfx942", "gfx950"))


def test_gpu_suites_do_not_compare_a_path_that_is_not_implemented(tmp_path):
    from moe_dispatch import gpu_suites

    missing = sorted(set(paths.PLANNED) - set(paths.registered()))
    assert missing
    lines, ok = gpu_suites.compare(missing[0], tmp_path)
    assert not ok and "not implemented" in lines[0]


def test_legacy_fused_moe_is_a_plain_function():
    assert inspect.isfunction(legacy.fused_moe)
    assert not hasattr(legacy.fused_moe, "cache_info")


def test_v2_api_without_a_path_raises():
    with pytest.raises(NotImplementedError):
        api.fused_moe(None, None, None, None, None)


def test_v2_api_counts_its_calls_under_entry_api():
    _fake_path("a4w4", out=lambda call: paths.DECLINED)
    _fake_path("a16w16", entry="op")
    paths.reset_outcomes()
    assert api.fused_moe(*_cpu_moe_args()) == ("v2", "a16w16")
    assert paths.outcomes() == {
        ("api", "a4w4", "declined"): 1,
        ("api", "a16w16", "served"): 1,
    }


def test_v2_api_matches_legacy_signature():
    old = inspect.signature(legacy.fused_moe).parameters
    new = inspect.signature(api.fused_moe).parameters
    assert list(old) == list(new)
    for name, p in old.items():
        q = new[name]
        assert p.kind == q.kind, name
        default = (
            getattr(p.default, "value", p.default) if name == "gate_mode" else p.default
        )
        assert default == q.default, name
