# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 dispatch pieces on the CPU.

Per-key equality with the legacy code over the whole snapshot runs in a subprocess
(tests/moe_dispatch/key_checks.py, differential.py); the rest are unit tests.
"""

import dataclasses
import inspect
import logging
import pathlib
import subprocess
import sys
import types

import pytest
import torch

import aiter.fused_moe as legacy
from aiter import ActivationType, QuantType, dtypes
from aiter.moev2 import api
from aiter.moev2.dispatch import (
    buf,
    kernel_names,
    ops,
    overrides,
    plan,
    resolver,
    tuned,
)
from aiter.moev2.problem import (
    FEATURES,
    ActSpec,
    FrozenMap,
    MoeProblem,
    ScaleSpec,
    Shape,
    WeightLayout,
    scale_spec,
)
from aiter.moev2.segments.base import Cap, Params, Segment

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))


def _subprocess_check(script):
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / script), "--check"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


@pytest.fixture(autouse=True)
def _gfx950():
    """Dispatch reads the arch, CU count and the arch's fp8; simulate an MI355."""
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx950"):
        yield


def test_problems_and_tuned_lookup_match_legacy_for_every_key():
    _subprocess_check("key_checks.py")


def _key(token=64, quant=QuantType.per_Tensor, fp8=None, **kwargs):
    fp8 = dtypes.fp8 if fp8 is None else fp8
    args = (token, 7168, 256, 256, 8, dtypes.bf16, fp8, fp8, quant, True)
    args += (ActivationType.Silu, False, 0, 0)
    return args, kwargs


def _problem(arch="gfx950", cu_num=256, **kwargs):
    args, kw = _key(**kwargs)
    return MoeProblem.from_legacy_key(*args, **kw, arch=arch, cu_num=cu_num)


# 2.1 problem


def test_problem_round_trips_through_the_legacy_key():
    for kwargs in (
        {},
        {"is_ep": True, "has_activation_scales": True},
        {"is_shuffled": False},
        {"is_shuffled": True, "opus_weights_shuffled": False},
        {"config_file": "/tmp/t.csv", "_disable_inline_sort": True},
        {"q_dtype_a2": dtypes.fp8, "input_dtype": dtypes.bf16, "swiglu_limit": 7.0},
    ):
        p = _problem(**kwargs)
        args, kw = p.legacy_key()
        again = MoeProblem.from_legacy_key(*args, **kw, arch="gfx950", cu_num=256)
        assert again == p and hash(again) == hash(p), kwargs


def test_problem_from_legacy_key_takes_the_legacy_parameters():
    legacy_names = list(
        inspect.signature(legacy.get_2stage_cfgs.__wrapped__).parameters
    )
    ours = [
        n
        for n, prm in inspect.signature(MoeProblem.from_legacy_key).parameters.items()
        if prm.kind is not prm.KEYWORD_ONLY
    ]
    assert ours == legacy_names


def test_problem_rejects_unknown_features_and_inconsistent_shuffle():
    with pytest.raises(ValueError, match="unknown MoeProblem features"):
        _problem().with_features(add={"telepathy"})
    with pytest.raises(ValueError, match="is_shuffled"):
        _problem(is_shuffled=False, opus_weights_shuffled=True)
    assert "telepathy" not in FEATURES


def test_problem_derived_predicates():
    p = _problem(is_ep=True, is_shuffled=True, opus_weights_shuffled=False)
    assert p.is_ep and p.any_shuffled and not p.fully_shuffled
    assert p.quant_type == QuantType.per_Tensor and p.is_g1u1
    assert p.rows_per_expert == 64 * 8 // 256
    assert p.layout.shuffle == "mixed"
    assert _problem(token=1024).shape.m_bucket == 1024


def test_a16w4_situv2_predicates_match_the_legacy_expressions():
    for out in (dtypes.bf16, dtypes.fp16, dtypes.fp32):
        for shuffled in (True, False):
            for doweight in (True, False):
                args = (32, 3072, 3072, 128, 4, out, dtypes.bf16, dtypes.fp4x2)
                args += (
                    QuantType.per_1x32,
                    True,
                    ActivationType.Situv2,
                    doweight,
                    0,
                    0,
                )
                p = MoeProblem.from_legacy_key(
                    *args, is_shuffled=shuffled, arch="gfx950", cu_num=256
                )
                legacy_format = (
                    p.quant_type == QuantType.per_1x32
                    and p.w1.dtype == dtypes.fp4x2
                    and p.act_in.dtype == dtypes.bf16
                    and p.activation.activation == ActivationType.Situv2
                )
                assert p.is_a16w4_situv2 == legacy_format
                assert p.a16w4_situv2_kernel_ok == (
                    legacy_format
                    and out in (dtypes.bf16, dtypes.fp16)
                    and shuffled
                    and not doweight
                )


def test_scale_spec_keeps_the_group_of_unquantized_dtypes():
    assert scale_spec(QuantType.per_1x32, dtypes.bf16).group != ()
    assert scale_spec(QuantType.per_1x32, dtypes.fp4x2).scale_dtype == dtypes.fp8_e8m0
    assert scale_spec(QuantType.No, dtypes.bf16).scale_dtype is None
    assert (
        not ScaleSpec(dtypes.bf16).is_quantized and ScaleSpec(dtypes.fp8).is_quantized
    )


def test_problem_str_is_readable():
    text = str(_problem(is_ep=True))
    assert "M=64" in text and "silu" in text and "features=ep_mask" in text


def test_build_problem_from_a_custom_op_call():
    hidden = torch.empty(3, 64, dtype=dtypes.bf16)
    w1 = torch.empty(4, 32, 64, dtype=dtypes.fp8)
    w2 = torch.empty(4, 64, 16, dtype=dtypes.fp8)
    w1.is_shuffled = w2.is_shuffled = True
    call = {
        name: prm.default
        for name, prm in inspect.signature(legacy.fused_moe).parameters.items()
    }
    call.update(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weight=torch.empty(3, 2),
        topk_ids=torch.zeros(3, 2, dtype=torch.int32),
        quant_type=QuantType.per_Tensor,
    )
    p = api.build_problem(api.op_call(call), arch="gfx950", cu_num=256)
    # Legacy keys the dispatch on the padded token count.
    assert (p.shape.m, p.shape.model_dim, p.shape.inter_dim) == (
        legacy.get_padded_M(3),
        64,
        16,
    )
    assert p.shape.experts == 4 and p.shape.topk == 2 and p.fully_shuffled
    assert p.quant_type == QuantType.per_Tensor and p.out_dtype == dtypes.bf16


# 2.2 ops, buffers, plans, capabilities


def test_op_chain_and_buffers():
    assert ops.chain(_problem())[0] is ops.Op.SORT
    assert ops.Op.TAIL not in ops.chain(_problem())
    assert str(buf.Buf("inter", ScaleSpec(dtypes.fp8)))
    with pytest.raises(ValueError):
        buf.Buf("bogus", ScaleSpec(dtypes.fp8))


def test_params_and_frozen_map():
    p = Params(kernel="k", ksplit=0)
    assert p.kernel == "k" and p.replace(ksplit=2).ksplit == 2 and p.ksplit == 0
    assert hash(p) == hash(Params({"ksplit": 0, "kernel": "k"}))
    with pytest.raises(AttributeError):
        _ = p.missing
    with pytest.raises(TypeError):
        p["kernel"] = "x"
    assert FrozenMap({"a": 1}) == FrozenMap({"a": 1})


def test_cap_rejects_with_a_readable_reason():
    fp8 = frozenset({scale_spec(QuantType.per_Tensor, dtypes.fp8)})
    cap = Cap(
        arch=frozenset({"gfx950"}),
        act_in=fp8,
        w1=fp8,
        w2=fp8,
        activation=frozenset({"silu"}),
        features=frozenset({"ep_mask"}),
    )
    assert cap.reject_reason(_problem()) is None
    assert cap.reject_reason(_problem(is_ep=True)) is None
    assert "arch gfx942" in cap.reject_reason(_problem(arch="gfx942"))
    assert "feature bias1" in cap.reject_reason(_problem(has_stage1_bias=True))
    assert "padding" in cap.reject_reason(
        _problem().__class__.from_legacy_key(
            *(_key()[0][:12] + (128, 0)), arch="gfx950", cu_num=256
        )
    )
    assert "extras tuned_table" in cap.reject_reason(_problem(config_file="/t.csv"))
    assert "act_in float8_e4m3fn/per_token not in" in cap.reject_reason(
        _problem(quant=QuantType.per_Token)
    )


def test_segment_without_capabilities_accepts_nothing():
    class Bare(Segment):
        name = "bare"

    assert not Bare().accepts(_problem())
    assert Bare().lossy_values == FrozenMap()


def test_plan_describes_its_structure():
    from aiter.moev2.migration import a8w8_pt

    plans, rejected = resolver.feasible_plans(_problem(), a8w8_pt.SEGMENTS)
    names = sorted(" | ".join(s.name for s in structure) for structure in plans)
    assert names == [
        "impl.whole_graph",
        "sort | quant.a1 | asm.stage1 | quant.inter | ck.gemm2",
        "sort | quant.a1 | ck.gemm1 | quant.inter | ck.gemm2",
    ]
    assert not rejected
    res = resolver.resolve(_problem(), a8w8_pt.SEGMENTS)
    assert res.plan.source in plan.SOURCES
    assert res.plan.structure() and "ck.gemm1" in res.plan.describe()


# 2.3 kernel names


def _csv_kernel_names():
    import pandas as pd

    names = set()
    configs = _REPO / "aiter" / "configs"
    for csv in sorted(configs.rglob("*fmoe*.csv")):
        if "untuned" in csv.name:
            continue
        df = pd.read_csv(csv)
        for column in ("kernelName1", "kernelName2"):
            if column in df.columns:
                names |= {n for n in df[column] if not kernel_names.is_empty(n)}
    return sorted(names)


def test_every_shipped_kernel_name_parses_and_round_trips():
    names = _csv_kernel_names()
    assert len(names) > 100
    for name in names:
        parsed = kernel_names.parse(name)
        assert parsed is not None and parsed.family in kernel_names.FAMILIES, name
        assert parsed.format() == name


def test_kernel_name_edge_cases():
    assert kernel_names.parse("") is None and kernel_names.parse(float("nan")) is None
    with pytest.raises(kernel_names.KernelNameError):
        kernel_names.parse("totally_new_kernel_family_x")
    assert kernel_names.try_parse("totally_new_kernel_family_x") is None
    # One shipped ASM name declares a length one more than its symbol.
    odd = "_ZN5aiter47fmoe_stage1_bf16_pertokenFp8_g1u1_16x128_4tg_pf2E"
    assert kernel_names.parse(odd).format() == odd


# 2.4 / 2.5 resolver, overrides, explain


def test_overrides_read_the_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setenv("AITER_KSPLIT", "-1")
    monkeypatch.setenv("AITER_USE_NT", "0")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "1")
    assert overrides.bypass_tuned() and overrides.ksplit_disabled()
    assert overrides.nt_override() == 0 and overrides.online_tune()
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG")
    monkeypatch.delenv("AITER_KSPLIT")
    monkeypatch.delenv("AITER_USE_NT")
    monkeypatch.delenv("AITER_ONLINE_TUNE")
    assert not overrides.bypass_tuned() and not overrides.ksplit_disabled()
    assert overrides.nt_override() == -1 and not overrides.online_tune()


def _resolve_with_row(monkeypatch, row):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setattr(tuned, "lookup", lambda problem: row)
    return resolver.resolve(_problem(), a8w8_pt.SEGMENTS)


@pytest.mark.parametrize(
    "row,reason",
    [
        ({"run_1stage": 1, "kernelName1": "", "kernelName2": ""}, "1-stage"),
        (
            {
                "kernelName1": "flydsl_moe1_afp8_wfp8_bf16_t32x128x128",
                "kernelName2": "",
            },
            "no segments",
        ),
        ({"kernelName1": " moe_ck2stages_gemm1_x", "kernelName2": ""}, "blanks"),
        ({"kernelName1": "brand_new_kernel", "kernelName2": ""}, "unknown kernel name"),
        ({"kernelName1": 3.0, "kernelName2": ""}, "not a string"),
    ],
)
def test_rows_of_unported_kernels_are_declined_silently(
    monkeypatch, caplog, row, reason
):
    res = _resolve_with_row(monkeypatch, row)
    assert res.plan is None and reason in res.declined
    with caplog.at_level(logging.INFO, logger="aiter"):
        res.emit()
    assert not res.logs and caplog.text == ""


def test_a_declined_call_runs_on_legacy(monkeypatch):
    from aiter.moev2 import paths
    from aiter.moev2.migration import a8w8_pt, local

    monkeypatch.setattr(a8w8_pt.LOCAL, "dispatch", lambda *a: None)
    monkeypatch.setattr(local.executor, "prologue", lambda call, remap: None)
    monkeypatch.setattr(api, "dispatch_key", lambda c: ((), {}, None))
    assert a8w8_pt.PATH.run({}) is paths.DECLINED


def test_explain_lists_problem_structures_and_choice(monkeypatch, caplog):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setenv("AITER_MOE_EXPLAIN", "1")
    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    res = resolver.resolve(_problem(), a8w8_pt.SEGMENTS)
    text = "\n".join(res.explain)
    assert "problem: gfx950 M=64" in text and "feasible structures (3)" in text
    assert "tuned lookup: miss" in text and "(fp8_per_tensor[otherwise])" in text
    assert "selected" in text
    with caplog.at_level(logging.INFO, logger="aiter"):
        res.emit()
    assert resolver.EXPLAIN_PREFIX + "problem:" in caplog.text
    monkeypatch.delenv("AITER_MOE_EXPLAIN")
    assert resolver.resolve(_problem(), a8w8_pt.SEGMENTS).explain == ()


def test_default_resolution_logs_like_legacy(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    res = resolver.resolve(_problem(), a8w8_pt.SEGMENTS)
    (_, first), (_, second) = res.logs
    assert first.startswith("run_1stage = False, xbf16 = False, ksplit = 0 q_type = ")
    assert second.startswith("[fused_moe] using 2stage default for ('gfx950', 256, 64,")
    assert res.metadata.stage1.func.__name__ == "ck_moe_stage1"


# 2.6 a8w8_pt


def test_no_one_stage_heuristic_for_per_tensor():
    # a8w8_pt has no 1-stage segment: the legacy default never picks 1-stage for it.
    for table in legacy.fused_moe_1stage_dict.values():
        for key in table:
            assert key[1] != QuantType.per_Tensor, key


def test_op_call_matches_what_legacy_passes_to_the_custom_op(monkeypatch):
    seen = {}

    def fake_op(**kwargs):
        seen.update(kwargs)
        return "out"

    monkeypatch.setattr(legacy, "fused_moe_", fake_op)
    scatter = types.SimpleNamespace(
        source_token_map=torch.zeros(3, dtype=torch.int32),
        arena_handle=11,
        combine_input_offset=12,
        slot_stride_bytes=13,
        max_tokens_per_rank=14,
        world_size=15,
        combine_quant_bits=8,
    )
    for extra in (
        {},
        {
            "stage2_scatter": scatter,
            "block_size_M": 64,
            "quant_type_a": QuantType.per_Tensor,
        },
    ):
        seen.clear()
        call = {
            name: prm.default
            for name, prm in inspect.signature(legacy.fused_moe).parameters.items()
        }
        call.update(
            hidden_states="h", w1="w1", w2="w2", topk_weight="tw", topk_ids="ti",
            activation=ActivationType.Gelu, quant_type=QuantType.per_Tensor, **extra,
        )  # fmt: skip
        assert legacy.fused_moe(**call) == "out"
        assert api.op_call(call) == seen


def _per_tensor_op_call():
    call = {
        name: prm.default
        for name, prm in inspect.signature(legacy.fused_moe_fake).parameters.items()
    }
    call.update(
        hidden_states=torch.empty(2, 64, dtype=dtypes.bf16),
        w1=torch.empty(4, 32, 64, dtype=dtypes.fp8),
        w2=torch.empty(4, 64, 16, dtype=dtypes.fp8),
        topk_weight=torch.empty(2, 2),
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        quant_type=QuantType.per_Tensor.value,
    )
    return call


def test_claims_only_per_tensor_fp8(monkeypatch):
    from aiter.moev2.migration import a8w8_pt, local

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    call = _per_tensor_op_call()
    assert a8w8_pt.claims(call)
    for change in (
        {"quant_type": QuantType.per_Token.value},
        {"activation": ActivationType.Swiglu.value},
        {"bias1": torch.empty(1)},
        {"hidden_pad": 64},
        {"quant_dtype_a2": dtypes.fp8},
        {"w1": torch.empty(4, 32, 64, dtype=torch.int8)},
        {"w1": torch.empty(4, 16, 64, dtype=dtypes.fp8)},  # g1u0
        {"dtype": torch.float32},
    ):
        assert not a8w8_pt.claims({**call, **change}), change
    monkeypatch.setenv("AITER_ONLINE_TUNE", "1")
    assert not a8w8_pt.claims(call)
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx1201")
    monkeypatch.delenv("AITER_ONLINE_TUNE")
    assert not a8w8_pt.claims(call)


def _a16w16_op_call(dtype=dtypes.bf16):
    call = _per_tensor_op_call()
    call.update(
        hidden_states=torch.empty(2, 64, dtype=dtype),
        w1=torch.empty(4, 32, 64, dtype=dtype),
        w2=torch.empty(4, 64, 16, dtype=dtype),
        w1_scale=None,
        w2_scale=None,
        quant_type=QuantType.No.value,
    )
    return call


def test_claims_only_unquantized_16_bit(monkeypatch):
    from aiter.moev2.migration import a8w8_pt, a16w16, local

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    for dtype in (dtypes.bf16, dtypes.fp16):
        call = _a16w16_op_call(dtype)
        assert a16w16.claims(call) and not a8w8_pt.claims(call)
    assert not a16w16.claims(_per_tensor_op_call())
    call = _a16w16_op_call()
    for change in (
        {"quant_type": QuantType.per_Token.value},
        {"quant_type_a": QuantType.per_Tensor.value},
        {"activation": ActivationType.Swiglu.value},
        {"bias2": torch.empty(1)},
        {"w2": torch.empty(4, 64, 16, dtype=dtypes.fp8)},
        {"w1": torch.empty(4, 16, 64, dtype=dtypes.bf16)},  # g1u0
        {"dtype": torch.float32},
    ):
        assert not a16w16.claims({**call, **change}), change
    monkeypatch.setattr(local, "get_gfx", lambda: "gfx1201")
    assert not a16w16.claims(call)


_LOCAL_PATHS = ("a8w8_pt", "a16w16", "a8w8_ptok", "a8w8_blockscale")


def _local_path_modules():
    import importlib

    return [importlib.import_module(f"aiter.moev2.migration.{n}") for n in _LOCAL_PATHS]


def _fp8_op_call(quant_type):
    call = _per_tensor_op_call()
    call["quant_type"] = quant_type.value
    return call


def test_the_paths_take_disjoint_keys():
    owners = {
        "a8w8_pt": _problem(),
        "a16w16": _problem(quant=QuantType.No, fp8=dtypes.bf16),
        "a8w8_ptok": _problem(quant=QuantType.per_Token),
        "a8w8_blockscale": _problem(quant=QuantType.per_1x128),
    }
    modules = _local_path_modules()
    for owner, p in owners.items():
        assert [m.NAME for m in modules if m.takes(p)] == [owner], p


def test_claims_per_token_and_blockscale_fp8(monkeypatch):
    from aiter.moev2.migration import a8w8_blockscale, a8w8_ptok, local

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    modules = _local_path_modules()
    for quant, owner in (
        (QuantType.per_Token, a8w8_ptok),
        (QuantType.per_1x128, a8w8_blockscale),
        (QuantType.per_128x128, a8w8_blockscale),  # remapped to per_1x128 by legacy
    ):
        call = _fp8_op_call(quant)
        assert [m.NAME for m in modules if m.claims(call)] == [owner.NAME], quant
        for change in (
            {"w1": torch.empty(4, 32, 64, dtype=torch.int8)},
            {"w1": torch.empty(4, 16, 64, dtype=dtypes.fp8)},  # g1u0
            {"activation": ActivationType.Swiglu.value},
            {"quant_type_a": QuantType.per_Tensor.value},
            {"bias1": torch.empty(1)},
        ):
            assert not owner.claims({**call, **change}), (quant, change)


@pytest.mark.parametrize("base", _LOCAL_PATHS)
def test_claims_and_takes_agree_on_key_fields(monkeypatch, base):
    """A claimed call's key is taken by the same path, and only by it.

    Changes here are visible in the dispatch key; call-only checks (bias tensors,
    padding, AITER_ONLINE_TUNE) are in `claims` alone.
    """
    from aiter.moev2.migration import local

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(api, "get_gfx_runtime", lambda: "gfx950")
    call = {
        "a8w8_pt": _per_tensor_op_call,
        "a16w16": _a16w16_op_call,
        "a8w8_ptok": lambda: _fp8_op_call(QuantType.per_Token),
        "a8w8_blockscale": lambda: _fp8_op_call(QuantType.per_1x128),
    }[base]()
    w1 = call["w1"]
    for change in (
        {},
        *({"quant_type": q.value} for q in QuantType.__members__.values()),
        {"w1": torch.empty(4, 16, 64, dtype=w1.dtype)},  # g1u0
        {"activation": ActivationType.Gelu.value},
        {"dtype": dtypes.fp16},
    ):
        changed = {**call, **change}
        problem = api.build_problem(changed, arch="gfx950", cu_num=256)
        for path in _local_path_modules():
            assert path.claims(changed) == path.takes(problem), (path.NAME, change)


def _default_resolve(monkeypatch, token, quant, inter_dim=256):
    from aiter.moev2.migration import a8w8_blockscale, a8w8_ptok

    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    p = _problem(token=token, quant=quant)
    p = dataclasses.replace(p, shape=dataclasses.replace(p.shape, inter_dim=inter_dim))
    path = a8w8_blockscale if quant == QuantType.per_1x128 else a8w8_ptok
    return resolver.resolve(p, path.SEGMENTS)


def test_per_token_fp8_defaults_hand_the_1stage_region_back(monkeypatch):
    res = _default_resolve(monkeypatch, 17, QuantType.per_Token)
    assert res.plan is None and "1-stage ASM fmoe is not ported" in res.declined
    res = _default_resolve(monkeypatch, 16, QuantType.per_Token, inter_dim=192)
    assert res.plan is None and "not inter_dim_128" in res.declined
    res = _default_resolve(monkeypatch, 16, QuantType.per_Token)
    assert res.plan.structure()[0] == (
        "sort", "quant.a1", "ck.gemm1", "quant.inter", "ck.gemm2"
    )  # fmt: skip
    assert (res.metadata.block_m, res.metadata.ksplit) == (32, 0)


def test_blockscale_defaults_take_the_legacy_block_m_and_ksplit(monkeypatch):
    calls = []

    def has_kernel(*args):
        calls.append(args)
        return 0

    res = _default_resolve(monkeypatch, 1, QuantType.per_1x128)
    assert (res.metadata.block_m, res.metadata.ksplit) == (16, 14)
    res = _default_resolve(monkeypatch, 33, QuantType.per_1x128)
    assert res.plan is None and "fmoe_fp8_blockscale_g1u1 is not ported" in res.declined

    import aiter

    monkeypatch.setattr(aiter, "fmoe_fp8_blockscale_g1u1_has_kernel", has_kernel)
    _default_resolve(monkeypatch, 16, QuantType.per_1x128)
    _default_resolve(monkeypatch, 33, QuantType.per_1x128, inter_dim=192)
    assert calls == []  # legacy asks the kernel table only past the shape checks
    res = _default_resolve(monkeypatch, 33, QuantType.per_1x128)
    assert calls == [(ActivationType.Silu.value, False, 32, 256, 7168)]
    assert res.plan.structure()[0][2] == "ck.gemm1"
    assert (res.metadata.block_m, res.metadata.ksplit) == (64, 0)


def test_a_tuned_asm_blockscale_row_runs_the_packed_stage1(monkeypatch):
    from aiter.moev2.migration import a8w8_blockscale

    row = {
        "block_m": 16,
        "ksplit": 0,
        "run_1stage": 0,
        "kernelName1": "_ZN5aiter59fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1_16x256_2tg_pf3E",
        "kernelName2": "moe_ck2stages_gemm2_256x16x128x256_1x4_MulABScaleExpertWeightA8W8blkscale_v1_Nswizzle0_Quant4_MulRoutedWeight1_F8_F8_B16",
    }
    monkeypatch.setattr(tuned, "lookup", lambda problem: row)
    res = resolver.resolve(
        _problem(quant=QuantType.per_1x128), a8w8_blockscale.SEGMENTS
    )
    assert res.plan.structure()[0] == (
        "sort", "quant.a1", "asm.stage1_1x128", "ck.gemm2"
    )  # fmt: skip
    assert res.metadata.stage1.func.__name__ == "asm_stage1"


def test_the_packed_stage1_buffer_splits_like_legacy():
    from aiter.moev2.segments.asm import AsmStage1Blockscale as S

    token, topk, inter, model = 5, 8, 256, 512
    a1 = torch.empty(token, model, dtype=dtypes.fp8)
    a1_scale = torch.empty(model // 128, token, dtype=torch.float32)
    a2 = S.out_buffer(a1, a1_scale, token, topk, inter, dtypes.fp8)
    assert a2.shape == (token + (token * 4 + 127) // 128, topk, inter)
    out, scale = S.split_out(a2, a1, a1_scale, token, topk, inter)
    assert out.shape == (token, topk, inter) and out.data_ptr() == a2.data_ptr()
    assert scale.dtype == torch.float32 and scale.shape == (token, topk * inter // 128)
    assert scale.data_ptr() == a2[token:].data_ptr()


def test_g1u0_is_not_claimed_and_has_no_ck_plan():
    """CK stage 1 always splits w1 into gate and up; g1u0 calls stay on legacy."""
    from aiter.moev2.migration import a8w8_pt, a16w16

    p = _problem(quant=QuantType.No, fp8=dtypes.bf16)
    assert resolver.resolve(p, a16w16.SEGMENTS).plan is not None
    g1u0 = dataclasses.replace(p, layout=dataclasses.replace(p.layout, g1u1=False))
    assert not a16w16.takes(g1u0)
    res = resolver.resolve(g1u0, a16w16.SEGMENTS)
    assert res.plan is None and "no listed structure is feasible" in res.declined
    fp8 = _problem()
    assert a8w8_pt.takes(fp8)
    assert not a8w8_pt.takes(
        dataclasses.replace(fp8, layout=dataclasses.replace(fp8.layout, g1u1=False))
    )


def test_tuned_cache_clear_also_resets_path_caches():
    for path in _local_path_modules():
        caches = (path.LOCAL.dispatch, path.LOCAL.resolve, path.LOCAL.relookup)
        assert all(cache in tuned._CACHES for cache in caches)


def test_v2_reads_and_resets_with_the_legacy_table(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    p = _problem()
    keys, _ = tuned.lookup_keys(p)
    key = keys[:7] + (str(p.activation.activation),) + keys[8:]
    row = {"kernelName1": "", "kernelName2": "", "block_m": 64, "ksplit": 0}
    monkeypatch.setattr(legacy, "cfg_2stages", ({key: row}, {}))
    assert tuned.lookup(p) is row
    a8w8_pt.LOCAL.resolve(p, False)
    assert a8w8_pt.LOCAL.resolve.cache_info().currsize == 1
    legacy.get_2stage_cfgs.cache_clear()
    assert a8w8_pt.LOCAL.resolve.cache_info().currsize == 0


def test_the_log_only_relookup_runs_after_the_sort(monkeypatch, caplog):
    from aiter.moev2 import hooks
    from aiter.moev2.migration import a8w8_pt, local
    from aiter.moev2.segments.quant import QuantA1

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    def sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, *args, **kw):
        seen.append(caplog.text.count(line))
        if len(seen) == 1:
            raise Stop
        empty = torch.empty(0, dtype=torch.int32)
        out = torch.empty(topk_ids.shape[0], model_dim, dtype=moebuf_dtype)
        return empty, empty, empty, empty, out

    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(local, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    monkeypatch.setattr(QuantA1, "run", stop)
    line, seen = "[fused_moe] using 2stage default", []
    with caplog.at_level(logging.INFO, logger="aiter"), hooks.set_sort_impl(sort):
        # Legacy logs the lookup once before sorting and once more after it.
        for _ in range(2):
            with pytest.raises(Stop):
                a8w8_pt.PATH.run(_per_tensor_op_call())
    assert seen == [1, 1] and caplog.text.count(line) == 2


def test_the_relookup_buckets_the_hidden_states_rows(monkeypatch):
    from aiter.moev2 import hooks
    from aiter.moev2.migration import a8w8_pt, local
    from aiter.moev2.segments.quant import QuantA1

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    def sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, *args, **kw):
        empty = torch.empty(0, dtype=torch.int32)
        out = torch.empty(topk_ids.shape[0], model_dim, dtype=moebuf_dtype)
        return empty, empty, empty, empty, out

    tokens = []

    def lookup(problem):
        tokens.append(problem.shape.m)

    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(local, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "lookup", lookup)
    monkeypatch.setattr(QuantA1, "run", stop)
    call = _per_tensor_op_call()
    call["hidden_states"] = torch.empty(3, 64, dtype=dtypes.bf16)
    # fused_moe_2stages buckets hidden_states.shape[0]; the first lookup, topk_ids rows.
    with hooks.set_sort_impl(sort), pytest.raises(Stop):
        a8w8_pt.PATH.run(call)
    assert tokens == [2, 4]


def test_shape_helpers_moved_verbatim():
    from aiter.moev2 import shapes

    assert legacy.get_padded_M is shapes.get_padded_M
    assert legacy.get_inter_dim is shapes.get_inter_dim
    assert shapes.get_inter_dim((4, 32, 64), (4, 64, 16)) == (4, 64, 16)


def test_shape_is_frozen():
    s = Shape(1, 2, 3, 4, 5)
    with pytest.raises(AttributeError):
        s.m = 3
    assert ActSpec("silu").activation == ActivationType.Silu
    assert WeightLayout("16x16", "separated").g1u1
