# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 dispatch pieces on the CPU.

Per-key equality with the legacy code over the whole snapshot runs in a subprocess
(tests/moe_dispatch/key_checks.py, differential.py); the rest are unit tests.
"""

import dataclasses
import functools
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


def _key(token=64, quant=QuantType.per_Tensor, fp8=None, w=None, **kwargs):
    fp8 = dtypes.fp8 if fp8 is None else fp8
    w = fp8 if w is None else w
    args = (token, 7168, 256, 256, 8, dtypes.bf16, fp8, w, quant, True)
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


def test_a_segment_reads_the_last_buffer_of_each_kind_before_it():
    from aiter.moev2.dispatch.buf import Buf
    from aiter.moev2.migration import a8w8_pt
    from aiter.moev2.problem import ScaleSpec
    from aiter.moev2.segments.ck import CkGemm1, CkGemm2
    from aiter.moev2.segments.sort import Sort

    aux = Buf("sorted", None, layout="aux_sorted")

    class AuxSort(Sort):
        name = "sort.aux"

        def produces(self, p, params):
            return aux

    class AuxGemm1(CkGemm1):
        name = "aux.gemm1"

        def reads(self, p):
            return (aux, Buf("tokens", p.act_in))

    class AuxGemm2(CkGemm2):
        name = "aux.gemm2"

        def reads(self, p):
            return (aux, Buf("inter", p.act_in))

    class Bf16Gemm2(CkGemm2):
        name = "bf16.gemm2"

        def reads(self, p):
            return (Buf("sorted", None), Buf("inter", ScaleSpec(dtypes.bf16)))

    extra = (AuxSort(), AuxGemm1(), AuxGemm2(), Bf16Gemm2())
    plans, rejected = resolver.feasible_plans(_problem(), a8w8_pt.SEGMENTS + extra)
    names = sorted(" | ".join(s.name for s in structure) for structure in plans)
    # The GEMM2s read the sort's layout from three segments back.
    assert names == [
        "impl.whole_graph",
        "sort | quant.a1 | asm.stage1 | quant.inter | ck.gemm2",
        "sort | quant.a1 | ck.gemm1 | quant.inter | ck.gemm2",
        "sort.aux | quant.a1 | aux.gemm1 | quant.inter | aux.gemm2",
    ]
    assert list(rejected) == ["bf16.gemm2"]
    assert "reads inter:" in rejected["bf16.gemm2"]
    assert "gets inter:" in rejected["bf16.gemm2"]


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
    monkeypatch.setattr(local.executor, "a16w4_unsupported", lambda c, q: None)
    monkeypatch.setattr(api, "dispatch_key", lambda c, *_: ((), {}, None))
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
        {"dtype": torch.float32},
    ):
        assert not a16w16.claims({**call, **change}), change
    # g1u0: the ranking runs the 1-stage `fmoe` keys and declines the rest.
    assert a16w16.claims({**call, "w1": torch.empty(4, 16, 64, dtype=dtypes.bf16)})
    monkeypatch.setattr(local, "get_gfx", lambda: "gfx1201")
    assert not a16w16.claims(call)


_LOCAL_PATHS = (
    "a8w8_pt",
    "a16w16",
    "a8w8_ptok",
    "a8w8_blockscale",
    "a4w4",
    "a16w4",
    "a16wi4",
    "a8w4",
    "mxfp8",
)


def _local_path_modules():
    import importlib

    return [importlib.import_module(f"aiter.moev2.migration.{n}") for n in _LOCAL_PATHS]


def _fp8_op_call(quant_type):
    call = _per_tensor_op_call()
    call["quant_type"] = quant_type.value
    return call


def _int_op_call(quant_type, weights):
    """int8 weights, or int4 packed eight to a uint32 (K / 8 columns)."""
    call = _fp8_op_call(quant_type)
    if weights == "i8":
        call.update(
            w1=torch.empty(4, 32, 64, dtype=dtypes.i8),
            w2=torch.empty(4, 64, 16, dtype=dtypes.i8),
        )
    else:
        call.update(
            w1=torch.empty(4, 32, 8, dtype=torch.uint32),
            w2=torch.empty(4, 64, 2, dtype=torch.uint32),
        )
    return call


def _a4w4_op_call():
    """MXFP4 activations and weights: fp4 packed two to a byte (K / 2 columns)."""
    call = _fp8_op_call(QuantType.per_1x32)
    call.update(
        w1=torch.empty(4, 32, 32, dtype=dtypes.fp4x2),
        w2=torch.empty(4, 64, 8, dtype=dtypes.fp4x2),
    )
    return call


def _a16w4_problem(
    activation=ActivationType.Situv2,
    g1u1=True,
    token=64,
    shape=(7168, 256, 256, 8),
    arch="gfx950",
    **kwargs,
):
    args = (token, *shape, dtypes.bf16, dtypes.bf16, dtypes.fp4x2)
    args += (QuantType.per_1x32, g1u1, activation, False, 0, 0)
    kwargs.setdefault("is_shuffled", True)
    return MoeProblem.from_legacy_key(*args, **kwargs, arch=arch, cu_num=256)


def _a16wi4_op_call():
    """bf16 activations into int4 weights: two to a byte (K / 2 columns)."""
    return {
        **_a4w4_op_call(),
        "w1": torch.empty(4, 32, 32, dtype=dtypes.i4x2),
        "w2": torch.empty(4, 64, 8, dtype=dtypes.i4x2),
    }


def _a16wi4_problem(activation=ActivationType.Silu, token=64, **kwargs):
    args = (token, 7168, 256, 384, 8, dtypes.bf16, dtypes.bf16, dtypes.i4x2)
    args += (QuantType.per_1x32, True, activation, False, 0, 0)
    kwargs.setdefault("is_shuffled", True)
    return MoeProblem.from_legacy_key(*args, **kwargs, arch="gfx950", cu_num=256)


def _a8w4_problem(activation=ActivationType.Silu, token=64, **kwargs):
    args = (token, 7168, 256, 256, 8, dtypes.bf16, dtypes.fp8, dtypes.fp4x2)
    args += (QuantType.per_1x32, True, activation, False, 0, 0)
    kwargs.setdefault("is_shuffled", True)
    return MoeProblem.from_legacy_key(*args, **kwargs, arch="gfx950", cu_num=256)


def _mxfp8_problem(activation=ActivationType.Silu, token=64, **kwargs):
    args = (token, 6144, 512, 256, 8, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    args += (QuantType.per_1x32, True, activation, False, 0, 0)
    kwargs.setdefault("is_shuffled", True)
    return MoeProblem.from_legacy_key(*args, **kwargs, arch="gfx950", cu_num=256)


def _a16w4_resolve(monkeypatch, problem, row=None):
    from aiter.moev2.migration import a16w4

    monkeypatch.setattr(tuned, "lookup", lambda p: row)
    return resolver.resolve(problem, a16w4.SEGMENTS)


def test_the_paths_take_disjoint_keys():
    i8, u32 = dtypes.i8, torch.uint32
    owners = (
        ("a8w8_pt", _problem()),
        ("a8w8_pt", _problem(fp8=i8)),
        ("a16w16", _problem(quant=QuantType.No, fp8=dtypes.bf16)),
        ("a8w8_ptok", _problem(quant=QuantType.per_Token)),
        ("a8w8_ptok", _problem(quant=QuantType.per_Token, fp8=i8)),
        ("a8w8_ptok", _problem(quant=QuantType.per_Token, w=u32)),
        ("a8w8_blockscale", _problem(quant=QuantType.per_1x128)),
        ("a4w4", _problem(quant=QuantType.per_1x32, fp8=dtypes.fp4x2)),
        ("a16w4", _a16w4_problem()),
        ("a16w4", _a16w4_problem(activation=ActivationType.Swiglu)),
        ("a16w4", _a16w4_problem(activation=ActivationType.Relu2, g1u1=False)),
        ("a16wi4", _a16wi4_problem()),
        ("a16wi4", _a16wi4_problem(activation=ActivationType.Situv2)),
        ("a8w4", _a8w4_problem()),
        ("a8w4", _a8w4_problem(activation=ActivationType.Swiglu)),
        ("mxfp8", _mxfp8_problem()),
        ("mxfp8", _mxfp8_problem(activation=ActivationType.Swiglu)),
    )
    modules = _local_path_modules()
    for owner, p in owners:
        assert [m.NAME for m in modules if m.takes(p)] == [owner], p
    for p in (_problem(w=u32), _problem(quant=QuantType.per_Token, w=torch.int4)):
        assert not any(m.takes(p) for m in modules), p


def test_claims_int8_and_int4_weights(monkeypatch):
    from aiter.moev2.migration import a8w8_pt, a8w8_ptok, local

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    modules = _local_path_modules()
    for quant, weights, owner in (
        (QuantType.per_Tensor, "i8", a8w8_pt),
        (QuantType.per_Token, "i8", a8w8_ptok),
        (QuantType.per_Token, "i4", a8w8_ptok),
    ):
        call = _int_op_call(quant, weights)
        assert [m.NAME for m in modules if m.claims(call)] == [owner.NAME], weights
        mixed = {**call, "w2": torch.empty(4, 64, 16, dtype=dtypes.fp8)}
        assert not owner.claims(mixed), weights
    assert not a8w8_pt.claims(_int_op_call(QuantType.per_Tensor, "i4"))


def test_a8w8_pt_runs_only_when_enabled(monkeypatch):
    from aiter.moev2 import envs, paths
    from aiter.moev2.migration import a8w8_pt, local

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(paths, "get_gfx_runtime", lambda: "gfx950")
    fp8, i8 = _per_tensor_op_call(), _int_op_call(QuantType.per_Tensor, "i8")
    for selection, routed in (
        ("", (False, False)),
        ("a8w8_pt", (True, True)),
        ("on", (True, True)),
    ):
        monkeypatch.setenv(envs.AITER_MOEV2, selection)
        envs.reload()
        got = tuple(paths.route_to_v2(c, "op") is a8w8_pt.PATH for c in (fp8, i8))
        assert got == routed, selection
    monkeypatch.delenv(envs.AITER_MOEV2)
    envs.reload()


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


@pytest.mark.parametrize(
    "base", _LOCAL_PATHS + ("a8w8_pt_i8", "a8w8_ptok_i8", "a8w8_ptok_i4")
)
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
        "a8w8_pt_i8": lambda: _int_op_call(QuantType.per_Tensor, "i8"),
        "a8w8_ptok_i8": lambda: _int_op_call(QuantType.per_Token, "i8"),
        "a8w8_ptok_i4": lambda: _int_op_call(QuantType.per_Token, "i4"),
        "a4w4": _a4w4_op_call,
        "a16w4": lambda: {**_a4w4_op_call(), "activation": ActivationType.Situv2.value},
        "a16wi4": _a16wi4_op_call,
        "a8w4": lambda: {**_a4w4_op_call(), "quant_dtype_a": dtypes.fp8},
        "mxfp8": lambda: _fp8_op_call(QuantType.per_1x32),
    }[base]()
    w1 = call["w1"]
    for change in (
        {},
        *({"quant_type": q.value} for q in QuantType.__members__.values()),
        {"w1": torch.empty(4, 16, w1.shape[2], dtype=w1.dtype)},  # g1u0
        {"activation": ActivationType.Gelu.value},
        {"activation": ActivationType.Swiglu.value},
        {"activation": ActivationType.Situv2.value},
        {"activation": ActivationType.Situv2.value, "quant_dtype_a": w1.dtype},
        {"activation": ActivationType.Swiglu.value, "quant_dtype_a": dtypes.bf16},
        {"activation": ActivationType.Swiglu.value, "quant_dtype_a": dtypes.fp8},
        {"activation": ActivationType.Relu2.value},
        {
            "activation": ActivationType.Relu2.value,
            "w1": torch.empty(4, 16, w1.shape[2], dtype=w1.dtype),
        },
        {"dtype": dtypes.fp16},
    ):
        changed = {**call, **change}
        problem = api.build_problem(changed, arch="gfx950", cu_num=256)
        for path in _local_path_modules():
            assert path.claims(changed) == path.takes(problem), (path.NAME, change)


def _default_resolve(monkeypatch, token, quant, inter_dim=256, arch="gfx950", **kw):
    from aiter.moev2.migration import a8w8_blockscale, a8w8_pt, a8w8_ptok

    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    p = _problem(token=token, quant=quant, arch=arch, **kw)
    p = dataclasses.replace(p, shape=dataclasses.replace(p.shape, inter_dim=inter_dim))
    path = {QuantType.per_1x128: a8w8_blockscale, QuantType.per_Tensor: a8w8_pt}
    return resolver.resolve(p, path.get(quant, a8w8_ptok).SEGMENTS)


def test_int8_defaults_run_asm_stage1_and_int4_runs_ck(monkeypatch):
    asm = ("sort", "quant.a1", "asm.stage1", "quant.inter", "ck.gemm2")
    for quant in (QuantType.per_Tensor, QuantType.per_Token):
        res = _default_resolve(monkeypatch, 64, quant, fp8=dtypes.i8)
        assert res.plan.structure()[0] == asm, quant
        assert res.metadata.stage1.func.__name__ == "asm_stage1"
        assert res.metadata.block_m in (16, 32, 64, 128)
    # gfx942 runs 1-stage `fmoe` for per-token int8 above 32 tokens.
    res = _default_resolve(
        monkeypatch, 33, QuantType.per_Token, arch="gfx942", fp8=dtypes.i8
    )
    _assert_default_1stage(res)
    res = _default_resolve(
        monkeypatch, 32, QuantType.per_Token, arch="gfx942", fp8=dtypes.i8
    )
    assert res.plan.structure()[0] == asm
    res = _default_resolve(monkeypatch, 64, QuantType.per_Token, w=torch.uint32)
    assert res.plan.structure()[0][2] == "ck.gemm1"
    assert res.metadata.stage1.func.__name__ == "ck_moe_stage1"


def test_asm_int8_has_no_doweight_stage1_kernel():
    from aiter.moev2.segments.asm import AsmStage1

    doweight = {"doweight_stage1"}
    i8 = _problem(quant=QuantType.per_Token, fp8=dtypes.i8).with_features(add=doweight)
    fp8 = _problem(quant=QuantType.per_Token).with_features(add=doweight)
    assert "doweight_stage1" in AsmStage1().reject_reason(i8)
    assert AsmStage1().reject_reason(fp8) is None


def _assert_default_1stage(res, xbf16=False):
    """Legacy's default 1-stage metadata: `fused_moe_1stage` without a kernel name,
    BLOCK_SIZE_M, no split-K, no stage 2."""
    assert res.plan.structure()[0] == ("sort", "asm.fmoe_1stage")
    md = res.metadata
    assert md.stage1.func.__name__ == "fused_moe_1stage" and md.stage2 is None
    assert (md.run_1stage, md.block_m, md.ksplit, md.flat) == (True, 32, 0, 0)
    assert md.stage1.keywords["kernelName"] == ""
    assert md.stage1.keywords["xbf16"] is xbf16
    assert "run_1stage = True" in res.logs[0][1]
    assert res.logs[-1][1].startswith(
        f"[fused_moe] using 1stage{' xbf16' if xbf16 else ''} default for "
    )


def test_per_token_fp8_defaults_run_1stage_fmoe_in_its_region(monkeypatch):
    _assert_default_1stage(_default_resolve(monkeypatch, 17, QuantType.per_Token))
    res = _default_resolve(monkeypatch, 16, QuantType.per_Token, inter_dim=192)
    _assert_default_1stage(res)
    res = _default_resolve(monkeypatch, 16, QuantType.per_Token)
    assert res.plan.structure()[0] == (
        "sort", "quant.a1", "ck.gemm1", "quant.inter", "ck.gemm2"
    )  # fmt: skip
    assert (res.metadata.block_m, res.metadata.ksplit) == (32, 0)


_NO_G1U0_PLAN = (
    "no feasible plan",
    "ranking a16w16_g1u0[otherwise]: no listed structure is feasible",
)


def test_g1u0_keys_run_1stage_fmoe_where_legacy_does(monkeypatch):
    from aiter.moev2.migration import a8w8_ptok, a16w16

    monkeypatch.setattr(tuned, "lookup", lambda problem: None)

    def g1u0(token, arch="gfx942", **kw):
        p = _problem(token=token, arch=arch, **kw)
        return dataclasses.replace(p, layout=dataclasses.replace(p.layout, g1u1=False))

    bf16 = {"quant": QuantType.No, "fp8": dtypes.bf16}
    int8 = {"quant": QuantType.per_Token, "fp8": dtypes.i8}
    for path, fmt, branch, one_stage, two_stage in (
        (a16w16, bf16, "CK 2-stage g1u0", 255, 256),
        (a8w8_ptok, int8, "ASM stage 1 + CK stage 2 g1u0", 33, 32),
    ):
        p = g1u0(one_stage, **fmt)
        assert path.takes(p)
        _assert_default_1stage(resolver.resolve(p, path.SEGMENTS))
        res = resolver.resolve(g1u0(two_stage, **fmt), path.SEGMENTS)
        assert res.declined.endswith(f": {branch} is not ported"), res.declined
        # gfx950 has no g1u0 key in fused_moe_1stage_dict: nothing runs g1u0 there.
        res = resolver.resolve(g1u0(one_stage, arch="gfx950", **fmt), path.SEGMENTS)
        assert res.declined in _NO_G1U0_PLAN, res.declined
    assert not a8w8_ptok.takes(g1u0(64, quant=QuantType.per_Token))


def test_blockscale_defaults_take_the_legacy_block_m_and_ksplit(monkeypatch):
    calls = []

    def has_kernel(*args):
        calls.append(args)
        return 0

    res = _default_resolve(monkeypatch, 1, QuantType.per_1x128)
    assert (res.metadata.block_m, res.metadata.ksplit) == (16, 14)
    _assert_default_1stage(_default_resolve(monkeypatch, 33, QuantType.per_1x128))
    monkeypatch.setenv("AITER_XBFLOAT16", "1")
    res = _default_resolve(monkeypatch, 33, QuantType.per_1x128)
    _assert_default_1stage(res, xbf16=True)
    res = _default_resolve(monkeypatch, 33, QuantType.per_1x128, arch="gfx942")
    _assert_default_1stage(res)  # xbf16 is gfx950 only
    monkeypatch.delenv("AITER_XBFLOAT16")

    import aiter

    monkeypatch.setattr(aiter, "fmoe_fp8_blockscale_g1u1_has_kernel", has_kernel)
    _default_resolve(monkeypatch, 16, QuantType.per_1x128)
    _default_resolve(monkeypatch, 33, QuantType.per_1x128, inter_dim=192)
    assert calls == []  # legacy asks the kernel table only past the shape checks
    res = _default_resolve(monkeypatch, 33, QuantType.per_1x128)
    assert calls == [(ActivationType.Silu.value, False, 32, 256, 7168)]
    assert res.plan.structure()[0][2] == "ck.gemm1"
    assert (res.metadata.block_m, res.metadata.ksplit) == (64, 0)


_ROW_1STAGE = {
    "block_m": 64,
    "ksplit": 0,
    "run_1stage": 1,
    "kernelName1": "_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_pf2_silu_32x128E",
    "kernelName2": float("nan"),
    "flat": 0.0,
    "xbf16": 0,
}


def _tuned_1stage(monkeypatch, row, **kw):
    from aiter.moev2.migration import a8w8_blockscale

    monkeypatch.setattr(tuned, "lookup", lambda problem: row)
    p = _problem(quant=QuantType.per_1x128, **kw)
    return resolver.resolve(p, a8w8_blockscale.SEGMENTS)


def test_a_tuned_1stage_row_runs_fused_moe_1stage_as_legacy_builds_it(monkeypatch):
    res = _tuned_1stage(monkeypatch, _ROW_1STAGE)
    assert res.plan.structure()[0] == ("sort", "asm.fmoe_1stage")
    md = res.metadata
    assert md.stage1.func.__name__ == "fused_moe_1stage" and md.stage2 is None
    assert md.stage1.keywords["kernelName"] == _ROW_1STAGE["kernelName1"]
    assert md.stage1.keywords["xbf16"] is False and md.stage1.keywords["flat"] == 0
    assert (md.run_1stage, md.block_m, md.ksplit) == (1, 64, 0)
    assert res.logs[-1][1].startswith(
        f"[fused_moe] using 1stage (kernelName1={_ROW_1STAGE['kernelName1']!r}, "
        "kernelName2=nan) for "
    )
    for row, xbf16 in (
        ({**_ROW_1STAGE, "xbf16": 1}, True),
        # Without the column, legacy reads it from the kernel name.
        (
            {
                k: v
                for k, v in _ROW_1STAGE.items()
                if k != "xbf16" and k != "kernelName1"
            }
            | {"kernelName1": "_ZN5aiter47fmoe_bf16_blockscaleBf16_g1u1_vs_silu_1tgE"},
            True,
        ),
        ({**_ROW_1STAGE, "kernelName2": "Null"}, False),
    ):
        res = _tuned_1stage(monkeypatch, row)
        assert res.metadata.stage1.keywords["xbf16"] is xbf16, row
        assert ("1stage xbf16 (" in res.logs[-1][1]) is xbf16


def test_a_flat_1stage_row_runs_on_the_flat_pass_through(monkeypatch):
    for flat in (1.0, 2.0):
        res = _tuned_1stage(monkeypatch, {**_ROW_1STAGE, "flat": flat}, is_ep=True)
        assert res.plan.structure()[0] == ("sort.flat", "asm.fmoe_1stage")
        assert res.metadata.flat == int(flat)
        assert res.metadata.stage1.keywords["flat"] == int(flat)
    res = _tuned_1stage(monkeypatch, _ROW_1STAGE)
    assert res.plan.structure()[0] == ("sort", "asm.fmoe_1stage")


def test_only_the_1stage_fmoe_reads_the_flat_pass_through():
    from aiter.moev2.migration import a4w4

    plans = resolver.feasible_plans(_a4w4_problem(), a4w4.SEGMENTS)[0]
    names = {tuple(s.name for s in plan) for plan in plans}
    assert {n for n in names if "sort.flat" in n} == {("sort.flat", "asm.fmoe_1stage")}
    assert {n for n in names if "asm.fmoe_1stage" in n} == {
        ("sort", "asm.fmoe_1stage"),
        ("sort.flat", "asm.fmoe_1stage"),
    }


def test_1stage_rows_outside_the_plain_form_go_back_to_legacy(monkeypatch):
    stage2 = "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightA8W8blkscale_v3_Nswizzle0_Quant4_MulRoutedWeight1_F8_F8_B16"
    res = _tuned_1stage(monkeypatch, {**_ROW_1STAGE, "kernelName2": stage2})
    assert res.declined.startswith("1-stage row with kernelName1=")


def test_a_1stage_row_for_another_activation_is_discarded(monkeypatch):
    monkeypatch.setattr(resolver, "ONE_STAGE_ACTIVATIONS", ("gelu",))
    res = _tuned_1stage(monkeypatch, _ROW_1STAGE)
    assert res.logs[0] == (
        logging.WARNING,
        (
            "[fused_moe] discarding 1-stage tuned config for unsupported activation "
            "ActivationType.Silu; using default heuristics"
        ),
    )
    assert res.plan.source == "default"


def test_the_bypass_ranks_a_problem_with_a_1stage_row(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    res = _tuned_1stage(monkeypatch, _ROW_1STAGE, token=16)
    assert res.plan.source == "override"
    assert res.plan.structure()[0][2] == "ck.gemm1"
    assert "(kernelName1='', kernelName2='')" in res.logs[-1][1]


def test_the_relookup_logs_a_1stage_choice(monkeypatch, caplog):
    from aiter.moev2.migration import a8w8_ptok

    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    tuned.clear_cache()
    p = _problem(token=16, quant=QuantType.per_Token)
    with caplog.at_level(logging.INFO, logger="aiter"):
        res = a8w8_ptok.LOCAL.relookup(p, 64)
    assert res.plan.structure()[0] == ("sort", "asm.fmoe_1stage")
    assert "[fused_moe] using 1stage default for" in caplog.text


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


def test_g1u0_has_no_ck_plan():
    """CK stage 1 always splits w1 into gate and up; g1u0 calls outside the 1-stage
    `fmoe` keys stay on legacy."""
    from aiter.moev2.migration import a8w8_pt, a16w16

    p = _problem(quant=QuantType.No, fp8=dtypes.bf16)
    assert resolver.resolve(p, a16w16.SEGMENTS).plan is not None
    g1u0 = dataclasses.replace(p, layout=dataclasses.replace(p.layout, g1u1=False))
    assert a16w16.takes(g1u0)
    res = resolver.resolve(g1u0, a16w16.SEGMENTS)
    assert res.plan is None and res.declined in _NO_G1U0_PLAN, res.declined
    plans = resolver.feasible_plans(g1u0, a16w16.SEGMENTS)[0]
    assert all(s.name != "ck.gemm1" for plan in plans for s in plan)
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


def _a4w4_problem(
    token=64,
    shape=(7168, 256, 256, 8),
    doweight=False,
    activation=ActivationType.Silu,
    **kwargs,
):
    args = (token, *shape, dtypes.bf16, dtypes.fp4x2, dtypes.fp4x2)
    args += (QuantType.per_1x32, True, activation, doweight, 0, 0)
    return MoeProblem.from_legacy_key(*args, **kwargs, arch="gfx950", cu_num=256)


def _a4w4_resolve(monkeypatch, problem, row=None):
    from aiter.moev2.migration import a4w4

    monkeypatch.setattr(tuned, "lookup", lambda p: row)
    return resolver.resolve(problem, a4w4.SEGMENTS)


_MXMOE_ROW = {
    "block_m": 16,
    "ksplit": 0,
    "run_1stage": 0,
    "kernelName1": "flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_xcd4",
    "kernelName2": "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16",
}
_MXMOE_SHAPE = (5120, 640, 128, 3)


def test_a4w4_defaults_follow_the_legacy_branches(monkeypatch):
    res = _a4w4_resolve(monkeypatch, _a4w4_problem(is_shuffled=True))
    assert res.plan.structure()[0] == (
        "sort", "quant.a1", "flydsl.gemm1", "quant.inter", "flydsl.gemm2"
    )  # fmt: skip
    assert (res.metadata.block_m, res.metadata.ksplit) == (32, -1)
    kn1 = res.metadata.stage1.keywords["kernelName"]
    assert kn1 == "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2"
    assert "using heuristic FlyDSL fallback" in res.logs[-1][1]
    # doweight_stage1 rules FlyDSL out: CK-Tile split-K at decode sizes, else CK.
    dw = _a4w4_problem(token=1, doweight=True, is_shuffled=True)
    res = _a4w4_resolve(monkeypatch, dw)
    assert res.plan.structure()[0] == (
        "sort", "quant.a1", "cktile.gemm1", "quant.inter", "cktile.gemm2"
    )  # fmt: skip
    stage1 = res.metadata.stage1
    assert stage1.func.__name__ == "cktile_moe_stage1"
    assert stage1.keywords["post_activation_layout"] == "auto"
    assert res.metadata.block_m == 16
    assert stage1.keywords["split_k"] == res.metadata.ksplit > 1
    assert res.metadata.stage2.func.__name__ == "cktile_moe_stage2"
    assert not res.logs or "CK-Tile" not in res.logs[-1][1]
    dw = _a4w4_problem(token=4096, doweight=True, is_shuffled=True)
    res = _a4w4_resolve(monkeypatch, dw)
    assert res.plan.structure()[0][2] == "ck.gemm1"
    assert res.metadata.stage1.func.__name__ == "ck_moe_stage1"


def test_a4w4_situv2_and_swiglu_defaults_follow_the_legacy_branches(monkeypatch):
    from aiter.moev2.dispatch import overrides

    situv2 = functools.partial(_a4w4_problem, activation=ActivationType.Situv2)
    res = _a4w4_resolve(monkeypatch, situv2(is_shuffled=True))
    assert res.plan.structure()[0] == (
        "sort.aux", "quant.a1", "mxmoe.gemm1", "mxmoe.gemm2"
    )  # fmt: skip
    assert (res.metadata.block_m, res.metadata.ksplit) == (64, 0)
    assert res.metadata.stage1.keywords["kernelName1"] == (
        "flydsl_mxmoe_g1_a4w4_64x256x256_situv2"
    )
    assert "using heuristic MXMOE fallback" in res.logs[-1][1]
    # Bias rules the MXMOE fallback out: the FlyDSL heuristic, which ignores it.
    res = _a4w4_resolve(monkeypatch, situv2(is_shuffled=True, has_stage1_bias=True))
    assert res.plan.structure()[0][2] == "flydsl.gemm1"
    assert not (res.metadata.has_bias or res.metadata.stage2_has_bias)
    res = _a4w4_resolve(monkeypatch, situv2(is_shuffled=False))
    assert "CK-Tile split-K MXFP4 or CK 2-stage SiTUv2 is not ported" in res.declined
    # The legacy MXMOE fallback needs `cfg is None`: a bypassed tuned row skips it.
    monkeypatch.setattr(overrides, "bypass_tuned", lambda: 1)
    row = {"block_m": 32, "ksplit": 0, "run_1stage": 0}
    row.update(
        kernelName1="flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2",
        kernelName2="flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2",
    )
    res = _a4w4_resolve(monkeypatch, situv2(is_shuffled=True), row)
    assert res.plan is None and "a tuned row is bypassed" in res.declined

    swiglu = functools.partial(_a4w4_problem, activation=ActivationType.Swiglu)
    res = _a4w4_resolve(monkeypatch, swiglu(is_shuffled=True))
    assert res.plan.structure()[0][2] == "flydsl.gemm1"
    # The legacy `enable_bias`: on for Swiglu with a 16-bit output, bias or not.
    assert res.metadata.has_bias and res.metadata.stage2_has_bias
    # Unshuffled: CK-Tile with a split-K of 2 or more, the key's bias, no layout option.
    res = _a4w4_resolve(
        monkeypatch,
        swiglu(is_shuffled=False, has_stage1_bias=True, has_stage2_bias=True),
    )
    assert res.plan.structure()[0][2] == "cktile.gemm1"
    stage1 = res.metadata.stage1
    assert stage1.keywords["split_k"] == res.metadata.ksplit >= 2
    assert "post_activation_layout" not in stage1.keywords
    assert res.metadata.has_bias and res.metadata.stage2_has_bias
    assert "routing to CK-Tile (ROCM-25478)" in res.logs[-1][1]


def test_a4w4_tuned_cktile_stage2_rows(monkeypatch):
    swiglu = functools.partial(
        _a4w4_problem, activation=ActivationType.Swiglu, is_shuffled=True
    )
    row = {"block_m": 32, "ksplit": 0, "run_1stage": 0}
    row.update(
        kernelName1="flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2",
        kernelName2="cktile_moe2_bm32",
    )
    res = _a4w4_resolve(monkeypatch, swiglu(), row)
    assert res.plan.structure()[0][-1] == "cktile.gemm2"
    assert res.metadata.stage2.func.__name__ == "cktile_moe_stage2"
    # The legacy tuned FlyDSL branch: `enable_bias` on both stages.
    assert res.metadata.has_bias and res.metadata.stage2_has_bias
    # CK-Tile's MXFP4 stage 2 is unsafe when inter_dim % 256: the row is discarded.
    res = _a4w4_resolve(monkeypatch, swiglu(shape=(7168, 384, 256, 8)), row)
    assert any(
        "discarding unsafe CK-Tile MXFP4 stage2 config for inter_dim=384" in m
        for _, m in res.logs
    )
    assert res.plan.structure()[0][-1] == "flydsl.gemm2"
    # Legacy builds the alias as a CK stage 2 name in its tuned branches.
    row["kernelName2"] = "swiglu_mxfp4_bf16_cktile"
    assert _a4w4_resolve(monkeypatch, swiglu(), row).plan is None


_ALIAS = "swiglu_mxfp4_bf16_cktile"


def test_a16w4_cktile_alias_rows(monkeypatch):
    swiglu = functools.partial(
        _a16w4_problem,
        activation=ActivationType.Swiglu,
        token=1,
        shape=(3072, 512, 128, 4),
    )
    row = {"block_m": 64, "ksplit": 4, "run_1stage": 0}
    row.update(kernelName1=_ALIAS, kernelName2=_ALIAS)
    # Legacy reads the alias as "a tuned row exists" and runs its CK-Tile branches with
    # the row's ksplit.
    res = _a16w4_resolve(monkeypatch, swiglu(), row)
    assert " | ".join(res.plan.structure()[0]) == (
        "sort | quant.a1 | cktile.gemm1 | quant.inter | cktile.gemm2"
    )
    stage1 = res.metadata.stage1
    assert stage1.func.__name__ == "cktile_moe_stage1"
    assert stage1.keywords["split_k"] == res.metadata.ksplit == 4
    assert stage1.keywords["post_activation_layout"] == "standard"
    # Not preshuffled, legacy builds ASM stage 1 with a CK stage 2 named after the alias.
    res = _a16w4_resolve(monkeypatch, swiglu(is_shuffled=False), row)
    assert "alias tuned row outside the CK-Tile branches" in res.declined
    row["kernelName1"] = "flydsl_moe1_abf16_wfp4_bf16_t16x128x256_w2"
    res = _a16w4_resolve(monkeypatch, swiglu(), row)
    assert res.plan is None


def test_a16w4_defaults(monkeypatch):
    # SiTUv2: the FlyDSL a16w4 heuristic.
    res = _a16w4_resolve(monkeypatch, _a16w4_problem())
    assert res.plan.structure()[0][2] == "flydsl.gemm1"
    assert "abf16_wfp4" in res.metadata.stage1.keywords["kernelName"]
    # Swiglu: CK-Tile, unless its stage 2 is unsafe (inter_dim % 256) and FlyDSL takes
    # the shape; gate and up interleaved, the CK-Tile gate-up kernel.
    swiglu = functools.partial(_a16w4_problem, activation=ActivationType.Swiglu)
    res = _a16w4_resolve(monkeypatch, swiglu())
    assert res.metadata.stage1.func.__name__ == "cktile_moe_stage1"
    assert res.metadata.stage1.keywords["split_k"] >= 2
    res = _a16w4_resolve(monkeypatch, swiglu(shape=(4096, 384, 128, 8)))
    assert res.plan.structure()[0][2] == "flydsl.gemm1"
    from aiter.ops.flydsl.moe_common import GateMode

    interleaved = swiglu(gate_mode=GateMode.INTERLEAVE, gate_up_interleave=True)
    res = _a16w4_resolve(monkeypatch, interleaved)
    assert res.metadata.stage1.func.__name__ == "cktile_moe_stage1"
    assert res.metadata.stage1.keywords["split_k"] == 1
    # Relu2 with a gate-only w1: CK-Tile on gfx950, legacy CK on gfx942.
    relu2 = functools.partial(
        _a16w4_problem, activation=ActivationType.Relu2, g1u1=False
    )
    res = _a16w4_resolve(monkeypatch, relu2())
    assert res.metadata.stage1.func.__name__ == "cktile_moe_stage1"
    assert res.metadata.stage1.keywords["split_k"] == 1
    res = _a16w4_resolve(monkeypatch, relu2(arch="gfx942"))
    assert "CK 2-stage a16w4 is not ported" in res.declined


def test_a8w4_defaults_and_tuned_rows(monkeypatch):
    from aiter.moev2.migration import a8w4
    from aiter.ops.flydsl.moe_common import GateMode

    def resolve(problem, row=None):
        monkeypatch.setattr(tuned, "lookup", lambda p: row)
        return resolver.resolve(problem, a8w4.SEGMENTS)

    # Gate and up interleaved: the FlyDSL heuristic with fp8 names (Silu, SiTUv2) and
    # the CK-Tile gate-up kernel (Swiglu); separated, CK and CK-Tile.
    interleaved = {"gate_mode": GateMode.INTERLEAVE, "gate_up_interleave": True}
    for activation in (ActivationType.Silu, ActivationType.Situv2):
        res = resolve(_a8w4_problem(activation, **interleaved))
        assert res.plan.structure()[0][2] == "flydsl.gemm1"
        kn1 = res.metadata.stage1.keywords["kernelName"]
        assert kn1.startswith("flydsl_moe1_afp8_wfp4") and kn1.endswith("_gui")
    res = resolve(_a8w4_problem(ActivationType.Swiglu, **interleaved))
    assert res.metadata.stage1.func.__name__ == "cktile_moe_stage1"
    assert res.metadata.stage1.keywords["split_k"] == 1 and res.metadata.block_m == 32
    res = resolve(_a8w4_problem())
    assert res.plan.structure()[0][2] == "ck.gemm1"
    res = resolve(_a8w4_problem(ActivationType.Swiglu))
    assert res.metadata.stage1.func.__name__ == "cktile_moe_stage1"
    # Tuned FlyDSL rows: a stage 1 quantizing to fp8, a layout GEMM2.
    row = {
        "kernelName1": "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8",
        "kernelName2": "flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2",
        "block_m": 32,
        "ksplit": 0,
    }
    res = resolve(_a8w4_problem(), row)
    assert res.plan.structure()[0][2] == "flydsl.gemm1_q"
    assert res.metadata.fuse_quant == "fp8"
    layout = {
        **row,
        "kernelName2": "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x256_reduce_nt_sbm32",
    }
    res = resolve(_a8w4_problem(), layout)
    assert res.plan.structure()[0][-1] == "flydsl_layout.gemm2"
    assert res.metadata.stage1.keywords["out_dtype"] == "fp8"
    # Opus rows: the stage 2 launch from the row's sort block_m; legacy discards them for
    # weights not both preshuffled or with expert_mask.
    opus = {
        "kernelName1": "opus_moe1_afp8_wfp4_bf16_t64x384_gs_qgb3_aff32_p2_fp8",
        "kernelName2": "opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3584",
        "block_m": 64,
        "ksplit": 0,
    }
    res = resolve(_a8w4_problem(), opus)
    assert res.plan.structure()[0][2:] == ("opus.gemm1", "opus.gemm2")
    assert res.metadata.stage1.keywords["block_m"] == 64
    assert res.metadata.stage2.keywords["launch"].sort_block_m == 64
    assert res.metadata.fuse_quant == "fp8" and not res.metadata.skip_inter_quant
    for kwargs in ({"opus_weights_shuffled": False}, {"is_ep": True}):
        res = resolve(_a8w4_problem(**kwargs), opus)
        assert res.plan.source != "tuned"
        assert any("Opus" in m for _, m in res.logs), kwargs
    # A `cktile_*` kernelName1: ASM stage 1 under that name, unless a default MXFP4
    # branch takes the row (CK-Tile with the row's split-K).
    asm = {**opus, "kernelName1": "cktile_a8w4_bm64"}
    res = resolve(_a8w4_problem(), asm)
    assert res.plan.structure()[0][2] == "asm.stage1_cktile"
    assert res.metadata.stage1.keywords["kernelName"] == "cktile_a8w4_bm64"
    assert (
        "default MXFP4 branch"
        in resolve(_a8w4_problem(), {**asm, "ksplit": 2}).declined
    )


def test_mxfp8_defaults_and_tuned_rows(monkeypatch):
    from aiter.moev2.migration import mxfp8
    from aiter.ops.flydsl.moe_common import GateMode

    def resolve(problem, row=None):
        monkeypatch.setattr(tuned, "lookup", lambda p: row)
        return resolver.resolve(problem, mxfp8.SEGMENTS)

    # The FlyDSL heuristic (gate/up interleaved, or w1 tagged as not interleaved) adds
    # the Swiglu bias with fp8 weights too; gate and up separated otherwise, CK.
    for layout in (
        {"gate_mode": GateMode.INTERLEAVE, "gate_up_interleave": True},
        {"gate_up_interleave": False},
    ):
        for activation in (ActivationType.Silu, ActivationType.Swiglu):
            res = resolve(_mxfp8_problem(activation, **layout))
            assert res.plan.structure()[0][2] == "flydsl.gemm1", layout
            kn1 = res.metadata.stage1.keywords["kernelName"]
            assert kn1.startswith("flydsl_moe1_afp8_wfp8") and kn1.endswith("_gui")
            swiglu = activation == ActivationType.Swiglu
            assert res.metadata.has_bias == res.metadata.stage2_has_bias == swiglu
    for activation in (ActivationType.Silu, ActivationType.Swiglu):
        res = resolve(_mxfp8_problem(activation))
        assert res.plan.structure()[0][2:] == ("ck.gemm1", "quant.inter", "ck.gemm2")
    # Tuned rows (GLM5, MiniMax-M3): a stage 1 quantizing to fp8 into a layout GEMM2 or
    # a FlyDSL stage 2; the tuned rule adds no Swiglu bias with fp8 weights.
    row = {
        "kernelName1": "flydsl_moe1_afp8_wfp8_bf16_t32x64x256_w4_gui_kw2_fp8",
        "kernelName2": "flydsl_moe2_layout_afp8_wfp8_bf16_t32x128x256_atomic_nt_sbm32",
        "block_m": 32,
        "ksplit": 0,
    }
    res = resolve(_mxfp8_problem(ActivationType.Swiglu), row)
    assert res.plan.source == "tuned"
    assert res.plan.structure()[0][-1] == "flydsl_layout.gemm2"
    assert res.metadata.stage1.keywords["out_dtype"] == "fp8"
    assert not res.metadata.has_bias and not res.metadata.stage2_has_bias
    row = {
        **row,
        "kernelName1": "flydsl_moe1_afp8_wfp8_bf16_t32x64x256_w2_gui_fp8",
        "kernelName2": "flydsl_moe2_afp8_wfp8_bf16_t32x128x128_atomic_bnt2",
    }
    res = resolve(_mxfp8_problem(ActivationType.Swiglu), row)
    assert res.plan.structure()[0][2:] == ("flydsl.gemm1_q", "flydsl.gemm2")
    assert res.metadata.fuse_quant == "fp8"
    assert not res.metadata.has_bias and not res.metadata.stage2_has_bias


def test_a16wi4_fallback_and_tuned_rows(monkeypatch):
    from aiter.moev2.migration import a16wi4

    def resolve(problem, row=None):
        monkeypatch.setattr(tuned, "lookup", lambda p: row)
        return resolver.resolve(problem, a16wi4.SEGMENTS)

    # The untuned fallback: one tile per token tier, no split-K, a warning.
    for token, tile_m in ((64, 16), (4096, 32), (16384, 64)):
        res = resolve(_a16wi4_problem(token=token))
        assert " | ".join(res.plan.structure()[0]) == (
            "sort | quant.a1 | flydsl.gemm1 | quant.inter | flydsl.gemm2"
        )
        kn1 = res.metadata.stage1.keywords["kernelName"]
        assert kn1.startswith(f"flydsl_moe1_abf16_wint4_bf16_t{tile_m}x128x128")
        assert res.metadata.block_m == tile_m and res.metadata.ksplit == 0
        assert any("untuned a16wi4 fallback" in m for _, m in res.logs)
    row = {"block_m": 16, "ksplit": 0, "run_1stage": 0}
    row.update(
        kernelName1="flydsl_moe1_abf16_wint4_bf16_t16x64x128",
        kernelName2="flydsl_moe2_abf16_wint4_bf16_t16x128x128_reduce",
    )
    res = resolve(_a16wi4_problem(activation=ActivationType.Situv2), row)
    assert res.metadata.stage2.keywords["kernelName"] == row["kernelName2"]
    assert not res.metadata.has_bias and not res.metadata.stage2_has_bias


def test_mxmoe_bias_flags_follow_the_kernel_names(monkeypatch):
    p = _a4w4_problem(
        token=8, shape=_MXMOE_SHAPE, is_shuffled=True, has_stage2_bias=True
    )
    res = _a4w4_resolve(monkeypatch, p, _MXMOE_ROW)
    assert res.plan.structure()[0][2] == "mxmoe.gemm1"
    assert res.metadata.stage2_has_bias and not res.metadata.has_bias


def test_claims_split_mxfp4_weights_by_activation_format(monkeypatch):
    from aiter.moev2.migration import a4w4, local
    from aiter.ops.flydsl.moe_common import GateMode

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    call = _a4w4_op_call()
    assert a4w4.claims(call)
    fp4 = {"quant_dtype_a": dtypes.fp4x2}
    for change, claimed in (
        # SiTUv2 defaults to bf16 activations, Swiglu at decode sizes too.
        ({"activation": ActivationType.Situv2.value}, False),
        ({"activation": ActivationType.Situv2.value, **fp4}, True),
        ({"activation": ActivationType.Swiglu.value}, False),
        ({"activation": ActivationType.Swiglu.value, **fp4}, True),
        ({"quant_dtype_a": dtypes.fp8}, False),
        ({"bias1": torch.empty(1), "bias2": torch.empty(1)}, True),
        ({"hidden_pad": 64, "intermediate_pad": 64}, True),
        ({"swiglu_limit": 7.0, "beta": 2.0, "linear_beta": 2.0}, True),
        ({"quant_dtype_a2": dtypes.fp8}, False),
        ({"gate_mode": GateMode.INTERLEAVE.value}, False),
        ({"activation": ActivationType.Relu2.value, **fp4}, False),
    ):
        assert a4w4.claims({**call, **change}) == claimed, change


def test_a8w4_leaves_legacy_grouped_gfx1250_calls_to_legacy(monkeypatch):
    from aiter.moev2.migration import a8w4, local
    from aiter.ops.flydsl.moe_common import GateMode

    monkeypatch.setattr(local, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(local, "get_gfx_runtime", lambda: "gfx950")
    for name in ("AITER_USE_GROUPED_GEMM", "GPU_ARCHS", "AITER_FORCE_GFX1250"):
        monkeypatch.delenv(name, raising=False)
    call = {
        **_a4w4_op_call(),
        "quant_dtype_a": dtypes.fp8,
        "gate_mode": GateMode.INTERLEAVE.value,
        "w1_scale": torch.empty(1),
        "w2_scale": torch.empty(1),
    }
    assert a8w4.claims(call)
    monkeypatch.setenv("AITER_USE_GROUPED_GEMM", "1")
    assert a8w4.claims(call)  # gfx950 alone: the grouped branch returns None
    monkeypatch.setenv("GPU_ARCHS", "gfx950;gfx1250")
    assert not a8w4.claims(call)
    assert a8w4.claims({**call, "gate_mode": GateMode.SEPARATED.value})
    assert a8w4.claims({**call, "w2_scale": None})
    monkeypatch.setenv("AITER_DISABLE_GROUPED_A8W4", "1")
    assert a8w4.claims(call)


def test_a4w4_tuned_rows_pick_the_mxmoe_and_layout_v2_segments(monkeypatch):
    p = _a4w4_problem(token=8, shape=_MXMOE_SHAPE, is_shuffled=True)
    res = _a4w4_resolve(monkeypatch, p, _MXMOE_ROW)
    assert res.plan.structure()[0] == (
        "sort.aux", "quant.a1", "mxmoe.gemm1", "mxmoe.gemm2"
    )  # fmt: skip
    m = res.metadata
    assert (m.block_m, m.output_aux, m.fuse_quant, m.prequant) == (
        16,
        "opus",
        "fp4",
        False,
    )
    row = {
        "block_m": 32,
        "ksplit": 0,
        "run_1stage": 0,
        "kernelName1": "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w4_fp4",
        "kernelName2": "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32",
    }
    res = _a4w4_resolve(monkeypatch, _a4w4_problem(is_shuffled=True), row)
    assert res.plan.structure()[0] == (
        "sort", "quant.a1", "flydsl.gemm1_layout", "flydsl_layout.gemm2"
    )  # fmt: skip
    assert res.metadata.skip_inter_quant and res.metadata.output_aux is False


def test_a4w4_rows_legacy_discards_or_overrides(monkeypatch):
    # Legacy runs the FlyDSL heuristic over a tuned CK row on preshuffled weights.
    ck = {
        "block_m": 32,
        "ksplit": 0,
        "run_1stage": 0,
        "kernelName1": "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
        "kernelName2": "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    }
    res = _a4w4_resolve(monkeypatch, _a4w4_problem(is_shuffled=True), ck)
    assert res.plan is None and "tuned CK row for shuffled MXFP4" in res.declined
    # An inline-sort row is discarded under _disable_inline_sort and with expert_mask.
    for kwargs in ({"_disable_inline_sort": True}, {"is_ep": True}):
        p = _a4w4_problem(token=8, shape=_MXMOE_SHAPE, is_shuffled=True, **kwargs)
        res = _a4w4_resolve(monkeypatch, p, _MXMOE_ROW)
        assert res.plan.structure()[0][2] == "flydsl.gemm1", kwargs
        warnings = [text for level, text in res.logs if level == logging.WARNING]
        assert warnings[0].startswith("[fused_moe] discarding tuned inline-sort config")


def test_a4w4_inline_sort_fallback_resolves_again_without_inline_sort(
    monkeypatch, caplog
):
    from aiter.moev2.migration import a4w4, local

    class Call(dict):
        gate_mode = block_size_M = None

        def __missing__(self, name):
            return None

    dispatched = []

    def dispatch(args, kwargs, arch, cu_num):
        dispatched.append(dict(kwargs))
        return fallback

    first = (types.SimpleNamespace(metadata=None), None)
    fallback = (types.SimpleNamespace(metadata=None), None)
    monkeypatch.setattr(local, "_is_mxfp4_inline_sort", lambda metadata: True)
    monkeypatch.setattr(local, "_mxfp4_inline_sort_unsupported", lambda *a, **k: "x")
    monkeypatch.setattr(a4w4.LOCAL, "dispatch", dispatch)
    with caplog.at_level(logging.WARNING, logger="aiter"):
        got = a4w4.LOCAL._inline_sort_fallback(
            Call(), (), {"_disable_inline_sort": False}, first
        )
    assert got is fallback and dispatched == [{"_disable_inline_sort": True}]
    assert (
        "inline-sort config is unsupported (x); using default heuristics" in caplog.text
    )
    # A fallback key v2 declines goes back to legacy before the warning: it logs it.
    fallback = None
    caplog.clear()
    got = a4w4.LOCAL._inline_sort_fallback(
        Call(), (), {"_disable_inline_sort": False}, first
    )
    assert got is None and "inline-sort" not in caplog.text


def test_the_relookup_ignores_disable_inline_sort(monkeypatch):
    from aiter.moev2.migration import a4w4

    seen = []
    monkeypatch.setattr(a4w4.LOCAL, "resolve", lambda p, relookup: seen.append(p))
    p = _a4w4_problem(_disable_inline_sort=True, config_file="/tmp/t.csv")
    a4w4.LOCAL._relookup(p, 128)
    assert dict(seen[0].extras) == {"tuned_table": "/tmp/t.csv"}


# 3.7 FHMoE


def test_fhmoe_helpers_moved_verbatim():
    import aiter.fhmoe as legacy_fhmoe
    from aiter.moev2.segments import flydsl_fhmoe

    for name in (
        "_validate_fhmoe_contract",
        "_flydsl_fhmoe_stage1_wrapper",
        "_flydsl_fhmoe_stage2_wrapper",
        "_validate_fhmoe_metadata_consistency",
        "_is_fhmoe_metadata_compatible",
        "_is_dsv4_i384_metadata",
        "_valid_fhmoe_kernel_geometry",
        "_config_has_mxfp8_fhmoe_geometry",
        "_config_has_mxfp8_fhmoe_contract",
        "_is_dsv4_i384_fhmoe_contract",
    ):
        assert getattr(legacy_fhmoe, name) is getattr(flydsl_fhmoe, name), name


def test_only_the_fhmoe_path_takes_shared_expert_keys():
    from aiter.moev2.migration import fhmoe
    from aiter.ops.flydsl.moe_common import GateMode

    interleave = {"gate_mode": GateMode.INTERLEAVE}
    taken = (
        _a4w4_problem(),
        _a8w4_problem(**interleave),
        _mxfp8_problem(**interleave),
        _a8w4_problem(swiglu_limit=7.0, **interleave),
        dataclasses.replace(
            _a4w4_problem(),
            layout=dataclasses.replace(
                _a4w4_problem().layout, hidden_pad=64, intermediate_pad=64
            ),
        ),
    )
    # Outside the `fhmoe_` contract or its activation formats.
    not_taken = (
        _a8w4_problem(),
        _mxfp8_problem(),
        _a4w4_problem(activation=ActivationType.Swiglu),
        _a4w4_problem(is_ep=True),
        _a4w4_problem(has_stage1_bias=True),
        _problem(quant=QuantType.per_Token),
    )
    modules = _local_path_modules() + [fhmoe]
    for p in taken:
        assert [m.NAME for m in modules if m.takes(api.fhmoe_problem(p))] == ["fhmoe"]
        assert not fhmoe.takes(p)
    for p in not_taken:
        assert not any(m.takes(api.fhmoe_problem(p)) for m in modules), p


def test_fhmoe_stages_run_the_flydsl_kernels_through_the_fhmoe_wrappers(monkeypatch):
    from aiter.moev2.migration import fhmoe, mxfp8
    from aiter.moev2.segments import flydsl_fhmoe
    from aiter.ops.flydsl.moe_common import GateMode

    monkeypatch.setattr(tuned, "lookup", lambda p: None)
    p = _mxfp8_problem(gate_mode=GateMode.INTERLEAVE, gate_up_interleave=True)
    plain = resolver.resolve(p, mxfp8.SEGMENTS)
    res = resolver.resolve(api.fhmoe_problem(p), fhmoe.SEGMENTS)
    assert res.plan.structure()[0] == (
        "sort",
        "quant.a1",
        "flydsl_fhmoe.gemm1",
        "quant.inter",
        "flydsl_fhmoe.gemm2",
    )
    assert res.metadata.block_m == plain.metadata.block_m
    assert res.logs == plain.logs
    for stage, wrapper in (
        ("stage1", flydsl_fhmoe._flydsl_fhmoe_stage1_wrapper),
        ("stage2", flydsl_fhmoe._flydsl_fhmoe_stage2_wrapper),
    ):
        got, want = getattr(res.metadata, stage), getattr(plain.metadata, stage)
        assert got.func is wrapper
        assert (got.args, got.keywords) == (want.args, want.keywords)
    c = {
        "shared_w1": "w1",
        "shared_w1_scale": "s1",
        "shared_w2": "w2",
        "shared_w2_scale": "s2",
        "shared_expert_id": 255,
        "swiglu_limit": 7.0,
        "clamp_shared": False,
    }
    stage1, stage2 = (
        res.plan.starting(op).segment for op in (ops.Op.GEMM1, ops.Op.GEMM2)
    )
    assert stage1.call_args(c) == {
        "shared_w1": "w1",
        "shared_w1_scale": "s1",
        "shared_expert_id": 255,
        "swiglu_limit": 7.0,
        "clamp_shared": False,
    }
    assert stage2.call_args(c) == {
        "shared_w2": "w2",
        "shared_w2_scale": "s2",
        "shared_expert_id": 255,
    }
    # Separated gate/up without the w1 tag: legacy raises, v2 declines.
    res = resolver.resolve(api.fhmoe_problem(_mxfp8_problem()), fhmoe.SEGMENTS)
    assert res.plan is None


def test_fhmoe_capability_checks_match_legacy_on_the_shipped_table(monkeypatch):
    import aiter.fhmoe as legacy_fhmoe
    from aiter.moev2.migration import fhmoe
    from aiter.ops.flydsl.moe_common import GateMode

    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG", raising=False)
    hy4 = {
        "model_dim": 6144,
        "inter_dim": 256,
        "experts": 257,
        "topk": 9,
        "dtype": torch.bfloat16,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": GateMode.INTERLEAVE,
        "doweight_stage1": False,
    }
    for m in (1, 7, 64, 1000, 8192, 70000):
        assert fhmoe.supports_dsv4_i384_fhmoe(m) == (
            legacy_fhmoe.supports_dsv4_i384_fhmoe(m)
        ), m
        assert fhmoe.supports_mxfp8_fhmoe_config(m, **hy4) == (
            legacy_fhmoe.supports_mxfp8_fhmoe_config(m, **hy4)
        ), m
    assert fhmoe.supports_dsv4_i384_fhmoe(64)
    assert fhmoe.supports_mxfp8_fhmoe_config(64, **hy4)
    with pytest.raises(ValueError, match="partially matches"):
        fhmoe._resolve_mxfp8_fhmoe_config_file(64, **{**hy4, "hidden_pad": 64})
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    assert not fhmoe.supports_dsv4_i384_fhmoe(64)


def test_fhmoe_call_raises_the_legacy_api_errors():
    import re

    tensor = torch.empty(0)
    weight = torch.empty(0, dtype=dtypes.fp8)
    call = {
        "hidden_states": tensor,
        "w1": weight,
        "w2": weight,
        "topk_weight": tensor,
        "topk_ids": tensor,
        "shared_w1": weight,
        "shared_w2": weight,
        "shared_w1_scale": tensor,
        "shared_w2_scale": tensor,
        "shared_expert_id": 0,
    }
    for change in (
        {},
        {"quant_type_a": QuantType.per_1x32},
        {"quant_dtype_a": dtypes.bf16},
        {"quant_dtype_a2": dtypes.fp8},
        {"a1_scale": tensor, "a2_scale": tensor},
    ):
        with pytest.raises(Exception) as legacy_error:
            legacy.fused_moe(**call, **change)
        with pytest.raises(
            type(legacy_error.value), match=re.escape(str(legacy_error.value))
        ):
            api.fused_moe(**call, **change)


def test_fhmoe_leaves_legacy_grouped_gfx1250_calls_to_legacy(monkeypatch):
    from aiter.moev2.migration import fhmoe
    from aiter.ops.flydsl.moe_common import GateMode

    for name in ("AITER_USE_GROUPED_GEMM", "GPU_ARCHS", "AITER_FORCE_GFX1250"):
        monkeypatch.delenv(name, raising=False)
    call = {
        "w1": torch.empty(1, dtype=dtypes.fp4x2),
        "gate_mode": GateMode.INTERLEAVE.value,
        "w1_scale": torch.empty(1),
        "w2_scale": torch.empty(1),
    }
    assert fhmoe.claims(call)
    monkeypatch.setenv("AITER_USE_GROUPED_GEMM", "1")
    monkeypatch.setenv("GPU_ARCHS", "gfx950;gfx1250")
    assert not fhmoe.claims(call)
    # The grouped branch runs fp8 activations into MXFP4 weights only.
    assert fhmoe.claims({**call, "w1": torch.empty(1, dtype=dtypes.fp8)})
    assert fhmoe.claims({**call, "gate_mode": GateMode.SEPARATED.value})


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
