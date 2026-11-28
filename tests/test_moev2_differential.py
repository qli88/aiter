# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code.

The CPU test compares dispatch plans and log lines on every snapshot key a v2 path
claims. The GPU test runs tests/moe_dispatch/gpu_differential.py for each registered
path: same kernels on the same inputs, same outputs and log lines as legacy. It runs
only with AITER_MOEV2_GPU_DIFFERENTIAL=1, so CI (which runs this file) skips it.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import json
import os
import pathlib
import subprocess
import sys
from functools import partial
from types import SimpleNamespace

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def test_whole_graph_launches_are_compared_without_a_stage1_record(monkeypatch):
    import torch
    from moe_dispatch import record
    from moe_dispatch.differential import compare_launches

    import aiter.fused_moe_registry as registry
    from aiter.fused_moe_registry import FusedMoeRequest

    def whole_graph(request, config):
        assert config == "unit_config"
        return request.hidden_states

    def request():
        return FusedMoeRequest(
            hidden_states=torch.ones(1, 2),
            w1=torch.ones(1, 2, 2),
            w2=torch.ones(1, 2, 2),
            topk_weight=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        )

    legacy_request, v2_request = request(), request()
    monkeypatch.setitem(registry._IMPLEMENTATIONS, "differential_test", whole_graph)
    kernel_name = registry.make_fused_moe_impl_kernel_name(
        "differential_test", "unit_config"
    )
    legacy_impl = registry.resolve_fused_moe_impl(kernel_name)
    v2_impl = registry.resolve_fused_moe_impl(kernel_name)
    binding = record.normalize(legacy_impl)
    assert binding["config"] == "unit_config"
    assert binding["implementation"].endswith(".<locals>.whole_graph")
    call = {"topk_ids": legacy_request.topk_ids, "expert_mask": None}
    problems = compare_launches(
        [("whole_graph", partial(legacy_impl, legacy_request))],
        [("whole_graph", partial(v2_impl, v2_request))],
        call,
        torch.empty(1, 2),
        torch.empty(1, 2),
    )
    assert not problems


def test_whole_graph_gpu_case_only_runs_on_its_tuned_architecture():
    from moe_dispatch import gpu_differential

    whole_graph = (1, 4096, 192, 192, 8, "Silu", False, "bf16", False)
    assert whole_graph in gpu_differential.cases("a8w8_pt", gfx="gfx942", cu_num=80)
    assert whole_graph not in gpu_differential.cases(
        "a8w8_pt", gfx="gfx950", cu_num=304
    )


def _kernel(*args, **kwargs):
    pass


def _stage2(a2, sorted_ids, block_m=32):
    return [("stage2", partial(_kernel, a2, sorted_ids, block_m=block_m))]


def _compare(legacy, v2, legacy2=None):
    import torch
    from moe_dispatch.differential import compare_launches

    call = {"topk_ids": torch.zeros(4, 2, dtype=torch.int32), "expert_mask": None}
    out = torch.empty(1)
    return compare_launches(legacy, v2, call, out, out, legacy2)


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_launch_floats_may_differ_by_rounding_only(dtype):
    import torch

    dtype = getattr(torch, dtype)
    ids = torch.arange(8, dtype=torch.int32)
    a2 = torch.linspace(0.5, 4, 8).to(dtype)
    # The next representable values up (positive, so the raw bits plus one).
    bits = torch.uint8 if a2.element_size() == 1 else torch.int16
    one_ulp = (a2.view(bits) + 1).view(dtype)
    assert not _compare(_stage2(a2, ids), _stage2(one_ulp, ids))
    far = (a2.float() * 1.5).to(dtype)
    problems = _compare(_stage2(a2, ids), _stage2(far, ids))
    assert list(problems) == [("stage2", 0)], problems
    # Legacy's own second run spreads as far: rounding of a non-repeatable kernel.
    assert not _compare(_stage2(a2, ids), _stage2(far, ids), _stage2(far, ids))


def test_launch_integers_and_arguments_stay_exact():
    import torch

    a2 = torch.ones(8, dtype=torch.bfloat16)
    ids = torch.arange(8, dtype=torch.int32)
    shifted = ids.clone()
    shifted[3] += 1
    spread = _stage2(a2, shifted)
    assert list(_compare(_stage2(a2, ids), _stage2(a2, shifted), spread)) == [
        ("stage2", 1)
    ]
    assert list(_compare(_stage2(a2, ids), _stage2(a2, ids, block_m=64))) == [
        ("stage2", "block_m")
    ]


def test_launch_non_finite_values_must_match():
    import torch

    ids = torch.arange(4, dtype=torch.int32)
    a2 = torch.tensor([1.0, 2.0, float("inf"), 4.0], dtype=torch.bfloat16)
    nan = a2.clone()
    nan[2] = float("nan")
    assert list(_compare(_stage2(a2, ids), _stage2(nan, ids))) == [("stage2", 0)]


def test_launch_equal_nans_pass_and_legacy_noise_excuses_no_new_ones():
    import torch

    ids = torch.arange(4, dtype=torch.int32)
    legacy = torch.tensor([1.0, 2.0, float("nan"), 4.0], dtype=torch.bfloat16)
    # Same NaN, and a rounding flip elsewhere so the tensors are not bitwise equal.
    flipped = legacy.clone()
    flipped[0] = (legacy[:1].view(torch.int16) + 1).view(torch.bfloat16)
    assert not _compare(_stage2(legacy, ids), _stage2(flipped, ids))
    # Legacy's second run has inf where the first has 3: v2 may match either run there,
    # but a NaN in another slot is still a difference.
    first = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    second = torch.tensor([1.0, 2.0, float("inf"), 4.0], dtype=torch.bfloat16)
    assert not _compare(_stage2(first, ids), _stage2(second, ids), _stage2(second, ids))
    new_nan = second.clone()
    new_nan[0] = float("nan")
    assert list(
        _compare(_stage2(first, ids), _stage2(new_nan, ids), _stage2(second, ids))
    ) == [("stage2", 0)]


_SLOTS = [(k << 24) | t for t in range(4) for k in range(2)]


def _sorted_launches(
    inter,
    a1_scale,
    slots=_SLOTS,
    func=_kernel,
    a1=None,
    out=None,
    num_valid=(12,),
    **stage1_kw,
):
    """A 2-stage recording over 4 tokens x topk 2 in blocks of 4: 8 routed rows of
    expert 0 (in the order of `slots`), then a block of padding (token id 4)."""
    import torch

    sorted_ids = torch.tensor(list(slots) + [4] * 4, dtype=torch.int32)
    stage1 = partial(
        func,
        torch.ones(4, 64, dtype=torch.bfloat16) if a1 is None else a1,
        None,
        None,
        sorted_ids,
        torch.zeros(3, dtype=torch.int32),
        torch.tensor(num_valid, dtype=torch.int32),
        out,
        2,
        block_m=4,
        a1_scale=a1_scale,
        **stage1_kw,
    )
    return [("stage1", stage1), ("stage2", partial(_kernel, inter, block_m=4))]


def _compare_sorted(legacy, v2, legacy2=None, **inputs):
    import torch
    from moe_dispatch.differential import compare_launches

    call = {
        "topk_ids": torch.zeros(4, 2, dtype=torch.int32),
        "expert_mask": None,
        "hidden_states": torch.empty(4, 64),
        "w1": torch.empty(2, 64, 64),
        "w2": torch.empty(2, 64, 32),
        **inputs,
    }
    out = torch.empty(1)
    return compare_launches(legacy, v2, call, out, out, legacy2)


def test_sorted_rows_compare_in_any_order_inside_an_expert():
    import torch

    from aiter import dtypes

    scale = torch.zeros(32, 8, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    raw = torch.randint(0, 256, (12, 16), dtype=torch.uint8)
    inter = raw.view(dtypes.fp4x2)
    # v2's sort put rows 0 and 5 of the expert the other way round, and stage 1 wrote
    # its sorted rows in that order: the same rows, so no difference.
    order = [5, 1, 2, 3, 4, 0, 6, 7]
    swapped = [_SLOTS[i] for i in order]
    moved = raw.clone()
    moved[:8] = raw[order]
    moved = moved.view(dtypes.fp4x2)
    legacy = _sorted_launches(inter, scale)
    assert not _compare_sorted(legacy, _sorted_launches(moved, scale, swapped))
    # The same sort, but the rows written in legacy's order: row 0 and 5 differ.
    problems = _compare_sorted(legacy, _sorted_launches(inter, scale, swapped))
    assert list(problems) == [("stage2", 0)], problems


def test_caller_weight_scales_compare_whole_whatever_the_sort_order():
    import torch

    from aiter import dtypes

    scale = torch.zeros(32, 8, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    inter = torch.zeros(12, 16, dtype=torch.bfloat16)
    # As many rows as a sorted MX scale, but the caller's: not read in sorted order.
    w1_scale = torch.randint(0, 256, (32, 8), dtype=torch.uint8).view(dtypes.fp8_e8m0)
    swapped = [_SLOTS[i] for i in [5, 1, 2, 3, 4, 0, 6, 7]]
    legacy = _sorted_launches(inter, scale, w1_scale=w1_scale)
    v2 = _sorted_launches(inter, scale, swapped, w1_scale=w1_scale)
    assert not _compare_sorted(legacy, v2, w1_scale=w1_scale)


def test_sort_layout_differences_are_still_reported():
    import torch

    from aiter import dtypes

    scale = torch.zeros(32, 8, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    inter = torch.zeros(12, 16, dtype=torch.bfloat16)
    # A routed row at a padding position: the layout differs, whatever the order.
    shifted = list(_SLOTS[:7]) + [4]
    legacy = _sorted_launches(inter, scale)
    v2 = _sorted_launches(inter, scale, shifted)
    v2[0][1].args[3][8] = _SLOTS[7]
    assert list(_compare_sorted(legacy, v2)) == [("stage1", "sort")]


def _int8_runs(stage1_func):
    """`run(n)`: a recording whose stage 2 reads an int8 tensor of zeros but one value
    `n`, after a stage 1 launched as `stage1_func`."""
    import torch

    from aiter import QuantType

    def run(n):
        a2 = torch.zeros(4, 2, 8, dtype=torch.int8)
        a2[1, 0, 3] = n
        stage1 = _sorted_launches(
            None, None, func=stage1_func, quant_type=QuantType.per_Token
        )[0]
        return [stage1, ("stage2", partial(_kernel, a2, block_m=4))]

    return run


def test_integers_after_an_unrepeatable_stage1_follow_legacy_spread():
    from aiter.moev2.segments.asm import asm_stage1

    run = _int8_runs(asm_stage1)
    # Legacy's two runs 3 apart: v2 may be up to 6 off.
    assert not _compare_sorted(run(0), run(6), run(3))
    assert list(_compare_sorted(run(0), run(7), run(3))) == [("stage2", 0)]
    # Legacy repeats itself on this case: one step.
    assert not _compare_sorted(run(0), run(1), run(0))
    assert list(_compare_sorted(run(0), run(2), run(0))) == [("stage2", 0)]


def test_integers_after_other_stage1_kernels_stay_bitwise_equal():
    run = _int8_runs(_kernel)
    assert list(_compare_sorted(run(0), run(1), run(1))) == [("stage2", 0)]


def test_unrepeatable_stage1_inputs_and_num_valid_ids_stay_bitwise_equal():
    import torch

    from aiter import QuantType
    from aiter.moev2.segments.asm import asm_stage1

    def run(a1_value, num_valid=(12, 4)):
        a1 = torch.zeros(4, 64, dtype=torch.int8)
        a1[0, 0] = a1_value
        return _sorted_launches(
            None,
            None,
            func=asm_stage1,
            a1=a1,
            num_valid=num_valid,
            quant_type=QuantType.per_Token,
        )[:1]

    # The int8 activations stage 1 reads are its input, not its output.
    assert list(_compare_sorted(run(0), run(1), run(0))) == [("stage1", 0)]
    # num_valid_ids is compared whole (the sort layout reads only its first value).
    problems = _compare_sorted(run(0), run(0, (12, 5)), run(0))
    assert list(problems) == [("stage1", 5)], problems


def test_only_failures_of_unrepeatable_legacy_cases_may_be_rerun():
    from moe_dispatch.differential import _LEGACY_NOISE, may_be_legacy_noise

    assert may_be_legacy_noise(f"a8w8_pt: values differ {_LEGACY_NOISE}ASM stage 1]")
    assert may_be_legacy_noise("a16wi4: legacy output not repeatable: ...")
    assert not may_be_legacy_noise("a4w4: launches differ: stage1[3]: values differ")


def test_asm_1x128_stage1_buffer_compares_rows_and_scales_in_their_own_dtypes():
    import torch

    from aiter import QuantType, dtypes
    from aiter.moev2.segments.asm import asm_stage1

    # 4 tokens x topk 2 x inter 128 fp8 rows, then 1 row holding 4 x 2 x 128 / 128 fp32
    # scales (32 bytes) and 224 bytes nobody writes.
    a1 = torch.zeros(4, 64, dtype=torch.uint8).view(dtypes.fp8)
    a1_scale = torch.ones(4, 1, dtype=torch.float32)
    rows = torch.randint(0, 100, (4, 2, 128), dtype=torch.uint8)
    scales = torch.linspace(0.5, 2.0, 8, dtype=torch.float32)

    def buffer(tail, scale_values=scales):
        extra = torch.cat([scale_values.view(torch.uint8), tail]).view(1, 2, 128)
        return torch.cat([rows, extra]).view(dtypes.fp8)

    def run(out):
        return _sorted_launches(
            torch.zeros(1),
            a1_scale,
            func=asm_stage1,
            a1=a1,
            out=out,
            quant_type=QuantType.per_1x128,
        )

    tail = torch.randint(0, 256, (224,), dtype=torch.uint8)
    garbage = tail.clone()
    garbage[:16] = 0x7F  # NaN as fp8, in bytes nobody writes
    assert not _compare_sorted(run(buffer(tail)), run(buffer(garbage)))
    far = scales.clone()
    far[3] *= 1.5
    problems = _compare_sorted(run(buffer(tail)), run(buffer(tail, far)))
    assert list(problems) == [("stage1", "6.scales")], problems


def test_sorted_fp4_rows_compare_bitwise_at_the_routed_rows_only():
    import torch

    from aiter import dtypes

    def fp4(raw):
        return raw.view(dtypes.fp4x2)

    scale = torch.zeros(32, 8, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    inter = torch.randint(0, 256, (12, 16), dtype=torch.uint8)
    padding = inter.clone()
    padding[10] ^= 1
    assert not _compare_sorted(
        _sorted_launches(fp4(inter), scale), _sorted_launches(fp4(padding), scale)
    )
    routed = inter.clone()
    routed[3, 0] ^= 1
    problems = _compare_sorted(
        _sorted_launches(fp4(inter), scale), _sorted_launches(fp4(routed), scale)
    )
    assert (
        list(problems) == [("stage2", 0)] and "values differ" in problems["stage2", 0]
    )


def test_sorted_mx_scales_compare_only_the_bytes_the_quant_kernels_write():
    import torch
    from moe_dispatch.differential import _sorted_scale_defined

    from aiter import dtypes

    inter = torch.zeros(12, 16, dtype=torch.bfloat16)
    raw = torch.randint(0, 256, (32, 8), dtype=torch.uint8)
    # model_dim 64: 2 groups of 32, padded to 8 columns; the 8 routed rows.
    defined = set(_sorted_scale_defined(torch.arange(8), 2, 8).tolist())
    unwritten = next(i for i in range(raw.numel()) if i not in defined)
    written = min(defined)

    def flipped(index):
        bits = raw.clone().view(-1)
        bits[index] ^= 1
        return bits.view(32, 8).view(dtypes.fp8_e8m0)

    legacy = _sorted_launches(inter, raw.view(dtypes.fp8_e8m0))
    assert not _compare_sorted(legacy, _sorted_launches(inter, flipped(unwritten)))
    problems = _compare_sorted(legacy, _sorted_launches(inter, flipped(written)))
    assert list(problems) == [("stage1", "a1_scale")], problems


def test_native_a4w4_scale_layout_has_16_row_chunks_and_duplicated_bytes():
    import torch
    from moe_dispatch.differential import _sorted_scale_defined

    # Sorted row 17, scale column 5, 16 padded columns: chunk 1 (128 dwords each),
    # lane 1, column group (ku 0, wave group 1) at dword 1 * 128 + 1 * 16 + 1, its
    # 16-bit half 1 (bytes 2 and 3). The generic layout: chunk 0, row half 1, byte 71.
    native = _sorted_scale_defined(torch.tensor([17]), 6, 16, native=True)
    assert native[10:12].tolist() == [145 * 4 + 2, 145 * 4 + 3]
    assert _sorted_scale_defined(torch.tensor([17]), 6, 16)[5].item() == 71


def test_native_a4w4_stage2_scale_compares_both_bytes_in_sorted_order():
    import torch

    inter = torch.zeros(12, 16, dtype=torch.bfloat16)
    raw = torch.randint(0, 256, (32, 8), dtype=torch.uint8)

    def run(a2_scale, slots=_SLOTS):
        stage1, (_, stage2) = _sorted_launches(
            inter, None, slots, native_scale_layout=True
        )
        return [stage1, ("stage2", partial(stage2, a2_scale=a2_scale))]

    # Inter dim 32: one scale column, the bytes 4 * row and 4 * row + 1 of chunk 0.
    swapped = [_SLOTS[i] for i in [5, 1, 2, 3, 4, 0, 6, 7]]
    moved = raw.clone().view(-1)
    moved[[0, 1, 20, 21]] = moved[[20, 21, 0, 1]]
    assert not _compare_sorted(run(raw), run(moved.view(32, 8), swapped))
    second_byte = raw.clone().view(-1)
    second_byte[1] ^= 1
    problems = _compare_sorted(run(raw), run(second_byte.view(32, 8)))
    assert list(problems) == [("stage2", "a2_scale")], problems


def test_every_gpu_case_variant_is_defined():
    from moe_dispatch import gpu_differential

    variants = {
        gpu_differential.case_variant(case)
        for path in gpu_differential._CASES
        for case in gpu_differential.cases(path, gfx="gfx942", cu_num=80)
    } - {None}
    assert variants <= set(gpu_differential._VARIANTS)
    for path in gpu_differential._CASES:
        for case in gpu_differential.cases(path, gfx="gfx942", cu_num=80):
            # Field 10 is the weight format, field 11 the call options.
            assert len(case) in (9, 10, 11), case
            assert len(case) < 10 or case[9] not in gpu_differential._VARIANTS, case
            assert len(case) < 11 or isinstance(case[10], dict), case
    assert set(gpu_differential._RAISES) <= variants


@pytest.mark.parametrize(
    "module, known",
    [
        ("module_moe_ck2stages_abi2_f8_f8_preshuffle_on_b16_silu_per_1x32_mulWeightStage2", True),
        ("module_moe_ck2stages_abi2_f8_fp4x2_preshuffle_on_b16_situv2_per_1x32_mulWeightStage2", True),
        ("module_moe_ck2stages_f8_f8_preshuffle_on_b16_swiglu_per_1x128_mulWeightStage2", True),
        ("module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2", False),
        ("module_moe_ck2stages_abi2_bf16_bf16_preshuffle_off_b16_silu_no_mulWeightStage2", False),
        ("module_moe_asm", False),
    ],
)  # fmt: skip
def test_only_legacy_modules_that_never_build_may_fail_to_build(module, known):
    from moe_dispatch import gpu_differential

    assert bool(gpu_differential._KNOWN_BUILD_FAILS.match(module)) is known


def test_perf_baseline_rejects_partial_token_results(monkeypatch):
    from moe_dispatch import perf_baseline

    result = SimpleNamespace(stdout='@@RESULT@@[{"token": 1, "us": 2.5}]\n', stderr="")
    monkeypatch.setattr(perf_baseline.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match=r"missing=\['16'\]"):
        perf_baseline.run_case(perf_baseline.CASES[0], (1, 16))


@pytest.fixture
def perf_check(monkeypatch, tmp_path, capsys):
    """Run `perf_baseline.main` on a fake gfx942 with baselines under tmp_path; the
    measurement returns `measured` and records which cases it was asked for."""
    from moe_dispatch import perf_baseline

    measured, asked = {}, []

    def measure(gfx, names=None):
        asked.append(names)
        return dict(measured)

    monkeypatch.setattr(perf_baseline, "_gfx", lambda: "gfx942")
    monkeypatch.setattr(perf_baseline, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(perf_baseline, "measure", measure)

    def check(*argv, baseline=None):
        if baseline is not None:
            (tmp_path / "perf_gfx942.json").write_text(
                json.dumps({"gfx": "gfx942", "us": baseline})
            )
        code = perf_baseline.main(["--check", *argv])
        return code, capsys.readouterr().out

    return SimpleNamespace(check=check, measured=measured, asked=asked)


def test_perf_check_rejects_requested_cases_without_a_baseline(perf_check):
    code, out = perf_check.check(
        "--case", "qwen3_235b_tp8_bf16", baseline={"dsv3_tp8_pt": {"1": 10.0}}
    )
    assert code == 1 and perf_check.asked == []
    assert "['qwen3_235b_tp8_bf16']" in out
    assert "--record --case qwen3_235b_tp8_bf16" in out


def test_perf_baseline_refuses_an_arch_without_cases(
    perf_check, monkeypatch, tmp_path, capsys
):
    from moe_dispatch import perf_baseline

    assert not any("gfx1250" in case[-1] for case in perf_baseline.CASES)
    monkeypatch.setattr(perf_baseline, "_gfx", lambda: "gfx1250")
    for mode in ("--record", "--check"):
        assert perf_baseline.main([mode]) == 1
        assert "no perf case runs on gfx1250" in capsys.readouterr().out
    assert perf_check.asked == [] and not list(tmp_path.iterdir())

    monkeypatch.setattr(perf_baseline, "_gfx", lambda: "gfx942")
    assert perf_baseline.main(["--record", "--case", "dsv3_tp8_a4w4"]) == 1
    assert "do not run on gfx942" in capsys.readouterr().out
    assert perf_check.asked == [] and not list(tmp_path.iterdir())


def test_perf_check_rejects_a_missing_file_and_other_arch_cases(perf_check):
    code, out = perf_check.check("--case", "dsv3_tp8_pt")
    assert code == 1 and "record one first" in out
    code, out = perf_check.check("--case", "dsv3_tp8_a4w4", baseline={})
    assert code == 1 and "do not run on gfx942" in out
    assert perf_check.asked == []


def test_perf_check_without_cases_needs_every_case_of_the_arch(perf_check):
    from moe_dispatch import perf_baseline

    gfx942 = [c[0] for c in perf_baseline.CASES if "gfx942" in c[-1]]
    code, out = perf_check.check(baseline={gfx942[0]: {"1": 10.0}})
    assert code == 1 and perf_check.asked == []
    assert str(gfx942[1:]) in out


def test_perf_check_compares_the_requested_cases(perf_check):
    baseline = {"dsv3_tp8_pt": {"1": 10.0}, "qwen3_235b_tp8_bf16": {"1": 10.0}}
    perf_check.measured["qwen3_235b_tp8_bf16"] = {"1": 10.5}
    code, _ = perf_check.check("--case", "qwen3_235b_tp8_bf16", baseline=baseline)
    assert code == 0 and perf_check.asked == [["qwen3_235b_tp8_bf16"]]
    perf_check.measured["qwen3_235b_tp8_bf16"] = {"error": "boom"}
    code, out = perf_check.check("--case", "qwen3_235b_tp8_bf16")
    assert code == 1 and "MISSING qwen3_235b_tp8_bf16 M=1" in out


def test_dispatch_plans_match_legacy():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "differential.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        proc.stdout[-6000:]
        + proc.stderr[-3000:]
        + "\nv2 must make the same choice as legacy for every key it claims."
    )


def _registered():
    from aiter.moev2 import paths

    return sorted(paths.registered())


def _path_supports_device(path_name, gfx=None):
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.moev2 import paths

    path = paths.registered()[path_name]
    gfx = get_gfx_runtime() if gfx is None else gfx
    return not path.supported_archs or gfx in path.supported_archs


def test_gpu_differential_architecture_filter():
    assert _path_supports_device("a8w8_pt", "gfx950")
    assert not _path_supports_device("a8w8_pt", "gfx1250")


@pytest.mark.parametrize("path_name", _registered())
def test_outputs_match_legacy(path_name):
    import torch

    if os.environ.get("AITER_MOEV2_GPU_DIFFERENTIAL") != "1":
        pytest.skip(
            "manual: AITER_MOEV2_GPU_DIFFERENTIAL=1 (non-repeatable kernels, "
            "arch-specific cases)"
        )
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not _path_supports_device(path_name):
        pytest.skip(f"{path_name} does not support this GPU architecture")
    proc = subprocess.run(
        [
            sys.executable,
            str(_HERE / "moe_dispatch" / "gpu_differential.py"),
            "--path",
            path_name,
        ],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
