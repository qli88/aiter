# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 legacy tuned rows as v2 plan-file rows (`dispatch.convert`), on the CPU.

A converted row holds the cells dispatch reads: run through dispatch, they resolve its
base call as the legacy row does, except that a 1-stage row's stage 2 (a blank cell or
"Null") logs as empty and the FHMoE table's absent `run_1stage` (False) reads as 0.
"""

import csv
import math
import pathlib
import re
import sys

import pytest

from aiter.moev2 import envs, paths
from aiter.moev2.dispatch import convert, resolver, tuned, tuned_plans
from aiter.moev2.engine import _canonical

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_MAIN_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
    "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,us2,"
    "kernelName2,us,run_1stage,xbf16,flat,nt"
)
_FP8 = "torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn"
_ROWS = {
    "ck_2stage": (
        f"gfx950,256,1,5120,1536,40,8,ActivationType.Silu,{_FP8},QuantType.per_Token,"
        "1,0,32,0,24.7726,moe_ck2stages_gemm1_256x32x64x256_1x4_TypeCast_v1_Nswizzle0_"
        "Quant2_MulRoutedWeight0_silu_F8_F8_B16,19.6546,moe_ck2stages_gemm2_256x32x128"
        "x256_1x4_TypeCast_v1_Nswizzle0_Quant2_MulRoutedWeight1_F8_F8_B16,44.4272,"
        "0,0,0,0"
    ),
    "asm_1stage_blank_stage2": (
        f"gfx950,256,8,6144,256,257,9,ActivationType.Silu,{_FP8},QuantType.per_1x128,"
        "1,0,16,0,68.9083,_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_pf2_silu_16x128E,"
        "0.0,,68.9083,1,0,0,0"
    ),
    "asm_1stage_null_stage2": (
        f"gfx950,256,1,5120,1024,128,1,ActivationType.Silu,{_FP8},QuantType.per_Token,"
        "1,1,32,0,0.0,_ZN5aiter46fmoe_bf16_pertokenFp8_g1u1_tkw1_silu_1tg_32x64E,0.0,"
        "Null,0.0,1,0,0,0"
    ),
    "flydsl_mxmoe": (
        "gfx950,256,1,6144,1536,129,5,ActivationType.Swiglu,torch.bfloat16,"
        "torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,"
        "21.457,flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_swiglu_xcd4,0.0,"
        "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16,21.457,0,0,0,0"
    ),
    "cktile_opus": (
        "gfx950,256,2048,7168,384,385,7,ActivationType.Silu,torch.bfloat16,"
        "torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,"
        "243.2977,cktile_a8w4_bm64,164.2256,opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_"
        "rbn3072,407.5233,0,0,0,0"
    ),
    "asm_1stage_flat_xbf16": (
        "gfx950,256,1,8192,256,512,10,ActivationType.Silu,torch.bfloat16,"
        "torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,"
        "20.87,_ZN5aiter47fmoe_bf16_pertokenMXfp4_g1u1_flat_vs_silu_16x32E,0.0,,20.49,"
        "1,1,1.0,0"
    ),
}
_IMPL = (
    "gfx942,80,{m},2048,128,257,9,ActivationType.Silu,torch.bfloat16,"
    "torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,1,0,16,0,"
    "11.48,impl__flydsl_gfx942__16_16_16_False:default:0,0.0,{k2},11.48,0,0,0,0"
)
_FHMOE_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,shared_expert_id,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,hidden_pad,intermediate_pad,"
    "gate_mode,block_m,ksplit,kernelName1,kernelName2"
)


def _fhmoe_row(shared_expert_id=384, gate_mode="GateMode.INTERLEAVE"):
    return (
        f"gfx950,256,1,7168,384,385,7,{shared_expert_id},ActivationType.Silu,"
        "torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,"
        f"1,0,0,0,{gate_mode},32,0,flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw4_fp8,"
        "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic"
    )


_FHMOE_FP8_ROW = (
    "gfx950,256,1,6144,256,257,9,256,ActivationType.Silu,torch.bfloat16,"
    "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x32,1,0,0,0,"
    "GateMode.INTERLEAVE,32,0,flydsl_moe1_afp8_wfp8_bf16_t32x64x256_w4_gui_kw2_fp8,"
    "flydsl_moe2_afp8_wfp8_bf16_t32x128x128_atomic"
)


@pytest.fixture(autouse=True)
def _gfx950():
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx950"):
        tuned.clear_cache()
        yield
    tuned.clear_cache()


def _table(tmp_path, header, rows, dedicated=False):
    path = tmp_path / ("fhmoe.csv" if dedicated else "fmoe.csv")
    path.write_text("\n".join((header, *rows)) + "\n")
    return str(path), tuned.read_table(
        str(path), tuned.index_columns(dedicated), dedicated
    )


def _outcome(problem, segments):
    resolution = resolver.resolve(problem, segments)
    assert resolution.plan is not None, resolution.declined

    def norm(text):
        text = re.sub(r" at 0x[0-9a-f]+", "", str(text))
        text = text.replace("run_1stage=False", "run_1stage=0")
        return re.sub(r"kernelName2=(nan|'Null')", "kernelName2=''", text)

    return (
        resolution.plan.source,
        norm(resolution.plan.describe()),
        norm(_canonical(resolution.metadata)),
        norm(resolution.logs),
    )


def _cells_table(problem, out):
    segments = {s.name: s for s in paths.registered()[out["path"]].segments}
    structure = tuple(
        segments[n] for n in tuned_plans.structure_names(out["structure"])
    )
    cells = resolver.candidate_row(
        structure, tuned_plans.decode_candidates(out["candidates"])
    )
    return {tuned.exact_key(problem): cells}, {}


def test_cells_read_a_row_as_dispatch_does():
    one_stage = {
        "kernelName1": "_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_pf2_silu_16x128E",
        "kernelName2": math.nan,
        "block_m": 16.0,
        "ksplit": 0,
        "run_1stage": 1,
        "nt": math.nan,
        "flat": 1,
    }
    assert convert.cells(one_stage) == {
        "kernelName1": one_stage["kernelName1"],
        "kernelName2": "",
        "block_m": 16,
        "ksplit": 0,
        "nt": 0,
        "run_1stage": 1,
        "xbf16": 0,
        "flat": 1,
    }
    assert convert.cells({**one_stage, "kernelName2": "Null"})["kernelName2"] == ""
    bf16 = {**one_stage, "kernelName1": "fmoe_blockscaleBf16_x"}
    assert convert.cells(bf16)["xbf16"] == 1
    assert convert.cells({**bf16, "xbf16": 0})["xbf16"] == 0
    two_stage = {"kernelName1": "a", "kernelName2": "b", "block_m": 32, "nt": 1}
    assert convert.cells({**two_stage, "flat": 1}) == {
        "kernelName1": "a",
        "kernelName2": "b",
        "block_m": 32,
        "ksplit": 0,
        "nt": 1,
        "run_1stage": 0,
        "xbf16": 0,
        "flat": 0,
    }


def _check_main_row(line, arch, cu_num, tmp_path, monkeypatch):
    _, table = _table(tmp_path, _MAIN_HEADER, [line])
    rows, skipped = convert.convert_table(table, arch, cu_num)
    assert skipped == []
    (out,) = rows
    (key,) = table[0]
    problem = convert.problem_of(
        dict(zip(tuned.index_columns(False), key)), arch, cu_num
    )
    assert tuple(out[c] for c in tuned_plans.KEY_COLUMNS) == tuned_plans.key_values(
        problem
    )
    assert out["mode"] == "legacy"
    assert out["e2e_us"] == out["us"] == line.split(",")[20]
    assert convert.paths_taking(problem) == [out["path"]]
    segments = paths.registered()[out["path"]].segments
    monkeypatch.setattr(tuned, "_shared_default_table", lambda: table)
    legacy = _outcome(problem, segments)
    monkeypatch.setattr(
        tuned, "_shared_default_table", lambda: _cells_table(problem, out)
    )
    assert _outcome(problem, segments) == legacy
    assert legacy[0] == "tuned"


@pytest.mark.parametrize("name", sorted(_ROWS))
def test_a_converted_row_resolves_as_its_legacy_row(name, tmp_path, monkeypatch):
    _check_main_row(_ROWS[name], "gfx950", 256, tmp_path, monkeypatch)


def test_a_converted_impl_row_resolves_as_its_legacy_row(tmp_path, monkeypatch):
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx942_cu80"):
        tuned.clear_cache()
        line = _IMPL.format(m=1, k2="")
        _check_main_row(line, "gfx942", 80, tmp_path, monkeypatch)


@pytest.mark.parametrize("line", [_fhmoe_row(), _FHMOE_FP8_ROW], ids=["dsv4", "fp8"])
def test_a_converted_fhmoe_row_resolves_as_its_legacy_row(line, tmp_path, monkeypatch):
    name, table = _table(tmp_path, _FHMOE_HEADER, [line], dedicated=True)
    rows, skipped = convert.convert_table(table, "gfx950", 256, fhmoe_table=name)
    assert skipped == []
    (out,) = rows
    assert out["path"] == "fhmoe"
    assert out["shared_expert_id"] == line.split(",")[7]
    assert out["gate_mode"] == "GateMode.INTERLEAVE"
    (key,) = table[0]
    problem = convert.problem_of(
        dict(zip(tuned.index_columns(True), key)), "gfx950", 256, name
    )
    segments = paths.registered()["fhmoe"].segments
    monkeypatch.setattr(tuned, "_dedicated_table", lambda f: table)
    legacy = _outcome(problem, segments)
    monkeypatch.setattr(tuned, "_dedicated_table", lambda f: _cells_table(problem, out))
    assert _outcome(problem, segments) == legacy


def test_rows_without_a_plan_row_say_why(tmp_path):
    bogus = _ROWS["ck_2stage"].replace("moe_ck2stages_gemm1_", "bogus_gemm1_", 1)
    gfx942 = _ROWS["ck_2stage"].replace("gfx950,256", "gfx942,304", 1)
    unread = _ROWS["ck_2stage"].replace("gfx950,256,1,", "gfx950,256,2,", 1)
    unread = unread.replace("ActivationType.Silu", "ActivationType.Bogus", 1)
    routed = _ROWS["cktile_opus"] + ",b1"
    lines = [bogus, gfx942, unread, routed]
    _, table = _table(tmp_path, _MAIN_HEADER + ",route_bucket", lines)
    rows, skipped = convert.convert_table(table, "gfx950", 256)
    assert rows == []
    reasons = [r for _, r in skipped]
    assert reasons[0].startswith("a8w8_ptok: dispatch declines the row: unknown kernel")
    assert reasons[1].startswith("the key is not read: ")
    assert reasons[2].endswith(
        ": Opus route columns the plan file does not hold: route_bucket"
    )
    assert len(reasons) == 3

    other_shared = _fhmoe_row(shared_expert_id=383)
    separated = _fhmoe_row(gate_mode="GateMode.SEPARATED")
    name, table = _table(
        tmp_path, _FHMOE_HEADER, [other_shared, separated], dedicated=True
    )
    rows, skipped = convert.convert_table(table, "gfx950", 256, fhmoe_table=name)
    assert rows == []
    named = dict(zip(_FHMOE_HEADER.split(","), other_shared.split(",")))
    looked_up = tuned.exact_key(convert.problem_of(named, "gfx950", 256, name))
    assert looked_up[tuned.index_columns(True).index("shared_expert_id")] == 384
    assert [r for _, r in skipped] == [
        f"looked up as {looked_up}",
        "an FHMoE key its calls look up in the main table",
    ]


def test_a_row_whose_cells_do_not_come_back_has_none(tmp_path):
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx942_cu80"):
        tuned.clear_cache()
        _, table = _table(
            tmp_path,
            _MAIN_HEADER,
            [_IMPL.format(m=1, k2=""), _IMPL.format(m=2, k2="stage2")],
        )
        rows, skipped = convert.convert_table(table, "gfx942", 80)
    assert [r["token"] for r in rows] == ["1"]
    ((key, reason),) = skipped
    assert key[2] == 2
    assert reason.endswith(": the cells do not come back: kernelName2 'stage2' vs ''")


def test_the_plan_file_reads_converted_rows(tmp_path, monkeypatch):
    _, table = _table(
        tmp_path, _MAIN_HEADER, [_ROWS["ck_2stage"], _ROWS["cktile_opus"]]
    )
    rows, _ = convert.convert_table(table, "gfx950", 256)
    plan_file = tmp_path / "converted.csv"
    with open(plan_file, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS + convert.INFO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_PLANS, str(plan_file))
    tuned.clear_cache()
    for key, out in zip(table[0], rows):
        problem = convert.problem_of(
            dict(zip(tuned.index_columns(False), key)), "gfx950", 256
        )
        assert tuned_plans.lookup(problem) == (
            (
                tuned_plans.structure_names(out["structure"]),
                tuned_plans.decode_candidates(out["candidates"]),
            ),
        )
