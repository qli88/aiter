# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 legacy tuned rows as v2 plan-file rows (`dispatch.convert`), on the CPU.

A converted row holds the cells dispatch reads: run through dispatch, they resolve its
base call as the legacy row does, except that a 1-stage row's stage 2 (a blank cell or
"Null") logs as empty and the FHMoE table's absent `run_1stage` (False) reads as 0.
"""

import csv
import json
import math
import pathlib
import re
import subprocess
import sys

import pytest

from aiter.moev2 import envs, paths
from aiter.moev2.dispatch import convert, resolver, tuned, tuned_plans, tuned_table
from aiter.moev2.engine import _canonical
from aiter.moev2.segments.base import Params

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
    "flydsl_layout_plain_stage1": (
        "gfx950,256,1024,7168,256,257,9,ActivationType.Silu,torch.bfloat16,"
        "torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,"
        "97.2441,flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4,103.6032,"
        "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64,200.8473,0,0,0,0"
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


def _plan_file(tmp_path, rows):
    path = tmp_path / "converted.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS + convert.INFO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _outcome_reading(path, problem, segments, monkeypatch):
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(path))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
    tuned.clear_cache()
    return _outcome(problem, segments)


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


_PLAIN = "flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4"
_LAYOUT_FP4 = "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64"
_LAYOUT_FP8 = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x256_atomic_nt_sbm32"


def test_a_plain_layout_stage1_name_is_held_as_its_twin():
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    assert convert.v2_kernel1(_PLAIN, _LAYOUT_FP4) == f"{_PLAIN}_fp4"
    plain8 = "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui"
    assert convert.v2_kernel1(plain8, _LAYOUT_FP8) == f"{plain8}_fp8"
    twin = get_flydsl_kernel_params(f"{_PLAIN}_fp4")
    assert twin == {**get_flydsl_kernel_params(_PLAIN), "out_dtype": "fp4"}
    token_major = "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce"
    for kernel1, kernel2 in (
        (f"{_PLAIN}_fp4", _LAYOUT_FP4),
        (f"{_PLAIN}_fp8", _LAYOUT_FP4),
        (_PLAIN, token_major),
        (_PLAIN, ""),
        ("flydsl_moe1_afp4_wfp4_bf16_t64x64x256_bogus", _LAYOUT_FP4),
        ("opus_moe1_afp8_wfp4_bf16_t32x384_pair_kw7_m3", _LAYOUT_FP8),
        ("moe_ck2stages_gemm1_x", _LAYOUT_FP4),
        ("", _LAYOUT_FP4),
    ):
        assert convert.v2_kernel1(kernel1, kernel2) == kernel1, kernel1
    row = {"kernelName1": _PLAIN, "kernelName2": _LAYOUT_FP4, "block_m": 64}
    assert convert.cells(row)["kernelName1"] == f"{_PLAIN}_fp4"
    where = "the row"
    assert tuned_table.legacy_cells(row, where)["kernelName1"] == f"{_PLAIN}_fp4"


def test_a_plain_layout_stage1_plan_row_runs_as_its_twin(tmp_path, monkeypatch):
    _, table = _table(tmp_path, _MAIN_HEADER, [_ROWS["flydsl_layout_plain_stage1"]])
    (out,), skipped = convert.convert_table(table, "gfx950", 256)
    assert skipped == []
    candidates = json.loads(out["candidates"])
    assert candidates["flydsl.gemm1_layout"]["kernel"] == f"{_PLAIN}_fp4"
    assert candidates["flydsl_layout.gemm2"]["kernel"] == _LAYOUT_FP4
    (key,) = table[0]
    problem = convert.problem_of(
        dict(zip(tuned.index_columns(False), key)), "gfx950", 256
    )
    segments = paths.registered()[out["path"]].segments
    twin = _outcome_reading(_plan_file(tmp_path, [out]), problem, segments, monkeypatch)
    plain_candidates = json.dumps(candidates).replace(f"{_PLAIN}_fp4", _PLAIN)
    plain_row = {**out, "candidates": plain_candidates}
    plain = _outcome_reading(
        _plan_file(tmp_path, [plain_row]), problem, segments, monkeypatch
    )
    assert twin[0] == plain[0] == "tuned"
    assert f"{_PLAIN}_fp4" in twin[1]
    assert tuple(t.replace(f"{_PLAIN}_fp4", _PLAIN) for t in twin) == plain


def test_the_differential_maps_legacy_s_plain_stage1_name():
    from moe_dispatch import differential

    renamed = _LAYOUT_FP4.replace("_afp4_", "_afp8_", 1)
    assert differential.legacy_stage1_as_v2(_PLAIN, _LAYOUT_FP4) == f"{_PLAIN}_fp4"
    assert differential.legacy_stage1_as_v2(_PLAIN, renamed) == f"{_PLAIN}_fp4"
    plain8 = "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui"
    assert differential.legacy_stage1_as_v2(plain8, _LAYOUT_FP8) == f"{plain8}_fp8"
    stage2 = {"fn": "fused_moe._flydsl_v2_stage2_wrapper"}
    record = {
        "block_m": 64,
        "stage1": {"fn": "f", "kw": {"kernelName": _PLAIN, "out_dtype": "fp4"}},
        "stage2": {**stage2, "kw": {"kernelName": _LAYOUT_FP4}},
    }
    mapped = differential.as_v2_record(record)
    assert mapped["stage1"]["kw"] == {
        "kernelName": f"{_PLAIN}_fp4",
        "out_dtype": "fp4",
    }
    assert {k: v for k, v in mapped.items() if k != "stage1"} == {
        k: v for k, v in record.items() if k != "stage1"
    }
    assert record["stage1"]["kw"]["kernelName"] == _PLAIN
    for unchanged in (
        {"error": "x"},
        {**record, "stage2": None},
        {**record, "stage2": {**stage2, "kw": {"kernelName": "opus_moe2_x"}}},
    ):
        assert differential.as_v2_record(unchanged) == unchanged
    line = (
        f"[fused_moe] using 2stage (kernelName1='{_PLAIN}', "
        f"kernelName2='{_LAYOUT_FP4}') for (key) "
    )
    one_stage = "[fused_moe] using 1stage (kernelName1='k', kernelName2='Null') for k "
    assert differential._as_v2_logs([(20, line), (20, one_stage), (30, "other")]) == [
        (20, line.replace(f"'{_PLAIN}'", f"'{_PLAIN}_fp4'")),
        (20, one_stage.replace("'Null'", "''")),
        (30, "other"),
    ]


def _check_main_row(line, arch, cu_num, tmp_path, monkeypatch):
    legacy_file, table = _table(tmp_path, _MAIN_HEADER, [line])
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
    legacy = _outcome_reading(legacy_file, problem, segments, monkeypatch)
    plan_file = _plan_file(tmp_path, rows)
    assert _outcome_reading(plan_file, problem, segments, monkeypatch) == legacy
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
    legacy = _outcome_reading(name, problem, segments, monkeypatch)
    plan_file = _plan_file(tmp_path, rows)
    assert _outcome_reading(plan_file, problem, segments, monkeypatch) == legacy
    assert legacy[0] == "tuned"


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


def test_the_committed_converted_tables_are_the_shipped_ones():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "convert_tables.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


def test_merged_tables_read_the_files_as_their_merge(tmp_path):
    full = tmp_path / "full.csv"
    full.write_text(f"{_MAIN_HEADER}\n{_ROWS['ck_2stage']}\n")
    header = _MAIN_HEADER.replace("gfx,", "", 1).replace(",xbf16,flat", "")
    blockscale_bf16 = (
        _ROWS["asm_1stage_blank_stage2"]
        .replace("gfx950,", "", 1)
        .replace("blockscaleFp8", "blockscaleBf16")
        .replace(",1,0,0,0", ",1,0")
    )
    lacking = tmp_path / "lacking.csv"
    lacking.write_text(f"{header}\n{blockscale_bf16}\n")
    tables = convert.merged_tables([full, lacking])
    assert list(tables) == [str(full), str(lacking)]
    (row,) = tables[str(lacking)][0].values()
    assert (row["xbf16"], row["flat"]) == (0, 0)
    assert convert.cells(row)["xbf16"] == 0
    alone = tuned.read_table(str(lacking), tuned.index_columns(False), False)
    assert convert.cells(next(iter(alone[0].values())))["xbf16"] == 1
    ((key, _),) = tables[str(lacking)][0].items()
    assert key[:2] == ("gfx950", 256)

    twice = tmp_path / "twice.csv"
    twice.write_text(f"{_MAIN_HEADER}\n{_ROWS['ck_2stage']}\n")
    with pytest.raises(ValueError, match="keys in two rows"):
        convert.merged_tables([full, twice])


def test_merged_tables_raise_for_keys_the_merge_reads_twice(tmp_path):
    row = _ROWS["ck_2stage"]
    no_gfx = tmp_path / "no_gfx.csv"
    no_gfx.write_text(
        f"{_MAIN_HEADER.replace('gfx,', '', 1)}\n{row[len('gfx950,'):]}\n"
    )
    full = tmp_path / "full.csv"
    full.write_text(f"{_MAIN_HEADER}\n{row}\n")
    within = tmp_path / "within.csv"
    within.write_text(f"{_MAIN_HEADER}\n{row}\n{row}\n")
    tagged = tmp_path / "tagged.csv"
    tagged.write_text(f"{_MAIN_HEADER},_tag\n{row},retuned\n")
    fallback = tmp_path / "fallback.csv"
    fallback.write_text(f"{_MAIN_HEADER},_tag\n{row},flydsl_fallback\n")
    for files in ([no_gfx, full], [within], [full, tagged]):
        with pytest.raises(ValueError, match="keys in two rows"):
            convert.merged_tables(files)
    tables = convert.merged_tables([full, fallback])
    assert (len(tables[str(full)][0]), len(tables[str(fallback)][0])) == (1, 0)


def test_the_shipped_sources_are_the_legacy_files():
    main, fhmoe = convert.shipped_sources()
    assert main[0] == convert.CONFIGS / "tuned_fmoe.csv" and len(main) > 1
    assert all(p.parent == convert.CONFIGS / "model_configs" for p in main[1:])
    assert fhmoe == convert.CONFIGS / "tuned_fhmoe.csv"


def test_converted_rows_read_back_as_the_legacy_rows_cells(tmp_path, monkeypatch):
    _, table = _table(
        tmp_path, _MAIN_HEADER, [_ROWS["ck_2stage"], _ROWS["cktile_opus"]]
    )
    rows, _ = convert.convert_table(table, "gfx950", 256)
    plan_file = _plan_file(tmp_path, rows)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(plan_file))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
    tuned.clear_cache()
    for key, row in table[0].items():
        problem = convert.problem_of(
            dict(zip(tuned.index_columns(False), key)), "gfx950", 256
        )
        found = tuned_table.lookup(problem)
        assert found == tuned_table.Found(cells=convert.cells(row))


# comm_fused_configs: comm_fused rows of the a8w4 row.

_DSV4 = _HERE.parent / "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"


def _dsv4_table(tmp_path, tokens=(1, 4096), edit=None, tag=""):
    """A legacy table of the shipped rows with comm_fused_configs at `tokens` (`edit`
    applied to each row's configs; `tag`: their `_tag`)."""
    with open(_DSV4, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [
            r
            for r in reader
            if r["comm_fused_configs"].strip() and int(r["token"]) in tokens
        ]
    for r in rows:
        r["_tag"] = tag
        if edit is not None:
            r["comm_fused_configs"] = edit(json.loads(r["comm_fused_configs"]))
    path = tmp_path / "fmoe.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, header)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _read(path, dedicated=False):
    return tuned.read_table(path, tuned.index_columns(dedicated), dedicated)


def _local(key):
    return convert.problem_of(dict(zip(tuned.index_columns(False), key)), "gfx950", 256)


def test_comm_fused_configs_become_comm_rows_of_the_a8w4_row(tmp_path, monkeypatch):
    path = _dsv4_table(tmp_path)
    table = _read(path)
    rows, skipped = convert.convert_table(table, "gfx950", 256)
    assert skipped == []
    assert [(r["token"], r["path"], r["comm"]) for r in rows] == [
        ("1", "a8w4", ""),
        ("1", "comm_fused", "ar"),
        ("4096", "a8w4", ""),
        ("4096", "comm_fused", "ar"),
        ("4096", "comm_fused", "rs"),
    ]
    local = {r["token"]: r for r in rows if r["path"] == "a8w4"}
    for (key, legacy), token in zip(table[0].items(), ("1", "4096")):
        configs = json.loads(legacy["comm_fused_configs"])
        for row in (r for r in rows if r["path"] == "comm_fused"):
            if row["token"] != token:
                continue
            entry = configs[row["comm"]]
            runner = Params(entry["params"], family=entry["family"])
            assert tuned_plans.decode_candidates(row["candidates"]) == {
                **tuned_plans.decode_candidates(local[token]["candidates"]),
                tuned_plans.RUNNER: runner,
            }
            assert row["structure"] == local[token]["structure"]
            assert (row["tp"], row["add_shared"], row["gpu_model"]) == (
                "8",
                "True",
                "mi355x",
            )
            assert float(row["e2e_us"]) == entry["us"] and row["mode"] == convert.MODE
            assert all(row[c] == "" for c in convert.INFO_COLUMNS)
    plan_file = _plan_file(tmp_path, rows)
    read = {}
    for name in (plan_file, path):
        monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(name))
        monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
        tuned.clear_cache()
        for key, legacy in table[0].items():
            problem = _local(key)
            assert tuned_table.lookup(problem) == tuned_table.Found(
                cells=convert.cells(legacy)
            )
            for comm in ("ar", "rs"):
                row = tuned_table.comm_lookup(problem, 8, comm, True, "mi355x")
                if comm not in json.loads(legacy["comm_fused_configs"]):
                    assert row is None
                    continue
                assert row.cells == convert.cells(legacy)
                read.setdefault((key, comm), []).append(row)
            assert tuned_table.comm_lookup(problem, 8, "ar", True, "mi350x") is None
            assert tuned_table.comm_lookup(problem, 4, "ar", True, "mi355x") is None
    assert len(read) == 3 and all(a == b for a, b in read.values())


def _set(where, value):
    def edit(configs):
        entry = configs["ar"]
        *path, last = where
        for part in path:
            entry = entry[part]
        entry[last] = value
        return json.dumps(configs)

    return edit


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda c: "{", "Expecting"),
        (lambda c: json.dumps([c]), "must be a JSON object"),
        (lambda c: json.dumps({**c, "ag": c["ar"]}), r"unsupported comm_fused modes"),
        (_set(("extra",), 1), r"unsupported comm_fused_configs\['ar'\] fields"),
        (_set(("tp",), 8.5), "is not a whole number"),
        (_set(("add_shared",), "yes"), "add_shared must be boolean"),
        (_set(("gpu_model",), " "), "gpu_model must be a non-empty string"),
        (_set(("params", "sort_block_m"), 128), "does not match CSV block_m=64"),
        (_set(("family",), "nosuch"), "nosuch"),
    ],
)
def test_comm_fused_configs_legacy_rejects_raise(tmp_path, monkeypatch, edit, match):
    from types import SimpleNamespace

    from aiter.ops.flydsl import comm_fused_moe_host as host

    path = _dsv4_table(tmp_path, tokens=(4096,), edit=edit)
    with pytest.raises(ValueError, match=match):
        convert.convert_table(_read(path), "gfx950", 256)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, path)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
    tuned.clear_cache()
    key = next(iter(_read(path)[0]))
    with pytest.raises(ValueError, match=match):
        tuned_table.lookup(_local(key))
    monkeypatch.setattr(
        host, "AITER_CONFIGS", SimpleNamespace(AITER_CONFIG_FMOE_FILE=path)
    )
    host._winner_table.cache_clear()
    try:
        with pytest.raises((ValueError, TypeError, KeyError)):
            host._winner_table()
    finally:
        host._winner_table.cache_clear()


def test_comm_fused_configs_without_an_a8w4_row_are_skipped(tmp_path):
    def bogus(configs):
        return json.dumps(configs)

    path = _dsv4_table(tmp_path, tokens=(4096,), edit=bogus)
    text = (
        pathlib.Path(path)
        .read_text()
        .replace("opus_moe1_afp8_wfp4_bf16_t64x384_gs_qgb3_aff32_p2_fp8", "bogus_moe1")
    )
    pathlib.Path(path).write_text(text)
    rows, skipped = convert.convert_table(_read(path), "gfx950", 256)
    assert rows == []
    reasons = [r for _, r in skipped]
    assert reasons[0].startswith("a8w4: ")
    assert reasons[1:] == ["comm_fused: no a8w4 row"] * 2


def test_comm_fused_configs_where_legacy_does_not_read_them_raise(
    tmp_path, monkeypatch
):
    untagged = tmp_path / "untagged"
    untagged.mkdir()
    key = next(iter(_read(_dsv4_table(untagged, tokens=(4096,)))[0]))
    path = _dsv4_table(tmp_path, tokens=(4096,), tag="flydsl_fallback")
    with pytest.raises(ValueError, match="on a flydsl_fallback row"):
        convert.merged_tables([path])
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, path)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
    tuned.clear_cache()
    with pytest.raises(ValueError, match="comm_fused_configs on rows the reader drops"):
        tuned_table.lookup(_local(key))

    fhmoe = tmp_path / "fhmoe.csv"
    configs = json.dumps(json.dumps({"rs": {"family": "direct"}}))
    fhmoe.write_text(f"{_FHMOE_HEADER},comm_fused_configs\n{_fhmoe_row()},{configs}\n")
    with pytest.raises(ValueError, match="comm_fused_configs in an FHMoE table"):
        convert.convert_table(_read(str(fhmoe), True), "gfx950", 256, str(fhmoe))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(fhmoe))
    tuned.clear_cache()
    with pytest.raises(ValueError, match="comm_fused_configs in an FHMoE table"):
        tuned_table.lookup(_local(key))


@pytest.mark.parametrize(
    "column, edit, match",
    [
        ("candidates", ('"sort_block_m": 64', '"sort_block_m": 128'), "CSV block_m=64"),
        ("candidates", ('"family": "window"', '"family": "nosuch"'), "nosuch"),
        ("gpu_model", ("mi355x", "MI355X"), "not a lowercase name"),
        ("add_shared", ("True", "1"), "not True or False"),
        ("tp", ("8", "8.5"), "not a whole number"),
        ("gpu_model", ("mi355x", ""), "some collective columns blank"),
    ],
)
def test_a_plan_file_s_comm_rows_legacy_rejects_raise(
    tmp_path, monkeypatch, column, edit, match
):
    table = _read(_dsv4_table(tmp_path, tokens=(4096,)))
    rows, _ = convert.convert_table(table, "gfx950", 256)
    row = next(r for r in rows if r["comm"] == "rs")
    assert edit[0] in row[column]
    row[column] = row[column].replace(*edit)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, str(_plan_file(tmp_path, rows)))
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
    tuned.clear_cache()
    with pytest.raises(ValueError, match=match):
        tuned_table.lookup(_local(next(iter(table[0]))))


def test_comm_fused_configs_on_the_second_row_of_a_key_raise(tmp_path, monkeypatch):
    path = _dsv4_table(tmp_path, tokens=(4096,))
    lines = pathlib.Path(path).read_text().splitlines()
    first = next(csv.DictReader(lines))
    first["comm_fused_configs"] = ""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(first))
        writer.writeheader()
        writer.writerow(first)
        f.write(lines[1] + "\n")
    key = next(iter(_read(path)[0]))
    assert not tuned_table.has_comm_configs(_read(path)[0][key])
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, path)
    monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, "replace")
    tuned.clear_cache()
    with pytest.raises(ValueError, match="the second of two rows of a key"):
        tuned_table.lookup(_local(key))
