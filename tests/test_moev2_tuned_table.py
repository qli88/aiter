# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""v2's tuned table (`dispatch.tuned_table`): the shipped tables, the user's
(`AITER_MOEV2_TUNED_TABLE` and its mode) and the hand-back for legacy's variables."""

import csv
import hashlib
import logging
import os
import pathlib
import sys
import types

import pytest

from aiter.moev2 import envs, paths
from aiter.moev2.dispatch import convert, tuned, tuned_plans, tuned_table

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
    "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,kernelName1,"
    "kernelName2,run_1stage"
)
_FP8 = "torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn"
_CK1 = "moe_ck2stages_gemm1_256x32x64x256_1x4_TypeCast_v1_Nswizzle0_Quant2"
_ASM = "_ZN5aiter46fmoe_bf16_pertokenFp8_g1u1_tkw1_silu_1tg_32x64E"
_FHMOE_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,shared_expert_id,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,hidden_pad,intermediate_pad,"
    "gate_mode,block_m,ksplit,kernelName1,kernelName2"
)
_FHMOE_FP8_ROW = (
    "gfx950,256,1,6144,256,257,9,256,ActivationType.Silu,torch.bfloat16,"
    "torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x32,1,0,0,0,"
    "GateMode.INTERLEAVE,32,0,flydsl_moe1_afp8_wfp8_bf16_t32x64x256_w4_gui_kw2_fp8,"
    "flydsl_moe2_afp8_wfp8_bf16_t32x128x128_atomic"
)


def _row(expert=40, kernel="k", act="Silu", token=1, one_stage=False, **cells):
    """A legacy row of a gfx950 per-token fp8 key with `expert` experts: a 2-stage CK
    row (kernel names `kernel`-suffixed), or a 1-stage ASM row."""
    values = {"block_m": 32, "ksplit": 0}
    if one_stage:
        values |= {"kernelName1": _ASM, "kernelName2": "", "run_1stage": 1}
    else:
        values |= {
            "kernelName1": f"{_CK1}_{kernel}",
            "kernelName2": f"moe_ck2stages_gemm2_{kernel}",
            "run_1stage": 0,
        }
    values |= cells
    key = (
        f"gfx950,256,{token},5120,1536,{expert},8,ActivationType.{act},{_FP8},"
        "QuantType.per_Token,1,0"
    )
    tail = ",".join(str(values[c]) for c in _HEADER.split(",")[14:])
    return f"{key},{tail}"


def _csv(tmp_path, name, rows, header=_HEADER):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def _problem(row, header=_HEADER):
    return convert.problem_of(
        dict(zip(header.split(","), row.split(","))), "gfx950", 256
    )


def _use(monkeypatch, files=None, mode=None):
    for name in (envs.AITER_MOEV2_TUNED_TABLE, envs.AITER_MOEV2_TUNED_TABLE_MODE):
        monkeypatch.delenv(name, raising=False)
    if files is not None:
        value = os.pathsep.join(map(str, files))
        monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE, value)
    if mode is not None:
        monkeypatch.setenv(envs.AITER_MOEV2_TUNED_TABLE_MODE, mode)
    tuned.clear_cache()


def _shipped(monkeypatch, path):
    table = tuned_table._user_file(str(path))[0]
    monkeypatch.setattr(tuned_table, "_shipped", lambda: (table, "shipped"))


def _kernel(found):
    return None if found.cells is None else found.cells["kernelName1"]


@pytest.fixture(autouse=True)
def _gfx950(monkeypatch):
    from moe_dispatch import arch_sim

    for name in envs.LEGACY_TUNED_TABLES:
        monkeypatch.delenv(name, raising=False)
    with arch_sim.simulated_arch("gfx950"):
        tuned.clear_cache()
        yield
    monkeypatch.undo()
    tuned.clear_cache()


def test_the_mode_is_required_and_known(monkeypatch):
    _use(monkeypatch)
    assert envs.tuned_table() == ((), None)
    _use(monkeypatch, ["a.csv"])
    with pytest.raises(ValueError, match="MODE must be one of"):
        envs.tuned_table()
    _use(monkeypatch, ["a.csv"], "merge")
    with pytest.raises(ValueError, match="not 'merge'"):
        envs.tuned_table()
    _use(monkeypatch, None, "replace")
    with pytest.raises(ValueError, match="is set without AITER_MOEV2_TUNED_TABLE"):
        envs.tuned_table()
    _use(monkeypatch, ["a.csv", "", "b.csv"], " Extend ")
    assert envs.tuned_table() == (("a.csv", "b.csv"), "extend")


def test_the_modes_combine_the_users_tables_with_the_shipped(tmp_path, monkeypatch):
    shipped = _csv(tmp_path, "shipped.csv", [_row(40, "s40"), _row(48, "s48")])
    user = _csv(tmp_path, "user.csv", [_row(40, "u40"), _row(56, "u56")])
    _shipped(monkeypatch, shipped)
    k40, k48, k56 = (_problem(_row(e)) for e in (40, 48, 56))

    def kernels(files, mode):
        _use(monkeypatch, files, mode)
        return [_kernel(tuned_table.lookup(p)) for p in (k40, k48, k56)]

    _use(monkeypatch)
    assert kernels(None, None) == [f"{_CK1}_s40", f"{_CK1}_s48", None]
    assert kernels([user], "replace") == [f"{_CK1}_u40", None, f"{_CK1}_u56"]
    assert kernels([user], "override") == [
        f"{_CK1}_u40",
        f"{_CK1}_s48",
        f"{_CK1}_u56",
    ]
    with pytest.raises(ValueError, match="keys in both the shipped tables and"):
        kernels([user], "extend")
    only_new = _csv(tmp_path, "new.csv", [_row(56, "u56")])
    assert kernels([only_new], "extend") == [
        f"{_CK1}_s40",
        f"{_CK1}_s48",
        f"{_CK1}_u56",
    ]
    again = _csv(tmp_path, "again.csv", [_row(56, "v56")])
    with pytest.raises(ValueError, match="keys in both .*new.csv and .*again.csv"):
        kernels([only_new, again], "replace")


def test_override_reads_the_shipped_tables_only_after_the_users_fallbacks(
    tmp_path, monkeypatch
):
    gelu = _row(40, act="Gelu", one_stage=True)
    _shipped(monkeypatch, _csv(tmp_path, "shipped.csv", [gelu]))
    # The user's row for Silu is the activation-agnostic row of the Gelu key.
    silu = _row(40, one_stage=True, block_m=64)
    user = _csv(tmp_path, "user.csv", [silu])
    _use(monkeypatch, [user], "override")
    found = tuned_table.lookup(_problem(gelu))
    assert found.cells["block_m"] == 64
    big = _row(40, "big", token=131072)
    _shipped(monkeypatch, _csv(tmp_path, "shipped2.csv", [big]))
    tier = _csv(tmp_path, "tier.csv", [_row(40, "tier", token=32768)])
    _use(monkeypatch, [tier], "override")
    assert _kernel(tuned_table.lookup(_problem(big))) == f"{_CK1}_tier"
    _use(monkeypatch, [_csv(tmp_path, "other.csv", [_row(48)])], "override")
    assert _kernel(tuned_table.lookup(_problem(big))) == f"{_CK1}_big"


def test_extend_falls_back_to_the_shipped_rows_first(tmp_path, monkeypatch):
    target = _problem(_row(40, act="Gelu", one_stage=True))
    shipped = _csv(tmp_path, "shipped.csv", [_row(40, one_stage=True, block_m=16)])
    _shipped(monkeypatch, shipped)
    user = _csv(tmp_path, "user.csv", [_row(40, act="Swiglu", one_stage=True)])
    _use(monkeypatch, [user], "extend")
    assert tuned_table.lookup(target).cells["block_m"] == 16


def test_a_legacy_table_reads_as_legacy_reads_it(tmp_path, monkeypatch):
    legacy = _csv(tmp_path, "t.csv", [_row(40, "a"), _row(40, "b"), _row(48, "c")])
    _use(monkeypatch, [legacy], "replace")
    found = tuned_table.lookup(_problem(_row(40)))
    primary, _ = tuned.read_table(str(legacy), tuned.index_columns(False), False)
    assert found == tuned_table.Found(cells=convert.cells(next(iter(primary.values()))))
    assert found.cells["kernelName1"] == f"{_CK1}_a"  # the first of two, as legacy
    # Rows whose key no call has are never found, as in legacy.
    assert tuned_table.legacy_key(("use_g1u1",), ("1",)) == ("?1",) + ("",) * 17
    assert tuned_table.legacy_key(("use_g1u1",), (2,))[0].startswith("?")
    assert tuned_table.legacy_key(("cu_num", "use_g1u1"), (80.0, 1))[:2] == (
        "80",
        "True",
    )


def test_a_plan_file_holds_cells_and_tuner_rows(tmp_path, monkeypatch):
    legacy = _csv(tmp_path, "t.csv", [_row(40, one_stage=True)])
    primary, _ = tuned.read_table(str(legacy), tuned.index_columns(False), False)
    rows, skipped = convert.convert_table((primary, {}), "gfx950", 256)
    assert rows and not skipped
    plan_file = tmp_path / "plans.csv"
    with open(plan_file, "w", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS + tuned_plans.INFO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _use(monkeypatch, [plan_file], "replace")
    problem = _problem(_row(40, one_stage=True))
    expected = tuned_table.Found(cells=convert.cells(next(iter(primary.values()))))
    assert tuned_table.lookup(problem) == expected
    # A legacy-mode row's cells fall back as legacy's row does.
    gelu = _problem(_row(40, act="Gelu", one_stage=True))
    assert tuned_table.lookup(gelu) == expected
    with open(plan_file, "a", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS + tuned_plans.INFO_COLUMNS)
        writer.writerow({**rows[0], "path": "a8w8_ptok_other", "mode": "e2e"})
    _use(monkeypatch, [plan_file], "replace")
    found = tuned_table.lookup(problem)
    assert found.cells == expected.cells and len(found.tuner) == 1
    with open(plan_file, "a", newline="") as f:
        writer = csv.DictWriter(f, tuned_plans.COLUMNS + tuned_plans.INFO_COLUMNS)
        writer.writerow({**rows[0], "path": "no_such_path"})
    _use(monkeypatch, [plan_file], "replace")
    with pytest.raises(ValueError, match="path 'no_such_path' is not registered"):
        tuned_table.lookup(problem)


@pytest.mark.parametrize(
    "header, row, match",
    [
        (_HEADER + ",route_bucket", _row() + ",b1", "Opus route columns"),
        (_HEADER, _row(block_m=32.5), "block_m 32.5 is not a whole number"),
        (_HEADER, _row(ksplit=""), "ksplit nan is not a whole number"),
        (
            _HEADER,
            _row(one_stage=True, kernelName2="moe_ck2stages_gemm2_x"),
            "a 1-stage row with kernelName2 'moe_ck2stages_gemm2_x'",
        ),
        ("token,model_dim", "1,2", "neither a v2 plan file"),
    ],
)
def test_rows_v2_cannot_hold_raise(tmp_path, monkeypatch, header, row, match):
    _use(monkeypatch, [_csv(tmp_path, "t.csv", [row], header)], "replace")
    with pytest.raises(ValueError, match=match):
        tuned_table.lookup(_problem(_row()))


def test_a_number_as_kernel_name_raises(tmp_path, monkeypatch):
    rows = [_row(kernelName1=123, kernelName2=456)]
    _use(monkeypatch, [_csv(tmp_path, "t.csv", rows)], "replace")
    with pytest.raises(ValueError, match="kernelName1 123 is not a kernel name"):
        tuned_table.lookup(_problem(_row()))


def test_a_dedicated_table_and_its_declarations(tmp_path, monkeypatch):
    from aiter.moev2.migration import fhmoe
    from aiter.ops.flydsl.moe_common import GateMode

    table = _csv(tmp_path, "fhmoe.csv", [_FHMOE_FP8_ROW], _FHMOE_HEADER)
    _use(monkeypatch, [table], "replace")
    geometry = {"model_dim": 6144, "inter_dim": 256, "experts": 257, "topk": 9}
    contract = {
        "dtype": convert._dtype("torch.bfloat16"),
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": GateMode.INTERLEAVE,
        "doweight_stage1": False,
    }
    assert fhmoe._declares_mxfp8(**geometry)
    assert fhmoe._declares_mxfp8(**geometry, **contract)
    assert not fhmoe._declares_mxfp8(**geometry, **{**contract, "hidden_pad": 128})
    assert not fhmoe._declares_mxfp8(**{**geometry, "topk": 8})
    problem = convert.problem_of(
        dict(zip(_FHMOE_HEADER.split(","), _FHMOE_FP8_ROW.split(","))),
        "gfx950",
        256,
        str(table),
    )
    assert problem.extras["tuned_table"] == str(table)
    assert tuned_table.lookup(problem).cells["block_m"] == 32
    _use(monkeypatch, [_csv(tmp_path, "main.csv", [_row()])], "replace")
    assert not fhmoe._declares_mxfp8(**geometry)
    assert tuned_table.lookup(problem) == tuned_table.Found()

    tiered = _FHMOE_FP8_ROW.replace("gfx950,256,1,", "gfx950,256,32768,", 1)
    _use(
        monkeypatch, [_csv(tmp_path, "tiered.csv", [tiered], _FHMOE_HEADER)], "replace"
    )
    named = dict(zip(_FHMOE_HEADER.split(","), tiered.split(",")))
    at = {
        m: convert.problem_of({**named, "token": str(m)}, "gfx950", 256, str(table))
        for m in (32768, 131072)
    }
    assert tuned_table.lookup(at[32768]).cells["block_m"] == 32
    assert tuned_table.lookup(at[131072]) == tuned_table.Found()


def test_the_converted_shipped_tables_raise_when_stale(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "model_configs").mkdir(parents=True)
    (configs / "moev2").mkdir()
    for name in ("tuned_fmoe.csv", "tuned_fhmoe.csv"):
        (configs / name).write_text(f"{_HEADER}\n")
        (configs / "moev2" / name).write_text(",".join(tuned_plans.COLUMNS) + "\n")

    def sources(names):
        with open(configs / "moev2" / tuned_table.SOURCES, "w", newline="") as f:
            f.write("source,sha256\n")
            for name in names:
                sha = hashlib.sha256((configs / name).read_bytes()).hexdigest()
                f.write(f"{name},{sha}\n")

    monkeypatch.setattr(tuned_table, "CONFIGS", configs)
    monkeypatch.setattr(tuned_table, "CONVERTED", configs / "moev2")
    tuned_table._shipped.cache_clear()
    try:
        sources(["tuned_fmoe.csv", "tuned_fhmoe.csv"])
        table, _ = tuned_table._shipped()
        assert table == tuned_table.Table()
        (configs / "tuned_fmoe.csv").write_text(f"{_HEADER}\n{_row()}\n")
        tuned_table._shipped.cache_clear()
        with pytest.raises(ValueError, match="tuned_fmoe.csv; rerun python tests"):
            tuned_table._shipped()
        sources(["tuned_fmoe.csv", "tuned_fhmoe.csv"])
        (configs / "model_configs" / "m_tuned_fmoe.csv").write_text(f"{_HEADER}\n")
        tuned_table._shipped.cache_clear()
        with pytest.raises(ValueError, match="m_tuned_fmoe.csv .not converted."):
            tuned_table._shipped()
        (configs / "model_configs" / "m_tuned_fmoe.csv").unlink()
        sources(["tuned_fmoe.csv", "tuned_fhmoe.csv"])
        with open(configs / "moev2" / tuned_table.SOURCES, "a") as f:
            f.write("model_configs/old_tuned_fmoe.csv,0\n")
        tuned_table._shipped.cache_clear()
        with pytest.raises(ValueError, match="old_tuned_fmoe.csv .gone."):
            tuned_table._shipped()
        (configs / "model_configs" / "m_tuned_fhmoe.csv").write_text("x\n")
        tuned_table._shipped.cache_clear()
        with pytest.raises(ValueError, match="m_tuned_fhmoe.csv into its FHMoE"):
            tuned_table._shipped()
    finally:
        tuned_table._shipped.cache_clear()


def test_the_converted_shipped_tables_are_the_legacy_ones():
    """Every row and activation-agnostic row of the shipped tables as legacy reads them
    (its merge of the default table) has its cells in the converted tables under its
    key, and nothing else is there, but for the rows no path takes
    (`golden/convert_skipped.json`)."""
    from aiter.jit.core import AITER_CONFIGS

    def text_keys(columns, rows):
        return {tuned_table.legacy_key(columns, k): convert.cells(v) for k, v in rows}

    main_columns, dedicated_columns = (tuned.index_columns(d) for d in (False, True))
    primary, fallback = tuned.read_table(
        AITER_CONFIGS.AITER_CONFIG_FMOE_FILE, main_columns, False
    )
    dedicated, _ = tuned.read_table(
        AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE, dedicated_columns, True
    )
    shipped, _ = tuned_table._shipped()
    want = text_keys(main_columns, primary.items())
    want |= text_keys(dedicated_columns, dedicated.items())
    skipped = want.keys() - shipped.cells.keys()
    assert len(skipped) == 36 and {k[0] for k in skipped} == {"gfx1201"}
    assert shipped.cells == {k: v for k, v in want.items() if k not in skipped}
    want = text_keys(main_columns, fallback.items())
    assert shipped.fallback == {k: v for k, v in want.items() if k[0] != "gfx1201"}
    assert shipped.tuner == {}


def test_v2_hands_its_calls_back_while_legacys_variables_are_set(
    monkeypatch, caplog, tmp_path
):
    path = types.SimpleNamespace(
        name="a16w16", entry="api", claims=lambda call: True, graduated_claims=None
    )
    monkeypatch.setattr(paths, "active", lambda: (path,))
    monkeypatch.setenv("AITER_MOEV2", "on")
    _use(monkeypatch)
    assert paths.route_to_v2({}) is path
    monkeypatch.setenv("AITER_CONFIG_FMOE", "/t.csv")
    _use(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="aiter"):
        assert paths.route_to_v2({}) is None
        assert paths.route_to_v2({}) is None
    warnings = [r for r in caplog.records if "AITER_CONFIG_FMOE set" in r.message]
    assert len(warnings) == 1
    with pytest.raises(ValueError, match="AITER_CONFIG_FMOE is set, which v2 does"):
        tuned_table.lookup(_problem(_row()))
    with pytest.raises(ValueError, match="AITER_CONFIG_FMOE is set, which v2 does"):
        tuned_table.digest()
    _use(monkeypatch, [_csv(tmp_path, "t.csv", [_row()])], "override")
    with pytest.raises(ValueError, match="are both set"):
        paths.route_to_v2({})
    with pytest.raises(ValueError, match="are both set"):
        tuned_table.lookup(_problem(_row()))
    monkeypatch.delenv("AITER_CONFIG_FMOE")
    _use(monkeypatch)
    assert paths.route_to_v2({}) is path
