# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Legacy tuned rows as v2 plan-file rows (`tuned_plans`, `mode` "legacy").

A legacy row gets one plan-file row per path taking its base call (its key with
shuffled weights, no bias, no expert mask: `problem_of`), keyed as dispatch looks the
call up. The structure is the one `match_row` puts the row on there; the candidates
hold the cells dispatch reads, which `resolver.candidate_row` gives back (`cells`):
each tunable segment's kernel (stage 1's kernelName1, stage 2's kernelName2, a 1-stage
segment's kernelName1) with the row's block_m, ksplit, nt, xbf16 and flat. A 1-stage
row's kernelName2 (empty, "Null" or a blank cell) is not kept: dispatch picks and
discards alike for the three, and logs it as empty. `e2e_us` is the row's `us`; `us1`,
`us2` and `us` (`INFO_COLUMNS`) are kept as the table reads them (numbers as Python
prints them, an int as a float in a column the merge reads as floats), and nothing
reads them.

A row gets none (`convert_table` gives the reason) when its key is not read, when
dispatch never looks its key up (an FHMoE row outside the contracts that read the
FHMoE table, whose calls read the main table; a key looked up as another), when no
path takes the key, and per path: no feasible plan, `match_row` declines the row,
it holds Opus route columns (`ROUTE_COLUMNS`), or its cells do not come back from
the candidates.
"""

import math
import os
import tempfile

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..segments.base import Params
from ..segments.defaults import BLOCK_SIZE_M
from ..segments.flydsl_fhmoe import (
    _config_has_mxfp8_fhmoe_contract,
    _config_has_mxfp8_fhmoe_geometry,
    _is_dsv4_i384_fhmoe_contract,
)
from . import resolver, tuned, tuned_plans, tuned_table
from .ops import Op
from .tuned_table import (  # noqa: F401
    CONFIGS,
    CONVERTED,
    ROUTE_COLUMNS,
    SOURCES,
    shipped_sources,
)

KEY_FIELDS = (
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
INFO_COLUMNS = tuned_plans.INFO_COLUMNS
MODE = tuned_plans.LEGACY_MODE
_SOURCE_COLUMN = "_moev2_source"


def _dtype(text):
    return getattr(torch, str(text).strip().replace("torch.", ""))


def _enum(cls, text):
    return getattr(cls, str(text).strip().split(".")[-1])


def _declares_mxfp8(fhmoe_table, args, gate_mode) -> bool:
    """Whether FHMoE table `fhmoe_table` declares the fp8-weight contract of the key
    `args` (as dispatch decides it reads that table); raises where dispatch raises,
    for a key of a geometry the table declares under another contract."""
    geometry = dict(zip(("model_dim", "inter_dim", "experts", "topk"), args[1:5]))
    if not _config_has_mxfp8_fhmoe_geometry(fhmoe_table, **geometry):
        return False
    request = dict(
        geometry,
        dtype=args[5],
        hidden_pad=args[12],
        intermediate_pad=args[13],
        gate_mode=gate_mode,
        doweight_stage1=args[11],
    )
    if not _config_has_mxfp8_fhmoe_contract(fhmoe_table, **request):
        raise ValueError(
            "MXFP8/FP8 FHMoE request partially matches a dedicated CSV contract but "
            "differs in dtype, padding, gate mode, or doweight_stage1"
        )
    return True


def problem_of(row, arch, cu_num, fhmoe_table=None):
    """The `MoeProblem` of an input row (a dict of strings); `fhmoe_table`: the row is
    an FHMoE key, of that table where dispatch reads it (the DSV4 I384 contract, or an
    fp8-weight contract the table declares)."""
    missing = [f for f in KEY_FIELDS if f not in row]
    if missing:
        raise ValueError(f"row without {missing}")
    dtype = _dtype(row["dtype"])
    q_dtype_a = _dtype(row["q_dtype_a"])
    activation = _enum(ActivationType, row["act_type"])
    args = (
        int(row["token"]),
        int(row["model_dim"]),
        int(row["inter_dim"]),
        int(row["expert"]),
        int(row["topk"]),
        dtype,
        q_dtype_a,
        _dtype(row["q_dtype_w"]),
        _enum(QuantType, row["q_type"]),
        bool(int(row["use_g1u1"])),
        activation,
        bool(int(row["doweight_stage1"])),
        int(row.get("hidden_pad") or 0),
        int(row.get("intermediate_pad") or 0),
    )
    kwargs = {"input_dtype": dtype}
    if fhmoe_table is not None:
        if row.get("gate_mode"):
            gate_mode = _enum(GateMode, row["gate_mode"])
        else:
            fp8 = q_dtype_a in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            gate_mode = GateMode.INTERLEAVE if fp8 else GateMode.SEPARATED
        shape, pads = args[1:5], args[12:14]
        doweight, q_dtype_w = args[11], args[7]
        dedicated = _is_dsv4_i384_fhmoe_contract(
            *shape,
            *pads,
            gate_mode,
            doweight,
            routed_mxfp4=q_dtype_w == torch.float4_e2m1fn_x2,
        ) or (q_dtype_w == dtypes.fp8 and _declares_mxfp8(fhmoe_table, args, gate_mode))
        kwargs.update(
            gate_mode=gate_mode, config_file=fhmoe_table if dedicated else None
        )
    return api.problem_from_key(
        args, kwargs, arch, cu_num, fhmoe=fhmoe_table is not None
    )


def paths_taking(problem, only=None):
    return [
        p.name
        for p in paths.registered().values()
        if p.segments
        and p.takes is not None
        and (not only or p.name in only)
        and p.takes(problem)
    ]


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _text(value) -> str:
    return "" if _blank(value) else str(value)


def cells(row) -> dict:
    """The cells of legacy tuned `row` dispatch reads, as it reads them, in
    `resolver.candidate_row`'s form."""
    nt, run_1stage, xbf16, flat = resolver._row_values(row)
    one_stage = bool(run_1stage)
    return {
        "kernelName1": _text(row.get("kernelName1")),
        "kernelName2": "" if one_stage else _text(row.get("kernelName2")),
        "block_m": int(row.get("block_m", BLOCK_SIZE_M)),
        "ksplit": int(row.get("ksplit", 0)),
        "nt": int(bool(nt)),
        "run_1stage": int(one_stage),
        "xbf16": int(bool(xbf16)),
        "flat": int(flat),
    }


def candidates_of(structure, row_cells) -> dict:
    """`{segment name: Params}` of `structure`'s tunable segments holding `row_cells`
    (`cells`): the kernel, `block_m`, and the ksplit, nt, xbf16 and flat that are set;
    raises for a tunable segment that runs neither stage's kernel."""
    shared = Params(
        block_m=row_cells["block_m"],
        **{f: row_cells[f] for f in ("ksplit", "flat") if row_cells[f]},
        **{f: True for f in ("nt", "xbf16") if row_cells[f]},
    )
    one_stage = len(structure) == 1 or resolver._one_stage(structure) is not None
    gemm1 = None if one_stage else resolver._gemm1(structure)
    candidates = {}
    for segment in structure:
        if not resolver.tunable(segment):
            continue
        if one_stage or segment is gemm1:
            kernel = row_cells["kernelName1"]
        elif segment.covers[0] == Op.GEMM2:
            kernel = row_cells["kernelName2"]
        else:
            raise ValueError(f"{segment.name} runs neither stage's kernel")
        candidates[segment.name] = shared.replace(kernel=kernel)
    return candidates


def plan_row(problem, segments, row):
    """`(structure, candidates)` of legacy tuned `row` on `problem`, one of the
    calls the row's key looks up, and a path's `segments`, or the reason (str) it has
    none."""
    plans, _ = resolver.feasible_plans(problem, segments)
    if not plans:
        return "no feasible plan"
    structure = resolver.match_row(plans, row)
    if isinstance(structure, str):
        return f"dispatch declines the row: {structure}"
    route = sorted(c for c in ROUTE_COLUMNS if not _blank(row.get(c)))
    if route:
        return f"Opus route columns the plan file does not hold: {', '.join(route)}"
    try:
        want = cells(row)
        candidates = candidates_of(structure, want)
        got = resolver.candidate_row(structure, candidates)
    except (TypeError, ValueError) as error:
        return f"the cells do not fit the structure: {error}"
    if got != want:
        diff = sorted(k for k in want if want[k] != got[k])
        return "the cells do not come back: " + ", ".join(
            f"{k} {want[k]!r} vs {got[k]!r}" for k in diff
        )
    return structure, candidates


def convert_table(table, arch, cu_num, fhmoe_table=None):
    """`(rows, skipped)` of a legacy tuned table read by `tuned.read_table` (its
    primary rows; `fhmoe_table`: the dedicated table's file name, the table being
    it), for its rows of `arch` and `cu_num` (the others are left out): the
    plan-file rows (dicts over `tuned_plans.COLUMNS + INFO_COLUMNS`), in table
    order, each row's comm_fused rows (one per `comm_fused_configs` entry: its a8w4
    row with the runner) after it, and `[(key, reason)]` for each row, each path of a
    row, or each comm_fused entry, that gets none. Raises for an entry legacy's winner
    table rejects."""
    primary, _ = table
    columns = tuned.index_columns(fhmoe_table is not None)
    rows, skipped = [], []
    for key, row in primary.items():
        named = {c: "" if _blank(v) else v for c, v in zip(columns, key)}
        if named["gfx"] != arch or int(named["cu_num"] or -1) != cu_num:
            continue
        if fhmoe_table is not None and tuned_table.has_comm_configs(row):
            raise ValueError(
                f"the row of {key}: comm_fused_configs in an FHMoE table, which legacy "
                "does not read"
            )
        entries = tuned_table.legacy_comm_entries(
            {**dict(zip(columns, key)), **row}, f"the row of {key}"
        )
        problem, local = _convert_row(key, row, named, fhmoe_table, rows, skipped)
        for tp, comm, add_shared, gpu_model, runner, us in entries:
            if local is None:
                skipped.append(
                    (key, f"comm_fused: no {tuned_plans.COMM_FUSED_LOCAL_PATH} row")
                )
                continue
            collective = tuned_plans.comm_key_values(
                problem, tp, comm, add_shared, gpu_model
            )
            out = dict(zip(tuned_plans.KEY_COLUMNS, collective))
            out.update(
                path=tuned_plans.COMM_FUSED_PATH,
                structure=local[0],
                candidates=tuned_plans.encode_candidates(
                    {**local[1], tuned_plans.RUNNER: runner}
                ),
                e2e_us=_text(us),
                mode=MODE,
            )
            out.update(dict.fromkeys(INFO_COLUMNS, ""))
            rows.append(out)
    return rows, skipped


def _convert_row(key, row, named, fhmoe_table, rows, skipped):
    """Append `row`'s plan-file rows (one per path taking its key) to `rows`, or the
    reasons it has none to `skipped`; `(problem, (structure, candidates) of its a8w4
    row, or None)`."""
    arch, cu_num = named["gfx"], int(named["cu_num"])
    try:
        problem = problem_of(named, arch, cu_num, fhmoe_table)
    except (AttributeError, TypeError, ValueError) as error:
        skipped.append((key, f"the key is not read: {error}"))
        return None, None
    if fhmoe_table is not None and problem.extras.get("tuned_table") is None:
        skipped.append((key, "an FHMoE key its calls look up in the main table"))
        return problem, None
    if tuned.exact_key(problem) != key:
        skipped.append((key, f"looked up as {tuned.exact_key(problem)}"))
        return problem, None
    names = paths_taking(problem)
    if not names:
        skipped.append((key, "no path takes the key"))
        return problem, None
    local = None
    for name in names:
        converted = plan_row(problem, paths.registered()[name].segments, row)
        if isinstance(converted, str):
            skipped.append((key, f"{name}: {converted}"))
            continue
        structure, candidates = converted
        out = dict(zip(tuned_plans.KEY_COLUMNS, tuned_plans.key_values(problem)))
        out.update(
            path=name,
            structure=tuned_plans.STRUCTURE_SEPARATOR.join(s.name for s in structure),
            candidates=tuned_plans.encode_candidates(candidates),
            e2e_us=_text(row.get("us")),
            mode=MODE,
        )
        out.update({c: _text(row.get(c)) for c in INFO_COLUMNS})
        rows.append(out)
        if name == tuned_plans.COMM_FUSED_LOCAL_PATH:
            local = out["structure"], candidates
    return problem, local


def merged_tables(files) -> dict:
    """`{file: (primary, {})}` of the legacy tuned CSVs `files` merged as the default
    table is (`update_config_files`: a lacking `gfx` column comes from `cu_num`, any
    other column a file lacks reads 0 in it; `tuned.read_table` reads the merge back),
    each `file` with its primary rows of the merge, as `convert_table` takes a table.
    Raises for a key in two rows: where the merge raises (also with the same `_tag`),
    and where `read_table` would keep the first of two rows, which in the legacy merge
    depends on the order the directory lists the files; and for `comm_fused_configs`
    on a row `read_table` drops."""
    import pandas as pd

    frames = [pd.read_csv(f) for f in files]
    frames = [tuned.ensure_gfx_column(f) if "gfx" not in f else f for f in frames]
    columns = list(dict.fromkeys(c for frame in frames for c in frame.columns))
    for frame in frames:
        for c in columns:
            if c not in frame.columns:
                frame[c] = 0
    merged = pd.concat(
        [
            f[columns].assign(**{_SOURCE_COLUMN: str(n)})
            for n, f in zip(files, frames)
            if not f.empty
        ],
        ignore_index=True,
    )
    index = tuned.index_columns(False)
    tag = merged.get("_tag", pd.Series("", index=merged.index)).fillna("")
    same_tag = merged[index].assign(_tag=tag).duplicated(keep=False)
    read = merged[tag != "flydsl_fallback"]
    first_kept = read.duplicated(subset=index, keep=False)
    twice = merged[same_tag | first_kept.reindex(merged.index, fill_value=False)]
    if len(twice):
        raise ValueError(f"keys in two rows:\n{twice.to_string(index=False)}")
    dropped = merged[tag == "flydsl_fallback"]
    for _, row in dropped.iterrows():
        if tuned_table.has_comm_configs(dict(row)):
            raise ValueError(
                "comm_fused_configs on a flydsl_fallback row, which the reader drops:\n"
                f"{row.to_frame().T.to_string(index=False)}"
            )
    with tempfile.TemporaryDirectory() as scratch:
        name = os.path.join(scratch, "merged.csv")
        merged.to_csv(name, index=False)
        primary, _ = tuned.read_table(name, tuned.index_columns(False), False)
    tables = {str(n): ({}, {}) for n in files}
    for key, row in primary.items():
        row = dict(row)
        tables[row.pop(_SOURCE_COLUMN)][0][key] = row
    return tables
