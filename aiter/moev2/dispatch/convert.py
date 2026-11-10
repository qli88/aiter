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
`us2` and `us` (`INFO_COLUMNS`) are kept as read (numbers as Python prints them), and
nothing reads them.

A row gets none (`convert_table` gives the reason) when its key is not read, when
dispatch never looks its key up (an FHMoE row outside the contracts that read the
FHMoE table, whose calls read the main table; a key looked up as another), when no
path takes the key, and per path: no feasible plan, `match_row` declines the row,
it holds Opus route columns (`ROUTE_COLUMNS`), or its cells do not come back from
the candidates.
"""

import math

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
from . import resolver, tuned, tuned_plans
from .ops import Op

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
MODE = "legacy"
# Read by Opus stage 2 into the launch metadata (`route_bucket_metadata`).
ROUTE_COLUMNS = (
    "route_bucket",
    "route_bucket_name",
    "expected_sorted_blocks",
    "expected_route_blocks",
    "min_sorted_blocks",
    "min_route_blocks",
    "max_sorted_blocks",
    "max_route_blocks",
)


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
    (`cells`); raises for a tunable segment that runs neither stage's kernel."""
    shared = Params(
        block_m=row_cells["block_m"],
        ksplit=row_cells["ksplit"],
        nt=bool(row_cells["nt"]),
        xbf16=bool(row_cells["xbf16"]),
        flat=row_cells["flat"],
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
    order, and `[(key, reason)]` for each row, or each path of a row, that gets
    none."""
    primary, _ = table
    columns = tuned.index_columns(fhmoe_table is not None)
    rows, skipped = [], []
    for key, row in primary.items():
        named = {c: "" if _blank(v) else v for c, v in zip(columns, key)}
        if named["gfx"] != arch or int(named["cu_num"] or -1) != cu_num:
            continue
        try:
            problem = problem_of(named, arch, cu_num, fhmoe_table)
        except (AttributeError, TypeError, ValueError) as error:
            skipped.append((key, f"the key is not read: {error}"))
            continue
        if fhmoe_table is not None and problem.extras.get("tuned_table") is None:
            skipped.append((key, "an FHMoE key its calls look up in the main table"))
            continue
        if tuned.exact_key(problem) != key:
            skipped.append((key, f"looked up as {tuned.exact_key(problem)}"))
            continue
        names = paths_taking(problem)
        if not names:
            skipped.append((key, "no path takes the key"))
            continue
        for name in names:
            converted = plan_row(problem, paths.registered()[name].segments, row)
            if isinstance(converted, str):
                skipped.append((key, f"{name}: {converted}"))
                continue
            structure, candidates = converted
            out = dict(zip(tuned_plans.KEY_COLUMNS, tuned_plans.key_values(problem)))
            out.update(
                path=name,
                structure=tuned_plans.STRUCTURE_SEPARATOR.join(
                    s.name for s in structure
                ),
                candidates=tuned_plans.encode_candidates(candidates),
                e2e_us=_text(row.get("us")),
                mode=MODE,
            )
            out.update({c: _text(row.get(c)) for c in INFO_COLUMNS})
            rows.append(out)
    return rows, skipped
