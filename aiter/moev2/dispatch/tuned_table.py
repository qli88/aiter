# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tuned table v2 dispatch reads: the shipped tables and the user's.

The shipped tables are the converted legacy tables (`convert`): one plan file under
`CONVERTED` per legacy file, read in the order `SOURCES` lists them. That file records
each legacy file's sha256; a converted table that no longer matches its legacy file, or
a legacy file added or removed, raises (the conversion is rerun with
`python tests/moe_dispatch/convert_tables.py --update`).

The user's tables are the files `AITER_MOEV2_TUNED_TABLE` names (`envs.tuned_table`),
each a v2 plan file (a header with `structure` and `candidates`) or a legacy tuned
CSV, read on its own (a legacy table with the FHMoE columns as a dedicated table), in
the listed order; a key in two of them raises. A legacy row holding Opus route columns,
a kernel name that is not text, a `block_m` or `ksplit` that is not a whole number, or
a 1-stage kernelName2 (other than empty or "Null") raises.
`AITER_MOEV2_TUNED_TABLE_MODE` combines them with the shipped tables: `replace` reads
the user's alone, `override` the user's first and the shipped tables for a key the
user's have no row for (exact or fallback), `extend` both as one table (a key in both
raises; the activation-agnostic fallback takes the shipped rows first).

A table holds, per key, the legacy row's cells (`resolver.candidate_row`'s form, from a
plan-file row of mode "legacy" or a legacy table's row), which `resolver.resolve` runs
as the legacy row with its fallbacks (`lookup`: exact key, then the activation-agnostic
row, then smaller token tiers; a dedicated table exact only), and the tuner's rows
(any other mode), by exact key, which `resolver.resolve` builds first. A comm_fused
row (`comm_lookup`, by exact key with the collective's columns) is the tuner's, or a
legacy row's runner with the cells of the a8w4 plan it runs with (`CommCells`: a
plan-file row of mode "legacy", or a legacy table's `comm_fused_configs` entry, checked
as legacy's winner table checks it).

v2 does not read legacy's `AITER_CONFIG_FMOE` or `AITER_CONFIG_FHMOE`: while one is set
the legacy entry points take their calls back (`paths.route_to_v2`); reading the table
then raises.
"""

import csv
import functools
import hashlib
import io
import json
import math
import pathlib
from dataclasses import dataclass, field

from aiter.ops.flydsl.mxfp4_kname import _is_mxfp4_kname

from .. import envs
from ..shapes import _PADDED_M_TIERS
from . import tuned, tuned_plans
from .tuned import _ACT_TYPE_DISABLED_KEY, cached

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "configs"
# The converted shipped tables, one per legacy file at the same path under it, and
# `SOURCES` (columns source, sha256: each legacy file, relative to `CONFIGS`, and its
# sha256).
CONVERTED = CONFIGS / "moev2"
SOURCES = "sources.csv"
# The dedicated table a call of the FHMoE path names (`MoeProblem.extras`); which rows
# it has is this module's.
FHMOE_TABLE = str(CONFIGS / "tuned_fhmoe.csv")
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
_LOCAL = len(tuned_plans.LOCAL_KEY_COLUMNS)
_MAIN = len(tuned.INDEX_COLUMNS)
_ACT = tuned.INDEX_COLUMNS.index("act_type")
_TOKEN = tuned.INDEX_COLUMNS.index("token")
_WHOLE_COLUMNS = frozenset(
    {
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "shared_expert_id",
        "hidden_pad",
        "intermediate_pad",
    }
)
_BOOL_COLUMNS = frozenset({"use_g1u1", "doweight_stage1"})


def shipped_sources():
    """`(main, fhmoe)`: the legacy files the default table merges (`tuned_fmoe.csv`,
    then the `model_configs/` tables `get_config_file` globs, here sorted) and the
    FHMoE table."""
    models = sorted(
        p
        for p in (CONFIGS / "model_configs").glob("*tuned_fmoe*.csv")
        if p.is_file() and "untuned" not in p.name
    )
    return [CONFIGS / "tuned_fmoe.csv", *models], CONFIGS / "tuned_fhmoe.csv"


def _model_fhmoe_tables() -> list:
    """The `model_configs/` FHMoE tables legacy merges into the FHMoE table."""
    return sorted(
        p
        for p in (CONFIGS / "model_configs").glob("*tuned_fhmoe*.csv")
        if p.is_file() and "untuned" not in p.name
    )


@dataclass
class Table:
    # local key (`tuned_plans.LOCAL_KEY_COLUMNS` values) -> the legacy row's cells
    cells: dict = field(default_factory=dict)
    # main key with `act_type` disabled -> cells, the first row of each
    fallback: dict = field(default_factory=dict)
    # key (`tuned_plans.KEY_COLUMNS` values) -> {path: (structure names, candidates)}
    tuner: dict = field(default_factory=dict)
    # key with the collective's columns -> the legacy comm_fused row's `CommCells`
    comm: dict = field(default_factory=dict)

    def all_keys(self) -> set:
        blank = ("",) * (len(tuned_plans.KEY_COLUMNS) - _LOCAL)
        return {k + blank for k in self.cells} | set(self.tuner) | set(self.comm)


@dataclass(frozen=True)
class CommCells:
    """A legacy comm_fused row: its runner (`Params`: the family and the config's
    params, flat) and the cells of the a8w4 plan it runs with."""

    runner: object
    cells: dict


@dataclass(frozen=True)
class Found:
    """A key's rows in the table: the tuner's (structure names, candidates) per path
    tuned, and the legacy row's cells (None: no row)."""

    tuner: tuple = ()
    cells: dict | None = None


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _disabled(key) -> tuple:
    return key[:_ACT] + (_ACT_TYPE_DISABLED_KEY,) + key[_ACT + 1 :]


def _at_token(key, token) -> tuple:
    return key[:_TOKEN] + (str(token),) + key[_TOKEN + 1 :]


def _falls_back(cells) -> bool:
    """Whether the legacy reader puts a row with `cells` in the activation-agnostic
    table."""
    return not (
        "moe_ck2stages" in cells["kernelName1"]
        or _is_mxfp4_kname(cells["kernelName1"])
        or _is_mxfp4_kname(cells["kernelName2"])
    )


def _fallback(cells) -> dict:
    fallback = {}
    for key, row_cells in cells.items():
        if not any(key[_MAIN:]) and _falls_back(row_cells):
            fallback.setdefault(_disabled(key), row_cells)
    return fallback


def _key_text(column, value) -> str:
    """A legacy table's key value as the key column holds it for the calls it equals
    (a value no call's equals gets one starting with "?")."""
    if column not in _WHOLE_COLUMNS | _BOOL_COLUMNS:
        return str(value)
    if isinstance(value, str):
        return f"?{value}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"?{value!r}"
    if not number.is_integer():
        return f"?{value!r}"
    if column in _WHOLE_COLUMNS:
        return str(int(number))
    return str(bool(number)) if number in (0, 1) else f"?{value!r}"


def legacy_key(columns, key) -> tuple:
    """The local key of a legacy table's row keyed `key` on `columns`."""
    text = tuple(_key_text(c, v) for c, v in zip(columns, key))
    return text + ("",) * (_LOCAL - len(text))


def legacy_cells(row, where) -> dict:
    """The cells of a user's legacy tuned `row` (`where`: the file and key, for the
    error); raises for a row v2 cannot hold as cells."""
    from .convert import cells
    from .resolver import _no_stage2

    route = sorted(c for c in ROUTE_COLUMNS if not _blank(row.get(c)))
    if route:
        raise ValueError(f"{where}: Opus route columns v2 does not read: {route}")
    for column in ("kernelName1", "kernelName2"):
        value = row.get(column)
        if not _blank(value) and not isinstance(value, str):
            raise ValueError(f"{where}: {column} {value!r} is not a kernel name")
    for column in ("block_m", "ksplit"):
        value = row.get(column)
        if column in row and not (
            not isinstance(value, (str, bool))
            and not _blank(value)
            and float(value).is_integer()
        ):
            raise ValueError(f"{where}: {column} {value!r} is not a whole number")
    try:
        row_cells = cells(row)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"{where}: {error!r}") from error
    if row_cells["run_1stage"] and not _no_stage2(row.get("kernelName2", "")):
        raise ValueError(
            f"{where}: a 1-stage row with kernelName2 {row['kernelName2']!r}, which "
            "v2 does not hold"
        )
    return row_cells


_COMM_MODES = ("ar", "rs")
_COMM_ENTRY_FIELDS = frozenset(
    {"family", "params", "tp", "add_shared", "gpu_model", "us"}
)


def _whole(value, what) -> int:
    if isinstance(value, bool) or not float(value).is_integer():
        raise ValueError(f"{what} {value!r} is not a whole number")
    return int(float(value))


def has_comm_configs(row) -> bool:
    """Whether legacy `row` has a `comm_fused_configs` value (`_comm_configs`)."""
    raw = row.get("comm_fused_configs")
    return not _blank(raw) and str(raw).strip() not in ("", "nan", "0")


def _check_runner(row, comm, tp, add_shared, entry) -> None:
    """Raise where legacy's winner table raises for the runner `entry` (`family`,
    `params`) of mode `comm` on legacy `row` (its key columns and `block_m`)."""
    from aiter.ops.flydsl.comm_fused_moe_host import ShapeKey, _config

    shape = ShapeKey(
        str(row["gfx"]),
        _whole(row["model_dim"], "model_dim"),
        _whole(row["inter_dim"], "inter_dim"),
        _whole(row["expert"], "expert"),
        _whole(row["topk"], "topk"),
        tp,
        add_shared=add_shared,
        comm=comm,
    )
    token = _whole(row["token"], "token")
    config = _config(entry, shape.kernel_shape(), token, comm)
    block_m = _whole(row["block_m"], "block_m")
    if config.sort_block_m != block_m:
        raise ValueError(
            f"comm_fused sort_block_m={config.sort_block_m} does not match "
            f"CSV block_m={block_m}"
        )


def legacy_comm_entries(row, where) -> list:
    """`[(tp, comm, add_shared, gpu_model, runner Params, us)]` of the
    `comm_fused_configs` of legacy tuned `row` (its key columns included; `where`: the
    file and key, for the error), checked as legacy's winner table checks them
    (`_winner_table`); raises where it raises."""
    from ..segments.base import Params

    if not has_comm_configs(row):
        return []
    try:
        configs = json.loads(row["comm_fused_configs"])
        if not isinstance(configs, dict):
            raise TypeError("comm_fused_configs must be a JSON object")
        unknown = sorted(set(configs) - set(_COMM_MODES))
        if unknown:
            raise ValueError(f"unsupported comm_fused modes: {unknown}")
        entries = []
        for comm, entry in configs.items():
            if not isinstance(entry, dict):
                raise TypeError(f"comm_fused_configs[{comm!r}] must be a JSON object")
            unknown = sorted(set(entry) - _COMM_ENTRY_FIELDS)
            if unknown:
                raise ValueError(
                    f"unsupported comm_fused_configs[{comm!r}] fields: {unknown}"
                )
            tp = _whole(entry["tp"], f"comm_fused_configs.{comm}.tp")
            add_shared = entry.get("add_shared")
            if not isinstance(add_shared, bool):
                raise TypeError(
                    f"comm_fused_configs[{comm!r}].add_shared must be boolean"
                )
            gpu_model = entry.get("gpu_model")
            if not isinstance(gpu_model, str) or not gpu_model.strip():
                raise ValueError(
                    f"comm_fused_configs[{comm!r}].gpu_model must be a non-empty string"
                )
            _check_runner(row, comm, tp, add_shared, entry)
            runner = Params(family=entry["family"], **entry["params"])
            entries.append(
                (
                    tp,
                    comm,
                    add_shared,
                    gpu_model.strip().lower(),
                    runner,
                    entry.get("us"),
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from error
    return entries


def comm_row_text(row) -> str:
    """A comm_fused row (`comm_lookup`'s) as text, equal for equal rows."""
    if isinstance(row, CommCells):
        return json.dumps(
            {"runner": dict(row.runner), "cells": row.cells}, sort_keys=True
        )
    names, candidates = row
    return json.dumps(
        {
            "structure": list(names),
            "candidates": tuned_plans.encode_candidates(candidates),
        }
    )


@cached
def _segments(path) -> dict:
    from .. import paths

    registered = paths.registered()
    if path not in registered:
        raise ValueError(f"path {path!r} is not registered")
    return {s.name: s for s in registered[path].segments}


def _plan_cells(path, structure, candidates) -> dict:
    from .resolver import candidate_row

    segments = _segments(path)
    names = tuned_plans.structure_names(structure)
    unknown = [n for n in names if n not in segments]
    if unknown:
        raise ValueError(f"segments {unknown} are not on path {path}")
    return candidate_row([segments[n] for n in names], candidates)


def _check_plan_runner(key, runner, cells) -> None:
    """Raise for a legacy comm_fused plan row (22-column `key`) whose collective
    columns or `runner` legacy's winner table would reject."""
    named = dict(zip(tuned_plans.KEY_COLUMNS, key))
    if named["add_shared"] not in ("True", "False"):
        raise TypeError(f"add_shared {named['add_shared']!r} is not True or False")
    if (
        not named["gpu_model"].strip()
        or named["gpu_model"] != named["gpu_model"].lower()
    ):
        raise ValueError(f"gpu_model {named['gpu_model']!r} is not a lowercase name")
    params = dict(runner)
    entry = {"family": params.pop("family", None), "params": params}
    tp = _whole(named["tp"], "tp")
    add_shared = named["add_shared"] == "True"
    _check_runner({**named, **cells}, named["comm"], tp, add_shared, entry)


def _plan_table(text, name) -> Table:
    table, seen = Table(), set()
    for row in tuned_plans.read_rows(text, name):
        key = tuple(row[c] for c in tuned_plans.KEY_COLUMNS)
        if (key, row["path"]) in seen:
            raise ValueError(f"{name}: two rows for {key} on {row['path']}")
        seen.add((key, row["path"]))
        try:
            if row.get("mode") != tuned_plans.LEGACY_MODE:
                table.tuner.setdefault(key, {})[row["path"]] = (
                    tuned_plans.structure_names(row["structure"]),
                    (
                        tuned_plans.decode_candidates(row["candidates"])
                        if row["candidates"]
                        else {}
                    ),
                )
                continue
            candidates = tuned_plans.decode_candidates(row["candidates"])
            comm_fused = row["path"] == tuned_plans.COMM_FUSED_PATH
            if any(key[_LOCAL:]) and not all(key[_LOCAL:]):
                raise ValueError("a legacy row with some collective columns blank")
            if comm_fused != any(key[_LOCAL:]):
                raise ValueError(
                    "a legacy comm_fused row without collective columns"
                    if comm_fused
                    else "a legacy row with collective columns"
                )
            if comm_fused:
                runner = candidates.pop(tuned_plans.RUNNER, None)
                if runner is None:
                    raise ValueError("a legacy comm_fused row without a runner")
                cells = _plan_cells(
                    tuned_plans.COMM_FUSED_LOCAL_PATH, row["structure"], candidates
                )
                _check_plan_runner(key, runner, cells)
                table.comm[key] = CommCells(runner, cells)
                continue
            row_cells = _plan_cells(row["path"], row["structure"], candidates)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"{name}: the row of {key} on {row['path']}: {error}")
        held = table.cells.setdefault(key[:_LOCAL], row_cells)
        if held != row_cells:
            raise ValueError(
                f"{name}: the rows of {key[:_LOCAL]} hold different cells: "
                f"{held} and {row_cells} (on {row['path']})"
            )
    table.fallback = _fallback(table.cells)
    return table


def _rows_with_comm_configs(name) -> int:
    with open(name, newline="") as f:
        return sum(has_comm_configs(row) for row in csv.DictReader(f))


def _legacy_table(name, header) -> Table:
    dedicated = set(tuned.DEDICATED_INDEX_COLUMNS) <= set(header)
    columns = tuned.index_columns(dedicated)
    primary, fallback = tuned.read_table(name, columns, dedicated)
    table, held = Table(), 0
    for key, row in primary.items():
        text = legacy_key(columns, key)
        where = f"{name}: the row of {key}"
        table.cells[text] = legacy_cells(row, where)
        if dedicated and has_comm_configs(row):
            raise ValueError(
                f"{where}: comm_fused_configs in an FHMoE table, which legacy does not "
                "read"
            )
        held += has_comm_configs(row)
        entries = legacy_comm_entries({**dict(zip(columns, key)), **row}, where)
        for tp, comm, add_shared, gpu_model, runner, _ in entries:
            collective = (str(tp), comm, str(add_shared), gpu_model)
            table.comm[text + collective] = CommCells(runner, table.cells[text])
    if held != _rows_with_comm_configs(name):
        raise ValueError(
            f"{name}: comm_fused_configs on rows the reader drops (a flydsl_fallback "
            "row, or the second of two rows of a key), which legacy's winner table reads"
        )
    for key, row in fallback.items():
        text = legacy_key(columns, key)
        table.fallback[text] = legacy_cells(row, f"{name}: the row of {key}")
    return table


def _combine(tables, names) -> Table:
    """One table of `tables` (named `names`, for the error); a key in two raises; the
    fallback takes the tables' in order."""
    combined, owner = Table(), {}
    for table, name in zip(tables, names):
        both = sorted(k for k in table.all_keys() if k in owner)
        if both:
            raise ValueError(
                f"keys in both {owner[both[0]]} and {name} ({len(both)}), e.g. "
                f"{dict(zip(tuned_plans.KEY_COLUMNS, both[0]))}"
            )
        owner.update(dict.fromkeys(table.all_keys(), name))
        combined.cells.update(table.cells)
        for key, by_path in table.tuner.items():
            combined.tuner[key] = dict(by_path)
        combined.comm.update(table.comm)
        for key, row_cells in table.fallback.items():
            combined.fallback.setdefault(key, row_cells)
    return combined


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@functools.cache
def _shipped() -> tuple:
    """`(Table, sha256 of the converted files)`; raises when they are stale."""
    with open(CONVERTED / SOURCES, newline="") as f:
        listed = [(r["source"], r["sha256"]) for r in csv.DictReader(f)]
    main, fhmoe = shipped_sources()
    legacy = sorted(p.relative_to(CONFIGS).as_posix() for p in (*main, fhmoe))
    unconverted = [p.relative_to(CONFIGS).as_posix() for p in _model_fhmoe_tables()]
    if unconverted:
        raise ValueError(
            f"legacy merges {', '.join(unconverted)} into its FHMoE table; v2 does not "
            f"convert model_configs FHMoE tables: name them in "
            f"{envs.AITER_MOEV2_TUNED_TABLE} instead"
        )
    stale = []
    if sorted(s for s, _ in listed) != legacy:
        listed_names = {s for s, _ in listed}
        stale += [f"{s} (not converted)" for s in legacy if s not in listed_names]
        stale += [f"{s} (gone)" for s in sorted(listed_names - set(legacy))]
    else:
        stale += [s for s, sha in listed if _sha256(CONFIGS / s) != sha]
    if stale:
        raise ValueError(
            f"the converted tuned tables in {CONVERTED} do not match the legacy "
            f"tables they come from: {', '.join(stale)}; rerun python "
            "tests/moe_dispatch/convert_tables.py --update"
        )
    digest, tables = hashlib.sha256(), []
    for source, _ in listed:
        data = (CONVERTED / source).read_bytes()
        digest.update(data)
        tables.append(_plan_table(data.decode(), str(CONVERTED / source)))
    return _combine(tables, [s for s, _ in listed]), digest.hexdigest()


@cached
def _user_file(name) -> tuple:
    """`(Table, sha256)` of a file `AITER_MOEV2_TUNED_TABLE` names, as read once."""
    data = pathlib.Path(name).read_bytes()
    text = data.decode()
    header = next(csv.reader(io.StringIO(text, newline="")), [])
    if tuned_plans.is_plan_file(header):
        table = _plan_table(text, name)
    elif "kernelName1" in header:
        table = _legacy_table(name, header)
    else:
        raise ValueError(
            f"{name} is neither a v2 plan file (columns structure, candidates) nor a "
            "legacy tuned table (column kernelName1)"
        )
    return table, hashlib.sha256(data).hexdigest()


def _check_variables(files):
    legacy = envs.legacy_tuned_tables()
    if legacy and files:
        raise ValueError(
            f"{', '.join(legacy)} and {envs.AITER_MOEV2_TUNED_TABLE} are both set; "
            f"v2 reads only {envs.AITER_MOEV2_TUNED_TABLE}: unset {', '.join(legacy)}"
        )
    if legacy:
        raise ValueError(
            f"{', '.join(legacy)} is set, which v2 does not read (the legacy entry "
            "points take their calls back while it is set); give v2 tables with "
            f"{envs.AITER_MOEV2_TUNED_TABLE} and {envs.AITER_MOEV2_TUNED_TABLE_MODE}"
        )


@cached
def _active() -> tuple:
    """The tables a key is looked up in, in order."""
    files, mode = envs.tuned_table()
    _check_variables(files)
    if not files:
        return (_shipped()[0],)
    user = _combine([_user_file(f)[0] for f in files], files)
    if mode == "replace":
        return (user,)
    if mode == "override":
        return (user, _shipped()[0])
    return (_combine([_shipped()[0], user], ["the shipped tables", "the user's"]),)


def _cells(table, problem, key):
    """The legacy `_lookup_cfg` over `table` with the local key `key` of `problem`."""
    cells = table.cells.get(key)
    if problem.extras.get("tuned_table") is not None:
        return cells
    if cells is None:
        cells = table.fallback.get(_disabled(key))
    token = problem.shape.m
    if cells is None and token > _PADDED_M_TIERS[0]:
        tier = _PADDED_M_TIERS.index(token) if token in _PADDED_M_TIERS else -1
        for smaller in reversed(_PADDED_M_TIERS[:tier]):
            at = _at_token(key, smaller)
            cells = table.cells.get(at) or table.fallback.get(_disabled(at))
            if cells is not None:
                break
    return cells


def lookup(problem) -> Found:
    """`problem`'s rows in the first table that has one. The tuner's rows are not
    looked up for a problem with a collective or a tail (the tuner tunes neither) and
    leave out comm_fused rows. Raises when a table cannot be read."""
    key = tuned_plans.key_values(problem)
    for table in _active():
        tuner = ()
        if problem.parallel is None and problem.tail is None:
            by_path = table.tuner.get(key, {})
            tuner = tuple(
                v for p, v in by_path.items() if p != tuned_plans.COMM_FUSED_PATH
            )
        cells = _cells(table, problem, key[:_LOCAL])
        if tuner or cells is not None:
            return Found(tuner, None if cells is None else dict(cells))
    return Found()


def comm_lookup(local, tp, comm, add_shared, gpu_model):
    """The TP call's comm_fused row in the first table that has one, or None: a tuner
    row's `(structure names, {segment name: Params})` (its runner under
    `tuned_plans.RUNNER`; none: the ordinary call won), or a legacy row's
    `CommCells`."""
    key = tuned_plans.comm_key_values(local, tp, comm, add_shared, gpu_model)
    for table in _active():
        row = table.tuner.get(key, {}).get(tuned_plans.COMM_FUSED_PATH)
        if row is None:
            row = table.comm.get(key)
        if row is not None:
            return row
    return None


def comm_tokens(local, tp, comm, add_shared, gpu_model) -> tuple:
    """The token counts of the comm_fused rows of `local`'s key at any token count and
    activation dtype (which follows the count; the rest of the key as `comm_lookup`'s)
    in any table, ascending: each still to be looked up at its count."""
    token = tuned_plans.KEY_COLUMNS.index("token")
    target = tuned_plans.any_token(
        tuned_plans.comm_key_values(local, tp, comm, add_shared, gpu_model)
    )
    return tuple(
        sorted(
            {
                int(k[token])
                for table in _active()
                for k in [
                    *(
                        k
                        for k, p in table.tuner.items()
                        if tuned_plans.COMM_FUSED_PATH in p
                    ),
                    *table.comm,
                ]
                if tuned_plans.any_token(k) == target
            }
        )
    )


@cached
def fhmoe_keys() -> tuple:
    """The FHMoE keys (dicts over `tuned_plans.LOCAL_KEY_COLUMNS`) of the tables."""
    keys = {
        k[:_LOCAL]: None
        for table in _active()
        for k in [*table.cells, *table.tuner]
        if any(k[_MAIN:_LOCAL])
    }
    return tuple(dict(zip(tuned_plans.LOCAL_KEY_COLUMNS, k)) for k in keys)


@cached
def digest() -> str:
    """The sha256 of the mode and the tables' contents as this process read them."""
    files, mode = envs.tuned_table()
    _check_variables(files)
    h = hashlib.sha256(f"{mode}\n".encode())
    if mode != "replace":
        h.update(_shipped()[1].encode())
    for name in files:
        h.update(f"{_user_file(name)[1]}\n".encode())
    return h.hexdigest()
