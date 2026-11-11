# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The v2 plan file's format, read only by v2 (`tuned_table`).

One CSV row per (key, path): the key in the tuned tables' index columns (the FHMoE
columns empty for other keys) and, for a TP call tuned with its collective (path
`comm_fused`), the collective's (`PARALLEL_COLUMNS`: TP size, "ar" or "rs", whether the
shared partial is added, the GPU model; empty for local keys), the path, the structure
(its segment names), each tunable segment's candidate (JSON; a comm_fused row's runner
under `comm_fused.stage2_rs`, none for the ordinary call), `e2e_us` and `mode`.

The v2 tuner (`python -m aiter.moev2.tuning`) writes one row per (key, path) it found
a winner for (the other structures, their times and why they failed are in its log
file). A row converted from a legacy tuned table (`convert`, `mode` "legacy") also
holds its times (`INFO_COLUMNS`), which are not read; `tuned_table` reads such a row as
the legacy row's cells (`resolver.candidate_row`), any other as a tuner row. A file
without the parallel columns (written before them) reads as local keys only; a file
with columns this one does not have (the tuner's log file) is not read.
"""

import csv
import io
import json

from ..segments.base import Params
from .tuned import DEDICATED_INDEX_COLUMNS, INDEX_COLUMNS

PARALLEL_COLUMNS = ("tp", "comm", "add_shared", "gpu_model")
LOCAL_KEY_COLUMNS = INDEX_COLUMNS + DEDICATED_INDEX_COLUMNS
KEY_COLUMNS = LOCAL_KEY_COLUMNS + PARALLEL_COLUMNS
COMM_FUSED_PATH = "comm_fused"
RUNNER = "comm_fused.stage2_rs"
COLUMNS = KEY_COLUMNS + ("path", "structure", "candidates", "e2e_us", "mode")
# A converted legacy row's times (`convert`), read by nothing.
INFO_COLUMNS = ("us1", "us2", "us")
STRUCTURE_SEPARATOR = " | "
LEGACY_MODE = "legacy"


def key_values(problem) -> tuple:
    """The key columns' values of `problem`, as written."""
    keys = tuple(str(v) for v in problem.log_keys())
    return keys + ("",) * (len(KEY_COLUMNS) - len(keys))


def comm_key_values(local, tp, comm, add_shared, gpu_model) -> tuple:
    """The key columns' values of a TP call: its local key `local` (a `MoeProblem`),
    then the collective's."""
    keys = key_values(local)[: len(LOCAL_KEY_COLUMNS)]
    return keys + (str(int(tp)), comm, str(bool(add_shared)), gpu_model)


def encode_candidates(candidates) -> str:
    """`{segment name: Params}` as JSON."""
    return json.dumps(
        {name: dict(params) for name, params in candidates.items()}, sort_keys=True
    )


def decode_candidates(text) -> dict:
    return {name: Params(values) for name, values in json.loads(text).items()}


def structure_names(text) -> tuple:
    return tuple(text.split(STRUCTURE_SEPARATOR))


def is_plan_file(fieldnames) -> bool:
    """Whether a CSV with header `fieldnames` is a plan file (else a legacy table)."""
    return {"structure", "candidates"} <= set(fieldnames or ())


def read_rows(text, name) -> list:
    """The rows (dicts) of plan file `name` with content `text`, its key columns
    filled ("" where blank or absent). Raises for columns a plan file does not have."""
    reader = csv.DictReader(io.StringIO(text, newline=""))
    other = [c for c in reader.fieldnames or () if c not in COLUMNS + INFO_COLUMNS]
    if other:
        raise ValueError(
            f"{name} has columns a v2 plan file does not ({', '.join(other)})"
        )
    rows = []
    for row in reader:
        rows.append(row | {c: row.get(c) or "" for c in KEY_COLUMNS})
    return rows


def any_token(key) -> tuple:
    """`key` (key values) without the columns that follow its token count: the token
    and the activation dtype."""
    free = {KEY_COLUMNS.index("token"), KEY_COLUMNS.index("q_dtype_a")}
    return tuple(v for i, v in enumerate(key) if i not in free)
