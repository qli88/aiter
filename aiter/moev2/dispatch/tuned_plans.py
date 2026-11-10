# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The v2 tuner's plan file (`python -m aiter.moev2.tuning`), read only by v2.

One CSV row per (key, path) the tuner found a winner for, and nothing else (the other
structures, their times and why they failed are in the tuner's log file): the key in
the tuned tables' index columns (the FHMoE columns empty for other keys) and, for a TP
call tuned with its collective (path `comm_fused`), the collective's
(`PARALLEL_COLUMNS`: TP size, "ar" or "rs", whether the shared partial is added, the
GPU model; empty for local keys), the path, the structure (its segment names), each
tunable segment's candidate (JSON; a comm_fused row's runner under
`comm_fused.stage2_rs`, none for the ordinary call), `e2e_us` and `mode`.

`resolver.resolve` builds a key's rows (one per path tuned; `resolver.build`) before
reading the tuned tables, and only while `AITER_MOEV2_TUNED_PLANS` names the file;
`comm_lookup` gives `CommFusedTemplate` a TP key's comm_fused row. Lookup is by exact
key: no activation-agnostic or smaller-tier fallback. A file without the parallel
columns (written before them) reads as local keys only; a file with columns this one
does not have (the tuner's log file) is not read.
"""

import csv
import json

from .. import envs
from ..segments.base import Params
from .tuned import DEDICATED_INDEX_COLUMNS, INDEX_COLUMNS, cached

PARALLEL_COLUMNS = ("tp", "comm", "add_shared", "gpu_model")
LOCAL_KEY_COLUMNS = INDEX_COLUMNS + DEDICATED_INDEX_COLUMNS
KEY_COLUMNS = LOCAL_KEY_COLUMNS + PARALLEL_COLUMNS
COMM_FUSED_PATH = "comm_fused"
RUNNER = "comm_fused.stage2_rs"
COLUMNS = KEY_COLUMNS + ("path", "structure", "candidates", "e2e_us", "mode")
STRUCTURE_SEPARATOR = " | "


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


@cached
def _winners(path):
    winners = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        other = [c for c in reader.fieldnames or () if c not in COLUMNS]
        if other:
            raise ValueError(
                f"{path} has columns a v2 plan file does not ({', '.join(other)})"
            )
        for row in reader:
            key = tuple(row.get(c) or "" for c in KEY_COLUMNS)
            if row["path"] in winners.setdefault(key, {}):
                raise ValueError(f"{path}: two rows for {key} on {row['path']}")
            winners[key][row["path"]] = (
                structure_names(row["structure"]),
                decode_candidates(row["candidates"]) if row["candidates"] else {},
            )
    return winners


def lookup(problem) -> tuple:
    """`(structure names, {segment name: Params})` of each of `problem`'s winners in
    the file `AITER_MOEV2_TUNED_PLANS` names (one per path tuned), in file order; empty
    with no file or row, or for a problem with a collective or a tail (the tuner tunes
    neither). Raises when the file cannot be read."""
    path = envs.tuned_plans_file()
    if not path or problem.parallel is not None or problem.tail is not None:
        return ()
    winners = _winners(path).get(key_values(problem), {})
    return tuple(v for p, v in winners.items() if p != COMM_FUSED_PATH)


def comm_lookup(local, tp, comm, add_shared, gpu_model):
    """`(structure names, {segment name: Params})` of the TP call's comm_fused
    row in the file `AITER_MOEV2_TUNED_PLANS` names (its runner under `RUNNER`; none:
    the ordinary call won), or None with no file or row."""
    path = envs.tuned_plans_file()
    if not path:
        return None
    key = comm_key_values(local, tp, comm, add_shared, gpu_model)
    return _winners(path).get(key, {}).get(COMM_FUSED_PATH)


def comm_tokens(local, tp, comm, add_shared, gpu_model) -> tuple:
    """The token counts of the comm_fused rows of `local`'s key at any token
    count and activation dtype (which follows the count; the rest of the key as
    `comm_lookup`'s), ascending: each still to be looked up at its count."""
    path = envs.tuned_plans_file()
    if not path:
        return ()
    token = KEY_COLUMNS.index("token")
    target = any_token(comm_key_values(local, tp, comm, add_shared, gpu_model))
    return tuple(
        sorted(
            {
                int(k[token])
                for k, by_path in _winners(path).items()
                if COMM_FUSED_PATH in by_path and any_token(k) == target
            }
        )
    )


def any_token(key) -> tuple:
    """`key` (key values) without the columns that follow its token count: the token
    and the activation dtype."""
    free = {KEY_COLUMNS.index("token"), KEY_COLUMNS.index("q_dtype_a")}
    return tuple(v for i, v in enumerate(key) if i not in free)
