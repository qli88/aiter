# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The v2 tuner's plan file (`python -m aiter.moev2.tuning`), read only by v2.

One CSV row per (key, path) the tuner found a winner for, and nothing else (the other
structures, their times and why they failed are in the tuner's log file): the key in
the tuned tables' index columns (the FHMoE columns empty for other keys), the path, the
structure (its segment names), each tunable segment's candidate (JSON), `e2e_us` and
`mode`.

`resolver.resolve` builds a key's rows (one per path tuned; `resolver.build`) before
reading the tuned tables, and only while `AITER_MOEV2_TUNED_PLANS` names the file.
Lookup is by exact key: no activation-agnostic or smaller-tier fallback. A file with
columns this one does not have (the tuner's log file) is not read.
"""

import csv
import json

from .. import envs
from ..segments.base import Params
from .tuned import DEDICATED_INDEX_COLUMNS, INDEX_COLUMNS, cached

KEY_COLUMNS = INDEX_COLUMNS + DEDICATED_INDEX_COLUMNS
COLUMNS = KEY_COLUMNS + ("path", "structure", "candidates", "e2e_us", "mode")
STRUCTURE_SEPARATOR = " | "


def key_values(problem) -> tuple:
    """The key columns' values of `problem`, as written."""
    keys = tuple(str(v) for v in problem.log_keys())
    return keys + ("",) * (len(KEY_COLUMNS) - len(keys))


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
            key = tuple(row[c] for c in KEY_COLUMNS)
            if row["path"] in winners.setdefault(key, {}):
                raise ValueError(f"{path}: two rows for {key} on {row['path']}")
            winners[key][row["path"]] = (
                structure_names(row["structure"]),
                decode_candidates(row["candidates"]),
            )
    return {key: tuple(by_path.values()) for key, by_path in winners.items()}


def lookup(problem) -> tuple:
    """`(structure names, {segment name: Params})` of each of `problem`'s winners in
    the file `AITER_MOEV2_TUNED_PLANS` names (one per path tuned), in file order; empty
    with no file or row, or for a problem with a collective or a tail (the tuner tunes
    neither). Raises when the file cannot be read."""
    path = envs.tuned_plans_file()
    if not path or problem.parallel is not None or problem.tail is not None:
        return ()
    return _winners(path).get(key_values(problem), ())
