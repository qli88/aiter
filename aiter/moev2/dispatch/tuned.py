# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The legacy tuned tables' columns and reader, and v2 dispatch's caches.

`read_table` is the CSV reader of the legacy `get_2stage_cfgs`; v2 reads a legacy table
with it only to convert it (`convert`, and `tuned_table` for a legacy table in
`AITER_MOEV2_TUNED_TABLE`). v2 dispatch looks its rows up in `tuned_table`.

Every cache in v2 dispatch that depends on tuned tables, environment variables or the
arch is declared with `@cached`, so `clear_cache()` resets all of them at once; tests call
it after changing AITER_MOEV2_TUNED_TABLE or patching the arch. The shipped converted
tables (`tuned_table._shipped`) are read once per process.
"""

import functools

from aiter import logger
from aiter.jit.utils.chip_info import gfx_from_cu_num
from aiter.ops.flydsl.mxfp4_kname import _is_mxfp4_kname

_CACHES: list = []


def cached(fn=None, *, maxsize=None):
    """`functools.lru_cache` that `clear_cache()` also clears."""

    def wrap(f):
        wrapped = functools.lru_cache(maxsize=maxsize)(f)
        _CACHES.append(wrapped)
        return wrapped

    return wrap(fn) if fn is not None else wrap


def clear_cache():
    """Clear every v2 dispatch cache and re-read the moev2 environment variables."""
    from .. import envs

    for cache in _CACHES:
        cache.cache_clear()
    envs.reload()


_ACT_TYPE_DISABLED_KEY = "__ignore__"

INDEX_COLUMNS = (
    "gfx",
    "cu_num",
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
# A dedicated table (`config_file=`, the FHMoE path) is also keyed on these.
DEDICATED_INDEX_COLUMNS = (
    "shared_expert_id",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
)


def index_columns(dedicated: bool) -> list:
    return list(INDEX_COLUMNS + (DEDICATED_INDEX_COLUMNS if dedicated else ()))


def ensure_gfx_column(df):
    """Guarantee a usable `gfx` column, migrating legacy cu_num-only CSVs."""
    if "gfx" not in df.columns:
        df = df.copy()
        df["gfx"] = df["cu_num"].map(gfx_from_cu_num)
        return df
    # Backfill placeholder/missing gfx (e.g. 0 filled when merging a config
    # that lacks the column against ones that have it).
    bad = df["gfx"].isna() | df["gfx"].astype(str).isin(["0", "", "nan", "None"])
    if bad.any():
        df = df.copy()
        df.loc[bad, "gfx"] = df.loc[bad, "cu_num"].map(gfx_from_cu_num)
    return df


def read_table(tune_file, _INDEX_COLS, dedicated):
    """Read one tuned CSV into `(primary, fallback)` dicts keyed on `_INDEX_COLS`.

    `fallback` ignores `act_type` and is empty for a dedicated table.
    """
    import pandas as pd

    df = pd.read_csv(tune_file)
    df = ensure_gfx_column(df)
    if "_tag" in df.columns:
        df = df[df["_tag"].fillna("") != "flydsl_fallback"]

    # Primary dict: keep original act_type for exact-match lookup.
    df_primary = df.copy()
    dup_mask = df_primary.duplicated(subset=_INDEX_COLS, keep="first")
    if dup_mask.any():
        if dedicated:
            raise ValueError(f"Duplicate dedicated FHMoE rows in {tune_file}")
        logger.warning(
            f"[fused_moe] duplicate tuned rows (primary) in {tune_file}; "
            f"keeping first match for {int(dup_mask.sum())} rows"
        )
        df_primary = df_primary.loc[~dup_mask]
    primary = df_primary.set_index(_INDEX_COLS).to_dict("index")

    if dedicated:
        return primary, {}

    # Fallback dict: disable act_type so any activation can match.
    df_fallback = df.copy()
    if "kernelName1" in df_fallback.columns:
        is_ck2stages = (
            df_fallback["kernelName1"]
            .fillna("")
            .astype(str)
            .str.contains("moe_ck2stages")
        )
        df_fallback = df_fallback.loc[~is_ck2stages]
    for column in ("kernelName1", "kernelName2"):
        if column in df_fallback.columns:
            is_mxfp4 = df_fallback[column].map(_is_mxfp4_kname)
            df_fallback = df_fallback.loc[~is_mxfp4]
    if "act_type" in df_fallback.columns:
        df_fallback["act_type"] = _ACT_TYPE_DISABLED_KEY
    dup_mask = df_fallback.duplicated(subset=_INDEX_COLS, keep="first")
    if dup_mask.any():
        logger.warning(
            f"[fused_moe] duplicate tuned rows after disabling act_type in {tune_file}; "
            f"keeping first match for {int(dup_mask.sum())} rows"
        )
        df_fallback = df_fallback.loc[~dup_mask]
    fallback = df_fallback.set_index(_INDEX_COLS).to_dict("index")

    return primary, fallback


def exact_key(problem):
    """The legacy-typed key of `problem`'s exact row."""
    keys = problem.log_keys()
    return keys[:7] + (str(problem.activation.activation),) + keys[8:]
