# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Environment variables read by aiter.moev2, one accessor each.

Values are read once and cached, because they are consulted on every MoE call (including
during graph capture). Tests that change them call `reload()`.
"""

import functools
import os

AITER_MOEV2 = "AITER_MOEV2"

_OFF = frozenset({"off", "0", "false", "none"})
_ON = frozenset({"on", "1", "true", "all"})


@functools.cache
def moev2_selection():
    """`AITER_MOEV2`: which migration paths run on aiter.moev2.

    Unset or empty: the graduated paths (the default).
    `off`: none; every call runs the legacy code.
    `on`: every registered path, graduated or not.
    `a4w4,fhmoe`: exactly the listed paths.

    Returns "default", "off", "on", or a frozenset of path names.
    """
    raw = os.environ.get(AITER_MOEV2, "").strip().lower()
    if not raw:
        return "default"
    if raw in _OFF:
        return "off"
    if raw in _ON:
        return "on"
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    return names or "default"


AITER_MOEV2_TUNED_TABLE = "AITER_MOEV2_TUNED_TABLE"
AITER_MOEV2_TUNED_TABLE_MODE = "AITER_MOEV2_TUNED_TABLE_MODE"
TUNED_TABLE_MODES = ("replace", "override", "extend")
# Legacy's own tuned-table variables; v2 does not read them (`legacy_tuned_tables`).
LEGACY_TUNED_TABLES = ("AITER_CONFIG_FMOE", "AITER_CONFIG_FHMOE")


@functools.cache
def tuned_table() -> tuple:
    """`(files, mode)`: `AITER_MOEV2_TUNED_TABLE`, the user's tuned tables v2 dispatch
    reads (`dispatch.tuned_table`; `os.pathsep`-separated, each a v2 plan file or a legacy
    tuned CSV), and `AITER_MOEV2_TUNED_TABLE_MODE`, how they combine with the shipped
    tables: `replace` (the shipped tables are not read), `override` (the shipped tables
    are read for a key only when the user's have no row for it, fallbacks included)
    or `extend` (one table; a key in both raises). `((), None)` with no file named.
    Raises for a mode not one of these, for files without a mode and for a mode
    without files."""
    raw = os.environ.get(AITER_MOEV2_TUNED_TABLE, "")
    files = tuple(f.strip() for f in raw.split(os.pathsep) if f.strip())
    mode = os.environ.get(AITER_MOEV2_TUNED_TABLE_MODE, "").strip().lower()
    if not files:
        if mode:
            raise ValueError(
                f"{AITER_MOEV2_TUNED_TABLE_MODE}={mode} is set without "
                f"{AITER_MOEV2_TUNED_TABLE}"
            )
        return (), None
    if mode not in TUNED_TABLE_MODES:
        raise ValueError(
            f"{AITER_MOEV2_TUNED_TABLE} is set; {AITER_MOEV2_TUNED_TABLE_MODE} must be "
            f"one of {', '.join(TUNED_TABLE_MODES)}, not {mode!r}"
        )
    return files, mode


@functools.cache
def legacy_tuned_tables() -> tuple:
    """The names in `LEGACY_TUNED_TABLES` that are set (non-empty)."""
    return tuple(n for n in LEGACY_TUNED_TABLES if os.environ.get(n, "").strip())


def reload():
    """Forget cached values so the next accessor call re-reads the environment."""
    moev2_selection.cache_clear()
    tuned_table.cache_clear()
    legacy_tuned_tables.cache_clear()
    from . import paths

    paths.reload()
