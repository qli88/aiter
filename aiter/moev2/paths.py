# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Migration paths: the unit that moves from the legacy MoE code to aiter.moev2.

A path claims a subset of `fused_moe` calls through a predicate over the call arguments.
The legacy entry points ask `route_to_v2(call, entry)` which path, if any, takes a call;
with no path active this costs one cached boolean check. A claimed call may still be
handed back: `run` returns `DECLINED` when the resolver finds no plan it may run (a tuned
row naming a kernel family the path has not ported, a ranking region marked legacy, no
feasible listed structure), and the legacy code then runs the call as before. While
legacy's own tuned-table variables are set, `route_to_v2` hands every call back.
The entry points run a routed call with `run(path, call)`, which counts its outcome
(`outcomes`; with `AITER_MOEV2_TRACE` set, also a trace file), so a run can show that
v2 served its calls rather than handing them back.

Predicates run on every call, including during graph capture: they may read dtypes,
shapes and Python values, never tensor contents.
"""

import collections
import importlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from . import envs

logger = logging.getLogger("aiter")

# Path names agreed for the migration. Only these are accepted in
# AITER_MOEV2, so a typo fails loudly instead of silently running the legacy code.
PLANNED = (
    "a16w16",
    "a8w8_pt",
    "a8w8_ptok",
    "a8w8_blockscale",
    "a4w4",
    "a8w4",
    "a16w4",
    "a16wi4",
    "mxfp8",
    "fhmoe",
    "comm_fused",
    "grouped_gfx1250",
    "mega_moe_v2",
    "mega_moe_tp",
    "mega_moe_gfx1250",
)


class _Declined:
    """`Path.run` result: run this call on the legacy code instead."""

    def __repr__(self):
        return "DECLINED"


DECLINED = _Declined()

# Where the legacy code consults the paths: "api" at the top of `fused_moe` (before the
# FHMoE dispatch, with the `fused_moe` parameters; skipped while torch.compile traces),
# "op" inside the `fused_moe_` custom op (with its parameters: enums as ints, the scatter
# context flattened), so torch.compile records the same op whichever code runs it;
# "fhmoe" inside the `fhmoe_` custom op, the same way, for calls with a shared expert;
# "comm_fused" at the top of `CommFusedMoeRuntime.run`, with `runtime` (the instance),
# its keyword arguments by name and `moe_args` (the `_fused_moe_impl` arguments);
# "mega_gfx1250" at the top of `MegaMoEGfx1250.forward`, with `mega` (the instance) and
# its arguments by name; "mega_v2" at the top of `MegaMoEV2.forward` and
# `forward_prequant`, with `mega`, `prequant` (which of the two), `x` (`x_bf16` or
# `x_q`), `scales` (None for `forward`) and the other arguments by name; "mega_tp" at
# the top of `MegaMoeTP.forward`, with `mtp` (the instance), `x` (`x_local`) and the
# other arguments by name.
ENTRIES = ("api", "op", "fhmoe", "comm_fused", "mega_gfx1250", "mega_v2", "mega_tp")


@dataclass(frozen=True)
class Path:
    """One migration path.

    `claims(call)` / `run(call)`: `call` maps the parameter names of the path's `entry`
    function to the call's arguments. `run` returns the output, or `DECLINED`.
    `graduated`: True, or the archs (`get_gfx_runtime()` values) the path runs on by
    default; the default needs a passed GPU gate on that arch.
    `graduated_claims(call)`: when the path runs by default (AITER_MOEV2 unset), it
    claims only the calls this also accepts, the formats its gate covered; None for all.
    A format added to a graduated path stays behind AITER_MOEV2 until its own gate.
    `dispatch_record(args, kwargs)`: for a legacy `get_2stage_cfgs(*args, **kwargs)` key,
    the v2 plan in the legacy snapshot record format, or None if the path does not take
    the key (not claimed, or declined). Used by the CPU differential test; the GPU cases
    of each path are in tests/moe_dispatch/gpu_differential.py.
    `segments`: the segments the path resolves plans from. `takes(problem)`: the
    `MoeProblem`s the path runs (its segments may accept more). The ranking checks use
    both.
    `resolve(call)`: the `Resolution` `run(call)` would dispatch on, without running
    it (shapes, dtypes and weight tags only, so meta tensors do), or None if the path
    declines the key. `MoeEngine.prepare` resolves its buckets with it; None for paths
    the engine does not run.
    """

    name: str
    claims: Callable[[Mapping], bool]
    run: Callable[[Mapping], object]
    graduated: bool | frozenset = False
    graduated_claims: Callable[[Mapping], bool] | None = None
    dispatch_record: Callable[[tuple, dict], object] | None = None
    entry: str = "api"
    segments: tuple = ()
    takes: Callable[[object], bool] | None = None
    resolve: Callable[[Mapping], object] | None = None

    def __post_init__(self):
        if self.entry not in ENTRIES:
            raise ValueError(f"unknown entry {self.entry!r}; known: {ENTRIES}")

    def graduated_here(self) -> bool:
        if isinstance(self.graduated, bool):
            return self.graduated
        try:
            return get_gfx_runtime() in self.graduated
        except (RuntimeError, KeyError):
            return False


# Modules that register the built-in paths; imported on first use, not at import time.
_BUILTIN_MODULES: tuple = (
    "aiter.moev2.migration.a8w8_pt",
    "aiter.moev2.migration.a16w16",
    "aiter.moev2.migration.a8w8_ptok",
    "aiter.moev2.migration.a8w8_blockscale",
    "aiter.moev2.migration.a4w4",
    "aiter.moev2.migration.a16w4",
    "aiter.moev2.migration.a16wi4",
    "aiter.moev2.migration.a8w4",
    "aiter.moev2.migration.mxfp8",
    "aiter.moev2.migration.fhmoe",
    "aiter.moev2.migration.comm_fused",
    "aiter.moev2.migration.grouped_gfx1250",
    "aiter.moev2.migration.mega_moe_gfx1250",
    "aiter.moev2.migration.mega_moe_v2",
    "aiter.moev2.migration.mega_moe_tp",
)

_REGISTRY: dict[str, Path] = {}
_builtin_loaded = False


def _load_builtin():
    global _builtin_loaded
    if not _builtin_loaded:
        _builtin_loaded = True
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)


def register(path: Path):
    _load_builtin()
    if path.name not in PLANNED:
        raise ValueError(f"unknown moev2 path {path.name!r}; add it to PLANNED first")
    if path.name in _REGISTRY:
        raise ValueError(f"moev2 path {path.name!r} is already registered")
    _REGISTRY[path.name] = path
    reload()


def unregister(name: str):
    _REGISTRY.pop(name, None)
    reload()


def registered():
    _load_builtin()
    return dict(_REGISTRY)


_active: tuple | None = None


def active() -> tuple:
    """Registered paths selected by AITER_MOEV2, in registration order."""
    global _active
    if _active is None:
        _active = _select()
    return _active


def _select() -> tuple:
    selection = envs.moev2_selection()
    if selection == "off":
        return ()
    _load_builtin()
    if selection == "on":
        return tuple(_REGISTRY.values())
    if selection == "default":
        return tuple(p for p in _REGISTRY.values() if p.graduated_here())
    unknown = sorted(selection - set(PLANNED))
    if unknown:
        raise ValueError(
            f"{envs.AITER_MOEV2} names unknown paths {unknown}; known: {', '.join(PLANNED)}"
        )
    missing = sorted(selection - set(_REGISTRY))
    if missing:
        logger.warning(
            "%s: paths %s are not implemented yet", envs.AITER_MOEV2, missing
        )
    return tuple(p for p in _REGISTRY.values() if p.name in selection)


def any_active() -> bool:
    return bool(active())


def route_to_v2(call: Mapping, entry: str = "api"):
    """The active path at `entry` that claims this call, or None for the legacy code.

    While legacy's tuned-table variables are set (`envs.legacy_tuned_tables`), which v2
    does not read, a claimed call goes to the legacy code (warned once); with
    `AITER_MOEV2_TUNED_TABLE` set too, raises."""
    by_default = envs.moev2_selection() == "default"
    used = False
    for path in active():
        if path.entry != entry:
            continue
        used = True
        if not path.claims(call):
            continue
        gate = path.graduated_claims
        if by_default and gate is not None and not gate(call):
            continue
        routed = _unless_legacy_tables(path)
        if routed is None:
            _record(entry, path.name, "legacy_tables", call)
        return routed
    if used:
        _record(entry, None, "unclaimed", call)
    return None


def run(path: Path, call: Mapping, entry: str | None = None):
    """`path.run(call)` for a path `route_to_v2` returned, its outcome counted under
    `entry` (the path's own by default)."""
    out = path.run(call)
    outcome = "declined" if out is DECLINED else "served"
    _record(entry or path.entry, path.name, outcome, call)
    return out


_outcomes: collections.Counter = collections.Counter()
_traced: set = set()
_trace_failed = False


def outcomes() -> dict:
    """`{(entry, path name, outcome): calls}` since `reset_outcomes`: the calls that
    reached a legacy entry point some active path uses, and every call of
    `api.fused_moe` (entry `api`, whatever AITER_MOEV2 says). Outcomes: `served` and
    `declined` (`run`; from a legacy entry point a declined call runs on the legacy
    code, from `api.fused_moe` the next path is tried), `unclaimed` (no active path
    took it: none claims it, or under the default selection its graduation gate
    refuses it; path name None), `legacy_tables` (claimed, handed back by
    `route_to_v2`). Calls replayed from a captured graph do not reach an entry and are
    not counted."""
    return dict(_outcomes)


def reset_outcomes():
    global _trace_failed
    _outcomes.clear()
    _traced.clear()
    _trace_failed = False


def _record(entry, name, outcome, call):
    _outcomes[entry, name, outcome] += 1
    file = envs.trace_file()
    if file and not _trace_failed:
        _trace(file, entry, name, outcome, call)


def _trace(file, entry, name, outcome, call):
    """Appends `{entry, path, outcome, call, pid}` the first time this process sees
    the outcome for a call of these argument shapes, dtypes and values. A file it
    cannot write is warned about once and no longer traced: the call goes on."""
    global _trace_failed
    record = {"entry": entry, "path": name, "outcome": outcome}
    record["call"] = {k: _describe(v) for k, v in call.items()}
    key = json.dumps(record, sort_keys=True)
    if key in _traced:
        return
    _traced.add(key)
    record["pid"] = os.getpid()
    line = (json.dumps(record, sort_keys=True) + "\n").encode()
    try:
        # One O_APPEND write per line, so lines of several processes do not interleave.
        fd = os.open(file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError as error:
        _trace_failed = True
        logger.warning("%s=%s: not traced: %s", envs.AITER_MOEV2_TRACE, file, error)


def _describe(value, nested=False):
    """`value` as JSON without reading tensor contents."""
    if isinstance(value, torch.Tensor):
        return f"{str(value.dtype).removeprefix('torch.')}{list(value.shape)}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)) or hasattr(
        type(value), "__members__"
    ):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_describe(v, True) for v in value]
    if isinstance(value, Mapping) and not nested:
        return {str(k): _describe(v, True) for k, v in value.items()}
    return type(value).__name__


_warned_legacy_tables = False


def _unless_legacy_tables(path):
    global _warned_legacy_tables
    legacy = envs.legacy_tuned_tables()
    if not legacy:
        return path
    if envs.tuned_table()[0]:
        raise ValueError(
            f"{', '.join(legacy)} and {envs.AITER_MOEV2_TUNED_TABLE} are both set; "
            f"v2 reads only {envs.AITER_MOEV2_TUNED_TABLE} and legacy only "
            f"{', '.join(legacy)}: unset one"
        )
    if not _warned_legacy_tables:
        _warned_legacy_tables = True
        logger.warning(
            "[fused_moe] %s set: the calls aiter.moev2 takes run on the legacy code, "
            "which reads it (v2 reads its tuned tables from %s)",
            ", ".join(legacy),
            envs.AITER_MOEV2_TUNED_TABLE,
        )
    return None


def reload():
    global _active, _warned_legacy_tables
    _active = None
    _warned_legacy_tables = False
