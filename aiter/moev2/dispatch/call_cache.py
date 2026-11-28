# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""What v2 decides for a call, cached per call signature: the path `route_to_v2` picks
and, for a local path, the dispatch decision of `LocalDispatch.run`.

Deciding afresh runs each active path's `why_not` and builds the dispatch key twice, on
every call, while a model repeats a few signatures on every layer and step. The decision
depends only on `signature(call, entry)`, which projects each parameter of the entry
function by its class in `PARAMS`:

- VALUE: by value (Python scalars, enums, dtypes);
- PRESENT: whether it is None;
- WEIGHT: dtype, shape and the `is_shuffled` / `gate_up_interleave` tags;
- ROWS: dtype, the shape after the first dimension, and the first dimension (the token
  count) bucketed with `get_padded_M`;
- EXEC: not at all; only the execution reads it, and the execution runs on every call;

and adds what the decision reads besides the parameters: the activation dtype (which
depends on the raw token count; memoized on its inputs and `ACTIVATION_ENV`), the arch
and CU count, AITER_ONLINE_TUNE, AITER_MOE_EXPLAIN, a registered presort and legacy's
grouped gfx1250 branch. What the dispatch caches read on a miss (tuned tables,
`overrides`) they already freeze until `tuned.clear_cache()`, which clears these caches
too (through `paths.reload()`).

AITER_MOEV2_CALL_CACHE, read on every call: unset or "1" uses the cache, "0" decides
every call afresh, "check" also decides every cache hit afresh and raises
`CallCacheMismatch` where the two differ (the tests and the GPU harnesses set it).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import dataclasses
import functools
import importlib
import inspect
import logging
import operator
import os
import threading
from operator import itemgetter

from aiter import presorted_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from ..policy import resolve_activation_dtype
from ..shapes import get_padded_M
from . import overrides

logger = logging.getLogger("aiter")

AITER_MOEV2_CALL_CACHE = "AITER_MOEV2_CALL_CACHE"
OFF, ON, CHECK = "0", "1", "check"

VALUE, PRESENT, WEIGHT, ROWS, EXEC = "value", "present", "weight", "rows", "exec"
KINDS = (VALUE, PRESENT, WEIGHT, ROWS, EXEC)

# Every parameter of the entry functions (`ENTRY_FUNCTIONS`), by how the signature keys it.
PARAMS = {
    "hidden_states": ROWS,
    "topk_ids": ROWS,
    "w1": WEIGHT,
    "w2": WEIGHT,
    "activation": VALUE,
    "quant_type": VALUE,
    "quant_type_a": VALUE,
    "quant_dtype_a": VALUE,
    "quant_dtype_a2": VALUE,
    "doweight_stage1": VALUE,
    "block_size_M": VALUE,
    "dtype": VALUE,
    "hidden_pad": VALUE,
    "intermediate_pad": VALUE,
    "swiglu_limit": VALUE,
    "beta": VALUE,
    "linear_beta": VALUE,
    "gate_mode": VALUE,
    "herd_topk": VALUE,
    "deterministic": VALUE,
    "expert_mask": PRESENT,
    "w1_scale": PRESENT,
    "w2_scale": PRESENT,
    "a1_scale": PRESENT,
    "a2_scale": PRESENT,
    "num_local_tokens": PRESENT,
    "bias1": PRESENT,
    "bias2": PRESENT,
    "w1_global_scale": PRESENT,
    "w2_global_scale": PRESENT,
    "ep_source_token_map": PRESENT,
    "stage2_scatter": PRESENT,
    "shared_w1": PRESENT,
    "shared_w2": PRESENT,
    "shared_w1_scale": PRESENT,
    "shared_w2_scale": PRESENT,
    "topk_weight": EXEC,
    "output": EXEC,
    "moe_sorting_dispatch_policy": EXEC,
    "splitk": EXEC,
    "herd_routed_scaling_factor": EXEC,
    "shared_expert_id": EXEC,
    "clamp_shared": EXEC,
    "ep_arena_handle": EXEC,
    "ep_combine_input_offset": EXEC,
    "ep_slot_stride_bytes": EXEC,
    "ep_max_tokens_per_rank": EXEC,
    "ep_world_size": EXEC,
    "ep_combine_quant": EXEC,
}

# `paths.ENTRIES`: the function whose parameters a call at that entry maps.
ENTRY_FUNCTIONS = {
    "api": ("aiter.fused_moe", "fused_moe"),
    "op": ("aiter.fused_moe", "fused_moe_"),
    "fhmoe": ("aiter.fhmoe", "fhmoe_"),
}

# Where `route_to_v2` leaves `(mode, signature, Entry)` in the call for
# `LocalDispatch.run`; the signature and the entry are None where the call is not cached.
SIGNATURE = "_moev2_call_signature"


class CallCacheMismatch(RuntimeError):
    """AITER_MOEV2_CALL_CACHE=check: a cached decision differs from a fresh one."""


class _Miss:
    def __repr__(self):
        return "MISS"


MISS = _Miss()


def mode() -> str:
    raw = os.environ.get(AITER_MOEV2_CALL_CACHE, ON).strip().lower()
    if raw in (OFF, "off", "false"):
        return OFF
    return CHECK if raw == CHECK else ON


def entry_params(entry) -> tuple:
    module, name = ENTRY_FUNCTIONS[entry]
    function = getattr(importlib.import_module(module), name)
    return tuple(inspect.signature(function).parameters)


def _getter(names):
    if not names:
        return lambda call: ()
    if len(names) == 1:
        name = names[0]
        return lambda call: (call[name],)
    return itemgetter(*names)


def _weight(w):
    return (
        w.dtype,
        tuple(w.shape),
        getattr(w, "is_shuffled", False),
        getattr(w, "gate_up_interleave", None),
    )


def _rows(t):
    shape = t.shape
    return t.dtype, tuple(shape[1:]), get_padded_M(shape[0])


class _Projection:
    def __init__(self, entry, names):
        unclassified = [n for n in names if n not in PARAMS]
        if unclassified:
            raise ValueError(
                f"entry {entry!r}: parameters {unclassified} are not in "
                "call_cache.PARAMS"
            )
        self.entry = entry
        self.names = {
            kind: tuple(n for n in names if PARAMS[n] == kind) for kind in KINDS
        }
        self.values = _getter(self.names[VALUE])
        self.present = _getter(self.names[PRESENT])
        self.weights = _getter(self.names[WEIGHT])
        self.rows = _getter(self.names[ROWS])

    def __call__(self, call):
        return (
            self.values(call),
            tuple(map(_is_none, self.present(call))),
            tuple(map(_weight, self.weights(call))),
            tuple(map(_rows, self.rows(call))),
        )


_is_none = functools.partial(operator.is_, None)


_projections: dict = {}
_lock = threading.Lock()


def _projection(entry):
    try:
        return _projections[entry]
    except KeyError:
        pass
    try:
        projection = _Projection(entry, entry_params(entry))
    except ValueError as error:
        logger.warning("moev2 call cache off at entry %r: %s", entry, error)
        projection = None
    with _lock:
        return _projections.setdefault(entry, projection)


# The variables `resolve_activation_dtype` reads.
ACTIVATION_ENV = (
    "AITER_BF16_FP8_MOE_BOUND",
    "AITER_SITUV2_A8W4",
    "AITER_SITUV2_A4W4",
    "AITER_FORCE_A8W4",
)


@functools.lru_cache(maxsize=4096)
def _resolve_activation_dtype(
    env,
    quant_type,
    q_dtype_w,
    activation,
    gate_mode,
    M,
    hidden_dtype,
    has_a1_scale,
    gfx,
    interleave,
):
    return resolve_activation_dtype(
        quant_type,
        q_dtype_w,
        activation=activation,
        gate_mode=gate_mode,
        M=M,
        hidden_dtype=hidden_dtype,
        has_a1_scale=has_a1_scale,
        gfx=gfx,
        gate_up_interleave=interleave,
    )


def _activation_dtype(call):
    w1 = call["w1"]
    return _resolve_activation_dtype(
        tuple(map(os.environ.get, ACTIVATION_ENV)),
        call["quant_type"],
        w1.dtype,
        call["activation"],
        call["gate_mode"],
        call["topk_ids"].shape[0],
        call["hidden_states"].dtype,
        call["a1_scale"] is not None,
        get_gfx_runtime(),
        getattr(w1, "gate_up_interleave", None),
    )


@functools.cache
def _local():
    from ..migration import local

    return local


STATE = (
    "arch",
    "cu_num",
    "online_tune",
    "explain",
    "presort_registered",
    "legacy_grouped_gfx1250",
)


def _state(call):
    return (
        get_gfx_runtime(),
        get_cu_num(),
        overrides.online_tune(),
        overrides.explain(),
        getattr(presorted_moe._PRESORTED_MOE, "aux", None) is not None,
        _local().legacy_grouped_gfx1250(call),
    )


def signature(call, entry):
    """The key of everything v2's decision for `call` at `entry` reads, or None where
    the call cannot be keyed (decide it afresh)."""
    projection = _projection(entry)
    if projection is None:
        return None
    try:
        key = (entry, projection(call), _activation_dtype(call), _state(call))
        hash(key)
    except (KeyError, TypeError, AttributeError, ValueError, RuntimeError):
        return None
    return key


def describe(key) -> dict:
    """`signature`'s key by field name, for error messages."""
    entry, (values, present, weights, rows), act_dtype, state = key
    names = _projections[entry].names
    fields = {"entry": entry}
    fields.update(zip(names[VALUE], values))
    fields.update((f"{n} is None", v) for n, v in zip(names[PRESENT], present))
    fields.update(zip(names[WEIGHT], weights))
    fields.update(zip(names[ROWS], rows))
    fields["activation dtype"] = act_dtype
    fields.update(zip(STATE, state))
    return fields


class Entry:
    """What is cached for one signature: the path `route_to_v2` picks, and the dispatch
    decisions of `LocalDispatch.run`, by dispatcher and forced arguments."""

    __slots__ = ("decisions", "path")

    def __init__(self, path):
        self.path = path
        self.decisions = {}


class Store:
    """`Entry`s by signature; `clear()` empties every store. Bounded: full, it starts
    over."""

    def __init__(self, maxsize=4096):
        self.maxsize = maxsize
        self._data = {}
        _STORES.append(self)

    def get(self, key):
        if key is None:
            return MISS
        return self._data.get(key, MISS)

    def put(self, key, value):
        if key is None:
            return
        if len(self._data) >= self.maxsize:
            self._data.clear()
        self._data[key] = value

    def clear(self):
        self._data.clear()

    def values(self):
        return list(self._data.values())


_STORES: list = []


def clear():
    for store in _STORES:
        store.clear()
    _resolve_activation_dtype.cache_clear()


def _differences(cached, fresh):
    if (
        dataclasses.is_dataclass(cached)
        and type(cached) is type(fresh)
        and not isinstance(cached, type)
    ):
        return {
            f.name: (getattr(cached, f.name), getattr(fresh, f.name))
            for f in dataclasses.fields(cached)
            if getattr(cached, f.name) != getattr(fresh, f.name)
        }
    return {"decision": (cached, fresh)}


def check(key, what, cached, fresh):
    """AITER_MOEV2_CALL_CACHE=check: raise unless `fresh` equals `cached`."""
    if cached is fresh or cached == fresh:
        return
    differences = "; ".join(
        f"{name}: cached {c!r}, fresh {f!r}"
        for name, (c, f) in _differences(cached, fresh).items()
    )
    signature_key, *forced = key if what == "dispatch" else (key,)
    raise CallCacheMismatch(
        f"moev2 call cache: the cached {what} decision differs from a fresh one "
        f"({differences}) for the signature {describe(signature_key)}"
        + (f" with {forced[0]}" if forced else "")
        + ". Something the decision reads is missing from call_cache.signature."
    )
