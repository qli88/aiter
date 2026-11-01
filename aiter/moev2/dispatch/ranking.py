# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Default ranking: the vocabulary of the table and its checks (§3.6 of the plan).

The table itself, and the rules it must follow, are in `ranking_table.py`.

An `Entry` covers one class of problems, given by `match(...)`: set membership on named
fields. Its `Region`s split the class by `when(...)`: integer thresholds and set
membership on named scalars, and named boolean predicates. `otherwise` is the part of the
class no region covers. Each region lists structures best first; the first feasible one
wins. A region can instead name what the legacy code runs there (`legacy=`), for a
branch not ported yet: its problems are declined back to the legacy code.

`check_table` checks the table statically; `check_problems` and `dead_lists` check it
over a set of problems (the dispatch snapshot key grid in CI).
"""

import functools
import itertools
from collections import Counter
from dataclasses import dataclass

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from ..problem import FrozenMap
from ..segments.base import Segment
from ..segments.ck_tile import (
    cktile_interleaved_swiglu,
    cktile_relu2,
    cktile_split_k,
    flydsl_mxfp4,
)
from ..segments.defaults import BLOCK_SIZE_M
from . import overrides
from .ops import Op

# Fields `match(...)` tests by set membership.
MATCH_FIELDS = {
    "arch": lambda p: p.arch,
    "quant_type": lambda p: p.quant_type,
    "act_dtype": lambda p: p.act_in.dtype,
    "w1_dtype": lambda p: p.w1.dtype,
    "w2_dtype": lambda p: p.w2.dtype,
    "out_dtype": lambda p: p.out_dtype,
    "activation": lambda p: p.activation.kind,
    "gate_mode": lambda p: p.layout.gate_mode,
    "shuffle": lambda p: p.layout.shuffle,
    "g1u1": lambda p: p.layout.g1u1,
    "shared": lambda p: None if p.shared is None else p.shared.placement,
}

# Integer scalars `when(...)` (and `match(...)`) can test.
SCALARS = {
    "m": lambda p: p.shape.m,
    "rows_per_expert": lambda p: p.rows_per_expert,
    "model_dim": lambda p: p.shape.model_dim,
    "inter_dim": lambda p: p.shape.inter_dim,
    "experts": lambda p: p.shape.experts,
    "topk": lambda p: p.shape.topk,
    "cu_num": lambda p: p.cu_num,
}

# fmt: off
# The keys of the legacy `fused_moe_1stage_dict` (aiter/fused_moe.py), whose problems the
# legacy default path runs on 1-stage ASM under a per-quant-type token condition.
# tests/test_moev2_ranking.py checks this copy against it. `_F8` stands for the arch's
# fp8, `dtypes.fp8` when looked up (legacy binds it at import; the same dtype in a GPU
# process, and the arch simulation of the tests swaps both).
_S, _G = ActivationType.Silu, ActivationType.Gelu
_V2 = ActivationType.Situv2
_NO, _TOK, _1X32, _1X128 = (
    QuantType.No, QuantType.per_Token, QuantType.per_1x32, QuantType.per_1x128
)
_F8 = "fp8"
_BF, _FP, _I8 = dtypes.bf16, dtypes.fp16, dtypes.i8
_F4, _I4 = dtypes.fp4x2, dtypes.i4x2
FMOE_1STAGE_KEYS = {
    # activation, quant_type, out dtype, q_dtype_a, q_dtype_w, g1u1, doweight_stage1
    "gfx942": frozenset({
        (_S, _NO,    _BF, _BF, _BF, False, False),
        (_S, _NO,    _FP, _FP, _FP, False, False),
        (_G, _TOK,   _BF, _F8, _I4, True,  False),
        (_S, _1X32,  _BF, _F4, _F4, True,  False),
        (_S, _TOK,   _BF, _I8, _I8, True,  False),
        (_G, _TOK,   _BF, _I8, _I8, True,  False),
        (_S, _TOK,   _BF, _F8, _F8, True,  False),
        (_G, _TOK,   _BF, _F8, _F8, True,  False),
        (_S, _1X128, _BF, _F8, _F8, True,  False),
        (_S, _TOK,   _BF, _I8, _I8, False, False),
        (_G, _TOK,   _BF, _I8, _I8, False, False),
    }),
    "gfx950": frozenset({
        (_S,  _1X32,  _BF, _F4, _F4, True,  False),
        (_V2, _1X32,  _BF, _BF, _F4, True,  False),
        (_S,  _1X128, _BF, _F8, _F8, True,  False),
        (_G,  _1X128, _BF, _F8, _F8, True,  False),
        (_S,  _TOK,   _BF, _BF, _BF, False, False),
        (_S,  _TOK,   _BF, _F8, _F8, True,  True),
        (_S,  _TOK,   _BF, _F8, _F8, True,  False),
        (_G,  _TOK,   _BF, _F8, _F8, True,  False),
    }),
}
# fmt: on


def _fmoe_1stage_key(p):
    return (
        p.activation.activation,
        p.quant_type,
        p.out_dtype,
        p.act_in.dtype,
        p.w1.dtype,
        p.layout.g1u1,
        "doweight_stage1" in p.features,
    )


@functools.cache
def _fmoe_1stage_keys(arch, fp8):
    return frozenset(
        tuple(fp8 if el is _F8 else el for el in key)
        for key in FMOE_1STAGE_KEYS.get(arch, ())
    )


def fmoe_1stage_keys(arch):
    """`FMOE_1STAGE_KEYS[arch]` with `_F8` read as `dtypes.fp8`."""
    return _fmoe_1stage_keys(arch, dtypes.fp8)


def _fmoe_1stage(p) -> bool:
    """The legacy `fused_moe_1stage_dict` has the problem's key for its arch."""
    return _fmoe_1stage_key(p) in fmoe_1stage_keys(p.arch)


def _inter_dim_128(p) -> bool:
    return p.shape.inter_dim % 128 == 0


def _blockscale_1stage_kernel(p) -> bool:
    """`fmoe_fp8_blockscale_g1u1` has a kernel for the problem (legacy's check)."""
    xbf16 = p.arch == "gfx950" and overrides.xbf16()
    s = p.shape
    has_kernel = aiter.fmoe_fp8_blockscale_g1u1_has_kernel(
        p.activation.activation.value, xbf16, BLOCK_SIZE_M, s.inter_dim, s.model_dim
    )
    return has_kernel == 1


def _mxmoe_fallback(p) -> bool:
    """The legacy `_mxmoe_fallback_ok`: the heuristic MXMOE kernels take MXFP4 SiTUv2."""
    s = p.shape
    return (
        p.out_dtype == dtypes.bf16
        and not p.is_ep
        and p.quant_type == QuantType.per_1x32
        and p.activation.activation == ActivationType.Situv2
        and p.act_in.dtype == dtypes.fp4x2
        and p.w1.dtype == dtypes.fp4x2
        and p.any_shuffled
        and p.layout.g1u1
        and "doweight_stage1" not in p.features
        and p.gate_mode != GateMode.INTERLEAVE
        and not {"bias1", "bias2"} & p.features
        and p.layout.hidden_pad == 0
        and p.layout.intermediate_pad == 0
        and s.model_dim % 256 == 0
        and s.inter_dim % 128 == 0
        and aiter.is_mxfp4_moe_shape_supported(
            s.experts, s.model_dim, s.inter_dim, s.topk
        )
        and overrides.mxmoe_fallback()
    )


def _padded(p) -> bool:
    """The hidden or intermediate dimension carries padding."""
    return bool(p.layout.hidden_pad or p.layout.intermediate_pad)


# Named boolean predicates `when(...)` can test (`when(fmoe_1stage=True)`).
PREDICATES = {
    "fmoe_1stage": _fmoe_1stage,
    "inter_dim_128": _inter_dim_128,
    "blockscale_1stage_kernel": _blockscale_1stage_kernel,
    "flydsl_mxfp4": flydsl_mxfp4,
    "mxmoe_fallback": _mxmoe_fallback,
    "cktile_split_k": cktile_split_k,
    "cktile_interleaved_swiglu": cktile_interleaved_swiglu,
    "cktile_relu2": cktile_relu2,
    "padded": _padded,
}

_GETTERS = {**MATCH_FIELDS, **SCALARS}


def _when_value(name, p) -> int:
    """A scalar's value, or a predicate's as 0/1."""
    if name in PREDICATES:
        return int(PREDICATES[name](p))
    return SCALARS[name](p)


_OPS = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "in": " in "}


def _as_set(name, value):
    values = (
        frozenset(value)
        if isinstance(value, (set, frozenset, list, tuple))
        else frozenset({value})
    )
    if not values:
        raise ValueError(f"empty value set for {name!r}")
    return values


def _show(value):
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, QuantType):
        return value.name
    return str(value)


def _show_set(values):
    shown = sorted(map(_show, values))
    return shown[0] if len(shown) == 1 else "{" + ", ".join(shown) + "}"


@dataclass(frozen=True)
class Match:
    """A problem class: every listed field takes one of its listed values."""

    sets: FrozenMap

    def holds(self, p) -> bool:
        return all(_GETTERS[name](p) in values for name, values in self.sets.items())

    def disjoint(self, other) -> bool:
        """True when some field both constrain has no common value."""
        return any(
            name in other.sets and not values & other.sets[name]
            for name, values in self.sets.items()
        )

    def __str__(self):
        return ", ".join(f"{k}={_show_set(v)}" for k, v in self.sets.items()) or "any"


def match(**fields) -> Match:
    """`match(quant_type=..., w1_dtype={...})`. Without `shared`, only problems without
    a shared expert match (`shared=None`)."""
    unknown = sorted(set(fields) - set(MATCH_FIELDS) - set(SCALARS))
    if unknown:
        raise ValueError(
            f"unknown match fields {unknown}; known: {sorted(MATCH_FIELDS)}"
        )
    fields.setdefault("shared", None)
    return Match(FrozenMap({k: _as_set(k, v) for k, v in fields.items()}))


@dataclass(frozen=True)
class Domain:
    """The integers `lo <= x < hi` (None: unbounded), restricted to `values` if set."""

    lo: int | None = None
    hi: int | None = None
    values: frozenset | None = None

    def _in_bounds(self, x) -> bool:
        return (self.lo is None or x >= self.lo) and (self.hi is None or x < self.hi)

    def __contains__(self, x) -> bool:
        return self._in_bounds(x) and (self.values is None or x in self.values)

    def __and__(self, other) -> "Domain":
        lo = max((b for b in (self.lo, other.lo) if b is not None), default=None)
        hi = min((b for b in (self.hi, other.hi) if b is not None), default=None)
        if self.values is None:
            values = other.values
        elif other.values is None:
            values = self.values
        else:
            values = self.values & other.values
        return Domain(lo, hi, values)

    @property
    def empty(self) -> bool:
        if self.lo is not None and self.hi is not None and self.lo >= self.hi:
            return True
        return self.values is not None and not any(map(self._in_bounds, self.values))


def _condition(op, threshold) -> Domain:
    if op == "in":
        values = _as_set("in", threshold)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"`in` takes integers, got {threshold!r}")
        return Domain(values=values)
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise TypeError(f"thresholds are integers, got {threshold!r}")
    return {
        "le": Domain(hi=threshold + 1),
        "lt": Domain(hi=threshold),
        "ge": Domain(lo=threshold),
        "gt": Domain(lo=threshold + 1),
    }[op]


@dataclass(frozen=True)
class When:
    """Conditions on named scalars and predicates, all of which must hold.

    `holds` tests them in the order written and stops at the first that fails, so a
    predicate that calls into a kernel library runs only where the legacy condition
    calls it.
    """

    domains: FrozenMap  # scalar or predicate (0/1) name -> Domain
    text: str
    order: tuple = ()  # the names in `domains`, as written

    def holds(self, p) -> bool:
        return all(_when_value(name, p) in self.domains[name] for name in self.order)

    def disjoint(self, other) -> bool:
        return any(
            name in other.domains and (d & other.domains[name]).empty
            for name, d in self.domains.items()
        )

    @property
    def empty(self) -> bool:
        return any(d.empty for d in self.domains.values())

    def __str__(self):
        return self.text


def when(**conditions) -> When:
    """`when(rows_per_expert__le=8, model_dim__gt=2048, topk__in={6, 8}, fmoe_1stage=True)`."""
    if not conditions:
        raise ValueError("when() needs at least one condition; use `otherwise`")
    domains, shown = {}, []
    for key, threshold in conditions.items():
        if key in PREDICATES:
            if not isinstance(threshold, bool):
                raise TypeError(
                    f"{key} is a predicate: True or False, got {threshold!r}"
                )
            d = Domain(values=frozenset({int(threshold)}))
            domains[key] = domains[key] & d if key in domains else d
            shown.append(key if threshold else f"not {key}")
            continue
        name, sep, op = key.rpartition("__")
        if not sep or name not in SCALARS or op not in _OPS:
            raise ValueError(
                f"bad condition {key!r}: use <scalar>__<op> with a scalar in "
                f"{sorted(SCALARS)} and an op in {sorted(_OPS)}, or a predicate in "
                f"{sorted(PREDICATES)}"
            )
        d = _condition(op, threshold)
        domains[name] = domains[name] & d if name in domains else d
        value = _show_set(_as_set(key, threshold)) if op == "in" else threshold
        shown.append(f"{name}{_OPS[op]}{value}")
    return When(FrozenMap(domains), ", ".join(shown), tuple(domains))


def _structure(text) -> tuple:
    """`"sort | quant.a1 | ck.gemm1"` -> `("sort", "quant.a1", "ck.gemm1")`."""
    names = tuple(n.strip() for n in text.split("|"))
    if not all(names):
        raise ValueError(f"empty segment name in structure {text!r}")
    return names


def _structures(texts) -> tuple:
    if isinstance(texts, str):
        raise TypeError(f"a structure list, not a string: {texts!r}")
    return tuple(_structure(t) for t in texts)


@dataclass(frozen=True)
class Region:
    when: When
    prefer: tuple = ()  # structures (segment-name tuples), best first
    # Instead of `prefer`: what the legacy code runs here (not ported yet); the problems
    # of this region are declined back to it.
    legacy: str | None = None
    # The legacy branch also needs `cfg is None`: with a tuned row kept under
    # AITER_BYPASS_TUNE_CONFIG it falls through to the next branch, so the problem is
    # declined.
    untuned_only: bool = False

    def __post_init__(self):
        object.__setattr__(self, "prefer", _structures(self.prefer))
        if (self.legacy is None) == (not self.prefer):
            raise ValueError(f"region [{self.when}]: give either prefer or legacy")


@dataclass(frozen=True)
class Entry:
    name: str
    match: Match
    regions: tuple = ()
    otherwise: tuple = ()  # structures for the rest of the class, best first

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "otherwise", _structures(self.otherwise))

    def lists(self):
        """`(label, structures, legacy)` for every region, then `otherwise` if any."""
        for r in self.regions:
            yield f"{self.name}[{r.when}]", r.prefer, r.legacy
        if self.otherwise:
            yield f"{self.name}[otherwise]", self.otherwise, None

    def region_for(self, p, tuned_row=False):
        """`(label, structures, legacy)` of the region `p` falls in.

        `otherwise` when no region holds (empty structures if the entry has none).
        `tuned_row`: a tuned row is kept although AITER_BYPASS_TUNE_CONFIG ignores it.
        """
        hits = [r for r in self.regions if r.when.holds(p)]
        if len(hits) > 1:
            raise RuntimeError(
                f"{self.name}: overlapping regions {[str(r.when) for r in hits]}"
            )
        if hits:
            r = hits[0]
            label = f"{self.name}[{r.when}]"
            if tuned_row and r.untuned_only:
                return label, (), "the next branch (a tuned row is bypassed)"
            return label, r.prefer, r.legacy
        return f"{self.name}[otherwise]", self.otherwise, None


def _default_table():
    from .ranking_table import RANKING

    return RANKING


def entry_for(p, table=None):
    table = _default_table() if table is None else table
    hits = [e for e in table if e.match.holds(p)]
    if len(hits) > 1:
        raise RuntimeError(f"overlapping ranking entries {[e.name for e in hits]}")
    return hits[0] if hits else None


def best(problem, plans, table=None, tuned_row=False):
    """`(label, plan, legacy)` for `problem`'s region, or None when no entry applies.

    `plan`: the first feasible structure of the region, from `plans` (the feasible
    segment tuples), None if none is; `label` names the entry and region. A region that
    hands the problem to the legacy code gives `(label, None, legacy)`. `tuned_row`: see
    `Entry.region_for`.
    """
    entry = entry_for(problem, table)
    if entry is None:
        return None
    label, prefer, legacy = entry.region_for(problem, tuned_row)
    if legacy is not None:
        return label, None, legacy
    by_names = {tuple(s.name for s in plan): plan for plan in plans}
    for names in prefer:
        if names in by_names:
            return label, by_names[names], None
    return label, None, None


def rankable(plans):
    """The plans the ranking chooses among: whole-graph plans have reach "tuned"."""
    return [plan for plan in plans if len(plan) > 1]


def _shape_free(segment) -> bool:
    """Feasibility comes only from `Cap`s, which have no shape condition."""
    cls = type(segment)
    return cls.reject_reason is Segment.reject_reason and cls.accepts is Segment.accepts


def _structure_errors(label, names, segments):
    unknown = [n for n in names if n not in segments]
    if unknown:
        return [f"{label}: unknown segments {unknown} in {' | '.join(names)}"]
    chain = [segments[n] for n in names]
    errors = []
    if chain[0].covers[0] != Op.SORT or chain[-1].covers[1] != Op.REDUCE:
        errors.append(f"{label}: {' | '.join(names)} does not cover SORT..REDUCE")
    for a, b in itertools.pairwise(chain):
        if b.covers[0] != a.covers[1] + 1:
            errors.append(
                f"{label}: {a.name} and {b.name} are not adjacent in the chain"
            )
    return errors


def check_table(table, segments) -> list:
    """Static problems with `table` given the known `segments`; empty when it passes."""
    errors = []
    by_name = {}
    for s in segments:
        if s.name in by_name and type(by_name[s.name]) is not type(s):
            errors.append(f"two different segments are named {s.name!r}")
        by_name[s.name] = s
    for name, n in Counter(e.name for e in table).items():
        if n > 1:
            errors.append(f"{n} entries are named {name!r}")
    for a, b in itertools.combinations(table, 2):
        if not a.match.disjoint(b.match):
            errors.append(
                f"entries {a.name} and {b.name} overlap: no field separates "
                f"({a.match}) from ({b.match})"
            )
    for e in table:
        if not e.regions and not e.otherwise:
            errors.append(f"{e.name}: no regions and no otherwise")
        for r in e.regions:
            if r.when.empty:
                errors.append(
                    f"{e.name}[{r.when}]: dead region, its conditions contradict"
                )
        for a, b in itertools.combinations(e.regions, 2):
            if not a.when.disjoint(b.when):
                errors.append(f"{e.name}: regions [{a.when}] and [{b.when}] overlap")
        for label, prefer, legacy in e.lists():
            if legacy is not None:
                continue
            if not prefer:
                errors.append(f"{label}: empty structure list")
                continue
            for names, n in Counter(prefer).items():
                if n > 1:
                    errors.append(f"{label}: {' | '.join(names)} listed {n} times")
            for names in prefer:
                errors += _structure_errors(label, names, by_name)
            for n in prefer[-1]:
                if n in by_name and not _shape_free(by_name[n]):
                    errors.append(
                        f"{label}: the last structure is not always feasible: "
                        f"{n} decides feasibility beyond its capabilities"
                    )
    return errors


def check_problems(problems, paths, table=None):
    """`(errors, hits)` over `(label, problem)` pairs.

    A problem is supported by a path (with `takes` and `segments`) when the path takes it
    and it has a rankable feasible plan there; it must land in exactly one entry and one
    region (or `otherwise`) whose last structure is feasible, or that hands it to the
    legacy code. `hits` counts problems per region label, for `dead_lists`; a region
    that hands its problems to the legacy code also counts the taken problems without a
    plan.
    """
    from .resolver import feasible_plans

    table = _default_table() if table is None else table
    errors, hits = [], Counter()
    for label, p in problems:
        for path in paths:
            if not path.takes(p):
                continue
            plans = rankable(feasible_plans(p, path.segments)[0])
            entries = [e for e in table if e.match.holds(p)]
            if not plans:
                if len(entries) == 1:
                    regions = [r for r in entries[0].regions if r.when.holds(p)]
                    if len(regions) == 1 and regions[0].legacy is not None:
                        hits[entries[0].region_for(p)[0]] += 1
                continue
            if len(entries) != 1:
                names = [e.name for e in entries]
                errors.append(f"{label}: lands in {len(entries)} entries {names}")
                continue
            entry = entries[0]
            regions = [r for r in entry.regions if r.when.holds(p)]
            if len(regions) > 1:
                shown = [str(r.when) for r in regions]
                errors.append(f"{label}: lands in regions {shown} of {entry.name}")
                continue
            region, prefer, legacy = entry.region_for(p)
            if legacy is None and not prefer:
                errors.append(f"{label}: no region of {entry.name} and no otherwise")
                continue
            hits[region] += 1
            if legacy is not None:
                continue
            feasible = {tuple(s.name for s in plan) for plan in plans}
            if prefer[-1] not in feasible:
                errors.append(
                    f"{label}: the last structure of {region} is not feasible "
                    f"({' | '.join(prefer[-1])})"
                )
    return errors, hits


def dead_lists(hits, table=None) -> list:
    """Regions (and non-empty `otherwise` lists) no supported problem reached."""
    table = _default_table() if table is None else table
    return [
        f"{label}: dead, no supported problem lands here"
        for e in table
        for label, _, _ in e.lists()
        if not hits[label]
    ]
