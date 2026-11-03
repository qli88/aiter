# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`MoeEngine`: one object per MoE layer geometry, shared by every layer of it.

`fused_moe` runs one call. An engine is created for a geometry (everything but the token
count and the weights), resolves a plan per token bucket in `prepare()`, checks that every
rank of its communicator resolved the same plans, and then runs any layer of that
geometry: `forward` takes the layer's weights (`MoeWeights`) on every call. It is the
entry point for structures that keep state or communicate; for local (single-GPU) MoE it
is an alternative to `fused_moe`.

Lifecycle: CREATED, `prepare()`, PREPARED, `capture()`, READY, `forward()`, ACTIVE, and
`release()` to RELEASED. Without captured buckets (`capture_max_tokens=0`) `prepare()`
ends in READY. A failed `health()` check moves the engine to FAILED, where it refuses
every call but `abort()`. `prepare()`, `capture()`, `reset()` and `release()` are
collective on the engine's communicator; `abort()` never communicates, so a rank can
tear down after a peer died.

Where the plans come from is the engine's template, a `PlanSource`. `LocalTemplate` is
the source of local MoE during the migration: each bucket is resolved through the paths
`aiter.moev2.api.fused_moe` uses, and `forward` calls `fused_moe` itself, so the kernels,
arguments and log lines are those of the functional API. Its per-call steps (the
activation format from the token count, the inline-sort fallback, the lookup after
sorting) stay per call; `prepare()` resolves what they pick for every count of a
bucket (`LocalTemplate.resolve`), so a bucket can hold one plan per activation format.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from .shapes import _PADDED_M_TIERS, get_padded_M


def buckets(max_tokens: int) -> tuple:
    """The token buckets of counts 1 to `max_tokens`: the `get_padded_M` values."""
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    top = get_padded_M(max_tokens)
    out = []
    b = 1
    while b <= top and b < _PADDED_M_TIERS[0]:
        out.append(b)
        b *= 2
    out += [tier for tier in _PADDED_M_TIERS if tier <= top]
    return tuple(out)


def bucket_of(tokens: int) -> int:
    """The bucket a call of `tokens` tokens runs in (0 tokens: bucket 1)."""
    return get_padded_M(tokens)


def bucket_range(bucket: int) -> tuple:
    """`(lowest, highest)` token count of `bucket` (counts 1 and up; highest None for
    the top tier)."""
    if bucket == _PADDED_M_TIERS[-1]:
        return bucket, None
    if bucket in _PADDED_M_TIERS:
        below = _PADDED_M_TIERS.index(bucket)
        lowest = _PADDED_M_TIERS[below - 1] if below else bucket // 2
        return lowest + 1, _PADDED_M_TIERS[below + 1] - 1
    return (bucket // 2 + 1 if bucket > 1 else 1), bucket


@dataclass(frozen=True)
class StepInfo:
    """Per-step values every rank agrees on. `max_tokens_across_ranks` picks the bucket
    in EP and TP, so every rank runs the same plan."""

    max_tokens_across_ranks: int


@dataclass
class MoeWeights:
    """One layer's weights, owned by the caller, in the layout the engine requires
    (`MoeEngine.weight_layout()`); the layout tags `fused_moe` reads (`is_shuffled`,
    `gate_up_interleave`) stay on the tensors."""

    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    bias1: torch.Tensor | None = None
    bias2: torch.Tensor | None = None


@dataclass(frozen=True)
class EngineContext:
    """What the engine hands to the stateful segments of its plans (`prepare`)."""

    comm: object | None  # the torch.distributed group; None on one process
    rank: int
    world_size: int
    device: torch.device | None
    memory_budget: int | None  # bytes the engine may use; None: no limit


@dataclass(frozen=True)
class Health:
    ok: bool
    reasons: tuple = ()


class EngineState(enum.Enum):
    CREATED = "created"
    PREPARED = "prepared"
    READY = "ready"
    ACTIVE = "active"
    FAILED = "failed"
    RELEASED = "released"


class EngineStateError(RuntimeError):
    """A call the engine's current state does not allow."""


class PlanDisagreement(RuntimeError):
    """The ranks of the communicator resolved different plans."""


@dataclass(frozen=True)
class BucketPlan:
    """What a `PlanSource` resolves for one bucket.

    `description`: the plan as text, shown by `explain`; `identity`: what else tells
    two plans apart (e.g. every kernel argument), the same in every process that
    resolves the same plan. Both are hashed for the agreement check. `run(x, topk_ids,
    topk_weights, weights, side_inputs, hints, out)`: runs a call; with `out` and
    `output_lifetime="caller"` it writes the result there. `warmup()`: runs the bucket
    once before graph capture (None: nothing to warm up). `stateful`: the stateful
    segments of the plan, whose lifecycle the engine drives. `side_inputs`: the names
    `forward` must pass; `hints`: the hints the plan can use.
    """

    bucket: int
    description: str
    run: Callable
    identity: str = ""
    warmup: Callable | None = None
    stateful: tuple = ()
    side_inputs: frozenset = frozenset()
    hints: frozenset = frozenset()
    output_lifetime: str = "caller"
    zero_tokens_ok: bool = True
    graph_capturable: bool = True


class PlanSource:
    """Where an engine's plans come from (its template).

    `resolve(bucket, graph_capture)`: the `BucketPlan` of a bucket, or raise
    NotImplementedError with the reason. `digests()`: names and digests of the data the
    plans come from (tuned tables), part of the agreement check. `check_weights(w)`:
    raise ValueError if `w` is not in the layout the plans were resolved for.
    `weight_layout()`: that layout. `serves_parallel`: whether its plans implement a
    `ParallelSpec` (their segments communicate); a source that does not runs only
    engines without one.
    """

    serves_parallel = False

    def resolve(self, bucket: int, graph_capture: bool) -> BucketPlan:
        raise NotImplementedError(type(self).__name__)

    def digests(self) -> dict:
        return {}

    def check_weights(self, weights: MoeWeights) -> None:
        return None

    def weight_layout(self):
        return None


def _aiter_version() -> str:
    try:
        from aiter._version import __version__
    except ImportError:
        return "unknown"
    return __version__


class MoeEngine:
    """See the module docstring; create one with `MoeEngine.create`."""

    def __init__(
        self,
        template: PlanSource,
        ctx: EngineContext,
        parallel,
        max_tokens: int,
        capture_max_tokens: int,
    ):
        self.template = template
        self.ctx = ctx
        self.parallel = parallel
        self.max_tokens = max_tokens
        self.capture_max_tokens = min(capture_max_tokens, max_tokens)
        self.buckets = buckets(max_tokens)
        self.captured = frozenset(
            b
            for b in self.buckets
            if self.capture_max_tokens and b <= bucket_of(self.capture_max_tokens)
        )
        self._state = EngineState.CREATED
        self._plans: dict = {}
        self._stateful: tuple = ()
        self._record: dict | None = None
        self._failure: tuple = ()

    @classmethod
    def create(
        cls,
        template: PlanSource,
        *,
        max_tokens: int,
        parallel=None,
        comm=None,
        capture_max_tokens: int = 0,
        memory_budget: int | None = None,
        device: torch.device | None = None,
    ) -> MoeEngine:
        """An engine for `template` (a `PlanSource`, e.g. `LocalTemplate`).

        `parallel`: the problem's `ParallelSpec` (None: local); EP and TP need `comm`, the
        group the plans run on and agree over. `max_tokens`: the largest token count
        `forward` takes (in EP/TP, the largest `StepInfo.max_tokens_across_ranks`).
        The buckets of counts up to `capture_max_tokens` are resolved for graph capture
        and warmed up by `capture()` (`captured`).
        """
        if parallel is not None and not template.serves_parallel:
            raise ValueError(
                f"{type(template).__name__} plans run no communication; it serves "
                "engines without a ParallelSpec"
            )
        if parallel is not None and comm is None:
            raise ValueError("a ParallelSpec needs the group it runs on: pass comm=")
        rank, world_size = 0, 1
        if comm is not None:
            import torch.distributed as dist

            rank = dist.get_rank(comm)
            world_size = dist.get_world_size(comm)
        if parallel is not None and parallel.world_size != world_size:
            raise ValueError(
                f"ParallelSpec.world_size={parallel.world_size}, but the group has "
                f"{world_size} ranks"
            )
        if capture_max_tokens < 0:
            raise ValueError(
                f"capture_max_tokens must be >= 0, got {capture_max_tokens}"
            )
        ctx = EngineContext(comm, rank, world_size, device, memory_budget)
        return cls(template, ctx, parallel, max_tokens, capture_max_tokens)

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def plan_hash(self) -> str:
        """The digest of the prepared plans every rank agreed on."""
        self._require("plan_hash", *_PREPARED_STATES, EngineState.FAILED)
        text = json.dumps(self._record, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()

    def explain(self) -> str:
        """The prepared plan of each bucket, one line each."""
        self._require("explain", *_PREPARED_STATES, EngineState.FAILED)
        return "\n".join(
            f"bucket {b}{' (captured)' if b in self.captured else ''}: "
            f"{plan.description}"
            for b, plan in self._plans.items()
        )

    def weight_layout(self):
        """The weight layout `forward` expects (`MoeWeights` must be in it)."""
        return self.template.weight_layout()

    # Lifecycle.

    def prepare(self) -> None:
        """Resolve a plan per bucket, check that every rank has the same plans, then
        prepare the stateful segments (compile, allocate; may be collective)."""
        self._require("prepare", EngineState.CREATED)
        try:
            plans = self._resolve_buckets()
            record = {
                "aiter": _aiter_version(),
                "digests": dict(self.template.digests()),
                "buckets": [
                    [b, b in self.captured, plan.description, plan.identity]
                    for b, plan in plans.items()
                ],
            }
        except Exception as exc:
            if self._collective():
                # The peers wait in the all-gather: tell them, then raise here.
                self._gather({"error": f"{type(exc).__name__}: {exc}"})
            raise
        if self._collective():
            _check_agreement(self._gather(record))
        stateful = []
        for plan in plans.values():
            for segment in plan.stateful:
                if all(segment is not s for s in stateful):
                    stateful.append(segment)
        for segment in stateful:
            segment.prepare(self.ctx)
        self._plans, self._stateful, self._record = plans, tuple(stateful), record
        self._state = EngineState.PREPARED if self.captured else EngineState.READY

    def _resolve_buckets(self) -> dict:
        plans = {}
        for b in self.buckets:
            captured = b in self.captured
            plan = self.template.resolve(b, captured)
            if plan.bucket != b:
                raise ValueError(f"the plan of bucket {b} is for bucket {plan.bucket}")
            if captured and not plan.graph_capturable:
                raise NotImplementedError(
                    f"bucket {b} is captured, but its plan is not graph-capturable: "
                    f"{plan.description}"
                )
            if self.parallel is not None and not plan.zero_tokens_ok:
                raise NotImplementedError(
                    f"bucket {b}: an EP/TP plan must run on ranks with no tokens: "
                    f"{plan.description}"
                )
            plans[b] = plan
        return plans

    def capture(self) -> None:
        """Warm up every captured bucket (compiles and first-call work happen here,
        not inside the caller's graph capture). The caller captures `forward` itself:
        the weights are per call, so a graph belongs to a layer, not to the engine."""
        self._require("capture", EngineState.PREPARED)
        for b, plan in self._plans.items():
            if b in self.captured and plan.warmup is not None:
                plan.warmup()
        self._state = EngineState.READY

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        weights: MoeWeights,
        step: StepInfo | None = None,
        side_inputs: Mapping | None = None,
        hints: Mapping | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one layer. `step` is required in EP/TP: the bucket comes from
        `step.max_tokens_across_ranks`, never from a hidden collective. `side_inputs`:
        exactly the ones the plan reads; `hints`: optional, passed only to plans that
        use them. With `out`, the result is written there and `out` is returned;
        without it, a plan whose output lives in a workspace returns a view that the
        next `forward` of this engine overwrites."""
        if self._state not in (EngineState.READY, EngineState.ACTIVE):
            needs = {
                EngineState.CREATED: "; call prepare() first",
                EngineState.PREPARED: "; call capture() first",
                EngineState.FAILED: "; only abort() is allowed",
            }.get(self._state, "")
            raise EngineStateError(
                f"forward needs a READY engine, it is {self._state.value}{needs}"
            )
        tokens = int(x.shape[0])
        if step is None:
            if self.parallel is not None:
                raise ValueError(
                    "forward needs step=StepInfo(max_tokens_across_ranks=...) in EP/TP: "
                    "every rank must pick the bucket from the same value"
                )
            agreed = tokens
        else:
            agreed = int(step.max_tokens_across_ranks)
            if agreed < tokens:
                raise ValueError(
                    f"step.max_tokens_across_ranks={agreed} is below this rank's "
                    f"{tokens} tokens"
                )
        if agreed > self.max_tokens:
            raise ValueError(f"{agreed} tokens exceed max_tokens={self.max_tokens}")
        plan = self._plans[bucket_of(agreed)]
        side = dict(side_inputs or {})
        unknown = sorted(set(side) - plan.side_inputs)
        missing = sorted(plan.side_inputs - set(side))
        if unknown or missing:
            raise ValueError(
                f"side_inputs: the plan reads {sorted(plan.side_inputs)}; "
                f"missing {missing}, not read {unknown}"
            )
        used = {k: v for k, v in (hints or {}).items() if k in plan.hints}
        self.template.check_weights(weights)
        result = plan.run(x, topk_ids, topk_weights, weights, side, used, out)
        if out is not None and result is not out:
            out.copy_(result)
            result = out
        self._state = EngineState.ACTIVE
        return result

    def health(self) -> Health:
        """The stateful segments' health; a failure moves the engine to FAILED."""
        self._require("health", *_PREPARED_STATES, EngineState.FAILED)
        if self._state == EngineState.FAILED:
            return Health(False, self._failure)
        reasons = tuple(r for r in (s.health() for s in self._stateful) if r)
        if reasons:
            self._failure = reasons
            self._state = EngineState.FAILED
        return Health(not reasons, reasons)

    def reset(self) -> None:
        """Restart the stateful segments' cross-rank state (collective)."""
        self._require("reset", EngineState.READY, EngineState.ACTIVE)
        for segment in self._stateful:
            segment.reset()

    def release(self) -> None:
        """Free the stateful segments on a healthy group (collective)."""
        self._require("release", EngineState.CREATED, *_PREPARED_STATES)
        for segment in self._stateful:
            segment.release()
        self._state = EngineState.RELEASED

    def abort(self) -> None:
        """Free this rank's state without communicating (after a peer failure)."""
        if self._state == EngineState.RELEASED:
            return
        for segment in self._stateful:
            segment.abort()
        self._state = EngineState.RELEASED

    # Internals.

    def _require(self, what, *states):
        if self._state not in states:
            allowed = ", ".join(s.value for s in states)
            only_abort = (
                "; only abort() is allowed" if self._state == EngineState.FAILED else ""
            )
            raise EngineStateError(
                f"{what} needs the engine {allowed}; it is {self._state.value}{only_abort}"
            )

    def _collective(self) -> bool:
        return self.ctx.comm is not None and self.ctx.world_size > 1

    def _gather(self, record) -> list:
        import torch.distributed as dist

        gathered = [None] * self.ctx.world_size
        dist.all_gather_object(gathered, record, group=self.ctx.comm)
        return gathered


_PREPARED_STATES = (EngineState.PREPARED, EngineState.READY, EngineState.ACTIVE)


def _check_agreement(gathered):
    """Raise PlanDisagreement, the same on every rank, unless all records are equal."""
    for rank, record in enumerate(gathered):
        if "error" in record:
            raise PlanDisagreement(f"rank {rank} could not prepare: {record['error']}")
    for rank, other in enumerate(gathered):
        if other != gathered[0]:
            raise PlanDisagreement(_difference(gathered[0], other, rank))


def _difference(reference, other, rank) -> str:
    """The first difference between rank 0's agreement record and `rank`'s."""
    head = f"ranks 0 and {rank} resolved different MoE plans: "
    if reference["aiter"] != other["aiter"]:
        return head + f"aiter {reference['aiter']} vs {other['aiter']}"
    if reference["digests"] != other["digests"]:
        names = sorted(set(reference["digests"]) | set(other["digests"]))
        differ = [
            n for n in names if reference["digests"].get(n) != other["digests"].get(n)
        ]
        return head + f"{', '.join(differ)} differ (a different file on one host?)"
    for mine, theirs in zip(reference["buckets"], other["buckets"]):
        if mine != theirs:
            shown = 2 if mine[2] != theirs[2] else 3
            return head + f"bucket {mine[0]}: {mine[shown]} vs {theirs[shown]}"
    return head + "different buckets"


def _canonical(value):
    """`value` as text that is the same in every process (no object addresses)."""
    if isinstance(value, functools.partial):
        args = [_canonical(a) for a in value.args]
        kwargs = [f"{k}={_canonical(v)}" for k, v in sorted(value.keywords.items())]
        return f"{_canonical(value.func)}({', '.join(args + kwargs)})"
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return f"{value.__module__}.{value.__qualname__}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        inner = ", ".join(
            f"{f.name}={_canonical(getattr(value, f.name))}"
            for f in dataclasses.fields(value)
        )
        return f"{type(value).__name__}({inner})"
    if isinstance(value, enum.Enum):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_canonical(v) for v in value) + ")"
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(f"{k}: {_canonical(v)}" for k, v in sorted(value.items()))
            + "}"
        )
    if isinstance(value, torch.dtype):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    if callable(value):
        kind = type(value)
        return f"{kind.__module__}.{kind.__qualname__}"
    return repr(value)


# Local MoE through the migration paths.

_PER_CALL = (
    "hidden_states",
    "w1",
    "w2",
    "topk_weight",
    "topk_ids",
    "w1_scale",
    "w2_scale",
    "bias1",
    "bias2",
    "output",
)
LOCAL_SIDE_INPUTS = frozenset({"a1_scale", "a2_scale", "num_local_tokens"})
# FHMoE (the shared expert) and the EP scatter are not served by MoeEngine yet.
_NOT_SUPPORTED = (
    "shared_w1",
    "shared_w2",
    "shared_w1_scale",
    "shared_w2_scale",
    "shared_expert_id",
    "clamp_shared",
    "stage2_scatter",
)


@functools.cache
def _fused_moe_defaults() -> dict:
    from . import api

    return {
        name: p.default
        for name, p in inspect.signature(api.fused_moe).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _weights_signature(weights: MoeWeights) -> dict:
    sig = {}
    for f in dataclasses.fields(MoeWeights):
        t = getattr(weights, f.name)
        sig[f.name] = (
            None
            if t is None
            else (
                tuple(t.shape),
                t.dtype,
                getattr(t, "is_shuffled", False),
                getattr(t, "gate_up_interleave", None),
            )
        )
    return sig


@dataclass
class LocalTemplate(PlanSource):
    """A local (single-GPU) MoE geometry for `MoeEngine`, as `fused_moe` arguments.

    `weights`: one layer's weights; their shapes, dtypes and layout tags are those of
    every layer (meta tensors do, but then `capture()` has nothing to warm up with).
    `topk`: the routed experts per token. `hidden_dtype`: the dtype of `x`.
    `options`: the `fused_moe` keyword arguments every call shares (activation,
    quant_type, gate_mode, doweight_stage1, dtype, padding, expert_mask, ...).
    `side_inputs`: the per-call `fused_moe` tensors the calls pass (of
    `LOCAL_SIDE_INPUTS`); each changes the dispatch key, so every call passes them.
    """

    weights: MoeWeights
    topk: int
    hidden_dtype: torch.dtype = torch.bfloat16
    options: Mapping = field(default_factory=dict)
    side_inputs: frozenset = frozenset()

    def __post_init__(self):
        self.side_inputs = frozenset(self.side_inputs)
        unknown_side = sorted(self.side_inputs - LOCAL_SIDE_INPUTS)
        if unknown_side:
            raise ValueError(
                f"side_inputs {unknown_side} are not per-call fused_moe tensors "
                f"({sorted(LOCAL_SIDE_INPUTS)})"
            )
        known = set(_fused_moe_defaults())
        bad = sorted(
            k
            for k in self.options
            if k not in known or k in _PER_CALL or k in LOCAL_SIDE_INPUTS
        )
        if bad:
            raise ValueError(f"options {bad} are not shared fused_moe arguments")
        unsupported = sorted(set(self.options) & set(_NOT_SUPPORTED))
        if unsupported:
            raise NotImplementedError(f"MoeEngine does not serve {unsupported} yet")
        self._signature = _weights_signature(self.weights)

    def _call(self, x, topk_ids, topk_weights, weights, side, out) -> dict:
        return {
            **_fused_moe_defaults(),
            **self.options,
            **side,
            "hidden_states": x,
            "w1": weights.w1,
            "w2": weights.w2,
            "topk_weight": topk_weights,
            "topk_ids": topk_ids,
            "w1_scale": weights.w1_scale,
            "w2_scale": weights.w2_scale,
            "bias1": weights.bias1,
            "bias2": weights.bias2,
            "output": out,
        }

    def _model_dim(self) -> int:
        from .shapes import get_inter_dim

        _, model_dim, _ = get_inter_dim(
            tuple(self.weights.w1.shape), tuple(self.weights.w2.shape)
        )
        return model_dim

    def _meta_call(self, bucket) -> dict:
        """A call of `bucket` tokens whose per-call tensors have no storage but the
        weights' device: per-call checks compare devices (the inline-sort fallback)."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        device = self.weights.w1.device
        with FakeTensorMode():
            x = torch.empty(
                (bucket, self._model_dim()), dtype=self.hidden_dtype, device=device
            )
            ids = torch.empty((bucket, self.topk), dtype=torch.int32, device=device)
            weights = torch.empty(
                (bucket, self.topk), dtype=torch.float32, device=device
            )
            side = {name: torch.empty(1, device=device) for name in self.side_inputs}
        return self._call(x, ids, weights, self.weights, side, None)

    def resolve(self, bucket: int, graph_capture: bool) -> BucketPlan:
        """The plans `fused_moe` runs for the counts of `bucket`. The dispatch key holds
        the bucket, but the activation dtype follows the count itself
        (`policy.activation_dtype_bounds`), so the bucket is resolved at its ends and on
        both sides of each bound inside it; counts between two that share a plan get
        it too. Several plans: one `<first>-<last> tokens: <plan>` part each."""
        from .policy import activation_dtype_bounds

        lowest, highest = bucket_range(bucket)
        counts = {lowest, bucket}
        for bound in activation_dtype_bounds():
            counts.update(
                m
                for m in (bound - 1, bound)
                if m >= lowest and (highest is None or m <= highest)
            )
        parts = []  # [first count, description, identity]
        for m in sorted(counts):
            plan = self._resolve_count(m)
            if not parts or parts[-1][1:] != list(plan):
                parts.append([m, *plan])
        if len(parts) == 1:
            description, identity = parts[0][1:]
        else:
            ends = [first - 1 for first, *_ in parts[1:]] + [highest]
            labels = [
                (
                    f"{first}+ tokens"
                    if end is None
                    else f"{first} tokens" if end == first else f"{first}-{end} tokens"
                )
                for (first, *_), end in zip(parts, ends)
            ]
            description = "; ".join(
                f"{label}: {p[1]}" for label, p in zip(labels, parts)
            )
            identity = "; ".join(f"{label}: {p[2]}" for label, p in zip(labels, parts))
        return BucketPlan(
            bucket,
            description,
            self._run,
            identity=identity,
            warmup=functools.partial(self._warmup, tuple(p[0] for p in parts)),
            side_inputs=self.side_inputs,
        )

    def _resolve_count(self, tokens) -> tuple:
        """`(description, identity)` of the plan `fused_moe` runs for `tokens` tokens."""
        from . import api

        for path, path_call in api.claiming_paths(self._meta_call(tokens)):
            if path.resolve is None:
                raise NotImplementedError(
                    f"path {path.name} claims {tokens} tokens, but MoeEngine does not "
                    "run it yet"
                )
            resolution = path.resolve(path_call)
            if resolution is None or resolution.plan is None:
                continue
            return (
                f"{path.name}: {resolution.plan.describe()}",
                _canonical(resolution.metadata),
            )
        raise NotImplementedError(f"{tokens} tokens: {api.NO_PATH}")

    def _run(self, x, topk_ids, topk_weights, weights, side, hints, out):
        from . import api

        return api.fused_moe(
            **self._call(x, topk_ids, topk_weights, weights, side, out)
        )

    def _warmup(self, counts):
        """Run once at each count (one per plan of the bucket)."""
        w1 = self.weights.w1
        if w1.is_meta or self.side_inputs:
            return
        experts, device = w1.shape[0], w1.device
        for m in counts:
            ids = (
                torch.arange(m * self.topk, dtype=torch.int32, device=device) % experts
            ).view(m, self.topk)
            x = torch.zeros(
                (m, self._model_dim()), dtype=self.hidden_dtype, device=device
            )
            topk_weights = torch.full(
                (m, self.topk), 1.0 / self.topk, dtype=torch.float32, device=device
            )
            self._run(x, ids, topk_weights, self.weights, {}, {}, None)

    def digests(self) -> dict:
        from .dispatch import tuned

        path = tuned.default_table_file()
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            digest = "missing"
        return {"tuned_table": digest}

    def check_weights(self, weights: MoeWeights) -> None:
        theirs = _weights_signature(weights)
        for name, mine in self._signature.items():
            if theirs[name] != mine:
                raise ValueError(
                    f"weights.{name} is {theirs[name]}, the engine was prepared for "
                    f"{mine} (shape, dtype, is_shuffled, gate_up_interleave)"
                )

    def weight_layout(self):
        from . import api

        return api.build_problem(api.op_call(self._meta_call(1))).layout
