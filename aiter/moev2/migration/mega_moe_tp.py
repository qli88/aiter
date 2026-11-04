# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mega_moe_tp`: `MegaMoeTP.forward` (gfx950, TP).

Each call runs one segment over the instance, the one legacy's switch picks:
`mega_moe_tp.bigm` from the instance's big-M threshold (its checks, input conversions,
then the ar or ag_rs launch), else `mega_moe_tp.small` (the small engine's forward). The
kernels, checks and errors are legacy's. `claims`: a gfx950 device (the only one the
instance is built on).

`MegaMoeTPTemplate` serves the instance to `MoeEngine`, with the fused tail as an
optional side input where legacy fuses it.
"""

from dataclasses import dataclass, fields

from aiter.jit.utils.chip_info import get_gfx_runtime

from .. import paths
from ..dispatch.plan import MoePlan, SegmentChoice
from ..engine import (
    BucketPlan,
    MoeWeights,
    PlanSource,
    _canonical,
    _weights_signature,
    bucket_range,
    file_digest,
)
from ..segments.mega_moe_tp import MegaMoeTPBigM, MegaMoeTPSmall

NAME = "mega_moe_tp"

# The fused tail's side inputs (`MegaMoeTP.forward`'s `tail`, in its order).
TAIL = ("res_in", "res_out", "norm_w")
TAIL_BF16 = "tail_bf16"


def claims(call) -> bool:
    return get_gfx_runtime() == "gfx950"


def forward(small: MegaMoeTPSmall, bigm: MegaMoeTPBigM, call):
    """`MegaMoeTP.forward` over the segments, in its order (`call`: its arguments by
    name, the instance as `mtp`)."""
    x, topk_weights, topk_ids = call["x"], call["topk_weights"], call["topk_ids"]
    out, tail, bf16 = call["out"], call["tail"], call["bf16"]
    if bigm.serves(x.shape[0]):
        bigm.check(x, topk_weights, topk_ids, out, tail)
        x, topk_weights, topk_ids = bigm.inputs(x, topk_weights, topk_ids)
        if bigm.mtp.cfg.comm_mode == "ar":
            return bigm.ar(x, topk_weights, topk_ids, out, tail)
        return bigm.ag_rs(x, topk_weights, topk_ids, out, tail, bf16)
    return small.forward(x, topk_weights, topk_ids, out, tail, bf16)


def run(call):
    mtp = call["mtp"]
    return forward(MegaMoeTPSmall.adopt(mtp), MegaMoeTPBigM.adopt(mtp), call)


# AITER_MOEV2=mega_moe_tp until its TP differential (big-M included) passes and it is
# asked to graduate.
PATH = paths.Path(NAME, claims, run, entry="mega_tp")
paths.register(PATH)


def _label(first: int, last: int) -> str:
    return f"{first} tokens" if first == last else f"{first}-{last} tokens"


@dataclass
class MegaMoeTPTemplate(PlanSource):
    """`MegaMoeTP` for `MoeEngine` (gfx950, TP).

    `mtp`: a `MegaMoeTP` the caller built (its constructor is collective) with one
    layer's weights. The caller owns it: neither `engine.release()` nor `abort()` frees
    it. After the engine is done, call `mtp.close()` on every rank (collective: it
    unmaps the IPC arena and closes the big-M kernel's SDMA queues), or synchronize every
    rank before dropping it (the arena is unmapped when collected, the SDMA queues are
    not closed). Other users of the instance (its own `forward`, another engine) join
    the same collective sequence: every rank calls them in the same order.

    Token counts are `forward`'s rows, as the instance takes them: ag_rs, this rank's
    rows, the same count on every rank (legacy's contract; `StepInfo` is that count); ar,
    the replicated batch. Each forward binds the layer's weights (legacy's
    `set_weights` when they are not the bound ones; shapes and dtypes of the example
    layer) and runs the instance's own small or big-M kernel. The result is legacy's: a
    view of the instance's buffers, valid until the next forward (`out=` to keep it).

    Per bucket the plan is `mega_moe_tp.small` below the big-M threshold and
    `mega_moe_tp.bigm` from it, both split at it when the bucket straddles it; a bucket
    whose counts are all above the instance's `max_local_tokens` has no plan. Big-M is not
    graph-capturable: a captured bucket that reaches it fails `prepare()`. `supports(n)`:
    the counts the instance takes (up to `max_local_tokens` local tokens, so ar takes
    `max_local_tokens * tp` rows; on the small engine: within its capacity, ar multiples
    of tp, fp8 activations within the dynamic schedule).

    The fused tail: calls may pass side inputs `res_in`, `res_out`, `norm_w` (`TAIL`,
    all three or none), and `tail_bf16` (`TAIL_BF16`, True) with them for legacy's
    `bf16=True`, where `engine.supports(n, side_inputs=...)` says: that variant is
    prepared (`tail`: the plain one, `tail_bf16`: the bf16 one; either or both) and
    legacy's `tail_ok` of the count holds (ag_rs on the LB schedule up to `TN_MAX` local
    tokens, or on big-M; never in ar). They return legacy's `(y, q, scale)`, with the
    bf16 rows for `tail_bf16`. Elsewhere the caller runs its own add + norm + quant, as
    legacy callers do.

    `capture()` runs the instance's `prepare` once, on every local count of the captured
    buckets that the small engine runs, with the tail variants (collective: compile, arm, run
    once, the watchdog check; and the big-M set-up when it is reachable). Without
    captured buckets nothing is prepared: as for the instance alone, each launch config
    compiles at its first forward. `health()` is the instance's `poll_errors()` (a sync),
    `reset()` its `reset()` (collective). `engine.memory()` is the instance's memory
    (its small engine's, and its big-M kernel's when a bucket reaches it or, as
    `capture()` sets big-M up, when a bucket is captured and the instance reaches it;
    the caller allocated the first), so `memory_budget` sees it (the segments' module
    docstring says what is not counted)."""

    mtp: object
    tail: bool = False
    tail_bf16: bool = False

    serves_parallel = True

    def __post_init__(self):
        mtp = self.mtp
        self.small = MegaMoeTPSmall.adopt(mtp)
        self.bigm = MegaMoeTPBigM.adopt(mtp)
        a = mtp._bigm_args
        self._signature = _weights_signature(
            MoeWeights(
                w1=a["w1"], w2=a["w2"], w1_scale=a["w1_scale"], w2_scale=a["w2_scale"]
            )
        )
        self._captured: set = set()
        self._prepared: frozenset | None = None

    # Token counts.

    def _ar(self) -> bool:
        return self.mtp.cfg.comm_mode == "ar"

    def _max_rows(self) -> int:
        top = self.mtp.max_local_tokens
        return top * self.mtp.cfg.world_size if self._ar() else top

    def _first_big(self) -> int | None:
        """The fewest rows legacy runs on the big-M kernel (None: never)."""
        threshold = self.mtp.bigm_m
        if not threshold:
            return None
        if self._ar():
            return (threshold - 1) * self.mtp.cfg.world_size + 1
        return threshold

    def _small_locals(self, first: int, last: int) -> list:
        """The small engine's local counts of rows `first..last` (ar: multiples of tp)."""
        if not self._ar():
            return [m for m in range(first, last + 1) if m > 0]
        tp = self.mtp.cfg.world_size
        return [r // tp for r in range(first, last + 1) if r > 0 and r % tp == 0]

    def _parts(self, bucket: int) -> list:
        """`[(segment, first, last)]`: the rows of the bucket each segment runs."""
        lowest, highest = bucket_range(bucket)
        top = self._max_rows()
        if lowest > top:
            raise NotImplementedError(
                f"{NAME}: bucket {bucket} starts at {lowest} tokens, above the "
                f"instance's {top} (max_local_tokens={self.mtp.max_local_tokens})"
            )
        last = top if highest is None else min(highest, top)
        first_big = self._first_big()
        if first_big is None or last < first_big:
            return [(self.small, lowest, last)]
        if lowest >= first_big:
            return [(self.bigm, lowest, last)]
        return [(self.small, lowest, first_big - 1), (self.bigm, first_big, last)]

    # PlanSource.

    def resolve(self, bucket: int, graph_capture: bool) -> BucketPlan:
        parts = self._parts(bucket)
        describe = [
            (s, first, last, MoePlan((SegmentChoice(s, s.params()),), "default"))
            for s, first, last in parts
        ]
        if len(describe) == 1:
            description = describe[0][3].describe()
        else:
            description = "; ".join(
                f"{_label(first, last)}: {plan.describe()}"
                for _, first, last, plan in describe
            )
        launch = []
        for s, first, last in parts:
            if s is self.small:
                launch = self._launch_configs(first, last)
                if graph_capture:
                    self._captured.update(
                        m
                        for m in self._small_locals(first, last)
                        if self._small_runs(m)
                    )
        cfg = self.mtp.cfg
        identity = _canonical(
            {
                "config": {
                    f.name: getattr(cfg, f.name)
                    for f in fields(cfg)
                    if f.name != "rank"
                },
                "threshold": self.mtp.bigm_m,
                "capacity": self.mtp.engine.mmax,
                "parts": [(s.name, first, last) for s, first, last in parts],
                "launch_configs": launch,
                "tail": (self.tail, self.tail_bf16),
                "tn_local": self.mtp.engine.tn_local,
            }
        )
        stateful = tuple(dict.fromkeys(s for s, _, _ in parts))
        if graph_capture and self.mtp._big(self.mtp.cfg.max_local_tokens):
            stateful = tuple(dict.fromkeys((*stateful, self.bigm)))
        return BucketPlan(
            bucket,
            description,
            self._run,
            identity=identity,
            warmup=self._warmup,
            stateful=stateful,
            optional_side_inputs=self._tail_inputs(),
            output_lifetime=self.small.output_lifetime,
            graph_capturable=all(s is self.small for s, _, _ in parts),
            supports=self._supports,
            supports_optional=self._supports_tail,
        )

    def _small_runs(self, local: int) -> bool:
        """The small engine runs this local count: within its capacity (an explicit
        `bigm_min_tokens` can leave counts above it that big-M does not take) and its
        schedules (fp8 activations stop at the dynamic one)."""
        if not 0 < local <= self.mtp.engine.mmax:
            return False
        try:
            self.mtp.launch_config(local)
        except ValueError:
            return False
        return True

    def _launch_configs(self, first: int, last: int) -> list:
        """The small engine's distinct launch configs of these rows, in count order."""
        seen = []
        for m in self._small_locals(first, last):
            if self._small_runs(m):
                cfg = self.mtp.launch_config(m)
                if cfg not in seen:
                    seen.append(cfg)
        return seen

    def _supports(self, rows: int) -> bool:
        if rows > self._max_rows():
            return False
        if self.bigm.serves(rows) or rows == 0:
            return True
        if self._ar() and rows % self.mtp.cfg.world_size:
            return False
        return self._small_runs(self.mtp._local(rows))

    def _tail_inputs(self) -> frozenset:
        if not (self.tail or self.tail_bf16):
            return frozenset()
        return frozenset(TAIL) | ({TAIL_BF16} if self.tail_bf16 else set())

    def _supports_tail(self, rows: int, names) -> bool:
        names = frozenset(names)
        prepared = self.tail_bf16 if TAIL_BF16 in names else self.tail
        return (
            prepared
            and names - {TAIL_BF16} == frozenset(TAIL)
            and self.mtp.tail_ok(self.mtp._local(rows))
        )

    def digests(self) -> dict:
        from aiter.jit.core import AITER_CONFIGS

        return {
            "mega_moe_tp_tuned": file_digest(
                AITER_CONFIGS.AITER_CONFIG_MEGA_MOE_TP_FILE
            ),
            "mega_moe_tp_bigm_tuned": file_digest(
                AITER_CONFIGS.AITER_CONFIG_MEGA_MOE_TP_BIGM_FILE
            ),
        }

    def check_weights(self, weights: MoeWeights) -> None:
        theirs = _weights_signature(weights)
        for name, mine in self._signature.items():
            if theirs[name] != mine:
                raise ValueError(
                    f"weights.{name} is {theirs[name]}, the engine was prepared for "
                    f"{mine} (shape, dtype, is_shuffled, gate_up_interleave)"
                )

    # Calls.

    def _warmup(self):
        """The instance's `prepare` on every local count of the captured buckets, again
        only when an engine captured counts not prepared yet."""
        captured = frozenset(self._captured)
        if self._prepared is not None and captured <= self._prepared:
            return
        self.mtp.prepare(
            sorted(captured),
            tail=self.tail,
            tail_bf16=self.tail_bf16,
        )
        self._prepared = captured

    def _bind(self, weights: MoeWeights) -> None:
        engine = self.mtp.engine
        new = (weights.w1, weights.w1_scale, weights.w2, weights.w2_scale)
        bound = (engine.w1, engine.w1s, engine.w2, engine.w2s)
        if all(t.data_ptr() == b.data_ptr() for t, b in zip(new, bound)):
            return
        self.mtp.set_weights(
            w1=weights.w1,
            w1_scale=weights.w1_scale,
            w2=weights.w2,
            w2_scale=weights.w2_scale,
        )

    def _run(self, x, topk_ids, topk_weights, weights, side, hints, out):
        tail = None
        bf16 = TAIL_BF16 in side
        if bf16 and side[TAIL_BF16] is not True:
            raise ValueError("side_inputs: tail_bf16 is a flag: pass True or omit it")
        if side:
            missing = [name for name in TAIL if name not in side]
            if missing:
                raise ValueError(
                    f"side_inputs: the fused tail needs {list(TAIL)}; missing {missing}"
                )
            tail = tuple(side[name] for name in TAIL)
        self._bind(weights)
        call = {
            "mtp": self.mtp,
            "x": x,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "out": out,
            "tail": tail,
            "bf16": bf16,
        }
        return forward(self.small, self.bigm, call)
