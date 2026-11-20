# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The `Segment` interface, capabilities and parameters.

A segment is a thin adapter over one kernel family that covers a span of the op chain.
It declares what it can run as `Cap`s; anything a problem has that no `Cap` declares
(a feature, a shared expert, a tail, padding, an extra) rejects the segment, so a new
input never reaches a kernel that would silently ignore it.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from dataclasses import dataclass

from ..dispatch.ops import Op
from ..problem import FrozenMap, MoeProblem


class Params(FrozenMap):
    """A segment's or plan's parameters: an immutable mapping with attribute access."""

    __slots__ = ()

    def __init__(self, items=None, **kwargs):
        super().__init__({**(items or {}), **kwargs})

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def replace(self, **kwargs) -> "Params":
        return Params(self, **kwargs)

    def __repr__(self):
        return "Params(" + ", ".join(f"{k}={v!r}" for k, v in self.items()) + ")"


def _short(dtype):
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class Cap:
    """One combination a segment supports. `None` means "any"; sets mean membership.

    `features` follows the subset rule: every feature the problem uses must be listed.
    """

    arch: frozenset | None = None
    act_in: frozenset | None = None  # ScaleSpecs
    w1: frozenset | None = None
    w2: frozenset | None = None
    out_dtype: frozenset | None = None
    activation: frozenset | None = None  # ActSpec.kind values
    gate_mode: frozenset | None = None
    shuffle: frozenset | None = None
    g1u1: frozenset | None = None
    features: frozenset = frozenset()
    pads: bool = False  # hidden/intermediate padding supported
    extras: frozenset = frozenset()  # extras keys understood
    shared: bool = False
    parallel: bool = False
    tail: bool = False

    def reject_reason(self, p: MoeProblem) -> str | None:
        """Why this capability does not cover `p`, or None if it does."""
        layout = p.layout
        checks = (
            ("arch", self.arch, p.arch, str),
            ("act_in", self.act_in, p.act_in, str),
            ("w1", self.w1, p.w1, str),
            ("w2", self.w2, p.w2, str),
            ("out dtype", self.out_dtype, p.out_dtype, _short),
            ("activation", self.activation, p.activation.kind, str),
            ("gate mode", self.gate_mode, layout.gate_mode, str),
            ("weight shuffle", self.shuffle, layout.shuffle, str),
            ("g1u1", self.g1u1, layout.g1u1, str),
        )
        for what, allowed, value, show in checks:
            if allowed is not None and value not in allowed:
                return f"{what} {show(value)} not in {{{', '.join(sorted(map(show, allowed)))}}}"
        missing = sorted(p.features - self.features)
        if missing:
            return f"feature {', '.join(missing)} not supported"
        if not self.pads and (layout.hidden_pad or layout.intermediate_pad):
            return "hidden/intermediate padding not supported"
        unknown = sorted(set(p.extras) - self.extras)
        if unknown:
            return f"extras {', '.join(unknown)} not understood"
        for what, allowed, value in (
            ("shared expert", self.shared, p.shared),
            ("parallel spec", self.parallel, p.parallel),
            ("tail", self.tail, p.tail),
        ):
            if value is not None and not allowed:
                return f"{what} not supported"
        return None


@dataclass(frozen=True)
class RowRejection:
    """A tuned row a segment cannot run. `warning` is logged as is; None is silent."""

    warning: str | None


NO_OPTIONS = FrozenMap()


@dataclass(frozen=True)
class DefaultKernels:
    """The kernels a segment picks when no tuned row applies, and the warning logged
    (None: none). `options`: extra stage 1 parameters (`Params`) of the choice."""

    kernel1: str
    kernel2: str
    block_m: int
    ksplit: int
    warning: str | None
    options: FrozenMap = NO_OPTIONS


REACHES = ("ranked", "tuned", "hibernating")


class Segment:
    """Base class of every segment: the reference for its attributes and methods.

    Attributes, grouped by who reads them. Adding one means documenting it here.

    Identity and dispatch (read by the resolver, the ranking, `coverage.py` and the
    executor):

    - `name`: unique; plans, logs, `AITER_MOE_EXPLAIN`, the ranking table and
      `hibernating.py` name the segment by it, and the executor finds `sort` and
      `quant.*` by it. Convention: `<family>.<stage>[_variant]`, e.g. `ck.gemm1`
      (`sort` and `quant.*` excepted).
    - `prefixes`: the kernel-name families (`kernel_names.FAMILIES`) whose tuned rows
      the segment runs (`owns`). Empty for segments no tuned row names (sort, quant).
    - `covers`: `(first, last)` ops of the chain (`dispatch.ops.Op`) it implements; the
      next segment of a plan starts at the op after `last`. A stage 1 covers
      `(GEMM1, ACT)`, or `(GEMM1, QUANT_INTER)` when it writes GEMM2's input format
      itself; a stage 2 `(GEMM2, REDUCE)`; the 1-stage ASM kernels `(QUANT_A1, REDUCE)`;
      a whole-graph implementation `(SORT, REDUCE)`.
    - `capabilities`: the `Cap`s it supports. A problem no `Cap` covers rejects the
      segment (`reject_reason`); no `Cap` at all accepts nothing.
    - `reach` (`REACHES`, checked by `coverage.segment_errors`): how dispatch gets to
      the segment without a tuned row. A tuned row runs the segment that owns its kernel
      name whatever its reach. "ranked": in some ranking entry; tuned rows may run it
      too, and are not checked (a retune that drops its last row leaves it reachable).
      "tuned": in no ranking entry, so run only through tuned rows, as in legacy; some
      shipped tuned row runs it. "hibernating": in no ranking entry and run by no
      shipped row, only by a table passed with AITER_CONFIG_FMOE or by a test; it needs
      a GPU test (tests/moe_dispatch/hibernating.py).

    Plan parameters (declared for the tuner and the engine; nothing reads them yet):

    - `structure_params`: parameters that change a boundary buffer, a wire format or a
      buffer size (sort `block_m`, a dispatch scheme, a capacity), so every segment of
      a plan and every rank must agree on them; the tuner enumerates their values as
      separate structures. Today `block_m` is a plan field (`MoePlan.params`, copied to
      `MOEMetadata.block_m`); `Sort` declares it here, unread.
    - `rank_local_params`: True if the local parameters (tiles, a bundle entry) may be
      picked from the rank's own tokens, because they never change the wire. False: all
      ranks use the same ones.

    Runtime and collective behavior (for the planned `MoeEngine`; nothing reads them
    yet):

    - `stateful`: keeps state across calls (symmetric memory, compiled bundles,
      workspaces), so the engine creates, prepares and releases one instance per
      structure.
    - `zero_tokens_ok`: runs, and joins its collectives, on a rank with no tokens. EP
      problems exclude segments without it: idle data-parallel ranks still call every
      layer.
    - `hints`: optional per-call inputs that only help speed, e.g. `"next_topk_ids"`
      (`MegaMoEGfx1250` prefetches the next layer's plan). The engine passes a hint only
      to segments that declare it; ignoring one is always correct, and hints are never
      part of the key.
    - `output_lifetime`: "caller" (the result is the caller's) or "until_next_forward"
      (a view of an internal buffer that the next forward overwrites; the engine copies
      it into `out=` when given).
    - `lossy_values`: structure values that change numerics, e.g.
      `{"p2p_quant": {"fp8_1x32"}}`; `comm_precision` filters them out.
      `default_lossy`: the subset the kernel uses by default today (allowed under
      `comm_precision="kernel_default"`).

    Methods. Every segment implements `reads` and `produces` and declares `name`,
    `covers` and `capabilities`; stages implement `build`, sort and quant segments
    `run`, which the executor calls. The rest have defaults:

    - Feasibility: `reject_reason` (from the `Cap`s), which the resolver calls;
      `accepts` is `reject_reason(p) is None`. A segment that overrides either cannot
      be in the last structure of a ranking region (`ranking._shape_free`).
    - Boundaries: `reads(p)`, the `Buf`s it reads, and `produces(p, params)`, the `Buf`
      it hands on. The resolver chains segments whose formats match.
    - Tuned rows: `owns` (from `prefixes`), `check_row` (the row is discarded, as legacy
      discards it, with an optional warning), `declines_row` (migration only: the row
      goes back to the legacy code), `row_params` (what the row sets besides the kernel
      names).
    - Without a tuned row: `default_kernels` (a stage 1 naming both stages' kernels),
      `default_params` (not called yet).
    - Stage 2 only: `stage2_kernel` (a rename of the tuned name, e.g. fp4 to fp8).
    - Launch: `build(p, params)` binds the kernel wrapper, `metadata_fields` the
      `MOEMetadata` fields it sets, `call_args(c)` the arguments taken from the call.

    Read off specific segments, not declared here: `flat` (`sort.flat`; the executor's
    ASM mode), `default_xbf16` (`asm.fmoe_1stage`; the resolver), and
    `transpose_a1_scale`, `out_buffer` and `split_out` (stage 1s that write GEMM2's
    input; the executor).
    """

    name: str = ""
    prefixes: tuple = ()
    covers: tuple = (Op.SORT, Op.SORT)
    capabilities: tuple = ()
    reach: str = "ranked"
    structure_params: frozenset = frozenset()
    rank_local_params: bool = False
    stateful: bool = False
    zero_tokens_ok: bool = True
    hints: frozenset = frozenset()
    output_lifetime: str = "caller"
    lossy_values: FrozenMap = FrozenMap()
    default_lossy: FrozenMap = FrozenMap()

    def reject_reason(self, p: MoeProblem) -> str | None:
        """None if some capability covers `p`, else the reason of the closest one."""
        if not self.capabilities:
            return "no capabilities declared"
        reasons = []
        for cap in self.capabilities:
            reason = cap.reject_reason(p)
            if reason is None:
                return None
            reasons.append(reason)
        return reasons[0] if len(reasons) == 1 else "; ".join(dict.fromkeys(reasons))

    def accepts(self, p: MoeProblem) -> bool:
        return self.reject_reason(p) is None

    def default_params(self, p: MoeProblem) -> Params:
        return Params()

    def owns(self, kernel) -> bool:
        """Whether this segment runs the parsed tuned kernel name `kernel`."""
        return kernel is not None and kernel.family in self.prefixes

    def reads(self, p: MoeProblem) -> tuple:
        """The boundary `Buf`s this segment reads, each from the last segment before it
        that produced that kind. `fmt=None` reads any format; `block_m` is a plan
        parameter, shared by every segment, so it is not compared."""
        raise NotImplementedError(self.name)

    def produces(self, p: MoeProblem, params: Params):
        """The `Buf` this segment hands to the next one."""
        raise NotImplementedError(self.name)

    def check_row(self, p: MoeProblem, row, kernel_name1="") -> RowRejection | None:
        """Whether this segment can run tuned `row` for `p` (None: it can)."""
        return None

    def declines_row(self, p: MoeProblem, row) -> str | None:
        """Why tuned `row` goes back to the legacy code for `p`, though owned, or None."""
        return None

    def default_kernels(self, p: MoeProblem) -> DefaultKernels | None:
        """For a stage 1 without a tuned row: the kernels it names for both stages (with
        block_m, ksplit and the warning logged), or None to keep the defaults."""
        return None

    def stage2_kernel(self, p: MoeProblem, kernel_name1, kernel_name2) -> str:
        """For a stage 2: the kernel name it runs for these tuned names."""
        return kernel_name2

    def row_params(self, p: MoeProblem, row) -> dict:
        """Parameters tuned `row` gives this segment beyond the kernel names."""
        return {}

    def metadata_fields(self, p: MoeProblem, params: Params) -> dict:
        """`MOEMetadata` fields this segment sets (`block_m` replaces the plan's)."""
        return {}

    def build(self, p: MoeProblem, params: Params):
        raise NotImplementedError(self.name)

    def call_args(self, c) -> dict:
        """For a stage: arguments from the call (`executor.Call`) its launch adds to the
        ones `build` binds, before the executor adds its own (`run_mxfp4`)."""
        return {}

    def __repr__(self):
        return f"<segment {self.name}>"
