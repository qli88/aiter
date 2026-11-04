# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segments `comm.dispatch_gfx1250` and `comm.combine_gfx1250`: the communication of
`MegaMoEGfx1250` (gfx1250, EP), around the P2P-scatter `grouped_gfx1250`.

The pipeline stays in `aiter.ops.flydsl.kernels.mega_moe_gfx1250`: one `MegaMoEGfx1250`
instance (its cco symmetric arena, dispatch variants, compact-plan slots and combine
reducers), owned by the dispatch segment, which adopts it or creates it at `prepare`.
All three segments of the plan work in that one arena: dispatch writes the received
tokens there (boundary A, `tokens`), GEMM2 scatters into its combine slots (boundary B,
`p2p_scatter`, described by the `Stage2ScatterContext` the combine segment hands out)
and combine reduces them into this rank's tokens. The combine segment has no state of
its own.

The methods are `MegaMoEGfx1250.forward` statement for statement, called in its order by
path `mega_moe_gfx1250` (`migration.mega_moe_gfx1250.forward`): `CommCombineGfx1250.bits`,
`CommDispatchGfx1250.check`, `begin`, `dispatch`, `tokens`, `CommCombineGfx1250.scatter`,
`CommDispatchGfx1250.routes`, `prefetch`, the grouped launch, `CommCombineGfx1250.combine`.
"""

from dataclasses import dataclass

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import FrozenMap, ScaleSpec
from .base import Cap, MemoryNeed, Params, Segment
from .grouped_gfx1250 import disabled as grouped_disabled
from .mxfp4 import mxfp4, mxfp8

# The dispatch wire's formats, by `dispatch_wire`.
WIRE_FORMATS = {"bf16": ScaleSpec(dtypes.bf16), "fp8": mxfp8(), "fp4": mxfp4()}


def _mega():
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import mega_moe

    return mega_moe


def token_layout(stage1_fused: bool) -> str:
    """The `layout` structure parameter: boundary A's rows grouped per expert by the
    compact plan, or one per received token (`Buf` rows `received`)."""
    return "compact_plan" if stage1_fused else "token_major"


@dataclass(frozen=True)
class MegaSpec:
    """A `MegaMoEGfx1250` pipeline: its constructor arguments besides the group's rank
    (`world_size` is checked against the engine's group). The dispatch and combine knobs
    are resolved (no environment default left)."""

    communicator: object  # the cco communicator the arena is allocated on
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tokens_per_rank: int
    activation: ActivationType = ActivationType.Silu
    gate_mode: int = GateMode.INTERLEAVE.value
    quant_type: QuantType = QuantType.per_1x32
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float = 0.0
    situ_beta: torch.Tensor | None = None
    situ_linear_beta: torch.Tensor | None = None
    dispatch_backend: str = "flydsl"
    dispatch_wire: str = "bf16"
    stage1_fused: bool = False
    combine_quant: str = "none"

    def config(self, rank: int = 0):
        """The pipeline's `MegaMoEConfig` (what sizes its arena)."""
        return _mega().MegaMoEConfig(
            rank=rank,
            world_size=self.world_size,
            hidden_dim=self.model_dim,
            max_tokens_per_rank=self.max_tokens_per_rank,
            experts_per_rank=self.experts // self.world_size,
            topk=self.topk,
            dispatch_backend=self.dispatch_backend,
            dispatch_wire=self.dispatch_wire,
            stage1_fused=self.stage1_fused,
            combine_quant=self.combine_quant,
        )


@dataclass
class Dispatched:
    """What `dispatch` returns: boundary A (the received tokens and their routes, the
    received count, the e8m0 rows of a quantizing wire) and the routing combine
    reduces with."""

    recv_x: torch.Tensor
    recv_weights: torch.Tensor
    recv_ids: torch.Tensor
    total_recv: torch.Tensor
    routing: object
    a1_scale: torch.Tensor | None = None


class CommDispatchGfx1250(Segment):
    """The FlyDSL (TDM) or MORI dispatch of a `MegaMoEGfx1250` pipeline, which this
    segment owns: `adopt(mega)` wraps an instance (nothing to prepare), `for_spec(spec)`
    creates one at `prepare(ctx)` on the engine's group. Structure: `dispatch_backend`
    (the kernels), `dispatch_wire` (A's format; fp8 and fp4 are lossy) and `layout`
    (A's row grouping, from `stage1_fused`). Hint: `next_topk_ids`, which starts the
    next layer's compact plan after this dispatch. Legacy has no health or reset, so
    neither has the segment; `release()` closes the pipeline."""

    name = "comm.dispatch_gfx1250"
    covers = (Op.COMM_IN, Op.COMM_IN)
    capabilities = (
        Cap(
            arch=frozenset({"gfx1250"}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            gate_mode=frozenset({"interleave"}),
            g1u1=frozenset({True}),
            parallel=True,
        ),
    )
    structure_params = frozenset({"dispatch_backend", "dispatch_wire", "layout"})
    stateful = True
    hints = frozenset({"next_topk_ids"})
    lossy_values = FrozenMap({"dispatch_wire": frozenset({"fp8", "fp4"})})

    def __init__(self, mega=None, spec: MegaSpec | None = None):
        self.mega = mega
        self.spec = spec

    @classmethod
    def adopt(cls, mega) -> "CommDispatchGfx1250":
        return cls(mega=mega)

    @classmethod
    def for_spec(cls, spec: MegaSpec) -> "CommDispatchGfx1250":
        return cls(spec=spec)

    def params(self):
        """The structure the pipeline was (or will be) built with."""
        if self.mega is not None:
            config = self.mega._config
            backend, wire, fused = (
                config.dispatch_backend,
                config.dispatch_wire,
                config.stage1_fused,
            )
        else:
            spec = self.spec
            backend, wire, fused = (
                spec.dispatch_backend,
                spec.dispatch_wire,
                spec.stage1_fused,
            )
        return Params(
            dispatch_backend=backend, dispatch_wire=wire, layout=token_layout(fused)
        )

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        rows = "received" if params.layout == "token_major" else params.layout
        return Buf("tokens", WIRE_FORMATS[params.dispatch_wire], rows=rows)

    def build(self, p, params):
        return self.dispatch

    def workspace_bytes(self) -> MemoryNeed:
        if self.mega is not None:
            return MemoryNeed()
        spec, mega = self.spec, _mega()
        layout = mega.pipeline_layout(
            spec.config(),
            activation=ActivationType(spec.activation),
            quant_type=QuantType(spec.quant_type),
            inter_dim=spec.inter_dim,
        )
        return MemoryNeed(symmetric=mega.SymmetricArena.layout(layout.regions)[2])

    def prepare(self, ctx) -> None:
        if self.mega is not None:
            return
        spec = self.spec
        if ctx.world_size != spec.world_size:
            raise ValueError(
                f"the pipeline is for {spec.world_size} ranks; the group has "
                f"{ctx.world_size}"
            )
        self.mega = _mega().MegaMoEGfx1250(
            communicator=spec.communicator,
            rank=ctx.rank,
            world_size=ctx.world_size,
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            topk=spec.topk,
            max_tokens_per_rank=spec.max_tokens_per_rank,
            activation=spec.activation,
            gate_mode=spec.gate_mode,
            quant_type=spec.quant_type,
            hidden_pad=spec.hidden_pad,
            intermediate_pad=spec.intermediate_pad,
            swiglu_limit=spec.swiglu_limit,
            situ_beta=spec.situ_beta,
            situ_linear_beta=spec.situ_linear_beta,
            dispatch_backend=spec.dispatch_backend,
            dispatch_wire=spec.dispatch_wire,
            stage1_fused=spec.stage1_fused,
            combine_quant=spec.combine_quant,
        )

    def release(self) -> None:
        if self.mega is not None:
            self.mega.close()

    # `MegaMoEGfx1250.forward`, in its order.

    def check(self, call) -> None:
        """`forward`'s argument checks (after the combine format's)."""
        mega = self.mega
        hidden_states = call["hidden_states"]
        topk_weights, topk_ids = call["topk_weights"], call["topk_ids"]
        w1, w2 = call["w1"], call["w2"]
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous bfloat16")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if grouped_disabled():
            raise RuntimeError(
                "MegaMoE fused stage2 scatter requires the grouped A8W4 kernel; "
                "AITER_DISABLE_GROUPED_A8W4=1 is not supported"
            )
        if call["w1_scale"] is None or call["w2_scale"] is None:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires both w1_scale and w2_scale"
            )
        if w1.dtype not in (torch.uint8, dtypes.fp4x2) or w2.dtype not in (
            torch.uint8,
            dtypes.fp4x2,
        ):
            raise ValueError(
                "MegaMoE fused stage2 scatter requires MXFP4 w1/w2 weights "
                f"(uint8 or fp4x2), got {w1.dtype} and {w2.dtype}"
            )
        expected_w1_shape, expected_w2_shape = weight_shapes(mega)
        if tuple(w1.shape) != expected_w1_shape:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires interleaved G1U1 w1 with "
                f"shape {expected_w1_shape}, got {tuple(w1.shape)}"
            )
        if tuple(w2.shape) != expected_w2_shape:
            raise ValueError(
                f"MegaMoE fused stage2 scatter requires w2 shape "
                f"{expected_w2_shape}, got {tuple(w2.shape)}"
            )
        token_count = int(hidden_states.shape[0])
        if token_count > mega.max_tokens_per_rank:
            raise ValueError(
                f"tokens={token_count} exceeds max_tokens_per_rank="
                f"{mega.max_tokens_per_rank}"
            )
        expected_shape = (token_count, mega.topk)
        if tuple(topk_weights.shape) != expected_shape:
            raise ValueError(
                f"topk_weights must have shape {expected_shape}, "
                f"got {tuple(topk_weights.shape)}"
            )
        if tuple(topk_ids.shape) != expected_shape:
            raise ValueError(
                f"topk_ids must have shape {expected_shape}, "
                f"got {tuple(topk_ids.shape)}"
            )

    def begin(self, token_count: int, recv_token_bound: int | None) -> None:
        """The compact plan's step geometry (compact layout only)."""
        mega = self.mega
        if mega._compact_plan:
            # Compact cannot slice recv_x -- the rows are grouped per expert, not
            # token-major -- so the bound is spent on the geometry instead: it
            # picks the plan's alignment, the GEMM's tile and the GEMM's grid.
            mega._begin_compact_step(
                mega._compact_recv_bound(token_count, recv_token_bound)
            )

    def dispatch(self, hidden_states, topk_weights, topk_ids) -> Dispatched:
        """The dispatch (collective)."""
        return Dispatched(*self.mega._dispatch(hidden_states, topk_weights, topk_ids))

    def tokens(self, d: Dispatched, recv_token_bound, a1_scale) -> Dispatched:
        """Boundary A: the received rows cut to `recv_token_bound` (token-major), and
        the e8m0 rows of a quantizing wire as `a1_scale`."""
        mega = self.mega
        compact = mega._compact_plan
        recv_x, recv_weights, recv_ids = d.recv_x, d.recv_weights, d.recv_ids
        if recv_token_bound is not None and not compact:
            bound = int(recv_token_bound)
            if bound <= 0 or bound > recv_x.shape[0]:
                raise ValueError(
                    f"recv_token_bound must be in (0, {recv_x.shape[0]}], "
                    f"got {recv_token_bound}"
                )
            recv_x = recv_x[:bound]
            recv_weights = recv_weights[:bound]
            recv_ids = recv_ids[:bound]
        if mega._config.is_quant_dispatch_wire:
            assert a1_scale is None, (
                "a1_scale is produced by the quantizing dispatch wire itself; a "
                "caller-supplied one would be silently discarded"
            )
            a1_scale = mega._recv_dispatch_scales()
            if recv_token_bound is not None and not compact:
                a1_scale = a1_scale[: int(recv_token_bound)]
        return Dispatched(
            recv_x, recv_weights, recv_ids, d.total_recv, d.routing, a1_scale
        )

    def routes(self, d: Dispatched) -> tuple:
        """`(topk_ids, topk_weight, num_local_tokens)` the grouped GEMM gets: the
        received routes and count, or the compact plan's placeholders."""
        mega = self.mega
        if mega._compact_plan:
            return mega._compact_dummy_ids, mega._compact_dummy_wts, None
        return d.recv_ids, d.recv_weights, d.total_recv

    def prefetch(self, next_topk_ids, next_recv_token_bound) -> None:
        """Hint `next_topk_ids`: start the next layer's compact plan on the side stream
        (compact layout only)."""
        mega = self.mega
        if (
            mega._compact_plan
            and next_topk_ids is not None
            and int(next_topk_ids.shape[0]) > 0
        ):
            # Dispatch of this layer is done, so tok_map[slot] is free. GEMM
            # still reads psum/masked_m of this slot; the next plan writes the
            # other slot and overlaps the expert GEMM (and later combine/RMS).
            mega._prefetch_next_compact_plan(next_topk_ids, next_recv_token_bound)

    def context(self, scatter):
        """The dispatch context the grouped launch gets: the FlyDSL dispatch's (its
        compact plan, its direct EP mask), none for MORI."""
        return scatter if self.mega._config.dispatch_backend == "flydsl" else None


def weight_shapes(mega) -> tuple:
    """The `(w1, w2)` shapes the pipeline takes: interleaved g1u1 MXFP4, local experts."""
    return (
        (mega.experts_per_rank, 2 * mega.inter_dim, mega.model_dim // 2),
        (mega.experts_per_rank, mega.model_dim, mega.inter_dim // 2),
    )


class CommCombineGfx1250(Segment):
    """The combine of the pipeline `dispatch` owns: the cross-device barrier, then the
    reduce of the slots GEMM2 scattered (boundary B) into this rank's tokens. Structure:
    `combine_quant`, B's format (mxfp8 and mxfp4 are lossy), which the pipeline must
    have built a reduce for. The result is a view of the pipeline's output buffer,
    overwritten by its next forward."""

    name = "comm.combine_gfx1250"
    covers = (Op.REDUCE, Op.COMM_OUT)
    capabilities = (
        Cap(
            arch=frozenset({"gfx1250"}),
            out_dtype=frozenset({dtypes.bf16}),
            features=frozenset({"scatter"}),
            parallel=True,
        ),
    )
    structure_params = frozenset({"combine_quant"})
    output_lifetime = "until_next_forward"
    lossy_values = FrozenMap({"combine_quant": frozenset({"mxfp8", "mxfp4"})})

    def __init__(self, dispatch: CommDispatchGfx1250):
        self.dispatch = dispatch

    def reads(self, p):
        return (Buf("p2p_scatter", None),)

    def produces(self, p, params):
        return Buf("out", ScaleSpec(dtypes.bf16))

    def build(self, p, params):
        return self.combine

    # `MegaMoEGfx1250.forward`, in its order.

    def bits(self, combine_quant) -> int:
        """The payload width of `combine_quant` (None: bf16), checked against what the
        pipeline built."""
        return self.dispatch.mega._resolve_combine_quant(combine_quant)

    def scatter(self, routing, bits: int):
        """Boundary B: the `Stage2ScatterContext` GEMM2 writes the combine slots by."""
        return self.dispatch.mega._scatter_context(routing, bits)

    def combine(self, routing, bits: int) -> torch.Tensor:
        """The combine (collective)."""
        return self.dispatch.mega._combine(routing, bits)
