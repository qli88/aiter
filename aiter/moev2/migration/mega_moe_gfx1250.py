# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mega_moe_gfx1250`: `MegaMoEGfx1250.forward` (gfx1250, EP).

The plan is `comm.dispatch_gfx1250 | grouped_gfx1250 (p2p_scatter) | comm.combine_gfx1250`
over the instance's pipeline. `forward` runs it as legacy does, without the nested
`fused_moe` call and without `set_flydsl_dispatch_context`: the grouped launch gets the
scatter context (boundary B) and the FlyDSL dispatch context as arguments.

`claims`: a gfx1250 device (todo 5.26), weights of the pipeline's shapes, and a nested
call legacy's grouped branch serves (`grouped_gemm_gfx1250_a8w4` returns a result; when it
returns None legacy falls through to the 2-stage dispatch, todo 5.40), decided before the
dispatch from the shapes and formats the dispatch hands on. Everything else stays legacy,
which raises for invalid arguments as before.

`MegaMoEGfx1250Template` serves the same plans to `MoeEngine`, whose dispatch segment then
creates the pipeline at `prepare`.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import overrides
from ..dispatch.plan import MoePlan, SegmentChoice
from ..engine import (
    BucketPlan,
    PlanSource,
    _canonical,
    _fused_moe_defaults,
    bucket_range,
)
from ..execute import executor
from ..policy import quant_remap
from ..segments.base import Params
from ..segments.comm_gfx1250 import (
    CommCombineGfx1250,
    CommDispatchGfx1250,
    MegaSpec,
    weight_shapes,
)
from ..segments.grouped_gfx1250 import COMBINE_FORMATS, GROUPED_P2P
from . import local

NAME = "mega_moe_gfx1250"

# `MegaMoEGfx1250.forward`'s arguments besides the instance (`mega`).
FORWARD_ARGS = (
    "hidden_states",
    "topk_weights",
    "topk_ids",
    "w1",
    "w2",
    "w1_scale",
    "w2_scale",
    "bias1",
    "bias2",
    "a1_scale",
    "a2_scale",
    "recv_token_bound",
    "next_topk_ids",
    "next_recv_token_bound",
    "combine_quant",
)


def _grouped_call(mega, call, hidden_states, topk_ids, topk_weight, a1_scale, nlt):
    """The `fused_moe_` arguments of legacy's nested `fused_moe` call (the scatter
    context goes to the launch instead)."""
    extra = {}
    if mega.activation == ActivationType.Situv2:
        extra = {"beta": mega.situ_beta, "linear_beta": mega.situ_linear_beta}
    return api.op_call(
        {
            **_fused_moe_defaults(),
            "hidden_states": hidden_states,
            "w1": call["w1"],
            "w2": call["w2"],
            "topk_weight": topk_weight,
            "topk_ids": topk_ids,
            "expert_mask": mega.expert_mask,
            "activation": mega.activation,
            "gate_mode": mega.gate_mode,
            "quant_type": mega.quant_type,
            "w1_scale": call["w1_scale"],
            "w2_scale": call["w2_scale"],
            "a1_scale": a1_scale,
            "a2_scale": call["a2_scale"],
            "bias1": call["bias1"],
            "bias2": call["bias2"],
            "hidden_pad": mega.hidden_pad,
            "intermediate_pad": mega.intermediate_pad,
            "dtype": dtypes.bf16,
            "num_local_tokens": nlt,
            "swiglu_limit": mega.swiglu_limit,
            **extra,
        }
    )


def grouped_unsupported(mega, call) -> str | None:
    """Why legacy's grouped branch would not serve the nested call, or None; decided
    before the dispatch, on meta tensors of the shapes and dtypes the dispatch hands
    on (the grouped checks read no values)."""
    config = mega._config
    wire = config.dispatch_wire_spec
    compact = mega._compact_plan
    rows = mega._compact_cap if compact else config.max_recv
    bound = call["recv_token_bound"]
    if bound is not None and not compact and 0 < int(bound) <= rows:
        rows = int(bound)

    def meta(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    width = config.dispatch_token_nbytes // wire.recv_dtype.itemsize
    a1_scale = call["a1_scale"]
    if config.is_quant_dispatch_wire:
        a1_scale = meta(rows, mega._compact_scale_row, dtype=torch.uint8)
    route_rows = 1 if compact else rows
    op = _grouped_call(
        mega,
        call,
        meta(rows, width, dtype=wire.recv_dtype),
        meta(route_rows, mega.topk, dtype=torch.int32),
        meta(route_rows, mega.topk, dtype=torch.float32),
        a1_scale,
        None if compact else meta(1, dtype=torch.int32),
    )
    return GROUPED_P2P.unsupported(
        executor.prologue(op, quant_remap), local.activation_dtype(op)
    )


def claims(call) -> bool:
    if get_gfx_runtime() != "gfx1250":
        return False
    mega = call["mega"]
    w1, w2 = call["w1"], call["w2"]
    if not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor):
        return False
    if (tuple(w1.shape), tuple(w2.shape)) != weight_shapes(mega):
        return False
    return grouped_unsupported(mega, call) is None


def forward(
    dispatch: CommDispatchGfx1250, combine: CommCombineGfx1250, call, claimed=False
):
    """`MegaMoEGfx1250.forward` over the segments, in its order (`call`: its arguments
    by name; `claimed`: `claims` accepted it)."""
    mega = dispatch.mega
    bits = combine.bits(call["combine_quant"])
    dispatch.check(call)
    reason = None if claimed else grouped_unsupported(mega, call)
    if reason is not None:
        raise NotImplementedError(
            f"{NAME}: grouped_gfx1250 does not serve this call ({reason})"
        )
    hidden_states = call["hidden_states"]
    recv_token_bound = call["recv_token_bound"]
    dispatch.begin(int(hidden_states.shape[0]), recv_token_bound)
    d = dispatch.dispatch(hidden_states, call["topk_weights"], call["topk_ids"])
    d = dispatch.tokens(d, recv_token_bound, call["a1_scale"])
    scatter = combine.scatter(d.routing, bits)
    topk_ids, topk_weight, num_local_tokens = dispatch.routes(d)
    dispatch.prefetch(call["next_topk_ids"], call["next_recv_token_bound"])
    op = _grouped_call(
        mega, call, d.recv_x, topk_ids, topk_weight, d.a1_scale, num_local_tokens
    )
    out = GROUPED_P2P.launch(
        executor.prologue(op, quant_remap),
        local.activation_dtype(op),
        stage2_scatter=scatter,
        dispatch_context=dispatch.context(scatter),
    )
    if out is None:
        raise RuntimeError(f"{NAME}: grouped_gfx1250 declined after the dispatch")
    return combine.combine(d.routing, bits)


def run(call):
    dispatch = CommDispatchGfx1250.adopt(call["mega"])
    return forward(dispatch, CommCombineGfx1250(dispatch), call, claimed=True)


# AITER_MOEV2=mega_moe_gfx1250 until its multi-GPU differential passes on gfx1250.
PATH = paths.Path(NAME, claims, run, entry="mega_gfx1250")
paths.register(PATH)


def _boundaries(dispatch, dispatch_params, grouped_params, combine, combine_params):
    """Boundaries A and B of the plan, checked against what the next segment reads."""
    a = dispatch.produces(None, dispatch_params)
    b = GROUPED_P2P.produces(None, grouped_params)
    for reader, params, have in (
        (GROUPED_P2P, grouped_params, a),
        (combine, combine_params, b),
    ):
        for need in reader.reads_with(None, params):
            problem = need.mismatch(have)
            if problem is not None:
                raise ValueError(f"{reader.name} {problem}")
    return a, b


@dataclass
class MegaMoEGfx1250Template(PlanSource):
    """`MegaMoEGfx1250` for `MoeEngine` (gfx1250, EP). `comm=`: the group the engine
    agrees over (a torch group or `GroupCoordinator`); `communicator`: the cco
    communicator the pipeline's arena is allocated on. The other fields are the
    constructor's; `dispatch_backend` and `dispatch_wire` default to `MEGA_DISPATCH`
    and `MEGA_DISPATCH_WIRE` (`overrides.py`), as there.

    Per bucket: `combine_quant`, the payload of the bucket's steps (one value for every
    bucket, or `{bucket: value}` with "none" elsewhere; legacy's per-step argument), and
    `recv_token_bound` (`{bucket: rows}`; elsewhere unbound, legacy's default). The
    pipeline builds a reduce for bf16 and for the one quantized format the buckets
    use, else for `MEGA_COMBINE_QUANT`'s, as the constructor does. Weights per call
    (`MoeWeights`: w1, w2, w1_scale, w2_scale, bias1, bias2); `side_inputs`: of
    a1_scale and a2_scale; hint `next_topk_ids` (the next layer's plan, with this
    bucket's bound). The result is a view of the pipeline's output buffer, which the
    next forward overwrites. `capture()` compiles the compact plans of the captured
    buckets (it cannot run a forward: that is a collective, and the weights are the
    caller's)."""

    communicator: object
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
    dispatch_backend: str | None = None
    dispatch_wire: str | None = None
    stage1_fused: bool = False
    combine_quant: str | Mapping = "none"
    recv_token_bound: Mapping = field(default_factory=dict)
    side_inputs: frozenset = frozenset()

    serves_parallel = True
    allowed_side_inputs = frozenset({"a1_scale", "a2_scale"})

    def __post_init__(self):
        self.side_inputs = frozenset(self.side_inputs)
        unknown = sorted(self.side_inputs - self.allowed_side_inputs)
        if unknown:
            raise ValueError(
                f"side_inputs {unknown} are not per-call inputs "
                f"({sorted(self.allowed_side_inputs)})"
            )
        if self.world_size <= 0 or self.experts % self.world_size:
            raise ValueError(
                f"experts={self.experts} must be divisible by "
                f"world_size={self.world_size}"
            )
        formats = (
            set(self.combine_quant.values())
            if isinstance(self.combine_quant, Mapping)
            else {self.combine_quant}
        )
        unknown = sorted(formats - set(COMBINE_FORMATS))
        if unknown:
            raise ValueError(
                f"combine_quant {unknown} not in {sorted(COMBINE_FORMATS)}"
            )
        quantized = sorted(formats - {"none"})
        if len(quantized) > 1:
            raise ValueError(
                f"the pipeline builds one quantized combine format; the buckets "
                f"use {quantized}"
            )
        self._built = quantized[0] if quantized else overrides.mega_combine_quant()
        self._dispatch = None

    def spec(self) -> MegaSpec:
        return MegaSpec(
            communicator=self.communicator,
            world_size=self.world_size,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            experts=self.experts,
            topk=self.topk,
            max_tokens_per_rank=self.max_tokens_per_rank,
            activation=self.activation,
            gate_mode=self.gate_mode,
            quant_type=self.quant_type,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            dispatch_backend=(
                self.dispatch_backend
                if self.dispatch_backend is not None
                else overrides.mega_dispatch_backend()
            ),
            dispatch_wire=(
                self.dispatch_wire
                if self.dispatch_wire is not None
                else overrides.mega_dispatch_wire()
            ),
            stage1_fused=bool(self.stage1_fused),
            combine_quant=self._built,
        )

    def _segments(self) -> tuple:
        if self._dispatch is None:
            self._dispatch = CommDispatchGfx1250.for_spec(self.spec())
            self._combine = CommCombineGfx1250(self._dispatch)
        return self._dispatch, self._combine

    def _combine_quant(self, bucket: int) -> str:
        if isinstance(self.combine_quant, Mapping):
            return self.combine_quant.get(bucket, "none")
        return self.combine_quant

    def resolve(self, bucket: int, graph_capture: bool) -> BucketPlan:
        dispatch, combine = self._segments()
        combine_quant = self._combine_quant(bucket)
        bound = self.recv_token_bound.get(bucket)
        dispatch_params = dispatch.params()
        grouped_params = Params(recv_token_bound=bound, combine_quant=combine_quant)
        combine_params = Params(combine_quant=combine_quant)
        plan = MoePlan(
            (
                SegmentChoice(dispatch, dispatch_params),
                SegmentChoice(GROUPED_P2P, grouped_params),
                SegmentChoice(combine, combine_params),
            ),
            "default",
            boundaries=_boundaries(
                dispatch, dispatch_params, grouped_params, combine, combine_params
            ),
        )
        spec = {k: v for k, v in vars(dispatch.spec).items() if k != "communicator"}
        for name in ("situ_beta", "situ_linear_beta"):
            if spec[name] is not None:
                spec[name] = float(spec[name])
        return BucketPlan(
            bucket,
            plan.describe(),
            functools.partial(self._run, bound, combine_quant),
            identity=_canonical(spec),
            warmup=(
                functools.partial(self._warmup, bucket, bound)
                if graph_capture and dispatch.spec.stage1_fused
                else None
            ),
            stateful=(dispatch,),
            side_inputs=self.side_inputs,
            hints=dispatch.hints,
            output_lifetime=combine.output_lifetime,
        )

    def _warmup(self, bucket, bound):
        """The compact plans the bucket's steps compile (`warmup_compact_plan`): its
        bound's, or without one each count's own (forward infers it per call)."""
        mega = self._dispatch.mega
        if bound is not None:
            mega.warmup_compact_plan(recv_token_bound=bound)
            return
        lowest, highest = bucket_range(bucket)
        if highest is None:
            highest = self.max_tokens_per_rank
        for tokens in range(lowest, min(highest, self.max_tokens_per_rank) + 1):
            mega.warmup_compact_plan(token_count=tokens)

    def _run(
        self, bound, combine_quant, x, topk_ids, topk_weights, weights, side, hints, out
    ):
        dispatch, combine = self._segments()
        call = {
            "mega": dispatch.mega,
            "hidden_states": x,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "w1": weights.w1,
            "w2": weights.w2,
            "w1_scale": weights.w1_scale,
            "w2_scale": weights.w2_scale,
            "bias1": weights.bias1,
            "bias2": weights.bias2,
            "a1_scale": side.get("a1_scale"),
            "a2_scale": side.get("a2_scale"),
            "recv_token_bound": bound,
            "next_topk_ids": hints.get("next_topk_ids"),
            "next_recv_token_bound": bound,
            "combine_quant": combine_quant,
        }
        return forward(dispatch, combine, call)
