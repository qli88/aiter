# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `mega_moe_v2`: `MegaMoEV2.forward` and `forward_prequant` (gfx950, EP).

The plan is one segment, `mega_moe_v2`, over the instance: `forward` runs its steps in
legacy's order, with the instance's own kernels and bundle entries. `claims`: a gfx950
device; elsewhere the call stays legacy.

`MegaMoEV2Template` serves the segment to `MoeEngine` with the instances vLLM builds
(`capacities`) and vLLM's per-layer handling: the shared-expert route, then the layer's
weights bound, then the forward.
"""

import functools
from dataclasses import dataclass

from aiter.jit.utils.chip_info import get_gfx_runtime

from .. import paths
from ..dispatch.plan import MoePlan, SegmentChoice
from ..engine import BucketPlan, MoeWeights, PlanSource, _canonical, _weights_signature
from ..segments.mega_moe_v2 import MegaMoEV2Segment, MegaV2Spec, _config

NAME = "mega_moe_v2"


def claims(call) -> bool:
    return get_gfx_runtime() == "gfx950"


def forward(segment: MegaMoEV2Segment, call):
    """`MegaMoEV2.forward` (or `forward_prequant`) over the segment, in its order
    (`call`: the hook's arguments by name)."""
    segment.check(call)
    x, scales = call["x"], call["scales"]
    wts, topk_ids, stream = call["wts"], call["topk_ids"], call["stream"]
    run_tokens = int(x.shape[0])
    config, prepared = None, False
    if not call["prequant"]:
        if segment.fixed_slot():
            x, scales = segment.quantize(x)
        else:
            config = segment.select(run_tokens)
            x, scales = segment.prepare_compact(x, topk_ids, run_tokens, config, stream)
            prepared = True
    if config is None:
        config = segment.select(run_tokens)
    mask = segment.combine_mask(call["mask_invalid_slots"])
    segment.stage1(x, wts, scales, topk_ids, stream, config, prepared)
    return segment.stage2(
        run_tokens,
        stream,
        call["slice_output"],
        config,
        topk_ids if mask else None,
    )


def run(call):
    return forward(MegaMoEV2Segment.adopt(call["mega"]), call)


# AITER_MOEV2=mega_moe_v2 until its EP8 differential passes and it is asked to graduate.
PATH = paths.Path(NAME, claims, run, entry="mega_v2")
paths.register(PATH)


def _prev_power_of_2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1) if n > 0 else 0


@dataclass
class MegaMoEV2Template(PlanSource):
    """`MegaMoEV2` for `MoeEngine` (gfx950, EP). `weights`: one layer's weights (the
    constructor needs real ones; each forward binds its own, of the same shapes and
    dtypes). `experts` and `topk`: the routed ones; with `fuse_shared_expert` each rank
    has one more local expert (its shared slot, the last) and each token one more route,
    appended by the engine (`routing.shared_route`). `max_tokens_per_rank`: the largest
    `StepInfo.max_tokens_across_ranks` (a power of two, vLLM's
    `next_power_of_2(max_num_batched_tokens)`). The other fields are the constructor's.
    `mask_invalid_slots`: the forward argument (None: AITER_MEGA_COMBINE_MASK).

    Instances (`capacities`), as vLLM builds them: one at `max_tokens_per_rank`, plus a
    fixed-slot one at the largest power of two up to AITER_MEGA_FIXED_SLOT_MAX_MTPR when
    that is smaller and its bundle plan is fixed-slot. A bucket runs on the smallest
    instance that holds it. The result is a view of the instance's output buffer, which
    the next forward overwrites. `capture()` loads each captured instance's stage 1 and
    stage 2 bundles (legacy's `preload_aot_bundles`, no GPU dispatch) and compiles the
    shared route; legacy's preload skips fixed-slot stage 1, which still compiles at the
    first forward (todo 2.25). The weight layout is the one vLLM shuffles to
    (`shuffle_weight_a16w4(16)`); `weight_layout()` does not name it yet (4.10)."""

    weights: MoeWeights
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tokens_per_rank: int
    quant: str = "a8w4"
    mega_scheme: str = "fixedslot"
    stage2_p2p_quant: str = "auto"
    swiglu_limit: float = 0.0
    fanout_masks: tuple = ()
    act: str = "silu"
    situ_beta: float = 1.0
    situ_linear_beta: float = 1.0
    fuse_shared_expert: bool = False
    mask_invalid_slots: bool | None = None

    serves_parallel = True

    def __post_init__(self):
        if self.world_size <= 0 or self.experts % self.world_size:
            raise ValueError(
                f"experts={self.experts} must be divisible by "
                f"world_size={self.world_size}"
            )
        m = self.max_tokens_per_rank
        if m <= 0 or m & (m - 1):
            raise ValueError(f"max_tokens_per_rank={m} must be a power of two")
        if self.weights.w1_scale is None or self.weights.w2_scale is None:
            raise ValueError("MegaMoEV2 needs w1_scale and w2_scale")
        self.fanout_masks = tuple(int(mask) for mask in self.fanout_masks)
        self.routed_per_rank = self.experts // self.world_size
        self._signature = _weights_signature(self.weights)
        self._segments = {}
        self.capacities = self._capacities()

    def spec(self, capacity: int) -> MegaV2Spec:
        shared = int(bool(self.fuse_shared_expert))
        return MegaV2Spec(
            world_size=self.world_size,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            experts=self.experts + shared * self.world_size,
            topk=self.topk + shared,
            quant=self.quant,
            max_tok_per_rank=capacity,
            mega_scheme=self.mega_scheme,
            stage2_p2p_quant=self.stage2_p2p_quant,
            swiglu_limit=float(self.swiglu_limit),
            fanout_masks=self.fanout_masks,
            act=self.act,
            situ_beta=float(self.situ_beta),
            situ_linear_beta=float(self.situ_linear_beta),
        )

    def _capacities(self) -> tuple:
        """vLLM's instances (`finalize_weights`), ascending."""
        top = self.max_tokens_per_rank
        fixed = _prev_power_of_2(_config().FIXED_SLOT_MAX_MTPR)
        if top > fixed and self.spec(fixed).bundle_plan().fixed_slot_dispatch:
            return (fixed, top)
        return (top,)

    def _segment(self, capacity: int) -> MegaMoEV2Segment:
        segment = self._segments.get(capacity)
        if segment is None:
            segment = MegaMoEV2Segment.for_spec(self.spec(capacity), self.weights)
            self._segments[capacity] = segment
        return segment

    def resolve(self, bucket: int, graph_capture: bool) -> BucketPlan:
        capacity = next((c for c in self.capacities if bucket <= c), None)
        if capacity is None:
            raise NotImplementedError(
                f"{NAME}: bucket {bucket} is above max_tokens_per_rank="
                f"{self.max_tokens_per_rank}"
            )
        segment = self._segment(capacity)
        plan = MoePlan((SegmentChoice(segment, segment.params()),), "default")
        identity = _canonical(
            {
                "spec": segment.spec,
                "fuse_shared_expert": bool(self.fuse_shared_expert),
                "mask_invalid_slots": self.mask_invalid_slots,
            }
        )
        return BucketPlan(
            bucket,
            plan.describe(),
            functools.partial(self._run, capacity),
            identity=identity,
            warmup=functools.partial(self._warmup, capacity),
            stateful=(segment,),
            output_lifetime=segment.output_lifetime,
        )

    def check_weights(self, weights: MoeWeights) -> None:
        theirs = _weights_signature(weights)
        for name, mine in self._signature.items():
            if theirs[name] != mine:
                raise ValueError(
                    f"weights.{name} is {theirs[name]}, the engine was prepared for "
                    f"{mine} (shape, dtype, is_shuffled, gate_up_interleave)"
                )

    def _warmup(self, capacity):
        segment = self._segments[capacity]
        if not segment.preload():
            return
        if self.fuse_shared_expert:
            import torch

            from ..routing import shared_route

            ids = torch.zeros(1, self.topk, dtype=torch.int32, device=segment.mega.dev)
            shared_route.append_shared_expert(
                torch.ones(ids.shape, dtype=torch.float32, device=ids.device),
                ids,
                shared_expert_id=shared_route.shared_expert_id(
                    segment.rank, self.routed_per_rank
                ),
                routed_per_rank=self.routed_per_rank,
            )

    def _run(self, capacity, x, topk_ids, topk_weights, weights, side, hints, out):
        segment = self._segments[capacity]
        if self.fuse_shared_expert:
            from ..routing import shared_route

            topk_weights, topk_ids = shared_route.append_shared_expert(
                topk_weights,
                topk_ids,
                shared_expert_id=shared_route.shared_expert_id(
                    segment.rank, self.routed_per_rank
                ),
                routed_per_rank=self.routed_per_rank,
            )
        segment.mega.bind(weights.w1, weights.w1_scale, weights.w2, weights.w2_scale)
        call = {
            "mega": segment.mega,
            "prequant": False,
            "x": x.contiguous(),
            "scales": None,
            "wts": topk_weights,
            "topk_ids": topk_ids,
            "stream": None,
            "slice_output": True,
            "mask_invalid_slots": self.mask_invalid_slots,
        }
        return forward(segment, call)
