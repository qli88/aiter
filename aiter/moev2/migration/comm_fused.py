# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `comm_fused`: `CommFusedMoeRuntime.run` (`aiter.ops.comm_fused_moe_runtime`).

Claims the runtime calls whose `fused_moe` arguments, padded to the runner's token
bucket, the a8w4 path claims. Runs them as the runtime does, without `_fused_moe_impl`
or `_stage2_override`: the padded call goes through the a8w4 plan of its key with the
stage 2 replaced by `comm_fused.stage2_rs` over the runtime's runner set. A padded call
the a8w4 path declines goes back to the legacy runtime, before any collective.

`CommFusedTemplate` serves the same plans to `MoeEngine`, whose segment then owns the
runner set (`prepare` creates it on the engine's `GroupCoordinator`).
"""

import functools
import inspect
from dataclasses import dataclass

import torch

from aiter import ActivationType, QuantType

from .. import api, paths
from ..dispatch.plan import SegmentChoice
from ..engine import (
    LOCAL_SIDE_INPUTS,
    BucketPlan,
    LocalTemplate,
    _canonical,
    _fused_moe_defaults,
    bucket_range,
)
from ..segments import comm_fused as stage2_rs
from ..segments.base import Params
from . import a8w4
from .local import with_stage2

NAME = "comm_fused"

# `CommFusedMoeRuntime.run` keyword arguments besides the `fused_moe` ones.
RUN_ARGS = (
    "shared_partial",
    "before_stage2",
    "before_stage2_for_rows",
    "before_shared_add",
    "stage2_stream",
    "reduce_scatter_sizes",
    "reuse_is_synchronized",
    "routes_may_be_missing",
)
COMM_FUSED_SIDE_INPUTS = frozenset(RUN_ARGS) - {"stage2_stream"}
# `fused_moe` parameters `_fused_moe_impl` does not take (the runtime raises TypeError).
NOT_IMPL_ARGS = frozenset(
    {
        "clamp_shared",
        "shared_expert_id",
        "shared_w1",
        "shared_w1_scale",
        "shared_w2",
        "shared_w2_scale",
        "splitk",
    }
)


@functools.cache
def _api_names() -> frozenset:
    return frozenset(inspect.signature(api.fused_moe).parameters) - NOT_IMPL_ARGS


def _api_call(moe_args):
    """The `fused_moe` call of `_fused_moe_impl` arguments (enums decoded), or None
    when one is not an argument of both, or `block_size_M` is 0 (`fused_moe` reads 0
    as -1, `_fused_moe_impl` does not)."""
    if not set(moe_args) <= _api_names() or moe_args.get("block_size_M") == 0:
        return None
    call = {**_fused_moe_defaults(), **moe_args}
    call["activation"] = ActivationType(call["activation"])
    call["quant_type"] = QuantType(call["quant_type"])
    if call["quant_type_a"] is not None:
        call["quant_type_a"] = QuantType(call["quant_type_a"])
    return call


def _padded_rows(runners, call):
    """The runner bucket the runtime pads the call to, or None when it raises."""
    sizes = call["reduce_scatter_sizes"]
    rows = int(call["moe_args"]["hidden_states"].shape[0])
    if sizes is not None:
        sizes = tuple(int(s) for s in sizes)
        if not sizes or any(s < 0 for s in sizes) or sum(sizes) != rows:
            return None
        rows = max(sizes) * len(sizes)
    bucket = stage2_rs.bucket_for(runners.configs, rows)
    return bucket if bucket >= rows else None


def claims(call) -> bool:
    fused_moe_call = _api_call(call["moe_args"])
    if fused_moe_call is None:
        return False
    bucket = _padded_rows(call["runtime"].runners, call)
    if bucket is None:
        return False
    topk_ids = fused_moe_call["topk_ids"]
    fused_moe_call["topk_ids"] = torch.empty(
        (bucket, topk_ids.shape[1]), dtype=topk_ids.dtype, device="meta"
    )
    op = api.op_call(fused_moe_call)
    return op is not None and a8w4.claims(op)


_ADOPTED: dict = {}


def _adopted(runners):
    """The segment over a runtime's runner set, one per set."""
    entry = _ADOPTED.get(id(runners))
    if entry is None or entry[0] is not runners:
        entry = _ADOPTED[id(runners)] = (
            runners,
            stage2_rs.CommFusedStage2.adopt(runners),
        )
    return entry[1]


def run_padded(segment, padded, fused):
    """Run a call `prepare_call` padded through the a8w4 plan with `segment` as its
    stage 2, or DECLINED (before the reuse wait) when the a8w4 path declines it."""
    op = api.op_call(padded)
    if a8w4.LOCAL.resolve_call(op) is None:
        return paths.DECLINED
    stage2_rs.wait_for_reuse(fused)
    op["comm_fused"] = fused
    choice = SegmentChoice(
        segment, Params(runner=stage2_rs.runner_label(fused.runner.config))
    )
    out = a8w4.LOCAL.run(op, stage2=choice)
    if out is paths.DECLINED:
        raise RuntimeError(f"{NAME}: the a8w4 path resolved this call, then declined")
    return stage2_rs.finish(fused, out)


def run(call):
    runners = call["runtime"].runners
    padded, fused = stage2_rs.prepare_call(
        runners, _api_call(call["moe_args"]), **{k: call[k] for k in RUN_ARGS}
    )
    return run_padded(_adopted(runners), padded, fused)


# AITER_MOEV2=comm_fused until the TP8 differential passes.
PATH = paths.Path(NAME, claims, run, entry="comm_fused")
paths.register(PATH)


@dataclass
class CommFusedTemplate(LocalTemplate):
    """`CommFusedMoeRuntime` for `MoeEngine` (gfx950, TP; `comm=` the TP
    `GroupCoordinator`): `LocalTemplate`'s fields for the a8w4 call, plus `tp_size`
    (the group's ranks), `comm` ("ar" or "rs") and `add_shared`, which pick the runner
    set as `create_flydsl_comm_fused_runners` does, and `stage2_stream` (the runtime's
    argument). `side_inputs`: of `LOCAL_SIDE_INPUTS` and the `run` arguments
    (`COMM_FUSED_SIDE_INPUTS`: the shared partial or its producers, ragged sizes, the
    reuse and missing-route flags).

    A bucket's plan names the runner and a8w4 plan of each runner bucket its counts
    pad to; counts no runner takes are not served (`engine.supports`), as
    `CommFusedMoeRuntime.supports` says. `capture()` warms up nothing: a warm-up would
    be a collective with a shared partial the caller owns.
    """

    tp_size: int = 1
    comm: str = "ar"
    add_shared: bool = True
    stage2_stream: object = None

    serves_parallel = True
    allowed_side_inputs = LOCAL_SIDE_INPUTS | COMM_FUSED_SIDE_INPUTS

    def __post_init__(self):
        super().__post_init__()
        self._segment = None

    def _stage2(self):
        if self._segment is None:
            from aiter.ops.flydsl.comm_fused_moe_host import comm_fused_shape

            from ..shapes import get_inter_dim

            experts, model_dim, inter_dim = get_inter_dim(
                tuple(self.weights.w1.shape), tuple(self.weights.w2.shape)
            )
            shape = comm_fused_shape(
                tp_size=self.tp_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=self.topk,
                comm=self.comm,
                add_shared=self.add_shared,
            )
            self._segment = stage2_rs.CommFusedStage2.for_shape(shape)
        return self._segment

    def resolve(self, bucket: int, graph_capture: bool) -> BucketPlan:
        segment = self._stage2()
        configs = segment.configs
        lowest, highest = bucket_range(bucket)
        # The runner bucket of a count changes only past `bucket` and past each runner
        # bucket in the range (the top tier pads to the smallest runner that holds it).
        counts = {lowest}
        for m in (bucket, *configs):
            if m >= lowest and (highest is None or m < highest):
                counts.update((m, m + 1))
        if highest is not None:
            counts.add(highest)
        parts = []  # [first count, runner bucket or None, description, identity]
        for m in sorted(counts):
            runner_bucket = stage2_rs.bucket_for(configs, m)
            if runner_bucket < m or runner_bucket not in configs:
                runner_bucket = None
            if parts and parts[-1][1] == runner_bucket:
                continue
            parts.append([m, runner_bucket, *self._describe(segment, runner_bucket)])
        if len(parts) == 1:
            description, identity = parts[0][2:]
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
            description = "; ".join(f"{lb}: {p[2]}" for lb, p in zip(labels, parts))
            identity = "; ".join(f"{lb}: {p[3]}" for lb, p in zip(labels, parts))
        return BucketPlan(
            bucket,
            description,
            self._run,
            identity=identity,
            stateful=(segment,),
            side_inputs=self.side_inputs,
            supports=functools.partial(stage2_rs.supports, configs),
        )

    def _describe(self, segment, runner_bucket) -> tuple:
        """`(description, identity)` of the plan of a runner bucket."""
        if runner_bucket is None:
            return "not served (no comm_fused runner)", "not served"
        resolution = a8w4.LOCAL.resolve_call(
            api.op_call(self._meta_call(runner_bucket))
        )
        if resolution is None or resolution.plan is None:
            raise NotImplementedError(
                f"{NAME}: the a8w4 path declines the {runner_bucket}-token call"
            )
        config = segment.configs[runner_bucket]
        choice = SegmentChoice(segment, Params(runner=stage2_rs.runner_label(config)))
        plan = with_stage2(resolution.plan, choice)
        return (
            f"{NAME} M={runner_bucket}: {plan.describe()}",
            f"{_canonical(resolution.metadata)}; {_canonical(config)}",
        )

    def _run(self, x, topk_ids, topk_weights, weights, side, hints, out):
        local = {k: v for k, v in side.items() if k in LOCAL_SIDE_INPUTS}
        run_args = {k: v for k, v in side.items() if k not in LOCAL_SIDE_INPUTS}
        segment = self._stage2()
        padded, fused = stage2_rs.prepare_call(
            segment.runners,
            self._call(x, topk_ids, topk_weights, weights, local, None),
            stage2_stream=self.stage2_stream,
            **run_args,
        )
        result = run_padded(segment, padded, fused)
        if result is paths.DECLINED:
            raise RuntimeError(f"{NAME}: the a8w4 path declined a prepared bucket")
        return result
