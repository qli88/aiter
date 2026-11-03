# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segment `comm_fused.stage2_rs`: the stage 2 of `CommFusedMoeRuntime` (gfx950, TP).

One runner per token bucket fuses GEMM2, the TP all-reduce or reduce-scatter and the
shared-expert add. The runners and their packed symmetric arena stay in
`aiter.ops.flydsl.comm_fused_moe_host`; this segment adapts a runner set. In a plan it
replaces the stage 2 of the local a8w4 plan of the call padded to the runner's bucket:
sorting, stage 1 and the intermediate quantization run as in that plan, and the runner
gets the launch the plan's own stage 2 would have got (its kernel name tells the runner
the intermediate layout).

`prepare_call`, `launch` and `finish` are `CommFusedMoeRuntime.run` statement for
statement: before `_fused_moe_impl`, its `stage2_override`, and after it.
"""

import functools
from dataclasses import dataclass, field

import torch

from aiter import dtypes

from ..dispatch.ops import Op
from .base import Cap, MemoryNeed, Segment
from .mxfp4 import mxfp4, mxfp8


def bucket_for(configs, tokens: int) -> int:
    """`CommFusedMoeRuntime.bucket_for`: the standard padded bucket, extended past its
    upper bound to the smallest runner bucket that holds `tokens`."""
    from ..shapes import get_padded_M

    bucket = int(get_padded_M(tokens))
    if bucket < tokens:
        larger = [candidate for candidate in configs if candidate >= tokens]
        if larger:
            bucket = min(larger)
    return bucket


def output_rows_for(configs, bucket: int, input_rows: int) -> int:
    """`_LazyRunners.output_rows_for`, over the runner configs."""
    config = configs[bucket]
    if config.requires_exact_m and input_rows != config.m:
        raise ValueError(f"{type(config).__name__} requires exact M={config.m}")
    output_rows, remainder = divmod(input_rows * config.output_rows, config.m)
    if remainder:
        raise ValueError(
            f"runner layout maps bucket M={config.m} to "
            f"{config.output_rows} output rows and cannot represent "
            f"input M={input_rows}"
        )
    return output_rows


def supports(configs, tokens: int) -> bool:
    """`CommFusedMoeRuntime.supports`."""
    bucket = bucket_for(configs, tokens)
    if bucket < tokens or bucket not in configs:
        return False
    try:
        output_rows_for(configs, bucket, tokens)
    except ValueError:
        return False
    return True


def runner_label(config) -> str:
    """The runner's kernel name (`config_name`), for plans and logs."""
    from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.config import (
        PipelineConfig,
    )

    if isinstance(config, PipelineConfig):
        from aiter.ops.flydsl.comm_fused_moe_host import config_name

        return config_name(config)
    return repr(config)


@dataclass
class CommFusedCall:
    """One call's runner and the `CommFusedMoeRuntime.run` arguments its stage 2
    reads; `final_output` is what the runner returned."""

    runner: object
    bucket: int
    raw_tokens: int
    output_rows: int
    device: torch.device
    shared_partial: torch.Tensor | None = None
    before_stage2: object = None
    before_stage2_for_rows: object = None
    before_shared_add: object = None
    stage2_stream: object = None
    reduce_scatter_sizes: tuple | None = None
    stage2_destination: torch.Tensor | None = None
    wait_before_write: bool = False
    runner_kwargs: dict = field(default_factory=dict)
    final_output: torch.Tensor | None = None


def prepare_call(
    runners,
    call,
    *,
    shared_partial=None,
    before_stage2=None,
    before_stage2_for_rows=None,
    before_shared_add=None,
    stage2_stream=None,
    reduce_scatter_sizes=None,
    reuse_is_synchronized=False,
    routes_may_be_missing=False,
):
    """`(call, CommFusedCall)`: the checks of `CommFusedMoeRuntime.run`, its runner
    choice and the call padded to the runner's bucket (a new mapping; `call` maps
    `fused_moe` argument names). The Direct reuse wait is left to `wait_for_reuse`."""
    if before_stage2 is not None and before_stage2_for_rows is not None:
        raise ValueError(
            "before_stage2 and before_stage2_for_rows are mutually exclusive"
        )
    call = dict(call)
    hidden_states = call["hidden_states"]
    raw_tokens = int(hidden_states.shape[0])
    runner_kwargs = {}
    sizes = None
    if reduce_scatter_sizes is None:
        bucket_input_rows = raw_tokens
    else:
        sizes = tuple(int(size) for size in reduce_scatter_sizes)
        if not sizes or any(size < 0 for size in sizes):
            raise ValueError(
                "reduce_scatter_sizes must contain non-negative rank sizes"
            )
        if sum(sizes) != raw_tokens:
            raise ValueError(
                "compact ragged input rows must equal sum(reduce_scatter_sizes): "
                f"input={raw_tokens}, sizes={sizes}"
            )
        bucket_input_rows = max(sizes) * len(sizes)
    bucket = bucket_for(runners.configs, bucket_input_rows)
    if bucket < bucket_input_rows:
        raise KeyError(
            f"no comm_fused bucket for required M={bucket_input_rows} "
            f"({raw_tokens} compact rows)"
        )
    runner = runners[bucket]
    if sizes is None:
        output_rows = runners.output_rows_for(bucket, raw_tokens)
    else:
        if len(sizes) != runner.config.shape.tp_size:
            raise ValueError(
                "reduce_scatter_sizes length must equal TP size: "
                f"sizes={len(sizes)}, TP={runner.config.shape.tp_size}"
            )
        if not runner.supports_ragged_m:
            raise ValueError(f"{type(runner).__name__} does not support ragged M")
        local_rows = sizes[runner.rank]
        rank_offset = sum(sizes[: runner.rank])
        if local_rows > runner.config.output_rows:
            raise ValueError(
                f"rank {runner.rank} has {local_rows} rows but runner "
                f"capacity is {runner.config.output_rows}"
            )
        output_rows = local_rows
        runner_kwargs.update(local_rows=local_rows, rank_offset=rank_offset)
    stage2_destination = runner.stage2_destination
    if stage2_destination is None and (
        routes_may_be_missing
        or call.get("expert_mask") is not None
        or call.get("num_local_tokens") is not None
    ):
        runner_kwargs["routes_may_be_missing"] = True
    if stage2_destination is not None:
        if call.get("output") is not None:
            raise RuntimeError("comm-fused Stage2 output is incompatible with output=")
        call["output"] = stage2_destination
    if bucket != raw_tokens:
        topk_weight = call["topk_weight"]
        topk_ids = call["topk_ids"]
        padded_hidden = hidden_states.new_zeros((bucket, hidden_states.shape[1]))
        padded_weight = topk_weight.new_zeros((bucket, topk_weight.shape[1]))
        padded_ids = topk_ids.new_zeros((bucket, topk_ids.shape[1]))
        padded_hidden[:raw_tokens].copy_(hidden_states)
        padded_weight[:raw_tokens].copy_(topk_weight)
        padded_ids[:raw_tokens].copy_(topk_ids)
        call["hidden_states"] = padded_hidden
        call["topk_weight"] = padded_weight
        call["topk_ids"] = padded_ids
    wait_before_write = False
    if reuse_is_synchronized and runner.supports_external_reuse_sync:
        runner_kwargs["reuse_is_synchronized"] = True
    elif runner.requires_reuse_wait_before_stage2_write:
        wait_before_write = True
        runner_kwargs["reuse_waited_before_stage2_write"] = True
    return call, CommFusedCall(
        runner=runner,
        bucket=bucket,
        raw_tokens=raw_tokens,
        output_rows=output_rows,
        device=hidden_states.device,
        shared_partial=shared_partial,
        before_stage2=before_stage2,
        before_stage2_for_rows=before_stage2_for_rows,
        before_shared_add=before_shared_add,
        stage2_stream=stage2_stream,
        reduce_scatter_sizes=sizes,
        stage2_destination=stage2_destination,
        wait_before_write=wait_before_write,
        runner_kwargs=runner_kwargs,
    )


def wait_for_reuse(fused: CommFusedCall) -> None:
    """Direct's pre-overwrite rank barrier, before sorting clears its destination
    (collective)."""
    if fused.wait_before_write:
        fused.runner.wait_for_stage2_destination_reuse()


def launch(fused: CommFusedCall, *, ordinary_stage2, stage2_args, stage2_kwargs):
    """The runner in place of `ordinary_stage2(*stage2_args, **stage2_kwargs)`, on
    `stage2_stream` when given; returns the stage 2 destination or the runner's output.
    """
    runner = fused.runner

    def run():
        current_shared = fused.shared_partial
        shared_join = fused.before_shared_add
        if fused.before_stage2_for_rows is not None:
            shared_rows = (
                fused.output_rows
                if fused.reduce_scatter_sizes is not None
                else runner.config.output_rows
            )
            current_shared = fused.before_stage2_for_rows(shared_rows)
        elif fused.before_stage2 is not None:
            current_shared = fused.before_stage2()
        add_shared = runner.config.shape.add_shared
        if add_shared and current_shared is None:
            raise RuntimeError("comm-fused Stage2 requires shared_partial")
        if (
            add_shared
            and fused.reduce_scatter_sizes is None
            and fused.bucket != fused.raw_tokens
            and current_shared.shape[0] != runner.config.output_rows
        ):
            # Padding reads the shared output: join its producer first, once.
            if shared_join is not None:
                shared_join()
                shared_join = None
            current_shared = runner.prepare_padded_shared_partial(
                current_shared, fused.raw_tokens
            )
        if add_shared:
            current_shared = runner.prepare_shared_partial(current_shared)
        fused.final_output = runner(
            shared_partial=current_shared,
            before_shared_add=shared_join,
            **fused.runner_kwargs,
            ordinary_stage2=ordinary_stage2,
            stage2_args=stage2_args,
            stage2_kwargs=stage2_kwargs,
        )
        return (
            fused.stage2_destination
            if fused.stage2_destination is not None
            else fused.final_output
        )

    if fused.stage2_stream is None:
        return run()
    caller_stream = torch.cuda.current_stream(fused.device)
    if fused.stage2_stream == caller_stream:
        return run()
    fused.stage2_stream.wait_stream(caller_stream)
    with torch.cuda.stream(fused.stage2_stream):
        output = run()
    caller_stream.wait_stream(fused.stage2_stream)
    return output


def finish(fused: CommFusedCall, output: torch.Tensor) -> torch.Tensor:
    """The call's result: the runner's output, cut to this rank's output rows."""
    if fused.final_output is not None:
        output = fused.final_output
    return (
        output if fused.output_rows == output.shape[0] else output[: fused.output_rows]
    )


class CommFusedStage2(Segment):
    """A runner set as a stateful segment: one per TP group and geometry.

    `adopt(runners)` wraps the set a `CommFusedMoeRuntime` was built with (already
    allocated: nothing to prepare). `for_shape(shape)` creates it at `prepare(ctx)`
    through `create_flydsl_comm_fused_runners` on the engine's `GroupCoordinator`,
    sharing that function's cache (one arena per group, geometry and MXFP8 fallback).
    `configs`: the runner config per token bucket. The runners have no health, reset or
    free in legacy, so neither has the segment.
    """

    name = "comm_fused.stage2_rs"
    covers = (Op.GEMM2, Op.REDUCE)
    capabilities = (
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp8()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16}),
            activation=frozenset({"silu"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "num_local_tokens"}),
            shared=True,
            parallel=True,
        ),
    )
    reach = "tuned"
    stateful = True

    def __init__(self, configs, runners=None, shape=None):
        self.configs = dict(configs)
        self.runners = runners
        self.shape = shape

    @classmethod
    def adopt(cls, runners) -> "CommFusedStage2":
        return cls(runners.configs, runners=runners)

    @classmethod
    def for_shape(cls, shape) -> "CommFusedStage2":
        from aiter.ops.flydsl.comm_fused_moe_host import comm_fused_configs

        configs, _ = comm_fused_configs(shape)
        return cls(configs, shape=shape)

    def workspace_bytes(self) -> MemoryNeed:
        if self.runners is not None:
            return MemoryNeed()
        from aiter.ops.flydsl.comm_fused_moe_host import runner_arena_bytes

        return MemoryNeed(symmetric=runner_arena_bytes(self.configs))

    def prepare(self, ctx) -> None:
        if self.runners is not None:
            return
        comm = ctx.comm
        if not hasattr(comm, "cpu_group"):
            raise ValueError(
                "comm_fused.stage2_rs needs the TP GroupCoordinator as comm="
            )
        if ctx.world_size != self.shape.tp:
            raise ValueError(
                f"the runner set is for TP{self.shape.tp}; the group has "
                f"{ctx.world_size} ranks"
            )
        from aiter.ops.flydsl.comm_fused_moe_host import (
            create_flydsl_comm_fused_runners,
        )

        s = self.shape
        self.runners = create_flydsl_comm_fused_runners(
            tp_group=comm,
            model_dim=s.model_dim,
            inter_dim=s.inter_dim,
            experts=s.experts,
            topk=s.topk,
            comm=s.comm,
            add_shared=s.add_shared,
        )

    def stage2_launch(self, c):
        return functools.partial(launch, c["comm_fused"])
