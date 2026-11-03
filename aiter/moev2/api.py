# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public entry points of aiter.moev2, with the legacy signatures.

`fused_moe` here always runs the v2 code: it ignores AITER_MOEV2 and raises when no
registered path takes the call. The legacy `aiter.fused_moe.fused_moe` is the entry
point that honours the switch.

`build_problem` turns a `fused_moe_` call into the `MoeProblem` dispatch resolves.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from . import paths
from .policy import _normalize_mxfp4_activation_params, resolve_activation_dtype
from .problem import MoeProblem, SharedExpertSpec, scale_spec
from .shapes import get_padded_M

if TYPE_CHECKING:
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext

    from .execute.executor import Call


def dispatch_key(c: Call, q_dtype_a=None, config_file=None):
    """`(args, kwargs, q_dtype_a)`: the legacy `get_2stage_cfgs` arguments for a call.

    Built exactly as `_fused_moe_impl` builds them, from the prologue's derived values.
    `q_dtype_a` and `config_file` are its `_q_dtype_a` and `_metadata_config_file`
    (FHMoE): the activation format unless the call names one, and a dedicated table.
    """
    forced_q_dtype_a = q_dtype_a
    hidden_states, w1, w2 = c["hidden_states"], c["w1"], c["w2"]
    q_dtype_w = w1.dtype
    q_dtype_a = resolve_activation_dtype(
        c.quant_type,
        q_dtype_w,
        activation=c.activation,
        gate_mode=c.gate_mode,
        M=c.M,
        hidden_dtype=hidden_states.dtype,
        has_a1_scale=c["a1_scale"] is not None,
        gfx=get_gfx_runtime(),
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
    )
    if c["quant_dtype_a"] is not None:
        q_dtype_a = c["quant_dtype_a"]
    elif forced_q_dtype_a is not None:
        q_dtype_a = forced_q_dtype_a
    situ_beta, situ_linear_beta, _ = _normalize_mxfp4_activation_params(
        c.activation, c["beta"], c["linear_beta"], c["swiglu_limit"]
    )
    args = (
        get_padded_M(c.M),
        c.model_dim,
        c.inter_dim,
        c.E,
        c.topk,
        c.dtype,
        q_dtype_a,
        q_dtype_w,
        c.quant_type,
        c.isG1U1,
        c.activation,
        c["doweight_stage1"],
        c["hidden_pad"],
        c["intermediate_pad"],
        getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
        c.gate_mode,
    )
    kwargs = {
        "is_ep": c["expert_mask"] is not None,
        "has_stage1_bias": c["bias1"] is not None,
        "has_stage2_bias": c["bias2"] is not None,
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
        "swiglu_limit": c["swiglu_limit"],
        "opus_weights_shuffled": getattr(w1, "is_shuffled", False)
        and getattr(w2, "is_shuffled", False),
        "gate_up_interleave": getattr(w1, "gate_up_interleave", None),
        "config_file": config_file,
        "_disable_inline_sort": False,
        "q_dtype_a2": c["quant_dtype_a2"],
        "input_dtype": hidden_states.dtype,
        "has_stage2_scatter": c["ep_source_token_map"] is not None,
        "has_activation_scales": c["a1_scale"] is not None or c["a2_scale"] is not None,
        "has_num_local_tokens": c["num_local_tokens"] is not None,
    }
    return args, kwargs, q_dtype_a


def problem_from_key(args, kwargs, arch=None, cu_num=None, fhmoe=False) -> MoeProblem:
    """The `MoeProblem` of a legacy key; `fhmoe`: as `fhmoe_` dispatches it (the key has
    no field for the shared expert)."""
    problem = MoeProblem.from_legacy_key(
        *args,
        **kwargs,
        arch=get_gfx_runtime() if arch is None else arch,
        cu_num=get_cu_num() if cu_num is None else cu_num,
    )
    return fhmoe_problem(problem) if fhmoe else problem


def fhmoe_problem(problem) -> MoeProblem:
    """`problem` with the FHMoE shared expert: the last routed row, fp8 1x32 weights.

    `clamp` stays at its default: `clamp_shared` changes the launch, not the plan.
    """
    shared_w = scale_spec(QuantType.per_1x32, dtypes.fp8)
    shared = SharedExpertSpec(
        "fused_row", problem.shape.experts - 1, shared_w, shared_w
    )
    return replace(problem, shared=shared)


def build_problem(call, arch=None, cu_num=None) -> MoeProblem:
    """The `MoeProblem` of a `fused_moe_` call (`call` maps its parameter names)."""
    from .execute.executor import prologue
    from .policy import quant_remap

    args, kwargs, _ = dispatch_key(prologue(call, quant_remap))
    return problem_from_key(args, kwargs, arch, cu_num)


def op_call(call) -> dict | None:
    """The `fused_moe_` arguments the legacy `fused_moe` passes for `call`.

    None when the legacy `fused_moe` would not reach `fused_moe_` (shared experts go to
    FHMoE).
    """
    has_shared_expert = (
        any(
            call[name] is not None
            for name in ("shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale")
        )
        or call["shared_expert_id"] != -1
    )
    if has_shared_expert:
        return None
    if not call["clamp_shared"]:
        raise ValueError("clamp_shared=False requires FHMoE shared-expert dispatch")
    block_size_M = call["block_size_M"] or -1
    scatter = call["stage2_scatter"]
    enable_ep_scatter = scatter is not None
    quant_type_a = call["quant_type_a"]
    return {
        "hidden_states": call["hidden_states"],
        "w1": call["w1"],
        "w2": call["w2"],
        "topk_weight": call["topk_weight"],
        "topk_ids": call["topk_ids"],
        "expert_mask": call["expert_mask"],
        "activation": call["activation"].value,
        "quant_type": call["quant_type"].value,
        "doweight_stage1": call["doweight_stage1"],
        "w1_scale": call["w1_scale"],
        "w2_scale": call["w2_scale"],
        "a1_scale": call["a1_scale"],
        "a2_scale": call["a2_scale"],
        "block_size_M": block_size_M,
        "num_local_tokens": call["num_local_tokens"],
        "moe_sorting_dispatch_policy": call["moe_sorting_dispatch_policy"],
        "dtype": call["dtype"],
        "hidden_pad": call["hidden_pad"],
        "intermediate_pad": call["intermediate_pad"],
        "bias1": call["bias1"],
        "bias2": call["bias2"],
        "swiglu_limit": call["swiglu_limit"],
        "beta": call["beta"],
        "linear_beta": call["linear_beta"],
        "gate_mode": call["gate_mode"],
        "ep_arena_handle": scatter.arena_handle if enable_ep_scatter else 0,
        "ep_combine_input_offset": (
            scatter.combine_input_offset if enable_ep_scatter else 0
        ),
        "ep_slot_stride_bytes": scatter.slot_stride_bytes if enable_ep_scatter else 0,
        "ep_max_tokens_per_rank": (
            scatter.max_tokens_per_rank if enable_ep_scatter else 0
        ),
        "ep_world_size": scatter.world_size if enable_ep_scatter else 0,
        "ep_combine_quant": (
            int(scatter.combine_quant_bits) if enable_ep_scatter else 0
        ),
        "ep_source_token_map": scatter.source_token_map if enable_ep_scatter else None,
        "output": call["output"],
        "quant_type_a": None if quant_type_a is None else quant_type_a.value,
        "quant_dtype_a": call["quant_dtype_a"],
        "quant_dtype_a2": call["quant_dtype_a2"],
    }


def fhmoe_call(call) -> dict | None:
    """The `fhmoe_` arguments the legacy `fused_moe` passes for `call`, after its checks
    of the shared-expert arguments (raised as it raises them).

    None when the call has no shared expert (it goes to `fused_moe_`).
    """
    has_shared_expert = (
        any(
            call[name] is not None
            for name in ("shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale")
        )
        or call["shared_expert_id"] != -1
    )
    if not has_shared_expert:
        return None
    w1 = call["w1"]
    routed_mxfp8 = w1 is not None and w1.dtype == dtypes.fp8
    unsupported = []
    if call["stage2_scatter"] is not None:
        unsupported.append("stage2_scatter")
    if call["quant_type_a"] is not None:
        unsupported.append("quant_type_a")
    fhmoe_quant_dtype_a = (
        dtypes.fp8
        if GateMode(call["gate_mode"]) == GateMode.INTERLEAVE
        else dtypes.fp4x2
    )
    quant_dtype_a = call["quant_dtype_a"]
    if quant_dtype_a is not None and quant_dtype_a != fhmoe_quant_dtype_a:
        unsupported.append("quant_dtype_a")
    if call["quant_dtype_a2"] is not None:
        unsupported.append("quant_dtype_a2")
    if routed_mxfp8 and call["a1_scale"] is not None:
        unsupported.append("a1_scale")
    if routed_mxfp8 and call["a2_scale"] is not None:
        unsupported.append("a2_scale")
    if unsupported:
        raise NotImplementedError(
            "FHMoE shared-expert dispatch does not support " + ", ".join(unsupported)
        )
    weights = (w1, call["w2"], call["shared_w1"], call["shared_w2"])
    if routed_mxfp8 and any(
        weight is not None and not getattr(weight, "is_shuffled", False)
        for weight in weights
    ):
        raise ValueError(
            "FHMoE weights must be AITER-preshuffled and marked "
            "is_shuffled=True; re-run the shuffle after tensor copies"
        )
    names = (
        "hidden_states",
        "w1",
        "w2",
        "topk_weight",
        "topk_ids",
        "expert_mask",
        "doweight_stage1",
        "w1_scale",
        "w2_scale",
        "a1_scale",
        "a2_scale",
        "num_local_tokens",
        "moe_sorting_dispatch_policy",
        "dtype",
        "hidden_pad",
        "intermediate_pad",
        "bias1",
        "bias2",
        "swiglu_limit",
        "gate_mode",
        "shared_w1",
        "shared_w2",
        "shared_w1_scale",
        "shared_w2_scale",
        "shared_expert_id",
        "output",
        "clamp_shared",
    )
    return {
        **{name: call[name] for name in names},
        "activation": call["activation"].value,
        "quant_type": call["quant_type"].value,
        "block_size_M": call["block_size_M"] or -1,
    }


NO_PATH = (
    "no aiter.moev2 path handles this fused_moe call yet; "
    "use aiter.fused_moe.fused_moe (the legacy code)"
)


def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask: torch.Tensor | None = None,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M=None,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = "separated",
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    stage2_scatter: Stage2ScatterContext | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: QuantType | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    clamp_shared: bool = True,
):
    """Same contract as `aiter.fused_moe.fused_moe`, served by aiter.moev2."""
    for path, path_call in claiming_paths(dict(locals())):
        out = path.run(path_call)
        if out is not paths.DECLINED:
            return out
    raise NotImplementedError(NO_PATH)


def claiming_paths(call):
    """`(path, path_call)` for each registered path that claims the `fused_moe` call
    (`call` maps its parameter names), in the order `fused_moe` tries them; `path_call`
    is the call in the arguments of the path's entry."""
    op = fhmoe = None
    for path in paths.registered().values():
        if path.entry == "op":
            op = op_call(call) if op is None else op
            path_call = op
        elif path.entry == "fhmoe":
            fhmoe = fhmoe_call(call) if fhmoe is None else fhmoe
            path_call = fhmoe
        else:
            path_call = call
        if path_call is not None and path.claims(path_call):
            yield path, path_call
