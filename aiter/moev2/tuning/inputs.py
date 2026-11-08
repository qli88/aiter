# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Real inputs for tuning a problem: a `fused_moe` call and its torch reference.

`build(problem)` makes the call as `op_tests/test_moe_2stage.py` does for the problem's
format (random activations, softmax top-k routing, weights quantized and shuffled
the way the kernels read them) and the reference output the same way that test does:
`torch_moe_stage1`, the intermediate quantized as the format quantizes it, then
`torch_moe_stage2`, on the quantized weights before shuffling. Weights are scaled by
1/sqrt(K) so each GEMM keeps values near 1 (unscaled, fp16 outputs reach the fp16
limit). The call must key on the problem (`problem_of`); a problem whose format is not
built here raises `Unsupported`.

An FHMoE problem (a shared expert, `fhmoe_`; fp8 activations, interleaved gates) is
built as `op_tests/test_fhmoe.py` does: the routed experts (fp4 or fp8) and the fp8
shared expert quantized per 1x32 and shuffled, routing
over every routed expert plus the shared expert as the last column (weight 1), and a
dequantized fp32 reference that quantizes the activations and the intermediate as the
kernels do.
"""

import inspect
from dataclasses import dataclass, replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .. import api
from ..execute.executor import prologue
from ..policy import quant_remap
from ..shapes import get_padded_M

# The problem fields a call may differ in from a problem built from a tuned-table key:
# the tables have no `gate_up_interleave` column, a shuffled call tags its weights.
_UNKEYED_LAYOUT = ("gate_up_interleave",)


class Unsupported(ValueError):
    """The problem's format has no input builder, or the built call keys elsewhere."""


@dataclass
class Inputs:
    call: dict  # `aiter.moev2.api.fused_moe` arguments, every parameter named
    reference: torch.Tensor  # the torch reference output


def fused_moe_defaults() -> dict:
    """The parameters of `api.fused_moe` with their defaults (None for the required)."""
    return {
        name: None if p.default is inspect.Parameter.empty else p.default
        for name, p in inspect.signature(api.fused_moe).parameters.items()
    }


def entry(call):
    """`(op, q_dtype_a, config_file, fhmoe)`: the `fused_moe_` arguments `call` reaches
    (those `fhmoe_` passes for a shared expert), with the activation format and the
    dedicated table its path dispatches with (None: from the call, the main table)."""
    op = api.op_call(call)
    if op is not None:
        return op, None, None, False
    from ..migration import fhmoe

    fh = api.fhmoe_call(call)
    gate_mode = GateMode(fh["gate_mode"])
    q_dtype_a = dtypes.fp8 if gate_mode == GateMode.INTERLEAVE else dtypes.fp4x2
    op = {**fh, **fhmoe._OP_DEFAULTS}
    return op, q_dtype_a, fhmoe.config_file(fh, gate_mode), True


def problem_of(call, arch=None, cu_num=None):
    """The `MoeProblem` dispatch keys `call` on (`call`: `api.fused_moe` arguments)."""
    op, q_dtype_a, config_file, fhmoe = entry(call)
    c = prologue(op, quant_remap)
    args, kwargs, _ = api.dispatch_key(c, q_dtype_a, config_file)
    return api.problem_from_key(args, kwargs, arch, cu_num, fhmoe)


def prologued(call):
    """`(c, q_dtype_a)`: the prologued call and its activation format, as the local
    paths hand them to `migration.local.execute`."""
    op, q_dtype_a, config_file, _ = entry(call)
    c = prologue(op, quant_remap)
    _, _, q_dtype_a = api.dispatch_key(c, q_dtype_a, config_file)
    return c, q_dtype_a


def check_keys_on(call, problem):
    """Raise `Unsupported` unless `call` keys on `problem` (but for `_UNKEYED_LAYOUT`
    where `problem` leaves it None)."""
    built = problem_of(call, problem.arch, problem.cu_num)
    layout = built.layout
    for name in _UNKEYED_LAYOUT:
        if getattr(problem.layout, name) is None:
            layout = replace(layout, **{name: None})
    built = replace(built, layout=layout)
    if built != problem:
        diff = [
            f"{name}: {getattr(built, name)!r} != {getattr(problem, name)!r}"
            for name in problem.__dataclass_fields__
            if getattr(built, name) != getattr(problem, name)
        ]
        raise Unsupported("the built call keys elsewhere: " + "; ".join(diff))


def _format(problem):
    """The format name of `problem`, as the test builds it, or raise `Unsupported`."""
    unsupported = []
    if problem.shared is not None:
        unsupported.append("a shared expert")
    if problem.parallel is not None or problem.tail is not None:
        unsupported.append("a collective or a tail")
    if problem.inter is not None:
        unsupported.append("a caller-fixed GEMM2 input format")
    if problem.layout.hidden_pad or problem.layout.intermediate_pad:
        unsupported.append("padding")
    extra = problem.features - {"doweight_stage1"}
    if extra:
        unsupported.append(f"features {sorted(extra)}")
    if problem.layout.shuffle != "16x16":
        unsupported.append(f"{problem.layout.shuffle} weights")
    if problem.gate_mode.value != "separated":
        unsupported.append(f"gate mode {problem.gate_mode.value}")
    if not problem.is_g1u1:
        unsupported.append("g1u0")
    a, w, q = problem.act_in.dtype, problem.w1.dtype, problem.quant_type
    name = {
        (QuantType.No, dtypes.bf16, dtypes.bf16): "a16w16",
        (QuantType.No, dtypes.fp16, dtypes.fp16): "a16w16",
        (QuantType.per_Tensor, dtypes.fp8, dtypes.fp8): "a8w8_pt",
        (QuantType.per_Tensor, dtypes.i8, dtypes.i8): "a8w8_pt",
        (QuantType.per_Token, dtypes.fp8, dtypes.fp8): "a8w8_ptok",
        (QuantType.per_Token, dtypes.i8, dtypes.i8): "a8w8_ptok",
        (QuantType.per_1x128, dtypes.fp8, dtypes.fp8): "a8w8_blockscale",
        (QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2): "a4w4",
        (QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2): "a16w4",
        (QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2): "a8w4",
        (QuantType.per_1x32, dtypes.fp8, dtypes.fp8): "mxfp8",
        (QuantType.per_1x32, dtypes.bf16, dtypes.i4x2): "a16wi4",
    }.get((q, a, w))
    if name is None:
        unsupported.append(f"format {q} {a} x {w}")
    if unsupported:
        raise Unsupported("no input builder for " + ", ".join(unsupported))
    return name


def _per_128x128_quant(weight, quant_dtype):
    import aiter

    E, dim1, dim2 = weight.shape
    blocks = weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
    blocks = blocks.permute(0, 1, 3, 2, 4).contiguous().view(E, -1, 128 * 128)
    qt, scale = aiter.pertoken_quant(blocks, quant_dtype=quant_dtype)
    qt = qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
    qt = qt.permute(0, 1, 3, 2, 4).contiguous().view(E, dim1, dim2)
    return qt, scale.view(E, dim1 // 128, dim2 // 128)


def _quant_weight(fmt, w, a_dtype):
    """`(quantized as the reference reads it, scale)` of weight `w`."""
    import aiter
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_i4_quant

    E = w.shape[0]
    if fmt == "a16w16":
        return w, None
    if fmt == "a8w8_pt":
        qt, scale = aiter.pertoken_quant(w.view(E, -1), quant_dtype=a_dtype)
        return qt.view(w.shape), scale
    if fmt == "a8w8_ptok":
        return aiter.pertoken_quant(w, quant_dtype=a_dtype)
    if fmt == "a8w8_blockscale":
        return _per_128x128_quant(w, dtypes.fp8)
    if fmt == "mxfp8":
        qt, scale = per_1x32_f8_scale_f8_quant(
            w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        return qt.view(w.shape), scale
    if fmt == "a16wi4":
        qt, scale = per_1x32_i4_quant(w)
        return qt.view(dtypes.i4x2), scale
    qt, scale = aiter.get_torch_quant(QuantType.per_1x32)(w, quant_dtype=dtypes.fp4x2)
    return qt.view(*w.shape[:-1], w.shape[-1] // 2), scale


def _kernel_weight(fmt, qt, scale, E, first):
    """`(weight, scale)` as the kernels read them: shuffled, scales shuffled."""
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_a16w4,
        shuffle_scale_for_int4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility import fp4_utils

    if fmt == "a16wi4":
        packed = pack_int8_to_packed_int4(shuffle_weight(qt.view(dtypes.i8), (16, 16)))
        shape = (qt.shape[0], qt.shape[1], qt.shape[2] // 2)
        packed = packed.view(shape).view(dtypes.i4x2)
        packed.is_shuffled = True
        return (
            packed,
            shuffle_scale_for_int4(scale, group_size=32).view(-1).contiguous(),
        )
    if fmt in ("a16w4", "a8w4"):
        # a16w4 reads gate and up as stored (GGUU), a8w4 interleaved (GUGU).
        gate_up = first and fmt == "a8w4"
        return shuffle_weight_a16w4(qt, 16, gate_up), shuffle_scale_a16w4(
            scale, E, gate_up
        )
    if fmt == "mxfp8":
        if first:
            return shuffle_weight_a16w4(qt, 16, True), shuffle_scale_a16w4(
                scale, E, True
            )
        return shuffle_weight_a16w4(qt, 16, False), fp4_utils.e8m0_shuffle(scale)
    shuffled = shuffle_weight(qt, layout=(16, 16))
    if fmt == "a4w4":
        return shuffled, fp4_utils.e8m0_shuffle(scale)
    return shuffled, scale


def _quant_act(fmt, x, a_dtype):
    """`(activations as the reference reads them, scale)` for GEMM1 or GEMM2 input
    `x` (2D, or 3D `[token, topk, inter]` for GEMM2)."""
    import aiter
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    if fmt in ("a16w16", "a16w4", "a16wi4", "mxfp8") or (
        fmt == "a8w4" and a_dtype != dtypes.fp8
    ):
        return x.to(dtypes.bf16) if fmt != "a16w16" else x, None
    if fmt == "a8w8_blockscale":
        qt, scale = aiter.pertoken_quant(
            x.view(*x.shape[:-1], -1, 128), quant_dtype=a_dtype
        )
        return qt.view(x.shape), scale.squeeze(-1)
    if fmt == "a8w4":
        return per_1x32_f8_scale_f8_quant(
            x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    quant_type = {
        "a8w8_pt": QuantType.per_Tensor,
        "a8w8_ptok": QuantType.per_Token,
        "a4w4": QuantType.per_1x32,
    }[fmt]
    return aiter.get_torch_quant(quant_type)(x, quant_dtype=a_dtype)


def _route(score, topk):
    """`(weights, ids)`: the softmax top-k of `score`, renormalized (`fused_topk`)."""
    weights, ids = torch.softmax(score.float(), dim=-1).topk(topk, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), ids.to(torch.int32)


_CALL_QUANT = {
    "a16w16": QuantType.No,
    "a8w8_pt": QuantType.per_Tensor,
    "a8w8_ptok": QuantType.per_Token,
    "a8w8_blockscale": QuantType.per_128x128,
}


def build(problem, seed=0, device="cuda") -> Inputs:
    """The call and reference for `problem`; raises `Unsupported` (no builder for its
    format, or the call would key elsewhere)."""
    from ..reference.torch_moe import torch_moe_stage1, torch_moe_stage2

    m = problem.shape.m
    if get_padded_M(m) != m:
        raise Unsupported(
            f"M={m} is not a dispatch bucket: a call with {m} tokens looks up "
            f"M={get_padded_M(m)}"
        )
    if problem.shared is not None:
        return _build_fhmoe(problem, seed, device)
    fmt = _format(problem)
    s, act = problem.shape, problem.activation
    token, H, inter, E, topk = s.m, s.model_dim, s.inter_dim, s.experts, s.topk
    dtype = problem.out_dtype
    a_dtype = problem.act_in.dtype
    doweight = "doweight_stage1" in problem.features
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    hidden = randn(token, H).to(problem.in_dtype or dtype)
    score = randn(token, E)
    topk_weight, topk_ids = _route(score, topk)
    weights = []
    for shape, first in (((E, 2 * inter, H), True), ((E, H, inter), False)):
        w = randn(*shape) / shape[-1] ** 0.5
        qt, scale = _quant_weight(fmt, w, a_dtype)
        del w
        weights.append((qt, scale, *_kernel_weight(fmt, qt, scale, E, first)))
    (w1_qt, w1_scale, w1, w1_kscale), (w2_qt, w2_scale, w2, w2_kscale) = weights

    activation = act.activation
    call = fused_moe_defaults()
    call.update(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        activation=activation,
        quant_type=_CALL_QUANT.get(fmt, QuantType.per_1x32),
        doweight_stage1=doweight,
        w1_scale=w1_kscale,
        w2_scale=w2_kscale,
        dtype=dtype,
        swiglu_limit=act.swiglu_limit,
    )
    if activation == ActivationType.Situv2:
        call.update(beta=act.situ_beta, linear_beta=act.situ_linear_beta)
    if problem_of(call, problem.arch, problem.cu_num).act_in.dtype != a_dtype:
        call["quant_dtype_a"] = a_dtype
    check_keys_on(call, problem)

    situ = {}
    if activation == ActivationType.Situv2:
        situ = {"situ_beta": act.situ_beta, "situ_linear_beta": act.situ_linear_beta}
    a1, a1_scale = _quant_act(fmt, hidden, a_dtype)
    reference_quant = quant_remap.get(call["quant_type"], call["quant_type"])
    out1 = torch_moe_stage1(
        a1,
        w1_qt,
        w2_qt,
        topk_weight,
        topk_ids,
        dtype=dtype,
        activation=activation,
        quant_type=reference_quant,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=doweight,
        swiglu_limit=act.swiglu_limit,
        **situ,
    )
    a2, a2_scale = _quant_act(fmt, out1, a_dtype)
    if fmt == "a8w8_blockscale":
        a2_scale = a2_scale.view(token, topk, -1)
    reference = torch_moe_stage2(
        a2.view(token, topk, -1),
        w1_qt,
        w2_qt,
        topk_weight,
        topk_ids,
        dtype=dtype,
        quant_type=reference_quant,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=not doweight,
    )
    return Inputs(call, reference)


def _fhmoe_format(problem):
    """The routed weight format of an FHMoE problem ("fp4" or "fp8"), or raise
    `Unsupported`. Activations must be fp8 with interleaved gates:
    `op_tests/test_fhmoe.py` has no reference for fp4 activations (separated gates)."""
    unsupported = []
    s = problem.shape
    if (problem.shared.placement, problem.shared.expert_id) != (
        "fused_row",
        s.experts - 1,
    ):
        unsupported.append(f"shared expert {problem.shared}")
    if problem.parallel is not None or problem.tail is not None:
        unsupported.append("a collective or a tail")
    if problem.inter is not None:
        unsupported.append("a caller-fixed GEMM2 input format")
    if problem.layout.hidden_pad or problem.layout.intermediate_pad:
        unsupported.append("padding")
    if problem.features:
        unsupported.append(f"features {sorted(problem.features)}")
    if problem.activation.activation != ActivationType.Silu or not problem.is_g1u1:
        unsupported.append(f"activation {problem.activation.activation}")
    w, a = problem.w1.dtype, problem.act_in.dtype
    found = {dtypes.fp4x2: "fp4", dtypes.fp8: "fp8"}.get(w)
    if (
        found is None
        or a != dtypes.fp8
        or problem.gate_mode != GateMode.INTERLEAVE
        or problem.quant_type != QuantType.per_1x32
    ):
        unsupported.append(
            f"format {problem.quant_type} {a} x {w}, {problem.gate_mode.value} gates"
        )
    if unsupported:
        raise Unsupported("no FHMoE input builder for " + ", ".join(unsupported))
    return found


def _mx_quant(w, fmt):
    """`(quantized, e8m0 scale as uint8 [..., cols // 32])` of bf16 `w`, per 1x32."""
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    if fmt == "fp4":
        qt, scale = per_1x32_f4_quant(w)
        qt = qt.view(torch.uint8).view(*w.shape[:-1], w.shape[-1] // 2)
    else:
        qt, scale = dynamic_mxfp8_quant(w, quant_dtype=dtypes.fp8)
    return qt, scale.view(torch.uint8).view(*w.shape[:-1], w.shape[-1] // 32)


def _mx_dequant(qt, scale, fmt):
    from aiter.utility import fp4_utils

    scale = fp4_utils.e8m0_to_f32(scale.view(dtypes.fp8_e8m0))
    values = fp4_utils.mxfp4_to_f32(qt.view(dtypes.fp4x2)) if fmt == "fp4" else qt
    return values.float() * scale.repeat_interleave(32, dim=-1)


def _act_qdq(x, intermediate=False):
    """`x` (fp32) quantized per 1x32 to fp8 and back, as the kernels quantize their
    GEMM inputs (`intermediate`: the scale the fused intermediate uses)."""
    from aiter.utility import fp4_utils
    from aiter.utility.mx_types import MxDtypeInt

    blocks = x.float().reshape(-1, 32)
    amax = blocks.abs().amax(dim=1)
    scale = (
        fp4_utils.f32_to_fused_moe_mxfp8_scale(amax)
        if intermediate
        else fp4_utils.f32_to_mx_e8m0_scale(amax, dtype=MxDtypeInt.FP8_E4M3)
    )
    scale = fp4_utils.e8m0_to_f32(scale).view(-1, 1)
    return ((blocks / scale).to(dtypes.fp8).float() * scale).view(x.shape)


def _build_fhmoe(problem, seed, device):
    import torch.nn.functional as F

    from aiter.ops.flydsl.moe_kernels import runtime_swiglu_limit
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    w_fmt = _fhmoe_format(problem)
    s, act = problem.shape, problem.activation
    token, H, inter, E, topk = s.m, s.model_dim, s.inter_dim, s.experts, s.topk
    dtype = problem.out_dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    def weights(experts, total, shape, fmt):
        """Raw (quantized, uint8 scale) of `total` experts, the first `experts` random,
        the rest zero weights with unit scales (the routed tensors' shared slot)."""
        cols = shape[1] // 2 if fmt == "fp4" else shape[1]
        qdtype = torch.uint8 if fmt == "fp4" else dtypes.fp8
        qt = torch.zeros((total, shape[0], cols), dtype=qdtype, device=device)
        scale = torch.full(
            (total, shape[0], shape[1] // 32), 0x7F, dtype=torch.uint8, device=device
        )
        for first in range(0, experts, 16):
            n = min(16, experts - first)
            q, sc = _mx_quant(randn(n, *shape) / shape[1] ** 0.5, fmt)
            qt[first : first + n], scale[first : first + n] = q, sc
        return qt, scale

    def kernel_layout(qt, scale, fmt, first):
        weight = qt.view(dtypes.fp4x2) if fmt == "fp4" else qt
        flat = scale.view(-1, scale.shape[-1]).view(dtypes.fp8_e8m0)
        weight = shuffle_weight_a16w4(weight, 16, first)
        flat = shuffle_scale_a16w4(flat, qt.shape[0], first)
        weight.is_shuffled = True
        return weight, flat

    hidden = randn(token, H).to(problem.in_dtype or dtype)
    routed_weight, routed_ids = _route(randn(token, E - 1), topk - 1)
    topk_ids = torch.cat(
        (routed_ids, torch.full_like(routed_ids[:, :1], problem.shared.expert_id)), 1
    )
    topk_weight = torch.cat((routed_weight, torch.ones_like(routed_weight[:, :1])), 1)
    w1_raw = weights(E - 1, E, (2 * inter, H), w_fmt)
    w2_raw = weights(E - 1, E, (H, inter), w_fmt)
    sw1_raw = weights(1, 1, (2 * inter, H), "fp8")
    sw2_raw = weights(1, 1, (H, inter), "fp8")
    w1, w1_scale = kernel_layout(*w1_raw, w_fmt, True)
    w2, w2_scale = kernel_layout(*w2_raw, w_fmt, False)
    shared_w1, shared_w1_scale = kernel_layout(*sw1_raw, "fp8", True)
    shared_w2, shared_w2_scale = kernel_layout(*sw2_raw, "fp8", False)

    call = fused_moe_defaults()
    call.update(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        dtype=dtype,
        swiglu_limit=act.swiglu_limit,
        gate_mode=problem.gate_mode.value,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_w1_scale=shared_w1_scale,
        shared_w2_scale=shared_w2_scale,
        shared_expert_id=problem.shared.expert_id,
    )
    check_keys_on(call, problem)

    limit = runtime_swiglu_limit(act.swiglu_limit, "silu")
    shared_limit = limit if call["clamp_shared"] else float("inf")
    x = _act_qdq(hidden.float())

    def expert(rows, w1_q, w1_s, w2_q, w2_s, fmt, clamp, round_bf16):
        gate_up = F.linear(x[rows], _mx_dequant(w1_q, w1_s, fmt))
        if round_bf16:
            gate_up = gate_up.to(dtypes.bf16).float()
        gate, up = gate_up.chunk(2, dim=-1)
        out = F.silu(gate.clamp(max=clamp)) * up.clamp(min=-clamp, max=clamp)
        if round_bf16:
            out = out.to(dtypes.bf16).float()
        out = _act_qdq(out, intermediate=True)
        return F.linear(out, _mx_dequant(w2_q, w2_s, fmt))

    reference = torch.zeros((token, H), dtype=torch.float32, device=device)
    for e in range(E - 1):
        rows, slots = (routed_ids == e).nonzero(as_tuple=True)
        if rows.numel():
            out = expert(
                rows, w1_raw[0][e], w1_raw[1][e], w2_raw[0][e], w2_raw[1][e],
                w_fmt, limit, False,
            )  # fmt: skip
            reference.index_add_(0, rows, out * routed_weight[rows, slots, None])
    every = torch.arange(token, device=device)
    reference += expert(
        every, sw1_raw[0][0], sw1_raw[1][0], sw2_raw[0][0], sw2_raw[1][0],
        "fp8", shared_limit, True,
    )  # fmt: skip
    return Inputs(call, reference.to(dtype))
