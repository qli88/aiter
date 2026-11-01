# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK-Tile 2-stage MoE (`moe_cktile2stages_gemm1/2`): MXFP4 and int4 weights.

`cktile_moe_stage1`, `cktile_moe_stage2` and `_is_cktile_mxfp4_stage2_name` moved here
verbatim from aiter/fused_moe.py, which imports them back.

Segments: `cktile.gemm1` runs legacy's four CK-Tile branches (gate/up interleaved,
Relu2, split-K, Swiglu without a kernelName1), which differ in the stage 1 split-K, the
activation layout and the `block_m`/`ksplit` of the metadata (`cktile_branch`): as a
default, or for a tuned row named `swiglu_mxfp4_bf16_cktile`, which legacy treats as a
tuned row without kernels (the branches with its split-K). A tuned `cktile_*`
kernelName1 goes to legacy's ASM stage 1 and stays unowned. `cktile.gemm2` also runs the
`cktile_*` stage 2 of tuned rows.
"""

import functools

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch import kernel_names, overrides
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import FrozenMap, ScaleSpec
from ..shapes import get_inter_dim
from .base import Cap, DefaultKernels, RowRejection, Segment
from .defaults import default_ksplit
from .flydsl import (
    _can_reroute_mxfp4_to_flydsl,
    _flydsl_mxfp4_layout_is_compatible,
    _normalize_bias_for_kernel,
)
from .mxfp4 import A16W4_ARCHS, a16w4, mxfp4, mxfp8, swiglu_bias


def cktile_moe_stage1(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    sorted_weights=None,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias1=None,
    topk_ids=None,
    activation=ActivationType.Silu,
    split_k=1,
    dtype=torch.bfloat16,
    kernel_name="",
    post_activation_layout="auto",
):
    token_num = hidden_states.shape[0]
    _, _n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2 * 2  # bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8

    expected_out_shape = (token_num, topk, D)
    if (
        out is None
        or tuple(out.shape) != expected_out_shape
        or out.dtype != dtype
        or out.device != hidden_states.device
    ):
        out = torch.empty(expected_out_shape, dtype=dtype, device=hidden_states.device)
    needs_post_activation = split_k > 1
    # Split-k reduces into a token-topk workspace and applies activation after
    # reduction. Non-split legacy A16W4 keeps CK-Tile's fused gate/up epilogue.
    workspace_rows = token_num * topk
    if needs_post_activation:
        tmp_out = torch.zeros(
            (workspace_rows, w1.shape[1]), dtype=dtype, device=out.device
        )
    else:
        tmp_out = out
    bias1 = _normalize_bias_for_kernel(bias1)
    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        tmp_out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a1_scale,
        w1_scale,
        None if needs_post_activation else bias1,
        activation,
        block_m,
        split_k,
        kernel_name,
    )

    if needs_post_activation:
        valid_out = tmp_out[: token_num * topk, :]
        if post_activation_layout == "auto":
            is_interleaved = (
                hasattr(torch, "float4_e2m1fn_x2")
                and w1.dtype == torch.float4_e2m1fn_x2
            )
        elif post_activation_layout == "interleaved":
            is_interleaved = True
        elif post_activation_layout == "standard":
            is_interleaved = False
        else:
            raise ValueError(
                f"Unsupported CK-Tile post activation layout: {post_activation_layout}"
            )

        if is_interleaved:
            if bias1 is not None:
                raise ValueError("CK-Tile interleaved split-k bias is not supported")
            inter_dim = out.shape[-1]
            if activation == ActivationType.Swiglu:
                from aiter.ops.flydsl.moe_kernels import (
                    flydsl_swiglu_and_mul_interleaved,
                )

                flydsl_swiglu_and_mul_interleaved(
                    valid_out.view(-1, inter_dim * 2),
                    out.view(-1, inter_dim),
                )
            elif activation == ActivationType.Silu:
                from aiter.ops.flydsl.moe_kernels import (
                    flydsl_silu_and_mul_interleaved,
                )

                flydsl_silu_and_mul_interleaved(
                    valid_out.view(-1, inter_dim * 2),
                    out.view(-1, inter_dim),
                    sorted_token_ids,
                    num_valid_ids,
                    token_num,
                    topk,
                )
            elif activation == ActivationType.Gelu:
                NLane = 16
                N0 = inter_dim // NLane
                flat = valid_out.view(-1, N0, 2, NLane)
                gate = flat[:, :, 0, :].reshape(-1, inter_dim)
                up = flat[:, :, 1, :].reshape(-1, inter_dim)
                out.view(-1, inter_dim).copy_(torch.nn.functional.gelu(gate) * up)
            else:
                raise ValueError(
                    f"Unsupported activation for interleaved split-k "
                    f"post-activation: {activation}"
                )
        else:
            if bias1 is not None and topk_ids is None:
                raise ValueError(
                    "topk_ids are required for CK-Tile split-k bias handling"
                )
            expert_ids = topk_ids.view(-1) if topk_ids is not None else None
            if bias1 is not None and activation == ActivationType.Silu:
                aiter.silu_and_mul_bias(out, valid_out, expert_ids, bias1)
            elif bias1 is not None and activation == ActivationType.Swiglu:
                aiter.swiglu_and_mul_bias(out, valid_out, expert_ids, bias1)
            elif bias1 is not None and activation == ActivationType.Gelu:
                aiter.gelu_and_mul_bias(out, valid_out, expert_ids, bias1)
            elif bias1 is not None:
                raise ValueError(
                    f"Unsupported activation for split-k post-activation "
                    f"with bias: {activation}"
                )
            elif activation == ActivationType.Silu:
                aiter.silu_and_mul(out, valid_out)
            elif activation == ActivationType.Swiglu:
                aiter.swiglu_and_mul(out, valid_out)
            elif activation == ActivationType.Gelu:
                aiter.gelu_and_mul(out, valid_out)
            else:
                raise ValueError(
                    f"Unsupported activation for split-k post-activation: {activation}"
                )
    return out


def cktile_moe_stage2(
    a2,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    w2_scale,
    a2_scale,
    block_m,
    activation=ActivationType.Swiglu,
    sorted_weights=None,
    zeros_out=False,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias2=None,
    kernel_name="",
):
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    # Defense in depth for direct callers and tuned metadata.
    if w2.dtype == dtypes.fp4x2 and inter_dim % 256 != 0:
        raise NotImplementedError(
            "CK-Tile MXFP4 stage2 requires inter_dim to be a multiple of 256; "
            f"got {inter_dim}"
        )
    bias2 = _normalize_bias_for_kernel(bias2)
    # print("Run cktile_moe_stage2: M=%d, N=%d, K=%d, topk=%d, expert=%d"%(a2.shape[0]*a2.shape[1], w2.shape[1], a2.shape[2], topk, w2.shape[0]))
    aiter.moe_cktile2stages_gemm2(
        a2,
        w2,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a2_scale,
        w2_scale,
        bias2,
        activation,
        block_m,
        kernel_name=kernel_name,
    )
    return out


def _is_cktile_mxfp4_stage2_name(kernel_name):
    return kernel_name.startswith("cktile_") or (
        kernel_name == "swiglu_mxfp4_bf16_cktile"
    )


def cktile_mxfp4_ok(p) -> bool:
    """The legacy `cktile_mxfp4_ok`: CK-Tile keeps an MXFP4 shape unless its stage 2 is
    unsafe (`inter_dim % 256`, `cktile_mxfp4_unsafe`) and the heuristic FlyDSL path can
    take it over."""
    s = p.shape
    return not (
        cktile_mxfp4_unsafe(p)
        and _can_reroute_mxfp4_to_flydsl(
            model_dim=s.model_dim,
            inter_dim=s.inter_dim,
            dtype=p.out_dtype,
            q_dtype_a=p.act_in.dtype,
            q_dtype_w=p.w1.dtype,
            q_type=p.quant_type,
            activation=p.activation.activation,
            is_shuffled=p.fully_shuffled,
            use_g1u1=p.layout.g1u1,
            doweight_stage1="doweight_stage1" in p.features,
            gate_mode=p.gate_mode,
            gate_up_interleave=p.layout.gate_up_interleave,
        )
    )


def cktile_mxfp4_unsafe(p) -> bool:
    """CK-Tile's MXFP4 stage 2 reads the wrong scale groups when `inter_dim % 256`."""
    return p.w1.dtype == dtypes.fp4x2 and p.shape.inter_dim % 256 != 0


def cktile_interleaved_swiglu(p) -> bool:
    """The legacy CK-Tile branch for MXFP4 Swiglu with gate and up not separated."""
    return (
        p.gate_mode != GateMode.SEPARATED
        and p.out_dtype in (dtypes.bf16, dtypes.fp16)
        and p.quant_type == QuantType.per_1x32
        and p.activation.activation == ActivationType.Swiglu
        and p.w1.dtype == dtypes.fp4x2
        and p.layout.gate_up_interleave in (None, True)
        and cktile_mxfp4_ok(p)
    )


def swiglu_mxfp4_bf16_cktile(p) -> bool:
    """The legacy `swiglu_mxfp4_bf16_cktile`: MXFP4 Swiglu on the standard layout."""
    gate_up_interleave = p.layout.gate_up_interleave
    return (
        p.quant_type == QuantType.per_1x32
        and p.activation.activation == ActivationType.Swiglu
        and p.w1.dtype == dtypes.fp4x2
        and p.any_shuffled
        and cktile_mxfp4_ok(p)
        and (
            gate_up_interleave is False
            # Callers that never tag w1.gate_up_interleave (None) keep the rule
            # from before #6478: bf16/fp16 activations on the standard layout.
            or (
                gate_up_interleave is None
                and p.act_in.dtype in (dtypes.bf16, dtypes.fp16)
            )
        )
    )


def cktile_relu2(p) -> bool:
    """The legacy CK-Tile branch for gate-only Relu2 (bf16 activations, MXFP4 weights)."""
    return (
        p.activation.activation == ActivationType.Relu2
        and not p.layout.g1u1
        and p.quant_type == QuantType.per_1x32
        and p.w1.dtype == dtypes.fp4x2
        and p.act_in.dtype == dtypes.bf16
        and p.out_dtype == dtypes.bf16
        and not {"scatter", "act_scales"} & p.features
        and p.fully_shuffled
        and p.arch == "gfx950"
    )


def flydsl_a16w4(p) -> bool:
    """The legacy `_is_a16w4`: bf16 activations into MXFP4 weights on the FlyDSL a16w4
    kernels, for SiTUv2, and for Swiglu where CK-Tile is unsafe (`cktile_mxfp4_ok`)."""
    activation = p.activation.activation
    return (
        p.out_dtype in (dtypes.bf16, dtypes.fp16)
        and p.quant_type == QuantType.per_1x32
        and p.act_in.dtype == dtypes.bf16
        and p.w1.dtype == dtypes.fp4x2
        and p.any_shuffled
        and p.layout.g1u1
        and "doweight_stage1" not in p.features
        and (
            activation == ActivationType.Situv2
            or (activation == ActivationType.Swiglu and not cktile_mxfp4_ok(p))
        )
    )


def flydsl_mxfp4(p) -> bool:
    """The legacy `use_mxfp4_flydsl`: the heuristic FlyDSL fallback takes the problem
    (bf16 activations: `_is_a16w4`). Legacy also tests it with a tuned row whose kernels
    are neither FlyDSL nor Opus."""
    return flydsl_a16w4(p) or (
        p.out_dtype in (dtypes.bf16, dtypes.fp16)
        and p.quant_type == QuantType.per_1x32
        and (
            p.activation.activation in (ActivationType.Swiglu, ActivationType.Situv2)
            or overrides.flydsl_force()
        )
        and p.act_in.dtype in (dtypes.fp4x2, dtypes.fp8)
        and p.w1.dtype in (dtypes.fp4x2, dtypes.fp8)
        and _flydsl_mxfp4_layout_is_compatible(
            p.act_in.dtype, p.gate_mode, p.layout.gate_up_interleave
        )
        and p.any_shuffled
        and p.layout.g1u1
        and "doweight_stage1" not in p.features
    )


def cktile_split_k(p, ksplit=None) -> bool:
    """The legacy CK-Tile split-K branch: preshuffled MXFP4 weights, not fp4 Swiglu, with
    a split-K (`ksplit`, else the default) above 1 or `swiglu_mxfp4_bf16_cktile`."""
    s = p.shape
    if ksplit is None:
        ksplit = default_ksplit(
            p.quant_type, s.m, s.topk, s.experts, s.inter_dim, s.model_dim
        )
    return (
        p.out_dtype in (dtypes.bf16, dtypes.fp16)
        and p.quant_type == QuantType.per_1x32
        and p.w1.dtype == dtypes.fp4x2
        and p.any_shuffled
        and not (
            p.activation.activation == ActivationType.Swiglu
            and p.act_in.dtype == dtypes.fp4x2
        )
        and (ksplit > 1 or swiglu_mxfp4_bf16_cktile(p))
    )


def tuned_ksplit(row) -> int:
    """The split-K legacy's branches see for a tuned row (`AITER_KSPLIT=-1`: 0)."""
    return 0 if overrides.ksplit_disabled() else int(row.get("ksplit", 0))


def cktile_default_kernels(p) -> DefaultKernels:
    """The legacy CK-Tile branches of `get_2stage_cfgs` without a tuned row, in its
    order. The ranking reaches CK-Tile only where one of them is legacy's choice."""
    s = p.shape
    ksplit = default_ksplit(
        p.quant_type, s.m, s.topk, s.experts, s.inter_dim, s.model_dim
    )
    return cktile_branch(p, ksplit, tuned_row=False)


def cktile_branch(p, ksplit, tuned_row) -> DefaultKernels | None:
    """The CK-Tile branch legacy takes for `p` with the split-K `ksplit` (the default,
    or a tuned row's), in its order: `block_m`, the metadata split-K, the stage 1
    options and the warning; None when legacy takes none of them. `tuned_row`: a
    non-empty kernelName1 rules out the last branch (Swiglu without one)."""
    token = p.shape.m
    tier_block_m = 16 if token < 2048 else 32 if token < 16384 else 64
    if cktile_interleaved_swiglu(p):
        # `get_block_m()`; the metadata keeps the default split-K, stage 1 does not split.
        block_m = 32 if p.act_in.dtype == dtypes.fp8 else tier_block_m
        return DefaultKernels("", "", block_m, ksplit, None, FrozenMap({"split_k": 1}))
    if cktile_relu2(p):
        return DefaultKernels("", "", tier_block_m, 1, None, FrozenMap({"split_k": 1}))
    if cktile_split_k(p, ksplit):
        standard = swiglu_mxfp4_bf16_cktile(p)
        split_k = max(int(ksplit), 2 if standard else 1)
        options = FrozenMap(
            {
                "split_k": split_k,
                "post_activation_layout": "standard" if standard else "auto",
            }
        )
        return DefaultKernels("", "", tier_block_m, split_k, None, options)
    if tuned_row or p.activation.activation != ActivationType.Swiglu:
        return None
    # Swiglu MXFP4 without a kernelName1.
    split_k = max(int(ksplit), 2)
    warning = (
        "[fused_moe] SwiGLU MXFP4 with unshuffled weights not supported "
        "by CK2stages codegen; routing to CK-Tile (ROCM-25478)"
    )
    return DefaultKernels(
        "", "", tier_block_m, split_k, warning, FrozenMap({"split_k": split_k})
    )


def cktile_caps():
    return (
        Cap(
            arch=A16W4_ARCHS,
            act_in=frozenset({a16w4()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"swiglu", "situv2", "relu2"}),
            gate_mode=frozenset({"separated", "interleave"}),
            g1u1=frozenset({True, False}),
            features=frozenset({"ep_mask", "bias1", "bias2", "doweight_stage1"}),
            pads=True,
            extras=frozenset({"disable_inline_sort"}),
        ),
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp4()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "gelu", "swiglu", "situv2"}),
            gate_mode=frozenset({"separated"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2", "doweight_stage1"}),
            pads=True,
            extras=frozenset({"disable_inline_sort"}),
        ),
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp8()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "swiglu", "situv2"}),
            gate_mode=frozenset({"separated", "interleave"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2", "doweight_stage1"}),
            pads=True,
            extras=frozenset({"disable_inline_sort"}),
        ),
    )


class CktileGemm1(Segment):
    """CK-Tile stage 1: a legacy default branch, also for an alias tuned row."""

    name = "cktile.gemm1"
    prefixes = (kernel_names.CKTILE,)
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        return cktile_caps()

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.CKTILE
            and kernel["alias"]
        )

    def declines_row(self, p, row):
        kn2 = str(row.get("kernelName2", "") or "").strip()
        if kn2.startswith("flydsl_"):
            # The tuned FlyDSL branch runs CK stage 1 under the alias name.
            return "alias kernelName1 with a FlyDSL stage 2"
        if kn2 != kernel_names.CKTILE_ALIAS and not kn2.startswith("cktile_"):
            # Legacy's CK-Tile branches run CK-Tile stage 2 whatever the row names.
            return "alias kernelName1 with a stage 2 that is not CK-Tile"
        if cktile_mxfp4_unsafe(p):
            return None  # `cktile.gemm2` discards it
        if not (cktile_interleaved_swiglu(p) or cktile_relu2(p)) and flydsl_mxfp4(p):
            return "the heuristic FlyDSL branch takes the alias tuned row"
        if cktile_branch(p, tuned_ksplit(row), tuned_row=True) is None:
            # Past the CK-Tile branches legacy runs ASM stage 1 under the alias name.
            return "alias tuned row outside the CK-Tile branches"
        return None

    @staticmethod
    def _branch(p, params):
        if not params.kernel:
            return cktile_default_kernels(p)
        return cktile_branch(p, int(params.ksplit), tuned_row=True)

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in))

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(p.out_dtype))

    def default_kernels(self, p):
        return cktile_default_kernels(p)

    def build(self, p, params):
        layout = p.layout
        options = self._branch(p, params).options
        kwargs = {}
        if "post_activation_layout" in options:
            kwargs["post_activation_layout"] = options["post_activation_layout"]
        return functools.partial(
            cktile_moe_stage1,
            n_pad_zeros=layout.intermediate_pad // 64 * 64 * (2 if layout.g1u1 else 1),
            k_pad_zeros=layout.hidden_pad // 128 * 128,
            activation=p.activation.activation,
            split_k=options["split_k"],
            dtype=p.out_dtype,
            **kwargs,
        )

    def metadata_fields(self, p, params):
        fields = {"has_bias": "bias1" in p.features}
        if params.kernel:
            branch = self._branch(p, params)
            fields.update(block_m=branch.block_m, ksplit=branch.ksplit)
        return fields


class CktileGemm2(Segment):
    """CK-Tile stage 2, after a CK-Tile stage 1 or as a tuned row's `cktile_*` stage 2."""

    name = "cktile.gemm2"
    prefixes = (kernel_names.CKTILE,)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return cktile_caps()

    def owns(self, kernel) -> bool:
        return kernel is not None and kernel.family == kernel_names.CKTILE

    def declines_row(self, p, row):
        kn1 = str(row.get("kernelName1", "") or "").strip()
        kn2 = str(row.get("kernelName2", "") or "").strip()
        if kn2 == kernel_names.CKTILE_ALIAS and kn1 != kn2:
            # Legacy's tuned branches build the alias as a CK stage 2 name.
            return "CK-Tile alias stage 2 after another stage 1"
        return None

    def reads(self, p):
        # bf16 when legacy keeps the intermediate unquantized (split-K on preshuffled
        # weights), else what `quant.inter` writes.
        return (Buf("sorted", None), Buf("inter", None))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        kn2 = str(row.get("kernelName2", "") or "").strip()
        if cktile_mxfp4_unsafe(p) and _is_cktile_mxfp4_stage2_name(kn2):
            return RowRejection(
                "[fused_moe] discarding unsafe CK-Tile MXFP4 stage2 config "
                f"for inter_dim={p.shape.inter_dim}; using default heuristics"
            )
        return None

    def build(self, p, params):
        layout = p.layout
        return functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=layout.hidden_pad // 64 * 64,
            k_pad_zeros=layout.intermediate_pad // 128 * 128,
            activation=p.activation.activation,
        )

    def metadata_fields(self, p, params):
        if not params.kernel or params.kernel1 == params.kernel:
            # A legacy CK-Tile branch (a default, or an alias row): the key's bias.
            return {"stage2_has_bias": "bias2" in p.features}
        if params.kernel1.startswith(("flydsl_", "opus_moe1_")):
            # The tuned FlyDSL/Opus branch: `enable_bias`.
            return {"stage2_has_bias": swiglu_bias(p)}
        return {}
