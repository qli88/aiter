# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whole-graph implementations named `impl__<name>__<config>` in tuned tables.

The implementation is looked up in `aiter.fused_moe_registry` (e.g. `flydsl_gfx942`,
registered by aiter.ops.flydsl) and runs the whole MoE from a `FusedMoeRequest`.
"""

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import (
    FusedMoeImplResolutionError,
    make_fused_moe_impl_kernel_name,
    resolve_fused_moe_impl,
)
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch import kernel_names
from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from .base import Params, RowRejection, Segment
from .ck import a8w8_per_tensor_caps, a8w8_per_token_caps, a16w16_caps

__all__ = ["FusedMoeImplResolutionError", "WholeGraph", "resolve_fused_moe_impl"]


class WholeGraph(Segment):
    name = "impl.whole_graph"
    reach = "tuned"
    prefixes = (kernel_names.IMPL,)
    covers = (Op.SORT, Op.REDUCE)

    @property
    def capabilities(self):
        return a8w8_per_tensor_caps() + a8w8_per_token_caps() + a16w16_caps()

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        if kernel_name1.startswith("impl__flydsl_"):
            unsupported = None
            if not p.fully_shuffled:
                unsupported = "both w1 and w2 must be marked is_shuffled=True"
            elif "bias1" in p.features or "bias2" in p.features:
                unsupported = "per-expert bias"
            elif "doweight_stage1" in p.features:
                unsupported = "doweight_stage1=True"
            elif p.in_dtype is not None and p.in_dtype != dtypes.bf16:
                unsupported = f"activation dtype {p.in_dtype}"
            elif "scatter" in p.features:
                unsupported = "stage2_scatter"
            elif "act_scales" in p.features:
                unsupported = "prequantized activations"
            elif "num_local_tokens" in p.features:
                unsupported = "num_local_tokens"
            elif p.layout.hidden_pad or p.layout.intermediate_pad:
                unsupported = "hidden/intermediate padding"
            elif p.gate_mode is not GateMode.SEPARATED:
                unsupported = f"gate mode {p.gate_mode.value!r}"
            elif p.activation.activation not in (
                ActivationType.Silu,
                ActivationType.Swiglu,
            ):
                unsupported = f"activation {p.activation.activation}"
            if unsupported is not None:
                return RowRejection(
                    f"[fused_moe] discarding FlyDSL whole-graph config for "
                    f"{p.log_keys()}: unsupported {unsupported}; using default heuristics"
                )
        if p.is_ep:
            return RowRejection(None)
        return None

    def param_space(self, p):
        """The configs legacy `FmoeTuner.e2e_tune` (`--e2e_tune`) times: those of
        `get_tune_space()` its shape probe (`Config.unsupported_reason`) keeps, where
        `run_flydsl_moe_gfx942` runs (bf16 output, g1u1; bf16 weights unquantized or
        fp8 fnuz weights per-token or per-tensor; the decode config up to 256 tokens,
        with `model_dim` a multiple of 256 and `inter_dim` of 64, as its split-K
        kernels assert), on the capabilities' archs (gfx942, gfx950); none where
        dispatch discards the row (`check_row`: e.g. an activation other than Silu or
        Swiglu, doweight, EP)."""
        if not self.accepts(p) or not p.is_g1u1 or p.out_dtype != dtypes.bf16:
            return ()
        operand_dtypes = {p.w1.dtype, p.w2.dtype, p.act_in.dtype}
        bf16 = p.quant_type == QuantType.No and operand_dtypes == {dtypes.bf16}
        fp8 = p.quant_type in (
            QuantType.per_Token,
            QuantType.per_Tensor,
        ) and operand_dtypes == {torch.float8_e4m3fnuz}
        if not (bf16 or fp8) or self.check_row(p, {}, "impl__flydsl_") is not None:
            return ()
        from aiter.ops.flydsl.fused_moe_gfx942 import Config, _Problem, get_tune_space

        s = p.shape
        probe = _Problem(
            batch=s.m,
            experts=s.experts,
            gateup_dim=2 * s.inter_dim,
            hidden_dim=s.model_dim,
            model_dim=s.model_dim,
            inter_dim=s.inter_dim,
            topk=s.topk,
            quant_type="no" if bf16 else "",
        )
        out = []
        for config_string in get_tune_space():
            config = Config.from_string(config_string)
            if config.unsupported_reason(probe) is not None:
                continue
            if not config.use_prefill and (
                s.m > 256 or s.model_dim % 256 or s.inter_dim % 64
            ):
                continue
            name = make_fused_moe_impl_kernel_name("flydsl_gfx942", config_string)
            out.append(Params(kernel=name, block_m=config.BLOCK_M))
        return tuple(out)

    def build(self, p, params):
        return params.impl
