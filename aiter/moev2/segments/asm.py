# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ASM kernels: stage 1 (`_ZN5aiter...fmoe_stage1...E`) and the 1-stage `fmoe` family."""

import csv
import functools
import os
from dataclasses import replace

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, mxfp4_moe_sort_fwd
from aiter import get_hip_quant as get_quant

from ..dispatch import kernel_names
from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec, scale_spec
from ..shapes import get_inter_dim
from .base import Cap, Params, Segment
from .ck import (
    a8w8_blockscale_caps,
    a8w8_per_tensor_caps,
    a8w8_per_token_caps,
    ck_module_builds,
    int8_per_tensor_caps,
    int8_per_token_caps,
)
from .defaults import BLOCK_SIZE_M, TUNE_BLOCK_MS, default_stage1_params, tunable


def asm_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,  # [token_num, topk, inter_dim]
    topk,
    block_m: int,
    kernelName: str = "",
    ksplit: int = 0,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
):
    dtype = dtypes.bf16  # out.dtype, asm only support bf16
    if quant_type != QuantType.per_1x128:
        out = out.view(dtype)
    device = out.device
    token_num, _, _ = out.shape
    E, _model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type == QuantType.per_Tensor:
        a1_scale = a1_scale.view(1, 1).repeat(token_num, 1)
        w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[1])
        quant_type = QuantType.per_Token

    tmp_out = out
    if ksplit > 0:
        tmp_out = torch.zeros(
            (token_num, topk, w1.shape[1]),
            dtype=dtypes.fp32,
            device=device,
        ).view(dtype)

    aiter.moe_stage1_g1u1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        inter_dim,
        kernelName,
        block_m,
        ksplit=ksplit,
        activation=activation,
        quant_type=quant_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
    )
    if ksplit > 0:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32))
        elif activation == ActivationType.Swiglu:
            aiter.swiglu_and_mul(out, tmp_out.view(dtypes.fp32))
        elif activation == ActivationType.Gelu:
            aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32))
        else:
            raise ValueError(
                f"Unsupported activation for split-k post-activation: {activation}"
            )
    return out


def fused_moe_1stage(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    isG1U1,
    block_size_M=32,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    xbf16=False,
    kernelName: str = "",
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: torch.Tensor | None = None,
    M: int | None = None,
    device=None,
    doweight_stage1: bool | None = None,
    flat: int = 0,
):
    if quant_type == QuantType.No and activation == ActivationType.Silu and not isG1U1:
        # pure bf16
        aiter.fmoe(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
        )
    elif quant_type == QuantType.per_Token and doweight_stage1 and isG1U1:
        a8_type = w1.dtype
        _, model_dim, _ = w2.shape

        a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
        a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
        aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)

        aiter.fmoe_g1u1_tkw1(
            moe_buf,
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a8_scale,
            w1_scale,
            w2_scale,
            kernelName,
            a2_scale,
            activation,
        )
    else:
        if xbf16:
            # xquant happens inside the asm kernel for per_1x128
            a1 = hidden_states
            a1_scale = torch.empty(0, device="cuda")
        else:
            quant_func = get_quant(quant_type)
            if hidden_states.dtype != q_dtype_a:
                if quant_type == QuantType.per_1x128:
                    quant_func = functools.partial(quant_func, transpose_scale=True)
                a1, a1_scale = quant_func(
                    hidden_states,
                    scale=a1_scale,
                    quant_dtype=q_dtype_a,
                    num_rows=num_local_tokens,
                )
            else:
                assert (
                    a1_scale is not None or quant_type == QuantType.No
                ), "a1_scale must be provided for quantized input for fused_moe"
                a1 = hidden_states
                if quant_type == QuantType.per_1x128:
                    scale_t = torch.empty_like(a1_scale)
                    aiter.partial_transpose(
                        scale_t, a1_scale, num_rows=num_local_tokens
                    )
                    a1_scale = scale_t

        token_num = hidden_states.shape[0]
        E, model_dim, _inter_dim = get_inter_dim(w1.shape, w2.shape)
        if quant_type == QuantType.per_1x32:
            # FLAT per_1x32 kernels work only with xbf16 on, non-flat ones only
            # with xbf16 off. FLAT: X stays bf16 and is dynamic-quantized to MXFP4
            # inside the kernel, so there is no host activation e8m0 scale to pack.
            # Non-flat: the kernel reads fp4 X, with a host scale that needs
            # sorting. The other pairing faults or gives NaN (todo 5.65).
            if not xbf16:
                a1_scale = mxfp4_moe_sort_fwd(
                    a1_scale,
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token_num,
                    cols=model_dim,
                )
            w1_scale = w1_scale.view(E, -1)
            w2_scale = w2_scale.view(E, -1)

        if quant_type == QuantType.per_1x128:
            fmoe_func = functools.partial(
                aiter.fmoe_fp8_blockscale_g1u1,
                fc_scale_blkn=128,
                fc_scale_blkk=128,
                block_size_M=block_size_M,
            )
        elif isG1U1:
            fmoe_func = aiter.fmoe_g1u1
        else:
            aiter.fmoe_int8_g1u0(
                moe_buf,
                a1,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                a1_scale,
                w1_scale,
                w2_scale,
                fc2_smooth_scale=None,
                activation=activation,
            )
            return moe_buf

        fmoe_func(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernelName,
            fc2_smooth_scale=None,
            activation=activation,
        )
    return moe_buf


def stage2_block_m(block_m):
    """Round a tuned block_m up to one CK stage 2 supports (16, 32, 64 or 128)."""
    # TODO: remove when stage2 support more size
    tmpList = [16, 32, 64, 128]
    if block_m not in tmpList:
        block_m = ([el for el in tmpList if block_m < el] + [128])[0]
    return block_m


def _stage1_kernel_table(p) -> str:
    """The ASM stage 1 kernel table legacy's `gen_2stages_asm1_task` reads for `p`."""
    from aiter.jit.core import AITER_ASM_DIR

    quant = "Int8" if p.act_in.dtype == dtypes.i8 else "Fp8"
    blockscale = "_blockscale" if p.quant_type == QuantType.per_1x128 else ""
    doweight = "_doweight" if "doweight_stage1" in p.features else ""
    return os.path.join(
        AITER_ASM_DIR,
        p.arch,
        "fmoe_2stages",
        f"fmoe_stage1_bf16_pertoken{quant}{blockscale}{doweight}_g1u1.csv",
    )


@functools.cache
def _stage1_kernels(path):
    """`(knl_name, tile_m)` rows of a kernel table; () when the arch ships none."""
    if not os.path.exists(path):
        return ()
    with open(path, newline="") as f:
        return tuple((row["knl_name"], int(row["tile_m"])) for row in csv.DictReader(f))


class AsmStage1(Segment):
    name = "asm.stage1"
    prefixes = (kernel_names.ASM,)
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        # The kernel tables (`get_cfg` in asm_moe_2stage.cu): fp8 with or without
        # doweight, int8 without; per-tensor runs as per-token; g1u1 only.
        int8 = tuple(
            replace(cap, features=cap.features - {"doweight_stage1"})
            for cap in int8_per_tensor_caps() + int8_per_token_caps()
        )
        caps = a8w8_per_tensor_caps() + a8w8_per_token_caps() + int8
        return tuple(replace(cap, g1u1=frozenset({True})) for cap in caps)

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in))

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(dtypes.bf16))

    def default_params(self, p):
        return default_stage1_params(p) if self.reach == "ranked" else None

    def build(self, p, params):
        return functools.partial(
            asm_stage1,
            kernelName=params.kernel,
            activation=p.activation.activation,
            quant_type=p.quant_type,
        )

    def param_space(self, p):
        if not (self.accepts(p) and tunable(p)):
            return ()
        return tuple(
            Params(kernel=name, block_m=tile_m)
            for name, tile_m in _stage1_kernels(_stage1_kernel_table(p))
            if tile_m in TUNE_BLOCK_MS
        )


class AsmStage1Blockscale(AsmStage1):
    """ASM stage 1 for 1x128 fp8: it writes the intermediate already quantized.

    It reads the A1 scales transposed (`quant.a1_transposed`) and writes fp8 rows
    followed by their fp32 scales into one buffer (`out_buffer`); `split_out` cuts that
    into the GEMM2 inputs.
    """

    name = "asm.stage1_1x128"
    reach = "tuned"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    @property
    def capabilities(self):
        return a8w8_blockscale_caps()

    def param_space(self, p):
        """None where its stage 2, CK, does not build (`ck.ck_module_builds`)."""
        return super().param_space(p) if ck_module_builds(p) else ()

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in, scales="transposed"))

    def produces(self, p, params):
        return Buf("inter", p.act_in)

    @staticmethod
    def out_buffer(a1, a1_scale, token_num, topk, inter_dim, q_dtype_a):
        ratio = a1_scale.element_size() // a1.element_size()
        return torch.empty(
            (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
            dtype=q_dtype_a,
            device=a1.device,
        )

    @staticmethod
    def split_out(a2, a1, a1_scale, token_num, topk, inter_dim):
        """`(a2, a2_scale)` from the `out_buffer` stage 1 wrote."""
        ratio = a1_scale.element_size() // a1.element_size()
        a2_scale = (
            a2[token_num:, ...]
            .view(-1)[: token_num * topk * inter_dim * ratio // 128]
            .view(dtypes.fp32)
            .view(token_num, -1)
        )
        return a2[:token_num, :, :], a2_scale


class AsmStage1Cktile(AsmStage1):
    """ASM stage 1 under a tuned mxfp8 x MXFP4 row's `cktile_*` kernelName1: legacy's
    last branch, which the row reaches when no default MXFP4 branch takes it. That
    branch passes no bias."""

    name = "asm.stage1_cktile"
    reach = "tuned"

    @property
    def capabilities(self):
        from .mxfp4 import a8w4_caps

        return tuple(
            replace(cap, features=cap.features - {"bias1", "bias2"})
            for cap in a8w4_caps()
        )

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.CKTILE
            and not kernel["alias"]
        )

    def declines_row(self, p, row):
        from .ck_tile import cktile_branch, flydsl_mxfp4, tuned_ksplit

        kn2 = str(row.get("kernelName2", "") or "").strip()
        if kn2.startswith("flydsl_"):
            # The tuned FlyDSL branch runs CK stage 1 under the name.
            return "cktile_* kernelName1 with a FlyDSL stage 2"
        ksplit = tuned_ksplit(row)
        if flydsl_mxfp4(p) or cktile_branch(p, ksplit, tuned_row=True) is not None:
            return "a default MXFP4 branch takes the cktile_* row"
        return None

    def param_space(self, p):
        return ()


# fmt: off
# The formats a tuned 1-stage row runs beyond the keys of the legacy
# `fused_moe_1stage_dict` (`ranking.FMOE_1STAGE_KEYS`), which only gate legacy's untuned
# default: legacy's tuner times these kernel tables and legacy dispatch runs the row
# without the dict. Same key layout; `_F8` stands for the arch's fp8.
_S, _G = ActivationType.Silu, ActivationType.Gelu
_TOK, _1X128 = QuantType.per_Token, QuantType.per_1x128
_F8 = "fp8"
_BF, _I8 = dtypes.bf16, dtypes.i8
FMOE_1STAGE_TUNED_KEYS = {
    "gfx942": frozenset({
        (_G, _1X128, _BF, _F8, _F8, True, False),
        (_S, _TOK,   _BF, _F8, _F8, True, True),
        (_G, _TOK,   _BF, _F8, _F8, True, True),
    }),
    "gfx950": frozenset({
        (_G, _TOK,   _BF, _F8, _F8, True, True),
        (_S, _TOK,   _BF, _I8, _I8, True, False),
        (_G, _TOK,   _BF, _I8, _I8, True, False),
    }),
}
# fmt: on


def fmoe_1stage_formats(arch):
    """The formats 1-stage `fmoe` runs on `arch`: the `fused_moe_1stage_dict` keys and
    `FMOE_1STAGE_TUNED_KEYS`."""
    from ..dispatch import ranking

    tuned = (
        tuple(dtypes.fp8 if el is _F8 else el for el in key)
        for key in FMOE_1STAGE_TUNED_KEYS.get(arch, ())
    )
    return ranking.fmoe_1stage_keys(arch) | frozenset(tuned)


def fmoe_1stage_caps():
    """One `Cap` per format of `fmoe_1stage_formats`, on the arch that lists it."""
    caps = []
    for arch in ("gfx942", "gfx950"):
        for key in sorted(fmoe_1stage_formats(arch), key=str):
            activation, quant_type, out_dtype, q_dtype_a, q_dtype_w, g1u1, doweight = (
                key
            )
            weights = frozenset({scale_spec(quant_type, q_dtype_w)})
            features = {"ep_mask", "num_local_tokens", "act_scales"}
            if doweight:
                features.add("doweight_stage1")
            caps.append(
                Cap(
                    arch=frozenset({arch}),
                    act_in=frozenset({scale_spec(quant_type, q_dtype_a)}),
                    w1=weights,
                    w2=weights,
                    out_dtype=frozenset({out_dtype}),
                    activation=frozenset({activation.name.lower()}),
                    gate_mode=frozenset({"separated"}),
                    g1u1=frozenset({g1u1}),
                    features=frozenset(features),
                )
            )
    return tuple(caps)


class AsmFmoe1Stage(Segment):
    """1-stage ASM `fmoe` (`fused_moe_1stage`): the activation quant, both GEMMs and the
    weighted sum in one launch, into the sort's output buffer."""

    name = "asm.fmoe_1stage"
    prefixes = (kernel_names.ASM,)
    covers = (Op.QUANT_A1, Op.REDUCE)

    @property
    def capabilities(self):
        return fmoe_1stage_caps()

    def owns(self, kernel) -> bool:
        return super().owns(kernel) and kernel.stage is None

    def reads(self, p):
        # The plain sort, or the FLAT pass-through for FLAT rows.
        return (
            Buf("sorted", None, rows=frozenset({"expert_sorted", "flat"})),
            CALL_INPUT,
        )

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def default_params(self, p):
        """Legacy's default 1-stage call: the launcher's heuristic kernel, `xbf16` for
        the gfx950 1x128 kernels under AITER_XBFLOAT16=1."""
        from ..dispatch import overrides

        xbf16 = (
            p.quant_type == QuantType.per_1x128
            and p.arch == "gfx950"
            and overrides.xbf16()
        )
        return Params(kernel="", block_m=BLOCK_SIZE_M, ksplit=0, xbf16=xbf16, flat=0)

    def build(self, p, params):
        return functools.partial(
            fused_moe_1stage,
            kernelName=params.get("kernel", ""),
            activation=p.activation.activation,
            quant_type=p.quant_type,
            xbf16=params.get("xbf16", False),
            flat=params.get("flat", 0),
        )

    def param_space(self, p):
        """Legacy's `gen_1stage_asm_task`: the kernels of the 1-stage tables (`xbf16`:
        the `blockscaleBf16` table for 1x128 fp8, which legacy reads on gfx950 only),
        `block_m` their `subGU_m`, `flat` from the table. Also per-1x32, which legacy's
        tuner does not time: the `pertokenMXfp4` table, a FLAT kernel with `xbf16`
        unless the input is fp4 already, a non-flat one without it and for fp4 X only.
        A kernel whose `subGU_n` does not divide `inter_dim`, or with `smf`, is left
        out."""
        if not (
            self.accepts(p)
            and tunable(p)
            and p.quant_type not in (QuantType.No, QuantType.per_Tensor)
            and p.w1.dtype != torch.int4
        ):
            return ()
        if p.quant_type == QuantType.per_1x32:
            # A FLAT kernel works only with `xbf16` on, a non-flat one only with it
            # off: a FLAT kernel quantizes bf16 X itself, a non-flat one reads fp4 X
            # and its sorted host scales. The launcher takes either X for any kernel
            # of the table, and the other pairing faults (non-flat, `xbf16` on) or
            # gives NaN (FLAT, `xbf16` off) (todo 5.65). It raises unless `model_dim`
            # is a positive multiple of 512.
            if p.shape.model_dim < 512 or p.shape.model_dim % 512:
                return ()
            fp4_x = p.act_in.dtype == dtypes.fp4x2
            bf16_x = p.in_dtype != dtypes.fp4x2
            return tuple(
                Params(kernel=name, block_m=tile_m, xbf16=bool(flat), flat=flat)
                for name, tile_m, flat in _fmoe_1stage_fits(p, "pertokenMXfp4")
                if (bf16_x if flat else fp4_x)
            )
        quant = {dtypes.i8: "Int8", dtypes.fp8: "Fp8"}.get(p.act_in.dtype, "")
        blockscale = p.quant_type == QuantType.per_1x128
        tables = [(("blockscale" if blockscale else "pertoken") + quant, False)]
        if blockscale and p.act_in.dtype == dtypes.fp8:
            tables.append(("blockscaleBf16", True))
        return tuple(
            Params(kernel=name, block_m=tile_m, xbf16=xbf16, flat=flat)
            for quant_name, xbf16 in tables
            for name, tile_m, flat in _fmoe_1stage_fits(p, quant_name)
        )


def _fmoe_1stage_fits(p, quant):
    """`(knl_name, subGU_m, flat)` of the table with quant part `quant` whose `subGU_n`
    divides `inter_dim`, without `smf`."""
    return tuple(
        (name, tile_m, flat)
        for name, tile_m, tile_n, smf, flat in _fmoe_1stage_kernels(
            _fmoe_1stage_table(p, quant)
        )
        if smf == 0 and p.shape.inter_dim % tile_n == 0
    )


def _fmoe_1stage_table(p, quant) -> str:
    """The path of the 1-stage kernel table with quant part `quant` (`pertokenFp8`, ...)
    for `p`, named as legacy's `gen_1stage_asm_task` names it: the bf16-output table,
    also for fp16 output (todo 5.53). Situv2, which legacy maps to no directory, has
    its per-1x32 table."""
    from aiter.jit.core import AITER_ASM_DIR

    act = {
        ActivationType.Silu: "silu",
        ActivationType.Gelu: "gelu",
        ActivationType.Situv2: "situv2",
    }.get(p.activation.activation, "")
    up = 1 if p.is_g1u1 else 0
    tkw1 = "_tkw1" if "doweight_stage1" in p.features else ""
    name = f"fmoe_bf16_{quant}_g1u{up}_{act}{tkw1}.csv"
    return os.path.join(AITER_ASM_DIR, p.arch, "fmoe", act, name)


@functools.cache
def _fmoe_1stage_kernels(path):
    """`(knl_name, subGU_m, subGU_n, smf, flat)` rows of a 1-stage kernel table; () when
    the arch ships none."""
    if not os.path.exists(path):
        return ()
    with open(path, newline="") as f:
        return tuple(
            (
                row["knl_name"],
                int(row["subGU_m"]),
                int(row["subGU_n"]),
                int(row["smf"]),
                int(float(row.get("flat") or 0)),
            )
            for row in csv.DictReader(f)
        )
