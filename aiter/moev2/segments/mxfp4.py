# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segments of the MXFP4 (`per_1x32` fp4) FlyDSL kernels: FlyDSL 2-stage and MXMOE.

Which buffers a stage writes depends on its kernel name (a fused fp4 output, the v2
scales of a layout GEMM2, the MXMOE scales), so each kind is its own segment and
the boundary check keeps the legacy pairings: a FlyDSL stage 1 is followed by
`quant.inter` (`flydsl.gemm1`) or writes the fp4 intermediate itself
(`flydsl.gemm1_q`, `flydsl.gemm1_layout`); a layout GEMM2 with the scatter epilog and
MXMOE read the auxiliary sort (`sort.aux`); MXMOE stage 1 reads the activations
`quant.a1` quantized (`mxmoe.gemm1`) or quantizes them itself (`mxmoe.quant_gemm1`).
"""

import dataclasses
import functools
import itertools

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
    get_flydsl_activation_name,
)
from aiter.ops.flydsl.mxfp4_kname import (
    MXFP4_G1_VARIANTS,
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    parse_flydsl_v2_gemm2_kernel,
)

from ..dispatch import kernel_names, overrides
from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec, scale_spec
from .base import Cap, DefaultKernels, Params, RowRejection, Segment
from .defaults import BLOCK_SIZE_M, TUNE_BLOCK_MS, default_nt, tunable
from .flydsl import (
    _flydsl_stage1_wrapper,
    _flydsl_stage2_wrapper,
    _flydsl_v2_stage2_wrapper,
)
from .mxmoe import (
    _is_inline_sort_cfg,
    _is_mxfp4_inline_sort,
    _make_mxfp4_metadata,
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_a4w4_stage2_fw,
)


def mxfp4():
    """fp4 values with one E8M0 scale per 32 (`QuantType.per_1x32`)."""
    return scale_spec(QuantType.per_1x32, dtypes.fp4x2)


def a16w4():
    """bf16 activations into MXFP4 weights: not quantized, `bf16/1x32` as `scale_spec`
    gives it."""
    return scale_spec(QuantType.per_1x32, dtypes.bf16)


def mxfp8():
    """fp8 values with one E8M0 scale per 32 (`QuantType.per_1x32`)."""
    return scale_spec(QuantType.per_1x32, dtypes.fp8)


A16W4_ARCHS = frozenset({"gfx942", "gfx950"})


def wi4():
    """int4 weights with one bf16 scale per 32 (`QuantType.per_1x32`, a16wi4)."""
    return scale_spec(QuantType.per_1x32, dtypes.i4x2)


def a4w4_caps(pads=False):
    """`pads`: the kernels take the hidden/intermediate padding (FlyDSL does, MXMOE
    does not)."""
    return (
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp4()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "gelu", "swiglu", "situv2"}),
            gate_mode=frozenset({"separated"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2"}),
            pads=pads,
            extras=frozenset({"disable_inline_sort"}),
        ),
    )


def a8w4_caps(pads=False):
    """mxfp8 activations into MXFP4 weights. The FlyDSL heuristic runs them with gate
    and up interleaved; tuned rows run on either layout. `pads`: as `a4w4_caps`."""
    return (
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp8()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "swiglu", "situv2"}),
            gate_mode=frozenset({"separated", "interleave"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2"}),
            pads=pads,
            extras=frozenset({"disable_inline_sort"}),
        ),
    )


def a16w4_caps():
    """The FlyDSL a16w4 (`abf16_wfp4`) kernels: SiTUv2, and Swiglu off CK-Tile."""
    return (
        Cap(
            arch=A16W4_ARCHS,
            act_in=frozenset({a16w4()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"swiglu", "situv2"}),
            gate_mode=frozenset({"separated"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2"}),
            pads=True,
            extras=frozenset({"disable_inline_sort"}),
        ),
    )


def a16wi4_caps():
    """The FlyDSL a16wi4 (`abf16_wint4`) kernels: Silu and SiTUv2."""
    return (
        Cap(
            arch=A16W4_ARCHS,
            act_in=frozenset({a16w4()}),
            w1=frozenset({wi4()}),
            w2=frozenset({wi4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "situv2"}),
            gate_mode=frozenset({"separated"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2"}),
            pads=True,
            extras=frozenset({"disable_inline_sort"}),
        ),
    )


def swiglu_bias(p, tuned=True) -> bool:
    """The legacy `enable_bias`: FlyDSL stages add the bias for Swiglu with a 16-bit
    output only, and ignore it otherwise. For a tuned row legacy also requires fp4
    weights; its heuristic does not."""
    return (
        p.activation.activation == ActivationType.Swiglu
        and p.out_dtype in (dtypes.bf16, dtypes.fp16)
        and p.quant_type == QuantType.per_1x32
        and (not tuned or p.w1.dtype == dtypes.fp4x2)
    )


def mxfp8_caps(pads=False):
    """mxfp8 activations and weights: the FlyDSL kernels on gfx950, Silu and Swiglu.
    `pads`: as `a4w4_caps`."""
    return (
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp8()}),
            w1=frozenset({mxfp8()}),
            w2=frozenset({mxfp8()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "swiglu"}),
            gate_mode=frozenset({"separated", "interleave"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask", "bias1", "bias2"}),
            pads=pads,
            extras=frozenset({"disable_inline_sort"}),
        ),
    )


def _row_names(row):
    return (
        str(row.get("kernelName1", "") or "").strip(),
        str(row.get("kernelName2", "") or "").strip(),
    )


TUNED_ROW_ACTIVATIONS = (
    ActivationType.Silu,
    ActivationType.Swiglu,
    ActivationType.Situv2,
)


def mxfp4_row_rejection(p, row) -> RowRejection | None:
    """The legacy `get_2stage_cfgs` checks, in its order, that discard a tuned row with
    an Opus, FlyDSL or MXMOE kernel. (Its stage 2 checks are with `cktile.gemm2` and
    `opus.gemm2`.)"""
    kn1, kn2 = _row_names(row)
    activation = p.activation.activation
    inline = _is_inline_sort_cfg(kn1, kn2)
    if (kn1.startswith("opus_") or kn2.startswith("opus_")) and not p.fully_shuffled:
        return RowRejection(
            f"[fused_moe] discarding Opus tuned config for {p.log_keys()}: "
            "both w1 and w2 must be marked is_shuffled=True; using default "
            "heuristics"
        )
    if kn1.startswith("opus_") and activation not in TUNED_ROW_ACTIVATIONS:
        return RowRejection(
            f"[fused_moe] discarding Opus tuned config for unsupported "
            f"activation {activation}; using default heuristics"
        )
    if (
        kn1.startswith("flydsl_") or kn2.startswith("flydsl_")
    ) and activation not in TUNED_ROW_ACTIVATIONS:
        return RowRejection(
            f"[fused_moe] discarding FlyDSL tuned config for unsupported "
            f"activation {activation}; using default heuristics"
        )
    if p.extras.get("disable_inline_sort") and inline:
        return RowRejection("[fused_moe] discarding tuned inline-sort config")
    if inline and p.is_ep:
        return RowRejection(
            "[fused_moe] discarding tuned inline-sort config with expert_mask"
        )
    if inline:
        inline_metadata = _make_mxfp4_metadata(
            kn1,
            kn2,
            p.gate_mode,
            row.get("ksplit", 0),
            block_m=row.get("block_m", BLOCK_SIZE_M),
            run_1stage=row.get("run_1stage", False),
            has_stage1_bias="bias1" in p.features,
            has_stage2_bias="bias2" in p.features,
        )
        if not _is_mxfp4_inline_sort(inline_metadata):
            return RowRejection(
                "[fused_moe] discarding tuned inline-sort config with "
                "incomplete metadata"
            )
    if not _is_mxfp4_kname(kn1):
        return None
    s = p.shape
    parsed_g1 = _parse_mxfp4_g1_kname(kn1)
    configured_act = parsed_g1["act"]
    try:
        expected_act = get_flydsl_activation_name(activation)
    except ValueError:
        expected_act = None
    swiglu_limit = p.activation.swiglu_limit
    reason = None
    if expected_act is None:
        reason = f"no MXMOE kernel for activation {activation!r}"
    elif configured_act != expected_act:
        reason = (
            f"activation {configured_act!r} does not match runtime {expected_act!r}"
        )
    elif swiglu_limit and expected_act not in ("silu", "swiglu"):
        reason = (
            f"MXMOE cannot apply swiglu_limit={swiglu_limit!r} to "
            f"activation {expected_act!r}"
        )
    elif not aiter.is_mxfp4_moe_shape_supported(
        s.experts, s.model_dim, s.inter_dim, s.topk
    ):
        reason = (
            "generated MXFP4 auxiliary kernels do not cover "
            f"(expert={s.experts}, model_dim={s.model_dim}, "
            f"inter_dim={s.inter_dim}, topk={s.topk})"
        )
    elif "bias1" in p.features and not parsed_g1.get("enable_bias", False):
        reason = "stage1 bias is present but kernelName1 lacks '_bias'"
    elif "bias2" in p.features and parse_flydsl_v2_gemm2_kernel(kn2) is None:
        reason = f"stage2 bias requires a flydsl_moe2_layout_ kernel, got {kn2!r}"
    elif p.is_ep:
        reason = "the MXMOE output_aux sort drops expert_mask"
    if reason is None:
        return None
    return RowRejection(
        f"[fused_moe] discarding MXMOE config: {reason}; using default heuristics"
    )


def flydsl_default_kernels(p) -> DefaultKernels:
    """The legacy heuristic FlyDSL fallback (`use_mxfp4_flydsl` without a tuned row)."""
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_kernel_name,
        get_flydsl_kernel_params,
        pick_flydsl_stage2_tile_k,
    )

    token = p.shape.m
    if p.w1.dtype == dtypes.i4x2:
        return _a16wi4_default_kernels(p)
    out_type = "bf16" if p.out_dtype == dtypes.bf16 else "f16"
    act = p.act_in.dtype
    a_type = "bf16" if act == dtypes.bf16 else "fp4" if act == dtypes.fp4x2 else "fp8"
    w_type = "fp8" if p.w1.dtype == dtypes.fp8 else "fp4"
    s2_tk = pick_flydsl_stage2_tile_k(p.shape.inter_dim)
    # Per token tier: (tile_m, stage1 suffix, stage2 suffix).
    if token < 2048:
        tile_m, s1_sfx, s2_sfx = 32, "_w2", "_bnt2"
    elif token < 4096:
        tile_m, s1_sfx, s2_sfx = 64, "_w3_bnt0", ""
    elif token < 16384:
        tile_m, s1_sfx, s2_sfx = 128, "_w2_bnt0", ""
    else:
        tile_m, s1_sfx, s2_sfx = 64, "_w4_bnt0", ""
    base_kn1 = flydsl_kernel_name(1, a_type, w_type, out_type, tile_m, 128, 256)
    base_kn2 = flydsl_kernel_name(
        2, a_type, w_type, out_type, tile_m, 128, s2_tk, "atomic"
    )
    kn1 = f"{base_kn1}{s1_sfx}"
    kn2 = f"{base_kn2}{s2_sfx}"
    # fp8 stage1 kernel names always carry a "_gui" suffix.
    if a_type == "fp8":
        kn1 = f"{kn1}_gui"
        base_kn1 = f"{base_kn1}_gui"
    if get_flydsl_kernel_params(kn1) is None:
        kn1 = base_kn1
    if get_flydsl_kernel_params(kn2) is None:
        kn2 = base_kn2
    warning = (
        f"[fused_moe] no tuned FlyDSL config for {p.log_keys()}, "
        f"using heuristic FlyDSL fallback ({kn1=}, {kn2=})"
    )
    return DefaultKernels(kn1, kn2, tile_m, -1, warning)


def _a16wi4_default_kernels(p) -> DefaultKernels:
    """The legacy untuned a16wi4 fallback: one tile per token tier, bf16 output names
    whatever the output dtype, no split-K (the kernels split K inside a block)."""
    from aiter.ops.flydsl.moe_kernels import flydsl_kernel_name

    token = p.shape.m
    tile_m = 16 if token < 2048 else 32 if token < 16384 else 64
    kn1 = flydsl_kernel_name(1, "bf16", "int4", "bf16", tile_m, 128, 128)
    kn2 = flydsl_kernel_name(2, "bf16", "int4", "bf16", tile_m, 128, 128, "atomic")
    warning = (
        f"[fused_moe] no tuned FlyDSL config for {p.log_keys()}, "
        f"using untuned a16wi4 fallback ({kn1=}, {kn2=})"
    )
    return DefaultKernels(kn1, kn2, tile_m, 0, warning)


def mxmoe_default_kernels(p) -> DefaultKernels:
    """The legacy heuristic MXMOE fallback (SiTUv2 without a tuned row)."""
    s = p.shape
    token, topk, expert = s.m, s.topk, s.experts
    bm = 64 if token < 512 else 128
    rows_per_expert = -(-token * topk // expert)
    g1_swz = min(6, max(1, -(-rows_per_expert // bm)))
    g1_sfx = f"_xcd{g1_swz}" if g1_swz > 1 else ""
    kn1 = f"flydsl_mxmoe_g1_a4w4_{bm}x256x256_situv2{g1_sfx}"
    kn2 = f"flydsl_moe2_layout_afp4_wfp4_bf16_t{bm}x256x128_reduce_sbm{bm}"
    warning = (
        f"[fused_moe] no tuned FlyDSL config for {p.log_keys()}, "
        f"using heuristic MXMOE fallback (kn1={kn1!r}, kn2={kn2!r})"
    )
    return DefaultKernels(kn1, kn2, bm, 0, warning)


def _stage1_default_params(p, branch: DefaultKernels):
    return Params(
        kernel=branch.kernel1,
        block_m=branch.block_m,
        ksplit=branch.ksplit,
        nt=default_nt(p),
    )


def _stage1_fused_quant(kernel_name) -> str:
    """What a FlyDSL stage 1 quantizes its output to, from its name ("" for none)."""
    options = kernel_name.split("_t")[-1]
    return "fp8" if "_fp8" in options else ("fp4" if "_fp4" in options else "")


@functools.cache
def _flydsl_kernels(lister, *args) -> dict:
    """A FlyDSL kernel registry (`aiter.ops.flydsl.moe_kernels.<lister>`); read only."""
    from aiter.ops.flydsl import moe_kernels

    return getattr(moe_kernels, lister)(*args)


def _split_k_fits(k, model_dim) -> bool:
    kb, kw, tk = k.get("k_batch", 1), k.get("k_wave", 1), k["tile_k"]
    if model_dim % kb or (model_dim // kb) % tk:
        return False
    k_per_batch = model_dim // kb
    return kw <= 1 or not (k_per_batch % kw or (k_per_batch // kw) % tk)


def _out_token(p) -> str:
    return "bf16" if p.out_dtype == dtypes.bf16 else "f16"


@functools.cache
def _flydsl_candidates(a, b, out, model_dim, inter_dim) -> tuple:
    """`(stage, kernel, block_m)` of the legacy tuner's `gen_flydsl_2stages_task`."""
    s1 = _flydsl_kernels("get_flydsl_stage1_kernels", a, b, out)
    s2 = _flydsl_kernels("get_flydsl_stage2_kernels", a, b, out)
    fused = {"fp8": "_fp8", "fp4": "_fp4"}.get(a)
    out_list = []
    for block_m in TUNE_BLOCK_MS:
        for name, k in s1.items():
            if k["tile_m"] != block_m or not _split_k_fits(k, model_dim):
                continue
            # a16w4: each of the 4 // k_wave N-waves needs whole 16-wide MFMA tiles.
            if a == "bf16" and k["tile_n"] % (16 * max(1, 4 // k.get("k_wave", 1))):
                continue
            if fused is None:
                names = (name,)
            elif k.get("k_batch", 1) > 1:
                names = (name + fused,)
            else:
                names = (name, name + fused)
            out_list += [(1, n, block_m) for n in names]
        for name, k in s2.items():
            tile_m = k["tile_m"]
            if block_m % tile_m or tile_m not in (block_m, block_m // 2):
                continue
            lds = tile_m * (k["tile_k"] * 2 + k["tile_n"] * 4)
            if a == "bf16" and (
                tile_m != block_m or lds > 160 * 1024 or inter_dim % k["tile_k"]
            ):
                continue
            sbm = "" if tile_m == block_m else f"_sbm{block_m}"
            out_list.append((2, name + sbm, block_m))
    return tuple(out_list)


@functools.cache
def _flydsl_int4_candidates(out, model_dim) -> tuple:
    """`(stage, kernel, block_m)` of the legacy tuner's `gen_flydsl_i4_2stages_task`."""
    s1 = _flydsl_kernels("get_flydsl_stage1_kernels_int4_bf16", out)
    s2 = _flydsl_kernels("get_flydsl_stage2_kernels_int4_bf16", out)
    out_list = []
    for block_m in TUNE_BLOCK_MS:
        for name, k in s1.items():
            kb, tk = k.get("k_batch", 1), k["tile_k"]
            k_tiles = (model_dim // kb) // tk
            if k["tile_m"] != block_m or (
                kb > 1
                and (
                    model_dim % kb
                    or (model_dim // kb) % tk
                    or k_tiles < 4
                    or k_tiles % 2
                )
            ):
                continue
            out_list.append((1, name, block_m))
        out_list += [
            (2, name, block_m)
            for name, k in s2.items()
            if k["tile_m"] == block_m
            and k.get("mode", "atomic") == "atomic"
            and k.get("persist") is None
        ]
    return tuple(out_list)


@functools.cache
def _flydsl_layout_candidates(a, b, model_dim, inter_dim) -> tuple:
    """`(stage, kernel, block_m)` of the legacy tuner's `gen_flydsl_v2_2stages_task`:
    the fused stage 1 without split-K, and the layout GEMM2s."""
    s1 = _flydsl_kernels("get_flydsl_stage1_kernels", a, b, "bf16")
    out_list = []
    for block_m in (32, 64, 128):
        out_list += [
            (1, f"{name}_{a}", block_m)
            for name, k in s1.items()
            if k.get("tile_m") == block_m and k.get("k_batch", 1) == 1
        ]
        s2 = _flydsl_kernels(
            "get_flydsl_stage2_v2_kernels", a, b, "bf16", block_m, model_dim, inter_dim
        )
        out_list += [(2, name, block_m) for name in s2]
    return tuple(out_list)


def flydsl_candidates(p, layout) -> tuple:
    """The legacy tuner's FlyDSL candidates for `p`: the layout ones (`layout`), else
    the token-major and int4 ones."""
    if p.quant_type != QuantType.per_1x32:
        return ()
    s = p.shape
    a, b = mx_token(p.act_in.dtype), mx_token(p.w1.dtype)
    if layout:
        if p.out_dtype != dtypes.bf16 or "doweight_stage1" in p.features:
            return ()
        if (a, b) not in (("fp4", "fp4"), ("fp8", "fp4"), ("fp8", "fp8")):
            return ()
        return _flydsl_layout_candidates(a, b, s.model_dim, s.inter_dim)
    if p.w1.dtype == dtypes.i4x2:
        return _flydsl_int4_candidates(_out_token(p), s.model_dim)
    if not b:
        return ()
    a = {dtypes.bf16: "bf16", dtypes.fp16: "fp16"}.get(p.act_in.dtype, a or "fp8")
    return _flydsl_candidates(a, b, _out_token(p), s.model_dim, s.inter_dim)


def flydsl_param_space(segment, p, layout) -> tuple:
    """The FlyDSL candidates `segment` runs: its stage, and the kernels it owns. None
    for an activation whose FlyDSL tuned rows dispatch discards (the legacy tuner lists
    them)."""
    if not (segment.accepts(p) and tunable(p)):
        return ()
    if p.activation.activation not in TUNED_ROW_ACTIVATIONS:
        return ()
    stage = 1 if Op.GEMM1 in segment.covers else 2
    return tuple(
        Params(kernel=name, block_m=block_m)
        for st, name, block_m in flydsl_candidates(p, layout)
        if st == stage and segment.owns(kernel_names.parse(name))
    )


def _mxmoe_g1_name(bm, bn, use_nt, inline_quant, hpf, act, k_wave, xcd, num_waves):
    """Legacy `Mxfp4FlydslTuner._g1_kname`, in `_parse_mxfp4_g1_kname`'s token order."""
    name = f"flydsl_mxmoe_g1_a4w4_{bm}x{bn}x256"
    name += "_f16in" if inline_quant else ""
    name += "_hpf" if hpf else ""
    name += "_nt" if use_nt else ""
    name += {"situv2": "_situv2", "swiglu": "_swiglu"}.get(act, "")
    name += f"_kw{k_wave}" if k_wave > 1 else ""
    name += f"_xcd{xcd}" if xcd else ""
    return name + ("_w2" if num_waves == 2 else "")


@functools.cache
def _mxmoe_candidates(act, model_dim, inter_dim) -> tuple:
    """`(block_m, stage 1 names, stage 2 names)` of the legacy
    `Mxfp4FlydslTuner._candidate_rows` at its full search (`--mxfp4-search-mode full`;
    the default prune by tokens per expert is a tuner filter), which pairs every name
    of a `block_m`'s stage 1 with every one of its stage 2: the a4w4 stage 1 variants
    the kernel's `_assert_supported` accepts, and the layout GEMM2s with that `tile_m`;
    a `block_m` without both is left out."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    stage1 = {}
    for bm, use_nt, inline_quant in sorted(MXFP4_G1_VARIANTS["fp4"]):
        for bn, num_waves, k_wave in itertools.product(
            (64, 128, 256), (4, 2), (1, 2, 4)
        ):
            for hpf in (False, True) if inline_quant else (False,):
                try:
                    _assert_supported(
                        D_HIDDEN=model_dim,
                        D_INTER=inter_dim,
                        BM=bm,
                        BN=bn,
                        BK=256,
                        use_nt=use_nt,
                        inline_quant=inline_quant,
                        prefetch_hidden=hpf,
                        act=act,
                        situ_beta=DEFAULT_SITUV2_BETA,
                        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
                        num_waves=num_waves,
                        k_wave=k_wave,
                        native_scale_layout=bm == 16,
                    )
                except NotImplementedError:
                    continue
                for xcd in (0, 2, 4):
                    name = _mxmoe_g1_name(
                        bm, bn, use_nt, inline_quant, hpf, act, k_wave, xcd, num_waves
                    )
                    stage1.setdefault(bm, []).append(name)
    out = []
    for bm, names in sorted(stage1.items()):
        s2 = _flydsl_kernels(
            "get_flydsl_stage2_v2_kernels",
            "fp4",
            "fp4",
            "bf16",
            bm,
            model_dim,
            inter_dim,
        )
        s2 = tuple(name for name, k in s2.items() if k["tile_m"] == bm)
        if s2:
            out.append((bm, tuple(names), s2))
    return tuple(out)


def mxmoe_param_space(segment, p) -> tuple:
    """The MXMOE candidates `segment` owns, of its stage, each in a pair dispatch does
    not discard (`mxfp4_row_rejection`: e.g. a shape the generated auxiliary kernels do
    not cover). None for fp16 output (the legacy tuner times bf16 whatever the key's
    dtype) or an activation MXMOE has no kernel for."""
    if not (segment.accepts(p) and tunable(p)) or p.out_dtype != dtypes.bf16:
        return ()
    if p.activation.activation not in TUNED_ROW_ACTIVATIONS:
        return ()
    act = get_flydsl_activation_name(p.activation.activation)
    gemm2 = Op.GEMM2 in segment.covers
    out = []
    for block_m, names1, names2 in _mxmoe_candidates(
        act, p.shape.model_dim, p.shape.inter_dim
    ):
        names, partners = (names2, names1) if gemm2 else (names1, names2)
        for name in names:
            if not segment.owns(kernel_names.parse(name)):
                continue
            pairs = (
                (partner, name) if gemm2 else (name, partner) for partner in partners
            )
            row = {"block_m": block_m, "ksplit": 0}
            if any(
                mxfp4_row_rejection(p, {**row, "kernelName1": kn1, "kernelName2": kn2})
                is None
                for kn1, kn2 in pairs
            ):
                out.append(Params(kernel=name, block_m=block_m))
    return tuple(out)


class _FlydslStage1(Segment):
    prefixes = (kernel_names.FLYDSL,)

    @property
    def capabilities(self):
        return a4w4_caps(pads=True) + a8w4_caps(pads=True) + mxfp8_caps(pads=True)

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def row_params(self, p, row):
        return {"tuned": True}

    def build(self, p, params):
        return functools.partial(
            _flydsl_stage1_wrapper,
            kernelName=params.kernel,
            activation=p.activation.activation,
            inter_dim_pad=p.layout.intermediate_pad,
            model_dim_pad=p.layout.hidden_pad,
        )

    def metadata_fields(self, p, params):
        return {"has_bias": swiglu_bias(p, tuned=params.get("tuned", False))}

    def param_space(self, p):
        return flydsl_param_space(self, p, layout=False)


class FlydslGemm1(_FlydslStage1):
    """A FlyDSL stage 1 writing the unquantized intermediate."""

    name = "flydsl.gemm1"
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        return (
            a4w4_caps(pads=True)
            + a8w4_caps(pads=True)
            + mxfp8_caps(pads=True)
            + a16w4_caps()
            + a16wi4_caps()
        )

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 1
            and _stage1_fused_quant(kernel.format()) == ""
        )

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(p.out_dtype))

    def default_params(self, p):
        return _stage1_default_params(p, flydsl_default_kernels(p))

    def default_warning(self, p):
        return flydsl_default_kernels(p).warning


class FlydslGemm1Quant(_FlydslStage1):
    """A FlyDSL stage 1 that quantizes its output to MXFP4 or mxfp8 (`_fp4` or `_fp8`
    in its options), in the v2 layout for an Opus layout GEMM2."""

    name = "flydsl.gemm1_q"
    reach = "tuned"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 1
            and _stage1_fused_quant(kernel.format()) != ""
        )

    def declines_row(self, p, row):
        kn1, _ = _row_names(row)
        fused = dtypes.fp8 if _stage1_fused_quant(kn1) == "fp8" else dtypes.fp4x2
        if kn1.startswith("flydsl_moe1_") and fused != p.act_in.dtype:
            return "FlyDSL stage 1 quantizes to another dtype than the activations"
        return None

    def produces(self, p, params):
        kernel = params.get("kernel")
        fmt = p.act_in if kernel is None else mx_inter(_stage1_fused_quant(kernel))
        return fused_inter(fmt, params.get("kernel2"))


def fused_inter(fmt, kernel2) -> Buf:
    """The intermediate a FlyDSL or Opus stage 1 quantizing it to `fmt` writes for
    GEMM2 `kernel2` (None before it is chosen): for a layout name (`_moe2_layout_`,
    legacy `skip_inter_quant`) the executor has it write the v2 layout, else
    token-major."""
    if kernel2 and "_moe2_layout_" in kernel2:
        return Buf("inter", fmt, rows="expert_sorted", scales="v2")
    return Buf("inter", fmt)


def _mx_dtypes() -> dict:
    # dtypes.fp8 depends on the arch, so it is read per call.
    return {"fp4": dtypes.fp4x2, "fp8": dtypes.fp8}


def mx_inter(a_dtype) -> ScaleSpec | None:
    """The per-1x32 intermediate a FlyDSL name's dtype token ("fp4", "fp8") stands
    for; None (any) for no token or another one."""
    dtype = _mx_dtypes().get(a_dtype)
    return None if dtype is None else scale_spec(QuantType.per_1x32, dtype)


def mx_token(dtype) -> str:
    """The FlyDSL name token ("fp4", "fp8") of a per-1x32 dtype; "" for another."""
    return next((t for t, d in _mx_dtypes().items() if d == dtype), "")


def _reads_mx_inter(cfg) -> bool:
    """A parsed layout GEMM2 name reads an fp4 or fp8 intermediate; legacy would hand
    stage 1 another token as `fuse_quant`, which no boundary describes, so such a name
    is not owned and legacy runs the call."""
    return cfg is not None and mx_inter(cfg["a_dtype"]) is not None


def layout_gemm2_inter(kernel2) -> ScaleSpec | None:
    """The intermediate a layout GEMM2 name reads; None before it is chosen."""
    cfg = parse_flydsl_v2_gemm2_kernel(kernel2) if kernel2 else None
    return None if cfg is None else mx_inter(cfg["a_dtype"])


def layout_gemm2_reads(reads, kernel) -> tuple:
    """`reads` (sort, intermediate) of a segment running layout GEMM2 `kernel`,
    narrowed to the sort block its name fixes (`sbm`; without it the kernel takes the
    sort's) and the intermediate format it reads."""
    cfg = parse_flydsl_v2_gemm2_kernel(kernel) if kernel else None
    if cfg is None:
        return reads
    sort, inter = reads
    fmt = mx_inter(cfg["a_dtype"])
    return (
        dataclasses.replace(sort, block_m=cfg["sort_block_m"] or None),
        inter if fmt is None else dataclasses.replace(inter, fmt=fmt),
    )


class FlydslGemm1Layout(_FlydslStage1):
    """A FlyDSL stage 1 writing the sorted, quantized input of a layout GEMM2."""

    name = "flydsl.gemm1_layout"
    reach = "tuned"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 1
        )

    def reads(self, p):
        return (
            Buf("sorted", None, rows=frozenset({"expert_sorted", "aux_sorted"})),
            Buf("tokens", p.act_in),
        )

    def produces(self, p, params):
        return Buf(
            "inter",
            layout_gemm2_inter(params.get("kernel2")),
            rows="expert_sorted",
            scales="v2",
        )

    def build(self, p, params):
        stage1 = super().build(p, params)
        stage1.keywords["out_dtype"] = self._a_dtype(params)
        return stage1

    @staticmethod
    def _a_dtype(params):
        return parse_flydsl_v2_gemm2_kernel(params.kernel2)["a_dtype"]

    def effective_kernel(self, p, params):
        """The kernel's registry params with the `out_dtype` `build` gives it: a plain
        stage 1 name under a layout GEMM2 runs as its `_fp4` / `_fp8` candidate. The
        name itself if the registry does not parse it."""
        from .flydsl import _get_flydsl_moe_kernels

        kernel = params.get("kernel", "")
        parsed = _get_flydsl_moe_kernels().get_flydsl_kernel_params(kernel)
        if parsed is None:
            return kernel
        kernel2 = parse_flydsl_v2_gemm2_kernel(params.get("kernel2") or "")
        if kernel2 is not None:
            parsed = {**parsed, "out_dtype": kernel2["a_dtype"]}
        return Params(parsed)

    def param_space(self, p):
        return flydsl_param_space(self, p, layout=True)


class FlydslGemm2(Segment):
    name = "flydsl.gemm2"
    prefixes = (kernel_names.FLYDSL,)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return (
            a4w4_caps(pads=True)
            + a8w4_caps(pads=True)
            + mxfp8_caps(pads=True)
            + a16w4_caps()
            + a16wi4_caps()
        )

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 2
        )

    def reads(self, p):
        return (Buf("sorted", None), Buf("inter", p.act_in))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def default_params(self, p):
        return Params(kernel=flydsl_default_kernels(p).kernel2)

    def build(self, p, params):
        return functools.partial(
            _flydsl_stage2_wrapper,
            kernelName=params.kernel,
            inter_dim_pad=p.layout.intermediate_pad,
            model_dim_pad=p.layout.hidden_pad,
        )

    def row_params(self, p, row):
        return {"tuned": True}

    def metadata_fields(self, p, params):
        return {"stage2_has_bias": swiglu_bias(p, tuned=params.get("tuned", False))}

    def param_space(self, p):
        return flydsl_param_space(self, p, layout=False)


class FlydslLayoutGemm2(Segment):
    """A FlyDSL v2 GEMM2 (`flydsl_moe2_layout_*`) with the atomic or reduce epilog."""

    name = "flydsl_layout.gemm2"
    reach = "tuned"
    prefixes = (kernel_names.FLYDSL_LAYOUT,)
    covers = (Op.GEMM2, Op.REDUCE)
    scatter = False

    @property
    def capabilities(self):
        return a4w4_caps() + a8w4_caps() + mxfp8_caps()

    def owns(self, kernel) -> bool:
        if kernel is None or kernel.family != kernel_names.FLYDSL_LAYOUT:
            return False
        cfg = parse_flydsl_v2_gemm2_kernel(kernel.format())
        return _reads_mx_inter(cfg) and (cfg["epilog"] == "scatter") == self.scatter

    def reads(self, p):
        return (
            Buf("sorted", None),
            Buf("inter", None, rows="expert_sorted", scales="v2"),
        )

    def reads_with(self, p, params):
        return layout_gemm2_reads(self.reads(p), params.get("kernel"))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def stage2_kernel(self, p, kernel_name1, kernel_name2):
        """An fp8 GEMM2 input (`q_dtype_a2`, else AITER_SITUV2_A4W4_FP8_INTER) renames
        an fp4 layout GEMM2 to its fp8 twin."""
        fp8_inter = (
            p.inter.dtype == dtypes.fp8
            if p.inter is not None
            else overrides.a4w4_fp8_inter()
        )
        if (
            fp8_inter
            and p.act_in.dtype == dtypes.fp4x2
            and kernel_name2.startswith("flydsl_moe2_layout_afp4_")
            and kernel_name1.startswith("flydsl_moe1_")
        ):
            return kernel_name2.replace("_afp4_", "_afp8_", 1)
        return kernel_name2

    def build(self, p, params):
        s = p.shape
        return functools.partial(
            _flydsl_v2_stage2_wrapper,
            kernelName=params.kernel,
            model_dim=s.model_dim,
            inter_dim=s.inter_dim,
            num_experts=s.experts,
        )

    def metadata_fields(self, p, params):
        return {"stage2_has_bias": swiglu_bias(p)}

    def param_space(self, p):
        return flydsl_param_space(self, p, layout=True)


class FlydslLayoutGemm2Scatter(FlydslLayoutGemm2):
    """A FlyDSL v2 GEMM2 with the scatter epilog: it reads the auxiliary sort."""

    name = "flydsl_layout.gemm2_scatter"
    scatter = True

    def reads(self, p):
        return (
            Buf("sorted", None, rows="aux_sorted"),
            Buf("inter", None, rows="expert_sorted", scales="v2"),
        )


def _mxmoe_quantizes_a1(kernel_name) -> bool:
    """Whether MXMOE stage 1 `kernel_name` is handed the caller's activations: legacy
    `prequant` is False and its override (any block_m but 16) does not apply."""
    p1 = _parse_mxfp4_g1_kname(kernel_name)
    return p1["BM"] == 16 and not (p1["a_dtype"] == "fp8" and not p1["inline_quant"])


class MxmoeGemm1(Segment):
    """MXMOE stage 1 (`flydsl_mxmoe_g1_*`) on the prequantized activations (`quant.a1`):
    it writes its own sorted, quantized intermediate."""

    name = "mxmoe.gemm1"
    prefixes = (kernel_names.MXMOE,)
    covers = (Op.GEMM1, Op.QUANT_INTER)

    @property
    def capabilities(self):
        return a4w4_caps()

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.MXMOE
            and kernel.stage == 1
            and _mxmoe_quantizes_a1(kernel.format()) == (self.covers[0] == Op.QUANT_A1)
        )

    def reads(self, p):
        return (Buf("sorted", None, rows="aux_sorted"), Buf("tokens", p.act_in))

    def produces(self, p, params):
        kernel = params.get("kernel")
        fmt = mx_inter(_parse_mxfp4_g1_kname(kernel)["out_dtype"]) if kernel else None
        return Buf("inter", fmt, rows="expert_sorted", scales="mxmoe")

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def build(self, p, params):
        return functools.partial(
            _mxfp4_a4w4_stage1_fw,
            kernelName1=params.kernel,
            interleave=(p.gate_mode == GateMode.INTERLEAVE),
        )

    def default_params(self, p):
        if self.reach != "ranked":
            return None
        return _stage1_default_params(p, mxmoe_default_kernels(p))

    def default_warning(self, p):
        return mxmoe_default_kernels(p).warning

    def metadata_fields(self, p, params):
        p1 = _parse_mxfp4_g1_kname(params.kernel)
        return {
            "block_m": p1["BM"],
            # Legacy's value before its run-time override; the run follows the plan.
            "prequant": p1["a_dtype"] == "fp8" and not p1["inline_quant"],
            "has_bias": "bias1" in p.features and p1.get("enable_bias", False),
        }

    def param_space(self, p):
        return mxmoe_param_space(self, p)


class MxmoeQuantGemm1(MxmoeGemm1):
    """MXMOE stage 1 handed the caller's activations, which it quantizes itself
    (block_m 16, legacy `prequant` False)."""

    name = "mxmoe.quant_gemm1"
    reach = "tuned"
    covers = (Op.QUANT_A1, Op.QUANT_INTER)

    def reads(self, p):
        return (Buf("sorted", None, rows="aux_sorted"), CALL_INPUT)


class MxmoeGemm2(Segment):
    """MXMOE stage 2: a native `flydsl_mxmoe_g2_*` kernel or a layout GEMM2."""

    name = "mxmoe.gemm2"
    prefixes = (kernel_names.MXMOE, kernel_names.FLYDSL_LAYOUT)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return a4w4_caps()

    def owns(self, kernel) -> bool:
        if kernel is None or kernel.stage != 2:
            return False
        if kernel.family == kernel_names.MXMOE:
            return True
        return kernel.family == kernel_names.FLYDSL_LAYOUT and _reads_mx_inter(
            parse_flydsl_v2_gemm2_kernel(kernel.format())
        )

    def reads(self, p):
        return (
            Buf("sorted", None, rows="aux_sorted"),
            Buf("inter", None, rows="expert_sorted", scales="mxmoe"),
        )

    def reads_with(self, p, params):
        return layout_gemm2_reads(self.reads(p), params.get("kernel"))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def default_params(self, p):
        return Params(kernel=mxmoe_default_kernels(p).kernel2)

    def build(self, p, params):
        return functools.partial(_mxfp4_a4w4_stage2_fw, kernelName2=params.kernel)

    def metadata_fields(self, p, params):
        is_layout_gemm2 = parse_flydsl_v2_gemm2_kernel(params.kernel) is not None
        return {"stage2_has_bias": "bias2" in p.features and is_layout_gemm2}

    def param_space(self, p):
        return mxmoe_param_space(self, p)
