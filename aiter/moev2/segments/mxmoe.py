# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MXFP4 kernels (`flydsl_mxmoe_g{1,2}_*`): the stage wrappers, their
`MOEMetadata`, and the inline-sort checks; aiter/fused_moe.py imports them from here.
The segments that run these kernels (`mxmoe.gemm1`, `mxmoe.gemm2`) are in `mxfp4.py`.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module is shared with the legacy MoE path (re-exported from aiter.fused_moe, run
by legacy MoE calls): changes must keep legacy behavior
(`python tests/moe_dispatch/snapshot.py --check`). Please coordinate any change in this
module with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import functools

import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl.moe_common import GateMode, _mxfp4_intermediate_enabled
from aiter.ops.flydsl.mxfp4_kname import (
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    _parse_mxfp4_g2_kname,
    native_scale_layout_for,
    parse_flydsl_v2_gemm2_kernel,
    parse_g2_kname_any,
)
from aiter.ops.opus import moe_stage2_a8w4 as _opus_a8w4

from ..execute.metadata import MOEMetadata
from ..shapes import _is_row_major_2d
from . import sort as moev2_sort
from .flydsl import (
    _empty_u8,
    _flydsl_v2_stage2_wrapper,
    _get_flydsl_moe_kernels,
    _mxfp4_scale_u8,
)
from .sort import AUX_SORT_OPUS, _empty_bf16


def _padded_scale_cols(size, group_size=32):
    """Scale columns per row, padded the way ``e8m0_shuffle`` pads them."""
    return (((size + group_size - 1) // group_size + 7) // 8) * 8


def _is_inline_sort_kname(kernel1):
    """Whether stage1 can use the generated BM16 inline-sort path."""
    try:
        p1 = _parse_mxfp4_g1_kname(kernel1)
        return (
            p1["BM"] == 16
            and p1["inline_quant"]
            and p1["a_dtype"] == "fp4"
            and p1["out_dtype"] == "fp4"
        )
    except (KeyError, TypeError, ValueError):
        return False


def _is_inline_sort_cfg(kernel1, kernel2):
    """Whether a tuned row targets the inline-sort two-stage path.

    An inline-quant stage1 paired with a plain mxmoe stage2 belongs to the
    mxmoe dispatcher instead, and must not be judged by this path's rules.
    """
    return (
        _is_inline_sort_kname(kernel1)
        and parse_flydsl_v2_gemm2_kernel(kernel2) is not None
    )


def _is_mxfp4_inline_sort(metadata):
    """Validate a config-driven inline-quant two-stage MXFP4 dispatch."""
    try:
        p1 = _parse_mxfp4_g1_kname(metadata.stage1.keywords["kernelName1"])
        p2 = parse_flydsl_v2_gemm2_kernel(metadata.stage2.keywords["kernelName2"])
        return (
            not metadata.run_1stage
            and metadata.output_aux
            and not metadata.prequant
            and metadata.fuse_quant == "fp4"
            and _is_inline_sort_kname(metadata.stage1.keywords["kernelName1"])
            and p2 is not None
            and int(metadata.block_m) == p1["BM"]
            and p2["sort_block_m"] == p1["BM"]
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _mxfp4_inline_sort_unsupported(
    metadata,
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_ids,
    *,
    gate_mode,
    expert_mask,
    bias1,
    bias2,
    a1_scale,
    a2_scale,
    stage2_scatter,
    block_size_M,
    hidden_pad,
    intermediate_pad,
):
    """Why a tuned inline-sort row cannot run this call, or "" if it can.

    Only called for metadata that already passed :func:`_is_mxfp4_inline_sort`.
    The tuned-row lookup key pins dtypes, quant type, activation, g1u1 and
    doweight_stage1, so only what the key cannot see is checked here.
    """
    if block_size_M is not None and int(block_size_M) != int(metadata.block_m):
        return (
            f"block_size_M={block_size_M} does not match tuned block_m="
            f"{metadata.block_m}"
        )
    if GateMode(gate_mode) not in (GateMode.SEPARATED, GateMode.INTERLEAVE):
        return f"unsupported gate mode {gate_mode}"
    if hidden_pad or intermediate_pad:
        return "hidden/intermediate padding"
    if hidden_states.dtype != dtypes.bf16:
        return "activations must be bf16"
    if not (getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False)):
        return "weights are not both preshuffled"
    if bias1 is not None and not metadata.has_bias:
        return "stage1 kernel does not support bias1"
    if bias2 is not None and not metadata.stage2_has_bias:
        return "stage2 kernel does not support bias2"
    for name, value in (
        ("a1_scale", a1_scale),
        ("a2_scale", a2_scale),
        ("stage2_scatter", stage2_scatter),
    ):
        if value is not None:
            return f"unsupported option {name}"

    if w1_scale is None or w2_scale is None:
        return "missing MXFP4 weight scales"
    scale_dtypes = (dtypes.fp8_e8m0, torch.uint8)
    if w1_scale.dtype not in scale_dtypes or w2_scale.dtype not in scale_dtypes:
        return "MXFP4 weight scales must be e8m0"
    experts, hidden, inter = w1.shape[0], hidden_states.shape[1], w2.shape[-1] * 2
    # Padded, not exact: a non-256-aligned inter_dim (Kimi-K3 I=384 gives 12
    # valid of 16 columns) makes the shuffled stride wider than the payload,
    # and the kernels address the padded stride.
    if w1_scale.numel() < experts * 2 * inter * _padded_scale_cols(hidden) or (
        w2_scale.numel() < experts * hidden * _padded_scale_cols(inter)
    ):
        return "MXFP4 weight scales are smaller than the padded stride"

    tensors = (w1, w2, topk_ids, w1_scale, w2_scale)
    if not _is_row_major_2d(hidden_states) or not all(
        tensor.is_contiguous() for tensor in tensors
    ):
        return "non-contiguous tensors"
    if len({tensor.device for tensor in (hidden_states, *tensors)}) != 1:
        return "cross-device tensors"

    p2 = parse_flydsl_v2_gemm2_kernel(metadata.stage2.keywords["kernelName2"])
    # EP sorts with the masked Opus kernel instead of this generated one.
    if expert_mask is None and not moev2_sort._mxfp4_aux_instance_supported(
        experts,
        topk_ids.shape[1],
        hidden,
        int(metadata.block_m),
        p2["epilog"] == "atomic",
    ):
        return "aux instance is not code-generated"
    return ""


def _mxfp4_a4w4_stage1(
    hidden_states,
    w1,
    w1_scale,
    a_quant,
    a_scale,
    bf16_zero,
    sorted_token_ids,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    *,
    inline_quant,
    NE,
    topk,
    D_HIDDEN,
    D_INTER,
    Kpad_inter,
    BM,
    BN,
    BK,
    a_dtype,
    out_dtype,
    act,
    situ_beta,
    situ_linear_beta,
    swiglu_limit,
    bias1,
    max_sorted,
    kernelName1,
    device,
    use_nt=False,
    interleave=False,
    num_waves=4,
    native_scale_layout=False,
    k_wave=1,
    prefetch_hidden=False,
    prequantized=False,
):
    if a_dtype == "fp8" and not inline_quant:
        if a_scale is None:
            raise ValueError("MXMOE FP8 input requires sorted E8M0 activation scales")
        a_scale_sorted_shuffled = a_scale
    elif a_dtype == "fp8":
        # Inline quant reads hidden_states directly and ignores both A buffers.
        a_scale_sorted_shuffled = _empty_u8(device)
    elif prequantized:
        # The Opus aux path can use fused_dynamic_mxfp4_quant_moe_sort before
        # stage1. Its returned E8M0 scale already has the exact flattened
        # [M/32, K/256, 4, 16, 4] layout consumed by this GEMM1, so avoid the
        # separate mxfp4_moe_quant + mxfp4_moe_sort_scales kernels.
        a_scale_sorted_shuffled = a_scale
    elif not inline_quant:
        aiter.mxfp4_moe_quant(
            a_input=hidden_states,
            a_quant=a_quant,
            a_scale=a_scale,
            bf16_zero_out=bf16_zero,
            NE=NE,
            TOPK=topk,
            D_HIDDEN=D_HIDDEN,
            MB=BM,
        )
        padded_rows = ((max_sorted + 31) // 32) * 32
        cols = D_HIDDEN // 32
        a_scale_sorted_shuffled = torch.empty(
            (padded_rows * cols * 2,), device=device, dtype=torch.uint8
        )
        aiter.mxfp4_moe_sort_scales(
            a_scale=a_scale,
            sorted_token_ids=sorted_token_ids,
            cumsum_tensor=cumsum_tensor,
            a_scale_sorted_shuffled=a_scale_sorted_shuffled,
            NE=NE,
            TOPK=topk,
            D_HIDDEN=D_HIDDEN,
            MB=BM,
            max_sorted=max_sorted,
        )
    else:
        # inline_quant: pass a tiny placeholder. gemm1 won't read it.
        a_scale_sorted_shuffled = _empty_u8(device)

    # -- gemm1: A_q x w1 -> inter (packed MXFP4, sorted layout) ----------
    # The flydsl port reads/writes D_INTER directly (no K-pad tail to zero).
    BM_MIN = 64
    inter_cols = D_INTER if out_dtype == "fp8" else D_INTER // 2
    inter_scale_cols = D_INTER // 32
    inter_scale_bytes = max_sorted * max((1024 // BM_MIN) * 4, inter_scale_cols * 2)
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for

    chunk_bytes = kas_per_chunk_dw_for(D_INTER) * 4
    if native_scale_layout:
        # The native BM16 layout addresses one *padded* chunk per M block:
        # kas_per_chunk_dw_for pads scale-N up to a multiple of 8 columns, so
        # the kernel's stride between chunks exceeds inter_scale_cols whenever
        # D_INTER // 32 is not a multiple of 8. Sizing on the unpadded width
        # under-allocates and the last blocks write past the buffer (D_INTER=1408
        # spans 49152 B where the unpadded figure gives 45056 B).
        chunks = (max_sorted + BM - 1) // BM
        inter_scale_bytes = max(inter_scale_bytes, chunks * chunk_bytes)
    else:
        # Regular chunks pack 32 rows. Expose their physical column stride so
        # consumers deriving the chunk stride from tensor metadata read it as-is.
        inter_scale_cols = _padded_scale_cols(D_INTER)
        inter_scale_bytes = max(
            inter_scale_bytes, ((max_sorted + 31) // 32) * chunk_bytes
        )
    inter_dtype = dtypes.fp8 if out_dtype == "fp8" else torch.uint8
    inter_sorted_quant = torch.empty(
        (max_sorted, inter_cols), device=device, dtype=inter_dtype
    )
    inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
    inter_scale_rows = (inter_scale_rows + 31) // 32 * 32
    inter_scale_dtype = dtypes.fp8_e8m0 if out_dtype == "fp8" else torch.uint8
    inter_sorted_shuffled_scale = torch.empty(
        (inter_scale_rows, inter_scale_cols),
        device=device,
        dtype=inter_scale_dtype,
    )

    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

    _xcd1 = _parse_mxfp4_g1_kname(kernelName1).get("xcd_swizzle", 0)
    flydsl_mxfp4_gemm1(
        a_quant=a_quant,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        w1_u8=w1,
        w1_scale_u8=_mxfp4_scale_u8(w1_scale),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        m_indices=m_indices,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        hidden_states=hidden_states,
        n_tokens=hidden_states.shape[0],
        BM=BM,
        use_nt=use_nt,
        inline_quant=inline_quant,
        prefetch_hidden=prefetch_hidden,
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=topk,
        BN=BN,
        BK=BK,
        interleave=interleave,
        xcd_swizzle=_xcd1,
        native_scale_layout=native_scale_layout,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        bias=bias1,
        num_waves=num_waves,
        k_wave=k_wave,
    )
    return inter_sorted_quant, inter_sorted_shuffled_scale


def _mxfp4_a4w4_stage2(
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2,
    w2_scale,
    cumsum_tensor,
    sorted_token_ids,
    sorted_expert_ids,
    sorted_weights,
    reverse_sorted,
    out_dst,  # caller's output buffer; written in-place (atomic accum or scatter_reduce dst)
    *,
    atomic,
    mxfp4out,
    kernelName2,
    M,
    max_sorted,
    NE,
    topk,
    D_HIDDEN,
    D_INTER,
    BM,
    device,
    use_nt=False,
    cshuffle=False,
    inter_real=None,  # w2.inter_real (unpadded inter for non-256-aligned shards)
    use_valid_token_count=False,
):
    _xcd2 = _parse_mxfp4_g2_kname(kernelName2).get("xcd_swizzle", 0)
    if atomic:
        out_buf = out_dst
    else:
        _mx_shape_ok = (
            BM == 128 and D_HIDDEN == 7168 and D_INTER == 512 and NE in (257, 385)
        )

        if _mx_shape_ok and _mxfp4_intermediate_enabled():
            flat_out_q = torch.empty(
                (max_sorted, D_HIDDEN // 2), dtype=torch.uint8, device=device
            )
            flat_out_scale = torch.empty(
                (max_sorted, D_HIDDEN // 32), dtype=torch.uint8, device=device
            )
            from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2

            flydsl_mxfp4_gemm2(
                inter_sorted_quant=inter_sorted_quant,
                inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
                w2_u8=w2,
                w2_scale_u8=_mxfp4_scale_u8(w2_scale),
                sorted_expert_ids=sorted_expert_ids,
                cumsum_tensor=cumsum_tensor,
                sorted_token_ids=sorted_token_ids,
                sorted_weights=sorted_weights,
                flat_out=flat_out_q,
                flat_out_scale=flat_out_scale,
                M_logical=M,
                max_sorted=max_sorted,
                BM=BM,
                use_nt=use_nt,
                atomic=False,
                mxfp4out=True,
                cshuffle=cshuffle,
                NE=NE,
                D_HIDDEN=D_HIDDEN,
                D_INTER=D_INTER,
                D_INTER_REAL=inter_real,
                topk=topk,
                xcd_swizzle=_xcd2,
            )
            # scatter_reduce fully overwrites each output row -> write the caller's
            # buffer directly (avoids a redundant (M, D_HIDDEN) D2D copy at the end).
            out = out_dst
            aiter.mxfp4_moe_scatter_reduce_q(
                flat_out_q=flat_out_q,
                flat_out_scale=flat_out_scale,
                reverse_sorted=reverse_sorted,
                sorted_weights=sorted_weights,
                out=out,
                NE=NE,
                TOPK=topk,
                D_HIDDEN=D_HIDDEN,
                MB=BM,
            )
            return out

        # `_f4out` requested on an unsupported shape -> drop it, run bf16.
        if mxfp4out:
            kernelName2 = kernelName2.replace("_f4out", "")

        # Non-atomic bf16: per-sorted-row staging; scatter_reduce afterwards.
        out_buf = torch.empty((max_sorted, D_HIDDEN), dtype=dtypes.bf16, device=device)

    from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2

    flydsl_mxfp4_gemm2(
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        w2_u8=w2,
        w2_scale_u8=_mxfp4_scale_u8(w2_scale),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        flat_out=out_buf,
        M_logical=M,
        max_sorted=max_sorted,
        BM=BM,
        use_nt=use_nt,
        atomic=atomic,
        mxfp4out=False,
        cshuffle=cshuffle,
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        D_INTER_REAL=inter_real,
        topk=topk,
        xcd_swizzle=_xcd2,
        use_valid_token_count=use_valid_token_count,
    )

    if atomic:
        return out_buf

    # -- scatter_reduce: per-(token, topk-slot) flat_out -> per-token out --
    # Write the caller's buffer directly (scatter_reduce overwrites every row), so the
    # trailing `moe_out.copy_(out)` becomes a no-op and the D2D copy is eliminated.
    out = out_dst
    aiter.mxfp4_moe_scatter_reduce(
        flat_out=out_buf,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        out=out,
        NE=NE,
        TOPK=topk,
        D_HIDDEN=D_HIDDEN,
        MB=BM,
    )
    return out


def _mxfp4_a4w4_stage1_fw(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    *,
    block_m=None,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
    kernelName1="",
    m_indices=None,
    moe_buf=None,
    interleave=False,
    bias1=None,
    swiglu_limit: float | None = None,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    native_scale_layout: bool | None = None,
    **_kwargs,
):
    device = hidden_states.device
    p1 = _parse_mxfp4_g1_kname(kernelName1)
    runtime_situ_beta = float(situ_beta)
    runtime_situ_linear_beta = float(situ_linear_beta)
    # Match generic FlyDSL v1/v2 semantics. MXMOE captures this value in the
    # compiled kernel closure; +inf represents an unclamped SiLU.
    runtime_swiglu_limit = _get_flydsl_moe_kernels().runtime_swiglu_limit(
        swiglu_limit, p1["act"]
    )
    BM = p1["BM"]
    if native_scale_layout is None:
        native_scale_layout = native_scale_layout_for(BM, p1["out_dtype"])
    inline_quant = p1["inline_quant"]
    if w1.element_size() == 1 and w1.dtype != torch.uint8:
        w1 = w1.view(torch.uint8)
    NE = w1.shape[0]
    # hidden_states may already be packed FP4, whose physical last dimension
    # is D_HIDDEN/2. w2 keeps the logical output dimension in mode 1 for both
    # packed and unpacked inputs, so it is the stable source of the GEMM1 K.
    D_HIDDEN = w2.shape[1]
    D_INTER = w1.shape[1] // 2
    Kpad_inter = ((D_INTER + 255) // 256) * 256
    M = hidden_states.shape[0]
    if m_indices is None:
        m_indices = (sorted_token_ids & 0xFFFFFF).contiguous()
    prequantized = (
        p1["a_dtype"] == "fp4"
        and not inline_quant
        and hidden_states.dtype == dtypes.fp4x2
        and a1_scale is not None
    )
    if p1["a_dtype"] == "fp8" or prequantized:
        a_quant = hidden_states
        a_scale = a1_scale
    else:
        a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
        a_scale = torch.empty((M, D_HIDDEN // 32), device=device, dtype=torch.uint8)

    bf16_zero = (
        moe_buf
        if (moe_buf is not None and moe_buf.numel() > 0 and not inline_quant)
        else _empty_bf16(device)
    )
    inter_sorted_quant, inter_sorted_scale = _mxfp4_a4w4_stage1(
        hidden_states,
        w1,
        w1_scale,
        a_quant,
        a_scale,
        bf16_zero,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        m_indices,
        inline_quant=inline_quant,
        NE=NE,
        topk=topk,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        Kpad_inter=Kpad_inter,
        BM=BM,
        BN=p1["BN"],
        BK=p1["BK"],
        a_dtype=p1["a_dtype"],
        out_dtype=p1["out_dtype"],
        act=p1["act"],
        situ_beta=runtime_situ_beta,
        situ_linear_beta=runtime_situ_linear_beta,
        swiglu_limit=runtime_swiglu_limit,
        bias1=bias1,
        max_sorted=sorted_token_ids.shape[0],
        kernelName1=kernelName1,
        device=device,
        use_nt=p1["use_nt"],
        # The caller supplies the weight layout independently of kernelName1.
        interleave=interleave,
        num_waves=p1.get("num_waves", 4),
        native_scale_layout=native_scale_layout,
        k_wave=p1.get("k_wave", 1),
        prefetch_hidden=p1.get("prefetch_hidden", False),
        prequantized=prequantized,
    )
    return inter_sorted_quant, inter_sorted_scale


def _mxfp4_a4w4_stage2_fw(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    moe_out,
    topk,
    *,
    w2_scale=None,
    a2_scale=None,
    block_m=None,
    sorted_weights=None,
    topk_weights=None,
    bias2=None,
    kernelName2="",
    reverse_sorted=None,
    use_valid_token_count=False,
    scatter_intermediate=None,
    expert_mask=None,
    topk_ids=None,
    **_kwargs,
):
    """``scatter_intermediate``: see ``_flydsl_v2_stage2_wrapper``. The native
    MXMOE GEMM2 ignores it and returns the complete ``moe_out``."""
    if scatter_intermediate is not None:
        scatter_intermediate.clear()  # no items of an earlier call

    device = inter_states.device
    # kernelName2's format selects the gemm2 family: a flydsl_moe2_layout name
    # means path B (v2 gemm2 behind the mxmoe front-end), else native mxmoe.
    cfg = parse_g2_kname_any(kernelName2)
    # Read inter_real BEFORE any w2.view() drops the attr. The flydsl port reads
    # D_INTER directly (D_INTER_REAL handles the unpadded shard); no K-pad needed.
    inter_real = getattr(w2, "inter_real", None)
    if w2.element_size() == 1 and w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)
    NE = w2.shape[0]
    D_HIDDEN = w2.shape[1]
    D_INTER = w1.shape[1] // 2
    M = moe_out.shape[0]
    if cfg["v2"]:
        # The mxmoe intermediate (inter_states + a2_scale) is byte-compatible
        # with gemm2_body_v2's native-BM scale-chunk layout at SBM=BM (verified
        # for BM in {16,32,64,128} x epilog {atomic,reduce}).
        if inter_real is not None and inter_real != D_INTER:
            # This path does not thread the v2 gemm2's K-pad skip (has_pad +
            # i32_kpad), so the pad columns would be accumulated instead of
            # skipped. v2 needs K aligned only to its BK, so an unpadded shard
            # is the intended input here.
            raise NotImplementedError(
                f"FlyDSL v2 stage2 requires an unpadded inter_dim shard, got "
                f"w2.inter_real={inter_real} with D_INTER={D_INTER}. Use a "
                f"native flydsl_mxmoe_g2 kernelName2 for pre-padded weights."
            )
        return _flydsl_v2_stage2_wrapper(
            inter_states=inter_states,
            w1=None,
            w2=w2,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=moe_out,
            topk=topk,
            kernelName=kernelName2,
            model_dim=D_HIDDEN,
            inter_dim=D_INTER,
            num_experts=NE,
            w2_scale=w2_scale,
            a2_scale=a2_scale,
            sorted_weights=sorted_weights,
            topk_weights=topk_weights,
            bias2=bias2,
            block_m=block_m,
            expert_mask=expert_mask,
            topk_ids=topk_ids,
            reverse_sorted=reverse_sorted,
            use_valid_token_count=use_valid_token_count,
            scatter_intermediate=scatter_intermediate,
        )
    if bias2 is not None:
        raise ValueError(f"MXMOE GEMM2 {kernelName2!r} does not support bias")
    out = _mxfp4_a4w4_stage2(
        inter_states,
        a2_scale,
        w2,
        w2_scale,
        num_valid_ids,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        reverse_sorted,
        moe_out,
        atomic=cfg["atomic"],
        mxfp4out=cfg["mxfp4out"],
        kernelName2=kernelName2,
        M=M,
        max_sorted=sorted_token_ids.shape[0],
        NE=NE,
        topk=topk,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        BM=cfg["BM"],
        device=device,
        use_nt=cfg["use_nt"],
        cshuffle=cfg["cshuffle"],
        inter_real=inter_real,
        use_valid_token_count=use_valid_token_count,
    )

    if out is not moe_out:
        moe_out.copy_(out)
    return moe_out


def _mxmoe_gemm2_sort_block_m(kernel_name2, sort_block_m):
    """GEMM2's sort block; a layout-v2 name without _sbm reads the actual one."""
    v2 = parse_flydsl_v2_gemm2_kernel(kernel_name2)
    if v2 is not None:
        return v2["sort_block_m"] or sort_block_m
    if _is_mxfp4_kname(kernel_name2):
        return _parse_mxfp4_g2_kname(kernel_name2)["BM"]
    return sort_block_m


def _make_mxfp4_metadata(
    kernel_name1,
    kernel_name2,
    gate_mode,
    ksplit,
    *,
    block_m=None,
    run_1stage=False,
    has_stage1_bias=False,
    has_stage2_bias=False,
    opus_launch=None,
    route_bucket_metadata=None,
):
    p1 = _parse_mxfp4_g1_kname(kernel_name1)
    # Use the stage2 dispatch parser so metadata only promises supported bias.
    is_layout_gemm2 = parse_flydsl_v2_gemm2_kernel(kernel_name2) is not None
    is_opus = _opus_a8w4.is_opus_a8w4_stage2_kernel(kernel_name2)
    if not (_is_mxfp4_kname(kernel_name2) or is_layout_gemm2 or is_opus):
        raise ValueError(
            "MXMOE GEMM1 requires a native or layout GEMM2 backend, "
            f"got {kernel_name2!r}"
        )
    resolved_block_m = p1["BM"] if block_m is None else int(block_m)
    if is_opus and opus_launch is None:
        opus_launch = _opus_a8w4.parse_stage2_config(
            {"kernelName2": kernel_name2}, resolved_block_m
        )
    stage2 = (
        functools.partial(
            _opus_a8w4.opus_a8w4_stage2_wrapper,
            kernelName=kernel_name2,
            launch=opus_launch,
        )
        if is_opus
        else functools.partial(_mxfp4_a4w4_stage2_fw, kernelName2=kernel_name2)
    )
    return MOEMetadata(
        stage1=functools.partial(
            _mxfp4_a4w4_stage1_fw,
            kernelName1=kernel_name1,
            interleave=(gate_mode == GateMode.INTERLEAVE),
            native_scale_layout=native_scale_layout_for(
                p1["BM"], p1["out_dtype"], kernel_name2
            ),
        ),
        stage2=stage2,
        block_m=resolved_block_m,
        ksplit=int(ksplit),
        run_1stage=bool(run_1stage),
        fuse_quant=p1["out_dtype"],
        output_aux=AUX_SORT_OPUS,
        prequant=p1["a_dtype"] == "fp8" and not p1["inline_quant"],
        has_bias=has_stage1_bias,
        stage2_has_bias=has_stage2_bias and is_layout_gemm2,
        **(route_bucket_metadata or {}),
    )
