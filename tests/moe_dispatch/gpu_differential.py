# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU differential of one migration path against the legacy MoE code.

Each case runs on the same inputs with AITER_MOEV2=off and AITER_MOEV2=<path>: the path
must run the call (not decline it), launch the same kernels on bitwise-equal inputs, give
the same output up to the GEMM2 atomics, and log the same dispatch lines on a first call
(see `differential.run_differential`).

    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt
    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt --synth-table

Shipped tables have per-tensor rows for gfx942 only (and bf16 rows for gfx942 and a few
gfx950 shapes). `--synth-table` adds the path's gfx942 CK rows, copied to the running
arch, to the arch's own rows in a temporary AITER_CONFIG_FMOE, so the tuned CK branch runs
on a gfx950 too. ASM and FlyDSL rows are not copied: their gfx942 configs give garbage on
gfx950 (legacy and v2 alike). For a8w8_blockscale, whose shipped ASM stage-1 rows are
gfx942 only, the arch's own CK rows are also added with stage 1 swapped for the arch's
ASM kernel of the same tile M (when it ships one), so that branch runs too; for a8w8_pt,
the gfx942 int8 rows whose ASM kernel the arch ships are copied as well.

A case that does not fit in the free GPU memory is reported as skipped; one that raises
the same error on legacy and v2 passes, with the error printed.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# (token, model_dim, inter_dim, experts, topk, activation, doweight_stage1, out dtype, ep)
# and optionally the weight format ("fp8" when absent, "i8", "i4": int4 in uint32, or
# "fp4"/"fp4u": MXFP4, preshuffled or not).
_A8W8_PT_DEFAULT = (
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16, 64, 256, 1024, 4096)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (1, 32, 512)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16, 512)),
    # int8 (ASM stage 1 + CK stage 2), and the shape of the gfx942 cu80 int8 row.
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False, "i8") for m in (1, 64, 1024)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "i8") for m in (32,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False, "i8") for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True, "i8") for m in (512,)),
    *((m, 6144, 4096, 8, 2, "Silu", False, "bf16", False, "i8") for m in (512,)),
)  # fmt: skip

_A16W16_DEFAULT = (
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16, 64, 256, 1024, 4096)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (1, 32, 512)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 2048, 256, 64, 6, "Gelu", True, "fp16", False) for m in (40,)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16, 512)),
    # Shapes with shipped gfx950 cu256 rows (tuned CK there, default elsewhere).
    *((m, 6144, 512, 192, 8, "Silu", False, "bf16", False) for m in (1, 512, 2048)),
    *((m, 4096, 384, 128, 8, "Silu", False, "bf16", False) for m in (64, 1024)),
)  # fmt: skip

# Outside gfx950's default 1-stage ASM region (`token > 16 or inter_dim % 128`, bf16
# output, Silu or Gelu, Silu only with doweight_stage1), which the path declines.
_A8W8_PTOK_DEFAULT = (
    # fp16 shapes stay small: larger ones overflow, and where the infinities land
    # differs between two legacy runs (GEMM2 atomics).
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16)),
    *((m, 2048, 256, 128, 8, "Gelu", False, "fp16", False) for m in (32, 512)),
    *((m, 4096, 512, 64, 4, "Gelu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16,)),
    *((m, 2048, 256, 64, 8, "Silu", False, "fp16", True) for m in (512,)),
    # Shapes with shipped gfx950 cu256 rows (ASM stage 1 + CK stage 2, and CK).
    *((m, 5120, 1536, 40, 8, "Silu", False, "bf16", False) for m in (1, 2, 256, 1024, 4096)),
    *((m, 5120, 1536, 41, 10, "Silu", False, "bf16", False) for m in (3, 8, 64)),
    *((m, 5120, 384, 161, 9, "Silu", False, "bf16", False) for m in (4, 2048)),
    *((m, 6144, 512, 257, 9, "Silu", False, "bf16", False) for m in (16, 1024)),
    # int8 (ASM stage 1 + CK stage 2; gfx942's 1-stage region is `token > 32`).
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False, "i8") for m in (1, 16, 32)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "i8") for m in (8, 512)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False, "i8") for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True, "i8") for m in (16,)),
    # int4 weights, fp8 activations (CK 2-stage), as `-q 3`.
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False, "i4") for m in (1, 16, 256)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "i4") for m in (8, 512)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False, "i4") for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False, "i4") for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True, "i4") for m in (16,)),
)  # fmt: skip

# Outside gfx950's default 1-stage ASM region (`token > 32`, bf16 output, no
# doweight_stage1, a kernel for the shape); token <= 32 runs CK with split-K.
_A8W8_BLOCKSCALE_DEFAULT = (
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16, 32)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (1, 32)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16,)),
    *((m, 7168, 256, 64, 8, "Silu", True, "bf16", True) for m in (512,)),
    # Shapes with shipped gfx950 cu256 rows (CK, some with split-K).
    *((m, 4096, 384, 128, 8, "Silu", False, "bf16", False) for m in (1, 256, 4096)),
    *((m, 4096, 384, 128, 8, "Silu", True, "bf16", False) for m in (8, 512)),
    *((m, 7168, 512, 257, 9, "Silu", False, "bf16", False) for m in (1, 128)),
    *((m, 4096, 1536, 16, 8, "Silu", False, "bf16", False) for m in (16, 512)),
)  # fmt: skip

# gfx950: shapes of each shipped cu256 row family, untuned shapes (the FlyDSL heuristic on
# preshuffled weights, CK otherwise), expert-parallel (MXMOE rows fall back to the
# heuristic) and doweight_stage1 (CK).
_A4W4 = (
    # FlyDSL stage 1 + layout-v2 stage 2.
    *((m, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4") for m in (1, 64, 1024)),
    *((m, 4096, 512, 512, 10, "Silu", False, "bf16", False, "fp4") for m in (16,)),
    # MXMOE stage 1 + layout-v2 stage 2.
    *((m, 5120, 640, 128, 3, "Silu", False, "bf16", False, "fp4") for m in (8, 64, 256)),
    *((m, 6144, 512, 257, 9, "Silu", False, "bf16", False, "fp4") for m in (2, 16)),
    # FlyDSL stage 1 + FlyDSL stage 2, and + CK stage 2.
    *((m, 4096, 1536, 16, 7, "Silu", False, "bf16", False, "fp4") for m in (16, 256)),
    *((m, 5120, 2304, 32, 2, "Silu", False, "bf16", False, "fp4") for m in (8,)),
    *((m, 3072, 768, 256, 8, "Silu", False, "bf16", False, "fp4") for m in (1, 2)),
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False, "fp4") for m in (64,)),
    # Untuned.
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4") for m in (5, 200)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False, "fp4") for m in (40,)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "fp4") for m in (32,)),
    # doweight_stage1 outside CK-Tile split-K (`token * topk > experts`), which the path
    # declines (and which faults on legacy at (8, 4096, 512, 64, 4)).
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False, "fp4") for m in (300, 1024)),
    *((m, 3072, 384, 128, 8, "Silu", False, "bf16", True, "fp4") for m in (16, 512)),
    *((m, 5120, 640, 64, 3, "Silu", False, "bf16", True, "fp4") for m in (64,)),
    # Not preshuffled: CK.
    *((m, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4u") for m in (16, 256)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4u") for m in (5,)),
)  # fmt: skip

_CASES = {
    "a8w8_pt": _A8W8_PT_DEFAULT,
    "a16w16": _A16W16_DEFAULT,
    "a8w8_ptok": _A8W8_PTOK_DEFAULT,
    "a8w8_blockscale": _A8W8_BLOCKSCALE_DEFAULT,
    "a4w4": _A4W4,
}
# The dispatch key's quant type and weight dtype of each path's tuned rows.
_ROWS = {
    "a8w8_pt": ("QuantType.per_Tensor", "float8"),
    "a16w16": ("QuantType.No", "16"),
    "a8w8_ptok": ("QuantType.per_Token", "float8"),
    "a8w8_blockscale": ("QuantType.per_1x128", "float8"),
    "a4w4": ("QuantType.per_1x32", "float4"),
}


def _per_128x128_quant(weight, quant_dtype):
    """op_tests/test_moe_2stage.py `weight_per_128x128_quant`."""
    import aiter

    E, dim1, dim2 = weight.shape
    blocks = weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
    blocks = blocks.permute(0, 1, 3, 2, 4).contiguous().view(E, -1, 128 * 128)
    qt, scale = aiter.pertoken_quant(blocks, quant_dtype=quant_dtype)
    qt = qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
    qt = qt.permute(0, 1, 3, 2, 4).contiguous().view(E, dim1, dim2)
    return qt, scale.view(E, dim1 // 128, dim2 // 128)


def _make_call(path_name, case):
    """Inputs as op_tests/test_moe_2stage.py builds them for `-q 1` (per-tensor fp8,
    path a8w8_pt), `-q 0` (unquantized, path a16w16), `-q 2` (per-token fp8, path
    a8w8_ptok), `-q 3` (per-token int4 weights, path a8w8_ptok), `-q 5` (128x128 fp8
    blocks, path a8w8_blockscale) or `-q 4` (MXFP4, path a4w4; `--preshuffle` for
    "fp4"); int8 weights as `-q 1`/`-q 2` with int8.
    Unlike op_tests, the weights are scaled by 1/sqrt(K) (see `weight`)."""
    import torch

    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_topk
    from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    token, model_dim, inter_dim, E, topk, act, doweight, out, ep, *rest = case
    weights = rest[0] if rest else "fp8"
    q_dtype = dtypes.i8 if weights in ("i8", "i4") else dtypes.fp8
    torch.manual_seed(token * 7919 + model_dim + inter_dim + E + topk)
    dtype = {"bf16": dtypes.bf16, "fp16": dtypes.fp16}[out]
    device = "cuda"
    hidden = torch.randn((token, model_dim), dtype=dtype, device=device)
    quant_type = {
        "a8w8_pt": QuantType.per_Tensor,
        "a8w8_ptok": QuantType.per_Token,
        "a8w8_blockscale": QuantType.per_128x128,
        "a4w4": QuantType.per_1x32,
    }.get(path_name, QuantType.No)

    def weight(shape):
        # Quantized one at a time, the bf16 copy freed: the GPU may be shared.
        # Scaled by 1/sqrt(K) so each GEMM keeps values near 1: unscaled, the fp16
        # outputs reach the fp16 limit, and whether a sum overflows then depends on
        # the order of the GEMM2 atomics.
        w = torch.randn(shape, dtype=dtype, device=device) / shape[-1] ** 0.5
        if weights == "i4":
            qt, scale = aiter.pertoken_quant(w, quant_dtype=dtypes.i8, dtypeMax=7)
            del w
            packed = shuffle_weight(qt, (16, 16), use_int4=True)
            return rearrange_4bit_elements(convert_int8_to_uint32_int4(packed)), scale
        if weights in ("fp4", "fp4u"):
            qt, scale = aiter.get_torch_quant(quant_type)(w, quant_dtype=dtypes.fp4x2)
            qt = qt.view(*shape[:-1], shape[-1] // 2)
            del w
            if weights == "fp4":
                qt = shuffle_weight(qt, layout=(16, 16))
            return qt, fp4_utils.e8m0_shuffle(scale)
        if quant_type == QuantType.per_Tensor:
            qt, scale = aiter.pertoken_quant(w.view(E, -1), quant_dtype=q_dtype)
        elif quant_type == QuantType.per_Token:
            qt, scale = aiter.pertoken_quant(w, quant_dtype=q_dtype)
        elif quant_type == QuantType.per_128x128:
            qt, scale = _per_128x128_quant(w, dtypes.fp8)
        else:
            qt, scale = w, None
        qt = qt.view(shape)
        del w
        return shuffle_weight(qt, layout=(16, 16)), scale

    w1_qt, w1_scale = weight((E, inter_dim * 2, model_dim))
    w2_qt, w2_scale = weight((E, model_dim, inter_dim))
    global_E = 2 * E if ep else E
    score = torch.randn((token, global_E), dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(hidden, score, topk, True)
    expert_mask = None
    if ep:
        expert_mask = torch.zeros(global_E, dtype=dtypes.i32, device=device)
        expert_mask[:E] = 1
    return {
        "hidden_states": hidden,
        "w1": w1_qt,
        "w2": w2_qt,
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
        "expert_mask": expert_mask,
        "activation": ActivationType.__members__[act],
        "quant_type": quant_type,
        "doweight_stage1": doweight,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }


def _synth_rows(path_name, table_file, gfx, cu_num, limit):
    """The table: (gfx, cu_num)'s own rows plus `path_name`'s gfx942 CK rows rewritten
    for it (own rows win on a shared key); the case tuples of a sample of the latter."""
    import pandas as pd

    from aiter.moev2.dispatch import kernel_names

    q_type, w_dtype = _ROWS[path_name]
    df = pd.read_csv(table_file)
    src = df[
        (df["gfx"] == "gfx942")
        & (df["cu_num"] == 80)
        & (df["q_type"] == q_type)
        & (df["q_dtype_w"].str.contains(w_dtype))
        & (df["run_1stage"] == 0)
        & (df["use_g1u1"] == 1)
    ].copy()

    def portable(name):
        parsed = kernel_names.try_parse(name)
        return parsed is None or parsed.family == kernel_names.CK2STAGES

    src = src[src["kernelName1"].map(portable) & src["kernelName2"].map(portable)]
    src["gfx"], src["cu_num"] = gfx, cu_num
    if w_dtype == "float8":
        fp8 = "torch.float8_e4m3fn" if gfx == "gfx950" else "torch.float8_e4m3fnuz"
        src["q_dtype_a"] = src["q_dtype_w"] = fp8
    # Spread the sample over kernel families, block_m, ksplit and token counts.
    src = src.sort_values(["kernelName1", "block_m", "ksplit", "token"])
    step = max(1, len(src) // limit)
    sample = src.iloc[::step].head(limit)
    own = df[(df["gfx"] == gfx) & (df["cu_num"] == cu_num)]
    extra = df.iloc[:0]
    if path_name == "a8w8_blockscale":
        extra = _asm_blockscale_rows(df, gfx, cu_num)
    elif path_name == "a8w8_pt":
        extra = _asm_int8_rows(df, gfx, cu_num)
    sample = pd.concat([extra, sample])
    table = pd.concat([extra, own, src]).drop_duplicates(_KEY_COLUMNS)
    cases = []
    for r in sample.itertuples():
        act = str(r.act_type).split(".")[-1]
        out = "bf16" if "bfloat16" in str(r.dtype) else "fp16"
        cases.append(
            (int(r.token), int(r.model_dim), int(r.inter_dim), int(r.expert),
             int(r.topk), act, bool(r.doweight_stage1), out, False, _weights(r))
        )  # fmt: skip
    return table, cases


def _weights(row):
    """The case weight format of a tuned row."""
    w = str(row.q_dtype_w)
    return "i8" if "int8" in w else "i4" if "uint32" in w or "int4" in w else "fp8"


_KEY_COLUMNS = [
    "gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk", "act_type",
    "dtype", "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1", "doweight_stage1",
]  # fmt: skip


def _asm_blockscale_rows(df, gfx, cu_num):
    """The arch's own 1x128 CK rows with stage 1 swapped for its ASM kernel of the same
    tile M: shipped ASM rows for this format exist for gfx942 only."""
    import pandas as pd

    listing = (
        _REPO / "hsa" / gfx / "fmoe_2stages"
        / "fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1.csv"
    )  # fmt: skip
    if not listing.exists():
        return df.iloc[:0]
    kernels = pd.read_csv(listing)
    kernels = kernels[kernels["tile_n"] == 128]  # every 1x128 inter_dim fits
    by_tile_m = {m: min(g["knl_name"]) for m, g in kernels.groupby("tile_m")}
    own = df[
        (df["gfx"] == gfx)
        & (df["cu_num"] == cu_num)
        & (df["q_type"] == "QuantType.per_1x128")
        & (df["run_1stage"] == 0)
        & (df["use_g1u1"] == 1)
        & (df["doweight_stage1"] == 0)
        & (df["ksplit"] == 0)
        & df["kernelName1"].str.startswith("moe_ck2stages_gemm1_")
        & df["block_m"].isin(list(by_tile_m))
    ].copy()
    own["kernelName1"] = own["block_m"].map(by_tile_m)
    return own


def _asm_int8_rows(df, gfx, cu_num):
    """The gfx942 per-tensor int8 rows (ASM stage 1 + CK stage 2) whose ASM kernel the
    arch also ships, copied to it."""
    import pandas as pd

    listing = (
        _REPO / "hsa" / gfx / "fmoe_2stages" / "fmoe_stage1_bf16_pertokenInt8_g1u1.csv"
    )
    if not listing.exists():
        return df.iloc[:0]
    shipped = set(pd.read_csv(listing)["knl_name"])
    rows = df[
        (df["gfx"] == "gfx942")
        & (df["q_type"] == "QuantType.per_Tensor")
        & (df["q_dtype_w"] == "torch.int8")
        & (df["run_1stage"] == 0)
        & df["kernelName1"].isin(shipped)
    ].copy()
    rows["gfx"], rows["cu_num"] = gfx, cu_num
    return rows


def cases(path_name, synth_cases=()):
    if path_name not in _CASES:
        raise SystemExit(f"no GPU cases for path {path_name}")
    return list(_CASES[path_name]) + list(synth_cases)


def _clear_dispatch_caches():
    import aiter.fused_moe as fm
    from aiter.moev2.migration import a4w4, a8w8_blockscale, a8w8_pt, a8w8_ptok, a16w16

    fm.get_2stage_cfgs.cache_clear()
    for path in (a8w8_pt, a16w16, a8w8_ptok, a8w8_blockscale, a4w4):
        for cache in (path.LOCAL.dispatch, path.LOCAL.resolve, path.LOCAL.relookup):
            cache.cache_clear()


def _raised(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def run_case(path_name, case):
    """Return a problem string, or None when legacy and v2 agree."""
    import torch

    import aiter.fused_moe as fm
    from moe_dispatch.differential import captured_logs, moev2_mode, run_differential

    call = _make_call(path_name, case)
    with moev2_mode("off"), captured_logs():
        legacy_error = _raised(lambda: fm.fused_moe(**call))  # warm-up: JIT logs here
    torch.cuda.synchronize()
    _clear_dispatch_caches()
    with moev2_mode("off"), captured_logs() as legacy_logs:
        _raised(lambda: fm.fused_moe(**call))
    _clear_dispatch_caches()
    with moev2_mode(path_name), captured_logs() as v2_logs:
        v2_error = _raised(lambda: fm.fused_moe(**call))
    if legacy_logs != v2_logs:
        return f"{case}: log lines differ\n    legacy: {legacy_logs}\n    v2:     {v2_logs}"
    if legacy_error or v2_error:
        if legacy_error != v2_error:
            return f"{case}: legacy raises {legacy_error}; v2 raises {v2_error}"
        print(f"     both raise {legacy_error[:160]}")
        return None
    try:
        run_differential(path_name, call)
    except AssertionError as exc:
        return f"{case}: {exc}"
    return None


def _child(args):
    sys.path.insert(0, str(_HERE.parent))
    import torch

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    synth_cases = json.loads(os.environ.get("_MOEV2_SYNTH_CASES", "[]"))
    todo = cases(args.path, [tuple(c) for c in synth_cases])
    print(
        f"{get_gfx_runtime()} cu{get_cu_num()} torch {torch.__version__}: {len(todo)} cases"
    )
    problems, skipped = [], []
    for case in todo:
        try:
            problem = run_case(args.path, case)
        except torch.OutOfMemoryError:
            # The GPU may be shared; a case that does not fit is reported, not compared.
            problem = None
            skipped.append(case)
        torch.cuda.empty_cache()
        status = (
            "SKIP" if skipped and skipped[-1] is case else "FAIL" if problem else "ok  "
        )
        print(f"{status} {case}", flush=True)
        if problem:
            problems.append(problem)
    for line in problems:
        print("  " + line)
    if skipped:
        print(f"{len(skipped)} cases skipped: out of GPU memory")
    print(f"{len(problems)} problems")
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True)
    parser.add_argument("--synth-table", action="store_true")
    parser.add_argument("--synth-limit", type=int, default=48)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child or not args.synth_table:
        return _child(args)

    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    rows, synth = _synth_rows(
        args.path, AITER_CONFIGS.AITER_CONFIG_FMOE_FILE, gfx, cu_num, args.synth_limit
    )
    table = pathlib.Path(tempfile.mkdtemp(prefix="moev2_gpu_")) / "tuned_fmoe.csv"
    rows.to_csv(table, index=False)
    print(f"synthesized {len(rows)} {args.path} rows for {gfx} cu{cu_num} in {table}")
    env = dict(
        os.environ, AITER_CONFIG_FMOE=str(table), _MOEV2_SYNTH_CASES=json.dumps(synth)
    )
    cmd = [sys.executable, __file__, "--path", args.path, "--child"]
    return subprocess.run(cmd, env=env, cwd=_REPO, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
