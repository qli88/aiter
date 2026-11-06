# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU differential of one migration path against the legacy MoE code.

Each case runs on the same inputs with AITER_MOEV2=off and AITER_MOEV2=<path>: the path
must run the call (not decline it), launch the same kernels on bitwise-equal inputs, give
the same output up to the GEMM2 atomics, and log the same dispatch lines on a first call
(see `differential.run_differential`).

    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt
    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt --synth-table

`--path comm_fused` runs the TP8 tests/moe_dispatch/comm_fused_differential.py instead;
`--path grouped_gfx1250` the GPU test of tests/test_moev2_grouped_gfx1250.py (gfx1250);
`--path mega_moe_gfx1250` has none yet (multi-GPU gfx1250) and fails; `--path mega_moe_v2`
runs the EP8 tests/moe_dispatch/mega_moe_v2_differential.py; `--path mega_moe_tp` the TP4
tests/moe_dispatch/mega_moe_tp_differential.py.

Shipped tables have per-tensor rows for gfx942 only (and bf16 rows for gfx942 and a few
gfx950 shapes). `--synth-table` adds the path's gfx942 CK rows, copied to the running
arch, to the arch's own rows in a temporary AITER_CONFIG_FMOE, so the tuned CK branch runs
on a gfx950 too. ASM and FlyDSL rows are not copied: their gfx942 configs give garbage on
gfx950 (legacy and v2 alike). For a8w8_blockscale, whose shipped ASM stage-1 rows are
gfx942 only, the arch's own CK rows are also added with stage 1 swapped for the arch's
ASM kernel of the same tile M (when it ships one), so that branch runs too; for a8w8_pt,
the gfx942 int8 rows whose ASM kernel the arch ships are copied as well; for a8w8_ptok,
1-stage rows are added for the formats only a tuned row runs on the arch
(`asm.FMOE_1STAGE_TUNED_KEYS`).

A case that does not fit in the free GPU memory is reported as skipped and fails the run
(it was not compared); one that raises the same error on legacy and v2 passes, with the
error printed.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from unittest import mock

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# (token, model_dim, inter_dim, experts, topk, activation, doweight_stage1, out dtype, ep)
# and optionally the weight format ("fp8" when absent, "i8", "i4": int4 in uint32,
# "fp4"/"fp4u": MXFP4, preshuffled or not, "wi4": `torch.int4` in 1x32 groups, or
# "mxfp8": fp8 in 1x32 groups) and a dict of call options: "q_a" (the
# `quant_dtype_a`: "fp4", "fp8", "bf16"), "bias" (fp32 per-expert bias1 and bias2),
# "swiglu_limit", "g1u0" (gate-only w1), "interleave" (gate/up interleaved w1, a16w4,
# a8w4 and mxfp8), "std_shuffle" (a16w4/a8w4 "fp4" and "mxfp8" weights shuffled as
# a4w4's, untagged), "env" (environment variables set for the legacy and the v2 runs).
_SWIGLU_GELUTANH = ("Swiglu", "GeluTanh")
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
    # Swiglu and GeluTanh: CK stage 1 (ASM for int8), the CK module of the activation.
    *((m, 7168, 256, 256, 8, a, False, "bf16", False) for a in _SWIGLU_GELUTANH for m in (1, 64, 1024)),
    *((m, 2048, 256, 64, 6, "Swiglu", True, "fp16", False) for m in (200,)),
    *((m, 4096, 384, 128, 8, a, False, "bf16", False, "i8") for a in _SWIGLU_GELUTANH for m in (16, 512)),
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
    *((m, 7168, 256, 256, 8, a, False, "bf16", False) for a in _SWIGLU_GELUTANH for m in (1, 64, 1024)),
    *((m, 2048, 256, 64, 6, a, False, "fp16", False) for a in _SWIGLU_GELUTANH for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Swiglu", True, "bf16", True) for m in (16,)),
)  # fmt: skip

# gfx950's default 1-stage ASM region is `token > 16 or inter_dim % 128` (bf16 output,
# Silu or Gelu, Silu only with doweight_stage1); most cases are outside it.
_A8W8_PTOK_DEFAULT = (
    # Default 1-stage ASM `fmoe` (`fmoe_g1u1`, `fmoe_g1u1_tkw1` with doweight_stage1).
    *((m, 4096, 384, 128, 8, "Silu", False, "bf16", False) for m in (64,)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (512,)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (300,)),
    *((m, 4096, 320, 64, 4, "Silu", False, "bf16", False) for m in (8,)),
    *((m, 2048, 256, 64, 8, "Silu", False, "bf16", True) for m in (256,)),
    # Shipped gfx950 cu256 1-stage rows (`fmoe_g1u1`, `fmoe_g1u1_tkw1`).
    *((m, 5120, 1536, 40, 8, "Silu", False, "bf16", False) for m in (4, 8)),
    *((m, 5120, 1024, 128, 1, "Silu", True, "bf16", False) for m in (1, 16)),
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
    # Swiglu and GeluTanh: no 1-stage kernel, so CK stage 1 (ASM for int8) at every
    # token count; and the shape of the gfx942 cu80 MiniMax-M3 rows.
    *((m, 4096, 384, 128, 8, a, False, "bf16", False) for a in _SWIGLU_GELUTANH for m in (1, 64, 512)),
    *((m, 6144, 384, 128, 4, "Swiglu", False, "bf16", False) for m in (16, 256)),
    *((m, 4096, 512, 64, 4, "Swiglu", True, "bf16", False) for m in (8, 300)),
    *((m, 7168, 256, 64, 8, "GeluTanh", False, "bf16", True) for m in (16,)),
    *((m, 7168, 256, 256, 8, a, False, "bf16", False, "i8") for a in _SWIGLU_GELUTANH for m in (16, 64)),
    *((m, 4096, 384, 128, 8, a, False, "bf16", False, "i4") for a in _SWIGLU_GELUTANH for m in (8, 512)),
)  # fmt: skip

# gfx950's default 1-stage ASM region is `token > 32` (bf16 output, no doweight_stage1,
# `inter_dim % 128 == 0`, a kernel for the shape); token <= 32 runs CK with split-K.
_XBF16 = {"env": {"AITER_XBFLOAT16": "1"}}
_A8W8_BLOCKSCALE_DEFAULT = (
    # Default 1-stage ASM `fmoe_fp8_blockscale_g1u1`, also its xbf16 variant.
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (512,)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (256,)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (512,)),
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False, "fp8", _XBF16) for m in (512,)),
    # Shipped gfx950 cu256 1-stage rows, fp8 and xbf16 (`blockscaleBf16`) kernels.
    *((m, 6144, 256, 257, 9, "Silu", False, "bf16", False) for m in (16,)),
    *((m, 6144, 256, 257, 9, "Silu", False, "bf16", True) for m in (16,)),
    *((m, 4096, 384, 128, 8, "Silu", False, "bf16", False) for m in (64,)),
    *((m, 4096, 1024, 288, 8, "Silu", False, "bf16", False) for m in (2,)),
    *((m, 2048, 128, 513, 11, "Silu", False, "bf16", False) for m in (512,)),
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
    # FLAT 1-stage rows (unsorted topk, in-kernel routing). Not with expert_mask, which
    # the FLAT kernels ignore (todo 5.25).
    *((m, 4096, 1024, 288, 8, "Silu", False, "bf16", False) for m in (4,)),
    *((m, 2048, 512, 512, 10, "Silu", False, "bf16", False) for m in (1,)),
    # Swiglu and GeluTanh: no 1-stage kernel; the Swiglu module fails to build. At 1
    # token CK split-K (ksplit 14), whose post-activation takes neither (both raise);
    # at 16 the tiles fill the CUs (ksplit 0).
    *((m, 7168, 256, 256, 8, a, False, "bf16", False) for a in _SWIGLU_GELUTANH for m in (1, 16, 512)),
    *((m, 4096, 384, 128, 8, "Swiglu", True, "bf16", False) for m in (64,)),
)  # fmt: skip

# gfx950: shapes of each shipped cu256 row family, untuned shapes (the FlyDSL heuristic on
# preshuffled weights, CK otherwise), expert-parallel (MXMOE rows fall back to the
# heuristic) and doweight_stage1 (CK).
_FP4_A = {"q_a": "fp4"}
_BIAS = {"bias": True}
_LIMIT = {"swiglu_limit": 7.0}
_FP4_A_BIAS = {**_FP4_A, **_BIAS}
_NO_FLYDSL = {"env": {"AITER_FLYDSL_FORCE": "0"}}
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
    # doweight_stage1 outside CK-Tile split-K (`token * topk > experts`): CK. With a
    # split-K above 1, CK-Tile faults on legacy and v2 alike (todo 5.5).
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False, "fp4") for m in (300, 1024)),
    # CK-Tile split-K (bf16 activations) without the FlyDSL heuristic; with bias, which
    # its stage 1 rejects.
    *((m, 2048, 512, 64, 6, "Silu", False, "bf16", False, "fp4", _NO_FLYDSL) for m in (4,)),
    *((m, 3072, 512, 128, 4, "Silu", False, "bf16", False, "fp4", _NO_FLYDSL) for m in (1,)),
    *((m, 2048, 512, 64, 6, "Silu", False, "bf16", False, "fp4", {**_NO_FLYDSL, **_BIAS}) for m in (4,)),
    *((m, 3072, 384, 128, 8, "Silu", False, "bf16", True, "fp4") for m in (16, 512)),
    *((m, 5120, 640, 64, 3, "Silu", False, "bf16", True, "fp4") for m in (64,)),
    # Not preshuffled: CK.
    *((m, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4u") for m in (16, 256)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4u") for m in (5,)),
    # Bias on Silu: legacy and v2 ignore it.
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4", _BIAS) for m in (64,)),
    # Swiglu: fp4 activations from 256 tokens (below, with quant_dtype_a). Tuned FlyDSL
    # + layout-v2 and MXMOE rows; untuned, the FlyDSL heuristic (with the bias it adds
    # for Swiglu), with bias tensors and swiglu_limit.
    *((m, 3072, 512, 128, 4, "Swiglu", False, "bf16", False, "fp4") for m in (256, 4096)),
    *((m, 6144, 768, 129, 5, "Swiglu", False, "bf16", False, "fp4", _FP4_A) for m in (8,)),
    *((m, 6144, 1536, 129, 5, "Swiglu", False, "bf16", False, "fp4", _FP4_A) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4") for m in (512,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _BIAS) for m in (512,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _LIMIT) for m in (512,)),
    # Not preshuffled: CK-Tile, which has no fp4-activation instance (todo 5.11).
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4u") for m in (512,)),
    *((m, 3072, 512, 128, 4, "Swiglu", False, "bf16", False, "fp4u", _FP4_A) for m in (64,)),
    # SiTUv2 with fp4 activations (quant_dtype_a): tuned FlyDSL + layout-v2 and MXMOE
    # rows; untuned, the MXMOE heuristic (a shape the MXMOE sort supports), or the
    # FlyDSL one (with expert_mask, another shape, with bias).
    *((m, 3584, 3072, 56, 16, "Situv2", False, "bf16", False, "fp4", _FP4_A) for m in (128,)),
    *((m, 3584, 384, 896, 16, "Situv2", False, "bf16", False, "fp4", _FP4_A) for m in (16,)),
    *((m, 7168, 256, 256, 8, "Situv2", False, "bf16", False, "fp4", _FP4_A) for m in (1024,)),
    *((m, 7168, 256, 256, 8, "Situv2", False, "bf16", True, "fp4", _FP4_A) for m in (1024,)),
    *((m, 4096, 512, 128, 8, "Situv2", False, "bf16", False, "fp4", _FP4_A) for m in (64,)),
    *((m, 4096, 512, 128, 8, "Situv2", False, "bf16", False, "fp4", _FP4_A_BIAS) for m in (64,)),
    # FLAT 1-stage rows (unsorted topk, in-kernel routing). Not with expert_mask, which
    # the FLAT kernels ignore (todo 5.25).
    *((m, 8192, 256, 512, 10, "Silu", False, "bf16", False, "fp4") for m in (1,)),
    *((m, 4096, 512, 256, 1, "Silu", False, "bf16", False, "fp4") for m in (8,)),
    *((m, 4096, 512, 513, 11, "Silu", False, "bf16", False, "fp4") for m in (4,)),
)  # fmt: skip

_G1U0 = {"g1u0": True}
_INTERLEAVE = {"interleave": True}

# gfx950, bf16 activations into MXFP4 weights: Swiglu below 256 tokens, SiTUv2, Relu2.
_A16W4 = (
    # Tuned: the `swiglu_mxfp4_bf16_cktile` rows (CK-Tile split-K with the row's ksplit),
    # FlyDSL Swiglu and SiTUv2 rows.
    *((m, 3072, 512, 128, 4, "Swiglu", False, "bf16", False, "fp4") for m in (1, 16, 128)),
    *((m, 6144, 768, 129, 5, "Swiglu", False, "bf16", False, "fp4") for m in (8, 64)),
    *((m, 3584, 512, 896, 16, "Situv2", False, "bf16", False, "fp4") for m in (16,)),
    # FLAT 1-stage rows (flat 1 and 2); with expert_mask, legacy and v2 both raise (the
    # a16w4 SiTUv2 guard).
    *((m, 3584, 384, 896, 16, "Situv2", False, "bf16", False, "fp4") for m in (1, 16)),
    *((m, 3584, 384, 896, 16, "Situv2", False, "bf16", True, "fp4") for m in (16,)),
    # Untuned Swiglu: CK-Tile split-K; FlyDSL where CK-Tile's stage 2 is unsafe
    # (`inter_dim % 256`); not preshuffled, CK-Tile with split-K 2 (ROCM-25478); bias.
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4") for m in (5, 128)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", True, "fp4") for m in (64,)),
    *((m, 4096, 384, 128, 8, "Swiglu", False, "bf16", False, "fp4") for m in (32,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4u") for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _BIAS) for m in (64,)),
    # Gate and up interleaved: the CK-Tile gate-up kernel (split-K 1).
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _INTERLEAVE) for m in (16, 128)),
    # Untuned SiTUv2: FlyDSL. With expert_mask, bias or routed weights in stage 1,
    # legacy and v2 both raise (the a16w4 SiTUv2 guard ahead of the kernels).
    *((m, 4096, 512, 128, 8, "Situv2", False, "bf16", False, "fp4") for m in (64, 1024)),
    *((m, 4096, 512, 128, 8, "Situv2", False, "bf16", True, "fp4") for m in (64,)),
    *((m, 4096, 512, 128, 8, "Situv2", False, "bf16", False, "fp4", _BIAS) for m in (64,)),
    *((m, 2048, 512, 64, 6, "Situv2", True, "bf16", False, "fp4") for m in (4,)),
    # Relu2 with a gate-only w1: CK-Tile.
    *((m, 7168, 256, 256, 8, "Relu2", False, "bf16", False, "fp4", _G1U0) for m in (1, 64, 512)),
)  # fmt: skip

# gfx950 and gfx942, bf16 activations into int4 weights ("wi4"): the tuned rows (gfx950:
# Kimi-K2, Silu; gfx942: Kimi-K3, SiTUv2) and the untuned fallback.
_A16WI4 = (
    *((m, 7168, 256, 384, 8, "Silu", False, "bf16", False, "wi4") for m in (1, 16, 128)),
    *((m, 3584, 3072, 112, 15, "Situv2", False, "bf16", False, "wi4") for m in (4,)),
    *((m, 2048, 768, 128, 8, "Silu", False, "bf16", False, "wi4") for m in (5, 300, 2048)),
    *((m, 2048, 768, 128, 8, "Situv2", False, "bf16", False, "wi4") for m in (64,)),
    *((m, 2048, 768, 128, 8, "Silu", False, "fp16", False, "wi4") for m in (40,)),
    *((m, 2048, 768, 128, 8, "Silu", False, "bf16", True, "wi4") for m in (64,)),
    *((m, 2048, 768, 128, 8, "Silu", False, "bf16", False, "wi4", _BIAS) for m in (64,)),
)  # fmt: skip

_FP8_A = {"q_a": "fp8"}
_FP8_A_BIAS = {**_FP8_A, **_BIAS}
_FP8_A_INTERLEAVE = {**_FP8_A, **_INTERLEAVE}
_FP8_A_STD = {**_FP8_A, "std_shuffle": True}

# gfx950, mxfp8 activations into MXFP4 weights (quant_dtype_a=fp8).
_A8W4 = (
    # Tuned FlyDSL rows: stage 1 quantizing to fp8 into a layout GEMM2 (Silu, Swiglu,
    # SiTUv2) or a FlyDSL stage 2, a plain stage 1 into a FlyDSL stage 2, a stage 1
    # quantizing to fp8 into a CK-Tile stage 2; with expert_mask.
    *((m, 4096, 256, 256, 6, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (8, 128)),
    *((m, 3072, 3072, 128, 4, "Swiglu", False, "bf16", False, "fp4", _FP8_A) for m in (16,)),
    *((m, 3584, 384, 896, 16, "Situv2", False, "bf16", False, "fp4", _FP8_A) for m in (4,)),
    *((m, 7168, 3072, 24, 6, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (64,)),
    *((m, 5120, 640, 128, 3, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (32, 256)),
    *((m, 3072, 3072, 128, 4, "Swiglu", False, "bf16", False, "fp4", _FP8_A) for m in (128,)),
    *((m, 4096, 256, 256, 6, "Silu", False, "bf16", True, "fp4", _FP8_A) for m in (128,)),
    # Tuned Opus rows: Opus stage 1 into a layout GEMM2 or an Opus stage 2 (atomic,
    # fp8), FlyDSL stage 1 (quantizing or not) into an Opus stage 2 or an Opus layout
    # one, and ASM stage 1 under a `cktile_*` kernelName1 into an Opus stage 2 (on
    # weights the FlyDSL heuristic does not take first). Legacy discards them with
    # expert_mask or weights not preshuffled.
    *((m, 7168, 384, 385, 7, "Silu", False, "bf16", False, "fp4", _FP8_A_STD) for m in (2048,)),
    *((m, 7168, 384, 384, 6, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (1, 2, 512, 2048)),
    *((m, 7168, 384, 384, 6, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (64,)),
    *((m, 7168, 384, 385, 7, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (16,)),
    *((m, 3584, 384, 896, 16, "Situv2", False, "bf16", False, "fp4", _FP8_A) for m in (512, 2048)),
    *((m, 7168, 384, 384, 6, "Silu", False, "bf16", True, "fp4", _FP8_A) for m in (2,)),
    *((m, 7168, 384, 384, 6, "Silu", False, "bf16", False, "fp4u", _FP8_A) for m in (2,)),
    # Untuned, gate and up separated: the FlyDSL heuristic on weights tagged with
    # `gate_up_interleave=False` (as -q 7 builds them; Swiglu with its bias); untagged,
    # CK (Silu, SiTUv2, with expert_mask), CK-Tile split-K, and CK-Tile for Swiglu
    # (with bias, not preshuffled).
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Situv2", False, "bf16", False, "fp4", _FP8_A) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4", _FP8_A_STD) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Situv2", False, "bf16", False, "fp4", _FP8_A_STD) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", True, "fp4", _FP8_A_STD) for m in (64,)),
    *((m, 2048, 768, 128, 8, "Silu", False, "bf16", False, "fp4", _FP8_A_STD) for m in (2,)),
    *((m, 4096, 384, 128, 8, "Situv2", False, "bf16", False, "fp4", _FP8_A_STD) for m in (4,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _FP8_A) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _FP8_A_BIAS) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _FP8_A_STD) for m in (64, 512)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", {**_FP8_A_STD, **_BIAS}) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4u", _FP8_A) for m in (64,)),
    # Gate and up interleaved: the FlyDSL heuristic (Silu, SiTUv2), the CK-Tile gate-up
    # kernel (Swiglu).
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "fp4", _FP8_A_INTERLEAVE) for m in (16, 1024)),
    *((m, 2048, 256, 64, 6, "Situv2", False, "bf16", False, "fp4", _FP8_A_INTERLEAVE) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", _FP8_A_INTERLEAVE) for m in (64,)),
)  # fmt: skip

_STD = {"std_shuffle": True}

# gfx950, mxfp8 activations and weights ("mxfp8", as -q 9 builds them).
_MXFP8 = (
    # Tuned FlyDSL rows (GLM5, MiniMax-M3): stage 1 quantizing to fp8 or not into a
    # layout GEMM2, stage 1 quantizing to fp8 into a FlyDSL stage 2 (Swiglu: the tuned
    # rule adds no bias with fp8 weights); with expert_mask.
    *((m, 6144, 512, 257, 9, "Silu", False, "bf16", False, "mxfp8") for m in (1, 128, 4096)),
    *((m, 6144, 256, 257, 9, "Silu", False, "bf16", False, "mxfp8", _FP8_A) for m in (8,)),
    *((m, 6144, 768, 129, 5, "Swiglu", False, "bf16", False, "mxfp8") for m in (2, 1024)),
    *((m, 6144, 384, 129, 5, "Swiglu", False, "bf16", False, "mxfp8") for m in (4, 2048)),
    *((m, 6144, 384, 129, 5, "Swiglu", False, "bf16", False, "mxfp8", _BIAS) for m in (4,)),
    *((m, 6144, 512, 257, 9, "Silu", False, "bf16", True, "mxfp8") for m in (16,)),
    # Untuned: the FlyDSL heuristic (gate and up interleaved, or separated with w1
    # tagged `gate_up_interleave=False`; Swiglu with its bias); untagged, CK (Silu,
    # Swiglu, `doweight_stage1`, expert_mask).
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "mxfp8", _INTERLEAVE) for m in (16, 1024)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "mxfp8", _INTERLEAVE) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "mxfp8", {**_INTERLEAVE, **_BIAS}) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "mxfp8") for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "mxfp8", _BIAS) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", False, "mxfp8", _STD) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "mxfp8", _STD) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Silu", True, "bf16", False, "mxfp8", _STD) for m in (64,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "bf16", True, "mxfp8", _STD) for m in (64,)),
)  # fmt: skip

# gfx950, FHMoE (calls with the shared expert, as op_tests/test_fhmoe.py builds them;
# `E` counts the shared row, `topk` does not): options "interleave" (default True),
# "clamp_shared", "block_size_M".
_SEPARATED = {"interleave": False}
_UNCLAMPED = {"clamp_shared": False}
_FHMOE = (
    # MXFP4 routed weights, the main table: fp8 activations into interleaved gate/up,
    # fp4 into separated; the FlyDSL heuristic.
    *((m, 256, 128, 9, 2, "Silu", False, "bf16", False, "fp4") for m in (4, 64)),
    *((m, 256, 128, 9, 2, "Silu", False, "bf16", False, "fp4", _SEPARATED) for m in (4, 64)),
    # DSV4 I384, the dedicated table: tuned token counts; beyond its last row, both raise.
    *((m, 7168, 384, 385, 6, "Silu", False, "bf16", False, "fp4") for m in (1, 128, 2048, 4096)),
    # MXFP8 routed weights: HY4 in the dedicated table (shared expert unclamped or
    # clamped), another geometry on the main table.
    *((m, 6144, 256, 257, 8, "Silu", False, "bf16", False, "mxfp8", _UNCLAMPED) for m in (4, 512)),
    *((m, 6144, 256, 257, 8, "Silu", False, "bf16", False, "mxfp8") for m in (16,)),
    *((m, 2048, 256, 65, 4, "Silu", False, "bf16", False, "mxfp8") for m in (8,)),
    # Both raise: a block size other than the selected kernel's, an unclamped shared
    # expert with MXFP4 routed weights.
    (4, 256, 128, 9, 2, "Silu", False, "bf16", False, "fp4", {"block_size_M": 64}),
    (4, 256, 128, 9, 2, "Silu", False, "bf16", False, "fp4", _UNCLAMPED),
)  # fmt: skip

_CASES = {
    "a8w8_pt": _A8W8_PT_DEFAULT,
    "a16w16": _A16W16_DEFAULT,
    "a8w8_ptok": _A8W8_PTOK_DEFAULT,
    "a8w8_blockscale": _A8W8_BLOCKSCALE_DEFAULT,
    "a4w4": _A4W4,
    "a16w4": _A16W4,
    "a16wi4": _A16WI4,
    "a8w4": _A8W4,
    "mxfp8": _MXFP8,
    "fhmoe": _FHMOE,
}
# The dispatch key's quant type and weight dtype of each path's tuned rows.
_ROWS = {
    "a8w8_pt": ("QuantType.per_Tensor", "float8"),
    "a16w16": ("QuantType.No", "16"),
    "a8w8_ptok": ("QuantType.per_Token", "float8"),
    "a8w8_blockscale": ("QuantType.per_1x128", "float8"),
    "a4w4": ("QuantType.per_1x32", "float4"),
    "a16w4": ("QuantType.per_1x32", "float4"),
    "a16wi4": ("QuantType.per_1x32", "int4"),
    "a8w4": ("QuantType.per_1x32", "float4"),
    "mxfp8": ("QuantType.per_1x32", "float8"),
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
    blocks, path a8w8_blockscale), `-q 4` (MXFP4, path a4w4; `--preshuffle` for "fp4")
    `-q 6` (bf16 into MXFP4, path a16w4), `-q 7` (mxfp8 into MXFP4, path a8w4) or
    `-q 8` (bf16 into int4, path a16wi4) or `-q 9` (mxfp8, path mxfp8); int8
    weights as `-q 1`/`-q 2` with int8. Unlike op_tests, the weights are scaled by
    1/sqrt(K) (see `weight`)."""
    import torch

    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_topk
    from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_i4_quant
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_a16w4,
        shuffle_scale_for_int4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility import fp4_utils

    if path_name == "fhmoe":
        return _make_fhmoe_call(case)
    token, model_dim, inter_dim, E, topk, act, doweight, out, ep, *rest = case
    weights = rest[0] if rest else "fp8"
    options = rest[1] if len(rest) > 1 else {}
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
        "a16w4": QuantType.per_1x32,
        "a16wi4": QuantType.per_1x32,
        "a8w4": QuantType.per_1x32,
        "mxfp8": QuantType.per_1x32,
    }.get(path_name, QuantType.No)

    def weight(shape, first):
        # Quantized one at a time, the bf16 copy freed: the GPU may be shared.
        # Scaled by 1/sqrt(K) so each GEMM keeps values near 1: unscaled, the fp16
        # outputs reach the fp16 limit, and whether a sum overflows then depends on
        # the order of the GEMM2 atomics.
        w = torch.randn(shape, dtype=dtype, device=device) / shape[-1] ** 0.5
        if weights == "mxfp8":
            qt, scale = per_1x32_f8_scale_f8_quant(
                w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
            )
            qt = qt.view(shape)
            del w
            if options.get("std_shuffle"):
                return shuffle_weight(qt, layout=(16, 16)), fp4_utils.e8m0_shuffle(
                    scale
                )
            # as -q 9 builds them (tagging w1.gate_up_interleave), with gate and up
            # interleaved for "interleave"
            if not first:
                return shuffle_weight_a16w4(qt, 16, False), fp4_utils.e8m0_shuffle(
                    scale
                )
            gate_up = options.get("interleave", False)
            return shuffle_weight_a16w4(qt, 16, gate_up), shuffle_scale_a16w4(
                scale, E, gate_up
            )
        if weights == "i4":
            qt, scale = aiter.pertoken_quant(w, quant_dtype=dtypes.i8, dtypeMax=7)
            del w
            packed = shuffle_weight(qt, (16, 16), use_int4=True)
            return rearrange_4bit_elements(convert_int8_to_uint32_int4(packed)), scale
        if weights == "wi4":
            # -q 8: int4 in [-7, 7] per 32, packed two to a byte after the shuffle.
            qt, scale = per_1x32_i4_quant(w)
            del w
            packed = pack_int8_to_packed_int4(
                shuffle_weight(qt.view(dtypes.i8), (16, 16))
            )
            qt = packed.view(shape[0], shape[1], shape[2] // 2).view(dtypes.i4x2)
            return (
                qt,
                shuffle_scale_for_int4(scale, group_size=32).view(-1).contiguous(),
            )
        if weights in ("fp4", "fp4u"):
            qt, scale = aiter.get_torch_quant(quant_type)(w, quant_dtype=dtypes.fp4x2)
            qt = qt.view(*shape[:-1], shape[-1] // 2)
            del w
            if (
                weights == "fp4"
                and path_name in ("a16w4", "a8w4")
                and not options.get("std_shuffle")
            ):
                # as -q 6 and -q 7 build them (tagging w1.gate_up_interleave); gate and
                # up interleaved for w1 with "interleave"
                gate_up = options.get("interleave", False) and shape[1] != model_dim
                return shuffle_weight_a16w4(qt, 16, gate_up), shuffle_scale_a16w4(
                    scale, E, gate_up
                )
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

    w1_qt, w1_scale = weight(
        (E, inter_dim * (1 if options.get("g1u0") else 2), model_dim), True
    )
    w2_qt, w2_scale = weight((E, model_dim, inter_dim), False)
    global_E = 2 * E if ep else E
    score = torch.randn((token, global_E), dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(hidden, score, topk, True)
    expert_mask = None
    if ep:
        expert_mask = torch.zeros(global_E, dtype=dtypes.i32, device=device)
        expert_mask[:E] = 1
    call = {
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
    if options.get("q_a") in ("fp4", "fp8", "bf16"):
        call["quant_dtype_a"] = {
            "fp4": dtypes.fp4x2,
            "fp8": dtypes.fp8,
            "bf16": dtypes.bf16,
        }[options["q_a"]]
    if options.get("bias"):
        # op_tests/test_moe_2stage.py: in [-1, 1], passed as fp32.
        call["bias1"] = torch.randn((E, inter_dim * 2), device=device).clamp(-1, 1)
        call["bias2"] = torch.randn((E, model_dim), device=device).clamp(-1, 1)
    if "swiglu_limit" in options:
        call["swiglu_limit"] = options["swiglu_limit"]
    if options.get("interleave"):
        call["gate_mode"] = "interleave"
    return call


_FHMOE_WEIGHTS = {}


def _make_fhmoe_call(case):
    """Inputs as op_tests/test_fhmoe.py builds them: routed MXFP4 ("fp4") or MXFP8
    ("mxfp8") weights and the FP8 shared expert as the last of the `E` rows, `topk`
    routed experts per token plus the shared one, `swiglu_limit` 10."""
    import torch

    from aiter.ops.flydsl.moe_common import GateMode

    sys.path.insert(0, str(_REPO / "op_tests"))
    import test_fhmoe

    token, model_dim, inter_dim, E, topk, act, doweight, out, ep, weights, *rest = case
    options = rest[0] if rest else {}
    assert (act, doweight, out, ep) == ("Silu", False, "bf16", False), case
    profile = test_fhmoe._Profile(model_dim, inter_dim, inter_dim, E, topk)
    interleave = options.get("interleave", True)
    built = (profile, interleave, weights)
    if built not in _FHMOE_WEIGHTS:
        _FHMOE_WEIGHTS.clear()  # one set at a time: the GPU may be shared
        torch.cuda.empty_cache()
        _FHMOE_WEIGHTS[built] = test_fhmoe._build_weights(
            profile, interleave, routed_mxfp8=weights == "mxfp8"
        )
    w = _FHMOE_WEIGHTS[built]
    hidden, _, _, all_weight, all_ids = test_fhmoe._route_inputs(
        profile, token, w.routed_w1.device, w.routed_expert_ids
    )
    gate_mode = GateMode.INTERLEAVE if interleave else GateMode.SEPARATED
    call = {
        "hidden_states": hidden,
        "w1": w.routed_w1,
        "w2": w.routed_w2,
        "topk_weight": all_weight,
        "topk_ids": all_ids,
        **test_fhmoe._common_kwargs(profile, w.routed_s1, w.routed_s2, gate_mode),
        "shared_w1": w.shared_w1,
        "shared_w2": w.shared_w2,
        "shared_w1_scale": w.shared_s1,
        "shared_w2_scale": w.shared_s2,
        "shared_expert_id": profile.shared_id,
    }
    for name in ("clamp_shared", "block_size_M"):
        if name in options:
            call[name] = options[name]
    return call


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
    elif path_name == "a8w8_ptok":
        extra = _asm_1stage_rows(df, gfx, cu_num)
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


_1STAGE_SHAPES = ((32, 4096, 512, 64, 8), (256, 7168, 256, 256, 8))


def _asm_1stage_rows(df, gfx, cu_num):
    """Per-token 1-stage rows for the formats only a tuned row runs on the arch
    (`asm.FMOE_1STAGE_TUNED_KEYS`; no shipped table has them): a persistent kernel of
    the format's table at each of `_1STAGE_SHAPES`, the smallest `subGU_n` at the first,
    the largest at the second (the non-persistent 32x128 fp8 kernels give wrong results,
    todo 5.54)."""
    import pandas as pd
    import torch

    from aiter import QuantType
    from aiter.moev2.segments import asm

    fp8 = "torch.float8_e4m3fn" if gfx == "gfx950" else "torch.float8_e4m3fnuz"
    rows = []
    for key in sorted(asm.FMOE_1STAGE_TUNED_KEYS.get(gfx, ()), key=str):
        act, q_type, _out, q_a, _q_w, _g1u1, doweight = key
        if q_type != QuantType.per_Token:
            continue
        int8 = q_a == torch.int8
        name = act.name.lower()
        table = (
            _REPO / "hsa" / gfx / "fmoe" / name
            / f"fmoe_bf16_pertoken{'Int8' if int8 else 'Fp8'}_g1u1_{name}"
            f"{'_tkw1' if doweight else ''}.csv"
        )  # fmt: skip
        kernels = pd.read_csv(table)
        kernels = kernels[(kernels["ps"] == 1) & (kernels["smf"] == 0)]
        for i, (token, model_dim, inter_dim, expert, topk) in enumerate(_1STAGE_SHAPES):
            fits = kernels[inter_dim % kernels["subGU_n"] == 0].sort_values("subGU_n")
            row = dict.fromkeys(df.columns)
            row.update(
                gfx=gfx, cu_num=cu_num, token=token, model_dim=model_dim,
                inter_dim=inter_dim, expert=expert, topk=topk,
                act_type=f"ActivationType.{act.name}", dtype="torch.bfloat16",
                q_dtype_a="torch.int8" if int8 else fp8,
                q_dtype_w="torch.int8" if int8 else fp8,
                q_type="QuantType.per_Token", use_g1u1=1,
                doweight_stage1=int(doweight), block_m=32, ksplit=0,
                kernelName1=fits["knl_name"].iloc[0 if i == 0 else -1], kernelName2="Null",
                run_1stage=1, xbf16=0, flat=0, nt=0,
            )  # fmt: skip
            rows.append({k: v for k, v in row.items() if k in df.columns})
    return pd.DataFrame(rows, columns=df.columns)


def cases(path_name, synth_cases=()):
    if path_name not in _CASES:
        raise SystemExit(f"no GPU cases for path {path_name}")
    return list(_CASES[path_name]) + list(synth_cases)


def _clear_dispatch_caches():
    import aiter.fhmoe as legacy_fhmoe
    import aiter.fused_moe as fm
    from aiter.moev2.migration import (
        a4w4,
        a8w4,
        a8w8_blockscale,
        a8w8_pt,
        a8w8_ptok,
        a16w4,
        a16w16,
        a16wi4,
        fhmoe,
        mxfp8,
    )

    fm.get_2stage_cfgs.cache_clear()
    paths = (
        a8w8_pt,
        a16w16,
        a8w8_ptok,
        a8w8_blockscale,
        a4w4,
        a16w4,
        a16wi4,
        a8w4,
        mxfp8,
        fhmoe,
    )
    for path in paths:
        for cache in (path.LOCAL.dispatch, path.LOCAL.resolve, path.LOCAL.relookup):
            cache.cache_clear()
    # The FHMoE capability checks resolve and log their token buckets once.
    for module in (legacy_fhmoe, fhmoe):
        module._supports_dsv4_i384_fhmoe_config.cache_clear()
        module._supports_mxfp8_fhmoe_config.cache_clear()


def _raised(fn):
    import torch

    try:
        fn()
    except torch.OutOfMemoryError:
        raise  # the case is skipped, not compared
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def run_case(path_name, case):
    """Return a problem string, or None when legacy and v2 agree."""
    env = case[10].get("env", {}) if len(case) > 10 else {}
    with mock.patch.dict(os.environ, env):
        return _run_case(path_name, case)


def _without_build_output(logs):
    """A JIT build that fails is retried on every call, so its log is in both runs, not
    only in the warm-up: compared by module, not by its time or compiler output (in
    the order ninja ran the jobs)."""
    stable = []
    for level, msg in logs:
        failed = re.search(r"failed jit build \[\w+\]", msg)
        stable.append(
            (level, failed.group(0) if failed else re.sub(r"cost [\d.]+s", "", msg))
        )
    return stable


def _run_case(path_name, case):
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
    if _without_build_output(legacy_logs) != _without_build_output(v2_logs):
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
        print(f"{len(skipped)} cases not compared (out of GPU memory): rerun them")
    print(f"{len(problems)} problems")
    return 1 if problems or skipped else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True)
    parser.add_argument("--synth-table", action="store_true")
    parser.add_argument("--synth-limit", type=int, default=48)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.path == "comm_fused":
        sys.path.insert(0, str(_HERE.parent))
        from moe_dispatch import comm_fused_differential

        return comm_fused_differential.main()
    if args.path == "mega_moe_v2":
        sys.path.insert(0, str(_HERE.parent))
        from moe_dispatch import mega_moe_v2_differential

        return mega_moe_v2_differential.main()
    if args.path == "mega_moe_tp":
        sys.path.insert(0, str(_HERE.parent))
        from moe_dispatch import mega_moe_tp_differential

        return mega_moe_tp_differential.main()
    if args.path == "grouped_gfx1250":
        test = _HERE.parent / "test_moev2_grouped_gfx1250.py"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{test}::test_gpu_outputs_match_legacy",
        ]
        return subprocess.run(cmd, cwd=_REPO, check=False).returncode
    if args.path == "mega_moe_gfx1250":
        print(
            "mega_moe_gfx1250: no multi-GPU gfx1250 differential yet; the CPU gate is "
            "tests/test_moev2_mega_moe_gfx1250.py"
        )
        return 1
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
