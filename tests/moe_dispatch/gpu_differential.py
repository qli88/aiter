# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU differential of one migration path against the legacy MoE code.

Each case runs on the same inputs with AITER_MOEV2=off and AITER_MOEV2=<path>: the path
must run the call (not decline it), launch the same kernels on equal inputs (exact, but
floating-point tensors within 1 ulp or legacy's own spread), give the same output up to
the GEMM2 atomics, and log the same dispatch lines on a first call (see
`differential.run_differential`). Besides plain calls, the cases cover inputs `claims`
accepts (`_VARIANTS`: all of them for a8w8_pt, `_GENERIC_VARIANTS` for the fp8 per-token,
blockscale and `per_1x32` paths); where legacy raises on one (`_RAISES`), v2 must run the
call and raise the same error.

    python tests/moe_dispatch/gpu_differential.py --path a8w8_pt
    python tests/moe_dispatch/gpu_differential.py --path a16w16 --synth-table

Shipped tables have per-tensor rows for gfx942 only (and bf16 rows for gfx942 and a few
gfx950 shapes). `--synth-table` adds the path's gfx942 CK rows, copied to the running
arch, to the arch's own rows in a temporary AITER_CONFIG_FMOE, so the tuned CK branch runs
on a gfx950 too. ASM and FlyDSL rows are not copied: their gfx942 configs give garbage on
gfx950 (legacy and v2 alike). For a8w8_blockscale, whose shipped ASM stage-1 rows are
gfx942 only, the arch's own CK rows are also added with stage 1 swapped for the arch's
ASM kernel of the same tile M (when it ships one), so that branch runs too; for a8w8_pt,
the gfx942 int8 rows whose ASM kernel the arch ships are copied as well.

A case that does not fit in the free GPU memory is reported as skipped.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
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
# a4w4's, untagged), "env" (environment variables set for the legacy and the v2 runs),
# "variant" (one of `_VARIANTS`, an input `claims` accepts beyond the plain call).
_SWIGLU_0 = {"variant": "swiglu_limit_0"}
_A8W8_PT_DEFAULT = (
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16, 64, 256, 1024, 4096)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (1, 32, 512)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16, 512)),
    *((m, 4096, 512, 64, 8, "Silu", False, "bf16", ep, "fp8", {"variant": v})
      for m, ep in ((16, False), (512, True))
      for v in ("num_local_tokens", "num_local_tokens_fp8_input")),
    *((m, 4096, 512, 64, 8, "Silu", False, "bf16", False, "fp8", {"variant": v})
      for v in ("a1_scale", "a1_a2_scales", "fp8_input", "output", "noncontiguous",
                "block_m_32", "block_m_128", "swiglu_limit_0", "swiglu_limit_inf")
      for m in (16, 512)),
    # Mixtral-8x7B (TP1) as vLLM calls it: swiglu_limit=0.0 on every call.
    *((m, 4096, 14336, 8, 2, "Silu", False, "bf16", False, "fp8", _SWIGLU_0) for m in (4, 256)),
    # vLLM passes swiglu_limit=0.0 with Gelu too; legacy leaves it unnormalized there.
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "fp8", _SWIGLU_0) for m in (32, 512)),
    # int8 (ASM stage 1 + CK stage 2), and the shape of the gfx942 cu80 int8 row.
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False, "i8") for m in (1, 64, 1024)),
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "i8") for m in (32,)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False, "i8") for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True, "i8") for m in (512,)),
    *((m, 6144, 4096, 8, 2, "Silu", False, "bf16", False, "i8") for m in (512,)),
)  # fmt: skip

_A8W8_PT_GFX942_CU80 = (
    # A shipped gfx942-cu80 row selects the FlyDSL whole-graph implementation.
    (1, 4096, 192, 192, 8, "Silu", False, "bf16", False),
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

# Variants that do not depend on the quant format (no static per-tensor scales).
_GENERIC_VARIANTS = ("output", "noncontiguous", "swiglu_limit_0", "swiglu_limit_inf")

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
    *((16, 7168, 256, 256, 8, "Silu", False, "bf16", False, "fp8", {"variant": v})
      for v in _GENERIC_VARIANTS),
    (300, 4096, 512, 64, 4, "Gelu", True, "bf16", False, "fp8", _SWIGLU_0),
)  # fmt: skip

# Outside gfx950's default 1-stage ASM region (`token > 32`, bf16 output, no
# doweight_stage1, a kernel for the shape); token <= 32 runs CK with split-K.
_A8W8_BLOCKSCALE_DEFAULT = (
    *((m, 7168, 256, 256, 8, "Silu", False, "bf16", False) for m in (1, 16, 32)),
    # The shape's shipped gfx950 1-stage Silu rows (4 to 128 tokens) serve Gelu too.
    *((m, 4096, 384, 128, 8, "Gelu", False, "bf16", False) for m in (1, 2)),
    *((m, 4096, 512, 64, 4, "Silu", True, "bf16", False) for m in (8, 300)),
    *((m, 2048, 256, 64, 6, "Silu", False, "fp16", False) for m in (5, 200)),
    *((m, 7168, 256, 64, 8, "Silu", False, "bf16", True) for m in (16,)),
    *((m, 7168, 256, 64, 8, "Silu", True, "bf16", True) for m in (512,)),
    # Shapes with shipped gfx950 cu256 rows (CK, some with split-K).
    *((m, 4096, 384, 128, 8, "Silu", False, "bf16", False) for m in (1, 256, 4096)),
    *((m, 4096, 384, 128, 8, "Silu", True, "bf16", False) for m in (8, 512)),
    *((m, 7168, 512, 257, 9, "Silu", False, "bf16", False) for m in (1, 128)),
    *((m, 4096, 1536, 16, 8, "Silu", False, "bf16", False) for m in (16, 512)),
    *((16, 7168, 256, 256, 8, "Silu", False, "bf16", False, "fp8", {"variant": v})
      for v in _GENERIC_VARIANTS),
    (2, 4096, 384, 128, 8, "Gelu", False, "bf16", False, "fp8", _SWIGLU_0),
)  # fmt: skip

# gfx950: shapes of each shipped cu256 row family, untuned shapes (the FlyDSL heuristic on
# preshuffled weights, CK otherwise), expert-parallel (MXMOE rows fall back to the
# heuristic), doweight_stage1 (CK) and the generic variants.
_FP4_A = {"q_a": "fp4"}
_BIAS = {"bias": True}
_LIMIT = {"swiglu_limit": 7.0}
_FP4_A_BIAS = {**_FP4_A, **_BIAS}
_NO_FLYDSL = {"env": {"AITER_FLYDSL_FORCE": "0"}}
_A4W4 = (
    # FlyDSL stage 1 + layout-v2 stage 2.
    *((m, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4") for m in (1, 64, 1024)),
    *((64, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4", {"variant": v})
      for v in _GENERIC_VARIANTS),
    *((m, 4096, 512, 512, 10, "Silu", False, "bf16", False, "fp4") for m in (16,)),
    # MXMOE stage 1 + layout-v2 stage 2; inline sort keeps strided hidden_states.
    *((m, 5120, 640, 128, 3, "Silu", False, "bf16", False, "fp4") for m in (8, 64, 256)),
    (8, 5120, 640, 128, 3, "Silu", False, "bf16", False, "fp4", {"variant": "noncontiguous"}),
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
    # Silu with swiglu_limit: FlyDSL clamps; CK (no fp4 clamp) and CK-Tile stage 1 raise.
    *((m, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4", _LIMIT) for m in (64,)),
    *((m, 3072, 384, 256, 8, "Silu", False, "bf16", False, "fp4u", _LIMIT) for m in (16,)),
    *((m, 2048, 512, 64, 6, "Silu", False, "bf16", False, "fp4", {**_NO_FLYDSL, **_LIMIT}) for m in (4,)),
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
)  # fmt: skip

_G1U0 = {"g1u0": True}
_INTERLEAVE = {"interleave": True}

# gfx950, bf16 activations into MXFP4 weights: Swiglu below 256 tokens, SiTUv2, Relu2.
_A16W4 = (
    # Tuned: the `swiglu_mxfp4_bf16_cktile` rows (CK-Tile split-K with the row's ksplit),
    # FlyDSL Swiglu and SiTUv2 rows; a 1-stage SwiGLU OAI row legacy discards for bias.
    *((m, 3072, 512, 128, 4, "Swiglu", False, "bf16", False, "fp4") for m in (1, 16, 128)),
    *((m, 6144, 768, 129, 5, "Swiglu", False, "bf16", False, "fp4") for m in (64,)),
    *((m, 6144, 768, 129, 5, "Swiglu", False, "bf16", False, "fp4", _BIAS) for m in (8,)),
    *((m, 3584, 512, 896, 16, "Situv2", False, "bf16", False, "fp4") for m in (16,)),
    # Untuned Swiglu: CK-Tile split-K; FlyDSL where CK-Tile's stage 2 is unsafe
    # (`inter_dim % 256`); not preshuffled, CK-Tile with split-K 2 (ROCM-25478); bias.
    *((m, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4") for m in (5, 128)),
    *((128, 2048, 256, 64, 6, "Swiglu", False, "bf16", False, "fp4", {"variant": v})
      for v in _GENERIC_VARIANTS),
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
    *((300, 2048, 768, 128, 8, "Silu", False, "bf16", False, "wi4", {"variant": v})
      for v in _GENERIC_VARIANTS),
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
    # quantizing to fp8 into a CK-Tile stage 2; MXMOE stage 1 into a layout GEMM2 (the
    # first shapes); with expert_mask, where legacy discards the MXMOE row.
    *((m, 4096, 256, 256, 6, "Silu", False, "bf16", False, "fp4", _FP8_A) for m in (8, 128)),
    *((128, 4096, 256, 256, 6, "Silu", False, "bf16", False, "fp4", {**_FP8_A, "variant": v})
      for v in _GENERIC_VARIANTS),
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
    *((128, 6144, 512, 257, 9, "Silu", False, "bf16", False, "mxfp8", {"variant": v})
      for v in _GENERIC_VARIANTS),
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
}
# Cases for one (gfx, cu_num) only: the shipped rows they select exist only there.
_ARCH_CASES = {("a8w8_pt", "gfx942", 80): _A8W8_PT_GFX942_CU80}
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
    1/sqrt(K) (see `weight`); then the case's options and variant are applied."""
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

    token, model_dim, inter_dim, E, topk, act, doweight, out, ep, *rest = case
    weights = rest[0] if rest else "fp8"
    q_dtype = dtypes.i8 if weights in ("i8", "i4") else dtypes.fp8
    options = rest[1] if len(rest) > 1 else {}
    variant = options.get("variant")
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
                gate_up = options.get("interleave", False) and first
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
    if variant is not None:
        _VARIANTS[variant](call)
    return call


def _a1_scale(call):
    import aiter
    from aiter import QuantType, dtypes

    quant = aiter.get_torch_quant(QuantType.per_Tensor)
    return quant(call["hidden_states"], quant_dtype=dtypes.fp8)


def _set_a1_scale(call):
    call["a1_scale"] = _a1_scale(call)[1]


def _set_a1_a2_scales(call):
    import torch

    _set_a1_scale(call)
    # The GEMM1 outputs stay near 1 (the weights are scaled by 1/sqrt(K)); a fixed
    # scale that keeps them inside the fp8 range.
    call["a2_scale"] = torch.full((1,), 8 / 224, dtype=torch.float32, device="cuda")


def _set_fp8_input(call):
    hidden = call["hidden_states"]
    call["dtype"] = hidden.dtype
    call["hidden_states"], call["a1_scale"] = _a1_scale(call)


def _set_num_local_tokens(call):
    import torch

    token = call["hidden_states"].shape[0]
    call["num_local_tokens"] = torch.tensor(
        [token - token // 4], dtype=torch.int32, device="cuda"
    )


def _set_output(call):
    import torch

    call["output"] = torch.empty_like(call["hidden_states"])


def _set_noncontiguous(call):
    import torch

    hidden = call["hidden_states"]
    wide = torch.zeros(
        (hidden.shape[0], 2 * hidden.shape[1]), dtype=hidden.dtype, device="cuda"
    )
    wide[:, : hidden.shape[1]] = hidden
    call["hidden_states"] = wide[:, : hidden.shape[1]]


def _set_num_local_tokens_fp8_input(call):
    _set_fp8_input(call)
    _set_num_local_tokens(call)


_VARIANTS = {
    "a1_scale": _set_a1_scale,
    "a1_a2_scales": _set_a1_a2_scales,
    "fp8_input": _set_fp8_input,
    "num_local_tokens": _set_num_local_tokens,
    "num_local_tokens_fp8_input": _set_num_local_tokens_fp8_input,
    "output": _set_output,
    "noncontiguous": _set_noncontiguous,
    "block_m_32": lambda call: call.update(block_size_M=32),
    "block_m_128": lambda call: call.update(block_size_M=128),
    "swiglu_limit_0": lambda call: call.update(swiglu_limit=0.0),
    "swiglu_limit_inf": lambda call: call.update(swiglu_limit=float("inf")),
}
# Variants legacy raises on: v2 must run them and raise the same error. The 2-stage
# path quantizes per tensor before GEMM2 (and before GEMM1 from bf16) with
# `per_tensor_quant_hip`, which takes no row count.
_RAISES = {
    "num_local_tokens": "num_rows is not supported for per_tensor_quant_hip",
    "num_local_tokens_fp8_input": "num_rows is not supported for per_tensor_quant_hip",
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

    keep = src["kernelName1"].map(portable).astype(bool)
    src = src[keep & src["kernelName2"].map(portable).astype(bool)]
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


def case_variant(case):
    """The `_VARIANTS` name a case applies, or None."""
    return case[10].get("variant") if len(case) > 10 else None


def cases(path_name, synth_cases=(), gfx=None, cu_num=None):
    if path_name not in _CASES:
        raise SystemExit(f"no GPU cases for path {path_name}")
    selected = list(_CASES[path_name])
    selected += _ARCH_CASES.get((path_name, gfx, cu_num), ())
    return selected + list(synth_cases)


def _clear_dispatch_caches():
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
    )
    for path in paths:
        for cache in (path.LOCAL.dispatch, path.LOCAL.resolve, path.LOCAL.relookup):
            cache.cache_clear()


def _raised(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def _v2_outcome(path_name, call):
    """What the path did with `call`: "not claimed", "declined", "raised" or "ran"."""
    import dataclasses

    import aiter.fused_moe as fm
    from aiter.moev2 import paths
    from moe_dispatch.differential import moev2_mode

    path = paths.registered()[path_name]
    outcome = ["not claimed"]

    def run(c):
        outcome[0] = "raised"
        out = path.run(c)
        outcome[0] = "declined" if out is paths.DECLINED else "ran"
        return out

    paths._REGISTRY[path_name] = dataclasses.replace(path, run=run)
    try:
        with moev2_mode(path_name):
            _raised(lambda: fm.fused_moe(**call))
    finally:
        paths._REGISTRY[path_name] = path
        paths.reload()
    return outcome[0]


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
        legacy_error = _raised(lambda: fm.fused_moe(**call))
    _clear_dispatch_caches()
    with moev2_mode(path_name), captured_logs() as v2_logs:
        v2_error = _raised(lambda: fm.fused_moe(**call))
    if _without_build_output(legacy_logs) != _without_build_output(v2_logs):
        return f"{case}: log lines differ\n    legacy: {legacy_logs}\n    v2:     {v2_logs}"
    expected = _RAISES.get(case_variant(case))
    if expected is not None:
        if not (legacy_error and legacy_error == v2_error and expected in v2_error):
            return (
                f"{case}: expected both to raise {expected!r}; "
                f"legacy raises {legacy_error}; v2 raises {v2_error}"
            )
        _clear_dispatch_caches()
        outcome = _v2_outcome(path_name, call)
        return None if outcome == "raised" else f"{case}: v2 {outcome}, not raised"
    if legacy_error or v2_error:
        return (
            f"{case}: expected successful execution; "
            f"legacy raises {legacy_error}; v2 raises {v2_error}"
        )
    try:
        run_differential(path_name, call)
    except AssertionError as exc:
        return f"{case}: {exc}"
    return None


def _child(args):
    import torch

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    synth_cases = json.loads(os.environ.get("_MOEV2_SYNTH_CASES", "[]"))
    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    todo = cases(args.path, [tuple(c) for c in synth_cases], gfx=gfx, cu_num=cu_num)
    print(f"{gfx} cu{cu_num} torch {torch.__version__}: {len(todo)} cases")
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
    # `tests` so `moe_dispatch` imports, and the repository root so `aiter` is
    # this checkout rather than an installed copy, even one earlier on the path.
    front = [str(_HERE.parent), str(_HERE.parents[1])]
    sys.path[:] = front + [p for p in sys.path if p not in front]
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
