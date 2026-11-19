# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The default ranking table: what runs when no tuned row applies (§3.6 of the plan).

The resolver consults it when the tuned table has no row for the problem, or when
AITER_BYPASS_TUNE_CONFIG ignores the row; a tuned row's kernel names otherwise pick the
plan. The vocabulary (`Entry`, `Region`, `match`, `when`, the predicates) and the checks
are in `ranking.py`.

How a problem is ranked:
1. The one `Entry` whose `match(...)` holds is its class. No entry: the local paths do
   not rank the problem.
2. The one `Region` of that entry whose `when(...)` holds; none: `otherwise`.
3. A region with `prefer=[...]` runs the first structure in the list that is feasible
   for the problem; a region with `legacy="..."` declines the problem back to the
   legacy code, which runs the branch named there.

Rules. Every one is enforced: "static" by `ranking.check_table` (in
tests/test_moev2_ranking.py), "grid" by `ranking.check_problems` and `dead_lists` over
the dispatch key grid (tests/moe_dispatch/ranking_checks.py --check, in CI), "runtime"
by an error in `entry_for` / `region_for`.

R1. No overlapping entries. No problem matches two entries: some `match` field must
    separate any two of them (static; runtime).
R2. No overlapping regions. Within one entry, no problem satisfies two `when`s: some
    condition must separate any two regions (static; runtime). Write out the negated
    conditions of the earlier legacy branches, e.g. `when(flydsl_mxfp4=False,
    cktile_split_k=True)` after `when(flydsl_mxfp4=True)`.
R3. Order does not matter. Because of R1 and R2, reordering entries or regions never
    changes a result; do not rely on order to express precedence (follows from R1, R2).
R4. Every problem has a default. Every supported problem of an entry's class lands in a
    region or in `otherwise`. Give `otherwise` unless the regions cover the whole class
    (e.g. a predicate and its negation); an entry needs at least one of the two (static;
    grid).
R5. Each region says exactly one thing: `prefer` (structures, best first) or `legacy`
    (the legacy branch not ported yet), never both or neither (when the table loads).
R6. Every list ends in a structure that is always feasible. The last structure of each
    `prefer` and `otherwise` list may only use segments whose feasibility comes from
    their capabilities alone (no shape-dependent `reject_reason` / `accepts`), so the
    list cannot run out (static); on the grid, the last structure is feasible for every
    problem that lands there (grid).
R7. Structures are well-formed. Each is a chain of known segment names, `|`-separated,
    covering SORT..REDUCE with adjacent ops, listed at most once per list (static).
R8. No dead regions. A region's conditions must not contradict each other (static), and
    every region and non-empty `otherwise` must be reached by at least one supported key
    of the grid (grid). Add the key to the grid rather than keep an unreachable region.
R9. Unique entry names; labels in logs and `explain` are `<entry>[<region>]` (static).
R10. Keep legacy behaviour. Each entry translates one branch of the legacy
    `get_2stage_cfgs` default path 1:1, with its full condition; the comment above the
    entry names that branch. A branch that is not ported is a `legacy=` region, not a
    guess at a structure. `untuned_only=True` marks a legacy branch that also needs
    `cfg is None` (under AITER_BYPASS_TUNE_CONFIG the kept row skips it).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import torch

from aiter import QuantType, dtypes

from .ranking import Entry, Region, match, when

FP8 = frozenset({torch.float8_e4m3fn, torch.float8_e4m3fnuz})
B16 = frozenset({torch.bfloat16, torch.float16})
CK_2STAGE = "sort | quant.a1 | ck.gemm1 | quant.inter | ck.gemm2"
ASM_2STAGE = "sort | quant.a1 | asm.stage1 | quant.inter | ck.gemm2"
FLYDSL_2STAGE = "sort | quant.a1 | flydsl.gemm1 | quant.inter | flydsl.gemm2"
FHMOE_2STAGE = "sort | quant.a1 | flydsl_fhmoe.gemm1 | quant.inter | flydsl_fhmoe.gemm2"
MXMOE_2STAGE = "sort.aux | quant.a1 | mxmoe.gemm1 | mxmoe.gemm2"
CKTILE_2STAGE = "sort | quant.a1 | cktile.gemm1 | quant.inter | cktile.gemm2"
ASM_1STAGE = "sort | asm.fmoe_1stage"
# The legacy 2-stage branches of g1u0 weights: their stage 1 kernels read `w1` as gate
# and up (todo 5.3).
CK_G1U0 = "CK 2-stage g1u0"
ASM_G1U0 = "ASM stage 1 + CK stage 2 g1u0"

RANKING = (
    # get_2stage_cfgs without a tuned row: fused_moe_1stage_dict has no per-tensor
    # entry (run_1stage stays False), and fp8 weights with an empty kernelName1 take
    # the CK branch (`not kernelName1 and q_dtype_w in [..., fp8]`).
    Entry(
        "fp8_per_tensor",
        match(quant_type=QuantType.per_Tensor, w1_dtype=FP8),
        otherwise=[CK_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row: bf16/fp16 weights with an empty kernelName1
    # take the CK branch. The QuantType.No keys of fused_moe_1stage_dict are all g1u0.
    Entry(
        "a16w16",
        match(quant_type=QuantType.No, w1_dtype=B16, g1u1=True),
        otherwise=[CK_2STAGE],
    ),
    # g1u0: keys in fused_moe_1stage_dict (gfx942, Silu) run 1-stage ASM `fmoe` when
    # `token < 256`; everything else takes the CK branch (no segment runs g1u0 there,
    # so the other keys have no feasible plan and are declined).
    Entry(
        "a16w16_g1u0",
        match(quant_type=QuantType.No, w1_dtype=B16, g1u1=False),
        regions=[
            Region(when(fmoe_1stage=True, m__lt=256), prefer=[ASM_1STAGE]),
            Region(when(fmoe_1stage=True, m__ge=256), legacy=CK_G1U0),
        ],
    ),
    # get_2stage_cfgs without a tuned row, per-token fp8 weights: keys in
    # fused_moe_1stage_dict run 1-stage ASM `fmoe` when `token > 16 or inter_dim % 128`;
    # everything else takes the CK branch (fp8 weights, empty kernelName1).
    Entry(
        "fp8_per_token",
        match(quant_type=QuantType.per_Token, w1_dtype=FP8),
        regions=[
            Region(when(fmoe_1stage=True, m__gt=16), prefer=[ASM_1STAGE]),
            Region(
                when(fmoe_1stage=True, m__le=16, inter_dim_128=False),
                prefer=[ASM_1STAGE],
            ),
        ],
        otherwise=[CK_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row, 1x128 fp8: keys in fused_moe_1stage_dict run
    # 1-stage ASM `fmoe_fp8_blockscale_g1u1` when `token > 32`, `inter_dim % 128 == 0`
    # and the kernel library has a kernel for the shape; everything else takes the CK
    # branch, with the 1x128 block_m and split-K defaults. Legacy also requires no
    # clamping swiglu_limit there, which holds for every 1x128 key a local path takes.
    Entry(
        "fp8_blockscale",
        match(quant_type=QuantType.per_1x128, w1_dtype=FP8),
        regions=[
            Region(
                when(
                    fmoe_1stage=True,
                    m__gt=32,
                    inter_dim_128=True,
                    blockscale_1stage_kernel=True,
                ),
                prefer=[ASM_1STAGE],
            ),
        ],
        otherwise=[CK_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row, int8 weights: per-token keys in
    # fused_moe_1stage_dict run 1-stage ASM `fmoe` when `token > 32`; everything else
    # takes the ASM branch (int8 is not in the CK branch's weight list), with block_m
    # rounded up to a CK stage 2 size.
    Entry(
        "int8_per_token",
        match(quant_type=QuantType.per_Token, w1_dtype=dtypes.i8, g1u1=True),
        regions=[Region(when(fmoe_1stage=True, m__gt=32), prefer=[ASM_1STAGE])],
        otherwise=[ASM_2STAGE],
    ),
    # g1u0: keys in fused_moe_1stage_dict (gfx942, Silu and Gelu) run 1-stage ASM
    # `fmoe_int8_g1u0` when `token > 32`; everything else takes the ASM branch (as for
    # a16w16_g1u0, the other keys have no feasible plan).
    Entry(
        "int8_per_token_g1u0",
        match(quant_type=QuantType.per_Token, w1_dtype=dtypes.i8, g1u1=False),
        regions=[
            Region(when(fmoe_1stage=True, m__gt=32), prefer=[ASM_1STAGE]),
            Region(when(fmoe_1stage=True, m__le=32), legacy=ASM_G1U0),
        ],
    ),
    Entry(
        "int8_per_tensor",
        match(quant_type=QuantType.per_Tensor, w1_dtype=dtypes.i8),
        otherwise=[ASM_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row, int4 weights packed in uint32: no key of
    # fused_moe_1stage_dict (its int4 key has the `torch.int4` dtype), and uint32
    # weights with an empty kernelName1 take the CK branch.
    Entry(
        "int4_per_token",
        match(quant_type=QuantType.per_Token, w1_dtype=torch.uint32),
        otherwise=[CK_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row, MXFP4 activations and weights, Silu or Gelu
    # (no per_1x32 key runs 1-stage by default): the heuristic FlyDSL fallback when
    # `use_mxfp4_flydsl` holds; else CK-Tile split-K for shuffled weights with a split-K
    # above 1; else the CK branch (fp4 weights, empty kernelName1), which ignores the
    # padding (the CK segments take none).
    Entry(
        "a4w4",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=dtypes.fp4x2,
            w1_dtype=dtypes.fp4x2,
            activation={"silu", "gelu"},
        ),
        regions=[
            Region(when(flydsl_mxfp4=True), prefer=[FLYDSL_2STAGE]),
            Region(
                when(flydsl_mxfp4=False, cktile_split_k=True), prefer=[CKTILE_2STAGE]
            ),
            Region(
                when(flydsl_mxfp4=False, cktile_split_k=False, padded=True),
                legacy="CK 2-stage, padding ignored",
            ),
        ],
        otherwise=[CK_2STAGE],
    ),
    # The same with Swiglu (the segments here run separated gate/up only, so the CK-Tile
    # branch for interleaved gate/up is never reached): the heuristic FlyDSL fallback;
    # else CK-Tile with a split-K of 2 or more (the CK-Tile split-K branch excludes fp4
    # Swiglu, the next one takes every Swiglu key without a kernelName1).
    Entry(
        "a4w4_swiglu",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=dtypes.fp4x2,
            w1_dtype=dtypes.fp4x2,
            activation="swiglu",
        ),
        regions=[
            Region(when(flydsl_mxfp4=True), prefer=[FLYDSL_2STAGE]),
            Region(when(flydsl_mxfp4=False), prefer=[CKTILE_2STAGE]),
        ],
    ),
    # The same with SiTUv2: the heuristic MXMOE fallback when `_mxmoe_fallback_ok`
    # holds; else the heuristic FlyDSL fallback; else CK-Tile split-K, or the CK branch
    # (which has no SiTUv2 instance).
    Entry(
        "a4w4_situv2",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=dtypes.fp4x2,
            w1_dtype=dtypes.fp4x2,
            activation="situv2",
        ),
        regions=[
            Region(when(mxmoe_fallback=True), prefer=[MXMOE_2STAGE], untuned_only=True),
            Region(
                when(mxmoe_fallback=False, flydsl_mxfp4=True), prefer=[FLYDSL_2STAGE]
            ),
            Region(
                when(mxmoe_fallback=False, flydsl_mxfp4=False),
                legacy="CK-Tile split-K MXFP4 or CK 2-stage SiTUv2",
            ),
        ],
    ),
    # get_2stage_cfgs without a tuned row, bf16 activations into MXFP4 weights (a16w4),
    # Swiglu: CK-Tile for gate/up interleaved; the FlyDSL a16w4 heuristic where CK-Tile
    # is unsafe (`_is_a16w4`); else CK-Tile split-K, or CK-Tile with a split-K of 2 or
    # more (`cktile_default_kernels` tells them apart).
    Entry(
        "a16w4_swiglu",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=dtypes.bf16,
            w1_dtype=dtypes.fp4x2,
            activation="swiglu",
        ),
        regions=[
            Region(when(cktile_interleaved_swiglu=True), prefer=[CKTILE_2STAGE]),
            Region(
                when(cktile_interleaved_swiglu=False, flydsl_mxfp4=True),
                prefer=[FLYDSL_2STAGE],
            ),
        ],
        otherwise=[CKTILE_2STAGE],
    ),
    # The same with SiTUv2: the FlyDSL a16w4 heuristic; else CK-Tile split-K; else the
    # CK branch (fp4 weights, empty kernelName1).
    Entry(
        "a16w4_situv2",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=dtypes.bf16,
            w1_dtype=dtypes.fp4x2,
            activation="situv2",
        ),
        regions=[
            Region(when(flydsl_mxfp4=True), prefer=[FLYDSL_2STAGE]),
            Region(
                when(flydsl_mxfp4=False, cktile_split_k=True), prefer=[CKTILE_2STAGE]
            ),
            Region(
                when(flydsl_mxfp4=False, cktile_split_k=False),
                legacy="CK 2-stage a16w4",
            ),
        ],
    ),
    # Gate-only Relu2: the CK-Tile Relu2 branch (gfx950, preshuffled); else CK-Tile
    # split-K; else the CK branch.
    Entry(
        "a16w4_relu2",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=dtypes.bf16,
            w1_dtype=dtypes.fp4x2,
            activation="relu2",
        ),
        regions=[
            Region(when(cktile_relu2=True), prefer=[CKTILE_2STAGE]),
            Region(
                when(cktile_relu2=False, cktile_split_k=True), prefer=[CKTILE_2STAGE]
            ),
            Region(
                when(cktile_relu2=False, cktile_split_k=False),
                legacy="CK 2-stage a16w4",
            ),
        ],
    ),
    # get_2stage_cfgs without a tuned row, mxfp8 activations into MXFP4 weights (a8w4),
    # Silu or SiTUv2: the heuristic FlyDSL fallback (gate/up interleaved); else CK-Tile
    # split-K; else the CK branch (fp4 weights, empty kernelName1), which ignores the
    # padding.
    Entry(
        "a8w4",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=FP8,
            w1_dtype=dtypes.fp4x2,
            activation={"silu", "situv2"},
        ),
        regions=[
            Region(when(flydsl_mxfp4=True), prefer=[FLYDSL_2STAGE]),
            Region(
                when(flydsl_mxfp4=False, cktile_split_k=True), prefer=[CKTILE_2STAGE]
            ),
            Region(
                when(flydsl_mxfp4=False, cktile_split_k=False, padded=True),
                legacy="CK 2-stage, padding ignored",
            ),
        ],
        otherwise=[CK_2STAGE],
    ),
    # The same with Swiglu: CK-Tile for gate/up interleaved; the FlyDSL fallback (gate
    # and up separated, w1 tagged `gate_up_interleave=False`); else CK-Tile split-K, or
    # CK-Tile with a split-K of 2 or more (`cktile_default_kernels` tells them apart).
    Entry(
        "a8w4_swiglu",
        match(
            quant_type=QuantType.per_1x32,
            act_dtype=FP8,
            w1_dtype=dtypes.fp4x2,
            activation="swiglu",
        ),
        regions=[
            Region(when(cktile_interleaved_swiglu=True), prefer=[CKTILE_2STAGE]),
            Region(
                when(cktile_interleaved_swiglu=False, flydsl_mxfp4=True),
                prefer=[FLYDSL_2STAGE],
            ),
        ],
        otherwise=[CKTILE_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row, mxfp8 activations and weights: the heuristic
    # FlyDSL fallback (gate/up interleaved, or w1 tagged `gate_up_interleave=False`);
    # else the CK branch (fp8 weights, empty kernelName1), which ignores the padding.
    Entry(
        "mxfp8",
        match(quant_type=QuantType.per_1x32, act_dtype=FP8, w1_dtype=FP8),
        regions=[
            Region(when(flydsl_mxfp4=True), prefer=[FLYDSL_2STAGE]),
            Region(
                when(flydsl_mxfp4=False, padded=True),
                legacy="CK 2-stage, padding ignored",
            ),
        ],
        otherwise=[CK_2STAGE],
    ),
    # get_2stage_cfgs without a tuned row, bf16 activations into int4 weights (a16wi4):
    # the untuned a16wi4 FlyDSL fallback, ahead of every other heuristic.
    Entry(
        "a16wi4",
        match(quant_type=QuantType.per_1x32, w1_dtype=dtypes.i4x2),
        otherwise=[FLYDSL_2STAGE],
    ),
    # FHMoE (`fhmoe_`): get_2stage_cfgs without a tuned row resolves the key as without
    # the shared expert, then `_use_fhmoe_wrappers` raises unless the plan is FlyDSL
    # 2-stage: the heuristic FlyDSL fallback where `use_mxfp4_flydsl` holds, run with
    # the FHMoE stages; legacy raises for every other default branch.
    Entry(
        "fhmoe",
        match(shared="fused_row"),
        regions=[
            Region(when(flydsl_mxfp4=True), prefer=[FHMOE_2STAGE]),
            Region(
                when(flydsl_mxfp4=False),
                legacy="not FlyDSL 2-stage (FHMoE raises)",
            ),
        ],
    ),
)
