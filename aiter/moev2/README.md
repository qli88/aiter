# aiter/moev2

## Status / ownership

`aiter.moev2` is under active development (the MoE v2 migration) and no path has
graduated. Owners: @qli88. Please coordinate with them before changing:

- any file under `aiter/moev2/`. The legacy MoE path runs helpers that live here:
  `policy.py`, `shapes.py`, `segments/defaults.py`, `segments/sort.py`, `ck_moe_stage1`
  and `_require_shuffled_gfx950_fp8` in `segments/ck.py`, `asm_stage1` in
  `segments/asm.py`, `execute/metadata.py`, `execute/output.py` (moved from
  `aiter/fused_moe.py`, which re-exports them) and the table helpers in
  `dispatch/tuned.py`. `aiter.fused_moe` also re-exports `routing/topk.py` and
  `reference/torch_moe.py` for its callers and tests. Changes must keep legacy behavior;
- the hook sites in `aiter/fused_moe.py` (comments starting `NOTE: aiter.moev2`);
- legacy code that `aiter.moev2` copies instead of calling, where a fix must be mirrored:
  `_fused_moe_impl` and `fused_moe_2stages` (`execute/executor.py`, `api.dispatch_key`),
  `get_2stage_cfgs` (`dispatch/resolver.py`, `dispatch/ranking.py`), and the arguments
  `fused_moe` passes to `fused_moe_` (`api.op_call`).

Checks after a change on either side (`tests/moe_dispatch/`):

    python tests/moe_dispatch/snapshot.py --check        # legacy dispatch unchanged
    python tests/moe_dispatch/snapshot.py --update       # after an intended change; paste its output into the PR
    python tests/moe_dispatch/differential.py --check    # v2 plans equal legacy's (CPU)
    AITER_MOEV2_GPU_DIFFERENTIAL=1 python -m pytest tests/test_moev2_differential.py  # outputs (GPU)
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt --compare                   # GPU suites, off vs v2

## Overview

MoE v2 rebuilds `fused_moe` dispatch and execution from small parts: a problem
description, segments (one per kernel family) and a plan that chains them. It lives
next to the legacy code (`aiter/fused_moe.py`, `aiter/fhmoe.py`,
`aiter/fused_moe_bf16_asm.py`) and takes over one group of calls (a *path*) at a time.

- `AITER_MOEV2` selects the paths: unset runs the graduated paths, `off` none, `on`
  every registered path, and `a8w8_pt,a4w4` exactly the listed ones. No path is
  graduated, so by default every call runs the legacy code.
- `aiter/moev2` never imports the legacy MoE modules; the legacy modules may import from
  it (`tests/test_moev2_imports.py` enforces this).
- Helpers that both sides need are moved here verbatim and imported back into
  `aiter/fused_moe.py` under their old names, so existing imports keep working.

Status: nine paths are registered, none graduated: `a8w8_pt` (per-tensor fp8 and int8),
`a16w16` (unquantized bf16/fp16), `a8w8_ptok` (per-token fp8 and int8, and int4 weights
with fp8 activations), `a8w8_blockscale` (fp8 in 1x128 / 128x128 blocks), `a4w4`
(MXFP4 activations and weights), `a16w4` (bf16 activations into MXFP4 weights),
`a16wi4` (bf16 activations into int4 weights), `a8w4` (mxfp8 activations into MXFP4
weights) and `mxfp8` (mxfp8 activations and weights).

## How a call flows

1. The legacy `fused_moe` (entry `api`) or `fused_moe_` (entry `op`) asks
   `paths.route_to_v2` whether an enabled path claims the call.
2. The path builds a `MoeProblem` from the call and resolves it to a plan: the tuned row
   if one applies (checked as the legacy code checks it), else the default ranking. If
   the row needs a kernel family the path has not ported, or the ranking sends the
   problem to legacy (or lists no feasible structure), the path returns `DECLINED` and
   the legacy code runs the call.
3. The executor runs the plan's segments in order.

Lifetime: *permanent* files stay after the migration. *To be removed* files exist only
for the migration and are deleted when every path has graduated and the legacy code
is retired (legacy `fused_moe.py` and `fhmoe.py` become re-export shims over
`aiter.moev2`).

## Package files

| File | Lifetime | Purpose |
|---|---|---|
| `__init__.py` | permanent | Package docstring: coexistence with the legacy code and the import rule. |
| `envs.py` | permanent; the `AITER_MOEV2` accessor is removed | `AITER_MOEV2` parsing, read once and cached (tests call `reload()`). |
| `paths.py` | to be removed | Path registry: `route_to_v2(call, entry)` picks the path for a call, `DECLINED` hands it back to legacy. Graduation can be scoped to archs, and `graduated_claims` keeps a format added to a graduated path behind `AITER_MOEV2` until its own gate. `Path.takes(problem)` names the dispatch keys a path runs, since one segment (CK) serves several paths. Logs the selection once; with `AITER_MOE_EXPLAIN=1`, why an active path does not claim a call. |
| `api.py` | permanent; `op_call` and `dispatch_key` are removed | `aiter.moev2.fused_moe` with the legacy signature (always v2; raises when no path takes the call; like legacy, clears a registered presort when the call returns or raises); `build_problem` and `dispatch_key` (a call to its `MoeProblem` and legacy key); `op_call` (the `fused_moe_` arguments legacy passes). |
| `problem.py` | permanent | `MoeProblem`: what one call must compute (formats, shapes, flags, never tensors). Round-trips the legacy `get_2stage_cfgs` key, so v2 can be compared with legacy key by key. |
| `hooks.py` | permanent; `legacy_kernel_bench` and `legacy_globals` are removed | Context managers for tests and benchmarks (kernel bench callable, sort implementation, `disable_fused_decode_sort_quant`) that replace assigning legacy module globals. `legacy_globals` forwards legacy's `kernel_bench_callable` and `_fused_decode_sort_quant_disabled` globals to v2 runs when they are set, and otherwise leaves the v2 hooks as the caller set them. |
| `policy.py` | permanent | Activation quantization policy, moved from `fused_moe.py`: `resolve_activation_dtype`, `quant_remap`, `_normalize_mxfp4_activation_params`. |
| `shapes.py` | permanent | Shape helpers, moved from `fused_moe.py`: `get_inter_dim`, `nextPow2`, `get_padded_M`, `_is_row_major_2d`. |
| `routing/topk.py` | permanent | Top-k routing, moved from `fused_moe.py`: `fused_topk`. |
| `reference/torch_moe.py` | permanent | PyTorch reference MoE that tests check backends against, moved from `fused_moe.py`: `torch_moe`, `torch_moe_stage1/2`, the activations and the per-1x32 dequant helpers. |
| `dispatch/ops.py` | permanent | The MoE op chain that every plan covers: sort, activation quant, GEMM1, activation, intermediate quant, GEMM2, reduce. |
| `dispatch/buf.py` | permanent | Boundary formats between segments, and why a produced format cannot be read where another is required; a reader may accept any layout. |
| `dispatch/plan.py` | permanent | A plan: the segments that cover the op chain, with their parameters; `starting(op)` finds the segment whose span starts at an op. |
| `dispatch/kernel_names.py` | permanent | One parser for every kernel-name family in tuned tables; `parse(n).format() == n` for every shipped name. |
| `dispatch/tuned.py` | permanent | Tuned-table reader (shared with legacy `get_2stage_cfgs`) and `lookup` with the legacy fallbacks; `@cached` and `clear_cache()`, which reset every v2 cache together. |
| `dispatch/ranking.py` | permanent | The vocabulary of the default ranking table (`Entry`, `Region`, `match`, `when`, the named predicates) and its static checks that it is unambiguous and complete. Predicates include `fmoe_1stage` (a copy of the keys of legacy `fused_moe_1stage_dict`), `blockscale_1stage_kernel` (the kernel library's check that legacy calls), `flydsl_mxfp4` (legacy `use_mxfp4_flydsl`), the legacy CK-Tile branches, the MXMOE fallback and padding. Regions can mark a branch that stays on legacy, and can apply to untuned problems only. A condition list is tested in the order written, so a predicate runs only where legacy evaluates it. The grid check ignores a 1-stage plan a problem's entry lists nowhere (legacy reaches it there through tuned rows only). |
| `dispatch/ranking_table.py` | permanent | The default ranking table itself (used when no tuned row applies) and the rules it follows: no overlapping entries or regions, a default for every problem, `prefer` or `legacy` in a region but not both, a feasible last structure, no dead regions. |
| `dispatch/resolver.py` | permanent | Resolves a `MoeProblem` to a plan in the legacy order, collecting the legacy log lines and raising where legacy raises (unshuffled fp8 CK weights on gfx950); declines rows for families no segment owns, post-reduction rows, 1-stage rows other than a named ASM kernel without a stage 2 (or whose `flat` legacy cannot parse) and a 1-stage plan with a finite SiLU limit (legacy raises), and says why. It runs the 1-stage ASM rows it keeps (FLAT ones on the pass-through sort) and, like legacy, retargets a 1-stage kernel tuned for the twin activation. Like legacy, it discards a post-reduction row for a weight layout that cannot use it, a 1-stage SiTUv2 ASM row on preshuffled MXFP4 weights, a 1-stage row for an activation the ASM kernels lack, and a 1-stage SwiGLU OAI row the call cannot run (no kernel name, not `flat=1`, bias, a `swiglu_limit` other than 7.0, gate/up not separated GGUU), warns, and falls back to the default heuristics; a dedicated table declines instead, as do the SiTUv2 rows legacy raises on. A segment can decline a row it owns (`declines_row`), name the kernels its stage 1 runs without a tuned row (`default_kernels`), the stage-2 kernel it runs for tuned names (`stage2_kernel`) and `MOEMetadata` fields (`metadata_fields`). Stage segments are found by the op they cover, so a stage 1 may span more than GEMM1. A segment left out of every plan by its boundaries is reported with the first mismatch. |
| `dispatch/overrides.py` | permanent | Debug overrides of dispatch decisions (`AITER_BYPASS_TUNE_CONFIG`, `AITER_USE_NT`, `AITER_XBFLOAT16`, `AITER_ONLINE_TUNE`, `AITER_FLYDSL_FORCE`, `AITER_SITUV2_A4W4_FP8_INTER`, ...), read where legacy reads them. |
| `dispatch/profile.py` | permanent | Measured runners-up per tuned key; empty for now. |
| `segments/base.py` | permanent | The `Segment` interface. A segment declares what it can run (`Cap`); anything a problem has that no `Cap` declares rejects the segment. It also declares the buffers it reads (`reads`), and a plan is feasible only if each matches what an earlier segment produced. Optional hooks let it decline an owned tuned row, pick default kernels, rename its stage-2 kernel and set `MOEMetadata` fields. |
| `segments/sort.py` | permanent | Token sorting: `moe_sorting` and its helpers, moved from `fused_moe.py`, which imports them (Opus, CK or FlyDSL backend; local topk ids, the adaptive auxiliary sort, FLAT's unsorted input), with the fused decode sort + MXFP4 quant; segment `sort` runs the plain sort through it, segment `sort.aux` the MXFP4 auxiliary sort and segment `sort.flat` the pass-through of the 1-stage FLAT kernels (flat 1 and 2). |
| `segments/quant.py` | permanent | Activation quantization before GEMM1 (with the scale transposed when the stage 1 kernel needs it) and between the GEMMs, including the sorted MXFP4 quant. |
| `segments/ck.py` | permanent | CK 2-stage GEMM kernels for per-tensor and per-token fp8 and int8, fp8 activations with int4 weights, 1x128 blockscale fp8, fp4 1x32 (declining tuned CK rows on preshuffled MXFP4 weights, which legacy overrides or keeps depending on its default MXFP4 choices) and bf16/fp16 (stage 1 g1u1 only); `ck_moe_stage1` and `_require_shuffled_gfx950_fp8` moved from `fused_moe.py`. |
| `segments/asm.py` | permanent | ASM stage-1 kernels: per-tensor and per-token fp8 and int8 (int8 without `doweight_stage1`), for 1x128 fp8 the kernel that also quantizes the intermediate, writing it and its scales into one buffer (it covers GEMM1 and the intermediate quant), and the stage 1 legacy runs for a tuned a8w4 `cktile_*` kernelName1 (`asm.stage1_cktile`); the 1-stage `fmoe` family (`asm.fmoe_1stage`: the legacy default regions and tuned rows, FLAT too). `asm_stage1`, `fused_moe_1stage` and `_retarget_asm_activation` moved from `fused_moe.py`. |
| `segments/ck_tile.py` | permanent | CK-Tile 2-stage kernels on MXFP4 weights: `cktile.gemm1` runs legacy's four CK-Tile branches (as a default or for a `swiglu_mxfp4_bf16_cktile` row), `cktile.gemm2` also the `cktile_*` stage 2 of tuned rows. `cktile_moe_stage1/2`, `_is_cktile_mxfp4_stage2_name` and `_cktile_gemm1_k_batch` moved from `fused_moe.py` (which imports them back). |
| `segments/opus.py` | permanent | Opus a8w4 stages, for tuned rows only: `opus.gemm1` (mxfp8 intermediate), `opus.gemm1_layout` (the sorted input of a FlyDSL layout GEMM2) and `opus.gemm2`; `_opus_stage2_fp8_enabled` moved from `fused_moe.py` (which imports it back). |
| `segments/flydsl.py` | permanent | FlyDSL 2-stage stage wrappers, stage-2 route-reduce queries and the MXFP4 reroute and swiglu bias predicates, moved from `fused_moe.py` (which imports them back). Patch `_stage2_scratch` here: `fused_moe.py` holds only an alias. |
| `segments/mxmoe.py` | permanent | MXMOE (`flydsl_mxmoe_*`) stage wrappers, `_make_mxfp4_metadata`, `_mxmoe_gemm2_sort_block_m` and the inline-sort checks, moved from `fused_moe.py` (which imports them back). |
| `segments/mxfp4.py` | permanent | The `per_1x32` FlyDSL segments for every activation format (fp4, mxfp8, bf16 into fp4 or int4 weights, mxfp8 weights): FlyDSL stage 1 (plain, fused fp4 output, layout-v2 output), FlyDSL and layout-v2 stage 2 (the latter with or without the scatter epilog), MXMOE stages 1 and 2 (fp4 or mxfp8 activations), the default FlyDSL, a16wi4 and MXMOE SiTUv2 heuristic kernels, and the legacy `get_2stage_cfgs` checks that discard a FlyDSL, Opus or MXMOE tuned row. |
| `segments/whole_graph.py` | permanent | Whole-graph implementations (`impl__<name>__<config>`) from `aiter.fused_moe_registry`. |
| `segments/defaults.py` | permanent | Default kernel parameters (`BLOCK_SIZE_M`, `get_block_size_M`, `get_ksplit`, `use_nt`), moved from `fused_moe.py`; `default_block_m` / `default_ksplit`, the 2-stage defaults of the legacy default branch per quant type. |
| `execute/executor.py` | permanent | Runs a plan on one call, statement for statement as the legacy `_fused_moe_impl` / `fused_moe_2stages`, including a stage 1 that quantizes its own output; `run_one_stage` for a 1-stage plan (the sort, then `fused_moe_1stage`); `run_mxfp4` for the `per_1x32` branches (auxiliary or fused decode sort, local topk ids, fp4/mxfp8/bf16 activation and intermediate quant, bias, the stage 1 `swiglu_limit` and the FlyDSL, MXMOE, Opus and CK-Tile stage arguments). The prologue normalizes a SiLU `swiglu_limit` as legacy does. |
| `execute/metadata.py` | permanent | `MOEMetadata` and `AsmMoeMode`, moved from `fused_moe.py`. |
| `execute/output.py` | permanent | The caller's `output=` buffer: validation and the final copy, moved from `fused_moe.py`. |
| `migration/__init__.py` | to be removed | The migration paths, one module per path. |
| `migration/local.py` | to be removed | Shared by the single-GPU paths at the `fused_moe_` entry: the common claims checks (`why_not` gives each refusal's reason; per-path allowances: activations, gate modes, bias, activation parameters, padding, `quant_dtype_a`, g1u0; calls legacy rejects with the a16w4 `NotImplementedError` are declined), the per-key cached resolve, the log-only second lookup after the sort (2-stage plans only, as in legacy; a call where it would raise or be declined is declined), the MXFP4 second resolutions of `_fused_moe_impl` (an MXMOE row the caller's `block_size_M` overrides, a tuned inline-sort row the call cannot run; a call whose fallback key v2 declines is declined before anything is logged, and the inline-sort check, which probes a JIT module, runs only after the first lookup has logged, as in legacy), the run and `dispatch_record`. |
| `migration/a8w8_pt.py` | to be removed | Path `a8w8_pt` (per-tensor fp8 and int8 weights): its claims, segments, `takes` and registration. |
| `migration/a16w16.py` | to be removed | Path `a16w16` (unquantized bf16/fp16, `QuantType.No`): its claims, segments, `takes` and registration. |
| `migration/a8w8_ptok.py` | to be removed | Path `a8w8_ptok` (per-token fp8 and int8 weights, and int4 weights packed in `uint32` with fp8 activations): its claims, segments, `takes` and registration. |
| `migration/a8w8_blockscale.py` | to be removed | Path `a8w8_blockscale` (fp8, 1x128 activations and 128x128 weight blocks; `per_128x128` calls too, which legacy remaps): its claims, segments, `takes` and registration. |
| `migration/a4w4.py` | to be removed | Path `a4w4` (MXFP4 activations and weights, `per_1x32`, gfx950; Silu, Gelu, Swiglu, SiTUv2, bias, activation parameters, padding): its claims, segments, `takes` and registration. Calls with `num_local_tokens`, `a1_scale` or activation scales find no 2-stage plan: only a tuned 1-stage row runs them, and the rest are declined. |
| `migration/a16w4.py` | to be removed | Path `a16w4` (bf16 activations into MXFP4 weights, gfx942 and gfx950): its claims, segments, `takes` and registration. |
| `migration/a16wi4.py` | to be removed | Path `a16wi4` (bf16 activations into int4 weights in 1x32 groups, gfx942 and gfx950): its claims, segments, `takes` and registration. |
| `migration/a8w4.py` | to be removed | Path `a8w4` (mxfp8 activations into MXFP4 weights, gfx950): its claims, segments, `takes` and registration. |
| `migration/mxfp8.py` | to be removed | Path `mxfp8` (mxfp8 activations and weights, gfx950): its claims, segments, `takes` and registration. |
| `tuning/__init__.py` | permanent | Empty subpackage; a later step adds the tuner. |

## Tests

| File | Lifetime | Purpose |
|---|---|---|
| `tests/test_moev2_imports.py` | permanent | `aiter/moev2` imports no legacy MoE module. |
| `tests/test_moev2_switch.py` | to be removed | `AITER_MOEV2` parsing and routing; the v2 entry point keeps the legacy signature; v2 runs see the legacy fused decode sort switch. |
| `tests/test_moev2_hooks.py` | permanent | The hooks restore the previous state. |
| `tests/test_moev2_differential.py` | to be removed | Runs the differential harness: dispatch on the CPU, outputs on the GPU (with `AITER_MOEV2_GPU_DIFFERENTIAL=1`; CI runs the rest). |
| `tests/test_moev2_dispatch.py` | permanent; the tests against legacy are removed | Unit tests of the problem, plans, capabilities and segment boundaries, kernel names, resolver and overrides, the executor (including the packed 1x128 stage 1 buffer), and the local paths (claims, `takes`, g1u0 outside the 1-stage keys left to legacy, the second-lookup declines, the 1-stage ASM defaults and tuned rows (FLAT, the twin-activation retarget, the declines), the 1x128 block_m and split-K defaults, the int8 and int4 weight defaults, the a4w4 defaults, row discards and MXFP4 fallbacks, the a16w4, a16wi4, a8w4 and mxfp8 defaults and tuned rows, the CK-Tile alias and stage-2 rows, the 1-stage SiTUv2 ASM row discard, the prologue's `swiglu_limit`, `graduated_claims`); the snapshot's feature variants are valid legacy calls. |
| `tests/test_moev2_ranking.py` | permanent | Unit tests of the ranking table and its checks, including predicates, their evaluation order, legacy regions and untuned-only regions. |
| `tests/test_moe_dispatch_snapshot.py` | permanent | Legacy dispatch matches the golden snapshot. |
| `tests/moe_dispatch/keys.py` | permanent; the feature variants are removed | Dispatch keys per simulated arch: every tuned row plus a grid of untuned keys. For the quant types v2 paths take (`FEATURE_QUANT_TYPES`), feature variants: the same call with one more `get_2stage_cfgs` feature (`num_local_tokens`, activation scales, bias, padding, ...), so the snapshot names the key whose decision changed. The `per_1x32` grid also runs on `MXFP4_SHAPES` and in the `MXFP4_VARIANTS` (bias, gate/up interleaved, tagged separated, padding). |
| `tests/moe_dispatch/arch_sim.py` | permanent | Runs dispatch as if on a given GPU arch, in a process without a GPU. |
| `tests/moe_dispatch/snapshot.py`, `record.py` | permanent | Generate, check or update the legacy dispatch snapshot (`golden/*.jsonl.gz`), read from frozen tuned tables (`golden/inputs/`). |
| `tests/moe_dispatch/differential.py` | to be removed | Requires v2 to make the same dispatch choice as legacy for every key a path claims, and counts the keys a path hands back to legacy as 1-stage or as a named legacy 2-stage branch (whose stage 1 the legacy record must run). `--env NAME=VALUE` compares with live legacy records under a dispatch override. |
| `tests/moe_dispatch/key_checks.py` | to be removed | Per key: the problem/legacy key round trip, and the tuned lookup returns legacy's row. |
| `tests/moe_dispatch/ranking_checks.py` | permanent | Every rankable snapshot key lands in one ranking entry and region, and every region is reached. |
| `tests/moe_dispatch/gpu_differential.py` | to be removed | On a GPU, a path launches the same kernels on the same inputs as legacy and gives the same output (`--path`; `--synth-table` adds tuned rows the shipped tables lack). The cases cover every weight format a path takes, with bias, `swiglu_limit`, gate/up layouts and environment overrides. Sorted buffers and MX scales are compared only where the kernels write them; a JIT build that fails is compared by its module. |
| `tests/moe_dispatch/inventory.py` | to be removed | Every name used from the legacy MoE modules, and every monkeypatch of them. |
| `tests/moe_dispatch/gpu_suites.py` | to be removed | The GPU test suites each path must pass with `AITER_MOEV2=<path>`; `--compare` runs them with `off` and with the path. |
| `tests/moe_dispatch/perf_baseline.py` | permanent | End-to-end MoE performance baseline (`golden/perf_gfx950.json`). |
