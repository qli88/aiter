# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The v2 tuning candidates (`Segment.param_space`) against the legacy tuner's.

For every tuner key the snapshot's `get_2stage_cfgs` keys give, on every simulated
arch, and per kernel family, the set of `(stage, kernel, block_m, nt)` the legacy
`FmoeTuner` generator lists equals the set v2's segments list, except for the known
differences below, each with its reason; a known difference holds only where v2 lists
nothing and no segment of the family accepts the problem, unless legacy raises (where
v2's guard is the fix, todo 5.46) or the reason says v2 declines on purpose. No
candidate may be listed twice. A tuner key is the 14 fields the legacy tuner tunes
on, per-128x128 tuned as per-1x128; legacy `tune()` skips g1u0 keys, and per-1x32 keys
off gfx950, so v2 must list nothing there. On gfx950 a few synthetic keys add what the
snapshot lacks. Candidates v2 adds by decision, beyond the legacy tuners, are rebuilt
here independently and count as legacy's under a known reason where they add any
(1-stage ASM 5.1k1, CK-Tile a16w4 5.1k2, FHMoE's widened rules 5.1i).

CK-Tile (`cktile`) is checked per snapshot call, since the call's weight layout picks
the branch: no legacy branch reads the tuner's `cktile_a8w4_bm*` stage 1 names, so each
goes through legacy dispatch itself (a one-row tuned table swapped into `fused_moe`,
`get_2stage_cfgs` read back). Where CK-Tile runs, it stands for the alias at the
`block_m` legacy ran (the alias row must run the same); where something else runs, it
is a known difference v2 declines on purpose. Opus stage 1 candidates are listed once
per legacy task: token-major output in `opus` (with stage 2), sorted output in
`opus_sorted`. A 1-stage candidate (`asm_1stage`) is `(asm_1stage[_xbf16], kernel,
block_m, flat)`. MXMOE (`mxmoe`) is the separate `Mxfp4FlydslTuner` at its full search,
each of its candidate pairs a stage 1 and a stage 2 tag. That tuner filters no key (it
times a g1u1 a4w4 bf16 call for any row); the harness gives it the a4w4 keys on gfx950,
where its kernels run, g1u0 included. FHMoE (`fhmoe`) is `tune_fhmoe.py` with its
filters widened for every FHMoE contract (decided in 5.1i), checked per snapshot
FHMoE call (the problem with its shared expert) and on synthetic gfx950 calls: legacy's
candidates, where its contract matches, must be in the widened rules, and v2 lists
exactly those. The whole-graph configs (`whole_graph`, tags `("impl", kernel, block_m,
False)`) are `FmoeTuner.e2e_tune`'s, on gfx942 and gfx950, where its kernels run.

    python tests/moe_dispatch/param_space_checks.py --check [--arch gfx950]
"""

import argparse
import collections
import contextlib
import functools
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

_BLOCK_MS = [16, 32, 64, 128]


def _legacy_tuner():
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    import torch

    import csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune as tune

    # The tuner module sets the default device to cuda on import.
    torch.set_default_device("cpu")
    return tune.FmoeTuner.__new__(tune.FmoeTuner)


@contextlib.contextmanager
def _legacy_pinned(gfx, cu_num):
    """What `arch_sim.simulated_arch` leaves to the arch the process runs on: the
    top-level `chip_info` copy (`get_asm_dir`, the CK lists), the cached
    `get_asm_dir`, and the fp8 keys of `dtype2str_dict`."""
    import torch

    from aiter.jit.core import get_asm_dir
    from aiter.ops import moe_op
    from moe_dispatch.arch_sim import _patched

    fakes = {
        "get_gfx": lambda: gfx,
        "get_gfx_runtime": lambda: gfx,
        "get_cu_num": lambda: cu_num,
    }
    top = sys.modules["chip_info"]
    originals = {f: getattr(top, f) for f in fakes}
    added = [
        t
        for t in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if t not in moe_op.dtype2str_dict
    ]
    with contextlib.ExitStack() as stack:
        for module in list(sys.modules.values()):
            for func, orig in originals.items():
                try:
                    bound = getattr(module, func, None)
                except Exception:  # noqa: BLE001, S112
                    continue
                if bound is orig:
                    stack.enter_context(_patched(module, func, fakes[func]))
        get_asm_dir.cache_clear()
        try:
            for t in added:
                moe_op.dtype2str_dict[t] = "f8"
            yield
        finally:
            get_asm_dir.cache_clear()
            for t in added:
                moe_op.dtype2str_dict.pop(t, None)


def tuner_keys(gfx, cu_num):
    """`{tuner key: [snapshot keys]}`; a synthetic key has none."""
    from aiter import QuantType
    from moe_dispatch import snapshot

    out = {}
    for key in snapshot.dispatch_keys(gfx, cu_num):
        token, model_dim, inter_dim, expert, topk, dtype, q_a, q_w, q_type, g1u1 = (
            key.args[:10]
        )
        activation, doweight = key.args[10], key.args[11]
        if q_type == QuantType.per_128x128:
            q_type = QuantType.per_1x128
        info = (gfx, cu_num, token, model_dim, inter_dim, expert, topk, activation)
        info += (dtype, q_a, q_w, q_type, g1u1, doweight)
        out.setdefault(info, []).append(key)
    for info in synthetic_keys(gfx, cu_num):
        out.setdefault(info, [])
    return out


def synthetic_keys(gfx, cu_num):
    """gfx950 per-1x32 keys the snapshot lacks: fp16 output, a4w4 Gelu, the gpt-oss
    model_dim 2880, a16w4 Swiglu with fp16 output or at `inter_dim % 256`, a8w4 Swiglu
    at every CK-Tile token tier, a4w4 g1u0 (at a shape the MXMOE auxiliary kernels
    cover); each with and without doweight."""
    if gfx != "gfx950":
        return []
    from aiter import ActivationType, QuantType, dtypes

    fp4, fp8 = dtypes.fp4x2, dtypes.fp8
    shapes = (
        (dtypes.fp16, fp4, ActivationType.Silu, 7168, 256),
        (dtypes.fp16, fp8, ActivationType.Swiglu, 7168, 256),
        (dtypes.bf16, fp4, ActivationType.Gelu, 7168, 256),
        (dtypes.bf16, fp8, ActivationType.Swiglu, 2880, 2880),
        (dtypes.bf16, dtypes.bf16, ActivationType.Situv2, 2880, 384),
        (dtypes.fp16, dtypes.bf16, ActivationType.Swiglu, 7168, 256),
        (dtypes.bf16, dtypes.bf16, ActivationType.Swiglu, 7168, 384),
    )
    tokens = [(shape, token) for token in (16, 512) for shape in shapes]
    tokens += [
        ((dtype, q_a, ActivationType.Swiglu, 7168, 256), token)
        for dtype, q_a in ((dtypes.bf16, fp8), (dtypes.fp16, fp8), (dtypes.bf16,) * 2)
        for token in (4096, 20000)
    ]
    g1u0 = [
        (gfx, cu_num, token, 7168, 256, 256, 8, ActivationType.Silu)
        + (dtypes.bf16, fp4, fp4, QuantType.per_1x32, False, False)
        for token in (16, 512)
    ]
    return g1u0 + [
        (gfx, cu_num, token, model_dim, inter_dim, 128, 4, activation)
        + (dtype, q_a, fp4, QuantType.per_1x32, True, doweight)
        for (dtype, q_a, activation, model_dim, inter_dim), token in tokens
        for doweight in (False, True)
    ]


def _synthetic_calls(info, gfx, cu_num):
    """The calls of a key without snapshot calls (synthetic, or only FHMoE calls with a
    `config_file`): the default layout, and the gate/up interleaved and standard MXFP4
    layouts (which pick the CK-Tile branch)."""
    from aiter.moev2.problem import MoeProblem
    from moe_dispatch.keys import _variant_kwargs

    args = (*info[2:7], *info[8:13], info[7], info[13], 0, 0)
    variants = [{}] + [_variant_kwargs(v, info[7]) for v in ("interleave", "standard")]
    return [
        MoeProblem.from_legacy_key(
            *args, **kwargs, arch=gfx, cu_num=cu_num
        ).legacy_key()
        for kwargs in variants
    ]


def legacy_tunes(info):
    """Whether legacy `tune()` tunes `info`: g1u1, per-1x32 only on gfx950."""
    from aiter import QuantType

    return bool(info[12]) and (info[11] != QuantType.per_1x32 or info[0] == "gfx950")


def _is_a8w4(info):
    import torch

    from aiter import QuantType, dtypes

    q_a, q_w, q_type = info[9], info[10], info[11]
    return (
        q_type == QuantType.per_1x32
        and q_w == dtypes.fp4x2
        and q_a in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    )


def _ck_module_fails(info):
    """The CK module of a 1x128 Swiglu key fails to build (`gridwise_moe_gemm_blockscale`
    static-asserts on the activation), so every CK candidate, and every ASM stage 1
    with its CK stage 2, fails when the tuner times it."""
    from aiter import ActivationType, QuantType

    return info[11] == QuantType.per_1x128 and info[7] == ActivationType.Swiglu


def _legacy_asm(tuner, info, call):
    if _ck_module_fails(info):
        return set()
    return {
        ("stage1", tag[2], tag[3], False)
        for (tag, *_rest) in tuner.gen_2stages_asm1_task(info, _BLOCK_MS)
    }


_ASM_1STAGE_ADDED = (
    "1-stage ASM: the tables legacy's tuner does not read, per-1x32 and the gfx942 "
    "`blockscaleBf16` (5.1k1)"
)


def _asm_1stage_added(info):
    """The 1-stage candidates added in 5.1k1, read from the kernel tables: on gfx950 the
    per-1x32 keys of legacy's `fused_moe_1stage_dict` with `model_dim` a multiple of
    512 (the launcher's MXFP4 check) from the `pertokenMXfp4` table (a FLAT kernel
    with `xbf16`, a non-flat one without it for fp4 X), on gfx942 1x128 fp8 the
    `blockscaleBf16` table (`xbf16`); `block_m` the `subGU_m`, `flat` from the table,
    without `smf` or a `subGU_n` that does not divide `inter_dim`."""
    import pandas as pd
    import torch

    import aiter.fused_moe as fm
    from aiter import QuantType, dtypes

    fp8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    gfx, inter_dim, act, q_type = info[0], info[4], info[7], info[11]
    name = str(act).split(".")[-1].lower()
    if q_type == QuantType.per_1x32 and gfx == "gfx950":
        key = (act, q_type, info[8], info[9], info[10], info[12], info[13])
        if key not in fm.fused_moe_1stage_dict[gfx] or info[3] % 512 or info[3] < 512:
            return set()
        quant, fp4_x = "pertokenMXfp4", info[9] == dtypes.fp4x2
    elif q_type == QuantType.per_1x128 and gfx == "gfx942" and info[9] in fp8:
        quant, fp4_x = "blockscaleBf16", None
    else:
        return set()
    tkw1 = "_tkw1" if info[13] else ""
    table = _REPO / "hsa" / gfx / "fmoe" / name
    table /= f"fmoe_bf16_{quant}_g1u{int(info[12])}_{name}{tkw1}.csv"
    if not table.exists():
        return set()
    df = pd.read_csv(table)
    df = df[(df["smf"] == 0) & (inter_dim % df["subGU_n"] == 0)]
    flats = df["flat"].fillna(0).astype(int) if "flat" in df else [0] * len(df)
    out = set()
    for kernel, tile_m, flat in zip(df["knl_name"], df["subGU_m"], flats):
        if fp4_x is None or flat:
            out.add(("asm_1stage_xbf16", kernel, int(tile_m), int(flat)))
        elif fp4_x:
            out.add(("asm_1stage", kernel, int(tile_m), int(flat)))
    return out


def _legacy_asm_1stage(tuner, info, call):
    """`gen_1stage_asm_task`: `(asm_1stage[_xbf16], kernel, block_m, flat)`, with the
    5.1k1 additions (`_asm_1stage_added`) as a known reason where they add any."""
    legacy = {
        (tag[1], tag[2], tag[3], int(tag[4]))
        for (tag, *_rest) in tuner.gen_1stage_asm_task(info)
    }
    added = _asm_1stage_added(info)
    return legacy | added, {_ASM_1STAGE_ADDED} if added - legacy else set()


def _legacy_ck(tuner, info, call):
    if _is_a8w4(info) or _ck_module_fails(info):
        return set()  # a8w4: the CK-Tile generator, family `cktile`
    return {
        (tag[1], tag[2], tag[3], bool(tag[6]))
        for (tag, *_rest) in tuner.gen_2stages_task(info, _BLOCK_MS)
    }


class _EveryKey(dict):
    """A tuned table whose every lookup finds `row`."""

    def __init__(self, row):
        super().__init__()
        self.row = row

    def get(self, key, default=None):
        return self.row


def _legacy_metadata(call, kernel1, kernel2, block_m, ksplit=0):
    """Legacy `get_2stage_cfgs` for `call` with a one-row tuned table naming `kernel1`
    and `kernel2`."""
    import aiter.fused_moe as fm

    row = {"block_m": block_m, "ksplit": ksplit, "kernelName1": kernel1}
    row |= {"kernelName2": kernel2, "run_1stage": 0, "xbf16": 0, "flat": 0}
    args, kwargs = call
    saved = fm.cfg_2stages
    fm.cfg_2stages = (_EveryKey(row), {})
    try:
        return fm.get_2stage_cfgs.__wrapped__(*args, **kwargs)
    finally:
        fm.cfg_2stages = saved


def _func_name(stage):
    return getattr(stage, "func", stage).__name__


def _legacy_runs(call, kernel1, kernel2, block_m, ksplit=0):
    """`(stage 1 function name, block_m, ksplit, stage 1 options)` legacy dispatch runs
    for `call` with a one-row tuned table naming `kernel1` and `kernel2`."""
    metadata = _legacy_metadata(call, kernel1, kernel2, block_m, ksplit)
    stage1 = metadata.stage1
    options = tuple(sorted(getattr(stage1, "keywords", {}).items(), key=str))
    return _func_name(stage1), metadata.block_m, metadata.ksplit, repr(options)


_CKTILE_A16W4_ADDED = (
    "CK-Tile a16w4: the alias row the shipped rows name, which no legacy tuner "
    "lists (5.1k2)"
)


def _cktile_a16w4_added(info, call):
    """The CK-Tile candidates added in 5.1k2 for a bf16 or fp16 x MXFP4 call: the alias
    row with split-K 0 through legacy dispatch, as `(stage, alias, block_m, False)` for
    both stages at the `block_m` legacy runs, where legacy runs CK-Tile for both and
    keeps the row (not at `inter_dim % 256`, where dispatch discards it). v2's stage 1
    candidate must carry the split-K the row ran with (0), which the tags leave out."""
    from aiter import QuantType, dtypes
    from aiter.moev2 import api
    from aiter.moev2.dispatch.kernel_names import CKTILE_ALIAS
    from aiter.moev2.segments.ck_tile import CktileGemm1

    a16w4 = (
        info[11] == QuantType.per_1x32
        and info[9] in (dtypes.bf16, dtypes.fp16)
        and info[10] == dtypes.fp4x2
    )
    if not a16w4 or info[4] % 256:
        return set()
    metadata = _legacy_metadata(call, CKTILE_ALIAS, CKTILE_ALIAS, 16)
    ran = (_func_name(metadata.stage1), _func_name(metadata.stage2))
    if ran != ("cktile_moe_stage1", "cktile_moe_stage2"):
        return set()
    ksplits = {
        p["ksplit"] for p in CktileGemm1().param_space(api.problem_from_key(*call))
    }
    if ksplits - {0}:
        raise RunsDifferently(
            f"v2 lists split-K {sorted(ksplits)}, the alias row ran 0"
        )
    return {
        (stage, CKTILE_ALIAS, metadata.block_m, False) for stage in ("stage1", "stage2")
    }


_LEGACY_STAGE1_RUNS = {
    "_flydsl_stage1_wrapper": "a tuned CK-Tile stage-1 row runs FlyDSL's default",
    "asm_stage1": "a tuned CK-Tile stage-1 row runs ASM stage 1 (todo 5.17)",
}


class RunsDifferently(Exception):
    """A legacy CK-Tile name runs no stage 1 the harness knows, or v2's explicit
    candidate does not run what the legacy name runs."""


def _legacy_cktile(tuner, info, call):
    """The legacy CK-Tile candidates as legacy dispatch runs them for `call`:
    `(candidates, dropped reasons)`. Stage 1 goes through a one-row table naming the
    candidate in both stages; where CK-Tile runs, the candidate becomes the alias at
    the `block_m` legacy runs, and the alias row with the split-K of v2's candidate at
    that `block_m` must run the same. Other keys: the a16w4 additions
    (`_cktile_a16w4_added`)."""
    from aiter.moev2 import api
    from aiter.moev2.dispatch.kernel_names import CKTILE_ALIAS
    from aiter.moev2.segments.ck_tile import CktileGemm1

    if not _is_a8w4(info):
        added = _cktile_a16w4_added(info, call)
        return added, {_CKTILE_A16W4_ADDED} if added else set()
    candidates, dropped = set(), set()
    for tag, *_rest in tuner.gen_2stages_task(info, _BLOCK_MS):
        stage, name, block_m = tag[1], tag[2], tag[3]
        if stage == "stage2" or info[4] % 256:
            # Stage 2 runs as named; at `inter_dim % 256` dispatch discards the row.
            candidates.add((stage, name, block_m, False))
            continue
        ran = _legacy_runs(call, name, name, block_m)
        if ran[0] in _LEGACY_STAGE1_RUNS:
            dropped.add(_LEGACY_STAGE1_RUNS[ran[0]])
            continue
        if ran[0] != "cktile_moe_stage1":
            raise RunsDifferently(f"{name} runs {ran}")
        v2_ksplit = {
            params["block_m"]: params["ksplit"]
            for params in CktileGemm1().param_space(api.problem_from_key(*call))
        }
        ksplit = v2_ksplit.get(ran[1], 0)
        alias = _legacy_runs(call, CKTILE_ALIAS, CKTILE_ALIAS, ran[1], ksplit)
        if alias != ran:
            raise RunsDifferently(f"{name} runs {ran}, the alias {alias}")
        candidates.add((stage, CKTILE_ALIAS, ran[1], False))
    return candidates, dropped


def _legacy_opus(tuner, info, call, sorted_out):
    """`gen_opus_2stages_task`: stage 1 with token-major (`sorted_out` False) or sorted
    output, or every stage 2 (`sorted_out` False)."""
    return {
        (tag[1], tag[2], tag[3], False)
        for (tag, *_rest) in tuner.gen_opus_2stages_task(info, _BLOCK_MS)
        if (tag[1] == "stage1" and (len(tag) > 4) == sorted_out)
        or (tag[1] == "stage2" and not sorted_out)
    }


def _tags(tasks):
    return {(tag[1], tag[2], tag[3], False) for (tag, *_rest) in tasks}


def _legacy_flydsl(tuner, info, call):
    return _tags(tuner.gen_flydsl_2stages_task(info, _BLOCK_MS)) | _tags(
        tuner.gen_flydsl_i4_2stages_task(info, _BLOCK_MS)
    )


@functools.cache
def _tune_fhmoe():
    sys.path.insert(0, str(_REPO / "op_tests" / "tuners"))
    import tune_fhmoe

    return tune_fhmoe


_FHMOE_WIDENED = "FHMoE: the rules widened for every contract (5.1i)"


def _fhmoe_widened(info, call):
    """The decided widening of `tune_fhmoe.py` (5.1i), rebuilt from the registries:
    every FHMoE contract with fp8 activations interleaved, MXFP4 (at 7168/384/385/7,
    the only shape `fhmoe_` reads the dedicated table for) or MXFP8 weights, bf16
    output, Silu, without doweight or pads; stage 1 fused or unfused, stage 2 any
    `tile_n`, persist and `b_nt`, `tile_n` and stage 2 `tile_k` dividing `inter_dim`;
    a stage 1 kept where it passes `_valid_fhmoe_kernel_geometry` with a stage 2."""
    import torch

    from aiter import ActivationType, QuantType, dtypes
    from aiter.moev2.segments.flydsl_fhmoe import _valid_fhmoe_kernel_geometry
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.moe_common import GateMode

    args, kwargs = call
    if (
        info[7] != ActivationType.Silu
        or info[8] != torch.bfloat16
        or info[9] != dtypes.fp8
        or info[10] not in (dtypes.fp8, dtypes.fp4x2)
        or info[11] != QuantType.per_1x32
        or not info[12]
        or info[13]
        or args[12]
        or args[13]
        or GateMode(kwargs["gate_mode"]) != GateMode.INTERLEAVE
        or (info[10] == dtypes.fp4x2 and info[3:7] != (7168, 384, 385, 7))
    ):
        return set()
    model_dim, inter_dim = info[3], info[4]
    w = "fp8" if info[10] == dtypes.fp8 else "fp4"
    params = moe_kernels.get_flydsl_kernel_params
    stage1 = moe_kernels.get_flydsl_stage1_kernels("fp8", w, "bf16")
    stage2 = moe_kernels.get_flydsl_stage2_kernels("fp8", w, "bf16")
    out = set()
    for bm in {int(k["tile_m"]) for k in stage1.values()}:
        s2 = {}
        for name, k in stage2.items():
            if bm % int(k["tile_m"]) or inter_dim % int(k["tile_k"]):
                continue
            name = name if int(k["tile_m"]) == bm else f"{name}_sbm{bm}"
            if params(name) and _tune_fhmoe()._effective_sort_block(name) == bm:
                s2[name] = params(name)
        for base, k in stage1.items():
            if (
                int(k["tile_m"]) != bm
                or int(k["tile_k"]) != 256
                or inter_dim % int(k["tile_n"])
                or k.get("gate_mode") != "interleave"
            ):
                continue
            for name in (base, base + "_fp8"):
                if params(name) and any(
                    _valid_fhmoe_kernel_geometry(params(name), p2, model_dim=model_dim)
                    for p2 in s2.values()
                ):
                    out.add(("stage1", name, bm, False))
                    out |= {("stage2", n, bm, False) for n in s2}
    return out


def _legacy_fhmoe(tuner, info, call):
    """`tune_fhmoe.py`'s `stage1_candidates` and `stage2_candidates` at every
    `block_m` of its stage 1 registry (it keeps the row's), for the calls of its
    contract (`_matches`), with the widening (`_fhmoe_widened`): `(candidates,
    dropped reasons)`; the widening counts as a known reason where it adds any."""
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.flydsl.moe_kernels import get_flydsl_stage1_kernels

    tf = _tune_fhmoe()
    args, kwargs = call
    fields = ("gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk")
    row = {f: str(v) for f, v in zip(fields, info[:7])}
    row |= {"shared_expert_id": str(info[5] - 1), "act_type": str(info[7])}
    row |= dict(
        zip(("dtype", "q_dtype_a", "q_dtype_w", "q_type"), map(str, info[8:12]))
    )
    row |= {"use_g1u1": str(int(info[12])), "doweight_stage1": str(int(info[13]))}
    row |= {"hidden_pad": str(args[12]), "intermediate_pad": str(args[13])}
    row |= {"gate_mode": str(GateMode(kwargs["gate_mode"])), "ksplit": "0"}
    legacy = set()
    if tf._matches(row, tf.Contract(), info[2]):
        stage1 = get_flydsl_stage1_kernels("fp8", "fp8", "bf16")
        for bm in {int(k["tile_m"]) for k in stage1.values()}:
            legacy |= {("stage1", n, bm, False) for n in tf.stage1_candidates(bm)}
            legacy |= {("stage2", n, bm, False) for n in tf.stage2_candidates(bm)}
    widened = _fhmoe_widened(info, call)
    if not legacy <= widened:
        raise RunsDifferently(f"the widening drops {sorted(legacy - widened)[:2]}")
    return widened, {_FHMOE_WIDENED} if widened - legacy else set()


def synthetic_fhmoe_calls(gfx, cu_num):
    """gfx950 FHMoE calls the snapshot lacks: HY4 with fp16 output, DeepSeek V4 with
    an intermediate pad, fp8 x MXFP4 separated, MXFP4 separated and at another shape,
    MXFP8 at other shapes (`model_dim` 2304: `_valid_fhmoe_kernel_geometry` drops
    `k_wave` 2 and 4; `inter_dim` 384)."""
    if gfx != "gfx950":
        return []
    from aiter import ActivationType, QuantType, dtypes
    from aiter.moev2.problem import MoeProblem
    from moe_dispatch.keys import _variant_kwargs

    silu, fp4, fp8 = ActivationType.Silu, dtypes.fp4x2, dtypes.fp8
    cases = (
        ((6144, 256, 257, 9), dtypes.fp16, fp8, fp8, "interleave", 0),
        ((7168, 384, 385, 7), dtypes.bf16, fp8, fp4, "interleave", 128),
        ((7168, 384, 385, 7), dtypes.bf16, fp8, fp4, "base", 0),
        ((7168, 256, 257, 9), dtypes.bf16, fp4, fp4, "base", 0),
        ((7168, 256, 257, 9), dtypes.bf16, fp8, fp4, "interleave", 0),
        ((7168, 256, 257, 9), dtypes.bf16, fp8, fp8, "interleave", 0),
        ((2304, 256, 257, 9), dtypes.bf16, fp8, fp8, "interleave", 0),
        ((7168, 384, 385, 7), dtypes.bf16, fp8, fp8, "interleave", 0),
    )
    out = []
    for shape, dtype, q_a, q_w, variant, pad in cases:
        info = (gfx, cu_num, 64, *shape, silu, dtype, q_a, q_w, QuantType.per_1x32)
        info += (True, False)
        args = (64, *shape, dtype, q_a, q_w, QuantType.per_1x32, True, silu, False)
        problem = MoeProblem.from_legacy_key(
            *args, 0, pad, **_variant_kwargs(variant, silu), arch=gfx, cu_num=cu_num
        )
        out.append((info, problem.legacy_key()))
    return out


def _legacy_e2e(tuner, info, call):
    """`FmoeTuner.e2e_tune` (`--e2e_tune`): the configs of `get_tune_space()` its shape
    probe keeps for a key without doweight, less those whose timing fails
    (`run_flydsl_moe_gfx942` raises: hidden states other than bf16, an activation
    other than Silu or Swiglu, fp8 weights other than fnuz per-token or per-tensor,
    g1u0 (its gate-up check); the decode config over 256 tokens, or with `model_dim`
    not a multiple of 256 or `inter_dim` of 64 (its split-K asserts)), and
    `QuantType.No` keys with weights other than bf16 (the tuner times bf16 weights
    whatever the key's; dispatch raises on the row's weight dtype)."""
    import torch

    from aiter import ActivationType, QuantType
    from aiter.fused_moe_registry import make_fused_moe_impl_kernel_name
    from aiter.ops.flydsl.fused_moe_gfx942 import Config, _Problem, get_tune_space

    token, model_dim, inter_dim, expert, topk, act, dtype = info[2:9]
    q_w, q_type, g1u1, doweight = info[10:14]
    if doweight:
        return set()
    bf16 = q_w == torch.bfloat16 and q_type == QuantType.No
    fp8 = q_w == torch.float8_e4m3fnuz and q_type in (
        QuantType.per_Token,
        QuantType.per_Tensor,
    )
    if (
        dtype != torch.bfloat16
        or act not in (ActivationType.Silu, ActivationType.Swiglu)
        or not (bf16 or fp8)
        or not g1u1
    ):
        return set()
    probe = _Problem(
        batch=token,
        experts=expert,
        gateup_dim=inter_dim * 2,
        hidden_dim=model_dim,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
        quant_type="no" if q_type == QuantType.No else "",
    )
    out = set()
    for config_string in get_tune_space():
        config = Config.from_string(config_string)
        if config.unsupported_reason(probe) is not None:
            continue
        if not config.use_prefill and (
            not 1 <= token <= 256 or model_dim % 256 or inter_dim % 64
        ):
            continue
        name = make_fused_moe_impl_kernel_name("flydsl_gfx942", config_string)
        out.add(("impl", name, config.BLOCK_M, False))
    return out


def _legacy_flydsl_layout(tuner, info, call):
    return _tags(tuner.gen_flydsl_v2_2stages_task(info, _BLOCK_MS))


@functools.cache
def _mxmoe_tuner(tune):
    tuner = tune.Mxfp4FlydslTuner.__new__(tune.Mxfp4FlydslTuner)
    tuner.keys = []
    return tuner


def _legacy_mxmoe(tuner, info, call):
    """`Mxfp4FlydslTuner._candidate_rows` at its full search, for the a4w4 keys (it
    tunes its own table and reads no dtype of the key)."""
    from aiter import QuantType, dtypes

    if (info[9], info[10], info[11]) != (
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
    ):
        return set()
    row = dict(zip(("token", "model_dim", "inter_dim", "expert", "topk"), info[2:7]))
    row["act_type"] = str(info[7])
    tune = sys.modules[type(tuner).__module__]
    candidates = _mxmoe_tuner(tune)._candidate_rows(row, full_search=True)
    return {("stage1", c["kernelName1"], c["block_m"], False) for c in candidates} | {
        ("stage2", c["kernelName2"], c["block_m"], False) for c in candidates
    }


Family = collections.namedtuple(
    "Family",
    "legacy segments per_call tunes fhmoe",
    defaults=(False, legacy_tunes, False),
)
"""`legacy(tuner, info, call)`: the legacy candidates, or `(candidates, dropped
reasons)`; `per_call`: checked for each snapshot call of the key (the call's layout
matters), else once for the key; `tunes(info)`: whether the legacy tuner tunes `info`
(else v2 must list nothing); `fhmoe`: checked for each FHMoE call (the snapshot's and
`synthetic_fhmoe_calls`), with the shared expert in the problem."""


def _families():
    from aiter.moev2.segments import (
        asm,
        ck,
        ck_tile,
        flydsl_fhmoe,
        mxfp4,
        opus,
        whole_graph,
    )

    fhmoe = (
        flydsl_fhmoe.FlydslFhmoeGemm1(),
        flydsl_fhmoe.FlydslFhmoeGemm1Quant(),
        flydsl_fhmoe.FlydslFhmoeGemm2(),
    )
    return {
        "asm": Family(
            _legacy_asm,
            (asm.AsmStage1(), asm.AsmStage1Blockscale(), asm.AsmStage1Cktile()),
        ),
        "asm_1stage": Family(_legacy_asm_1stage, (asm.AsmFmoe1Stage(),)),
        "ck": Family(_legacy_ck, (ck.CkGemm1(), ck.CkGemm2())),
        "cktile": Family(
            _legacy_cktile,
            (ck_tile.CktileGemm1(), ck_tile.CktileGemm2()),
            per_call=True,
        ),
        "flydsl": Family(
            _legacy_flydsl,
            (mxfp4.FlydslGemm1(), mxfp4.FlydslGemm1Quant(), mxfp4.FlydslGemm2())
            + fhmoe,
        ),
        "flydslv2": Family(
            _legacy_flydsl_layout,
            (
                mxfp4.FlydslGemm1Layout(),
                mxfp4.FlydslLayoutGemm2(),
                mxfp4.FlydslLayoutGemm2Scatter(),
            ),
        ),
        "opus": Family(
            functools.partial(_legacy_opus, sorted_out=False),
            (opus.OpusGemm1(), opus.OpusGemm2()),
        ),
        "opus_sorted": Family(
            functools.partial(_legacy_opus, sorted_out=True),
            (opus.OpusGemm1Layout(),),
        ),
        "mxmoe": Family(
            _legacy_mxmoe,
            (mxfp4.MxmoeGemm1(), mxfp4.MxmoeQuantGemm1(), mxfp4.MxmoeGemm2()),
            tunes=lambda info: info[0] == "gfx950",
        ),
        "whole_graph": Family(
            _legacy_e2e,
            (whole_graph.WholeGraph(),),
            tunes=lambda info: info[0] in ("gfx942", "gfx950"),
        ),
        "fhmoe": Family(
            _legacy_fhmoe,
            fhmoe,
            tunes=lambda info: info[0] == "gfx950",
            fhmoe=True,
        ),
    }


def _stage(segment):
    from aiter.moev2.dispatch.ops import Op

    return "stage1" if segment.covers[0] <= Op.GEMM1 <= segment.covers[1] else "stage2"


def _candidate(segment, params):
    """The legacy tuner's tag for a v2 candidate: `(stage, kernel, block_m, nt)`, for a
    1-stage segment `(asm_1stage[_xbf16], kernel, block_m, flat)`."""
    from aiter.moev2.dispatch.ops import Op

    kernel, block_m = params["kernel"], params["block_m"]
    if segment.name == "impl.whole_graph":
        return "impl", kernel, block_m, False
    if segment.covers[0] <= Op.GEMM1 and segment.covers[1] >= Op.GEMM2:
        stage = "asm_1stage_xbf16" if params.get("xbf16") else "asm_1stage"
        return stage, kernel, block_m, int(params.get("flat", 0))
    return _stage(segment), kernel, block_m, bool(params.get("nt", False))


def v2_candidates(segments, problem):
    """`(candidates, twice)`: what `segments` list for `problem`, and what they list
    more than once."""
    listed = collections.Counter(
        _candidate(s, params) for s in segments for params in s.param_space(problem)
    )
    return set(listed), {c for c, n in listed.items() if n > 1}


def _activation(info):
    return str(info[7]).split(".")[-1].lower()


def _ck_activation_unmapped(info):
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_ck2stages_common import (
        ACT_OP_MAP,
    )

    return _activation(info) not in ACT_OP_MAP


Known = collections.namedtuple("Known", "reason declined", defaults=(False,))
"""A known difference; `declined`: v2's segments accept the problem and list nothing
on purpose."""


def known_difference(family, info, raised):
    """The `Known` reason legacy's candidates for `info` are not v2's (v2 lists none),
    or None."""
    import torch

    import aiter
    from aiter import QuantType, dtypes

    gfx, q_a, q_w, q_type = info[0], info[9], info[10], info[11]
    fp8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if raised is not None:
        if (
            family == "ck"
            and isinstance(raised, KeyError)
            and _ck_activation_unmapped(info)
        ):
            return Known(f"legacy raises: CK maps no {_activation(info)} (todo 5.46)")
        if (
            family == "asm_1stage"
            and isinstance(raised, TypeError)
            and gfx not in ("gfx942", "gfx950")
        ):
            return Known(f"legacy raises: no 1-stage table names on {gfx} (todo 5.52)")
        if (
            family == "mxmoe"
            and isinstance(raised, ValueError)
            and _activation(info) not in ("silu", "swiglu", "situv2")
        ):
            return Known(f"legacy raises: MXMOE has no {_activation(info)} kernel")
        return None
    if gfx == "gfx1201":
        return Known("gfx1201: no v2 path (todo 1b.2)")
    if family.startswith("flydsl") and _activation(info) not in (
        "silu",
        "swiglu",
        "situv2",
    ):
        return Known(
            f"FlyDSL {_activation(info)}: dispatch discards the tuned row (todo 5.48)",
            declined=True,
        )
    if family == "flydsl" and q_type == QuantType.per_1x32 and info[13]:
        return Known(
            "per-1x32 doweight: the FlyDSL capabilities leave it out (todo 1e.7)"
        )
    if family == "cktile" and info[4] % 256:
        return Known(
            "CK-Tile MXFP4 at inter_dim % 256: dispatch discards the stage 2 row",
            declined=True,
        )
    if family == "opus" and _activation(info) not in ("silu", "swiglu", "situv2"):
        return Known(f"Opus stage 2 for {_activation(info)}: no a8w4 stage 1 runs it")
    if family == "opus" and info[13]:
        return Known(
            "Opus stage 2 doweight: `opus.gemm2` leaves it out; its only legacy "
            "partner is FlyDSL stage 1 (todo 1e.7)"
        )
    if family == "mxmoe" and not info[12]:
        return Known(
            "MXMOE g1u0: the capabilities leave it out; the legacy tuner times g1u1 "
            "whatever the key's"
        )
    if family == "mxmoe" and info[8] != dtypes.bf16:
        return Known(
            "MXMOE fp16 output: the legacy tuner times bf16 whatever the key's dtype",
            declined=True,
        )
    if family == "mxmoe" and info[13]:
        return Known(
            "MXMOE doweight: the capabilities leave it out; the legacy tuner applies "
            "the routed weights in stage 2 whatever the key's"
        )
    if family == "mxmoe" and not aiter.is_mxfp4_moe_shape_supported(
        info[5], info[3], info[4], info[6]
    ):
        return Known(
            "MXMOE shape outside the generated auxiliary kernels: dispatch discards "
            "the tuned row (todo 5.61)",
            declined=True,
        )
    if family == "asm_1stage" and q_w == torch.uint32:
        return Known(
            "int4 weights: legacy's 1-stage filter tests torch.int4, not uint32 "
            "(todo 5.47)"
        )
    if family == "asm":
        if q_type == QuantType.per_1x32 and q_a in fp8 and q_w in fp8:
            return Known(
                "mxfp8: legacy lists the per-token fp8 ASM kernels, which take no "
                "1x32 scales"
            )
        if q_w == torch.uint32:
            return Known(
                "int4 weights: legacy's ASM filter tests torch.int4, not uint32 "
                "(todo 5.47)"
            )
    return None


def _combo(info):
    from moe_dispatch.keys import _short

    return " ".join(_short(v) for v in info[7:12]) + f" dw={int(info[13])}"


def check_arch(arch, tuner):
    """`(counts, known, problems)`; `problems` maps a kind of difference to its
    first example and the number of keys that show it."""
    import io

    from aiter.moev2 import api
    from aiter.moev2.problem import MoeProblem
    from moe_dispatch import arch_sim
    from moe_dispatch.differential import captured_logs

    problems = {}
    counts = collections.Counter()
    known = collections.Counter()
    families = _families()

    def problem_at(family, info, what, example):
        kind = f"{family} {_combo(info)}: {what}"
        first, n = problems.get(kind, (f"{info[2:7]} {example}", 0))
        problems[kind] = (first, n + 1)

    def compare(name, family, info, problem, call):
        raised, dropped = None, ()
        try:
            # The legacy generators print when a kernel table is missing.
            with contextlib.redirect_stdout(io.StringIO()):
                legacy = (
                    family.legacy(tuner, info, call) if family.tunes(info) else set()
                )
        except RunsDifferently as exc:
            problem_at(name, info, "runs differently", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            raised, legacy = exc, set()
        if isinstance(legacy, tuple):
            legacy, dropped = legacy
        known.update(dropped)
        v2, twice = v2_candidates(family.segments, problem)
        counts[f"{name} legacy"] += len(legacy)
        counts[f"{name} v2"] += len(v2)
        if twice:
            problem_at(name, info, "listed twice", sorted(twice)[:2])
        if v2 - legacy:
            problem_at(name, info, "v2-only", sorted(v2 - legacy)[:2])
        if not (legacy - v2 or raised is not None):
            return
        k = known_difference(name, info, raised)
        if k is None and raised is not None:
            problem_at(name, info, "legacy raises", repr(raised))
        elif k is None:
            problem_at(name, info, "legacy-only", sorted(legacy - v2)[:2])
        elif v2:
            problem_at(name, info, f"partly known ({k.reason})", "")
        elif (
            raised is None
            and not k.declined
            and (accepting := [s.name for s in family.segments if s.accepts(problem)])
        ):
            problem_at(name, info, f"known ({k.reason}) but accepted", accepting)
        else:
            known[k.reason] += 1

    with contextlib.ExitStack() as stack:
        gfx, cu_num = stack.enter_context(arch_sim.simulated_arch(arch))
        stack.enter_context(captured_logs())
        stack.enter_context(_legacy_pinned(gfx, cu_num))
        fhmoe_calls = []
        for info, keys in tuner_keys(gfx, cu_num).items():
            counts["keys"] += 1
            problem = MoeProblem.from_legacy_key(
                *info[2:7], *info[8:13], info[7], info[13], 0, 0,
                arch=gfx, cu_num=cu_num,
            )  # fmt: skip
            calls = [
                (key.args, key.call_kwargs())
                for key in keys
                if not key.call_kwargs().get("config_file")
            ] or _synthetic_calls(info, gfx, cu_num)
            fhmoe_calls += [
                (info, (key.args, key.call_kwargs()))
                for key in keys
                if key.section == "fhmoe"
            ]
            for name, family in families.items():
                if family.fhmoe:
                    continue
                if not family.per_call:
                    compare(name, family, info, problem, None)
                    continue
                for call in calls:
                    counts[f"{name} calls"] += 1
                    compare(name, family, info, api.problem_from_key(*call), call)
        for info, call in fhmoe_calls + synthetic_fhmoe_calls(gfx, cu_num):
            problem = api.problem_from_key(*call, fhmoe=True)
            for name, family in families.items():
                if family.fhmoe:
                    counts[f"{name} calls"] += 1
                    compare(name, family, info, problem, call)
    return counts, known, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup()
    from moe_dispatch import arch_sim

    tuner = _legacy_tuner()
    failed = False
    for arch in args.arch or list(arch_sim.ARCHS):
        counts, known, problems = check_arch(arch, tuner)
        print(
            f"{arch}: {counts['keys']} tuner keys; "
            + ", ".join(
                f"{f} {counts[f + ' legacy']}/{counts[f + ' v2']} candidates"
                for f in _families()
            )
            + f"; {len(problems)} problems"
        )
        for reason, n in sorted(known.items()):
            print(f"  known ({n} keys or calls): {reason}")
        for kind, (first, n) in sorted(problems.items()):
            print(f"  {kind} ({n} keys or calls)\n    first: {first}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
