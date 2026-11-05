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
snapshot lacks.

CK-Tile (`cktile`) is checked per snapshot call, since the call's weight layout picks
the branch: no legacy branch reads the tuner's `cktile_a8w4_bm*` stage 1 names, so each
goes through legacy dispatch itself (a one-row tuned table swapped into `fused_moe`,
`get_2stage_cfgs` read back). Where CK-Tile runs, it stands for the alias at the
`block_m` legacy ran (the alias row must run the same); where something else runs, it
is a known difference v2 declines on purpose. Opus stage 1 candidates are listed once
per legacy task: token-major output in `opus` (with stage 2), sorted output in
`opus_sorted`. A 1-stage candidate (`asm_1stage`) is `(asm_1stage[_xbf16], kernel,
block_m, flat)`.

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
    model_dim 2880, a8w4 Swiglu at every CK-Tile token tier."""
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
    )
    tokens = [(shape, token) for token in (16, 512) for shape in shapes]
    tokens += [
        ((dtype, fp8, ActivationType.Swiglu, 7168, 256), token)
        for dtype in (dtypes.bf16, dtypes.fp16)
        for token in (4096, 20000)
    ]
    return [
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


def _legacy_asm(tuner, info, call):
    return {
        ("stage1", tag[2], tag[3], False)
        for (tag, *_rest) in tuner.gen_2stages_asm1_task(info, _BLOCK_MS)
    }


def _legacy_asm_1stage(tuner, info, call):
    """`gen_1stage_asm_task`: `(asm_1stage[_xbf16], kernel, block_m, flat)`."""
    return {
        (tag[1], tag[2], tag[3], int(tag[4]))
        for (tag, *_rest) in tuner.gen_1stage_asm_task(info)
    }


def _legacy_ck(tuner, info, call):
    if _is_a8w4(info):
        return set()  # the CK-Tile generator: family `cktile`
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


def _legacy_runs(call, kernel1, kernel2, block_m, ksplit=0):
    """`(stage 1 function name, block_m, ksplit, stage 1 options)` legacy dispatch runs
    for `call` with a one-row tuned table naming `kernel1` and `kernel2`."""
    import aiter.fused_moe as fm

    row = {"block_m": block_m, "ksplit": ksplit, "kernelName1": kernel1}
    row |= {"kernelName2": kernel2, "run_1stage": 0, "xbf16": 0, "flat": 0}
    args, kwargs = call
    saved = fm.cfg_2stages
    fm.cfg_2stages = (_EveryKey(row), {})
    try:
        metadata = fm.get_2stage_cfgs.__wrapped__(*args, **kwargs)
    finally:
        fm.cfg_2stages = saved
    stage1 = metadata.stage1
    func = getattr(stage1, "func", stage1)
    options = tuple(sorted(getattr(stage1, "keywords", {}).items(), key=str))
    return func.__name__, metadata.block_m, metadata.ksplit, repr(options)


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
    that `block_m` must run the same."""
    from aiter.moev2 import api
    from aiter.moev2.dispatch.kernel_names import CKTILE_ALIAS
    from aiter.moev2.segments.ck_tile import CktileGemm1

    if not _is_a8w4(info):
        return set()
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


def _legacy_flydsl_layout(tuner, info, call):
    return _tags(tuner.gen_flydsl_v2_2stages_task(info, _BLOCK_MS))


Family = collections.namedtuple("Family", "legacy segments per_call", defaults=(False,))
"""`legacy(tuner, info, call)`: the legacy candidates, or `(candidates, dropped
reasons)`; `per_call`: checked for each snapshot call of the key (the call's layout
matters), else once for the key."""


def _families():
    from aiter.moev2.segments import asm, ck, ck_tile, flydsl_fhmoe, mxfp4, opus

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
    }


def _stage(segment):
    from aiter.moev2.dispatch.ops import Op

    return "stage1" if Op.GEMM1 in segment.covers else "stage2"


def _candidate(segment, params):
    """The legacy tuner's tag for a v2 candidate: `(stage, kernel, block_m, nt)`, for a
    1-stage segment `(asm_1stage[_xbf16], kernel, block_m, flat)`."""
    from aiter.moev2.dispatch.ops import Op

    kernel, block_m = params["kernel"], params["block_m"]
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

    from aiter import QuantType

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
        return None
    if gfx == "gfx1201":
        return Known("gfx1201: no v2 path (todo 1b.2)")
    if q_type != QuantType.per_1x32 and _activation(info) not in ("silu", "gelu"):
        return Known(
            "Swiglu/GeluTanh outside per-1x32: v2 claims Silu and Gelu (todo 1c.4)"
        )
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
                    family.legacy(tuner, info, call) if legacy_tunes(info) else set()
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
            for name, family in families.items():
                if not family.per_call:
                    compare(name, family, info, problem, None)
                    continue
                for call in calls:
                    counts[f"{name} calls"] += 1
                    compare(name, family, info, api.problem_from_key(*call), call)
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
