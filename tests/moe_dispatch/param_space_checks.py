# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The v2 tuning candidates (`Segment.param_space`) against the legacy tuner's.

For every tuner key the snapshot's `get_2stage_cfgs` keys give, on every simulated
arch, and per kernel family, the set of `(stage, kernel, block_m, nt)` the legacy
`FmoeTuner` generator lists equals the set v2's segments list, except for the known
differences below, each with its reason; a known difference holds only where v2 lists
nothing and, unless legacy raises (where v2's guard is the fix, todo 5.46), no segment
of the family accepts the problem. A tuner key is the 14 fields the legacy
tuner tunes on, after its own filters (`tune()`): per-1x32 only on gfx950,
per-128x128 tuned as per-1x128; it tunes no g1u0 key, so v2 must list nothing there.

    python tests/moe_dispatch/param_space_checks.py --check [--arch gfx950]
"""

import argparse
import collections
import contextlib
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
    from aiter import QuantType
    from moe_dispatch import snapshot

    out = {}
    for key in snapshot.dispatch_keys(gfx, cu_num):
        token, model_dim, inter_dim, expert, topk, dtype, q_a, q_w, q_type, g1u1 = (
            key.args[:10]
        )
        activation, doweight = key.args[10], key.args[11]
        if q_type == QuantType.per_1x32 and gfx != "gfx950":
            continue
        if q_type == QuantType.per_128x128:
            q_type = QuantType.per_1x128
        info = (gfx, cu_num, token, model_dim, inter_dim, expert, topk, activation)
        info += (dtype, q_a, q_w, q_type, g1u1, doweight)
        out.setdefault(info, None)
    return list(out)


def _is_a8w4(info):
    import torch

    from aiter import QuantType, dtypes

    q_a, q_w, q_type = info[9], info[10], info[11]
    return (
        q_type == QuantType.per_1x32
        and q_w == dtypes.fp4x2
        and q_a in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    )


def _legacy_asm(tuner, info):
    return {
        ("stage1", tag[2], tag[3], False)
        for (tag, *_rest) in tuner.gen_2stages_asm1_task(info, _BLOCK_MS)
    }


def _legacy_ck(tuner, info):
    if _is_a8w4(info):
        return set()  # the CK-Tile generator (5.1c)
    return {
        (tag[1], tag[2], tag[3], bool(tag[6]))
        for (tag, *_rest) in tuner.gen_2stages_task(info, _BLOCK_MS)
    }


def _families():
    from aiter.moev2.segments import asm, ck

    return {
        "asm": (
            _legacy_asm,
            (asm.AsmStage1(), asm.AsmStage1Blockscale(), asm.AsmStage1Cktile()),
        ),
        "ck": (_legacy_ck, (ck.CkGemm1(), ck.CkGemm2())),
    }


def _stage(segment):
    from aiter.moev2.dispatch.ops import Op

    return "stage1" if Op.GEMM1 in segment.covers else "stage2"


def v2_candidates(segments, problem):
    return {
        (_stage(s), params["kernel"], params["block_m"], bool(params.get("nt", False)))
        for s in segments
        for params in s.param_space(problem)
    }


def _activation(info):
    return str(info[7]).split(".")[-1].lower()


def _ck_activation_unmapped(info):
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_ck2stages_common import (
        ACT_OP_MAP,
    )

    return _activation(info) not in ACT_OP_MAP


def known_difference(family, info, raised):
    """The reason legacy's candidates for `info` are not v2's (v2 lists none), or
    None."""
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
            return f"legacy raises: CK maps no {_activation(info)} (todo 5.46)"
        return None
    if gfx == "gfx1201":
        return "gfx1201: no v2 path (todo 1b.2)"
    if q_type != QuantType.per_1x32 and _activation(info) not in ("silu", "gelu"):
        return "Swiglu/GeluTanh outside per-1x32: v2 claims Silu and Gelu (todo 1c.4)"
    if family == "asm":
        if q_type == QuantType.per_1x32 and q_a in fp8 and q_w in fp8:
            return (
                "mxfp8: legacy lists the per-token fp8 ASM kernels, which take no "
                "1x32 scales"
            )
        if q_w == torch.uint32:
            return (
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

    with contextlib.ExitStack() as stack:
        gfx, cu_num = stack.enter_context(arch_sim.simulated_arch(arch))
        stack.enter_context(captured_logs())
        stack.enter_context(_legacy_pinned(gfx, cu_num))
        for info in tuner_keys(gfx, cu_num):
            counts["keys"] += 1
            problem = MoeProblem.from_legacy_key(
                *info[2:7], *info[8:13], info[7], info[13], 0, 0,
                arch=gfx, cu_num=cu_num,
            )  # fmt: skip
            for family, (legacy_fn, segments) in families.items():
                raised = None
                try:
                    # The legacy generators print when a kernel table is missing.
                    with contextlib.redirect_stdout(io.StringIO()):
                        legacy = legacy_fn(tuner, info) if info[12] else set()
                except Exception as exc:  # noqa: BLE001
                    raised, legacy = exc, set()
                v2 = v2_candidates(segments, problem)
                counts[f"{family} legacy"] += len(legacy)
                counts[f"{family} v2"] += len(v2)
                if v2 - legacy:
                    problem_at(family, info, "v2-only", sorted(v2 - legacy)[:2])
                if not (legacy - v2 or raised is not None):
                    continue
                reason = known_difference(family, info, raised)
                if reason is None and raised is not None:
                    problem_at(family, info, "legacy raises", repr(raised))
                elif reason is None:
                    problem_at(family, info, "legacy-only", sorted(legacy - v2)[:2])
                elif v2:
                    problem_at(family, info, f"partly known ({reason})", "")
                elif raised is None and (
                    accepting := [s.name for s in segments if s.accepts(problem)]
                ):
                    problem_at(
                        family, info, f"known ({reason}) but accepted", accepting
                    )
                else:
                    known[reason] += 1
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
            print(f"  known ({n} keys): {reason}")
        for kind, (first, n) in sorted(problems.items()):
            print(f"  {kind} ({n} keys)\n    first: {first}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
