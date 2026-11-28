# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Differential test of aiter.moev2 against the legacy MoE code.

CPU (`--check`): for every key of the legacy snapshot that a registered v2 path claims,
the v2 plan (in the snapshot record format) must equal the legacy golden record, and the
log lines v2 dispatch emits must equal the ones `get_2stage_cfgs` emits for the key. There
is no allowlist for claimed keys: the revamp keeps today's behavior, and a bug fix lands
in the legacy code first, with the snapshot regenerated. A key a path takes and declines
although legacy runs it must be listed in `known_declines.py`.

    python tests/moe_dispatch/differential.py --check [--arch gfx950]
    python tests/moe_dispatch/differential.py --check --env AITER_FLYDSL_FORCE=0

`--env NAME=VALUE` sets a dispatch-affecting variable for legacy and v2 alike; the legacy
records are then generated live rather than read from the golden files (which hold the
default environment).

GPU: `run_differential(path_name, call)` runs one call through AITER_MOEV2=off and
AITER_MOEV2=<path> and compares the kernel launches and outputs (see
tests/moe_dispatch/gpu_differential.py).

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import sys
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent


@contextlib.contextmanager
def captured_logs():
    """Collect `(level, message)` of the aiter logger at INFO, printing nothing.

    Every record is kept except one: the AITER_MOEV2 selection line that
    `aiter.moev2.paths` logs once per selection (after start-up or a reload), e.g.
    "AITER_MOEV2='a8w8_pt': a8w8_pt on aiter.moev2, every other call on the legacy
    code ...". `paths` tags that record with the attribute `SELECTION_RECORD`.

    The differential compares the lines of a legacy run (AITER_MOEV2=off) with those of
    a v2 run (AITER_MOEV2=<path>). The selection line names the AITER_MOEV2 value, so it
    can never match between the two, and it says nothing about how a call was
    dispatched. Compared, it fails every case. Dispatch lines ("using 2stage ...",
    "run_1stage = ...", warnings) are still compared exactly. Lines logged under
    AITER_MOE_EXPLAIN=1 are not tagged and are kept; the harness does not set it.
    """
    import logging

    from aiter.moev2.paths import SELECTION_RECORD

    logger = logging.getLogger("aiter")
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            # The selection line, not a dispatch line; see the docstring.
            if not getattr(record, SELECTION_RECORD, False):
                records.append((record.levelno, record.getMessage()))

    handler = _Collect(logging.DEBUG)
    saved = logger.handlers[:], logger.level, logger.propagate
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield records
    finally:
        logger.handlers[:], level, logger.propagate = saved[0], saved[1], saved[2]
        logger.setLevel(level)


def _legacy_logs(key):
    import aiter.fused_moe as fm

    with captured_logs() as logs:
        try:
            fm.get_2stage_cfgs.__wrapped__(*key.args, **key.call_kwargs())
        except Exception:  # noqa: BLE001, S110
            pass
    return logs


# The stage-1 functions of the legacy 2-stage branches a ranking region hands back.
_CKTILE = "fused_moe.cktile_moe_stage1"
_LEGACY_2STAGE = {
    "CK-Tile split-K MXFP4 or CK 2-stage SiTUv2": (_CKTILE, "fused_moe.ck_moe_stage1"),
    "CK 2-stage, padding ignored": ("fused_moe.ck_moe_stage1",),
    "CK 2-stage a16w4": ("fused_moe.ck_moe_stage1",),
    "CK 2-stage g1u0": ("fused_moe.ck_moe_stage1",),
    "ASM stage 1 + CK stage 2 g1u0": ("fused_moe.asm_stage1",),
    # A bypassed tuned row turns the MXMOE fallback off (AITER_BYPASS_TUNE_CONFIG).
    "the next branch (a tuned row is bypassed)": (
        "fused_moe._flydsl_stage1_wrapper",
        _CKTILE,
        "fused_moe.ck_moe_stage1",
    ),
}


def v2_decline(path, key, fhmoe=False):
    """`(problem, reason)` when `path` takes `key` (as `fhmoe_` dispatches it if
    `fhmoe`), else None; `reason` is why v2 declines it (the resolver, or the second
    lookup after the sort as `dispatch_record` checks it), None when it runs it."""
    from aiter.moev2 import api
    from aiter.moev2.dispatch import resolver

    if path.takes is None:
        return None
    try:
        problem = api.problem_from_key(key.args, key.call_kwargs(), fhmoe=fhmoe)
    except Exception:  # noqa: BLE001 - key_checks.py reports these
        return None
    if not path.takes(problem):
        return None
    reason = resolver.resolve(problem, path.segments).declined
    # `dispatch_record` is bound to the path's `LocalDispatch`.
    relookup = getattr(
        getattr(path.dispatch_record, "__self__", None), "relookup_decline", None
    )
    if reason is None and relookup is not None:
        reason = relookup(problem, problem.shape.m)
    return problem, reason


def legacy_decline(path, key, declined=None):
    """`(reason, stage-1 functions)` when `path` hands `key` back to a named legacy
    branch, else None; the functions are None for a 1-stage branch.

    A 1-stage tuned row the path does not run and a ranking region marked `legacy=`
    name the branch. Under AITER_BYPASS_TUNE_CONFIG legacy runs no tuned row, so such a
    row names none. `declined`: `v2_decline(path, key)`, if already known.
    """
    from aiter.moev2.dispatch import overrides

    declined = declined or v2_decline(path, key)
    reason = None if declined is None else declined[1]
    if reason is None:
        return None
    if reason.startswith(("1-stage row with ", "no segment runs 1-stage ")):
        return None if overrides.bypass_tuned() else (reason, None)
    if reason.startswith("ranking ") and reason.endswith(" is not ported"):
        legacy = reason.split(": ", 1)[1].removesuffix(" is not ported")
        if legacy.startswith("1-stage "):
            return reason, None
        # An unlisted branch matches no stage 1: compare() reports it.
        return reason, _LEGACY_2STAGE.get(legacy, ())
    return None


def legacy_fhmoe(key):
    """`(record, logs)` of `key` as `fhmoe_` dispatches it: `get_2stage_cfgs`, then
    `_use_fhmoe_wrappers` (an error record where either raises)."""
    import aiter.fused_moe as fm
    from aiter.fhmoe import _use_fhmoe_wrappers
    from moe_dispatch import record

    with captured_logs() as logs:
        try:
            metadata = fm.get_2stage_cfgs.__wrapped__(*key.args, **key.call_kwargs())
            got = record.normalize(_use_fhmoe_wrappers(metadata))
        except Exception as exc:  # noqa: BLE001
            got = record.error_record(exc)
    return got, logs


def compare_fhmoe(path, key, check_logs=True):
    """`(outcome, problem)` of a path of the `fhmoe` entry for `key` as `fhmoe_`
    dispatches it (the shared expert added): outcome None when the path does not take
    it, "claimed" when its plan must equal the legacy one, "raises" when it declines a
    key legacy raises on, "declined" when it hands back a key legacy runs."""
    from aiter.moev2 import api
    from moe_dispatch import record

    try:
        problem = api.problem_from_key(key.args, key.call_kwargs(), fhmoe=True)
    except Exception:  # noqa: BLE001 - key_checks.py reports these
        return None, None
    if not path.takes(problem):
        return None, None
    with captured_logs() as v2_logs:
        try:
            plan = path.dispatch_record(key.args, key.call_kwargs())
        except Exception as exc:  # noqa: BLE001
            plan = record.error_record(exc)
    expected, legacy_logs = legacy_fhmoe(key)
    if plan is None:
        return ("raises" if "error" in expected else "declined"), None
    label = f"{key.label()}|fhmoe"
    if check_logs and v2_logs != legacy_logs:
        return "claimed", (
            f"{label} [{path.name}] log lines differ\n"
            f"    legacy: {legacy_logs}\n"
            f"    v2:     {v2_logs}"
        )
    got = plan if isinstance(plan, dict) else record.normalize(plan)
    if got != expected:
        return "claimed", (
            f"{label} [{path.name}]\n"
            f"    legacy: {json.dumps(expected, sort_keys=True)}\n"
            f"    v2:     {json.dumps(got, sort_keys=True)}"
        )
    return "claimed", None


OUTCOMES = {
    "claimed": "claimed by v2",
    "one_stage": "handed back as legacy 1-stage",
    "two_stage": "as a legacy 2-stage branch",
    "raises": "declined where legacy raises",
    "declined": "otherwise declined where legacy runs",
}
# The outcomes of keys legacy runs and v2 declines: each must be in known_declines.py.
_LEGACY_RUNS = ("one_stage", "two_stage", "declined")


def compare(arch, check_logs=True, live=False):
    """`(outcomes, problems, known)`: `{outcome: {label: path name}}` for each of
    `OUTCOMES`, a list of strings, and the count of declines per `known_declines.py`
    entry. `live`: generate the legacy records instead of reading golden/.

    `one_stage`: keys a path hands back as legacy 1-stage keys, which must be 1-stage
    in the legacy record. `two_stage`: keys a ranking region hands back to a named legacy
    2-stage branch, whose stage 1 the legacy record must run. Paths of the `fhmoe` entry
    see every key with the shared expert added (`compare_fhmoe`; labels end in `|fhmoe`,
    the legacy records are generated). Every decline of a key legacy runs must be in
    `known_declines.py`.
    """
    from aiter.moev2 import paths
    from aiter.moev2.dispatch import tuned
    from moe_dispatch import arch_sim, known_declines, record, snapshot

    if live:
        import logging

        import aiter.fused_moe as fm

        logger = logging.getLogger("aiter")
        level = logger.level
        golden = snapshot.generate(arch)
        logger.setLevel(level)
        fm.get_2stage_cfgs.cache_clear()
    else:
        golden = snapshot.read(arch)
    checkers = [p for p in paths.registered().values() if p.dispatch_record is not None]
    fhmoe_paths = [p for p in checkers if p.entry == "fhmoe"]
    checkers = [p for p in checkers if p.entry != "fhmoe"]
    outcomes = {name: {} for name in OUTCOMES}
    claimed = outcomes["claimed"]
    one_stage, two_stage = outcomes["one_stage"], outcomes["two_stage"]
    problems = []
    known = Counter()
    seen = set()

    def check_known(label, path, declined):
        problem, reason = declined
        if reason is None:
            problems.append(
                f"{label} [{path.name}] declined without a reason where legacy runs "
                "it; known_declines.py cannot list it"
            )
            return
        entry = known_declines.find(path.name, problem, reason)
        if entry is None:
            problems.append(
                f"{label} [{path.name}] declined ({reason}) where legacy runs it, and "
                "known_declines.py does not list it"
            )
        else:
            known[entry.name] += 1

    with arch_sim.simulated_arch(arch) as (gfx, cu_num):
        keys = snapshot.dispatch_keys(gfx, cu_num)
        # Load the tuned table once up front so no key's logs include the load warnings.
        with captured_logs():
            if keys:
                _legacy_logs(keys[0])
            tuned._default_table()
        for key in keys:
            label = key.label()
            if label in seen:  # a row repeated in the tuned table
                continue
            seen.add(label)
            for path in checkers:
                with captured_logs() as v2_logs:
                    try:
                        plan = path.dispatch_record(key.args, key.call_kwargs())
                    except Exception as exc:  # noqa: BLE001
                        plan = record.error_record(exc)
                if plan is None:
                    with captured_logs():
                        declined = v2_decline(path, key)
                        handed_back = legacy_decline(path, key, declined)
                    if declined is None:
                        continue
                    expected = snapshot.resolve_same_as_base(golden, label)
                    if handed_back is None:
                        if "error" in expected:
                            outcomes["raises"][label] = path.name
                        else:
                            outcomes["declined"][label] = path.name
                            check_known(label, path, declined)
                        continue
                    reason, stage1 = handed_back
                    if stage1 is None:
                        one_stage[label] = path.name
                        if not expected.get("run_1stage"):
                            problems.append(
                                f"{label} [{path.name}] declined ({reason}) but legacy "
                                f"runs it 2-stage: {json.dumps(expected, sort_keys=True)}"
                            )
                            continue
                    else:
                        two_stage[label] = path.name
                        got = expected.get("stage1")
                        got = got.get("fn") if isinstance(got, dict) else got
                        if expected.get("run_1stage") or got not in stage1:
                            problems.append(
                                f"{label} [{path.name}] declined ({reason}) but legacy "
                                f"does not run {stage1}: "
                                f"{json.dumps(expected, sort_keys=True)}"
                            )
                            continue
                    check_known(label, path, declined)
                    continue
                if check_logs:
                    legacy_logs = _legacy_logs(key)
                    if v2_logs != legacy_logs:
                        problems.append(
                            f"{label} [{path.name}] log lines differ\n"
                            f"    legacy: {legacy_logs}\n"
                            f"    v2:     {v2_logs}"
                        )
                if label in claimed:
                    problems.append(
                        f"{label}: claimed by {claimed[label]} and {path.name}"
                    )
                    continue
                claimed[label] = path.name
                got = plan if isinstance(plan, dict) else record.normalize(plan)
                expected = snapshot.resolve_same_as_base(golden, label)
                if got != expected:
                    problems.append(
                        f"{label} [{path.name}]\n"
                        f"    legacy: {json.dumps(expected, sort_keys=True)}\n"
                        f"    v2:     {json.dumps(got, sort_keys=True)}"
                    )
            fhmoe_label = f"{label}|fhmoe"
            for path in fhmoe_paths:
                if any(fhmoe_label in by_label for by_label in outcomes.values()):
                    break
                outcome, problem = compare_fhmoe(path, key, check_logs)
                if outcome is not None:
                    outcomes[outcome][fhmoe_label] = path.name
                if problem is not None:
                    problems.append(problem)
                if outcome == "declined":
                    with captured_logs():
                        declined = v2_decline(path, key, fhmoe=True)
                    check_known(fhmoe_label, path, declined)
    return outcomes, problems, known


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    parser.add_argument(
        "--no-logs", action="store_true", help="skip the log comparison"
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set for legacy and v2; compares with live legacy records",
    )
    args = parser.parse_args(argv)

    # `tests` so `moe_dispatch` imports, and the repository root so `aiter` is
    # this checkout rather than an installed copy, even one earlier on the path.
    front = [str(_HERE.parent), str(_HERE.parents[1])]
    sys.path[:] = front + [p for p in sys.path if p not in front]
    from moe_dispatch import snapshot

    snapshot._setup()
    # After _setup, which drops the dispatch-affecting variables of the caller.
    for item in args.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
    from moe_dispatch import arch_sim, known_declines

    failed = False
    unmatched = {entry.name for entry in known_declines.KNOWN}
    for arch in args.arch or list(arch_sim.ARCHS):
        outcomes, problems, known = compare(
            arch, check_logs=not args.no_logs, live=bool(args.env)
        )
        unmatched -= set(known)

        def summary(by_label):
            counts = Counter(by_label.values())
            return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"

        counts = ", ".join(
            f"{len(outcomes[name])} {what} ({summary(outcomes[name])})"
            for name, what in OUTCOMES.items()
        )
        print(f"{arch}: {counts}, {len(problems)} problems")
        runs = sum(len(outcomes[name]) for name in _LEGACY_RUNS)
        print(
            f"  {runs} declined where legacy runs, {sum(known.values())} of them known:"
        )
        for name, n in sorted(known.items()):
            print(f"    {n:6d} {name}")
        for line in problems[:40]:
            print("  " + line)
        failed |= bool(problems)
    if not args.arch:
        # Not a failure: some entries hold only under an environment (e.g.
        # AITER_BYPASS_TUNE_CONFIG), but a stale entry shows up here.
        for name in sorted(unmatched):
            print(f"known entry matching no decline here: {name}")
    return 1 if failed else 0


@contextlib.contextmanager
def moev2_mode(value):
    """Run with AITER_MOEV2=value, restoring the previous setting afterwards."""
    from aiter.moev2 import envs

    old = os.environ.get(envs.AITER_MOEV2)
    os.environ[envs.AITER_MOEV2] = value
    envs.reload()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(envs.AITER_MOEV2, None)
        else:
            os.environ[envs.AITER_MOEV2] = old
        envs.reload()


def _bits(t):
    """`t` with floating-point values as integers of the same width: equal NaNs compare
    equal, and -0.0 differs from 0.0."""
    import torch

    if t.is_floating_point():
        width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        return t.view(width[t.element_size()])
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _expand(launch):
    """`{where: value}` of a recorded launch, dataclass arguments by field."""
    pairs = [(i, x) for i, x in enumerate(launch.args)]
    pairs += list(launch.keywords.items())
    expanded = []
    for where, x in pairs:
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            expanded.extend(
                (f"{where}.{field.name}", getattr(x, field.name))
                for field in dataclasses.fields(x)
            )
        else:
            expanded.append((where, x))
    return dict(expanded)


def _float_gap(x, y, noise, z=None):
    """Why floating-point `x` (legacy) and `y` (v2) are not close, or None.

    Close: where either value is non-finite, `y` has the bits of `x` or of `z` (legacy's
    second run, if any); elsewhere within 1 ulp of their own dtype at their magnitude (a
    rounding flip) or within twice `noise` (how far legacy's two runs differ on the same
    argument).
    """
    import torch

    xf, yf = x.float(), y.float()
    finite = xf.isfinite() & yf.isfinite()
    odd = ~finite
    if odd.any():
        match = _bits(x)[odd] == _bits(y)[odd]
        if z is not None:
            match |= _bits(z)[odd] == _bits(y)[odd]
        if not bool(match.all()):
            return "non-finite values differ"
    if not finite.any():
        return None
    gap = (xf - yf).abs()[finite]
    info = torch.finfo(x.dtype)
    peak = torch.maximum(xf.abs(), yf.abs())[finite].clamp_min(info.tiny)
    mantissa = {
        torch.float32: 23,
        torch.float16: 10,
        torch.bfloat16: 7,
        torch.float8_e5m2: 2,
    }.get(x.dtype, 3)
    ulp = torch.exp2(peak.log2().floor() - mantissa)
    allowed = torch.maximum(ulp, torch.full_like(ulp, 2 * noise))
    if bool((gap <= allowed).all()):
        return None
    worst = int((gap - allowed).argmax())
    return (
        f"differ beyond 1 ulp and twice legacy's own spread ({noise:.3g}): "
        f"{int((gap > allowed).sum())} of {gap.numel()} values, worst "
        f"{gap[worst].item():.3g} at magnitude {peak[worst].item():.3g}"
    )


def _sorted_scale_defined(rows, groups, padded_groups, native=False):
    """Flat indices of the bytes the sorted MX quant kernels write in their swizzled
    e8m0 scale (`mx_scale_shuffle_idx` in csrc/include/mx_quant_utils.h): the sorted
    positions `rows` (routed rows, not block padding, in the order given), the `groups`
    columns (not the pad to 8). `native`: the FlyDSL a4w4 stage 1 `native_scale_layout`
    (`store_scales` in aiter/ops/flydsl/kernels/mxfp4_gemm1.py), one chunk per 16 rows
    instead of 32, each scale written to both bytes of its 16-bit half."""
    import torch

    x = rows[:, None]
    y = torch.arange(groups, device=rows.device)[None, :]
    chunk = 16 if native else 32
    idx = (
        (x // chunk * padded_groups) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
    )
    if native:
        return torch.stack([idx, idx + 1], -1).flatten()
    return (idx + (x % 32) // 16).flatten()


def _sort_view(stage1, at, block_m, token_num):
    """One run's sort, read so that sorted-order buffers compare without depending on
    the order of the rows inside an expert.

    A sort lays out one segment per expert, in expert order, each padded to `block_m`
    rows; the expert's routed rows fill the start of its segment and the rest is
    padding. That layout (`sorted_expert_ids`, `num_valid_ids`, which positions are
    padding, the expert at each position) is fixed by the routing and compared exactly.
    The order of the routed rows inside a segment is not: the adaptive aux sort of
    a4w4 / a8w4 claims positions with atomics, and two legacy runs order them
    differently about half the time. So each run's routed rows are read in `order`, its
    own positions sorted by the row's (token, slot) id (`token | slot << 24`, unique per
    routed row): the same row of each run lines up, whatever position its sort gave it.
    """
    num_valid = int(stage1.args[at["num_valid"]][0])
    ids = stage1.args[at["ids"]][:num_valid]
    blocks = stage1.args[at["blocks"]][: num_valid // block_m]
    routed = (ids & 0xFFFFFF) < token_num
    positions = routed.nonzero().flatten()
    return {
        "num_valid": num_valid,
        "ids": ids,
        "routed": routed,
        "order": positions[ids[positions].argsort()],
        "padding": (~routed).nonzero().flatten(),
        "experts": blocks.repeat_interleave(block_m)[:num_valid],
    }


def _unrepeatable_stage1(launch):
    """Why legacy does not repeat this stage-1 launch on the same inputs, or None.

    The kernels listed here give a different output from run to run (todo 5.128).
    `compare_launches` then compares stage 1's integer output and the integer arguments
    of the launches after it (bitwise otherwise; not the sort buffers or stage 1's
    inputs) like the floating-point ones: within twice what legacy's two runs differ,
    and at least one step. The final output is not relaxed (`_check_output`): a case
    where legacy's output does not repeat still fails. A failure of such a case is
    marked (`may_be_legacy_noise`) so the GPU differential may rerun it."""
    from aiter import QuantType
    from aiter.moev2.segments.asm import asm_stage1
    from aiter.moev2.segments.flydsl import _flydsl_stage1_wrapper

    if (
        launch.func is asm_stage1
        and launch.keywords.get("quant_type") != QuantType.per_1x128
    ):
        # gfx950, int8 per-tensor Gelu (32, 4096, 384, E128 top8), legacy alone: stage 1
        # differs by up to 2.1 between clean runs (the int8 GEMM2 input by up to 29
        # steps), and some runs write garbage (1e32..2e38) into a few values.
        return "ASM stage 1 fmoe_stage1_bf16_pertoken{Int8,Fp8}_g1u1_*"
    if launch.func is _flydsl_stage1_wrapper and str(
        launch.keywords.get("kernelName", "")
    ).startswith("flydsl_moe1_abf16_wint4"):
        # gfx950, fp16 (40, 2048, 768, E128 top8), legacy alone: the output differs by
        # up to 1.6 at a peak of 1.9 on every run; that case fails on the output.
        return "FlyDSL stage 1 flydsl_moe1_abf16_wint4_*"
    return None


def _int_gap(x, y, noise):
    """Why integer `x` (legacy) and `y` (v2) are not close, or None: within twice
    `noise` (how far legacy's two runs differ on the same argument), at least 1."""
    gap = (x.int() - y.int()).abs()
    allowed = max(1, 2 * noise)
    if bool((gap <= allowed).all()):
        return None
    return (
        f"differ beyond twice legacy's own spread ({noise}, at least 1): "
        f"{int((gap > allowed).sum())} of {gap.numel()} values, worst {gap.max().item()}"
    )


def compare_launches(legacy, v2, call, out_legacy, out_v2, legacy2=None):
    """`{(stage, argument): problem}` between two `kernel_bench_callable` recordings.

    Same stages, same functions, equal non-tensor arguments, and tensors over what the
    kernels define. Sorted order is compared independently of the order of the rows
    inside an expert (`_sort_view`): the sort layout exactly (`sorted_expert_ids`,
    `num_valid_ids`, the padding positions, the expert of each row), and every buffer in
    sorted order row by row, each run's rows taken in (token, slot) order: the sort
    buffers (`sorted_ids`, `sorted_weights`, `m_indices`: routed rows, then padding;
    the rest past `num_valid_ids` is `torch.empty`), `reverse_sorted` as the id at each
    position it gives, MX scales at the routed rows and the `K / 32` groups
    (`_sorted_scale_defined`), and other sorted buffers at the routed rows (stage 1
    skips block padding). `[token, topk, inter]` buffers are compared at the routed
    slots that are local (stage 1 skips the others), the fp8 rows and fp32 scales of
    the ASM 1x128 stage 1 buffer each in its own dtype (`_split_1x128_out`), and nothing
    for the output buffer, for the `out` argument of a stage 1 that writes a buffer of
    its own (`_own_stage1_buffer`), nor for the output of a split-K stage 1 (accumulated
    with atomics). The sort buffers, MX scales, fp4 values and every integer tensor must
    be bitwise equal, but for the integer tensors after a stage 1 legacy does not repeat
    (`_unrepeatable_stage1`), compared like floating-point ones. Floating-point tensors
    (activations, intermediates and their scales) may differ by 1 ulp of their dtype,
    or by twice what the same argument differs between `legacy` and `legacy2`, a second
    legacy run: some stage-1 kernels are not repeatable. `legacy2` counts only where its
    sort layout equals `legacy`'s. A non-finite value must equal one of legacy's runs.
    The caller's own tensors (weights and their scales) compare whole. A 1-stage ASM
    launch writes the output to its `moe_buf`, and the FLAT one takes the topk ids and
    weights unsorted, which are compared as inputs.
    """
    import torch

    from aiter import dtypes
    from aiter.moev2.segments.asm import fused_moe_1stage
    from aiter.moev2.shapes import get_inter_dim

    if [n for n, _ in legacy] != [n for n, _ in v2]:
        return {
            ("", ""): f"stages differ: {[n for n, _ in legacy]} vs {[n for n, _ in v2]}"
        }
    if legacy2 is not None and [n for n, _ in legacy2] != [n for n, _ in legacy]:
        legacy2 = None
    roles = {}
    block_m = num_valid = sorted_len = model_dim = inter_dim = relaxed = None
    native = False
    views = {}  # "legacy" / "v2" / "legacy2": `_sort_view` of that run
    # The caller's tensors (weights and their scales, ...) compare whole, whatever
    # their shape.
    inputs = {
        v.untyped_storage().data_ptr()
        for v in call.values()
        if isinstance(v, torch.Tensor) and v.numel()
    }
    topk_ids, expert_mask = call["topk_ids"], call.get("expert_mask")
    skip = {out_legacy.data_ptr(), out_v2.data_ptr()}
    problems = {}
    if "stage1" in dict(legacy):
        stage1 = dict(legacy)["stage1"]
        v2_stage1 = dict(v2)["stage1"]
        if stage1.func is fused_moe_1stage:
            # (hidden_states, w1, w2, topk, sorted_ids, sorted_weights,
            # sorted_expert_ids, num_valid_ids, moe_buf, isG1U1, block_size_M, ...);
            # moe_buf is the output.
            skip |= {stage1.args[8].data_ptr(), v2_stage1.args[8].data_ptr()}
            block_m = int(stage1.args[10])
            at = {"ids": 4, "weights": 5, "blocks": 6, "num_valid": 7}
            if stage1.keywords.get("flat"):
                # The FLAT pass-through: the topk ids and weights, so every tensor whole.
                at = {}
        elif "block_m" in stage1.keywords:
            # Executor and legacy both call stage 1 as (a1, w1, w2, sorted_ids,
            # sorted_expert_ids, num_valid_ids, out, topk, block_m=..., ...).
            block_m = int(stage1.keywords["block_m"])
            at = {"ids": 3, "blocks": 4, "num_valid": 5}
        else:
            return {("stage1", ""): f"stage1: {stage1.func} is not a known launch"}
        relaxed = _unrepeatable_stage1(stage1)
        native = bool(stage1.keywords.get("native_scale_layout"))
        if at:
            for role, i in at.items():
                role = "ids" if role == "weights" else role
                roles[stage1.args[i].data_ptr()] = role
                roles[v2_stage1.args[i].data_ptr()] = role
            sorted_len = len(stage1.args[at["ids"]])
            token_num = topk_ids.shape[0]
            views["legacy"] = _sort_view(stage1, at, block_m, token_num)
            views["v2"] = _sort_view(v2_stage1, at, block_m, token_num)
            num_valid = views["legacy"]["num_valid"]
            if legacy2 is not None:
                views["legacy2"] = _sort_view(
                    dict(legacy2)["stage1"], at, block_m, token_num
                )
            # The layout must match exactly; only the order inside a segment may not.
            for run in [r for r in ("v2", "legacy2") if r in views]:
                mine, ref = views[run], views["legacy"]
                same = (
                    mine["num_valid"] == num_valid
                    and torch.equal(mine["routed"], ref["routed"])
                    and torch.equal(
                        mine["experts"][mine["order"]], ref["experts"][ref["order"]]
                    )
                )
                if same:
                    continue
                if run == "legacy2":
                    legacy2 = None
                    del views["legacy2"]
                else:
                    return {
                        ("stage1", "sort"): (
                            "stage1: sort layouts differ (num_valid_ids, padding "
                            "positions or the expert of a (token, slot))"
                        )
                    }
            for _, launch in legacy + v2 + (legacy2 or []):
                for name in ("sorted_weights", "m_indices"):
                    if launch.keywords.get(name) is not None:
                        roles[launch.keywords[name].data_ptr()] = "ids"
                if launch.keywords.get("reverse_sorted") is not None:
                    roles[launch.keywords["reverse_sorted"].data_ptr()] = "reverse"
        model_dim = call["hidden_states"].shape[-1]
        _, _, inter_dim = get_inter_dim(call["w1"].shape, call["w2"].shape)
    slots = None if expert_mask is None else expert_mask[topk_ids.long()].bool()

    def routed(t):
        """`t` at the local routed slots if it is a `[token, topk, ...]` buffer."""
        if t.dim() == 3 and t.shape[:2] == topk_ids.shape and slots is not None:
            return _bits(t)[slots].view(t.dtype)
        return t

    def spread(x, z):
        """How much legacy's two runs differ in one floating-point argument, over the
        values finite in both (`_float_gap` checks the others)."""
        xf, zf = x.float(), z.float()
        finite = xf.isfinite() & zf.isfinite()
        return (xf - zf).abs()[finite].max().item() if finite.any() else 0.0

    def defined(name, where, role, t, run):
        """The part of argument `t` of `run` ("legacy", "v2" or "legacy2") the kernels
        define, rows in sorted order taken in that run's `_sort_view` order, and whether
        it must be bitwise equal."""
        view = views.get(run)
        rows = t.shape[0] if t.dim() == 2 else 0
        sorted_rows = view is not None and rows >= sorted_len
        scale = t.element_size() == 1 and (
            t.dtype == dtypes.fp8_e8m0 or where in ("a1_scale", "a2_scale")
        )
        if role == "ids":
            # Routed rows by (token, slot), then the padding (same positions each run).
            t = _bits(t[:num_valid])
            return torch.cat([t[view["order"]], t[view["padding"]]]), True
        if role == "blocks":
            return t[: num_valid // block_m], True
        if role == "num_valid":
            return t, True
        if role == "input":
            return t, not t.is_floating_point() or t.dtype == dtypes.fp4x2
        if role == "reverse":
            # The sorted position of each (token, slot): compared as the id the run's
            # own sort holds there, which does not depend on the order.
            outside = torch.full_like(t, -1)
            if not num_valid:
                return outside, True
            inside = (t >= 0) & (t < num_valid)
            at_id = view["ids"][t.clamp(0, num_valid - 1).long()]
            return torch.where(inside, at_id, outside), True
        if scale and sorted_rows and rows % 32 == 0:
            groups = (model_dim if name == "stage1" else inter_dim) // 32
            index = _sorted_scale_defined(
                view["order"], groups, -groups // 8 * -8, native and name != "stage1"
            )
            return _bits(t).reshape(-1)[index], True
        if sorted_rows and rows == sorted_len:
            t = _bits(t)[:num_valid][view["order"]].view(t.dtype)
        else:
            t = routed(t)
        return t, not t.is_floating_point() or t.dtype == dtypes.fp4x2

    seconds = [None] * len(legacy) if legacy2 is None else [l for _, l in legacy2]
    for (name, a), (_, b), second_launch in zip(legacy, v2, seconds):
        second = {} if second_launch is None else _expand(second_launch)
        if a.func != b.func:
            problems[name, ""] = f"{name}: {a.func} vs {b.func}"
            continue
        if len(a.args) != len(b.args) or set(a.keywords) != set(b.keywords):
            problems[name, ""] = f"{name}: arguments differ: {a} vs {b}"
            continue
        pairs = [(i, x, y) for i, (x, y) in enumerate(zip(a.args, b.args))]
        pairs += [(k, a.keywords[k], b.keywords[k]) for k in a.keywords]
        if name == "stage1" and _own_stage1_buffer(a):
            pairs = [pair for pair in pairs if pair[0] != 6]
        split = name == "stage1" and _split_1x128_out(a, topk_ids.shape[0])
        if split:
            other = dict(_split_1x128_out(b, topk_ids.shape[0]))
            pairs = [pair for pair in pairs if pair[0] != 6]
            pairs += [(where, x, other[where]) for where, x in split]
            if second_launch is not None:
                second.update(_split_1x128_out(second_launch, topk_ids.shape[0]))
        if name == "stage1" and (a.keywords.get("split_k") or 1) > 1:
            skip |= {a.args[6].data_ptr(), b.args[6].data_ptr()}
        expanded = []
        for where, x, y in pairs:
            if dataclasses.is_dataclass(x) and dataclasses.is_dataclass(y):
                if type(x) is not type(y):
                    problems[name, where] = (
                        f"{name}[{where}]: {type(x).__name__} vs {type(y).__name__}"
                    )
                    continue
                expanded.extend(
                    (
                        f"{where}.{field.name}",
                        getattr(x, field.name),
                        getattr(y, field.name),
                    )
                    for field in dataclasses.fields(x)
                )
            else:
                expanded.append((where, x, y))
        pairs = expanded
        for where, x, y in pairs:
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                if x != y:
                    problems[name, where] = f"{name}[{where}]: {x!r} vs {y!r}"
                continue
            if x.data_ptr() in skip:
                continue
            if x.dtype != y.dtype or x.shape != y.shape:
                problems[name, where] = (
                    f"{name}[{where}]: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}"
                )
                continue
            role = roles.get(x.data_ptr())
            if role is None and x.numel() and x.untyped_storage().data_ptr() in inputs:
                role = "input"
            dx, exact = defined(name, where, role, x, "legacy")
            dy, _ = defined(name, where, role, y, "v2")
            if torch.equal(_bits(dx), _bits(dy)):
                continue
            reason = "values differ"
            # After a stage 1 legacy does not repeat, the integer tensors quantized
            # from its output get the floating-point rule (`_unrepeatable_stage1`);
            # stage 1's inputs are not its output.
            loose_int = (
                exact
                and relaxed
                and role is None
                and (name != "stage1" or where == 6)
                and dx.dtype == x.dtype
                and not dx.is_floating_point()
                and dx.dtype != dtypes.fp4x2
            )
            if not exact or loose_int:
                z = second.get(where)
                like_x = isinstance(z, torch.Tensor) and z.dtype == x.dtype
                z = (
                    defined(name, where, role, z, "legacy2")[0]
                    if like_x and z.shape == x.shape
                    else None
                )
                if loose_int:
                    noise = 0 if z is None else (dx.int() - z.int()).abs().max().item()
                    reason = _int_gap(dx, dy, noise)
                    if reason is not None:
                        reason += f" (after {relaxed}, which legacy does not repeat)"
                else:
                    noise = 0.0 if z is None else spread(dx, z)
                    reason = _float_gap(dx, dy, noise, z)
                if reason is None:
                    continue
            problems[name, where] = (
                f"{name}[{where}]: {x.dtype}{tuple(x.shape)} {reason}"
            )
    return problems


def _split_1x128_out(launch, token_num):
    """`[("6.rows", fp8 rows), ("6.scales", fp32 scales)]` of the `out` buffer of an ASM
    1x128 stage 1, or None for another launch. That kernel writes its fp8 rows, then
    their fp32 scales as raw bytes, into one buffer (`AsmStage1Blockscale.out_buffer`),
    and the bytes after the scales are never written: each part is compared in its own
    dtype, the way `split_out` hands them to stage 2."""
    from aiter import QuantType
    from aiter.moev2.segments.asm import AsmStage1Blockscale, asm_stage1

    out = launch.args[6] if len(launch.args) > 6 else None
    if (
        launch.func is not asm_stage1
        or launch.keywords.get("quant_type") != QuantType.per_1x128
        or out is None
    ):
        return None
    _, topk, inter_dim = out.shape
    rows, scales = AsmStage1Blockscale.split_out(
        out, launch.args[0], launch.keywords["a1_scale"], token_num, topk, inter_dim
    )
    return [("6.rows", rows), ("6.scales", scales)]


def _own_stage1_buffer(launch) -> bool:
    """The FlyDSL a16w-mix stage 1 (bf16 into fp4 or int4 weights) writes a sorted
    intermediate it allocates and returns, and leaves its `out` argument unwritten."""
    from aiter.moev2.segments.flydsl import _flydsl_stage1_wrapper
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    if launch.func is not _flydsl_stage1_wrapper:
        return False
    parsed = get_flydsl_kernel_params(launch.keywords.get("kernelName", ""))
    return (
        parsed is not None
        and parsed.get("a_dtype") == "bf16"
        and parsed.get("b_dtype") in ("fp4", "int4")
    )


_NOISE_ULPS = 16


def _ulp(t):
    import torch

    finite = t.float()[t.isfinite()]
    peak = finite.abs().max().item() if finite.numel() else 0.0
    if peak == 0:
        return 0.0
    mantissa = {torch.bfloat16: 7, torch.float16: 10}.get(t.dtype, 23)
    return 2.0 ** (int(torch.tensor(peak).log2().floor()) - mantissa)


def entry_call(path, call):
    """`call` (legacy fused_moe arguments by name) as the mapping `path.claims` reads.
    The legacy hooks pass every parameter, defaults included."""
    import inspect

    import aiter.fused_moe as fm
    from aiter.moev2 import api

    bound = inspect.signature(fm.fused_moe).bind(**call)
    bound.apply_defaults()
    return {"api": dict, "op": api.op_call, "fhmoe": api.fhmoe_call}[path.entry](
        bound.arguments
    )


def run_differential(path_name, call, *, bitwise=False):
    """Run `call` (legacy fused_moe arguments by name) on legacy and on `path_name`.

    The path must claim and run the call (not decline it), launch the same kernels with
    the same arguments (`compare_launches`, against both legacy runs: exact but for
    floating-point tensors), and produce the same output: bitwise with
    `bitwise=True`, else within the larger of 4 ulps of the output's peak and twice the
    difference between two legacy runs (atomic accumulation in GEMM2 and in a split-K
    stage 1 is not deterministic), which fails above `_NOISE_ULPS` ulps of the peak.
    """
    import dataclasses

    import torch

    import aiter.fused_moe as fm
    from aiter.moev2 import paths

    path = paths.registered()[path_name]
    entry = entry_call(path, call)

    def launch(mode):
        bench, previous = [], fm.kernel_bench_callable
        fm.kernel_bench_callable = bench
        if call.get("output") is not None:
            # Legacy and v2 share the caller's buffer: what one leaves unwritten must
            # not pass as the other's result.
            call["output"].fill_(float("nan"))
        try:
            with moev2_mode(mode):
                out = fm.fused_moe(**call)
        finally:
            fm.kernel_bench_callable = previous
        torch.cuda.synchronize()
        if call.get("output") is not None and out is not call["output"]:
            raise AssertionError(f"{mode}: the result is not the caller's output=")
        return out, bench

    ref_raw, legacy_launches = launch("off")
    ref = ref_raw.clone()
    ref2_raw, legacy2_launches = launch("off")
    ref2 = ref2_raw.clone()
    results = []

    def run(c):
        out = path.run(c)
        results.append(out)
        return out

    paths._REGISTRY[path_name] = dataclasses.replace(path, run=run)
    try:
        with moev2_mode(path_name):
            routed = paths.route_to_v2(entry, path.entry)
            if routed is None or routed.name != path_name:
                raise AssertionError(f"path {path_name} does not claim this call")
        out_raw, v2_launches = launch(path_name)
        out = out_raw.clone()
    finally:
        paths._REGISTRY[path_name] = path
        paths.reload()
    if len(results) != 1 or results[0] is paths.DECLINED:
        raise AssertionError(f"path {path_name} did not run this call: {results}")
    stage1 = dict(legacy_launches).get("stage1")
    unrepeatable = stage1 is not None and _unrepeatable_stage1(stage1)
    try:
        problems = compare_launches(
            legacy_launches, v2_launches, call, ref_raw, out_raw, legacy2_launches
        )
        if problems:
            raise AssertionError(
                f"{path_name}: launches differ: "
                + "; ".join(list(problems.values())[:6])
            )
        if bitwise:
            assert ref.dtype == out.dtype and torch.equal(
                _bits(ref), _bits(out)
            ), f"{path_name}: output differs from legacy"
        else:
            _check_output(path_name, ref, ref2, out)
    except Exception as exc:
        if not unrepeatable or isinstance(exc, torch.OutOfMemoryError):
            raise
        raise AssertionError(
            f"{type(exc).__name__}: {exc} {_LEGACY_NOISE}{unrepeatable}]"
        ) from exc
    return ref, out


_LEGACY_NOISE = "[legacy does not repeat "


def may_be_legacy_noise(problem):
    """Whether a failure may come from legacy not repeating itself: its stage 1 is in
    `_unrepeatable_stage1`, or its two runs already differ too much."""
    return _LEGACY_NOISE in problem or "legacy output not repeatable" in problem


def _same_non_finite(a, b):
    """The values of `a` and `b` agree where `a` is not finite: NaN for NaN (any
    payload), the same infinity."""
    import torch

    where = ~a.isfinite()
    if not torch.equal(where, ~b.isfinite()):
        return False
    x, y = a[where].float(), b[where].float()
    return bool(((x.isnan() & y.isnan()) | (x == y)).all())


def _check_output(path_name, ref, ref2, out):
    """`out` matches legacy `ref` within the noise between two legacy runs, which may
    not exceed `_NOISE_ULPS` ulps of the output's peak."""
    finite = ref.isfinite()

    def gap(x):
        diff = (x.float()[finite] - ref.float()[finite]).abs()
        return diff.max().item() if diff.numel() else 0.0

    noise = gap(ref2) if _same_non_finite(ref, ref2) else float("inf")
    ulp = _ulp(ref)
    assert noise <= _NOISE_ULPS * ulp, (
        f"{path_name}: legacy output not repeatable: two legacy runs differ by {noise}, "
        f"above {_NOISE_ULPS} ulps of the peak ({_NOISE_ULPS * ulp}); a legacy problem, "
        "v2 cannot be validated on this case"
    )
    assert _same_non_finite(
        ref, out
    ), f"{path_name}: non-finite outputs differ from legacy"
    err = gap(out)
    tol = max(4 * ulp, 2 * noise)
    assert (
        err <= tol
    ), f"{path_name}: max |v2 - legacy| {err} > {tol} (legacy noise {noise})"


if __name__ == "__main__":
    sys.exit(main())
