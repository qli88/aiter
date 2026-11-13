# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Differential test of aiter.moev2 against the legacy MoE code.

CPU (`--check`): for every key of the legacy snapshot that a registered v2 path takes,
the v2 plan (in the snapshot record format) must equal the legacy golden record, and the
log lines v2 dispatch emits must equal the ones `get_2stage_cfgs` emits for the key. There
is no allowlist: the revamp keeps today's behavior, and a bug
fix lands in the legacy code first, with the snapshot regenerated.

    python tests/moe_dispatch/differential.py --check [--arch gfx950]
    python tests/moe_dispatch/differential.py --check --env AITER_FLYDSL_FORCE=0

`--env NAME=VALUE` sets a dispatch-affecting variable for legacy and v2 alike; the legacy
records are then generated live rather than read from the golden files (which hold the
default environment).

GPU: `run_differential(path_name, call)` runs one call through AITER_MOEV2=off and
AITER_MOEV2=<path> and compares the kernel launches and outputs (see
tests/moe_dispatch/gpu_differential.py). It raises `LegacyNotRepeatable` where legacy's
own runs explain every difference of v2 (by element for launch arguments; by element,
within a small bound for its row, for outputs), so the comparison does not
decide, and `KnownDecline` where the path
declines the call as `known_declines.py` lists it.
"""

import argparse
import contextlib
import itertools
import json
import os
import pathlib
import re
import sys
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent


@contextlib.contextmanager
def captured_logs():
    """Collect `(level, message)` of the aiter logger at INFO, printing nothing."""
    import logging

    logger = logging.getLogger("aiter")
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
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


# v2 holds a 1-stage row without its kernelName2 (`convert.cells`), which legacy logs as
# the table reads it: "Null", a blank cell (nan) or empty.
_ONE_STAGE_KERNEL2 = re.compile(
    r"(using 1stage(?: xbf16)? \(kernelName1='[^']*', kernelName2=)(?:'Null'|nan|'')\)"
)


def legacy_stage1_as_v2(kernel_name1, kernel_name2) -> str:
    """The stage 1 name v2 runs where legacy runs `kernel_name1` before the GEMM2
    `kernel_name2` it launches: `convert.v2_kernel1` of the tuned names (an fp8 GEMM2
    input renames an fp4 layout GEMM2 after an fp4 stage 1, `stage2_kernel`)."""
    from aiter.moev2.dispatch.convert import v2_kernel1

    if kernel_name1.startswith("flydsl_moe1_afp4_"):
        kernel_name2 = kernel_name2.replace(
            "flydsl_moe2_layout_afp8_", "flydsl_moe2_layout_afp4_", 1
        )
    return v2_kernel1(kernel_name1, kernel_name2)


# A 2-stage row's names as both log them.
_TWO_STAGE_KERNELS = re.compile(
    r"(using 2stage \(kernelName1=')([^']*)(', kernelName2=')([^']*)('\))"
)


def _two_stage_as_v2(match):
    head, kernel1, middle, kernel2, tail = match.groups()
    return head + legacy_stage1_as_v2(kernel1, kernel2) + middle + kernel2 + tail


def _as_v2_logs(logs):
    """Legacy's log lines `logs` with a 1-stage row's kernelName2 and a 2-stage
    row's kernelName1 (`legacy_stage1_as_v2`) as v2 logs them."""
    return [
        (
            level,
            _TWO_STAGE_KERNELS.sub(
                _two_stage_as_v2, _ONE_STAGE_KERNEL2.sub(r"\1'')", line)
            ),
        )
        for level, line in logs
    ]


def as_v2_record(record):
    """Legacy's dispatch record with its stage 1 name as v2 runs it
    (`legacy_stage1_as_v2`)."""
    stage1, stage2 = record.get("stage1"), record.get("stage2")
    if not (isinstance(stage1, dict) and isinstance(stage2, dict)):
        return record
    kernel1 = stage1.get("kw", {}).get("kernelName")
    kernel2 = stage2.get("kw", {}).get("kernelName")
    if not (isinstance(kernel1, str) and isinstance(kernel2, str)):
        return record
    renamed = legacy_stage1_as_v2(kernel1, kernel2)
    if renamed == kernel1:
        return record
    kw = {**stage1["kw"], "kernelName": renamed}
    return {**record, "stage1": {**stage1, "kw": kw}}


def _legacy_logs(key):
    import aiter.fused_moe as fm

    with captured_logs() as logs:
        try:
            fm.get_2stage_cfgs.__wrapped__(*key.args, **key.call_kwargs())
        except Exception:  # noqa: BLE001, S110
            pass
    return _as_v2_logs(logs)


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
    `fhmoe`) and its resolver declines it, else None."""
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
    return None if reason is None else (problem, reason)


def legacy_decline(path, key):
    """`(reason, stage-1 functions)` when `path` hands `key` back to a named legacy
    branch, else None; the functions are None for a 1-stage branch.

    A 1-stage tuned row the path does not run and a ranking region marked `legacy=`
    name the branch.
    """
    declined = v2_decline(path, key)
    if declined is None:
        return None
    reason = declined[1]
    if reason.startswith(("1-stage row with ", "no segment runs 1-stage ")):
        return reason, None
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
            got = as_v2_record(record.normalize(_use_fhmoe_wrappers(metadata)))
        except Exception as exc:  # noqa: BLE001
            got = record.error_record(exc)
    return got, _as_v2_logs(logs)


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
        # Load legacy's tuned table once up front so no key's logs include its load warnings.
        with captured_logs():
            if keys:
                _legacy_logs(keys[0])
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
                        handed_back = legacy_decline(path, key)
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
                expected = as_v2_record(snapshot.resolve_same_as_base(golden, label))
                if got != expected:
                    problems.append(
                        f"{label} [{path.name}]\n"
                        f"    legacy: {json.dumps(expected, sort_keys=True)}\n"
                        f"    v2:     {json.dumps(got, sort_keys=True)}"
                    )
            fhmoe_label = f"{label}|fhmoe"
            for path in fhmoe_paths:
                if any(fhmoe_label in seen for seen in outcomes.values()):
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

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup(env=args.env)
    from moe_dispatch import arch_sim

    failed = False
    for arch in args.arch or list(arch_sim.ARCHS):
        outcomes, problems, known = compare(
            arch, check_logs=not args.no_logs, live=bool(args.env)
        )

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


class LegacyNotRepeatable(AssertionError):
    """Legacy differs from itself where the comparison needs it to repeat: undecided,
    not a v2 failure."""


class KnownDecline(AssertionError):
    """The path declines the call (legacy runs it) as `known_declines.py` lists it."""


# The calls of `dirty_allocator`; cross_checkout's workers take every other one each.
_FILLS = itertools.count()
# The largest request the caching allocator serves from its small pool.
_SMALL_REQUEST = 1 << 20
_ALLOC_CONF = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
)


def fill_byte(n):
    """The byte `dirty_allocator` writes on its `n`-th call: 96 values in turn, each of
    them finite repeated as fp32, bf16, fp16, fp8 (e4m3fn, e4m3fnuz, e5m2, e5m2fnuz)
    and e8m0."""
    return 0x10 + n % 96


def _free_blocks(segments, device, stream):
    """The requests that take every inactive block of the caching allocator that
    `stream` on `device` can reuse, largest first (it serves the smallest free block
    that fits): small-pool blocks in pieces the small pool serves."""
    sizes = []
    for segment in segments:
        if (segment["device"], segment["stream"]) != (device, stream):
            continue
        if tuple(segment["segment_pool_id"]) != (0, 0):
            continue
        small = segment["segment_type"] == "small"
        for block in segment["blocks"]:
            if block["state"] != "inactive":
                continue
            n = block["size"]
            while small and n > _SMALL_REQUEST:
                sizes.append(_SMALL_REQUEST)
                n -= _SMALL_REQUEST
            sizes.append(n)
    return sorted(sizes, reverse=True)


def dirty_allocator():
    """Fill every freed block of the caching allocator with a byte new to each call, so
    the next run's `torch.empty` buffers hold other bytes than any run before, at every
    offset (back-to-back runs often repeat bitwise even where the kernels do not, and
    bytes no kernel writes then look repeatable). A 256 MiB block first, so that memory
    not cached yet is filled too. Taking each block whole relies on the allocator's
    default configuration (no expandable segments, no rounding to powers of two)."""
    import torch

    for name in _ALLOC_CONF:
        if os.environ.get(name):
            raise RuntimeError(
                f"dirty_allocator needs the default caching allocator: unset {name}"
            )
    value = fill_byte(next(_FILLS))
    torch.full((256 << 20,), value, dtype=torch.uint8, device="cuda")
    sizes = _free_blocks(
        torch.cuda.memory_snapshot(),
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
    )
    held = [torch.full((n,), value, dtype=torch.uint8, device="cuda") for n in sizes]
    del held


def entry_call(path, call):
    """`call` (legacy fused_moe arguments by name) as `path`'s legacy hook passes it:
    every parameter, defaults included."""
    import inspect

    import aiter.fused_moe as fm
    from aiter.moev2 import api

    bound = inspect.signature(fm.fused_moe).bind(**call)
    bound.apply_defaults()
    return {"api": dict, "op": api.op_call, "fhmoe": api.fhmoe_call}[path.entry](
        bound.arguments
    )


def known_decline(path_name, call):
    """`(reason, entry)`: the resolver's last decline when `path_name` runs `call` again
    with v2's caches cleared (its log lines dropped), and the `known_declines.py` entry
    covering it (None if none does); None when the path does not decline it there."""
    from unittest import mock

    from aiter.moev2 import paths
    from aiter.moev2.dispatch import resolver, tuned
    from moe_dispatch import known_declines

    path = paths.registered()[path_name]
    resolve = resolver.resolve
    declined = []

    def recording(problem, *args, **kwargs):
        resolution = resolve(problem, *args, **kwargs)
        if resolution.declined is not None:
            declined.append((problem, resolution.declined))
        return resolution

    with moev2_mode(path_name), captured_logs():
        tuned.clear_cache()
        with mock.patch.object(resolver, "resolve", recording):
            out = path.run(entry_call(path, call))
    if out is not paths.DECLINED or not declined:
        return None
    problem, reason = declined[-1]
    return reason, known_declines.find(path_name, problem, reason)


def _sorted_scale_defined(sorted_ids, token_num, groups, padded_groups):
    """Flat indices of the bytes the sorted MX quant kernels write in their swizzled
    e8m0 scale (`mx_scale_shuffle_idx` in csrc/include/mx_quant_utils.h): the rows of
    routed tokens (not block padding), the `groups` columns (not the pad to 8)."""
    import torch

    rows = ((sorted_ids & 0xFFFFFF) < token_num).nonzero().flatten()
    x = rows[:, None]
    y = torch.arange(groups, device=rows.device)[None, :]
    idx = (
        (x // 32 * padded_groups) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )
    return idx.flatten()


def _raw(t):
    """`t`'s bits as integers, so that equal NaNs compare equal."""
    import torch

    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return t.view(width[t.element_size()])


def compare_launches(
    legacy, v2, call, out_legacy, out_v2, *, as_v2=True, positions=None
):
    """`{(stage, argument): problem}` between two `kernel_bench_callable` recordings.

    Same stages, same functions, equal non-tensor arguments, and bitwise-equal tensors
    over what the kernels define: the sort buffers up to `num_valid_ids` (the rest is
    `torch.empty`), MX scales in sorted order at the routed rows and the `K / 32` groups
    (`_sorted_scale_defined`), other buffers in sorted order at the routed rows (stage 1
    skips block padding), `[token, topk, inter]` buffers at the slots in the sorted ids
    (stage 1 writes no others: not the slots of experts that are not local, nor the
    repeats of an expert in one token's top-k), stage 1's 2-D activation and its scale
    (`[token, ...]`) at the tokens in the sorted ids (an activation quant can leave the
    rows of tokens without a local expert unwritten; the ASM 1x128 stage 1 reads its
    scale transposed, `[K / 128, rows]` in memory, `_transposed_scale`), and nothing for
    the output buffers (a problem where only one side passes its own),
    for the `out` argument of a stage 1 that writes a buffer of its own
    (`_own_stage1_buffer`); only dtypes and shapes for the output of a split-K stage 1
    (accumulated with atomics) and stage 2's activation and `a2_scale` made from it.
    `as_v2`: the legacy stage 1 kernel name is compared as v2 names it; off
    for two legacy runs. `positions`: a dict that gets, for each problem that is only
    "values differ", a bool tensor of where the compared elements differ; the elements
    are the same ones in every comparison against the same `legacy` recording.
    """
    import torch

    from aiter import dtypes
    from aiter.moev2.segments.asm import fused_moe_1stage
    from aiter.moev2.shapes import get_inter_dim

    if [n for n, _ in legacy] != [n for n, _ in v2]:
        return {
            ("", ""): f"stages differ: {[n for n, _ in legacy]} vs {[n for n, _ in v2]}"
        }
    stage1 = dict(legacy)["stage1"]
    v2_stage1 = dict(v2)["stage1"]
    skip = {out_legacy.data_ptr(), out_v2.data_ptr()}
    if stage1.func is fused_moe_1stage:
        # (hidden_states, w1, w2, topk, sorted_ids, sorted_weights, sorted_expert_ids,
        # num_valid_ids, moe_buf, isG1U1, block_size_M, ...); moe_buf is the output.
        skip |= {stage1.args[8].data_ptr(), v2_stage1.args[8].data_ptr()}
        block_m = int(stage1.args[10])
        at = {"ids": 4, "weights": 5, "blocks": 6, "num_valid": 7}
        if stage1.keywords.get("flat"):
            # The FLAT pass-through: the topk ids and weights, so every tensor whole.
            at = {}
    elif "block_m" in stage1.keywords:
        # (a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, out, topk,
        # block_m=..., ...)
        block_m = int(stage1.keywords["block_m"])
        at = {"ids": 3, "blocks": 4, "num_valid": 5}
    else:
        return {("stage1", ""): f"stage1: {stage1.func} is not a known launch"}
    num_valid = int(stage1.args[at["num_valid"]][0]) if at else None
    roles = {}
    for role, i in at.items():
        if role != "num_valid":
            role = "ids" if role == "weights" else role
            roles[stage1.args[i].data_ptr()] = roles[v2_stage1.args[i].data_ptr()] = (
                role
            )
    for _, launch in legacy + v2:
        for name in ("sorted_weights", "m_indices"):
            if launch.keywords.get(name) is not None:
                roles[launch.keywords[name].data_ptr()] = "ids"
    topk_ids = call["topk_ids"]
    slots = None
    sorted_ids = stage1.args[at["ids"]] if at else None
    if sorted_ids is not None:
        routed = sorted_ids[:num_valid]
        routed_rows = (routed & 0xFFFFFF) < topk_ids.shape[0]
        slots = torch.zeros(topk_ids.shape, dtype=torch.bool, device=topk_ids.device)
        routed = routed[routed_rows].long()
        slots[routed & 0xFFFFFF, routed >> 24] = True
    model_dim = call["hidden_states"].shape[-1]
    _, _, inter_dim = get_inter_dim(call["w1"].shape, call["w2"].shape)
    problems = {}
    for (name, a), (_, b) in zip(legacy, v2):
        if a.func is not b.func:
            problems[name, ""] = f"{name}: {a.func} vs {b.func}"
            continue
        if len(a.args) != len(b.args) or set(a.keywords) != set(b.keywords):
            problems[name, ""] = f"{name}: arguments differ: {a} vs {b}"
            continue
        pairs = [(i, x, y) for i, (x, y) in enumerate(zip(a.args, b.args))]
        pairs += [(k, a.keywords[k], b.keywords[k]) for k in a.keywords]
        if as_v2 and name == "stage1" and isinstance(a.keywords.get("kernelName"), str):
            kernel2 = dict(legacy).get("stage2")
            kernel2 = getattr(kernel2, "keywords", {}).get("kernelName")
            if isinstance(kernel2, str):
                pairs = [
                    (
                        (k, legacy_stage1_as_v2(x, kernel2), y)
                        if k == "kernelName"
                        else (k, x, y)
                    )
                    for k, x, y in pairs
                ]
        if name == "stage1" and _own_stage1_buffer(a):
            pairs = [pair for pair in pairs if pair[0] != 6]
        unvalued = set()
        if name == "stage1" and _split_k_stage1(a):
            unvalued = {6}
        if name == "stage2" and _split_k_stage1(stage1):
            unvalued = {0, "a2_scale"}
        for where, x, y in pairs:
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                if x != y:
                    problems[name, where] = f"{name}[{where}]: {x!r} vs {y!r}"
                continue
            if (x.data_ptr() in skip) != (y.data_ptr() in skip):
                problems[name, where] = (
                    f"{name}[{where}]: only one side passes its output buffer here"
                )
                continue
            if x.data_ptr() in skip:
                continue
            if x.dtype != y.dtype or x.shape != y.shape:
                problems[name, where] = (
                    f"{name}[{where}]: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}"
                )
                continue
            if where in unvalued:
                continue
            role = roles.get(x.data_ptr())
            rows = x.shape[0] if x.dim() == 2 else 0
            scale = x.element_size() == 1 and (
                x.dtype == dtypes.fp8_e8m0 or where in ("a1_scale", "a2_scale")
            )
            if sorted_ids is None:
                pass  # FLAT: no sorted buffers
            elif scale and rows >= len(sorted_ids) and rows % 32 == 0:
                groups = (model_dim if name == "stage1" else inter_dim) // 32
                defined = _sorted_scale_defined(
                    sorted_ids[:num_valid], topk_ids.shape[0], groups, -groups // 8 * -8
                )
                x, y = (_raw(t).reshape(-1)[defined] for t in (x, y))
            elif role is None and rows == len(sorted_ids):
                x, y = (_raw(t)[:num_valid][routed_rows] for t in (x, y))
            elif role == "ids":
                x, y = x[:num_valid], y[:num_valid]
            elif role == "blocks":
                x, y = x[: num_valid // block_m], y[: num_valid // block_m]
            elif x.dim() == 3 and x.shape[:2] == topk_ids.shape and slots is not None:
                x, y = (_raw(t)[slots] for t in (x, y))
            elif (
                name == "stage1"
                and where in (0, "a1_scale")
                and rows == topk_ids.shape[0]
                and slots is not None
            ):
                tokens = slots.any(1)
                if where == "a1_scale" and _transposed_scale(a, x):
                    n = call.get("num_local_tokens")
                    n = rows if n is None else int(n.reshape(-1)[0])
                    groups = x.shape[1]
                    x, y = (
                        _raw(t).reshape(-1)[: groups * n].view(groups, n)[:, tokens[:n]]
                        for t in (x, y)
                    )
                else:
                    x, y = (_raw(t)[tokens] for t in (x, y))
            if not torch.equal(_raw(x), _raw(y)):
                problems[name, where] = (
                    f"{name}[{where}]: {x.dtype}{tuple(x.shape)} values differ"
                )
                if positions is not None:
                    positions[name, where] = _raw(x) != _raw(y)
    return problems


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


def _split_k_stage1(launch) -> bool:
    """The stage 1 accumulates K splits into its output with atomics, in no fixed order:
    CK-Tile `split_k`, the CK 1x128 `splitk`, the ASM `ksplit` (legacy never passes
    one), the FlyDSL `_kb{n}` kernels."""
    from aiter import QuantType
    from aiter.moev2.segments.asm import asm_stage1
    from aiter.moev2.segments.ck import ck_moe_stage1
    from aiter.moev2.segments.flydsl import _flydsl_stage1_wrapper
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    keywords = launch.keywords
    if launch.func is asm_stage1:
        return (keywords.get("ksplit") or 0) > 0
    if launch.func is ck_moe_stage1:
        return (keywords.get("splitk") or 1) > 1 and keywords.get(
            "quant_type"
        ) == QuantType.per_1x128
    if launch.func is _flydsl_stage1_wrapper:
        parsed = get_flydsl_kernel_params(keywords.get("kernelName") or "") or {}
        return parsed.get("k_batch", 1) > 1
    return (keywords.get("split_k") or 1) > 1


def _transposed_scale(launch, scale) -> bool:
    """The ASM 1x128 stage 1 takes its activation scale as the quant wrote it with
    `transpose_scale=True`: shape `[token, K / 128]`, memory `[K / 128, rows]` (rows:
    `num_local_tokens` when given, else `token`)."""
    from aiter import QuantType
    from aiter.moev2.segments.asm import asm_stage1

    return (
        launch.func is asm_stage1
        and launch.keywords.get("quant_type") == QuantType.per_1x128
        and scale.is_contiguous()
    )


def _mantissa(dtype):
    import torch

    return {
        torch.bfloat16: 7,
        torch.float16: 10,
        torch.float64: 52,
        torch.float8_e4m3fn: 3,
        torch.float8_e4m3fnuz: 3,
        torch.float8_e5m2: 2,
        torch.float8_e5m2fnuz: 2,
    }.get(dtype, 23)


def _ulp(peak, dtype):
    """One unit in the last place of `dtype` at magnitude `peak` (0 for integers)."""
    import math

    if peak == 0 or not dtype.is_floating_point:
        return 0.0
    return 2.0 ** (math.floor(math.log2(peak)) - _mantissa(dtype))


def _ulps(t, dtype):
    """`_ulp` of `dtype` at each element of `t`, a float tensor."""
    import torch

    if not dtype.is_floating_point:
        return torch.zeros_like(t)
    _, exponent = torch.frexp(t)
    unit = torch.ldexp(torch.ones_like(t), exponent - 1 - _mantissa(dtype))
    return unit.where(t != 0, 0)


# Legacy runs taken, beyond the first two, when v2's launches or output differ from
# legacy's first run, to see whether legacy differs from itself there too.
_MORE_LEGACY_RUNS = 4


def unexplained_launches(problems, positions, varies):
    """The `compare_launches` problems of v2 that legacy's own runs do not explain: any
    other than "values differ" (a function, an argument, a dtype or a shape), and values
    that differ at an element where every legacy run equals the first. `positions`: v2's
    differing elements per problem; `varies`: the elements where some legacy run differs
    from the first, per argument."""
    unexplained = {}
    for key, problem in problems.items():
        if key not in positions:
            unexplained[key] = problem
            continue
        alone = positions[key] & ~varies[key] if key in varies else positions[key]
        if alone.any():
            unexplained[key] = (
                f"{problem} at {int(alone.sum())} of {alone.numel()} elements where "
                "legacy's runs agree"
            )
    return unexplained


# The bound for a row of an output, in ulps of its peak: twice the largest difference of
# the quietest legacy run that differs there, kept within these. Atomic accumulation in
# GEMM2 varies by a few ulps; where legacy varies by more, only its runs at that element
# widen the bound. The upper one also sets which elements are calm.
_WHOLE_ULPS = (4, 16)


def output_outside(ref, runs, out, *, bitwise=False):
    """Why `out` is farther from legacy's first output `ref` than legacy's other outputs
    `runs` let it be, or None. `out` and every run must have `ref`'s dtype and shape.
    Where every run has `ref`'s NaN or infinity, `out` must have it too, and an element
    whose NaN or infinity differs between legacy's runs is not compared; with `bitwise`,
    an element every run repeats bitwise must be repeated bitwise. Every other element
    must be within twice the largest difference of a legacy run there, or within a bound
    for its row (the last dimension; a 1-D output is one row): none with `bitwise`, else
    `_WHOLE_ULPS` at the row's peak over its calm elements, where every legacy run is
    within 16 ulps of the element's own magnitude (garbage that legacy reads, in one
    run or in all, widens neither the peak nor the bound if it changes between runs,
    as the fill of `dirty_allocator` makes it; a row without a calm element has no
    bound of its own)."""
    import torch

    shapes = [(f"legacy run {i}", r) for i, r in enumerate(runs, 2)] + [("out", out)]
    for label, x in shapes:
        if (x.dtype, x.shape) != (ref.dtype, ref.shape):
            return (
                f"{label} is {x.dtype}{tuple(x.shape)}, legacy run 1 "
                f"{ref.dtype}{tuple(ref.shape)}"
            )
    if ref.numel() == 0:
        return None
    finite, nan = ref.isfinite(), ref.isnan()

    def same_state(x):
        return (
            (x.isfinite() == finite) & (x.isnan() == nan) & (finite | nan | (x == ref))
        )

    stable = torch.ones_like(finite)
    for r in runs:
        stable &= same_state(r)
    wrong = stable & ~same_state(out)
    if wrong.any():
        return (
            f"NaN or infinity other than legacy's at {int(wrong.sum())} elements where "
            f"legacy's {len(runs) + 1} runs agree on them"
        )
    if bitwise:
        exact = torch.ones_like(finite)
        for r in runs:
            exact &= _raw(r) == _raw(ref)
        changed = exact & (_raw(out) != _raw(ref))
        if changed.any():
            return (
                f"bits other than legacy's at {int(changed.sum())} elements where "
                f"legacy's {len(runs) + 1} runs repeat bitwise"
            )
    keep = stable & finite
    base = ref.float().where(keep, 0)

    def gap(x):
        return (x.float().where(keep, 0) - base).abs()

    spread = torch.zeros_like(base)
    for r in runs:
        spread = torch.maximum(spread, gap(r))
    bound = 2 * spread
    whole = torch.zeros_like(base)
    if not bitwise:
        low, high = _WHOLE_ULPS
        calm = keep & (spread <= high * _ulps(base, ref.dtype))
        width = base.shape[-1] if base.dim() > 1 else base.numel()

        def by_row(x):
            return x.reshape(-1, width)

        peak = by_row(base.abs().where(calm, 0)).amax(1)
        unit = _ulps(peak, ref.dtype)
        quiet = torch.full_like(peak, float("inf"))
        for r in runs:
            row = by_row(gap(r).where(calm, 0)).amax(1)
            quiet = torch.minimum(quiet, row.where(row > 0, float("inf")))
        quiet = quiet.where(quiet.isfinite(), 0)
        row = torch.minimum(torch.maximum(low * unit, 2 * quiet), high * unit)
        whole = row[:, None].expand(-1, width).reshape(base.shape)
        bound = torch.maximum(bound, whole)
    over = gap(out) > bound
    if not over.any():
        return None
    worst = (gap(out) - bound).where(over, float("-inf")).argmax()
    return (
        f"{int(over.sum())} elements beyond what legacy's {len(runs) + 1} runs vary "
        f"(worst: off by {gap(out).flatten()[worst].item()}, where legacy runs differ by "
        f"up to {spread.flatten()[worst].item()} and the bound is "
        f"{bound.flatten()[worst].item()}; its row's bound is "
        f"{whole.flatten()[worst].item()})"
    )


def run_differential(path_name, call, *, bitwise=False):
    """Run `call` (legacy fused_moe arguments by name) on legacy and on `path_name`.

    The path must claim and run the call (not decline it), launch the same kernels on the
    same inputs as legacy's first run (`compare_launches`), and produce an output of the
    same dtype and shape that matches legacy's first run: bitwise with `bitwise=True`,
    else within `_check_output` of legacy's first two runs and within `output_outside`
    of them (atomic accumulation in GEMM2 and in a split-K stage 1 is not
    deterministic). The allocator is dirtied before each run.

    Raises `KnownDecline` when the path declines the call as `known_declines.py` lists
    it. Where v2 does not pass, legacy runs up to `_MORE_LEGACY_RUNS` more times, and
    `LegacyNotRepeatable` (undecided) is raised when legacy's own runs explain every
    difference: v2's launch arguments differ from legacy's first run only at elements
    where another legacy run differs from it too (`unexplained_launches`), and v2's
    output is within `output_outside` of all legacy runs. Anything else fails. v2 runs
    once: a later legacy run never stands in for the first.
    """
    import dataclasses

    import torch

    import aiter.fused_moe as fm
    from aiter.moev2 import paths

    path = paths.registered()[path_name]
    entry = entry_call(path, call)

    def launch(mode):
        dirty_allocator()
        bench = []
        fm.kernel_bench_callable = bench
        try:
            with moev2_mode(mode):
                out = fm.fused_moe(**call)
        finally:
            fm.kernel_bench_callable = None
        torch.cuda.synchronize()
        return out, bench

    ref, legacy_launches = launch("off")
    more, varies = [], {}

    def legacy_again():
        again, launches = launch("off")
        where = {}
        compare_launches(
            legacy_launches, launches, call, ref, again, as_v2=False, positions=where
        )
        for key, differ in where.items():
            varies[key] = varies[key] | differ if key in varies else differ
        more.append(again)

    legacy_again()
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
        out, v2_launches = launch(path_name)
    finally:
        paths._REGISTRY[path_name] = path
        paths.reload()
    if len(results) == 1 and results[0] is paths.DECLINED:
        declined = known_decline(path_name, call)
        if declined is not None and declined[1] is not None:
            raise KnownDecline(
                f"{path_name} declines this call: {declined[1].name} ({declined[0]})"
            )
        reason = "" if declined is None else f" ({declined[0]})"
        raise AssertionError(f"path {path_name} declined this call{reason}")
    if len(results) != 1:
        raise AssertionError(f"path {path_name} did not run this call: {results}")
    positions = {}
    problems = compare_launches(
        legacy_launches, v2_launches, call, ref, out, positions=positions
    )
    while len(more) <= _MORE_LEGACY_RUNS and any(
        key in positions for key in unexplained_launches(problems, positions, varies)
    ):
        legacy_again()
    differ = unexplained_launches(problems, positions, varies)
    if differ:
        raise AssertionError(
            f"{path_name}: launches differ: " + "; ".join(list(differ.values())[:6])
        )
    if (ref.dtype, ref.shape) != (out.dtype, out.shape):
        raise AssertionError(
            f"{path_name}: output {ref.dtype}{tuple(ref.shape)} vs "
            f"{out.dtype}{tuple(out.shape)}"
        )
    undecided = []
    if problems:
        undecided.append(
            "v2 differs from legacy's first run only where legacy runs differ: "
            + "; ".join(list(problems.values())[:6])
        )
    if bitwise:
        if not torch.equal(_raw(ref), _raw(out)):
            undecided.append(f"{path_name}: output differs from legacy")
    else:
        try:
            _check_output(path_name, ref, more[0], out)
            far = output_outside(ref, more[:1], out)
            if far is not None:
                undecided.append(
                    f"{path_name}: v2's output against 2 legacy runs: {far}"
                )
        except AssertionError as exc:
            undecided.append(str(exc))
    if not undecided:
        return ref, out
    while len(more) <= _MORE_LEGACY_RUNS:
        legacy_again()
    far = output_outside(ref, more, out, bitwise=bitwise)
    if far is not None:
        raise AssertionError(
            f"{path_name}: v2's output: {far}; " + "; ".join(undecided)
        )
    raise LegacyNotRepeatable(
        f"{path_name}: " + "; ".join(undecided) + f"; v2's output is within what "
        f"legacy's {len(more) + 1} runs vary"
    )


def _check_output(path_name, ref, ref2, out, names=("legacy", "v2")):
    """`out` matches legacy `ref` within the noise between two legacy runs. `names`:
    what the messages call the reference and `out`."""
    import torch

    ref_name, out_name = names
    finite = ref.isfinite()

    def same_non_finite(x):
        """NaN where `ref` has NaN, the same infinities where it has those."""
        nan = ref.isnan()
        inf = ~finite & ~nan
        return (
            torch.equal(finite, x.isfinite())
            and torch.equal(nan, x.isnan())
            and torch.equal(ref[inf], x[inf])
        )

    def gap(x):
        diff = (x.float()[finite] - ref.float()[finite]).abs()
        return diff.max().item() if diff.numel() else 0.0

    noise = gap(ref2) if same_non_finite(ref2) else float("inf")
    peak = ref.float()[finite].abs().max().item() if finite.any() else 0.0
    if not noise <= peak:
        raise LegacyNotRepeatable(
            f"{path_name}: {ref_name} output not reproducible (noise {noise}, peak "
            f"{peak})"
        )
    assert same_non_finite(
        out
    ), f"{path_name}: non-finite outputs differ from {ref_name}"
    err = gap(out)
    tol = max(4 * _ulp(peak, ref.dtype), 2 * noise)
    assert err <= tol, (
        f"{path_name}: max |{out_name} - {ref_name}| {err} > {tol} ({ref_name} "
        f"noise {noise})"
    )


if __name__ == "__main__":
    sys.exit(main())
