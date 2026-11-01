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
tests/moe_dispatch/gpu_differential.py).
"""

import argparse
import contextlib
import json
import os
import pathlib
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


def legacy_decline(path, key):
    """`(reason, stage-1 functions)` when `path` hands `key` back to a named legacy
    branch, else None; the functions are None for a 1-stage branch.

    A 1-stage tuned row the path does not run and a ranking region marked `legacy=`
    name the branch.
    """
    from aiter.moev2 import api
    from aiter.moev2.dispatch import resolver

    if path.takes is None:
        return None
    try:
        problem = api.problem_from_key(key.args, key.call_kwargs())
    except Exception:  # noqa: BLE001 - key_checks.py reports these
        return None
    if not path.takes(problem):
        return None
    reason = resolver.resolve(problem, path.segments).declined or ""
    if reason.startswith(("1-stage row with ", "no segment runs 1-stage ")):
        return reason, None
    if reason.startswith("ranking ") and reason.endswith(" is not ported"):
        legacy = reason.split(": ", 1)[1].removesuffix(" is not ported")
        if legacy.startswith("1-stage "):
            return reason, None
        # An unlisted branch matches no stage 1: compare() reports it.
        return reason, _LEGACY_2STAGE.get(legacy, ())
    return None


def compare(arch, check_logs=True, live=False):
    """`(claimed, one_stage, two_stage, problems)`: `{label: path name}` three times and
    a list of strings. `live`: generate the legacy records instead of reading golden/.

    `one_stage`: keys a path hands back as legacy 1-stage keys, which must be 1-stage
    in the legacy record. `two_stage`: keys a ranking region hands back to a named legacy
    2-stage branch, whose stage 1 the legacy record must run.
    """
    from aiter.moev2 import paths
    from aiter.moev2.dispatch import tuned
    from moe_dispatch import arch_sim, record, snapshot

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
    claimed, one_stage, two_stage, problems = {}, {}, {}, []
    with arch_sim.simulated_arch(arch) as (gfx, cu_num):
        keys = snapshot.dispatch_keys(gfx, cu_num)
        # Load the tuned table once up front so no key's logs include the load warnings.
        with captured_logs():
            if keys:
                _legacy_logs(keys[0])
            tuned._default_table()
        for key in keys:
            label = key.label()
            if label in claimed:
                continue
            for path in checkers:
                with captured_logs() as v2_logs:
                    try:
                        plan = path.dispatch_record(key.args, key.call_kwargs())
                    except Exception as exc:  # noqa: BLE001
                        plan = record.error_record(exc)
                if plan is None:
                    with captured_logs():
                        handed_back = legacy_decline(path, key)
                    if handed_back is None:
                        continue
                    reason, stage1 = handed_back
                    expected = snapshot.resolve_same_as_base(golden, label)
                    if stage1 is None:
                        one_stage[label] = path.name
                        if not expected.get("run_1stage"):
                            problems.append(
                                f"{label} [{path.name}] declined ({reason}) but legacy "
                                f"runs it 2-stage: {json.dumps(expected, sort_keys=True)}"
                            )
                        continue
                    two_stage[label] = path.name
                    got = expected.get("stage1")
                    got = got.get("fn") if isinstance(got, dict) else got
                    if expected.get("run_1stage") or got not in stage1:
                        problems.append(
                            f"{label} [{path.name}] declined ({reason}) but legacy "
                            f"does not run {stage1}: {json.dumps(expected, sort_keys=True)}"
                        )
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
    return claimed, one_stage, two_stage, problems


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

    snapshot._setup()
    # After _setup, which drops the dispatch-affecting variables of the caller.
    for item in args.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
    from moe_dispatch import arch_sim

    failed = False
    for arch in args.arch or list(arch_sim.ARCHS):
        claimed, one_stage, two_stage, problems = compare(
            arch, check_logs=not args.no_logs, live=bool(args.env)
        )

        def summary(by_label):
            counts = Counter(by_label.values())
            return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"

        print(
            f"{arch}: {len(claimed)} keys claimed by v2 ({summary(claimed)}), "
            f"{len(one_stage)} handed back as legacy 1-stage ({summary(one_stage)}), "
            f"{len(two_stage)} as a legacy 2-stage branch ({summary(two_stage)}), "
            f"{len(problems)} problems"
        )
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


def _bits(t):
    import torch

    return t.view(torch.uint8) if t.element_size() == 1 else t


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


def compare_launches(legacy, v2, call, out_legacy, out_v2):
    """`{(stage, argument): problem}` between two `kernel_bench_callable` recordings.

    Same stages, same functions, equal non-tensor arguments, and bitwise-equal tensors
    over what the kernels define: the sort buffers up to `num_valid_ids` (the rest is
    `torch.empty`), MX scales in sorted order at the routed rows and the `K / 32` groups
    (`_sorted_scale_defined`), other buffers in sorted order at the routed rows (stage 1
    skips block padding), `[token, topk, inter]` buffers at the routed slots that
    are local (stage 1 skips the others), and nothing for the output buffer, for the
    `out` argument of a stage 1 that writes a buffer of its own (`_own_stage1_buffer`),
    nor for the output of a split-K stage 1 (accumulated with atomics).
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
    topk_ids, expert_mask = call["topk_ids"], call.get("expert_mask")
    slots = None if expert_mask is None else expert_mask[topk_ids.long()].bool()
    sorted_ids = stage1.args[at["ids"]] if at else None
    if sorted_ids is not None:
        routed_rows = (sorted_ids[:num_valid] & 0xFFFFFF) < topk_ids.shape[0]
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
        if name == "stage1" and _own_stage1_buffer(a):
            pairs = [pair for pair in pairs if pair[0] != 6]
        if name == "stage1" and (a.keywords.get("split_k") or 1) > 1:
            skip |= {a.args[6].data_ptr(), b.args[6].data_ptr()}
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
                x, y = (_bits(t).reshape(-1)[defined] for t in (x, y))
            elif role is None and rows == len(sorted_ids):
                x, y = (_bits(t)[:num_valid][routed_rows] for t in (x, y))
            elif role == "ids":
                x, y = x[:num_valid], y[:num_valid]
            elif role == "blocks":
                x, y = x[: num_valid // block_m], y[: num_valid // block_m]
            elif x.dim() == 3 and x.shape[:2] == topk_ids.shape and slots is not None:
                x, y = (_bits(t)[slots] for t in (x, y))
            if not torch.equal(_bits(x), _bits(y)):
                problems[name, where] = (
                    f"{name}[{where}]: {x.dtype}{tuple(x.shape)} values differ"
                )
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


def _ulp(t):
    import torch

    finite = t.float()[t.isfinite()]
    peak = finite.abs().max().item() if finite.numel() else 0.0
    if peak == 0:
        return 0.0
    mantissa = {torch.bfloat16: 7, torch.float16: 10}.get(t.dtype, 23)
    return 2.0 ** (int(torch.tensor(peak).log2().floor()) - mantissa)


def run_differential(path_name, call, *, bitwise=False):
    """Run `call` (legacy fused_moe arguments by name) on legacy and on `path_name`.

    The path must claim and run the call (not decline it), launch the same kernels on the
    same inputs (`compare_launches`), and produce the same output: bitwise with
    `bitwise=True`, else within the larger of 4 ulps of the output's peak and twice the
    difference between two legacy runs (atomic accumulation in GEMM2 and in a split-K
    stage 1 is not deterministic).
    """
    import dataclasses
    import inspect

    import torch

    import aiter.fused_moe as fm
    from aiter.moev2 import api, paths

    path = paths.registered()[path_name]
    bound = inspect.signature(fm.fused_moe).bind(**call)
    bound.apply_defaults()
    # The legacy hooks pass every parameter, defaults included; check the same mapping.
    entry_call = (
        dict(bound.arguments) if path.entry == "api" else api.op_call(bound.arguments)
    )

    def launch(mode):
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
    ref2, _ = launch("off")
    results = []

    def run(c):
        out = path.run(c)
        results.append(out)
        return out

    paths._REGISTRY[path_name] = dataclasses.replace(path, run=run)
    try:
        with moev2_mode(path_name):
            routed = paths.route_to_v2(entry_call, path.entry)
            if routed is None or routed.name != path_name:
                raise AssertionError(f"path {path_name} does not claim this call")
        out, v2_launches = launch(path_name)
    finally:
        paths._REGISTRY[path_name] = path
        paths.reload()
    if len(results) != 1 or results[0] is paths.DECLINED:
        raise AssertionError(f"path {path_name} did not run this call: {results}")
    problems = compare_launches(legacy_launches, v2_launches, call, ref, out)
    if problems:
        raise AssertionError(
            f"{path_name}: launches differ: " + "; ".join(list(problems.values())[:6])
        )
    if bitwise:
        assert torch.equal(ref, out), f"{path_name}: output differs from legacy"
        return ref, out
    _check_output(path_name, ref, ref2, out)
    return ref, out


def _check_output(path_name, ref, ref2, out):
    """`out` matches legacy `ref` within the noise between two legacy runs."""
    import torch

    finite = ref.isfinite()

    def gap(x):
        diff = (x.float()[finite] - ref.float()[finite]).abs()
        return diff.max().item() if diff.numel() else 0.0

    noise = gap(ref2) if torch.equal(finite, ref2.isfinite()) else float("inf")
    peak = ref.float()[finite].abs().max().item() if finite.any() else 0.0
    assert (
        noise <= peak
    ), f"{path_name}: legacy output not reproducible (noise {noise}, peak {peak})"
    assert torch.equal(finite, out.isfinite()) and torch.equal(
        ref[~finite], out[~finite]
    ), f"{path_name}: non-finite outputs differ from legacy"
    err = gap(out)
    tol = max(4 * _ulp(ref), 2 * noise)
    assert (
        err <= tol
    ), f"{path_name}: max |v2 - legacy| {err} > {tol} (legacy noise {noise})"


if __name__ == "__main__":
    sys.exit(main())
