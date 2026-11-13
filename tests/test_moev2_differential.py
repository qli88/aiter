# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code.

The CPU test compares dispatch plans and log lines on every snapshot key a v2 path
claims. The GPU test runs tests/moe_dispatch/gpu_differential.py for each registered
path: same kernels on the same inputs, same outputs and log lines as legacy (for
`comm_fused`, the TP8 tests/moe_dispatch/comm_fused_differential.py, on 8 GPUs; for
`mega_moe_v2`, the EP8 tests/moe_dispatch/mega_moe_v2_differential.py, on 8 gfx950
GPUs; for `mega_moe_tp`, the TP4 tests/moe_dispatch/mega_moe_tp_differential.py, on 4
gfx950 GPUs; for `grouped_gfx1250`, the GPU test of tests/test_moev2_grouped_gfx1250.py, on
gfx1250).
"""

import functools
import pathlib
import re
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def test_dispatch_plans_match_legacy():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "differential.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        proc.stdout[-6000:]
        + proc.stderr[-3000:]
        + "\nv2 must make the same choice as legacy for every key it claims."
    )


def _registered():
    from aiter.moev2 import paths

    return sorted(paths.registered())


@pytest.mark.parametrize("path_name", _registered())
def test_outputs_match_legacy(path_name):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if path_name == "comm_fused" and torch.cuda.device_count() < 8:
        pytest.skip("the comm_fused differential runs TP8: needs 8 GPUs")
    if path_name == "mega_moe_v2":
        from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() != "gfx950" or torch.cuda.device_count() < 8:
            pytest.skip("the mega_moe_v2 differential runs EP8 on gfx950")
    if path_name == "mega_moe_tp":
        from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() != "gfx950" or torch.cuda.device_count() < 4:
            pytest.skip("the mega_moe_tp differential runs TP4 on gfx950")
    if path_name == "mega_moe_gfx1250":
        pytest.skip(
            "no multi-GPU gfx1250 differential yet; the CPU gate is "
            "tests/test_moev2_mega_moe_gfx1250.py"
        )
    proc = subprocess.run(
        [
            sys.executable,
            str(_HERE / "moe_dispatch" / "gpu_differential.py"),
            "--path",
            path_name,
        ],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


def test_check_output_compares_non_finite_values_by_position():
    import torch
    from moe_dispatch.differential import LegacyNotRepeatable, _check_output

    nan, inf = float("nan"), float("inf")
    ref = torch.tensor([1.0, nan, -inf, 2.0], dtype=torch.bfloat16)
    ref2 = torch.tensor([1.0078125, nan, -inf, 2.0], dtype=torch.bfloat16)
    _check_output("p", ref, ref2, ref.clone())
    for out, says in (
        ([1.0, 0.0, -inf, 2.0], "non-finite outputs differ from legacy"),
        ([1.0, nan, inf, 2.0], "non-finite outputs differ from legacy"),
        ([1.5, nan, -inf, 2.0], "max |v2 - legacy|"),
    ):
        with pytest.raises(AssertionError, match=re.escape(says)) as raised:
            _check_output("p", ref, ref2, torch.tensor(out, dtype=torch.bfloat16))
        assert not isinstance(raised.value, LegacyNotRepeatable)
    moved = torch.tensor([nan, 1.0, -inf, 2.0], dtype=torch.bfloat16)
    with pytest.raises(LegacyNotRepeatable, match="reference output not reproducible"):
        _check_output("p", ref, moved, ref.clone(), ("reference", "candidate"))


def _xc_reply(tmp_path, label, outs, logs=(), **extra):
    """A cross_checkout worker reply with outputs `outs` (saved to `tmp_path`)."""
    import torch

    runs = []
    for i, out in enumerate(outs):
        file = tmp_path / f"{label}{i}.pt"
        torch.save(out, file)
        runs.append({"output": str(file), "launches": []})
    reply = {"inputs": {}, "error": None, "logs": list(logs), "runs": runs}
    return {**reply, "inputs_after": {}, **extra}


def test_cross_checkout_drops_table_load_lines_with_v2_off(tmp_path):
    import torch

    sys.path.insert(0, str(_HERE / "moe_dispatch"))
    from moe_dispatch import cross_checkout

    out = torch.ones(4, dtype=torch.bfloat16)
    used = (20, "[fused_moe] using x")
    loaded = (30, "[fused_moe] duplicate tuned rows (primary) in t.csv; keeping first")
    ref = _xc_reply(tmp_path, "r", (out, out), (loaded, used))
    cand = _xc_reply(tmp_path, "c", (out, out), (used,))
    assert cross_checkout.compare(ref, cand, False) == []
    cand = _xc_reply(tmp_path, "c", (out, out), (used, used))
    assert "log lines differ" in cross_checkout.compare(ref, cand, False)[0]


def test_cross_checkout_reports_known_declines_and_others(tmp_path):
    import torch

    sys.path.insert(0, str(_HERE / "moe_dispatch"))
    from moe_dispatch import cross_checkout

    out = torch.ones(4, dtype=torch.bfloat16)
    logs = ((20, "[fused_moe] using x"),)
    outcomes = [["api", "a8w4", "declined", 3]]
    reason = "a default MXFP4 branch takes the cktile_* row"
    known = [["a8w4", reason, "cktile_* rows a default MXFP4 branch takes (5.59)"]]
    ref = _xc_reply(tmp_path, "r", (out, out), logs)
    cand = _xc_reply(tmp_path, "c", (out, out), logs, outcomes=outcomes, declines=known)
    info = {}
    assert cross_checkout.compare(ref, cand, True, info) == []
    assert info["known"] == [known[0][2]]
    unknown = [["a8w4", "no feasible plan", None]]
    cand = _xc_reply(
        tmp_path, "c", (out, out), logs, outcomes=outcomes, declines=unknown
    )
    info = {}
    problems = cross_checkout.compare(ref, cand, True, info)
    assert "known" not in info
    assert any("v2 did not serve" in p and "no feasible plan" in p for p in problems)


def test_cross_checkout_reports_a_passing_known_decline_as_known(tmp_path, capsys):
    import io
    import json
    from types import SimpleNamespace

    import torch

    sys.path.insert(0, str(_HERE / "moe_dispatch"))
    from moe_dispatch import cross_checkout

    out = torch.ones(4, dtype=torch.bfloat16)
    logs = ((20, "[fused_moe] using x"),)
    outcomes = [["api", "a8w4", "declined", 3]]
    reason = "a default MXFP4 branch takes the cktile_* row"
    known = [["a8w4", reason, "cktile_* rows a default MXFP4 branch takes (5.59)"]]
    unknown = [["a8w4", "no feasible plan", None]]

    class Worker:
        def __init__(self, label, replies):
            self.label, self.device, self.replies = label, 0, iter(replies)

        def send(self, request):
            return True

        def read(self, timeout, n):
            return next(self.replies)

    workers = [
        Worker(
            "reference",
            [_xc_reply(tmp_path, f"r{i}", (out, out), logs) for i in (0, 1)],
        ),
        Worker(
            "candidate",
            [
                _xc_reply(
                    tmp_path, f"c{i}", (out, out), logs, outcomes=outcomes, declines=d
                )
                for i, d in enumerate((known, unknown))
            ],
        ),
    ]
    todo = [("a8w4", [i], "builtin") for i in (0, 1)]
    args = SimpleNamespace(timeout=1, ref="ref", cand_env=[])
    results = io.StringIO()
    counts, _ = cross_checkout._run(todo, workers, args, True, results)
    statuses = [json.loads(line)["status"] for line in results.getvalue().splitlines()]
    assert statuses == ["KNOWN", "FAIL"]
    assert (counts["known"], counts["failed"], counts["undecided"]) == (1, 1, 0)
    assert capsys.readouterr().out.startswith("KNOWN ")


def test_cross_checkout_workers_never_write_the_same_fill_byte():
    sys.path.insert(0, str(_HERE / "moe_dispatch"))
    from moe_dispatch import cross_checkout
    from moe_dispatch.differential import fill_byte

    bytes_ = {}
    for label in ("reference", "candidate"):
        fills = cross_checkout._fills(label)
        bytes_[label] = [fill_byte(next(fills)) for _ in range(200)]
        assert all(a != b for a, b in zip(bytes_[label], bytes_[label][1:]))
    assert not set(bytes_["reference"]) & set(bytes_["candidate"])


def test_cross_checkout_decides_by_how_far_the_reference_runs_vary(tmp_path):
    import torch

    sys.path.insert(0, str(_HERE / "moe_dispatch"))
    from moe_dispatch import cross_checkout

    def bf16(*values):
        return torch.tensor(values, dtype=torch.bfloat16)

    def compare(refs, cands):
        info = {}
        ref = _xc_reply(tmp_path, "r", refs)
        problems = cross_checkout.compare(
            ref, _xc_reply(tmp_path, "c", cands), False, info
        )
        return problems, info

    good, near, far = bf16(1.0, 2.0), bf16(1.0, 2.015625), bf16(1.0, 2.5)
    garbage = bf16(1.0, 3e36)
    # Within `_check_output` of a reference whose first run a race hit, but beyond the
    # reference's runs where they agree; a candidate run of another dtype.
    raced, clean = bf16(1.0, 2.0, 1e30), bf16(1.0, 2.0, 3.0)
    problems, _ = compare((raced, clean), (clean, bf16(1.0, 50.0, 3.0)))
    assert "candidate run 2 against 2 reference runs" in problems[0]
    assert compare((raced, clean), (clean, clean))[0] == []
    problems, _ = compare((good, good), (good, good.float()))
    assert "vs torch.float32(2,)" in problems[0]
    # The reference not reproducible at one element: undecided there, failed elsewhere.
    problems, info = compare((good, garbage), (good, good))
    assert problems == [] and info["output"] == "tolerance"
    assert "reference output not reproducible" in info["undecided"]
    assert "undecided" not in compare((good, garbage), (good, bf16(1.5, 2.0)))[1]
    assert compare((good, garbage), (good, bf16(1.5, 2.0)))[0]
    problems, info = compare((good, near), (good, far))
    assert "undecided" not in info and problems
    # A later reference run as far out as the candidate's: undecided, not a failure.
    problems, info = compare((good, near, near, far), (good, far))
    assert (
        problems == [] and "no farther from the reference's first" in info["undecided"]
    )
    # A candidate far beyond every reference run fails, whatever the reference does.
    problems, info = compare(
        (good, near, near, bf16(1.0, 2.125)), (good, bf16(1, 1e30))
    )
    assert "undecided" not in info and "beyond what legacy's 4 runs vary" in problems[0]
    # The same with a reference whose first two runs repeat bitwise.
    problems, info = compare((good, good, good, near), (good, near))
    assert problems == [] and info["output"] == "bitwise"
    assert "which its runs 3 to 4 do not repeat" in info["undecided"]
    # Held to the row bound there, not to bits, even where all its runs repeat.
    first, later = bf16(4.0, 2.0), bf16(4.0, 2.0625)
    problems, info = compare((first, first, first, later), (first, bf16(4.0625, 2)))
    assert problems == [] and "do not repeat" in info["undecided"]
    problems, info = compare((good, good, good, near), (good, far))
    assert "undecided" not in info and "candidate run 2 output: " in problems[0]
    problems, info = compare((good, good, good, good), (good, near))
    assert "undecided" not in info and "repeats bitwise" in problems[0]


def test_output_outside_compares_each_element_with_legacy_runs():
    import torch
    from moe_dispatch.differential import output_outside

    nan, inf = float("nan"), float("inf")

    def bf16(*values):
        return torch.tensor(values, dtype=torch.bfloat16)

    ref = bf16(1.0, 2.0, nan, inf, 4.0)
    runs = [bf16(1.0, 2.0, nan, inf, 4.0), bf16(1.0, 2.0, nan, 5.0, 6.0)]
    # Element 4 varies by 2 in legacy, so 4 is allowed there; element 3 is not compared.
    assert output_outside(ref, runs, bf16(1.0, 2.0, nan, 1.0, 8.0)) is None
    assert output_outside(ref, runs, bf16(1.0, 2.0, nan, -inf, 0.0)) is None
    assert "off by 0.5" in output_outside(ref, runs, bf16(1.5, 2.0, nan, inf, 4.0))
    assert "off by 8.0" in output_outside(ref, runs, bf16(1.0, 2.0, nan, inf, 12.0))
    for wrong in (bf16(1.0, nan, nan, inf, 4.0), bf16(1.0, 2.0, 0.0, inf, 4.0)):
        assert "NaN or infinity" in output_outside(ref, runs, wrong)
    # Within 4 ulps of the peak (0.125 for a peak of 4), unless bitwise.
    close = bf16(1.0625, 2.0, nan, inf, 4.0)
    assert output_outside(ref, runs, close) is None
    assert output_outside(ref, runs, close, bitwise=True)
    # The quietest run sets the bound of a row, between 4 and 16 ulps of its peak over
    # the calm elements (0.125 to 0.5 at 4).
    ref = bf16(1.0, 2.0, 4.0)
    assert output_outside(ref, [bf16(1.125, 2.0, 4.0)], bf16(1.0, 2.25, 4.0)) is None
    assert output_outside(ref, [bf16(1.125, 2.0, 4.0)], bf16(1.0, 2.375, 4.0))
    found = output_outside(ref, [bf16(1.0, 2.0, 4.5)], bf16(1.0, 2.75, 4.0))
    assert "its row's bound is 0.5" in found
    noisy = [bf16(1.0, 2.0, 4.03125), bf16(1.0, 2.0, 4.5)]
    assert "its row's bound is 0.125" in output_outside(ref, noisy, bf16(1, 2.5, 4))
    # A run that repeats the row exactly says nothing of its noise.
    quiet = [ref.clone(), bf16(1.0, 2.0, 4.25)]
    assert output_outside(ref, quiet, bf16(1.0, 2.5, 4.0)) is None
    # An element legacy moves by more than 16 of its ulps is not calm: neither its
    # magnitude nor its noise widens the bound (4 ulps of 2 here).
    found = output_outside(ref, [bf16(1.0, 2.0, 6.0)], bf16(1.0, 2.5, 4.0))
    assert "its row's bound is 0.0625" in found
    assert output_outside(ref, [bf16(1.0, 2.0, 6.0)], bf16(1.0, 2.0, 7.0)) is None
    assert output_outside(bf16(1, 2, 3), [bf16(1, 2, 6)], bf16(1, 8, 3))
    # A race in legacy's first run, or garbage it reads in every run, widens neither
    # the peak nor the bound elsewhere.
    ref = bf16(1.0, 2.0, 1e30)
    runs = [bf16(1.0, 2.0, 3.0)] * 5
    assert output_outside(ref, runs, bf16(1.0, 2.0, 3.0)) is None
    assert "off by 48.0" in output_outside(ref, runs, bf16(1.0, 50.0, 3.0))
    ref = bf16(1.0, 2.0, 3e20)
    runs = [bf16(1.0, 2.0, g) for g in (1e20, 5e20, 2e20, 7e20, 4e20)]
    assert output_outside(ref, runs, bf16(1.0, 2.0, 6e20)) is None
    assert "off by 498.0" in output_outside(ref, runs, bf16(1.0, 500.0, 2e20))
    # Each row has its own bound: a large row does not widen a small one, and a row
    # with no calm element has none.
    ref = torch.tensor([[1.0, 2.0], [1e20, 1.0]], dtype=torch.bfloat16)
    out = torch.tensor([[1.0, 2.5], [1e20, 1.0]], dtype=torch.bfloat16)
    assert "its row's bound is 0.0625" in output_outside(ref, [ref.clone()], out)
    ref = torch.tensor([[1e20, 3e20]], dtype=torch.bfloat16)
    run = torch.tensor([[2e20, 1e20]], dtype=torch.bfloat16)
    within = torch.tensor([[2e20, 5e20]], dtype=torch.bfloat16)
    assert output_outside(ref, [run], within) is None
    found = output_outside(
        ref, [run], torch.tensor([[1e20, 8e20]], dtype=torch.bfloat16)
    )
    assert "its row's bound is 0.0" in found


def test_output_outside_checks_bits_shapes_and_empty_outputs():
    import torch
    from moe_dispatch.differential import output_outside

    zero = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    negative = torch.tensor([-0.0, 1.0], dtype=torch.bfloat16)
    assert output_outside(zero, [zero], negative) is None
    assert "bits other than legacy's at 1" in output_outside(
        zero, [zero], negative, bitwise=True
    )
    nan = torch.tensor([0x7FC0, 0], dtype=torch.int16).view(torch.bfloat16)
    other_nan = torch.tensor([0x7FC1, 0], dtype=torch.int16).view(torch.bfloat16)
    assert output_outside(nan, [nan], other_nan, bitwise=True)
    assert output_outside(nan, [other_nan], other_nan, bitwise=True) is None
    for shape in ((0, 8), (4, 0)):
        empty = torch.zeros(shape, dtype=torch.bfloat16)
        assert output_outside(empty, [empty], empty) is None
    assert "legacy run 2 is torch.bfloat16(3,)" in output_outside(
        zero, [torch.zeros(3, dtype=torch.bfloat16)], zero
    )
    assert "out is torch.float32(2,)" in output_outside(zero, [zero], zero.float())


def test_ulp_follows_the_output_format():
    import torch
    from moe_dispatch.differential import _ulp

    assert _ulp(4.0, torch.bfloat16) == 2.0**-5
    assert _ulp(4.0, torch.float16) == 2.0**-8
    assert _ulp(4.0, torch.float32) == 2.0**-21
    assert _ulp(4.0, torch.float8_e4m3fn) == 0.5
    assert _ulp(4.0, torch.float8_e5m2fnuz) == 1.0
    assert _ulp(4.0, torch.int32) == 0.0
    assert _ulp(0.0, torch.bfloat16) == 0.0


def _differential_runner(monkeypatch):
    """`run_differential` on fake legacy and v2 calls: `run(legacy, v2, ...)` gives the
    verdict and how often each side ran. `legacy`: legacy's outputs, run by run; `v2`:
    v2's output (or `paths.DECLINED`); `launches` / `v2_launch`: the one launch
    argument each run records (zeros by default); `decline`: what `known_decline`
    returns."""
    import dataclasses
    import os

    import torch
    from moe_dispatch import differential
    from moe_dispatch.differential import (
        KnownDecline,
        LegacyNotRepeatable,
        run_differential,
    )

    import aiter.fused_moe as fm
    from aiter.moev2 import paths

    @dataclasses.dataclass(frozen=True)
    class FakePath:
        name: str
        entry: str
        run: object

    def compare(legacy, v2, call, out_legacy, out_v2, *, as_v2=True, positions=None):
        (_, x), (_, y) = legacy[0], v2[0]
        if x.shape != y.shape:
            return {("stage1", 0): "stage1[0]: shapes differ"}
        if torch.equal(x, y):
            return {}
        if positions is not None:
            positions["stage1", 0] = x != y
        return {("stage1", 0): "stage1[0]: values differ"}

    def run(legacy, v2, bitwise=False, launches=None, v2_launch=None, decline=None):
        outs = iter(legacy)
        args = iter(launches or [torch.zeros(3)] * 6)
        calls = []
        path = FakePath("a16w16", "api", lambda call: v2)

        def fused_moe(**call):
            if os.environ["AITER_MOEV2"] == "off":
                calls.append("legacy")
                fm.kernel_bench_callable.append(("stage1", next(args)))
                return next(outs)
            calls.append("v2")
            launch = torch.zeros(3) if v2_launch is None else v2_launch
            fm.kernel_bench_callable.append(("stage1", launch))
            return paths._REGISTRY["a16w16"].run(call)

        monkeypatch.setattr(paths, "_REGISTRY", {})
        monkeypatch.setattr(paths, "registered", lambda: {"a16w16": path})
        monkeypatch.setattr(
            paths, "route_to_v2", lambda call, entry: paths._REGISTRY["a16w16"]
        )
        monkeypatch.setattr(fm, "fused_moe", fused_moe)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(differential, "dirty_allocator", lambda: None)
        monkeypatch.setattr(differential, "entry_call", lambda path, call: call)
        monkeypatch.setattr(differential, "compare_launches", compare)
        monkeypatch.setattr(differential, "known_decline", lambda name, call: decline)
        try:
            run_differential("a16w16", {}, bitwise=bitwise)
            outcome = "ok"
        except KnownDecline as exc:
            outcome = f"KNOWN {exc}"
        except LegacyNotRepeatable as exc:
            outcome = f"LEGACY {exc}"
        except AssertionError as exc:
            outcome = f"FAIL {exc}"
        return outcome, calls.count("legacy"), calls.count("v2")

    return run


def test_run_differential_judges_v2_against_legacy_first_run(monkeypatch):
    import torch

    run = _differential_runner(monkeypatch)

    def bf16(*values):
        return torch.tensor(values, dtype=torch.bfloat16)

    good = bf16(1.0, 2.0, 4.0)
    assert run([good] * 6, good) == ("ok", 2, 1)
    # A race in legacy's first run hides nothing where its runs agree.
    outcome, legacy, v2 = run([bf16(1, 2, 1e30)] + [bf16(1, 2, 3)] * 5, bf16(1, 50, 3))
    assert outcome.startswith("FAIL") and "off by 48.0" in outcome
    assert (legacy, v2) == (6, 1)
    # v2 equal to legacy's later runs, not its first two: undecided, never ok.
    later = bf16(1.0, 2.5, 4.0)
    outcome, legacy, v2 = run([good, good] + [later] * 4, later)
    assert outcome.startswith("LEGACY") and (legacy, v2) == (6, 1)
    # Within legacy's spread over six runs, beyond it over two.
    runs = [good, bf16(1, 2, 4.25), good, bf16(1, 2, 6), good, good]
    assert run(runs, bf16(1, 2, 6))[0].startswith("LEGACY")
    assert run(runs, bf16(1, 2, 9))[0].startswith("FAIL")
    outcome, legacy, _ = run([good] * 6, good.float())
    assert outcome.startswith("FAIL") and "torch.float32" in outcome and legacy == 2
    # Legacy's bits in another 16-bit format.
    outcome, legacy, _ = run([good] * 6, good.view(torch.float16), bitwise=True)
    assert outcome.startswith("FAIL") and "torch.float16" in outcome and legacy == 2
    assert run([good] * 6, good.clone(), bitwise=True) == ("ok", 2, 1)


def test_run_differential_judges_launches_and_declines(monkeypatch):
    from types import SimpleNamespace

    import torch

    from aiter.moev2 import paths

    run = _differential_runner(monkeypatch)
    out = torch.ones(3, dtype=torch.bfloat16)
    zero, moved = torch.zeros(3), torch.tensor([1.0, 0.0, 0.0])
    # v2's launch differs where a later legacy run does too: undecided, never ok;
    # legacy runs until that run explains it, then six times for the output.
    launches = [zero, zero, moved, zero, zero, zero]
    outcome, legacy, v2 = run([out] * 6, out, launches=launches, v2_launch=moved)
    assert outcome.startswith("LEGACY") and "values differ" in outcome
    assert (legacy, v2) == (6, 1)
    # Where every legacy run agrees: a failure, after the extra runs.
    outcome, legacy, _ = run(
        [out] * 6, out, launches=launches, v2_launch=torch.tensor([0.0, 1.0, 0.0])
    )
    assert outcome.startswith("FAIL") and "where legacy's runs agree" in outcome
    assert legacy == 6
    # Another kind of launch problem fails at once.
    outcome, legacy, _ = run([out] * 6, out, v2_launch=torch.zeros(4))
    assert outcome.startswith("FAIL") and "shapes differ" in outcome and legacy == 2
    # A decline: known by `known_declines.py`, or not.
    entry = SimpleNamespace(name="cktile_* rows a default MXFP4 branch takes (5.59)")
    outcome, _, _ = run([out] * 6, paths.DECLINED, decline=("why", entry))
    assert outcome.startswith("KNOWN") and "5.59" in outcome
    outcome, _, _ = run([out] * 6, paths.DECLINED, decline=("why", None))
    assert outcome.startswith("FAIL") and "declined this call (why)" in outcome


def test_unexplained_launches_takes_only_elements_where_legacy_varies():
    import torch
    from moe_dispatch.differential import unexplained_launches

    key = ("stage1", 0)
    problems = {key: "stage1[0]: values differ", ("stage1", ""): "not a known launch"}
    v2 = {key: torch.tensor([False, True, True])}
    varies = {
        key: torch.tensor([False, True, True]),
        ("stage1", ""): torch.tensor([True]),
    }
    assert list(unexplained_launches(problems, v2, varies)) == [("stage1", "")]
    del problems["stage1", ""]
    assert unexplained_launches(problems, v2, varies) == {}
    varies[key] = torch.tensor([True, True, False])
    found = unexplained_launches(problems, v2, varies)
    assert "at 1 of 3 elements where legacy's runs agree" in found[key]
    assert key in unexplained_launches(problems, v2, {})


def test_the_allocator_fill_is_finite_in_every_float_format():
    import torch
    from moe_dispatch.differential import fill_byte

    values = {fill_byte(n) for n in range(200)}
    assert len(values) == 96
    for byte in values:
        raw = torch.full((8,), byte, dtype=torch.uint8)
        for dtype in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ):
            assert raw.view(dtype).float().isfinite().all(), (byte, dtype)
        assert byte != 0xFF  # e8m0 NaN


def test_the_allocator_fill_takes_every_free_block_the_stream_can_reuse():
    from moe_dispatch.differential import _free_blocks

    def segment(blocks, kind="large", device=0, stream=0, pool=(0, 0)):
        return {
            "device": device,
            "stream": stream,
            "segment_type": kind,
            "segment_pool_id": pool,
            "blocks": [{"size": n, "state": state} for n, state in blocks],
        }

    mib = 1 << 20
    segments = [
        segment([(12 * mib, "inactive")]),
        segment([(8 * mib, "active_allocated"), (20 * mib, "inactive")]),
        segment([(2 * mib, "inactive")], "small"),
        segment([(512, "active_allocated"), (1536, "inactive")], "small"),
        segment([(30 * mib, "inactive")], device=1),
        segment([(30 * mib, "inactive")], stream=7),
        segment([(30 * mib, "inactive")], pool=(1, 0)),
        segment([(4 * mib, "active_awaiting_free")]),
    ]
    assert _free_blocks(segments, 0, 0) == [20 * mib, 12 * mib, mib, mib, 1536]


def test_compare_launches_skips_rows_of_tokens_without_a_local_expert():
    import torch
    from moe_dispatch.differential import compare_launches

    def stage1(*args, **kwargs):
        pass

    token, topk = 4, 2
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32)
    # Expert 0 and 1 local: token 2 has no local expert (expert 3 is not local either).
    routed = [0 | 0 << 24, 3 | 1 << 24, 0 | 1 << 24, 1 | 0 << 24]
    call = {
        "topk_ids": topk_ids,
        "hidden_states": torch.zeros(token, 8, dtype=torch.bfloat16),
        "w1": torch.zeros(4, 16, 8),
        "w2": torch.zeros(4, 8, 8),
    }

    def launch(a1, w1_scale=None):
        w1_scale = torch.zeros(4, 8) if w1_scale is None else w1_scale
        sorted_ids = torch.tensor(routed + [token] * 4, dtype=torch.int32)
        blocks = torch.tensor([0, 1], dtype=torch.int32)
        out = torch.zeros(token, topk, 8)
        args = (a1, call["w1"], call["w2"], sorted_ids, blocks, torch.tensor([4]), out)
        stage = functools.partial(stage1, *args, topk, block_m=2, w1_scale=w1_scale)
        return [("stage1", stage)], out

    a1 = torch.arange(token * 8, dtype=torch.uint8).reshape(token, 8)
    legacy, out_legacy = launch(a1)
    unwritten = a1.clone()
    unwritten[2] = 255
    v2, out_v2 = launch(unwritten)
    assert compare_launches(legacy, v2, call, out_legacy, out_v2) == {}
    read = a1.clone()
    read[3] = 255
    v2, out_v2 = launch(read)
    positions = {}
    found = compare_launches(legacy, v2, call, out_legacy, out_v2, positions=positions)
    assert list(found) == [("stage1", 0)]
    assert positions["stage1", 0].shape == (3, 8)
    assert positions["stage1", 0].sum() == 8 and positions["stage1", 0][2].all()
    # A per-expert tensor with as many rows as tokens is compared whole.
    scale = torch.zeros(4, 8)
    scale[2] = 1
    v2, out_v2 = launch(a1, scale)
    found = compare_launches(legacy, v2, call, out_legacy, out_v2)
    assert list(found) == [("stage1", "w1_scale")]
    # Compared by bits: -0.0 is not 0.0, and equal NaNs are equal.
    v2, out_v2 = launch(a1, torch.zeros(4, 8).neg())
    found = compare_launches(legacy, v2, call, out_legacy, out_v2)
    assert list(found) == [("stage1", "w1_scale")]
    nan = torch.zeros(4, 8)
    nan[0, 0] = float("nan")
    legacy, out_legacy = launch(a1, nan)
    v2, out_v2 = launch(a1, nan.clone())
    assert compare_launches(legacy, v2, call, out_legacy, out_v2) == {}
    # The output buffer is skipped only where both sides pass theirs.
    found = compare_launches(legacy, v2, call, torch.zeros(1), out_v2)
    assert "only one side passes its output buffer" in found["stage1", 6]


def test_compare_launches_skips_what_a_split_k_stage1_accumulates():
    import torch
    from moe_dispatch.differential import compare_launches

    from aiter import QuantType
    from aiter.moev2.segments.asm import asm_stage1
    from aiter.moev2.segments.ck import ck_moe_stage1
    from aiter.moev2.segments.flydsl import _flydsl_stage1_wrapper

    def cktile_stage1(*args, **kwargs):
        pass

    def stage2(*args, **kwargs):
        pass

    flydsl = "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3"

    token, topk = 2, 2
    call = {
        "topk_ids": torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        "hidden_states": torch.zeros(token, 256, dtype=torch.bfloat16),
        "w1": torch.zeros(2, 16, 256),
        "w2": torch.zeros(2, 256, 8),
    }
    sorted_ids = torch.tensor(
        [0 | 0 << 24, 1 | 1 << 24, 0 | 1 << 24, 1 | 0 << 24], dtype=torch.int32
    )
    blocks, num_valid = torch.tensor([0, 1], dtype=torch.int32), torch.tensor([4])
    moe_out = torch.zeros(token, 256)

    def launch(func, splits, out, a2, a2_scale):
        args = (torch.zeros(token, 256, dtype=torch.uint8), call["w1"], call["w2"])
        args += (sorted_ids, blocks, num_valid)
        stage1 = functools.partial(func, *args, out, topk, block_m=2, **splits)
        args = (a2, call["w1"], call["w2"], sorted_ids, blocks, num_valid, moe_out)
        stage = functools.partial(stage2, *args, topk, a2_scale=a2_scale, block_m=2)
        return [("stage1", stage1), ("stage2", stage)]

    def moved(t):
        t = t.clone()
        t.view(-1)[0] += 1
        return t

    out = torch.zeros(token, topk, 8)
    a2, a2_scale = torch.zeros(token, topk, 8), torch.ones(token, 1)
    per_1x128, per_token = QuantType.per_1x128, QuantType.per_Token
    for func, splits, skipped in (
        (ck_moe_stage1, {"quant_type": per_1x128, "splitk": 2}, True),
        (ck_moe_stage1, {"quant_type": per_1x128, "splitk": 1}, False),
        (ck_moe_stage1, {"quant_type": per_token, "splitk": 2}, False),
        (asm_stage1, {"quant_type": per_1x128, "ksplit": 2}, True),
        (asm_stage1, {"quant_type": per_1x128, "ksplit": 0}, False),
        (cktile_stage1, {"split_k": 2}, True),
        (cktile_stage1, {"split_k": 1}, False),
        (_flydsl_stage1_wrapper, {"kernelName": f"{flydsl}_kb2_bnt0"}, True),
        (_flydsl_stage1_wrapper, {"kernelName": f"{flydsl}_bnt0"}, False),
    ):
        legacy = launch(func, splits, out, a2, a2_scale)
        for i, changed in enumerate(
            [(moved(out), a2, a2_scale), (out, moved(a2), a2_scale)]
            + [(out, a2, moved(a2_scale))]
        ):
            v2 = launch(func, splits, *changed)
            found = compare_launches(legacy, v2, call, torch.zeros(1), torch.zeros(1))
            assert (found == {}) == skipped, (func.__name__, splits, i)
        # Their dtypes and shapes are compared still, and stage 2's other arguments.
        for i, changed in enumerate(
            [(out.bfloat16(), a2, a2_scale), (out, a2[:1], a2_scale)]
            + [(out, a2, None)]
        ):
            v2 = launch(func, splits, *changed)
            found = compare_launches(legacy, v2, call, torch.zeros(1), torch.zeros(1))
            assert len(found) == 1, (func.__name__, splits, i)
        v2 = launch(func, splits, out.clone(), a2, a2_scale)
        found = compare_launches(legacy, v2, call, torch.zeros(1), v2[0][1].args[6])
        assert "only one side passes its output buffer" in found["stage1", 6]
        v2 = launch(func, splits, out, a2, a2_scale)
        name, stage = v2[1]
        args = stage.args[:2] + (moved(stage.args[2]),) + stage.args[3:]
        v2[1] = name, functools.partial(stage.func, *args, **stage.keywords)
        found = compare_launches(legacy, v2, call, torch.zeros(1), torch.zeros(1))
        assert list(found) == [("stage2", 2)]
        v2 = launch(func, splits, out, a2, a2_scale)
        v2[1] = ("stage2", functools.partial(v2[1][1], w2_scale=torch.ones(1)))
        legacy[1] = ("stage2", functools.partial(legacy[1][1], w2_scale=torch.zeros(1)))
        found = compare_launches(legacy, v2, call, torch.zeros(1), torch.zeros(1))
        assert list(found) == [("stage2", "w2_scale")]


def test_dirty_allocator_needs_the_default_allocator(monkeypatch):
    from moe_dispatch import differential

    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(RuntimeError, match="needs the default caching allocator"):
        differential.dirty_allocator()


def test_dirty_allocator_fills_the_freed_blocks_of_the_gpu(monkeypatch):
    import itertools

    import torch
    from moe_dispatch import differential
    from moe_dispatch.differential import fill_byte

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    sizes = (40 << 20, 5 << 20, 700 << 10, 3000)
    held = [torch.zeros(n, dtype=torch.uint8, device="cuda") for n in sizes]
    del held
    reserved = torch.cuda.memory_reserved()
    monkeypatch.setattr(differential, "_FILLS", itertools.count(7))
    differential.dirty_allocator()
    assert torch.cuda.memory_reserved() <= reserved + (256 << 20)
    reused = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in sizes]
    assert all(bool((t == fill_byte(7)).all()) for t in reused)


def test_compare_launches_reads_the_asm_1x128_scale_transposed():
    import torch
    from moe_dispatch.differential import compare_launches

    from aiter import QuantType
    from aiter.moev2.segments.asm import asm_stage1

    token, topk, groups = 4, 2, 2
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32)
    # Token 2 has no local expert (0 and 1 local).
    routed = [0 | 0 << 24, 3 | 1 << 24, 0 | 1 << 24, 1 | 0 << 24]
    call = {
        "topk_ids": topk_ids,
        "hidden_states": torch.zeros(token, 256, dtype=torch.bfloat16),
        "w1": torch.zeros(4, 16, 256),
        "w2": torch.zeros(4, 256, 8),
    }

    def launch(a1_scale, quant_type=QuantType.per_1x128):
        sorted_ids = torch.tensor(routed + [token] * 4, dtype=torch.int32)
        blocks = torch.tensor([0, 1], dtype=torch.int32)
        out = torch.zeros(token, topk, 8)
        a1 = torch.zeros(token, 256, dtype=torch.uint8)
        args = (a1, call["w1"], call["w2"], sorted_ids, blocks, torch.tensor([4]), out)
        stage = functools.partial(
            asm_stage1, *args, topk, block_m=2, quant_type=quant_type, a1_scale=a1_scale
        )
        return [("stage1", stage)], out

    # Shape [token, groups], memory [groups, token]: element g * token + t.
    scale = torch.arange(token * groups, dtype=torch.float32).reshape(token, groups)
    legacy, out_legacy = launch(scale)
    for flat, differs in ((4, True), (2, False), (6, False)):
        other = scale.clone()
        other.view(-1)[flat] = -1
        v2, out_v2 = launch(other)
        positions = {}
        found = compare_launches(
            legacy, v2, call, out_legacy, out_v2, positions=positions
        )
        assert (list(found) == [("stage1", "a1_scale")]) == differs, flat
        if differs:
            assert positions["stage1", "a1_scale"].shape == (groups, 3)
    # Row-major for another stage 1 quant: token 2 is row 2.
    legacy, out_legacy = launch(scale, QuantType.per_Token)
    other = scale.clone()
    other.view(-1)[4] = -1
    v2, out_v2 = launch(other, QuantType.per_Token)
    assert compare_launches(legacy, v2, call, out_legacy, out_v2) == {}


def test_gpu_differential_clears_the_dispatch_caches_of_every_builtin_path(monkeypatch):
    import importlib
    import pkgutil
    from types import SimpleNamespace

    from moe_dispatch import gpu_differential

    from aiter.moev2 import migration

    cleared = set()
    locals_ = []
    for module in pkgutil.iter_modules(migration.__path__):
        name = f"{migration.__name__}.{module.name}"
        local = getattr(importlib.import_module(name), "LOCAL", None)
        if local is not None:
            locals_.append(name)
            for attr in ("dispatch", "resolve", "relookup"):
                clear = functools.partial(cleared.add, (name, attr))
                monkeypatch.setattr(local, attr, SimpleNamespace(cache_clear=clear))
    gpu_differential._clear_decision_caches()
    assert "aiter.moev2.migration.a16w16" in locals_
    assert cleared == {
        (name, attr) for name in locals_ for attr in ("dispatch", "resolve", "relookup")
    }


def test_gpu_differential_maps_verdicts_to_its_exit_status(monkeypatch):
    from types import SimpleNamespace

    import torch
    from moe_dispatch import differential, gpu_differential
    from moe_dispatch.differential import KnownDecline, LegacyNotRepeatable

    import aiter.fused_moe as fm
    from aiter.jit.utils import chip_info

    def raising(exc):
        def run_differential(path_name, call):
            if exc is not None:
                raise exc

        return run_differential

    monkeypatch.setattr(gpu_differential, "_make_call", lambda path, case: {})
    monkeypatch.setattr(gpu_differential, "_clear_dispatch_caches", lambda: None)
    monkeypatch.setattr(gpu_differential, "_clear_decision_caches", lambda: None)
    monkeypatch.setattr(fm, "fused_moe", lambda **call: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    for exc, verdict in (
        (None, "ok"),
        (KnownDecline("listed"), "KNOWN"),
        (LegacyNotRepeatable("noisy"), "LEGACY"),
        (AssertionError("off"), "FAIL"),
        (ValueError("bug"), "FAIL"),
    ):
        monkeypatch.setattr(differential, "run_differential", raising(exc))
        assert gpu_differential._run_case("a16w16", "case")[0] == verdict, exc
    oom = torch.OutOfMemoryError("full")
    monkeypatch.setattr(differential, "run_differential", raising(oom))
    with pytest.raises(torch.OutOfMemoryError):
        gpu_differential._run_case("a16w16", "case")

    def run_case(path_name, verdict):
        if verdict == "SKIP":
            raise torch.OutOfMemoryError("full")
        return verdict, None

    monkeypatch.setattr(gpu_differential, "run_case", run_case)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    args = SimpleNamespace(path="a16w16")
    for verdicts, status in (
        (["ok", "KNOWN"], 0),
        (["ok", "LEGACY", "KNOWN"], 2),
        (["LEGACY", "FAIL"], 1),
        (["LEGACY", "SKIP"], 1),
    ):
        assert gpu_differential._child(args, verdicts) == status, verdicts
