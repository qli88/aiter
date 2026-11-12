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
    from moe_dispatch.differential import _check_output

    nan, inf = float("nan"), float("inf")
    ref = torch.tensor([1.0, nan, -inf, 2.0], dtype=torch.bfloat16)
    ref2 = torch.tensor([1.0078125, nan, -inf, 2.0], dtype=torch.bfloat16)
    _check_output("p", ref, ref2, ref.clone())
    for out, says in (
        ([1.0, 0.0, -inf, 2.0], "non-finite outputs differ from legacy"),
        ([1.0, nan, inf, 2.0], "non-finite outputs differ from legacy"),
        ([1.5, nan, -inf, 2.0], "max |v2 - legacy|"),
    ):
        with pytest.raises(AssertionError, match=re.escape(says)):
            _check_output("p", ref, ref2, torch.tensor(out, dtype=torch.bfloat16))
    moved = torch.tensor([nan, 1.0, -inf, 2.0], dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="reference output not reproducible"):
        _check_output("p", ref, moved, ref.clone(), ("reference", "candidate"))
