# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug overrides of dispatch decisions, read where the legacy code reads them.

Each is read when a dispatch decision is made (not cached here): decisions are cached per
problem, so this matches the legacy reads inside the cached `get_2stage_cfgs`.
"""

import os


def bypass_tuned() -> int:
    """`AITER_BYPASS_TUNE_CONFIG`: ignore tuned rows; use the default heuristics."""
    return int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0"))


def ksplit_disabled() -> bool:
    """`AITER_KSPLIT=-1`: run tuned rows with ksplit 0."""
    return int(os.environ.get("AITER_KSPLIT", "0")) == -1


def nt_override() -> int:
    """`AITER_USE_NT`: -1 (default) keeps the tuned `nt` column, 0/1 forces it."""
    return int(os.environ.get("AITER_USE_NT", "-1"))


def xbf16() -> bool:
    """`AITER_XBFLOAT16=1`: the gfx950 1-stage fp8 blockscale kernel's bf16 variant."""
    return int(os.environ.get("AITER_XBFLOAT16", "0")) == 1


def online_tune() -> bool:
    """`AITER_ONLINE_TUNE=1`: tune missing keys on the fly (legacy code only, for now)."""
    return os.environ.get("AITER_ONLINE_TUNE", "0") == "1"


def explain() -> bool:
    """`AITER_MOE_EXPLAIN=1`: log how each new problem is resolved."""
    return os.environ.get("AITER_MOE_EXPLAIN", "0") == "1"


def flydsl_force() -> bool:
    """`AITER_FLYDSL_FORCE` (default 1): MXFP4 defaults to FlyDSL for every activation,
    not only Swiglu and SiTUv2."""
    return os.environ.get("AITER_FLYDSL_FORCE", "1") == "1"


def a4w4_fp8_inter() -> bool:
    """`AITER_SITUV2_A4W4_FP8_INTER=1`: fp8 intermediate for tuned a4w4 FlyDSL rows
    with a layout GEMM2 (when the call sets no `quant_dtype_a2`)."""
    return os.environ.get("AITER_SITUV2_A4W4_FP8_INTER", "0") == "1"


def mxmoe_fallback() -> bool:
    """`AITER_MXMOE_FALLBACK` (default 1): MXFP4 SiTUv2 without a tuned row defaults to
    the heuristic MXMOE kernels."""
    return os.environ.get("AITER_MXMOE_FALLBACK", "1") == "1"
