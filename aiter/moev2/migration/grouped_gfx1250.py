# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `grouped_gfx1250`: the gfx1250 grouped MoE (segment `grouped_gfx1250`).

Runs the `fused_moe_` calls on a gfx1250 device that `_fused_moe_impl` hands to
`grouped_gemm_gfx1250_a8w4` and that it serves instead of returning None. `claims`:
interleaved gate/up, the grouped switches on, both weight scales, gfx1250 named by the
build (`local.legacy_grouped_gfx1250`), no `stage2_scatter` or FlyDSL dispatch context
(`MegaMoEGfx1250`, step 4.6b) and no `quant_type_a` other than `quant_type` (legacy
raises). `run` follows `_fused_moe_impl` up to that call: the prologue checks and the
activation format; it declines what the function would not serve (not g1u1 per_1x32,
not Silu/Swiglu/SiTUv2, not fp4 or fp8 activations into fp4x2/uint8 weights, EP with
`doweight_stage1`), so legacy falls through to its 2-stage dispatch as before; then the
launch and the copy into `output`. On other devices the build variables can name
gfx1250 too (todo 5.26); those calls stay legacy.
"""

from aiter import QuantType
from aiter.jit.utils.chip_info import get_gfx_runtime

from .. import paths
from ..execute import executor
from ..execute.output import _return_output
from ..policy import quant_remap
from ..segments.grouped_gfx1250 import GROUPED, dispatch_context
from . import local

NAME = "grouped_gfx1250"


def claims(call) -> bool:
    if get_gfx_runtime() != "gfx1250" or not local.legacy_grouped_gfx1250(call):
        return False
    if call["ep_source_token_map"] is not None:
        return False
    if dispatch_context() is not None:
        return False
    quant_type_a = call["quant_type_a"]
    return quant_type_a is None or QuantType(quant_type_a) == QuantType(
        call["quant_type"]
    )


def run(call):
    c = executor.prologue(call, quant_remap)
    q_dtype_a = local.activation_dtype(call)
    if GROUPED.unsupported(c, q_dtype_a) is not None:
        return paths.DECLINED
    out = GROUPED.launch(c, q_dtype_a)
    if out is None:
        # Before any launch: legacy reruns the checks and falls through.
        return paths.DECLINED
    return _return_output(out, c["output"])


# AITER_MOEV2=grouped_gfx1250 until its GPU differential passes on gfx1250.
PATH = paths.Path(NAME, claims, run, entry="op")
paths.register(PATH)
