# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`MOEMetadata`: the stage callables and flags one dispatch decision produces."""

from collections.abc import Callable
from dataclasses import dataclass

from aiter.fused_moe_registry import BoundFusedMoeImpl


@dataclass
class MOEMetadata:
    stage1: Callable | None
    stage2: Callable | None
    block_m: int
    ksplit: int
    run_1stage: bool = False
    has_bias: bool = False
    use_non_temporal_load: bool = True
    fuse_quant: str = ""
    stage2_has_bias: bool = False
    flat: bool = False
    output_aux: bool | str = False
    prequant: bool = True
    skip_inter_quant: bool = False
    route_bucket: str = ""
    expected_sorted_blocks: int | None = None
    min_sorted_blocks: int | None = None
    max_sorted_blocks: int | None = None
    full_impl: BoundFusedMoeImpl | None = None
