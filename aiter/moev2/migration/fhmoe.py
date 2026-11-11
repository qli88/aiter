# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Path `fhmoe`: FHMoE, the shared expert computed as the last routed row (`aiter.fhmoe`).

Claims `fhmoe_` calls (the legacy `fused_moe` sends calls with shared-expert tensors
there) on gfx950 outside AITER_ONLINE_TUNE, except the interleaved MXFP4 calls legacy's
grouped gfx1250 branch takes. `run` follows `fhmoe_`: the contract checks, the
activation format from the gate layout (fp8 interleaved, fp4 separated), the dedicated
table for the DSV4 I384 contract and for MXFP8 geometries it declares (raising where
legacy raises), then the `_fused_moe_impl` dispatch with the FHMoE stages and the
selected block size enforced. Legacy runs only FlyDSL 2-stage plans and raises on any
other; v2 has no other plan for these keys, so it declines them and legacy raises.
"""

import os

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fhmoe_contract import FHMOE_BUFFER_MAX_BYTES
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from .. import paths
from ..dispatch import overrides, tuned, tuned_table
from ..segments.flydsl import (
    _flydsl_route_output_nbytes,
    _stage2_runtime_uses_route_reduce,
)
from ..segments.flydsl_fhmoe import (
    FlydslFhmoeGemm1,
    FlydslFhmoeGemm1Quant,
    FlydslFhmoeGemm2,
    _is_dsv4_i384_fhmoe_contract,
    _is_dsv4_i384_metadata,
    _is_fhmoe_metadata_compatible,
    _validate_fhmoe_contract,
    _validate_fhmoe_metadata_consistency,
)
from ..segments.mxfp4 import mxfp4, mxfp8
from ..segments.quant import QuantA1, QuantInter
from ..segments.sort import Sort
from ..shapes import get_padded_M
from . import local

NAME = "fhmoe"
ARCHS = ("gfx950",)
SEGMENTS = (
    Sort(),
    QuantA1(),
    FlydslFhmoeGemm1(),
    FlydslFhmoeGemm1Quant(),
    QuantInter(),
    FlydslFhmoeGemm2(),
)

# `fused_moe_` arguments `fhmoe_` passes `_fused_moe_impl` at their defaults.
_OP_DEFAULTS = {
    "beta": None,
    "linear_beta": None,
    "ep_source_token_map": None,
    "quant_type_a": None,
    "quant_dtype_a": None,
    "quant_dtype_a2": None,
}
# Problem features `_validate_fhmoe_contract` rejects before the dispatch.
_CONTRACT_EXCLUDES = frozenset({"ep_mask", "bias1", "bias2"})
# What the legacy capability checks treat as "not supported".
_PROBE_ERRORS = (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError)


def claims(call) -> bool:
    if overrides.online_tune():
        return False
    if not local.same_arch(ARCHS):
        return False
    return not (call["w1"].dtype == dtypes.fp4x2 and local.legacy_grouped_gfx1250(call))


def takes(problem) -> bool:
    """The dispatch keys this path runs: the key-level counterpart of `claims` and of
    the `fhmoe_` contract."""
    interleave = problem.gate_mode == GateMode.INTERLEAVE
    return (
        local.key_args_supported(
            problem,
            ARCHS,
            activations=("silu",),
            gate_modes=(GateMode.SEPARATED, GateMode.INTERLEAVE),
            act_params=True,
            shared="fused_row",
        )
        and not problem.features & _CONTRACT_EXCLUDES
        and problem.quant_type == QuantType.per_1x32
        and problem.w1 in (mxfp4(), mxfp8())
        and problem.act_in == (mxfp8() if interleave else mxfp4())
        and (interleave or problem.w1 == mxfp4())
    )


LOCAL = local.LocalDispatch(SEGMENTS, takes, fhmoe=True)


def _fhmoe_config_file() -> str:
    return tuned_table.FHMOE_TABLE


def _declares_mxfp8(model_dim, inter_dim, experts, topk, **contract) -> bool:
    """Whether the tuned table has an fp8-weight FHMoE key of this geometry, and with
    `contract` (dtype, hidden_pad, intermediate_pad, gate_mode, doweight_stage1) of the
    MXFP8/FP8 contract with those values: the legacy `_config_has_mxfp8_fhmoe_geometry`
    and `_config_has_mxfp8_fhmoe_contract` on the table."""
    want = {
        "model_dim": str(model_dim),
        "inter_dim": str(inter_dim),
        "expert": str(experts),
        "topk": str(topk),
        "shared_expert_id": str(experts - 1),
        "q_dtype_w": str(dtypes.fp8),
    }
    if contract:
        want |= {
            "act_type": str(ActivationType.Silu),
            "dtype": str(contract["dtype"]),
            "q_dtype_a": str(dtypes.fp8),
            "q_type": str(QuantType.per_1x32),
            "use_g1u1": str(True),
            "doweight_stage1": str(bool(contract["doweight_stage1"])),
            "hidden_pad": str(int(contract["hidden_pad"])),
            "intermediate_pad": str(int(contract["intermediate_pad"])),
            "gate_mode": str(contract["gate_mode"]),
        }
    return any(
        all(key[c] == v for c, v in want.items()) for key in tuned_table.fhmoe_keys()
    )


def _probe_metadata(
    token,
    model_dim,
    inter_dim,
    experts,
    topk,
    dtype,
    q_dtype_w,
    doweight_stage1,
    hidden_pad,
    intermediate_pad,
    gate_mode,
    config_file,
):
    """The FHMoE metadata of a probe key (the legacy `get_2stage_cfgs` arguments the
    capability checks pass); raises NotImplementedError where `_use_fhmoe_wrappers`
    does (no FHMoE plan, or stage tiles that differ from `block_m`)."""
    args = (
        token,
        model_dim,
        inter_dim,
        experts,
        topk,
        dtype,
        dtypes.fp8,
        q_dtype_w,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        True,
        gate_mode,
    )
    dispatched = LOCAL.dispatch(
        args, (("config_file", config_file),), get_gfx_runtime(), get_cu_num()
    )
    if dispatched is None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require the "
            "two-stage FlyDSL path"
        )
    metadata = dispatched[0].metadata
    _validate_fhmoe_metadata_consistency(metadata)
    return metadata


@tuned.cached
def _supports_dsv4_i384_fhmoe_config(max_tokens: int, config_file: str) -> bool:
    """`aiter.fhmoe._supports_dsv4_i384_fhmoe_config`, resolved by this path."""
    try:
        required_tokens = {
            get_padded_M(1 << exponent) for exponent in range(max_tokens.bit_length())
        }
        required_tokens.add(get_padded_M(max_tokens))
        for token in required_tokens:
            metadata = _probe_metadata(
                token,
                7168,
                384,
                385,
                7,
                torch.bfloat16,
                dtypes.fp4x2,
                False,
                0,
                0,
                GateMode.INTERLEAVE,
                config_file,
            )
            if not _is_dsv4_i384_metadata(metadata):
                return False
    except _PROBE_ERRORS:
        return False
    return True


def supports_dsv4_i384_fhmoe(max_tokens: int) -> bool:
    """`aiter.fhmoe.supports_dsv4_i384_fhmoe`, resolved by this path."""
    if type(max_tokens) is not int or max_tokens <= 0:
        return False
    try:
        if int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")) != 0:
            return False
        config_file = _fhmoe_config_file()
    except _PROBE_ERRORS:
        return False
    return _supports_dsv4_i384_fhmoe_config(max_tokens, config_file)


@tuned.cached
def _supports_mxfp8_fhmoe_config(
    max_tokens: int,
    config_file: str,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
    force_reduce: bool,
    stage2_fp8: bool,
) -> bool:
    """`aiter.fhmoe._supports_mxfp8_fhmoe_config`, resolved by this path."""
    try:
        max_bucket = get_padded_M(max_tokens)
        max_metadata = None
        required_tokens = {
            get_padded_M(1 << exponent) for exponent in range(max_tokens.bit_length())
        }
        required_tokens.add(max_bucket)
        for token in required_tokens:
            metadata = _probe_metadata(
                token,
                model_dim,
                inter_dim,
                experts,
                topk,
                dtype,
                dtypes.fp8,
                doweight_stage1,
                hidden_pad,
                intermediate_pad,
                gate_mode,
                config_file,
            )
            if not _is_fhmoe_metadata_compatible(
                metadata,
                model_dim=model_dim,
                dtype=dtype,
                q_dtype_a=dtypes.fp8,
                q_dtype_w=dtypes.fp8,
                gate_mode=gate_mode,
            ):
                return False
            if token == max_bucket:
                max_metadata = metadata
        if max_metadata is None:
            return False
        if _stage2_runtime_uses_route_reduce(
            max_metadata.stage2,
            max_tokens,
            model_dim,
            force_reduce=force_reduce,
            stage2_fp8=stage2_fp8,
        ):
            output_element_size = torch.empty((), dtype=dtype).element_size()
            if (
                _flydsl_route_output_nbytes(
                    max_metadata.stage2,
                    max_tokens,
                    topk,
                    model_dim,
                    output_element_size,
                    stage2_fp8=stage2_fp8,
                )
                > FHMOE_BUFFER_MAX_BYTES
            ):
                return False
    except _PROBE_ERRORS:
        return False
    return True


def supports_mxfp8_fhmoe_config(max_tokens: int, **request) -> bool:
    """`aiter.fhmoe.supports_mxfp8_fhmoe_config`, resolved by this path."""
    if type(max_tokens) is not int or max_tokens <= 0:
        return False
    try:
        if get_gfx_runtime() != "gfx950":
            return False
        if int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")) != 0:
            return False
        config_file = _fhmoe_config_file()
    except _PROBE_ERRORS:
        return False
    return _supports_mxfp8_fhmoe_config(
        max_tokens,
        config_file,
        **request,
        force_reduce=os.environ.get("AITER_FLYDSL_FORCE_REDUCE", "0") == "1",
        stage2_fp8=os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1",
    )


def _resolve_mxfp8_fhmoe_config_file(max_tokens: int, **request) -> str | None:
    """`aiter.fhmoe._resolve_mxfp8_fhmoe_config_file`, resolved by this path."""
    config_file = _fhmoe_config_file()
    geometry = {k: request[k] for k in ("model_dim", "inter_dim", "experts", "topk")}
    if not _declares_mxfp8(**geometry):
        return None
    if not _declares_mxfp8(**request):
        raise ValueError(
            "MXFP8/FP8 FHMoE request partially matches a dedicated CSV "
            "contract but differs in dtype, padding, gate mode, or "
            "doweight_stage1"
        )
    if not supports_mxfp8_fhmoe_config(max_tokens, **request):
        raise NotImplementedError(
            "The dedicated MXFP8/FP8 FHMoE config does not cover "
            f"this request at M={max_tokens}"
        )
    return config_file


def config_file(call, gate_mode):
    """The dedicated table `fhmoe_` dispatches the call with (None: the main table),
    raising where it raises."""
    hidden_states, w1, w2 = call["hidden_states"], call["w1"], call["w2"]
    experts = w1.shape[0]
    model_dim = w2.shape[1]
    inter_dim = w2.shape[2] * (model_dim // w1.shape[-1])
    topk = call["topk_ids"].shape[1]
    if _is_dsv4_i384_fhmoe_contract(
        model_dim,
        inter_dim,
        experts,
        topk,
        call["hidden_pad"],
        call["intermediate_pad"],
        gate_mode,
        call["doweight_stage1"],
        routed_mxfp4=w1.dtype == dtypes.fp4x2,
    ):
        if not supports_dsv4_i384_fhmoe(hidden_states.shape[0]):
            raise NotImplementedError(
                "The active FHMoE config does not cover this DSV4 I384 "
                f"token shape: M={hidden_states.shape[0]}"
            )
        return _fhmoe_config_file()
    if w1.dtype == dtypes.fp8:
        dtype = hidden_states.dtype if call["dtype"] is None else call["dtype"]
        return _resolve_mxfp8_fhmoe_config_file(
            hidden_states.shape[0],
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            dtype=dtype,
            hidden_pad=call["hidden_pad"],
            intermediate_pad=call["intermediate_pad"],
            gate_mode=gate_mode,
            doweight_stage1=call["doweight_stage1"],
        )
    return None


def run(call):
    gate_mode = GateMode(call["gate_mode"])
    w1 = call["w1"]
    _validate_fhmoe_contract(
        call["hidden_states"],
        w1,
        call["w2"],
        call["w1_scale"],
        call["w2_scale"],
        call["shared_w1"],
        call["shared_w2"],
        call["shared_w1_scale"],
        call["shared_w2_scale"],
        call["shared_expert_id"],
        ActivationType(call["activation"]),
        QuantType(call["quant_type"]),
        gate_mode,
        call["expert_mask"],
        call["bias1"],
        call["bias2"],
    )
    q_dtype_a = dtypes.fp8 if gate_mode == GateMode.INTERLEAVE else dtypes.fp4x2
    if not call["clamp_shared"] and w1.dtype != dtypes.fp8:
        raise ValueError(
            "clamp_shared=False requires MXFP8 routed weights; "
            "MXFP4/FP8 FHMoE keeps the shared expert clamped"
        )
    table = config_file(call, gate_mode)
    return LOCAL.run({**call, **_OP_DEFAULTS}, q_dtype_a=q_dtype_a, config_file=table)


# AITER_MOEV2=fhmoe until tests/moe_dispatch/gpu_differential.py passes on an arch.
PATH = paths.Path(
    NAME,
    claims,
    run,
    dispatch_record=LOCAL.dispatch_record,
    entry="fhmoe",
    segments=SEGMENTS,
    takes=takes,
)
paths.register(PATH)
