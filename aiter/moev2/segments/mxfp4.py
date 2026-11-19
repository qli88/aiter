# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segments of the MXFP4 (`per_1x32` fp4) FlyDSL kernels: FlyDSL 2-stage and MXMOE.

Which buffers a stage writes depends on its kernel name (a fused fp4 output, the v2
sorted layout of a layout GEMM2, the MXMOE layout), so each kind is its own segment and
the boundary check keeps the legacy pairings: a FlyDSL stage 1 is followed by
`quant.inter` (`flydsl.gemm1`) or writes the fp4 intermediate itself
(`flydsl.gemm1_q`, `flydsl.gemm1_layout`); a layout GEMM2 with the scatter epilog and
MXMOE read the auxiliary sort (`sort.aux`).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import functools

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode, get_flydsl_activation_name
from aiter.ops.flydsl.mxfp4_kname import (
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    native_scale_layout_for,
    parse_flydsl_v2_gemm2_kernel,
)

from ..dispatch import kernel_names, overrides
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec, scale_spec
from .base import Cap, DefaultKernels, RowRejection, Segment
from .defaults import BLOCK_SIZE_M
from .flydsl import (
    _flydsl_stage1_wrapper,
    _flydsl_stage2_wrapper,
    _flydsl_v2_stage2_wrapper,
)
from .mxmoe import (
    _is_inline_sort_cfg,
    _is_mxfp4_inline_sort,
    _make_mxfp4_metadata,
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_a4w4_stage2_fw,
    _mxmoe_gemm2_sort_block_m,
)
from .sort import AUX_SORT_OPUS


def mxfp4():
    """fp4 values with one E8M0 scale per 32 (`QuantType.per_1x32`)."""
    return scale_spec(QuantType.per_1x32, dtypes.fp4x2)


def a4w4_caps():
    return (
        Cap(
            arch=frozenset({"gfx950"}),
            act_in=frozenset({mxfp4()}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "gelu"}),
            gate_mode=frozenset({"separated"}),
            g1u1=frozenset({True}),
            features=frozenset({"ep_mask"}),
            extras=frozenset({"disable_inline_sort", "exclude_mxmoe"}),
        ),
    )


def _row_names(row):
    return (
        str(row.get("kernelName1", "") or "").strip(),
        str(row.get("kernelName2", "") or "").strip(),
    )


def mxfp4_row_rejection(p, row) -> RowRejection | None:
    """The legacy `get_2stage_cfgs` checks, in its order, that discard a tuned row with a
    FlyDSL or MXMOE kernel. (Its Opus, CK-Tile and post-reduction checks concern rows no
    segment here runs.)"""
    kn1, kn2 = _row_names(row)
    activation = p.activation.activation
    inline = _is_inline_sort_cfg(kn1, kn2)
    if (kn1.startswith("flydsl_") or kn2.startswith("flydsl_")) and activation not in (
        ActivationType.Silu,
        ActivationType.Swiglu,
        ActivationType.Situv2,
    ):
        return RowRejection(
            f"[fused_moe] discarding FlyDSL tuned config for unsupported "
            f"activation {activation}; using default heuristics"
        )
    if p.extras.get("disable_inline_sort") and inline:
        return RowRejection("[fused_moe] discarding tuned inline-sort config")
    if inline:
        inline_metadata = _make_mxfp4_metadata(
            kn1,
            kn2,
            p.gate_mode,
            row.get("ksplit", 0),
            block_m=row.get("block_m", BLOCK_SIZE_M),
            run_1stage=row.get("run_1stage", False),
            has_stage1_bias="bias1" in p.features,
            has_stage2_bias="bias2" in p.features,
        )
        if not _is_mxfp4_inline_sort(inline_metadata):
            return RowRejection(
                "[fused_moe] discarding tuned inline-sort config with "
                "incomplete metadata"
            )
    if not _is_mxfp4_kname(kn1):
        return None
    s = p.shape
    parsed_g1 = _parse_mxfp4_g1_kname(kn1)
    configured_act = parsed_g1["act"]
    try:
        expected_act = get_flydsl_activation_name(activation)
    except ValueError:
        expected_act = None
    swiglu_limit = p.activation.swiglu_limit
    reason = None
    if expected_act is None:
        reason = f"no MXMOE kernel for activation {activation!r}"
    elif configured_act != expected_act:
        reason = (
            f"activation {configured_act!r} does not match runtime {expected_act!r}"
        )
    elif swiglu_limit and expected_act not in ("silu", "swiglu"):
        reason = (
            f"MXMOE cannot apply swiglu_limit={swiglu_limit!r} to "
            f"activation {expected_act!r}"
        )
    elif p.extras.get("exclude_mxmoe"):
        reason = "excluded for this invocation"
    elif parsed_g1["inline_quant"] and p.in_dtype not in (None, dtypes.bf16):
        reason = f"inline GEMM1 reads BF16 and cannot accept {p.in_dtype} input"
    elif not (
        int(row.get("block_m", BLOCK_SIZE_M))
        == parsed_g1["BM"]
        == _mxmoe_gemm2_sort_block_m(kn2, parsed_g1["BM"])
    ):
        reason = (
            f"sort block_m={row.get('block_m', BLOCK_SIZE_M)} and GEMM2 {kn2!r} "
            f"must both match GEMM1 BM={parsed_g1['BM']}"
        )
    # BM16 makes the masked Opus aux sort use dispatch_policy=1, which hangs at >= 819
    # experts (#8014).
    elif p.is_ep and parsed_g1["BM"] == 16:
        reason = "BM16 MXMOE is not supported under EP yet"
    elif p.is_ep and activation != ActivationType.Situv2:
        reason = f"MXMOE under EP is only validated for SiTUv2, got {activation}"
    elif p.is_ep and parse_flydsl_v2_gemm2_kernel(kn2) is None:
        reason = f"EP requires a flydsl_moe2_layout_ GEMM2, got {kn2!r}"
    elif p.is_ep and parse_flydsl_v2_gemm2_kernel(kn2)["epilog"] == "scatter":
        reason = "the scatter GEMM2 epilog is not validated under EP"
    elif not p.is_ep and not aiter.is_mxfp4_moe_shape_supported(
        s.experts, s.model_dim, s.inter_dim, s.topk
    ):
        reason = (
            "generated MXFP4 auxiliary kernels do not cover "
            f"(expert={s.experts}, model_dim={s.model_dim}, "
            f"inter_dim={s.inter_dim}, topk={s.topk})"
        )
    elif "bias2" in p.features and parse_flydsl_v2_gemm2_kernel(kn2) is None:
        reason = f"stage2 bias requires a flydsl_moe2_layout_ kernel, got {kn2!r}"
    if reason is None:
        return None
    return RowRejection(
        f"[fused_moe] discarding MXMOE config ({kn1!r}, {kn2!r}): {reason}; "
        f"using default heuristics"
    )


def flydsl_default_kernels(p) -> DefaultKernels:
    """The legacy heuristic FlyDSL fallback (`use_mxfp4_flydsl` without a tuned row)."""
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_kernel_name,
        get_flydsl_kernel_params,
        pick_flydsl_stage2_tile_k,
    )

    token = p.shape.m
    out_type = "bf16" if p.out_dtype == dtypes.bf16 else "f16"
    a_type = "fp4" if p.act_in.dtype == dtypes.fp4x2 else "fp8"
    w_type = "fp8" if p.w1.dtype == dtypes.fp8 else "fp4"
    s2_tk = pick_flydsl_stage2_tile_k(p.shape.inter_dim)
    # Per token tier: (tile_m, stage1 suffix, stage2 suffix).
    if token < 2048:
        tile_m, s1_sfx, s2_sfx = 32, "_w2", "_bnt2"
    elif token < 4096:
        tile_m, s1_sfx, s2_sfx = 64, "_w3_bnt0", ""
    elif token < 16384:
        tile_m, s1_sfx, s2_sfx = 128, "_w2_bnt0", ""
    else:
        tile_m, s1_sfx, s2_sfx = 64, "_w4_bnt0", ""
    base_kn1 = flydsl_kernel_name(1, a_type, w_type, out_type, tile_m, 128, 256)
    base_kn2 = flydsl_kernel_name(
        2, a_type, w_type, out_type, tile_m, 128, s2_tk, "atomic"
    )
    kn1 = f"{base_kn1}{s1_sfx}"
    kn2 = f"{base_kn2}{s2_sfx}"
    # fp8 stage1 kernel names always carry a "_gui" suffix.
    if a_type == "fp8":
        kn1 = f"{kn1}_gui"
        base_kn1 = f"{base_kn1}_gui"
    if get_flydsl_kernel_params(kn1) is None:
        kn1 = base_kn1
    if get_flydsl_kernel_params(kn2) is None:
        kn2 = base_kn2
    warning = (
        f"[fused_moe] no tuned FlyDSL config for {p.log_keys()}, "
        f"using heuristic FlyDSL fallback ({kn1=}, {kn2=})"
    )
    return DefaultKernels(kn1, kn2, tile_m, -1, warning)


def _stage1_fused_quant(kernel_name) -> str:
    """What a FlyDSL stage 1 quantizes its output to, from its name ("" for none)."""
    options = kernel_name.split("_t")[-1]
    return "fp8" if "_fp8" in options else ("fp4" if "_fp4" in options else "")


class _FlydslStage1(Segment):
    prefixes = (kernel_names.FLYDSL,)

    @property
    def capabilities(self):
        return a4w4_caps()

    def reads(self, p):
        return (Buf("sorted", None), Buf("tokens", p.act_in))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def build(self, p, params):
        return functools.partial(
            _flydsl_stage1_wrapper,
            kernelName=params.kernel,
            activation=p.activation.activation,
            inter_dim_pad=p.layout.intermediate_pad,
            model_dim_pad=p.layout.hidden_pad,
        )


class FlydslGemm1(_FlydslStage1):
    """A FlyDSL stage 1 writing the unquantized intermediate."""

    name = "flydsl.gemm1"
    covers = (Op.GEMM1, Op.ACT)

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 1
            and _stage1_fused_quant(kernel.format()) == ""
        )

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(p.out_dtype))

    def default_kernels(self, p):
        return flydsl_default_kernels(p)


class FlydslGemm1Quant(_FlydslStage1):
    """A FlyDSL stage 1 that quantizes its output to MXFP4 (`_fp4` in its options)."""

    name = "flydsl.gemm1_q"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 1
            and _stage1_fused_quant(kernel.format()) == "fp4"
        )

    def produces(self, p, params):
        return Buf("inter", mxfp4())

    def metadata_fields(self, p, params):
        return {"fuse_quant": "fp4"}


class FlydslGemm1Layout(_FlydslStage1):
    """A FlyDSL stage 1 writing the sorted, quantized input of a layout GEMM2."""

    name = "flydsl.gemm1_layout"
    covers = (Op.GEMM1, Op.QUANT_INTER)

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 1
        )

    def reads(self, p):
        return (Buf("sorted", None, None), Buf("tokens", p.act_in))

    def produces(self, p, params):
        # fp4 or fp8, as the GEMM2 name says.
        return Buf("inter", None, "v2_sorted")

    def build(self, p, params):
        stage1 = super().build(p, params)
        stage1.keywords["out_dtype"] = self._a_dtype(params)
        return stage1

    def metadata_fields(self, p, params):
        return {"fuse_quant": self._a_dtype(params)}

    @staticmethod
    def _a_dtype(params):
        return parse_flydsl_v2_gemm2_kernel(params.kernel2)["a_dtype"]


class FlydslGemm2(Segment):
    name = "flydsl.gemm2"
    prefixes = (kernel_names.FLYDSL,)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return a4w4_caps()

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.FLYDSL
            and kernel.stage == 2
        )

    def reads(self, p):
        return (Buf("sorted", None), Buf("inter", p.act_in))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def build(self, p, params):
        return functools.partial(
            _flydsl_stage2_wrapper,
            kernelName=params.kernel,
            inter_dim_pad=p.layout.intermediate_pad,
            model_dim_pad=p.layout.hidden_pad,
        )


class FlydslLayoutGemm2(Segment):
    """A FlyDSL v2 GEMM2 (`flydsl_moe2_layout_*`) with the atomic or reduce epilog."""

    name = "flydsl_layout.gemm2"
    prefixes = (kernel_names.FLYDSL_LAYOUT,)
    covers = (Op.GEMM2, Op.REDUCE)
    scatter = False

    @property
    def capabilities(self):
        return a4w4_caps()

    def owns(self, kernel) -> bool:
        if kernel is None or kernel.family != kernel_names.FLYDSL_LAYOUT:
            return False
        cfg = parse_flydsl_v2_gemm2_kernel(kernel.format())
        return cfg is not None and (cfg["epilog"] == "scatter") == self.scatter

    def reads(self, p):
        return (Buf("sorted", None), Buf("inter", None, "v2_sorted"))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def stage2_kernel(self, p, kernel_name1, kernel_name2):
        """An fp8 GEMM2 input (`q_dtype_a2`, else AITER_SITUV2_A4W4_FP8_INTER) renames
        an fp4 layout GEMM2 to its fp8 twin."""
        fp8_inter = (
            p.inter.dtype == dtypes.fp8
            if p.inter is not None
            else overrides.a4w4_fp8_inter()
        )
        if (
            fp8_inter
            and p.act_in.dtype == dtypes.fp4x2
            and kernel_name2.startswith("flydsl_moe2_layout_afp4_")
            and kernel_name1.startswith("flydsl_moe1_")
        ):
            return kernel_name2.replace("_afp4_", "_afp8_", 1)
        return kernel_name2

    def build(self, p, params):
        s = p.shape
        return functools.partial(
            _flydsl_v2_stage2_wrapper,
            kernelName=params.kernel,
            model_dim=s.model_dim,
            inter_dim=s.inter_dim,
            num_experts=s.experts,
        )

    def metadata_fields(self, p, params):
        return {
            "skip_inter_quant": True,
            "output_aux": AUX_SORT_OPUS if self.scatter else False,
        }


class FlydslLayoutGemm2Scatter(FlydslLayoutGemm2):
    """A FlyDSL v2 GEMM2 with the scatter epilog: it reads the auxiliary sort."""

    name = "flydsl_layout.gemm2_scatter"
    scatter = True

    def reads(self, p):
        return (Buf("sorted", None, "aux_sorted"), Buf("inter", None, "v2_sorted"))


class MxmoeGemm1(Segment):
    """MXMOE stage 1 (`flydsl_mxmoe_g1_*`): quantizes its input unless handed fp4,
    and writes its own sorted, quantized intermediate."""

    name = "mxmoe.gemm1"
    prefixes = (kernel_names.MXMOE,)
    covers = (Op.GEMM1, Op.QUANT_INTER)

    @property
    def capabilities(self):
        return a4w4_caps()

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.MXMOE
            and kernel.stage == 1
        )

    def reads(self, p):
        return (Buf("sorted", None, "aux_sorted"), Buf("tokens", None))

    def produces(self, p, params):
        return Buf("inter", None, "mxmoe_sorted")

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def build(self, p, params):
        p1 = _parse_mxfp4_g1_kname(params.kernel)
        return functools.partial(
            _mxfp4_a4w4_stage1_fw,
            kernelName1=params.kernel,
            interleave=(p.gate_mode == GateMode.INTERLEAVE),
            native_scale_layout=native_scale_layout_for(
                p1["BM"], p1["out_dtype"], params.kernel2
            ),
        )

    def metadata_fields(self, p, params):
        p1 = _parse_mxfp4_g1_kname(params.kernel)
        return {
            "block_m": p1["BM"],
            "fuse_quant": p1["out_dtype"],
            "output_aux": AUX_SORT_OPUS,
            "prequant": p1["a_dtype"] == "fp8" and not p1["inline_quant"],
        }


class MxmoeGemm2(Segment):
    """MXMOE stage 2: a native `flydsl_mxmoe_g2_*` kernel or a layout GEMM2."""

    name = "mxmoe.gemm2"
    prefixes = (kernel_names.MXMOE, kernel_names.FLYDSL_LAYOUT)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return a4w4_caps()

    def owns(self, kernel) -> bool:
        if kernel is None or kernel.stage != 2:
            return False
        if kernel.family == kernel_names.MXMOE:
            return True
        return (
            kernel.family == kernel_names.FLYDSL_LAYOUT
            and parse_flydsl_v2_gemm2_kernel(kernel.format()) is not None
        )

    def reads(self, p):
        return (Buf("sorted", None, "aux_sorted"), Buf("inter", None, "mxmoe_sorted"))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        return mxfp4_row_rejection(p, row)

    def build(self, p, params):
        return functools.partial(_mxfp4_a4w4_stage2_fw, kernelName2=params.kernel)
