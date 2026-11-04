# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segment `mega_moe_v2`: `MegaMoEV2` (gfx950, EP), the whole layer from dispatch to
combine in one segment.

The kernels stay in `aiter.ops.flydsl.kernels.mega_moe`: one `MegaMoEV2` instance (its
MORI symmetric-heap buffers, AOT bundles and bundle plan) per structure, owned by the
segment, which adopts it or creates it at `prepare`. Structure: `dispatch` (fixed-slot or
compact, from the bundle plan of `capacity`), `capacity` (`max_tok_per_rank`, which
sizes the heap) and `p2p_quant` (GEMM2's payload to the peers; fp8 is lossy). The bundle
entry (the stage 1 and stage 2 tiles) is picked per call from this rank's own tokens
(`rank_local_params`): the wire does not depend on it.

The methods are `MegaMoEV2.forward` and `forward_prequant` step for step, called in
their order by path `mega_moe_v2` (`migration.mega_moe_v2.forward`): `check`, then
`quantize` (fixed-slot) or `select` and `prepare_compact` (compact: the count exchange
with the quant fused in), then `select` if not yet, `stage1` (dispatch, GEMM1) and
`stage2` (GEMM2 into the peers' combine buffers, then the combine).
"""

from dataclasses import dataclass

import torch

from aiter import dtypes

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import FrozenMap, ScaleSpec
from .base import Cap, MemoryNeed, Params, Segment
from .mxfp4 import mxfp4


def _mega():
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_v2

    return mega_moe_v2


def _config():
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config

    return mega_moe_config


@dataclass(frozen=True)
class MegaV2Spec:
    """A `MegaMoEV2` instance: its constructor arguments besides the rank and the
    weights (`world_size` is checked against the engine's group)."""

    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    quant: str
    max_tok_per_rank: int
    mega_scheme: str = "fixedslot"
    stage2_p2p_quant: str = "auto"
    swiglu_limit: float = 0.0
    fanout_masks: tuple = ()
    act: str = "silu"
    situ_beta: float = 1.0
    situ_linear_beta: float = 1.0

    def bundle_plan(self):
        """The instance's bundle plan, as its constructor builds it."""
        return _config().build_mega_moe_bundle_plan(
            self.max_tok_per_rank,
            a_dtype="fp4" if self.quant == "a4w4" else "fp8",
            p2p_quant=self.stage2_p2p_quant,
            experts_per_rank=self.experts // self.world_size,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            world_size=self.world_size,
        )


def _num_cu() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def instance_need(spec: MegaV2Spec, weights, num_cu: int) -> MemoryNeed:
    """The bytes `MegaMoEV2(...)` of this spec allocates and holds, as its constructor
    (and its `FlyDSLDispatchCombineIntraNodeOp` and group-major op) does: MORI
    symmetric-heap tensors (`symmetric`, their bytes without the heap's alignment) and
    device tensors (`device`, with the copies of weights that are not contiguous, which
    the first `bind` frees). `num_cu`: the device's CU count."""
    from aiter.ops.flydsl.kernels import flydsl_dispatch_combine_intranode_op as op_mod
    from aiter.ops.flydsl.kernels.mega_moe import dispatch as disp

    config = _config()
    plan = spec.bundle_plan()
    W, H, I, K = spec.world_size, spec.model_dim, spec.inter_dim, spec.topk
    epr, mtpr = spec.experts // W, spec.max_tok_per_rank
    fp4 = spec.quant == "a4w4"
    compact = not plan.fixed_slot_dispatch
    unit = 128 if compact else 32
    scale_dim = H // 32
    indexed = compact and mtpr >= config.INDEXED_PAYLOAD_MIN_MTPR
    a_dtype = torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn
    cfg = op_mod.FlyDSLDispatchCombineConfig(
        rank=0,
        world_size=W,
        hidden_dim=H,
        max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr,
        num_experts_per_token=K,
        combine_dtype=torch.bfloat16,
        dispatch_dtype=a_dtype,
        scale_dim=scale_dim,
        scale_type_size=1,
        enable_std_moe=False,
        enable_group_major=True,
        gm_unit_size=unit,
        gm_scheme=spec.mega_scheme,
        gm_compact=compact,
        gm_indexed_payload=indexed,
        max_total_recv_tokens=W,
    )
    sym = dev = 0

    # FlyDSLDispatchCombineIntraNodeOp: _alloc_buffers, the P2P tables, the flag.
    mt, mr, mr_worst = mtpr, cfg.effective_max_recv, cfg.max_recv
    legacy_tb = H * cfg.max_token_type_size if cfg.max_token_type_size > 0 else 0
    disp_tb = max(cfg.dispatch_token_bytes, legacy_tb)
    comb_tb = max(cfg.combine_token_bytes, legacy_tb)
    sym += 2 * ((mr * disp_tb + 1) // 2) + mr * K * 8
    sym += mr * cfg.scale_bytes if cfg.scale_bytes > 0 else 1
    sym += 4 + W * 4 + mr * 4
    sym += 2 * ((max(mr_worst, mt * K) * comb_tb + 1) // 2) + mr_worst * K * 4
    sym += 2 * ((mt * comb_tb + 1) // 2) + mt * K * 4 + W * 8
    dev += W * 4 + 3 * 4 + mt * K * 4 + 4 + 4 + 8 + 8
    dev += 10 * W * 8 + 8

    # FlyDSLDispatchGroupMajorOp (its total_recv is replaced by the op's).
    ll_cap = -(-W * mtpr // unit) * unit
    scale_n_i32 = (scale_dim + 3) // 4 if scale_dim > 0 else 0
    row_bytes = op_mod._token_bytes_for(a_dtype, H)
    if compact:
        nvm = W * mtpr * K + (epr + 2) * unit
    else:
        nvm = epr * ll_cap + 256
    max_blocks = -(-nvm // unit)
    payload_rows = W * mtpr + 1 if indexed else nvm
    sym += W * 4 + epr * 8 + payload_rows * row_bytes
    sym += max(1, payload_rows * scale_n_i32) * 4 + nvm * 12 + W * 4
    dev += 8 + 8 + W * 4
    tables = 8
    if compact:
        sym += epr * 8 + W * 8 + W * W * epr * 4 + W * 4
        dev += 8 + 4 + 3 * W * epr * 4
        tables += 6
    dev += tables * W * 8

    # MegaMoEV2: _build_fused_stage1 (metadata, workspace, buffers, the dispatch table).
    metadata_blocks = -(-nvm // 32)
    dev += max(max_blocks, metadata_blocks) * 8 + metadata_blocks * 4 + 4 + 8
    prepare_cu = max(entry.config.stage1.num_dispatch_cu for entry in plan.entries)
    keys = (prepare_cu + 1) * (num_cu + 1)
    slots = config.fixed_stage1_epoch_slot_count(num_cu)
    experts = W * epr
    segments = experts + W
    tile_blocks = 2 * metadata_blocks
    dev += segments * 4 * (3 + num_cu) + mtpr * K * 8 + 8 + slots * 12
    dev += keys * 16 + 8 + 128 * 4 + 4 + epr * 4 + 4 + W * 8 + 2 * W * 4
    sym += W * segments * 4 + 2 * W * 4 + experts * 8 + 2 * W * 4 + W * 4
    sym += 4 * tile_blocks * 4 + 8 + 4
    dev += 10 * W * 8
    dev += nvm * (I // 2 if fp4 else I)
    dev += -(-nvm // 256) * 256 * (-(-(I // 32) // 8) * 8) + I
    dev += mtpr * (H // 2 if fp4 else H) + mtpr * scale_dim
    dev += disp.DISPATCH_TABLE_SIZE * 8

    # _build_fused_stage2: the combine placeholder.
    dev += H * 2

    dev += sum(
        t.nbytes
        for t in (weights.w1, weights.w1_scale, weights.w2, weights.w2_scale)
        if not t.is_contiguous()
    )
    return MemoryNeed(device=dev, symmetric=sym)


def _structure(plan) -> Params:
    p2p = sorted({entry.config.p2p_quant for entry in plan.entries})
    return Params(
        dispatch="fixed_slot" if plan.fixed_slot_dispatch else "compact",
        capacity=plan.mtpr,
        p2p_quant=",".join(p2p),
    )


class MegaMoEV2Segment(Segment):
    """A `MegaMoEV2` instance, which this segment owns: `adopt(mega)` wraps one (nothing
    to prepare), `for_spec(spec, weights)` creates one at `prepare(ctx)` on the engine's
    group with one layer's weights (the constructor needs them; each forward binds its
    own). Legacy has no health, reset or close, so neither has the segment; `release()`
    drops the instance. `workspace_bytes()`: what the constructor will allocate
    (`instance_need`); nothing for an adopted one (the hook's)."""

    name = "mega_moe_v2"
    covers = (Op.COMM_IN, Op.COMM_OUT)
    capabilities = (
        Cap(
            arch=frozenset({"gfx950"}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16}),
            features=frozenset({"invalid_slots"}),
            shared=True,
            parallel=True,
        ),
    )
    structure_params = frozenset({"dispatch", "capacity", "p2p_quant"})
    rank_local_params = True
    stateful = True
    output_lifetime = "until_next_forward"
    lossy_values = FrozenMap({"p2p_quant": frozenset({"fp8_blockwise_1x32"})})
    default_lossy = FrozenMap({"p2p_quant": frozenset({"fp8_blockwise_1x32"})})

    def __init__(self, mega=None, spec: MegaV2Spec | None = None, weights=None):
        self.mega = mega
        self.spec = spec
        self.weights = weights
        self.rank = None if mega is None else mega.rank
        self._preloaded = False

    @classmethod
    def adopt(cls, mega) -> "MegaMoEV2Segment":
        return cls(mega=mega)

    @classmethod
    def for_spec(cls, spec: MegaV2Spec, weights) -> "MegaMoEV2Segment":
        return cls(spec=spec, weights=weights)

    def params(self) -> Params:
        """The structure the instance was (or will be) built with."""
        if self.mega is not None:
            return _structure(self.mega._bundle_plan)
        return _structure(self.spec.bundle_plan())

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        return Buf("out", ScaleSpec(dtypes.bf16))

    def build(self, p, params):
        return self

    def workspace_bytes(self) -> MemoryNeed:
        if self.spec is None:
            return MemoryNeed()
        return instance_need(self.spec, self.weights, _num_cu())

    def prepare(self, ctx) -> None:
        if self.mega is not None:
            return
        spec = self.spec
        if ctx.world_size != spec.world_size:
            raise ValueError(
                f"the instance is for {spec.world_size} ranks; the group has "
                f"{ctx.world_size}"
            )
        w = self.weights
        self.rank = ctx.rank
        self._preloaded = False
        self.mega = _mega().MegaMoEV2(
            rank=ctx.rank,
            world_size=spec.world_size,
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            topk=spec.topk,
            quant=spec.quant,
            w1=w.w1,
            w1_scale=w.w1_scale,
            w2=w.w2,
            w2_scale=w.w2_scale,
            max_tok_per_rank=spec.max_tok_per_rank,
            mega_scheme=spec.mega_scheme,
            stage2_p2p_quant=spec.stage2_p2p_quant,
            swiglu_limit=spec.swiglu_limit,
            fanout_masks=spec.fanout_masks,
            act=spec.act,
            situ_beta=spec.situ_beta,
            situ_linear_beta=spec.situ_linear_beta,
        )

    def release(self) -> None:
        if self.spec is not None:
            self.mega = None
            self._preloaded = False

    def preload(self) -> bool:
        """Load the instance's stage 1 and stage 2 bundles (`preload_aot_bundles`, no
        GPU dispatch) once; False if already loaded."""
        if self._preloaded:
            return False
        self.mega.preload_aot_bundles()
        self._preloaded = True
        return True

    # `MegaMoEV2.forward` and `forward_prequant`, in their order.

    def check(self, call) -> None:
        """The arguments checks of `forward` (or of `forward_prequant`)."""
        mega = self.mega
        x = call["x"]
        run_tokens = int(x.shape[0])
        if run_tokens > mega.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={mega.mtpr}")
        if call["prequant"]:
            return
        wts, topk_ids = call["wts"], call["topk_ids"]
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")

    def fixed_slot(self) -> bool:
        return self.mega._s1_fixed_slot

    def quantize(self, x) -> tuple:
        """Fixed-slot: the MX 1x32 quant of the tokens, `(x_q, scales)`."""
        return self.mega.quantize(x)

    def select(self, run_tokens: int):
        """The bundle entry's config for this rank's tokens (the instance's
        `_select_config`, which also makes it the active entry)."""
        return self.mega._select_config(run_tokens)

    def prepare_compact(self, x, topk_ids, run_tokens, config, stream) -> tuple:
        """Compact: the count exchange (collective) with the quant fused in; the
        quantized tokens and scales it wrote."""
        mega = self.mega
        fx = _fx()
        prepare_stream = stream
        if prepare_stream is None:
            prepare_stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        elif hasattr(prepare_stream, "cuda_stream"):
            prepare_stream = fx.Stream(prepare_stream.cuda_stream)
        mega._run_compact_prepare(
            topk_ids,
            run_tokens,
            config.stage1,
            prepare_stream,
            quant_input=x,
        )
        return mega._s1_quant_x[:run_tokens], mega._s1_quant_scale[:run_tokens]

    def combine_mask(self, mask_invalid_slots) -> bool:
        """Whether combine skips -1 slots (None: the instance's default)."""
        if mask_invalid_slots is None:
            return self.mega._combine_mask
        return mask_invalid_slots

    def stage1(self, x, wts, scales, topk_ids, stream, config, prepared) -> None:
        """Dispatch and GEMM1 (with the count exchange unless `prepared`)."""
        self.mega._run_fused_stage1(
            x,
            wts,
            scales,
            topk_ids,
            stream=stream,
            config=config.stage1,
            prepared=prepared,
        )

    def stage2(self, run_tokens, stream, slice_output, config, combine_ids):
        """GEMM2 into the peers' combine buffers, then the combine (collective); a view
        of the instance's output buffer."""
        return self.mega._run_stage2(
            run_tokens, stream, slice_output, config, combine_ids
        )


def _fx():
    import flydsl.expr as fx

    return fx
