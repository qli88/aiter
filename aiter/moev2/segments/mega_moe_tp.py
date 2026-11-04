# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segments `mega_moe_tp.small` and `mega_moe_tp.bigm`: `MegaMoeTP` (gfx950, TP), the
whole layer from the TP gather to the reduce in one launch, with the fused tail where
legacy fuses it.

The kernels stay in `aiter.ops.flydsl.kernels.mega_moe_tp`. Both segments wrap one
`MegaMoeTP` instance (`adopt`), which the caller builds and closes: `small` is its
small-batch engine (`MegaMoeTPEngine`, graph-capturable), `bigm` its big-M kernel
(`MegaMoeTPBigM`, chunked over SDMA, not capturable, set up on first use or by the
instance's `prepare`). A call runs the one legacy's switch picks (`serves`: the
instance's `_big` of the call's local tokens). Structure: `comm_mode` (ag_rs or ar),
`comm_dtype` (the wire, fp8 is lossy) and the capacity (`capacity`: the small engine's
local tokens; big-M from `threshold` up to the instance's `max_local_tokens`). The two
tuned CSVs stay the kernels' own tables.

The instance has one watchdog and one cross-rank state: the small segment, which every
engine has (bucket 1 is below the switch), reports `health()` (legacy's
`poll_errors()`, a sync, with `check_errors`' reasons) and runs `reset()` (legacy's
`reset()`, collective) for both kernels; the big-M segment has neither. Neither frees the instance (`release`,
`abort`): the caller's `close()` does.

`workspace_bytes()` is the instance's memory the segment runs on, though the caller
allocated it: small, its engine's arena (`symmetric`) and buffers (`device`), read off
the instance; big-M, what its constructor allocates (computed from the config; zero
when big batches are unreachable). `symmetric` here is torch memory (IPC-mapped), not
a MORI heap. Not counted: MORI's SDMA queues, the unit table growing for a tuned plan
with more chunks, big-M's chunk table per batch size, the small engine's converted
inputs of its last call (kept), per-call temporaries.
"""

import torch

from aiter import dtypes

from ..dispatch.buf import CALL_INPUT, Buf
from ..dispatch.ops import Op
from ..problem import FrozenMap, ScaleSpec
from .base import Cap, MemoryNeed, Params, Segment
from .mxfp4 import mxfp4


def _engine_module():
    from aiter.ops.flydsl.kernels.mega_moe_tp import mega_moe_tp

    return mega_moe_tp


_LOSSY = FrozenMap({"comm_dtype": frozenset({"fp8"})})


class _MegaMoeTPSegment(Segment):
    covers = (Op.COMM_IN, Op.COMM_OUT)
    stateful = True
    output_lifetime = "until_next_forward"
    lossy_values = _LOSSY
    default_lossy = _LOSSY

    def __init__(self, mtp):
        self.mtp = mtp

    @classmethod
    def adopt(cls, mtp):
        return cls(mtp)

    def reads(self, p):
        return (CALL_INPUT,)

    def produces(self, p, params):
        return Buf("out", ScaleSpec(dtypes.bf16))

    def build(self, p, params):
        return self

    def serves(self, rows: int) -> bool:
        """Whether legacy's switch runs a call of `rows` input rows on the big-M
        kernel (`rows`: ag_rs, this rank's; ar, the replicated batch)."""
        mtp = self.mtp
        return mtp._big(mtp._local(rows))


class MegaMoeTPSmall(_MegaMoeTPSegment):
    """The instance's small-batch engine; also the instance's health and reset (see the
    module docstring)."""

    name = "mega_moe_tp.small"
    capabilities = (
        Cap(
            arch=frozenset({"gfx950"}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16}),
            activation=frozenset({"silu", "swiglu", "situv2"}),
            shared=True,
            parallel=True,
            tail=True,
        ),
    )
    structure_params = frozenset({"comm_mode", "comm_dtype", "capacity"})

    def params(self) -> Params:
        cfg = self.mtp.cfg
        return Params(
            comm_mode=cfg.comm_mode,
            comm_dtype=cfg.comm_dtype,
            capacity=self.mtp.engine.mmax,
        )

    def workspace_bytes(self) -> MemoryNeed:
        engine = self.mtp.engine
        buffers = (engine.ctrl, engine.routes, engine.y, engine.proutes, engine.xg)
        return MemoryNeed(
            device=sum(t.nbytes for t in buffers),
            symmetric=engine.arena.storage.nbytes,
        )

    def prepare(self, ctx) -> None:
        cfg = self.mtp.cfg
        if (ctx.rank, ctx.world_size) != (cfg.rank, cfg.world_size):
            raise ValueError(
                f"the instance is rank {cfg.rank} of {cfg.world_size}; the engine's "
                f"group has rank {ctx.rank} of {ctx.world_size}"
            )

    def forward(self, x, topk_weights, topk_ids, out, tail, bf16):
        """The small engine's forward (legacy's last line of `MegaMoeTP.forward`)."""
        return self.mtp.engine(x, topk_weights, topk_ids, out, tail=tail, bf16=bf16)

    def health(self) -> str | None:
        """Legacy's `poll_errors()` (a sync) of the small engine and the big-M kernel;
        each flagged kernel's reason is the one `check_errors()` raises with."""
        mtp = self.mtp
        reasons = []
        if mtp._bigm is not None and mtp._bigm.poll_errors():
            reasons.append(
                "MegaMoeTP: a big-M wait on a peer gave up (results invalid)"
            )
        err = mtp.engine.poll_errors()
        if err:
            names = [n for b, n in _engine_module()._ERR_NAMES if err & b]
            reasons.append(
                f"MegaMoeTP rank {mtp.engine.rank}: a wait inside the kernel timed out "
                f"({'|'.join(names)}); the output is invalid. Ranks out of step "
                "(different call counts or local token counts) need reset()."
            )
        return "; ".join(reasons) or None

    def reset(self) -> None:
        """Legacy's `reset()` (collective): the small engine's and the big-M kernel's."""
        self.mtp.reset()


class MegaMoeTPBigM(_MegaMoeTPSegment):
    """The instance's big-M kernel: legacy's big-M branch of `MegaMoeTP.forward`, step
    for step (`check`, `inputs`, then `ar` or `ag_rs`)."""

    name = "mega_moe_tp.bigm"
    capabilities = (
        Cap(
            arch=frozenset({"gfx950"}),
            w1=frozenset({mxfp4()}),
            w2=frozenset({mxfp4()}),
            out_dtype=frozenset({dtypes.bf16}),
            activation=frozenset({"silu", "swiglu"}),
            shared=True,
            parallel=True,
            tail=True,
        ),
    )
    structure_params = frozenset({"comm_mode", "comm_dtype", "threshold", "capacity"})

    def params(self) -> Params:
        cfg = self.mtp.cfg
        return Params(
            comm_mode=cfg.comm_mode,
            comm_dtype=cfg.comm_dtype,
            threshold=self.mtp.bigm_m,
            capacity=cfg.max_local_tokens,
        )

    def workspace_bytes(self) -> MemoryNeed:
        mtp = self.mtp
        if not mtp._big(mtp.cfg.max_local_tokens):
            return MemoryNeed()
        return bigm_need(mtp.cfg)

    def check(self, x, topk_weights, topk_ids, out, tail) -> None:
        """The small engine's input checks up to `max_local_tokens` (ar batches need
        not be a multiple of tp), of `out` and of the tail."""
        engine = self.mtp.engine
        engine._local_tokens(
            x,
            topk_weights,
            topk_ids,
            limit=self.mtp.cfg.max_local_tokens,
            uneven=True,
        )
        engine._check_out(out, x.shape[0])
        if tail is not None:
            engine._check_tail(tail, x.shape[0])

    def inputs(self, x, topk_weights, topk_ids) -> tuple:
        """The kernel's input types: contiguous rows, f32 weights, int32 ids."""
        return (
            x.contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )

    def ar(self, x, topk_weights, topk_ids, out, tail):
        """ar: this rank's rows of the replicated batch, every rank's output gathered
        in the kernel (raises with a tail)."""
        return self.mtp._bigm_ar(x, topk_weights, topk_ids, out, tail)

    def ag_rs(self, x, topk_weights, topk_ids, out, tail, bf16):
        """ag_rs: the big-M kernel on this rank's rows, with the fused tail (bf16: and
        the tail's bf16 rows)."""
        mtp = self.mtp
        local = bool(mtp.engine.tn_local)
        res = mtp.bigm().moe(x, topk_weights, topk_ids, out=out, tail=tail, local=local)
        if tail is not None and bf16:
            res = (*res, mtp._bigm_rows(tail[1], tail[2], local))
        return res


def bigm_need(cfg) -> MemoryNeed:
    """The bytes `MegaMoeTPBigM(...)` allocates for this `MegaMoeTPConfig`, as
    `MegaMoeTP.bigm()` builds it: its `SdmaComm` arena (`symmetric`) and the device
    buffers of its constructor and `SdmaComm.commit`."""
    from aiter.ops.flydsl.kernels.mega_moe_tp import mega_moe_tp_config as c

    tp, H, K1, NE = cfg.world_size, cfg.model_dim, cfg.topk, cfg.experts
    T = -(-cfg.max_local_tokens // c.PAD) * c.PAD * tp
    RB = c.i8_row_bytes(H)
    regions = (
        T * H // 2,
        T * H // 32,
        T * K1 * 4,
        T * K1 * 4,
        T * RB,
        T * RB,
        T * H * (2 if cfg.comm_mode == "ar" else 1),
        T * 4,
        2 * T * H * 2,
        tp * 64,
    )
    align = _engine_module()._ALIGN
    cursor = 0
    for n in regions:
        cursor = -(-cursor // align) * align + max(n, 1)
    m = T // tp
    uslots = 2 * (-(-tp * m * K1 // c.UNIT_ROWS) // 8) + c.CMAX * (
        -(-NE // 8) + c.UQ + 2
    )
    device = (
        tp * 8  # bases
        + 2 * tp * c.NQ * 4  # consumed
        + 8 * 8  # offs
        + 3 * m * H * 2  # normed, resout, ybuf
        + T * K1 * (H * 2 + 4 + 4)  # routes, lrix, lwt
        + 8 * uslots * 4 * 4  # utab
        + (c.BIGM_CTRL_INTS + 32) * 4  # ctrl
        + m * H * 2
        + m * K1 * 4
        + m * K1 * 4
        + m * H * 2
        + T * H
        + T * 4  # pads
    )
    return MemoryNeed(device=device, symmetric=-(-cursor // align) * align)
