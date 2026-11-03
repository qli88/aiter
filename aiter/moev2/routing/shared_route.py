# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared-expert route `fused_row_per_rank`: one extra local expert per rank, and one
extra route per token to its own rank's shared slot.

The kernel is vLLM's `_append_shared_expert_kernel`
(`vllm/models/deepseek_v4/amd/mega_moe.py`), so the ids and weights are the ones vLLM
hands to `MegaMoEV2` today.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _append_shared_expert_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    num_tokens,
    shared_expert_id,
    TOPK: tl.constexpr,
    ROUTED_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < num_tokens
    for k in tl.static_range(TOPK):
        e = tl.load(topk_ids_ptr + rows * TOPK + k, mask=mask, other=0)
        w = tl.load(topk_weights_ptr + rows * TOPK + k, mask=mask, other=0.0)
        e = tl.where(e >= 0, e + e // ROUTED_PER_RANK, e)
        tl.store(out_ids_ptr + rows * (TOPK + 1) + k, e, mask=mask)
        tl.store(out_weights_ptr + rows * (TOPK + 1) + k, w, mask=mask)
    tl.store(
        out_ids_ptr + rows * (TOPK + 1) + TOPK,
        tl.zeros_like(rows) + shared_expert_id,
        mask=mask,
    )
    tl.store(
        out_weights_ptr + rows * (TOPK + 1) + TOPK,
        tl.full([BLOCK_M], 1.0, tl.float32),
        mask=mask,
    )


def shared_expert_id(rank: int, routed_per_rank: int) -> int:
    """The global id of `rank`'s shared slot: the last of its `routed_per_rank + 1`."""
    return rank * (routed_per_rank + 1) + routed_per_rank


def append_shared_expert(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    shared_expert_id: int,
    routed_per_rank: int,
) -> tuple:
    """`(weights, ids)` with one more column: routed ids remapped to the per-rank
    (routed + shared) layout (-1 stays -1), then a weight-1.0 route to
    `shared_expert_id`. Weights are float32."""
    num_tokens, topk = topk_ids.shape
    out_ids = torch.empty(
        (num_tokens, topk + 1), dtype=topk_ids.dtype, device=topk_ids.device
    )
    out_weights = torch.empty(
        (num_tokens, topk + 1), dtype=torch.float32, device=topk_ids.device
    )
    if num_tokens:
        block_m = 64
        _append_shared_expert_kernel[(triton.cdiv(num_tokens, block_m),)](
            topk_ids.contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            out_ids,
            out_weights,
            num_tokens,
            shared_expert_id,
            TOPK=topk,
            ROUTED_PER_RANK=routed_per_rank,
            BLOCK_M=block_m,
        )
    return out_weights, out_ids
