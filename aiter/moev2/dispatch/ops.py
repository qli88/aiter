# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MoE op chain: every plan covers it with segments, in order."""

import enum


class Op(enum.IntEnum):
    COMM_IN = 0  # EP dispatch or TP all-gather (only with a ParallelSpec)
    SORT = 1  # route tokens to experts: sorted ids, weights, expert ids, output buffer
    QUANT_A1 = 2  # activations into GEMM1's input format
    GEMM1 = 3  # gate/up projection
    ACT = 4  # activation (and gate * up)
    QUANT_INTER = 5  # intermediate into GEMM2's input format
    GEMM2 = 6  # down projection
    REDUCE = 7  # weighted sum over each token's experts, into the output
    COMM_OUT = 8  # EP combine, TP reduce-scatter/all-reduce (only with a ParallelSpec)
    TAIL = 9  # fused epilogue after the MoE (only when the problem has a tail)


_COMM = (Op.COMM_IN, Op.COMM_OUT)


def chain(problem) -> tuple:
    """The ops a plan for `problem` must cover."""
    ops = tuple(
        op
        for op in Op
        if op is not Op.TAIL and (problem.parallel is not None or op not in _COMM)
    )
    return ops + (Op.TAIL,) if problem.tail is not None else ops
