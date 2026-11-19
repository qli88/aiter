# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Boundary formats between segments.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from dataclasses import dataclass

from ..problem import ScaleSpec

KINDS = ("tokens", "sorted", "inter", "routed_out", "p2p_scatter", "out")


@dataclass(frozen=True)
class Buf:
    # fmt None: "sorted" (routing, not values), the caller's own hidden_states, or, in
    # `Segment.reads`, any format. layout None, in `Segment.reads` only: any layout.
    kind: str
    fmt: ScaleSpec | None
    layout: str | None = "default"  # "aux_sorted", "v2_sorted", "opus_bucketed", ...
    block_m: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown Buf kind {self.kind!r}; known: {KINDS}")

    def mismatch(self, have: "Buf | None") -> str | None:
        """Why `have` cannot be read where this `Buf` is required, or None."""
        if have is None:
            return f"reads {self}, which no earlier segment produces"
        if (self.layout is not None and have.layout != self.layout) or (
            self.fmt is not None and have.fmt != self.fmt
        ):
            return f"reads {self}, gets {have}"
        return None

    def __str__(self):
        parts = [self.kind]
        if self.fmt is not None:
            parts.append(str(self.fmt))
        if self.layout is None:
            parts.append("any layout")
        elif self.layout != "default":
            parts.append(self.layout)
        if self.block_m is not None:
            parts.append(f"bm={self.block_m}")
        return ":".join(parts)


# The caller's hidden_states, before any segment.
CALL_INPUT = Buf("tokens", None)
