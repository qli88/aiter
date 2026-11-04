# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Boundary formats between segments."""

import operator
from dataclasses import dataclass

from ..problem import ScaleSpec

KINDS = ("tokens", "sorted", "inter", "routed_out", "p2p_scatter", "out")

# The row orders each kind may have; the first is its default.
ROWS = {
    # "received": one row per token a dispatch received on this rank; "compact_plan":
    # those rows grouped per expert by the gfx1250 compact plan.
    "tokens": ("token_major", "received", "compact_plan"),
    # "aux_sorted": the MXFP4 sort with its auxiliary outputs; "flat": the topk ids and
    # weights unsorted, for kernels that route themselves.
    "sorted": ("expert_sorted", "aux_sorted", "flat"),
    "inter": ("token_major", "expert_sorted"),
    "routed_out": ("token_major",),
    "p2p_scatter": ("token_major",),
    "out": ("token_major",),
}
# "v2": the FlyDSL v2 layout GEMM2's arrangement; "mxmoe": MXMOE stage 1's (its native
# BM16 chunks or v2's, by tile), read by the MXMOE stage 2.
SCALES = ("plain", "v2", "mxmoe")

_KIND_DEFAULT = object()


def _check(name, value, allowed):
    if value is None or (isinstance(value, str) and value in allowed):
        return
    if isinstance(value, frozenset) and value and all(v in allowed for v in value):
        return
    raise ValueError(f"Buf {name} {value!r}; known: {allowed}")


def _admits(need, have) -> bool:
    if need is None:
        return True
    return have in need if isinstance(need, frozenset) else have == need


def _field(value) -> str:
    return "|".join(sorted(value)) if isinstance(value, frozenset) else str(value)


@dataclass(frozen=True)
class Buf:
    # fmt None: "sorted" (routing, not values), the caller's own hidden_states, or, in
    # `Segment.reads`/`reads_with`, any format. There only, rows, scales, pad or
    # block_m None: any; a frozenset of rows or scales: any of those. `replace` with
    # another kind keeps the rows, which must be valid for it.
    kind: str
    fmt: ScaleSpec | None
    rows: str | frozenset | None = _KIND_DEFAULT
    scales: str | frozenset | None = "plain"  # SCALES
    pad: int | None = 0  # row padding the reader must honor
    block_m: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown Buf kind {self.kind!r}; known: {KINDS}")
        if self.rows is _KIND_DEFAULT:
            object.__setattr__(self, "rows", ROWS[self.kind][0])
        _check(f"{self.kind} rows", self.rows, ROWS[self.kind])
        _check("scales", self.scales, SCALES)
        if self.pad is not None:
            try:
                pad = None if isinstance(self.pad, bool) else operator.index(self.pad)
            except TypeError:
                pad = None
            if pad is None or pad < 0:
                raise ValueError(f"Buf pad {self.pad!r}: a row count >= 0 or None")
            object.__setattr__(self, "pad", pad)

    @property
    def concrete(self) -> bool:
        """One `rows`, `scales` and `pad` value each, as `Segment.produces` returns."""
        return (
            isinstance(self.rows, str)
            and isinstance(self.scales, str)
            and self.pad is not None
        )

    def mismatch(self, have: "Buf | None") -> str | None:
        """Why `have` cannot be read where this `Buf` is required, or None."""
        if have is None:
            return f"reads {self}, which no earlier segment produces"
        if not (
            _admits(self.rows, have.rows)
            and _admits(self.scales, have.scales)
            and (self.fmt is None or have.fmt == self.fmt)
            and (self.pad is None or have.pad == self.pad)
            and (self.block_m is None or have.block_m == self.block_m)
        ):
            return f"reads {self}, gets {have}"
        return None

    def __str__(self):
        parts = [self.kind]
        if self.fmt is not None:
            parts.append(str(self.fmt))
        if self.rows is None:
            parts.append("any rows")
        elif self.rows != ROWS[self.kind][0]:
            parts.append(_field(self.rows))
        if self.scales is None:
            parts.append("any scales")
        elif self.scales != "plain":
            parts.append(f"scales={_field(self.scales)}")
        if self.pad is None:
            parts.append("any pad")
        elif self.pad:
            parts.append(f"pad={self.pad}")
        if self.block_m is not None:
            parts.append(f"bm={self.block_m}")
        return ":".join(parts)


# The caller's hidden_states, before any segment.
CALL_INPUT = Buf("tokens", None)
