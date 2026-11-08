# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuning candidates (plan §4.2): a problem's feasible structures, each with the
`param_space` of its tunable segments (`resolver.tunable`).

A structure is a segment chain of `feasible_plans`; a combination picks one candidate
per tunable segment, agreeing on `block_m` and `nt`, and `resolver.build` turns it into
the plan dispatch runs for it (or the reason it would not). The filters read the legacy
tuner's env (`Filters.from_env`).
"""

import itertools
import os
import re
from dataclasses import dataclass

from ..dispatch.resolver import feasible_plans, tunable

# The legacy `TUNE_ONLY` names: the segments of the candidates each generator lists.
# Legacy's default when unset ("flydslv2") is not kept: unset or empty tunes all. The
# FHMoE, mxmoe and whole-graph candidates come from other legacy tuners, which read no
# `TUNE_ONLY`: a `TUNE_ONLY` leaves them out.
TUNE_ONLY_SEGMENTS = {
    "asm": ("asm.stage1", "asm.stage1_1x128", "asm.stage1_cktile", "asm.fmoe_1stage"),
    "cktile": ("ck.gemm1", "ck.gemm2", "cktile.gemm1", "cktile.gemm2"),
    "flydsl": ("flydsl.gemm1", "flydsl.gemm1_q", "flydsl.gemm2"),
    "flydslv2": (
        "flydsl.gemm1_layout",
        "flydsl_layout.gemm2",
        "flydsl_layout.gemm2_scatter",
    ),
    "flydsli4": ("flydsl.gemm1", "flydsl.gemm2"),
    "opus": ("opus.gemm1", "opus.gemm1_layout", "opus.gemm2"),
}
# The segments whose legacy lists `AITER_FMOE_TUNE_EXCLUDE_KERNELS` filters: the
# 2-stage ASM stage 1 and the CK stage 2.
EXCLUDE_SEGMENTS = frozenset(
    TUNE_ONLY_SEGMENTS["asm"][:3] + ("ck.gemm2", "cktile.gemm2")
)


@dataclass(frozen=True)
class Filters:
    """Which candidates are tuned. `only`: segment names, or families (the name
    before the dot, e.g. "ck" or "flydsl_layout"), empty for all; `drop`: segment
    names dropped; `exclude`: kernel names of `EXCLUDE_SEGMENTS` containing one of
    these are dropped; `regex`: only kernel names it matches (`re.search`) are
    kept."""

    only: frozenset = frozenset()
    exclude: tuple = ()
    regex: str | None = None
    drop: frozenset = frozenset()

    @classmethod
    def from_env(cls, environ=None) -> "Filters":
        """`TUNE_ONLY`, else `OPUS_ONLY=1` for the Opus segments only;
        `OPUS_SKIP_CKTILE=1`; `AITER_FMOE_TUNE_EXCLUDE_KERNELS`;
        `TUNE_MOE_KERNEL_REGEX`."""
        env = os.environ if environ is None else environ
        names = [n for n in env.get("TUNE_ONLY", "").split(",") if n]
        if not names and env.get("OPUS_ONLY", "0") == "1":
            names = ["opus"]
        only = set()
        for name in names:
            if name not in TUNE_ONLY_SEGMENTS:
                raise ValueError(
                    f"TUNE_ONLY={name!r}: known names {sorted(TUNE_ONLY_SEGMENTS)}"
                )
            only.update(TUNE_ONLY_SEGMENTS[name])
        skip_cktile = env.get("OPUS_SKIP_CKTILE", "0") == "1"
        exclude = env.get("AITER_FMOE_TUNE_EXCLUDE_KERNELS", "").split(",")
        return cls(
            frozenset(only),
            tuple(p.strip() for p in exclude if p.strip()),
            env.get("TUNE_MOE_KERNEL_REGEX") or None,
            frozenset(TUNE_ONLY_SEGMENTS["cktile"] if skip_cktile else ()),
        )

    def keeps(self, segment, params) -> bool:
        family = segment.name.split(".")[0]
        if self.only and not {segment.name, family} & self.only:
            return False
        if segment.name in self.drop:
            return False
        kernel = str(params.get("kernel", ""))
        if segment.name in EXCLUDE_SEGMENTS and any(p in kernel for p in self.exclude):
            return False
        return self.regex is None or re.search(self.regex, kernel) is not None


@dataclass(frozen=True)
class StructureCandidates:
    """A feasible structure and `{segment name: candidates}` for its tunable
    segments, in chain order."""

    structure: tuple
    candidates: dict

    @property
    def name(self) -> str:
        return " | ".join(s.name for s in self.structure)

    def block_ms(self) -> tuple:
        """The `block_m` values every tunable segment has a candidate for, ascending
        (a candidate without `block_m` takes any)."""
        common = None
        for cs in self.candidates.values():
            values = {c.get("block_m") for c in cs}
            if None not in values:
                common = values if common is None else common & values
        return tuple(sorted(common or ()))

    def combinations(self, block_m=None):
        """Each `{segment name: Params}` with one candidate per tunable segment whose
        candidates agree on `block_m` and `nt` (a field a candidate leaves out agrees
        with any value), by ascending `block_m`; only that `block_m` if given."""
        names = list(self.candidates)
        for bm in (block_m,) if block_m is not None else self.block_ms() or (None,):
            lists = [
                [c for c in cs if c.get("block_m", bm) == bm]
                for cs in self.candidates.values()
            ]
            for chosen in itertools.product(*lists):
                if len({c["nt"] for c in chosen if "nt" in c}) <= 1:
                    yield dict(zip(names, chosen))


NO_FILTERS = Filters()


def structures(problem, segments, filters=NO_FILTERS):
    """`(tunable, skipped)`: the `StructureCandidates` of `problem`'s feasible
    structures over `segments`, and `{structure name: reason}` for those with a
    tunable segment left without candidates."""
    plans, _ = feasible_plans(problem, segments)
    found, skipped = [], {}
    for structure in plans:
        candidates, reason = {}, None
        for segment in filter(tunable, structure):
            listed = segment.param_space(problem)
            kept = tuple(c for c in listed if filters.keeps(segment, c))
            if not kept:
                reason = (
                    f"{segment.name}: {'filtered out' if listed else 'no candidates'}"
                )
                break
            candidates[segment.name] = kept
        entry = StructureCandidates(structure, candidates)
        if reason is None:
            found.append(entry)
        else:
            skipped[entry.name] = reason
    return found, skipped
