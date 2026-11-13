# How to add a new MoE kernel to AITER (moev2)

This tutorial is for kernel authors who want a new MoE kernel to run through the moev2
dispatch (`aiter/moev2`). It covers the three kinds of change, from smallest to largest:

| You are adding | Typical v2 work |
|---|---|
| A new **instance** of a family v2 already runs (a CK tile, a FlyDSL tile, an ASM kernel in `hsa/<arch>`) | Usually none: add it to the family's instance list, tune, ship the rows |
| A new **capability** of an existing kernel (an activation, a dtype, a feature) | Widen the segment's `Cap`s |
| A new **kernel family** (a new stage 1, stage 2, a fused multi-op kernel, a whole-graph implementation) | A new segment, its kernel-name grammar, tests; this is most of the tutorial |

Read "The transition rule" first: it decides where your kernel has to land.

---

## 1. The transition rule (read this first)

No moev2 path has graduated. With `AITER_MOEV2` unset, the default, every call still
runs the legacy code in `aiter/fused_moe.py` and `aiter/fhmoe.py`. v2 runs a call only
when you enable its path explicitly, for example `AITER_MOEV2=a8w4`, or
`AITER_MOEV2=a8w4,a4w4`, or `on`. It is read once per process; tests that change it call
`aiter.moev2.envs.reload()`.

So until the legacy code is retired, two dispatchers exist. That has consequences for a
new kernel:

- **Kernel wrappers are shared.** Put the Python wrapper of your kernel under
  `aiter/ops/...` (or in a `aiter/moev2/segments/*.py` module that legacy imports back,
  as `segments/opus.py` does for `_opus_stage2_fp8_enabled`). There must be one copy.
- **Shipped tuned rows are read by both sides.** The shipped tables
  (`aiter/configs/tuned_fmoe.csv`, `aiter/configs/model_configs/*tuned_fmoe*.csv`,
  `tuned_fhmoe.csv`) are legacy's. v2 reads a converted copy under `aiter/configs/moev2/`,
  which must equal a fresh conversion (`tests/moe_dispatch/convert_tables.py --check`;
  v2 raises at runtime on a stale copy). No CI workflow regenerates it: run
  `convert_tables.py --update` yourself. You cannot ship a row only to v2.
- **A row naming a family legacy does not know is a bug for default users.** Legacy
  classifies kernel names with its own `startswith` checks. An unknown stage-1 name goes
  to its ASM branch (`asm_stage1`). An unknown stage-2 name goes to
  `aiter.ck_moe_stage2_fwd`, where CK logs "ck kernel not found" and runs its heuristic
  instance, so another kernel runs silently.
  So:
  - **Existing family:** ship rows as today.
  - **New family, and default users should get it:** add the legacy branch too (in
    `get_2stage_cfgs` in `aiter/fused_moe.py`, calling the shared wrapper), ship the
    rows, and keep both sides equal. The CPU and GPU differentials fail when they
    diverge.
  - **New family, v2 only:** ship no row naming it. Users opt in with the path switch
    plus their own tuned table:

    ```bash
    AITER_MOEV2=a16w16 \
    AITER_MOEV2_TUNED_TABLE=/path/to/my_plans.csv \
    AITER_MOEV2_TUNED_TABLE_MODE=override \
    python my_model.py
    ```

    `my_plans.csv` is a v2 plan file written by the v2 tuner (section 6). The segment's
    `reach` is then `"hibernating"`, and it needs a GPU test (section 5.6).
- **No behavior change for existing keys.** While v2 is being brought to parity with
  legacy (the migration), a new segment may not change what any key legacy serves runs.
  The CPU differential (`differential.py --check`), which compares v2's plan of every
  grid key with legacy's recorded decision, must stay clean. In practice, a new segment
  is reached through tuned rows, not through the default ranking (5.5).
- **`aiter.moev2` imports on CPU, without legacy.** No `aiter/moev2` module may import
  `aiter.fused_moe`, `aiter.fhmoe` or `aiter.fused_moe_bf16_asm`, and every module must
  import on a machine without a GPU (`tests/test_moev2_imports.py`). So the wrapper
  module your segment imports, and the instance list behind its `param_space`, must be
  plain Python at import time: load compiled modules inside the functions that launch.
  `build_checks.py` calls `param_space` on CPU.

---

## 2. Five concepts

Everything below uses these. The reference for every attribute and method is the
docstring of `Segment` in `aiter/moev2/segments/base.py`.

**The problem.** `MoeProblem` (`aiter/moev2/problem.py`) is one dispatch key: shape
(`m`, `model_dim`, `inter_dim`, `experts`, `topk`), arch and CU count, `act_in` / `w1` /
`w2` as `ScaleSpec`s (dtype, quant group, scale dtype), `out_dtype`, activation, layout
(gate mode, shuffle, g1u1, padding) and `features` (`doweight_stage1`, `ep_mask`,
`bias1`, ...).

**The op chain.** Every plan covers the ops of `aiter/moev2/dispatch/ops.py` in order:

```text
SORT, QUANT_A1, GEMM1, ACT, QUANT_INTER, GEMM2, REDUCE   (+ COMM_IN/COMM_OUT, TAIL)
```

**Segments.** A segment is a thin adapter over one kernel family that covers a span of
the chain (`covers = (first, last)`). For example `ck.gemm1` covers `(GEMM1, ACT)`,
`ck.gemm2` covers `(GEMM2, REDUCE)`, the 1-stage ASM kernel covers `(QUANT_A1, REDUCE)`,
and a whole-graph implementation covers `(SORT, REDUCE)`. A plan is a chain of segments
whose spans are adjacent, and whose boundary formats match: each segment declares the
`Buf`s it `reads` and the `Buf` it `produces` (`aiter/moev2/dispatch/buf.py`).

**Paths.** A path (`aiter/moev2/migration/<path>.py`) claims a family of calls (for
example `a16w16`: bf16/fp16, unquantized) and lists the segments it may combine
(`SEGMENTS`). `AITER_MOEV2=<path>` switches it on.

**How a plan is chosen.** The resolver (`aiter/moev2/dispatch/resolver.py`):

1. finds the feasible structures: segment chains that cover the op chain, whose
   `Cap`s accept the problem and whose boundaries match (`feasible_plans`);
2. if a tuned table has a row for the key, runs it. A shipped row (the converted legacy
   tables) runs the structure whose segments own its kernel names (`owns`, `check_row`,
   `row_params`), falling back to a smaller token tier as legacy does. A row of a v2
   plan file (the tuner's output, section 6) names its structure and each segment's
   candidate; it matches its exact key only, is built by `resolver.build`, and is
   refused unless each candidate is in that segment's `param_space(p)`;
3. otherwise consults the default ranking (`aiter/moev2/dispatch/ranking_table.py`) and
   runs the first feasible structure it lists, at each segment's `default_params`.

The tuner (`python -m aiter.moev2.tuning`) enumerates the same structures and, per
segment, the candidates of `param_space(p)`, and measures them end to end. A new segment
with a `param_space` is tuned with no tuner change.

To see all of this, set `AITER_MOE_EXPLAIN=1`: for each new problem (resolutions are
cached, so a repeated problem logs once), the log lists the problem, the feasible
structures, each rejected segment with the reason, and the choice.

---

## 3. Case A: a new instance of an existing family

For CK, FlyDSL, Opus, ASM, MXMOE and CK-Tile, each segment's `param_space(p)` reads the
family's own instance list, filtered by the problem the way the legacy tuner filters it.
Examples: `ck_param_space` reads the CK instance dicts
(`gemm_moe_ck2stages_common.py`), `opus_param_space` reads
`opus_a8w4_stage1_instances_for_shape` and `get_opus_a8w4_stage2_kernels`.

So for a new instance:

1. Add it to the family's list (and build it, for compiled families).
2. Check that both tuners see it:

   ```bash
   python tests/moe_dispatch/param_space_checks.py --check
   ```

   This compares, on every simulated arch, the candidates of v2's segments with the
   legacy tuner's generators, for the families listed in its `_families()`. Adding the
   instance to the shared list keeps them equal.
3. Tune the shapes it is meant for (section 6) and ship the rows in the legacy tables as
   today. Then regenerate v2's copy:

   ```bash
   python tests/moe_dispatch/convert_tables.py --update
   ```

4. Run the CPU gates of section 7.

Every CPU gate except `convert_tables.py` reads a frozen copy of the tables
(`tests/moe_dispatch/golden/inputs/`), so that tuning PRs do not change them. Your new
rows are invisible to them until the copy is refrozen, in a PR of its own: refresh the
copy, update the two golden files, then run all the gates of section 7, since the
coverage and the known declines can change too:

```bash
python tests/moe_dispatch/snapshot.py --refresh-inputs
python tests/moe_dispatch/snapshot.py --update
python tests/moe_dispatch/row_params_checks.py --update
```

If the instance's name does not follow its family's grammar, `kernel_names.parse` raises
`KernelNameError`, and dispatch declines the row ("unknown kernel name ..."): fix the
name or the grammar (5.2).

---

## 4. Case B: a new capability of an existing kernel

Each segment says what it supports with `Cap`s (`segments/base.py`). A field left `None`
means "any"; a set means membership; `features` follows the subset rule (every feature
the problem uses must be listed). Anything a problem has that no `Cap` declares rejects
the segment, so a new input never reaches a kernel that would silently ignore it.

To add, say, Gelu to a stage 1 that now supports it:

1. Implement it in the kernel and its wrapper.
2. Add `"gelu"` to the `activation` set of the segment's `Cap`s (often a shared helper,
   such as `_cap` in `segments/ck.py` or `a8w4_caps` in `segments/mxfp4.py`).
3. If the new combination is also a new tuning space, make `param_space` list it.
4. Check that the path lets the call in. A path's `claims(call)` and `takes(problem)`
   filter calls before any segment sees them (`aiter/moev2/migration/local.py`):
   `local.claims` limits activation, gate mode, activation parameters, bias and
   padding (its `activations=`, `gate_modes=`, `act_params=`, `bias=`, `pads=`
   arguments), and `local.key_args_supported` the same at the key level, except bias
   and padding. `claims` takes `ActivationType` values, `key_args_supported` lower-case
   names (`"gelu"`). Widening a `Cap` alone does not route a new activation or feature
   to v2; widen the path's arguments too, in both functions.
5. Run `coverage_checks.py --check` (section 7). It prints, per path, the grid keys of
   each kind (`plan`, `gap`, `legacy`, `row`); compare the counts before and after.

Remember the transition rule: if legacy does not run that combination the same way,
keys that legacy serves would change. The CPU differential (`differential.py --check`)
fails on any such key. Either legacy gets the same change, or the combination stays a
gap that v2 declines back to legacy until the migration is complete.

---

## 5. Case C: a new kernel family, step by step

The running example is a hypothetical family **Acme**: a new bf16 stage 2 for the
`a16w16` path (unquantized bf16/fp16), named in tuned rows as

```text
acme_moe2_bf16_t64x256_w4        (stage 2, tile 64 x 256, 4 waves)
```

A bf16 stage 2 runs through the plain two-stage executor (`run_two_stage` in
`aiter/moev2/execute/executor.py`). Per-1x32 formats (MXFP4, mxfp8) run through
`run_mxfp4` instead, which adds `call_args` and `stage2_launch`; copy the closest
segment in `segments/mxfp4.py` (the FlyDSL and MXMOE segments; their wrappers are in
`segments/flydsl.py` and `segments/mxmoe.py`), `segments/opus.py` or
`segments/ck_tile.py` for those.

### 5.1 Write the kernel wrapper

Put it next to the other ops (for example `aiter/ops/acme_moe.py`). The executor calls a
stage 2 like this (from `run_two_stage`):

```python
metadata.stage2(
    a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, moe_out, topk,
    w2_scale=..., a2_scale=..., block_m=..., sorted_weights=...,
)
```

and a stage 1 like this:

```python
metadata.stage1(
    a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
    block_m=..., a1_scale=..., w1_scale=..., sorted_weights=...,
)  # returns a2
```

`metadata.stage1` / `stage2` are whatever your segment's `build` returns, usually a
`functools.partial` of the wrapper with the kernel name and problem-fixed arguments bound.
So the wrapper's signature is:

```python
def acme_moe_stage2(
    a2,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    *,
    kernelName,
    w2_scale=None,
    a2_scale=None,
    block_m=None,
    sorted_weights=None,
): ...
```

`sorted_weights` is passed to stage 2 only when the routing weights are applied there
(no `doweight_stage1`). The output buffer `out` is accumulated into.

### 5.2 Teach `kernel_names.py` the family's grammar

v2 classifies tuned kernel names only through `aiter/moev2/dispatch/kernel_names.py`.
`parse(name)` returns a `KernelName(family, stage, fields)`, and
`parse(n).format() == n` must hold for every shipped name. Add:

```python
ACME = "acme"  # acme_moe{1,2}_<dtype>_t<M>x<N>[_opts]

_ACME = re.compile(
    r"acme_moe(?P<stage>[12])_(?P<dtype>[a-z0-9]+)_t(?P<tile_m>\d+)x(?P<tile_n>\d+)"
    r"(?P<opts>(?:_[A-Za-z0-9]+)*)"
)


def _parse_acme(m):
    fields = {
        **_ints(m, "tile_m", "tile_n"),
        "dtype": m["dtype"],
        "opts": _opts(m["opts"]),
    }
    return int(m["stage"]), fields


def _format_acme(k):
    f = k.fields
    return f"acme_moe{k.stage}_{f['dtype']}_t{f['tile_m']}x{f['tile_n']}{_join_opts(f['opts'])}"
```

Then add `ACME` to `FAMILIES`, `_format_acme` to `_FORMATTERS`, and the
`(ACME, _ACME, _parse_acme)` triple to the loop in `parse`. Check that no existing
pattern also matches your names. `tests/test_moev2_dispatch.py::
test_every_shipped_kernel_name_parses_and_round_trips` covers shipped names; add an
edge-case test of your own grammar (5.6).

### 5.3 Write the segment

Create `aiter/moev2/segments/acme.py`. The class attributes and methods, in the order
the resolver uses them:

```python
"""Acme bf16 stage 2 (`aiter.ops.acme_moe`), v2 only: a user's tuned table runs it."""

import functools

from aiter.ops.acme_moe import acme_moe_stage2, acme_stage2_instances

from ..dispatch import kernel_names
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from .base import Params, RowRejection, Segment
from .ck import a16w16_caps
from .defaults import TUNE_BLOCK_MS, tunable


class AcmeGemm2(Segment):
    name = "acme.gemm2"  # <family>.<stage>[_variant]; unique
    prefixes = (kernel_names.ACME,)  # the families whose tuned rows it runs
    covers = (Op.GEMM2, Op.REDUCE)  # a stage 2
    reach = "hibernating"  # no shipped row runs it; see 5.5

    @property
    def capabilities(self):
        # Start from a helper that describes the format; narrow it to what the kernel
        # really supports. Anything not declared rejects the segment.
        return a16w16_caps()

    def owns(self, kernel) -> bool:
        return (
            kernel is not None
            and kernel.family == kernel_names.ACME
            and kernel.stage == 2
        )

    # Boundaries: what it reads from earlier segments, what it hands on.
    def reads(self, p):
        return (Buf("sorted", None), Buf("inter", p.act_in))

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    # Tuned rows.
    def check_row(self, p, row, kernel_name1=""):
        if p.shape.model_dim % 256:
            return RowRejection(
                "[fused_moe] Acme stage 2 needs model_dim % 256 == 0; "
                "using default heuristics"
            )
        return None

    def row_params(self, p, row):
        return {}  # what the row sets besides the kernel names

    # Launch.
    def build(self, p, params):
        return functools.partial(acme_moe_stage2, kernelName=params.kernel)

    # Tuning: one Params per candidate.
    def param_space(self, p):
        if not (self.accepts(p) and tunable(p)) or p.shape.model_dim % 256:
            return ()
        return tuple(
            Params(kernel=name, block_m=block_m)
            for block_m in TUNE_BLOCK_MS
            for name in acme_stage2_instances(block_m=block_m)
        )
```

What each part is for:

- **`capabilities`**: what the kernel can compute. `reject_reason(p)` comes from the
  `Cap`s. Override it only for a check a `Cap` cannot express, and keep the last
  structure of a ranking list free of such checks (ranking rule R6).
- **`reads` / `produces`**: the resolver chains segments whose formats match. A field of
  a `Buf` in `reads` left `None` accepts any value. `produces` returns one concrete value
  per field. If your kernel fixes a format only once its kernel is chosen, narrow it in
  `reads_with(p, params)` (as `OpusGemm2` does) or in `produces(p, params)` from
  `params`. If your stage 1 also quantizes its output for stage 2, its `covers` ends at
  `Op.QUANT_INTER` and it produces stage 2's input format. Under `run_two_stage` it
  must then provide `out_buffer` (the buffer it writes) and `split_out` (the values and
  scales from it), as `AsmStage1Blockscale` does (`segments/asm.py`); `run_mxfp4`
  drives a fused quant through `fuse_quant` instead.
- **`check_row`**: a row this segment cannot run is discarded, with the warning legacy
  would log (or `RowRejection(None)` for a silent discard), and dispatch falls back to
  the default ranking. `declines_row` is different: it sends the call back to legacy,
  and exists only for migration parity. A new family should not need it.
- **`row_params`**: whatever the row sets besides `kernelName1/2`, for example
  `{"block_m": int(row["block_m"])}`.
- **`structure_params`**: the parameters, set by the segment's own `params`, that
  change a boundary buffer or a buffer size, so that every segment of a plan must agree
  on them. The resolver hands a segment the plan's value of each one it declares and
  does not set, and fails a plan in which a segment sets one to another value; the
  tuner measures each value as a separate structure. For example the Opus stage 1
  declares `block_m`, which its kernel fixes. A segment that runs at the plan's
  `block_m`, as `CkGemm2` and the Acme sample do, declares nothing.
- **`param_space`**: the candidates the tuner may try for `p`. Return `()` for a problem
  the segment does not accept, one that is not tuned (`defaults.tunable`), or one whose
  row dispatch would discard. Each `Params` holds `kernel` and the structure values the
  other segments must share (`block_m`, `nt`, `ksplit`). Only the fields a tuned row has
  a column for are allowed: `kernel`, `block_m`, `ksplit`, `nt`, `xbf16` and `flat`
  (`resolver.candidate_row` raises on any other). Encode any other option in the kernel
  name.
- **`default_params`**: only for a segment the default ranking lists (`reach =
  "ranked"`). It returns the parameters the segment runs at without a tuned row. Leave
  it `None` otherwise. (`test_only_ranked_segments_have_default_params` checks a few
  fixed problems, none of them a16w16, so it will not catch a mistake here.)
- **`metadata_fields`**: `MOEMetadata` fields the segment sets
  (`aiter/moev2/execute/metadata.py`), for example `has_bias`. Do not set `fuse_quant`,
  `skip_inter_quant` or `output_aux`: the resolver derives them from the boundaries and
  asserts if a segment sets them.
- **Stateful kernels** (symmetric memory, compiled bundles, workspaces) also declare
  `stateful = True` and implement `workspace_bytes`, `prepare`, `health`, `reset`,
  `release` and `abort`. Those run under `MoeEngine`; read `segments/mega_moe_v2.py`.
- **A fused kernel** (for example sort + GEMM1 in one launch) is one segment with
  `covers=(Op.SORT, Op.ACT)`. The resolver accepts such a plan, but no executor runs it
  yet: `run_two_stage` takes a separate sort segment and a separate A1 quant, so a new
  fusion also needs executor work. **A whole-graph implementation** can reuse `WholeGraph`
  (`segments/whole_graph.py`): register it in `aiter.fused_moe_registry` and name its
  rows `impl__<name>__<config>`.

### 5.4 Add it to a path

Add an instance to the path's `SEGMENTS` (`aiter/moev2/migration/a16w16.py` here):

```python
from ..segments.acme import AcmeGemm2

SEGMENTS = (
    Sort(),
    QuantA1(),
    CkGemm1(),
    QuantInter(),
    CkGemm2(),
    AcmeGemm2(),
    WholeGraph(),
    AsmFmoe1Stage(),
)
```

The resolver now finds `sort | quant.a1 | ck.gemm1 | quant.inter | acme.gemm2` feasible
wherever the `Cap`s and the boundaries allow it.

If no existing path claims your format (a new quant type or dtype combination), write a
new path module modeled on `migration/a16w16.py`:

- `claims(call)`: which calls it takes;
- `takes(problem)`: the same, at the key level. `tests/test_moev2_dispatch.py::
  test_claims_and_takes_agree_on_key_fields` and `test_the_paths_take_disjoint_keys`
  check that `claims` and `takes` agree and that no two paths take a key;
- `LOCAL = local.LocalDispatch(SEGMENTS, takes)` and `paths.register(paths.Path(...))`
  with `segments=`, `takes=`, `dispatch_record=`, `resolve=LOCAL.resolve_call` (used by
  `MoeEngine.prepare`) and `entry="op"`, as `a16w16.py` does. Without `segments` and
  `takes`, the ranking, coverage, row-params and build checks skip the path silently;
  without `dispatch_record`, the CPU differential does;
- `ARCHS`: the archs it claims. Claim an arch only where a GPU differential has run.

Then register the path's name where the code and the tests list paths:

- `aiter/moev2/paths.py`: `PLANNED` (`AITER_MOEV2` rejects any other name) and
  `_BUILTIN_MODULES`, which imports the module on first use;
- `tests/test_moev2_dispatch.py`: `_LOCAL_PATHS`, the call factory of each path (the
  `call = {...}` dict; a missing path is a `KeyError`) and the `owners` list;
- `tests/moe_dispatch/gpu_suites.py`: `PATHS` (`--path` accepts only these);
- `tests/moe_dispatch/gpu_differential.py`: `_CASES`, the inputs of the path's format
  in `_make_call`, and for `--model-table` the path's quant type in `_ROWS`, `row_path`
  and `_row_case`;
- `tests/moe_dispatch/cross_checkout.py`: `PATHS`;
- ranking entries for the keys it takes, or `tests/moe_dispatch/known_declines.py`
  entries; otherwise `coverage_checks.py` reports each key legacy runs as a gap.

### 5.5 Decide how dispatch reaches it: `reach`

`reach` says how dispatch gets to the segment without a tuned row.
`coverage_checks.py --check` enforces it:

| `reach` | Meaning | Checked |
|---|---|---|
| `"tuned"` | Only tuned rows run it, as in legacy | It is in no ranking entry, and some shipped row runs it |
| `"hibernating"` | Nothing ships that runs it; only a user table (`AITER_MOEV2_TUNED_TABLE`) or a test | It is in no ranking entry, no shipped row runs it, and it has a GPU test in `tests/moe_dispatch/hibernating.py` |
| `"ranked"` | Some default-ranking structure lists it | It is in a ranking entry |

A v2-only family with no shipped rows starts as `"hibernating"`. Register its GPU test:

```python
# tests/moe_dispatch/hibernating.py
TESTS = {"acme.gemm2": "tests/test_moev2_acme.py::test_acme_gemm2_matches_reference"}
```

Once rows naming it ship (legacy branch added, section 1), it becomes `"tuned"`.

**Ranking it** (making it a default without a tuned row) is a behavior change, and waits
until the migration is complete; it then comes as a PR of its own. Today the
ranking table only translates legacy: rule R10 (in the docstring of `ranking_table.py`)
requires each entry to port one legacy `get_2stage_cfgs` branch 1:1, and rule R8
requires every region to be reached by some key of the grid. A region for keys legacy
does not serve breaks the CPU differential instead: there v2 would plan a key on which
legacy raises. A provider may propose where a new structure should rank; the refit
report (`aiter/moev2/tuning/refit_ranking.py`, GPU) measures how good a position is
against the tuner's best plans.

### 5.6 Tests

**CPU unit tests.** Add a `tests/test_moev2_acme.py` (or extend the closest
`tests/test_moev2_*.py`), run with `HIP_VISIBLE_DEVICES=` (section 7). Test:

- the grammar: names parse, round-trip, and are not claimed by another family;
- `owns`: your names only, stage 2 only;
- the `Cap`s: a problem you support is accepted, and each unsupported field rejects with
  a readable reason (`segment.reject_reason(p)`);
- the chain: the plans of `resolver.feasible_plans(p, SEGMENTS)` (it returns
  `(plans, rejected)`) contain your structure for a supported problem, and not for an
  unsupported one;
- tuned rows: with a one-row v2 plan file naming your structure and an Acme candidate
  (`AITER_MOEV2_TUNED_TABLE`; `tests/test_moev2_tuning.py` writes such a row with
  `cli._plan_row` and `tuned_plans.COLUMNS`), the call at that exact key resolves to
  your structure, and a candidate outside
  `param_space(p)` is refused with its warning. `resolver.build(p, structure,
  candidates)` builds a plan from chosen candidates directly. Once legacy-format rows
  name the family (a legacy branch exists), test also that `check_row` refuses what it
  should and that `row_params` reads the row;
- `param_space`: lists the instances the kernel supports, and nothing for problems it
  refuses (`tunable`, `accepts`);
- `build`: binds the kernel name.

Make each test fail if you break the line it is about (delete a `Cap` field, return the
wrong stage in `owns`, drop a `check_row` condition): a test that still passes then
proves nothing.

**GPU test of the kernel.** Compare its output with the torch reference
(`aiter/moev2/reference/torch_moe.py`) on a few shapes, through
`fused_moe` with `AITER_MOEV2=<path>` and a one-row tuned table naming your kernel.
For a hibernating segment this is the test listed in `hibernating.py`. It sits next to
the CPU tests, so it must skip when no GPU is visible (`if not
torch.cuda.is_available(): pytest.skip("needs a GPU")`).

**GPU differential**, when legacy runs the family too: add cases to the path's tuple in
`tests/moe_dispatch/gpu_differential.py` (`_CASES`), or tuned rows to a shipped table,
which `--model-table` picks up. It compares, per case, legacy and v2: the dispatch log
lines (of a call after a warm-up, with the cached dispatch decisions cleared), the
errors, every kernel launch (function, scalar arguments, tensor dtypes, shapes and
values; not the output buffers, nor the values of a split-K stage 1's output, made
from partials it accumulates with atomics, and of stage 2's activation and scale
made from it) and the outputs:

```bash
python tests/moe_dispatch/gpu_differential.py --path a16w16
python tests/moe_dispatch/gpu_differential.py --path a16w16 --model-table all
```

Each case ends `ok`, `FAIL`, `KNOWN` (v2 declines it as `known_declines.py` lists it) or
`LEGACY`: legacy's own runs explain every difference of v2 from legacy's first run, so
the case is undecided. That is, v2's launch arguments differ only at elements where
another legacy run differs too, and each element of its output is no farther off than
twice what legacy's runs vary there, or than a bound for its row of 4 to 16 ulps of the
row's peak, taken over the elements where legacy's runs agree within 16 ulps (some
legacy kernels race or read memory nothing wrote; GEMM2's atomics vary by a few ulps).
`ok` needs equal launches, and an output within `_check_output` and within the same
check against legacy's first two runs. The exit status
is 1 on a failure or a skipped case, else 2 on an undecided case, else 0. A `LEGACY`
case says nothing about v2 where legacy varies: find the legacy kernel that varies (run
legacy alone). A `FAIL` stands; rerunning it only helps to diagnose it.

### 5.7 The kernel-name map

`aiter/moev2/kernel_name_map.csv` lists every legacy kernel-name pattern with its v2
segment, v2 name, parameters and status. No code reads it, but reviewers do. A v2-only
family has no legacy name and needs no row. When you add the legacy branch, add a row
for the family:

```text
acme_moe2_*,2,,acme.gemm2,(same),block_m=the row's,mapped: an Acme bf16 stage 2 instance
```

---

## 6. Tune it

The v2 tuner needs no change for a new segment: it reads `param_space`.

```bash
python -m aiter.moev2.tuning my_keys.csv -o my_plans.csv --paths a16w16 --devices 0,1,2,3
```

- `my_keys.csv` is in the legacy untuned format (`token,model_dim,inter_dim,expert,
  topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1`), or use
  `--keys-from <tuned csv>` to retune the keys of a tuned table.
- `my_plans.csv` is a v2 plan file: one row per (key, path), the winning structure
  (segment names) and each tunable segment's candidate. `<output>.log.csv` holds every
  structure measured, its time, and why the others failed.
- Use the plan file directly with `AITER_MOEV2_TUNED_TABLE` (section 1).
- The legacy `TUNE_ONLY` filter keeps working; a new family's segments are left out of
  it unless you add them to `TUNE_ONLY_SEGMENTS` in `aiter/moev2/tuning/candidates.py`.

For rows to ship to every user, tune with the legacy tuner too (it must know the family:
the legacy side of section 1), ship the rows in the legacy table, and regenerate the v2
copy with `convert_tables.py --update`.

`param_space_checks.py --check` compares v2's candidates with the legacy tuner's
generators, but only for the families its `_families()` lists: a new segment is not
checked until you add a `Family` there. Its known differences excuse only candidates
the legacy side lists and v2 does not; a candidate only v2 lists is always a failure. So
the legacy side of a new `Family` must rebuild your candidates independently (as
several families there already do; see that file's docstring). A v2-only family has no legacy
generator to compare with, and the check does not cover it.

---

## 7. Checks before you send the PR

CPU, no GPU needed: the unit tests and nine gates (each must pass; outputs of the gates
must not change for keys your change does not mean to touch). The gates hide the GPUs
themselves. pytest runs the GPU tests too when a GPU is visible; an empty
`HIP_VISIBLE_DEVICES` hides them, so that only the CPU tests run and the GPU tests skip:

```bash
HIP_VISIBLE_DEVICES= python -m pytest -q tests/test_moev2_*.py
for gate in differential coverage_checks key_checks ranking_checks build_checks \
            param_space_checks row_params_checks snapshot; do
    python tests/moe_dispatch/$gate.py --check || echo "FAILED: $gate"
done
python tests/moe_dispatch/convert_tables.py --check
black aiter/moev2 tests/moe_dispatch tests/test_moev2_*.py
ruff check aiter/moev2 tests/moe_dispatch tests/test_moev2_*.py
```

What each gate guards:

| Gate | Fails when |
|---|---|
| `differential.py` | v2 and legacy dispatch a grid key differently (any path) |
| `snapshot.py` | legacy's dispatch decisions changed |
| `coverage_checks.py` | a key legacy runs has no v2 plan (unless in `known_declines.py`), or a segment's `reach` is wrong |
| `key_checks.py` | a key's problem does not round-trip to the legacy key, or v2's tuned-table lookup gives other cells than legacy's |
| `ranking_checks.py` | the ranking table breaks a rule, or `default_plan` of the pick differs from `resolve` |
| `build_checks.py` | building a shipped row's chosen candidates (`resolver.build`) differs from what `resolve` runs for the row |
| `convert_tables.py` | the converted tables under `aiter/configs/moev2/` differ from a fresh conversion of the legacy tables |
| `param_space_checks.py` | v2's candidates differ from the legacy tuner's (beyond the known differences) |
| `row_params_checks.py` | the counts of shipped rows whose choice is not one of the segment's candidates differ, in either direction, from `golden/row_params.json` |

GPU, on each arch the path claims:

```bash
python tests/moe_dispatch/gpu_differential.py --path <path>             # built-in cases
python tests/moe_dispatch/gpu_differential.py --path <path> --model-table all
python tests/moe_dispatch/gpu_suites.py --path <path> --compare --log-dir /tmp/suites
```

`gpu_suites --compare` runs the path's op tests with `AITER_MOEV2=off` and with the path,
and fails where v2 fails and legacy passes.

---

## 8. Debugging

- **`AITER_MOE_EXPLAIN=1`**: per new problem, the problem, the feasible structures, each
  rejected segment with its reason (the first `Cap` field that rejects, or the boundary
  that does not match: `reads <buf>, gets <buf>`, or `reads <buf>, which no earlier
  segment produces`), and the choice.
- **"no feasible plan"**: no chain of the path's segments covers the problem. Read the
  explain output: usually a `Cap` field, or a boundary your segment reads that nothing
  before it produces.
- **A shipped tuned row does not run**: `coverage_checks.py --check -v` lists each
  shipped row (of the frozen copy, section 3) that does not run, with why: discarded by
  a `check_row`, or declined. For a row of your own plan file, the warning
  "moev2 tuned plan ... not used" gives the reason (logged with or without
  `AITER_MOE_EXPLAIN`).
- **"metadata_fields sets [...], which boundaries set"**: your segment sets
  `fuse_quant`, `skip_inter_quant` or `output_aux`; derive them from `reads`/`produces`
  instead.
- **The resolver says the chosen kernels disagree** (`plan_mismatch`): two segments
  disagree on a `structure_params` value, such as `block_m`, or a `Buf` a segment reads
  with its chosen kernel (`reads_with`) does not match what the segment before it
  produces.
- **v2 declined the call** (it ran on legacy): `aiter.moev2.paths.outcomes()` counts
  served / declined / unclaimed calls per path, and `AITER_MOEV2_TRACE=<file>` writes one
  JSON line per distinct outcome and call shape, from every process.

---

## 9. Checklist

- [ ] Kernel wrapper under `aiter/ops/` (one copy, shared with legacy).
- [ ] Decided where it lands (section 1): legacy branch too, or v2-only with no shipped
      rows.
- [ ] `kernel_names.py`: family constant, pattern, parser, formatter, `FAMILIES`,
      `_FORMATTERS`, `parse`; names round-trip.
- [ ] Segment: `name`, `prefixes`, `covers`, `capabilities`, `reach`, `owns`, `reads`,
      `produces`, `build`; `check_row`, `row_params`, `param_space` as needed;
      `structure_params` if it fixes `block_m` or another boundary value;
      `default_params` only if ranked.
- [ ] Added to the path's `SEGMENTS` (or a new path with `claims` / `takes` / `ARCHS`).
- [ ] `reach` holds: `hibernating.py` entry for a hibernating segment.
- [ ] Dispatch unchanged for keys legacy serves (the CPU differential clean).
- [ ] `kernel_name_map.csv` row.
- [ ] CPU unit tests that fail when the code they cover is broken; GPU test against the
      torch reference; GPU differential cases where legacy runs it too.
- [ ] CPU suite, the nine gates, black and ruff; GPU differential and suites on each arch.
