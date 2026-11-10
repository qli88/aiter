# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`python -m aiter.moev2.tuning`: tune keys with the two-level search, into the v2
plan file (`dispatch.tuned_plans`), which v2 dispatch reads while
`AITER_MOEV2_TUNED_PLANS` names it.

Keys come from CSVs in the legacy tuner's input format (`untuned_fmoe.csv` columns:
token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w,
q_type, use_g1u1, doweight_stage1), or from tuned tables (`--keys-from`; rows of
another gfx or CU count are skipped). Values are read as written, as the legacy tuner
does. A key is the call's key as dispatch looks it up, with the defaults of a plain call
(shuffled weights, no bias, no expert mask); `--fhmoe` reads FHMoE keys (`gate_mode`
from the row, else interleaved for fp8 activations), keyed in the FHMoE table
(`--fhmoe-table`) where dispatch reads it (the DSV4 I384 contract), else in the main
table. Rows with an empty `gfx` or `cu_num` are this GPU's. Each key is tuned on every registered path taking it. Keys already in the
output file (with that path) are skipped, so a stopped run resumes; to retune a key,
write another file. The output file gets each key's winner per path, nothing else; a
key with no winner gets no row, so a resumed run tunes it again. Every structure
measured (its time, rank and level-1 sum) and why the others did not run go to the
log file (`--log-file`; default: the output's path with the suffix `.log.csv`),
which nothing reads. A file there that is not such a log (other columns, unreadable)
is left as it is, and the log goes to `<name>.1` (the first such name that is new or
such a log). The candidate filters are the legacy tuner's env (`TUNE_ONLY`,
`AITER_FMOE_TUNE_EXCLUDE_KERNELS`, ...; `candidates.Filters.from_env`).
"""

import argparse
import contextlib
import csv
import os
import pathlib
import sys
import time

import torch

from aiter import ActivationType, QuantType
from aiter.ops.flydsl.moe_common import GateMode

from .. import api, paths
from ..dispatch import tuned_plans
from ..segments.flydsl_fhmoe import _is_dsv4_i384_fhmoe_contract
from . import task as tuning_task
from .candidates import Filters
from .pool import Pool
from .search import Settings, tune

KEY_FIELDS = (
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
LOG_COLUMNS = tuned_plans.KEY_COLUMNS + (
    "path",
    "structure",
    "candidates",
    "rank",
    "e2e_us",
    "mode",
    "cov",
    "sum_us",
    "reason",
    "notes",
)


def _dtype(text):
    return getattr(torch, str(text).strip().replace("torch.", ""))


def _enum(cls, text):
    return getattr(cls, str(text).strip().split(".")[-1])


def problem_of(row, arch, cu_num, fhmoe_table=None):
    """The `MoeProblem` of an input row (a dict of strings); `fhmoe_table`: the row is
    an FHMoE key, of that table where dispatch reads it."""
    missing = [f for f in KEY_FIELDS if f not in row]
    if missing:
        raise ValueError(f"row without {missing}")
    dtype = _dtype(row["dtype"])
    q_dtype_a = _dtype(row["q_dtype_a"])
    activation = _enum(ActivationType, row["act_type"])
    args = (
        int(row["token"]),
        int(row["model_dim"]),
        int(row["inter_dim"]),
        int(row["expert"]),
        int(row["topk"]),
        dtype,
        q_dtype_a,
        _dtype(row["q_dtype_w"]),
        _enum(QuantType, row["q_type"]),
        bool(int(row["use_g1u1"])),
        activation,
        bool(int(row["doweight_stage1"])),
        int(row.get("hidden_pad") or 0),
        int(row.get("intermediate_pad") or 0),
    )
    kwargs = {"input_dtype": dtype}
    if fhmoe_table is not None:
        if row.get("gate_mode"):
            gate_mode = _enum(GateMode, row["gate_mode"])
        else:
            fp8 = q_dtype_a in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            gate_mode = GateMode.INTERLEAVE if fp8 else GateMode.SEPARATED
        shape, pads = args[1:5], args[12:14]
        doweight, q_dtype_w = args[11], args[7]
        dedicated = _is_dsv4_i384_fhmoe_contract(
            *shape,
            *pads,
            gate_mode,
            doweight,
            routed_mxfp4=q_dtype_w == torch.float4_e2m1fn_x2,
        )
        kwargs.update(
            gate_mode=gate_mode, config_file=fhmoe_table if dedicated else None
        )
    return api.problem_from_key(
        args, kwargs, arch, cu_num, fhmoe=fhmoe_table is not None
    )


def read_keys(files, arch, cu_num, fhmoe_table=None, log=print):
    """`[(label, MoeProblem)]` of the rows of `files`, in order, without repeats; rows of
    another gfx or CU count are skipped."""
    seen, keys, other = set(), [], 0
    for name in files:
        with open(name, newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                gfx, cus = row.get("gfx") or arch, int(row.get("cu_num") or cu_num)
                if gfx != arch or cus != cu_num:
                    other += 1
                    continue
                problem = problem_of(row, arch, cu_num, fhmoe_table)
                if problem not in seen:
                    seen.add(problem)
                    keys.append((f"{name}:{n}", problem))
    if other:
        log(f"skipped {other} rows of another gfx or CU count")
    return keys


def paths_taking(problem, only=None):
    return [
        p.name
        for p in paths.registered().values()
        if p.segments
        and p.takes is not None
        and (not only or p.name in only)
        and p.takes(problem)
    ]


def _log_row(problem, path, outcome) -> dict:
    row = dict(zip(tuned_plans.KEY_COLUMNS, tuned_plans.key_values(problem)))
    e2e = outcome.e2e
    row.update(
        path=path,
        structure=tuned_plans.STRUCTURE_SEPARATOR.join(outcome.structure),
        candidates=(
            tuned_plans.encode_candidates(outcome.candidates)
            if outcome.candidates is not None
            else ""
        ),
        rank="" if outcome.rank is None else outcome.rank,
        e2e_us="" if e2e is None else f"{e2e.median_us:.3f}",
        mode="" if e2e is None else e2e.mode,
        cov="" if e2e is None else f"{e2e.cov:.4f}",
        sum_us="" if outcome.sum_us is None else f"{outcome.sum_us:.3f}",
        reason=outcome.reason or "",
        notes=outcome.notes,
    )
    return row


def _plan_row(problem, path, outcome) -> dict:
    row = _log_row(problem, path, outcome)
    return {c: row[c] for c in tuned_plans.COLUMNS}


def _new_file(output) -> bool:
    return not os.path.exists(output) or os.path.getsize(output) == 0


def _done(output) -> set:
    if not os.path.exists(output):
        return set()
    with open(output, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(tuned_plans.COLUMNS):
            raise ValueError(f"{output} has other columns than a v2 plan file")
        return {
            (tuple(r[c] for c in tuned_plans.KEY_COLUMNS), r["path"]) for r in reader
        }


@contextlib.contextmanager
def _open(name, columns):
    """`name` opened to append rows of `columns`, with their header when new."""
    new = _new_file(name)
    pathlib.Path(name).parent.mkdir(parents=True, exist_ok=True)
    with open(name, "a", newline="") as out:
        if new:
            csv.DictWriter(out, columns).writeheader()
        yield out


def _is_log(name) -> bool:
    if _new_file(name):
        return True
    try:
        with open(name, newline="") as f:
            return next(csv.reader(f), None) == list(LOG_COLUMNS)
    except (OSError, ValueError, csv.Error):
        return False


def _open_log(name, log):
    actual, n = name, 0
    while not _is_log(actual):
        n += 1
        actual = f"{name}.{n}"
    if actual != name:
        log(f"{name} is not a log of these columns: logging to {actual}")
    return _open(actual, LOG_COLUMNS)


def default_devices() -> list:
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
    if visible:
        return [d.strip() for d in visible.split(",") if d.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m aiter.moev2.tuning", description=__doc__.splitlines()[0]
    )
    parser.add_argument("inputs", nargs="*", help="keys, legacy untuned CSV format")
    parser.add_argument("--keys-from", nargs="*", default=[], help="tuned CSVs")
    parser.add_argument("-o", "--output", required=True, help="the v2 plan file")
    parser.add_argument("--log-file", help="default: OUTPUT with the suffix .log.csv")
    parser.add_argument("--fhmoe", action="store_true", help="the keys are FHMoE's")
    parser.add_argument("--fhmoe-table", help="default: AITER_CONFIG_FHMOE_FILE")
    parser.add_argument("--paths", nargs="*", help="only these paths")
    parser.add_argument("--devices", help="HIP device ids, comma-separated")
    parser.add_argument("--timeout", type=float, default=600.0, help="per task, s")
    defaults = Settings()
    for field, value in vars(defaults).items():
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=type(value), default=value
        )
    args = parser.parse_args(argv)
    if not args.inputs and not args.keys_from:
        parser.error("no keys: give input CSVs or --keys-from")

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from ..migration import fhmoe

    def log(line):
        print(f"[moev2 tune] {line}", file=sys.stderr, flush=True)

    arch, cu_num = get_gfx(), get_cu_num()
    table = None
    if args.fhmoe:
        table = args.fhmoe_table or fhmoe._fhmoe_config_file()
    keys = read_keys(args.inputs + args.keys_from, arch, cu_num, table, log)
    settings = Settings(**{f: getattr(args, f) for f in vars(defaults)})
    filters = Filters.from_env()
    devices = (
        [d.strip() for d in args.devices.split(",") if d.strip()]
        if args.devices
        else default_devices()
    )
    if args.log_file is None:
        args.log_file = str(pathlib.Path(args.output).with_suffix("")) + ".log.csv"
    if os.path.realpath(args.log_file) == os.path.realpath(args.output):
        parser.error("--log-file is the output file")
    done = _done(args.output)
    log(f"{arch} cu{cu_num}: {len(keys)} keys, devices {devices}")
    pool = Pool(tuning_task.run, devices=devices, timeout=args.timeout)
    with pool, _open(args.output, tuned_plans.COLUMNS) as out, _open_log(
        args.log_file, log
    ) as log_out:
        writer = csv.DictWriter(out, tuned_plans.COLUMNS)
        log_writer = csv.DictWriter(log_out, LOG_COLUMNS)

        def evaluate(tasks):
            return [r for _, r in sorted(pool.map(tasks), key=lambda t: t[0])]

        for label, problem in keys:
            for path in paths_taking(problem, args.paths) or [None]:
                if path is None:
                    log(f"{label}: no registered path takes it")
                    continue
                if (tuned_plans.key_values(problem), path) in done:
                    log(f"{label} {path}: already in {args.output}")
                    continue
                start = time.time()
                outcomes = tune(
                    problem,
                    path,
                    evaluate,
                    settings,
                    filters,
                    lambda line, k=label, p=path: log(f"{k} {p}: {line}"),
                )
                writer.writerows(
                    _plan_row(problem, path, o) for o in outcomes if o.rank == 1
                )
                out.flush()
                log_writer.writerows(_log_row(problem, path, o) for o in outcomes)
                log_out.flush()
                win = next((o for o in outcomes if o.rank == 1), None)
                summary = (
                    f"{tuned_plans.STRUCTURE_SEPARATOR.join(win.structure)} "
                    f"{win.e2e.median_us:.1f} us"
                    if win is not None
                    else f"no plan ({outcomes[0].reason})"
                )
                log(f"{label} {path}: {summary} ({time.time() - start:.0f} s)")
    return 0
