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

`--comm-fused` tunes TP keys instead (`comm_search`, path `comm_fused`): each row's
comm modes (`--comm`, else those of its `comm_fused_configs`) with `--tp` ranks and
the shared add (`--add-shared`, else the row's), keyed also on this GPU's model. Level
1 of the local key runs on every device, the collective runs on a group of the first
`--tp`. Runners start from the comm_fused winners of the key in the input tables and
`--seeds-from` at token counts within `--seed-span` times M, plus their families'
defaults; `--runner-axis FAMILY:FIELD=V1,V2` adds configs.
"""

import argparse
import contextlib
import csv
import functools
import io
import os
import pathlib
import signal
import sys
import time

import torch

from ..dispatch import tuned_plans
from ..dispatch.convert import paths_taking, problem_of
from . import collective, comm_search
from . import task as tuning_task
from .candidates import Filters
from .group import Group
from .pool import Pool
from .search import Settings, tune

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


def _log_row(keys, path, outcome) -> dict:
    row = dict(zip(tuned_plans.KEY_COLUMNS, keys))
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


def _plan_row(keys, path, outcome) -> dict:
    row = _log_row(keys, path, outcome)
    return {c: row[c] for c in tuned_plans.COLUMNS}


def _key_writes(keys, path, outcomes, out, log_out) -> list:
    """The writes (`_append`'s) of a key's `outcomes`: its winner to the plan file
    `out`, then every outcome to the log file `log_out`."""
    return [
        (
            out,
            tuned_plans.COLUMNS,
            [_plan_row(keys, path, o) for o in outcomes if o.rank == 1],
        ),
        (log_out, LOG_COLUMNS, [_log_row(keys, path, o) for o in outcomes]),
    ]


def _new_file(output) -> bool:
    return not os.path.exists(output) or os.path.getsize(output) == 0


def _done(output) -> set:
    if _new_file(output):
        return set()
    with open(output, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(tuned_plans.COLUMNS):
            missing = [c for c in tuned_plans.COLUMNS if c not in reader.fieldnames]
            raise ValueError(
                f"{output} has other columns than a v2 plan file"
                + (f" (no {', '.join(missing)}): write a new file" if missing else "")
            )
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
            out.flush()
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


def _append(writes):
    """Write each `(out, columns, rows)` of `writes` (a key's) in one write and flush
    them before handling a SIGTERM or SIGINT that arrives meanwhile, so a stopped run
    leaves no partial key."""
    texts = []
    for out, columns, rows in writes:
        if not rows:
            continue
        text = io.StringIO()
        csv.DictWriter(text, columns).writerows(rows)
        texts.append((out, text.getvalue()))
    # A thread's signal mask does not hold a process-wide signal another thread takes,
    # while Python handlers all run on the main thread, as this does.
    held = []
    previous = {
        signum: signal.signal(signum, lambda signum, frame: held.append(signum))
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for out, text in texts:
            out.write(text)
            out.flush()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)
    if held:
        signal.raise_signal(held[0])


def _table_rows(name) -> tuple:
    stat = os.stat(name)
    return _read_rows(name, stat.st_mtime_ns, stat.st_size)


@functools.cache
def _read_rows(name, mtime_ns, size) -> tuple:
    with open(name, newline="") as f:
        return tuple(csv.DictReader(f))


def _same_key(row, local, arch, cu_num) -> bool:
    """Whether `row` is `local`'s key at another token count."""
    gfx, cus = row.get("gfx") or arch, int(row.get("cu_num") or cu_num)
    if gfx != arch or cus != cu_num:
        return False
    try:
        other = problem_of(row, arch, cu_num)
    except (ValueError, KeyError, AttributeError):
        return False
    return tuned_plans.any_token(
        tuned_plans.key_values(other)
    ) == tuned_plans.any_token(tuned_plans.key_values(local))


def read_seeds(files, problem, arch, cu_num) -> list:
    """The `comm_search.Seed`s of `problem` (a `CommProblem`): the comm_fused winners
    in the rows of `files` of its key at any token count, of its TP size, comm mode and
    shared add, any GPU model, in file order."""
    from aiter.ops.flydsl.comm_fused_moe_host import _comm_configs

    seeds = {}
    for name in files:
        for row in _table_rows(name):
            entry = _comm_configs(row).get(problem.comm)
            if (
                entry is None
                or int(entry["tp"]) != problem.tp
                or bool(entry["add_shared"]) != problem.add_shared
                or not _same_key(row, problem.local, arch, cu_num)
            ):
                continue
            seed = comm_search.Seed(
                collective.spec_of_record(entry),
                int(row["token"]),
                str(entry["gpu_model"]).strip().lower(),
                float(entry["us"]) if entry.get("us") is not None else None,
            )
            seeds.setdefault((seed.spec, seed.token, seed.gpu_model), seed)
    return list(seeds.values())


def read_comm_keys(files, arch, cu_num, tp, gpu_model, comms=None, add_shared=None):
    """`[(label, CommProblem)]` of the rows of `files`: each row's comm modes
    (`comms`, for every row; else those of its `comm_fused_configs` entries for `tp`),
    with the shared add of `add_shared`, else of its entry for `tp` (else added), in order,
    without repeats."""
    from aiter.ops.flydsl.comm_fused_moe_host import _comm_configs

    seen, keys = set(), []
    for name in files:
        with open(name, newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                gfx, cus = row.get("gfx") or arch, int(row.get("cu_num") or cu_num)
                if gfx != arch or cus != cu_num:
                    continue
                entries = {
                    c: e
                    for c, e in _comm_configs(row).items()
                    if int(e.get("tp", tp)) == tp
                }
                for comm in comms or sorted(entries):
                    shared = add_shared
                    if shared is None:
                        shared = bool(entries.get(comm, {}).get("add_shared", True))
                    problem = collective.CommProblem(
                        problem_of(row, arch, cu_num), tp, comm, shared, gpu_model
                    )
                    if problem not in seen:
                        seen.add(problem)
                        keys.append((f"{name}:{n} {comm}", problem))
    return keys


def _value(text):
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        return text.strip()


def parse_axes(specs) -> dict:
    """`{family: {field: values}}` of `FAMILY:FIELD=V1,V2` specs."""
    axes = {}
    for spec in specs:
        head, sep, values = spec.partition("=")
        family, colon, field = head.partition(":")
        if not sep or not colon or not values.strip():
            raise ValueError(f"runner axis {spec!r} is not FAMILY:FIELD=V1,V2")
        axes.setdefault(family.strip(), {})[field.strip()] = tuple(
            _value(v) for v in values.split(",")
        )
    return axes


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
    comm = parser.add_argument_group(
        "comm_fused",
        "--comm-fused tunes each key's TP call with its collective (path comm_fused): "
        "the a8w4 plan with a comm_fused runner as its stage 2, against the plan "
        "dispatch resolves then the TP collective, on a group of --tp devices",
    )
    comm.add_argument("--comm-fused", action="store_true")
    comm.add_argument("--tp", type=int, default=8)
    comm.add_argument(
        "--comm",
        nargs="*",
        choices=collective.COMMS,
        help="these modes on every row; default: the modes of the row's entries for --tp",
    )
    comm.add_argument(
        "--add-shared",
        type=int,
        choices=(0, 1),
        help="default: the row's entry (else 1)",
    )
    comm.add_argument(
        "--seeds-from", nargs="*", default=[], help="more tables to take runners from"
    )
    comm.add_argument(
        "--runner-axis",
        action="append",
        default=[],
        metavar="FAMILY:FIELD=V1,V2",
        help="runner configs to try: each value of FIELD on each runner of FAMILY",
    )
    defaults = comm_search.CommSettings()
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
    if args.comm_fused:
        return _main_comm(args, arch, cu_num, filters, devices, log)
    table = None
    if args.fhmoe:
        table = args.fhmoe_table or fhmoe._fhmoe_config_file()
    keys = read_keys(args.inputs + args.keys_from, arch, cu_num, table, log)
    settings = Settings(**{f: getattr(args, f) for f in vars(Settings())})
    done = _done(args.output)
    log(f"{arch} cu{cu_num}: {len(keys)} keys, devices {devices}")
    pool = Pool(tuning_task.run, devices=devices, timeout=args.timeout)
    with pool, _open(args.output, tuned_plans.COLUMNS) as out, _open_log(
        args.log_file, log
    ) as log_out:

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
                row_keys = tuned_plans.key_values(problem)
                _append(_key_writes(row_keys, path, outcomes, out, log_out))
                log(
                    f"{label} {path}: {_summary(outcomes)} ({time.time() - start:.0f} s)"
                )
    return 0


def _summary(outcomes) -> str:
    win = next((o for o in outcomes if o.rank == 1), None)
    if win is None:
        return f"no plan ({outcomes[0].reason})"
    runner = ""
    if win.candidates and comm_search.RUNNER in win.candidates:
        runner = f" + {win.candidates[comm_search.RUNNER]['family']} runner"
    elif comm_search.ORDINARY_NOTE in win.notes:
        runner = " (ordinary)"
    structure = tuned_plans.STRUCTURE_SEPARATOR.join(win.structure)
    return f"{structure}{runner} {win.e2e.median_us:.1f} us"


def _main_comm(args, arch, cu_num, filters, devices, log):
    from aiter.ops.flydsl.comm_fused_moe_host import get_gpu_model

    if len(devices) < args.tp:
        raise SystemExit(f"--tp {args.tp} needs {args.tp} devices, have {devices}")
    group_devices = devices[: args.tp]
    gpu_model = get_gpu_model()
    if not gpu_model:
        raise SystemExit("unknown GPU model: the comm_fused rows key on it")
    tables = args.inputs + args.keys_from
    add_shared = None if args.add_shared is None else bool(args.add_shared)
    keys = read_comm_keys(
        tables, arch, cu_num, args.tp, gpu_model, args.comm, add_shared
    )
    axes = parse_axes(args.runner_axis)
    settings = comm_search.CommSettings(
        **{f: getattr(args, f) for f in vars(comm_search.CommSettings())}
    )
    done = _done(args.output)
    log(
        f"{arch} cu{cu_num} {gpu_model}: {len(keys)} TP{args.tp} keys, "
        f"group {group_devices}"
    )
    path = tuned_plans.COMM_FUSED_PATH
    pending = []
    for label, problem in keys:
        row_keys = tuned_plans.comm_key_values(
            problem.local, problem.tp, problem.comm, problem.add_shared, gpu_model
        )
        if (row_keys, path) in done:
            log(f"{label}: already in {args.output}")
            continue
        local_paths = paths_taking(problem.local, ["a8w4"])
        if not local_paths:
            log(f"{label}: the a8w4 path does not take its local key")
            continue
        pending.append((label, problem, row_keys, local_paths[0]))
    if not pending:
        return 0
    with contextlib.ExitStack() as stack:
        pool = stack.enter_context(
            Pool(tuning_task.run, devices=devices, timeout=args.timeout)
        )
        group = stack.enter_context(
            Group(
                collective.run,
                group_devices,
                timeout=args.timeout,
                setup=collective.setup,
                teardown=collective.teardown,
            )
        )
        out = stack.enter_context(_open(args.output, tuned_plans.COLUMNS))
        log_out = stack.enter_context(_open_log(args.log_file, log))

        def evaluate_local(tasks):
            return [r for _, r in sorted(pool.map(tasks), key=lambda t: t[0])]

        def evaluate_group(tasks):
            return [
                group.run(i, task, timeout=args.timeout * len(_plans_of(task)))
                for i, task in enumerate(tasks)
            ]

        for label, problem, row_keys, local_path in pending:
            start = time.time()
            outcomes = comm_search.tune(
                problem,
                local_path,
                evaluate_local,
                evaluate_group,
                read_seeds(tables + args.seeds_from, problem, arch, cu_num),
                axes,
                settings,
                filters,
                lambda line, k=label: log(f"{k}: {line}"),
            )
            _append(_key_writes(row_keys, path, outcomes, out, log_out))
            log(f"{label}: {_summary(outcomes)} ({time.time() - start:.0f} s)")
    return 0


def _plans_of(task) -> tuple:
    return task.plans if isinstance(task, collective.CommRivals) else (task,)
