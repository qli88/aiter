# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Legacy MoE dispatch snapshot: generate, check or update the golden files.

    python tests/moe_dispatch/snapshot.py --check            # compare with golden/
    python tests/moe_dispatch/snapshot.py --update           # rewrite golden/
    python tests/moe_dispatch/snapshot.py --arch gfx950 --out /tmp/snap
    python tests/moe_dispatch/snapshot.py --refresh-inputs   # re-freeze tuned tables

Runs on CPU: GPUs are hidden and dispatch-affecting environment variables are dropped
(arch_sim.prepare_process). Legacy dispatch reads a frozen copy of the tuned tables
(golden/inputs/), v2 dispatch its conversion, so tuning PRs do not change the snapshot;
only dispatch logic does.
Re-freeze the inputs when new kinds of tuned rows should be covered, in a PR of its own.

One gzip-compressed JSON-lines file per simulated arch, sorted by key. A call variant
whose result equals the `base` variant is stored as "=base".
"""

import argparse
import gzip
import io
import json
import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
GOLDEN_DIR = _HERE / "golden"
INPUTS_DIR = GOLDEN_DIR / "inputs"
FORMAT_VERSION = 1
# Legacy dispatch reads these, v2 dispatch their conversion (`FROZEN_CONVERTED`,
# checked by convert_tables.py).
FROZEN_TABLES = {"fmoe": "tuned_fmoe.csv", "fhmoe": "tuned_fhmoe.csv"}
FROZEN_CONVERTED = {"fmoe": "tuned_fmoe.moev2.csv", "fhmoe": "tuned_fhmoe.moev2.csv"}
SAME_AS_BASE = "=base"


def read_input(name) -> bytes:
    return gzip.decompress((INPUTS_DIR / f"{name}.gz").read_bytes())


def write_input(name, data: bytes):
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(INPUTS_DIR / f"{name}.gz", "wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", mtime=0
    ) as f:
        f.write(data)


def _setup(frozen=True, env=()):
    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import arch_sim

    if not frozen:
        arch_sim.prepare_process(env=env)
        return
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="moe_dispatch_inputs_"))
    for name in (*FROZEN_TABLES.values(), *FROZEN_CONVERTED.values()):
        (tmp / name).write_bytes(read_input(name))
    arch_sim.prepare_process(
        {k: tmp / name for k, name in FROZEN_TABLES.items()},
        [tmp / name for name in FROZEN_CONVERTED.values()],
        env,
    )


def _variant_label(label, variant):
    head, current, rest = label.split("|", 2)
    return f"{head}|{variant}|{rest}" if current != variant else label


def dispatch_keys(gfx, cu_num):
    """`get_2stage_cfgs` keys for one arch; call inside `arch_sim.simulated_arch`."""
    from aiter.utility import dtypes
    from moe_dispatch import keys

    table = keys.tuned_table()
    return (
        keys.tuned_keys(gfx, cu_num, dtypes.fp8, table)
        + keys.fhmoe_keys(gfx, cu_num, dtypes.fp8)
        + keys.grid_keys(dtypes.fp8, table)
    )


def resolve_same_as_base(records, label):
    rec = records.get(label)
    return records.get(_variant_label(label, "base")) if rec == SAME_AS_BASE else rec


def generate(arch):
    """Return {label: record} for one simulated arch."""
    import logging

    import aiter.fused_moe as fm
    from moe_dispatch import arch_sim, keys, record

    logging.getLogger("aiter").setLevel(logging.ERROR)
    out = {}
    with arch_sim.simulated_arch(arch) as (gfx, cu_num):
        from aiter.utility import dtypes

        arch_fp8 = dtypes.fp8
        for key in dispatch_keys(gfx, cu_num):
            label = key.label()
            if label in out:
                continue
            # Errors are dispatch results too: record them, do not raise.
            try:
                metadata = fm.get_2stage_cfgs(*key.args, **key.call_kwargs())
                out[label] = record.normalize(metadata)
            except Exception as exc:  # noqa: BLE001
                out[label] = record.error_record(exc)
        for kw in keys.activation_dtype_keys(arch_fp8):
            label = "act_dtype|" + "|".join(
                f"{k}={keys._short(v)}" for k, v in kw.items()
            )
            try:
                out[label] = record.normalize(fm.resolve_activation_dtype(**kw))
            except Exception as exc:  # noqa: BLE001
                out[label] = record.error_record(exc)
    for label, rec in list(out.items()):
        if label.split("|")[1] not in ("base", "") and not label.startswith(
            "act_dtype"
        ):
            base = out.get(_variant_label(label, "base"))
            if base is not None and base == rec:
                out[label] = SAME_AS_BASE
    return out


def golden_path(arch, directory=GOLDEN_DIR):
    return pathlib.Path(directory) / f"{arch}.jsonl.gz"


def write(arch, records, directory=GOLDEN_DIR):
    path = golden_path(arch, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    text.write(json.dumps({"format": FORMAT_VERSION, "arch": arch}) + "\n")
    for label in sorted(records):
        text.write(
            json.dumps({"key": label, "rec": records[label]}, sort_keys=True) + "\n"
        )
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(text.getvalue().encode())
    return path


def read(arch, directory=GOLDEN_DIR):
    path = golden_path(arch, directory)
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        if header.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format {header.get('format')} != {FORMAT_VERSION}"
            )
        return {(line := json.loads(raw))["key"]: line["rec"] for raw in f}


def diff(expected, actual, limit=40):
    """Human-readable differences; empty list when identical."""
    lines = []
    for label in sorted(set(expected) | set(actual)):
        if label not in actual:
            lines.append(f"- missing: {label}")
        elif label not in expected:
            lines.append(
                f"+ new:     {label}  {json.dumps(actual[label], sort_keys=True)}"
            )
        elif expected[label] != actual[label]:
            lines.append(f"~ changed: {label}")
            lines.append(f"    golden: {json.dumps(expected[label], sort_keys=True)}")
            lines.append(f"    now:    {json.dumps(actual[label], sort_keys=True)}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines"]
    return lines


def refresh_inputs():
    """Freeze the current merged runtime tuned tables as the snapshot inputs, with
    their conversion."""
    from aiter.jit.core import AITER_CONFIGS
    from moe_dispatch import convert_tables

    sources = {
        "fmoe": AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
        "fhmoe": AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE,
    }
    for key, name in FROZEN_TABLES.items():
        write_input(name, pathlib.Path(sources[key]).read_bytes())
        print(f"{sources[key]} -> {INPUTS_DIR / name}.gz")
    convert_tables._frozen_problems(update=True)
    print(f"converted -> {', '.join(FROZEN_CONVERTED.values())}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare with golden files")
    mode.add_argument("--update", action="store_true", help="rewrite golden files")
    mode.add_argument(
        "--refresh-inputs", action="store_true", help="re-freeze tuned tables"
    )
    parser.add_argument("--out", help="write snapshots to this directory instead")
    args = parser.parse_args(argv)

    if args.refresh_inputs:
        _setup(frozen=False)
        refresh_inputs()
        return 0

    _setup()
    from moe_dispatch import arch_sim

    archs = args.arch or list(arch_sim.ARCHS)
    failed = False
    for arch in archs:
        records = generate(arch)
        if args.check:
            lines = diff(read(arch), records)
            status = "identical" if not lines else "DIFFERENT"
            print(f"{arch}: {len(records)} keys, {status}")
            for line in lines:
                print("  " + line)
            failed |= bool(lines)
        else:
            directory = args.out or GOLDEN_DIR
            path = write(arch, records, directory)
            print(f"{arch}: {len(records)} keys -> {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
