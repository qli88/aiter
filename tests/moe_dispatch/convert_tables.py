# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The converted shipped tuned tables (`aiter/configs/moev2/`) against the legacy ones.

Converts the legacy tables (`convert.shipped_sources`: the default table's files, read
as their merge by `convert.merged_tables`, and the FHMoE table) on every simulated arch
into v2 plan-file rows (`convert.convert_table`): one file per legacy file at the same
path under `aiter/configs/moev2/`, each arch's rows in table order, the archs in the
order the file first has them. `sources.csv` there records each legacy file's sha256.
A row of a gfx and CU count no simulated arch has is an error. `--check` fails when a
converted file or `sources.csv` differs from the committed one, a committed file is not
produced, or the rows given no plan-file row differ from `golden/convert_skipped.json`
(counts per legacy file and reason; the `flydsl_fallback` rows, which the read drops
and dispatch never sees, are not counted); `--update` rewrites them all.

    python tests/moe_dispatch/convert_tables.py --check [-v]
    python tests/moe_dispatch/convert_tables.py --update
"""

import argparse
import csv
import hashlib
import io
import json
import pathlib
import sys
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent
SKIPPED = _HERE / "golden" / "convert_skipped.json"


def _csv(columns, rows) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def convert_all():
    """`(files, skipped)`: `{path under CONVERTED: text}` of every converted file and
    `sources.csv`, and `{legacy file: [(key, reason)]}`."""
    from aiter.moev2.dispatch import convert, tuned, tuned_plans
    from moe_dispatch import arch_sim

    main, fhmoe = convert.shipped_sources()
    tables = {
        pathlib.Path(n): (t, None) for n, t in convert.merged_tables(main).items()
    }
    tables[fhmoe] = (
        tuned.read_table(str(fhmoe), tuned.index_columns(True), True),
        fhmoe,
    )
    archs = {v: k for k, v in arch_sim.ARCHS.items()}
    order = {}
    for source, ((primary, _), _) in tables.items():
        combos = dict.fromkeys((k[0], int(k[1])) for k in primary)
        unknown = [c for c in combos if c not in archs]
        if unknown:
            raise ValueError(f"{source}: rows of {unknown}, which no arch simulates")
        order[source] = list(combos)
    converted, skipped = {}, {s: [] for s in tables}
    for arch in arch_sim.ARCHS:
        with arch_sim.simulated_arch(arch) as (gfx, cu_num):
            tuned.clear_cache()
            for source, (table, dedicated) in tables.items():
                if (gfx, cu_num) not in order[source]:
                    continue
                fhmoe_table = None if dedicated is None else str(dedicated)
                rows, missed = convert.convert_table(table, gfx, cu_num, fhmoe_table)
                converted[(source, (gfx, cu_num))] = rows
                skipped[source] += missed
    columns = tuned_plans.COLUMNS + convert.INFO_COLUMNS
    files, sources = {}, []
    for source in tables:
        name = source.relative_to(convert.CONFIGS).as_posix()
        rows = [r for combo in order[source] for r in converted[(source, combo)]]
        files[name] = _csv(columns, rows)
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        sources.append({"source": name, "sha256": sha})
    files[convert.SOURCES] = _csv(("source", "sha256"), sources)
    named = {s.relative_to(convert.CONFIGS).as_posix(): v for s, v in skipped.items()}
    return files, named


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="list the rows")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import arch_sim

    arch_sim.prepare_process(None)
    from aiter.moev2.dispatch import convert

    if not convert.CONFIGS.is_relative_to(_HERE.parents[1]):
        print(f"aiter is imported from {convert.CONFIGS.parent}, not this checkout")
        return 1
    files, skipped = convert_all()
    counts = {
        s: dict(sorted(Counter(r for _, r in rows).items()))
        for s, rows in sorted(skipped.items())
        if rows
    }
    rows = sum(t.count("\n") - 1 for n, t in files.items() if n != convert.SOURCES)
    print(f"{len(files) - 1} converted files, {rows} rows")
    for source, by_reason in counts.items():
        for reason, n in by_reason.items():
            print(f"  no plan-file row: {n:5d} {source}: {reason}")
        if args.verbose:
            for key, reason in skipped[source]:
                print(f"    {key}: {reason}")
    if args.update:
        for name in convert.CONVERTED.rglob("*.csv"):
            name.unlink()
        for name, text in files.items():
            path = convert.CONVERTED / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        SKIPPED.write_text(json.dumps(counts, indent=1) + "\n")
        print(f"wrote {convert.CONVERTED} and {SKIPPED}")
        return 0
    committed = {
        p.relative_to(convert.CONVERTED).as_posix(): p.read_text()
        for p in convert.CONVERTED.rglob("*.csv")
    }
    problems = [f"not committed: {n}" for n in sorted(files.keys() - committed.keys())]
    problems += [f"not produced: {n}" for n in sorted(committed.keys() - files.keys())]
    problems += [
        f"differs: {n}"
        for n in sorted(files.keys() & committed.keys())
        if files[n] != committed[n]
    ]
    expected = json.loads(SKIPPED.read_text()) if SKIPPED.exists() else {}
    if counts != expected:
        problems.append(f"the rows given no plan-file row differ from {SKIPPED.name}")
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problems")
    if problems:
        print(
            "if the change is intended, run python tests/moe_dispatch/convert_tables.py "
            "--update and commit aiter/configs/moev2/ and "
            f"{SKIPPED.relative_to(_HERE.parents[1])}"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
