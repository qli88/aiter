# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MoE cases of tests/moe_dispatch/gpu_differential.py on two checkouts of aiter.

The reference is another checkout (a worktree of the branch point of `main`), the
candidate this one; both run with AITER_MOEV2 unset unless `--cand-env` sets it. Each
checkout runs the cases in a worker process whose `import aiter` is that checkout's
(`PYTHONPATH`). Per case the worker builds the call (with its own aiter, from the same
seed), clears the dispatch caches, calls `fused_moe` once (the first call: dispatch
log lines, JIT builds), then twice more (the reference six times), the allocator
dirtied before each (`differential.dirty_allocator`), recording the kernel launches
(`kernel_bench_callable`) and the output. The two checkouts must have:

- bitwise-equal inputs, before and after the calls;
- the same error, if either raises (nothing else is compared then);
- the same log lines on the first call (checkout paths, JIT build, table merge and
  table load lines left out);
- the same launches: stages, function names, non-tensor arguments, tensor dtypes and
  shapes (a case where none are recorded says so);
- the same output: where the reference's first two runs are bitwise equal, both
  candidate runs bitwise equal to them; else each candidate run within
  `differential._check_output` and `differential.output_outside` of the reference's
  first two runs (GEMM2 atomics); the dtype and shape of both candidate runs.
  `results.jsonl` (`output`) and the last lines say which. A candidate run that fails
  this is compared with all six reference runs (`differential.output_outside`, not
  bitwise): it fails where it is farther from the reference's first run than those
  runs let it be, and is undecided (LEGACY) otherwise, unless all six repeat bitwise;
- no v2 outcome in the candidate (`aiter.moev2.paths.outcomes`) with AITER_MOEV2
  off or unset.

With `--cand-env AITER_MOEV2=<path>` the candidate runs v2: the reference's log lines
and stage 1 kernel names are compared as v2 names them (`differential._as_v2_logs`,
`legacy_stage1_as_v2`), and v2 must serve every call (`outcomes` all `served`). A call
v2 declines as `known_declines.py` lists it (`differential.known_decline`) is compared
as legacy against legacy and reported as `KNOWN` when nothing else differs.

    python tests/moe_dispatch/cross_checkout.py --ref ../aiter_main --path a8w8_blockscale
    python tests/moe_dispatch/cross_checkout.py --ref ../aiter_main --path all \\
        --model-table all --devices 6,7 --log-dir /tmp/xc
    python tests/moe_dispatch/cross_checkout.py --ref ../aiter_main --path a4w4 \\
        --case '[64, 7168, 256, 384, 8, "Silu", false, "bf16", false, "fp4", {"q_a": "fp4"}]'

`--devices`: one GPU (the two workers take turns) or two (one each, same GPU model).
A case that runs out of GPU memory, crashes a worker, hits a harness error or exceeds
`--timeout` is reported and fails the run (the worker is restarted). Exit status: 1
when a case differs, else 2 when one is undecided, else 0.
"""

import argparse
import collections
import contextlib
import json
import os
import pathlib
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
_NAME = pathlib.Path(__file__).name
_REPO = _HERE.parents[1]
# Single-GPU paths with `gpu_differential` cases.
PATHS = (
    "a8w8_pt",
    "a16w16",
    "a8w8_ptok",
    "a8w8_blockscale",
    "a4w4",
    "a16w4",
    "a16wi4",
    "a8w4",
    "mxfp8",
    "fhmoe",
)
_V2_OFF = ("", "off", "0", "false", "none")
# Log lines that depend on what ran before rather than on dispatch: aiter.jit.core's
# builds and imports, and legacy's one-time merge of its tuned tables (v2 reads its
# own; the file order is the directory listing's).
_SKIPPED_LINE = re.compile(
    r"^(import \[|start build \[|finish build \[|waiting for baton|"
    r"failed jit build \[|broke stale lock|merge tuned file |"
    r"\[\w+\] prebuilt \.so targets |JIT build \[[^\]]*\] succeeded, but publishing )"
)
# The process and colour prefixes aiter.jit puts before those.
_LOG_PREFIX = re.compile(r"^(\x1b\[[0-9;]*m|\[pid=\d+[^\]]*\] ?)+")
# Logged where a table is first read: by legacy or v2, and again by a restarted worker.
_TABLE_LOAD = re.compile(r"^\[fused_moe\] duplicate tuned rows ")
# Both checkouts' roots, which a worker writes as <checkout> in errors and log lines.
_ROOTS_VARIABLE = "_MOEV2_XC_ROOTS"
# Reference runs beyond the two the candidate's are compared with: whether the reference
# itself leaves the bitwise value or the tolerance its first two runs give.
_MORE_REFERENCE_RUNS = 4


def _without_roots(text, roots):
    # Longest first: one checkout can be nested in the other (../aiter/.worktrees/main).
    for root in sorted(roots, key=len, reverse=True):
        text = re.sub(re.escape(root) + r"(?![\w.-])", "<checkout>", text)
    return text


# The worker: one per checkout.


def _describe(value):
    import torch

    if isinstance(value, torch.Tensor):
        return f"{str(value.dtype).removeprefix('torch.')}{list(value.shape)}"
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value) and hasattr(value, "__qualname__"):
        return value.__qualname__
    return re.sub(r" at 0x[0-9a-f]+", "", repr(value))


def _launch(stage, call):
    func = getattr(call, "func", call)
    return {
        "stage": stage,
        "func": getattr(func, "__qualname__", type(func).__name__),
        "args": _describe(list(getattr(call, "args", ()))),
        "kw": _describe(dict(getattr(call, "keywords", {}) or {})),
    }


def _digest(t):
    import hashlib

    import torch

    data = t.detach().contiguous()
    if data.element_size() != 1:
        data = data.view(torch.uint8)
    return hashlib.blake2b(
        data.view(torch.uint8).cpu().numpy().tobytes(), digest_size=16
    ).hexdigest()


def _dispatch_line(msg):
    return not _SKIPPED_LINE.match(_LOG_PREFIX.sub("", msg))


def _clear_dispatch_caches():
    """Legacy's per-key dispatch caches (and v2's in a checkout that has it), so each
    case resolves and logs from scratch, whatever ran before in the worker."""
    import aiter.fhmoe as legacy_fhmoe
    import aiter.fused_moe as fm

    fm.get_2stage_cfgs.cache_clear()
    legacy_fhmoe._supports_dsv4_i384_fhmoe_config.cache_clear()
    legacy_fhmoe._supports_mxfp8_fhmoe_config.cache_clear()
    try:
        import aiter.moev2  # noqa: F401
    except ImportError:
        return
    import gpu_differential as gd

    gd._clear_dispatch_caches()


def _worker_case(request, shm, label):
    import gpu_differential as gd
    import torch
    from differential import captured_logs, dirty_allocator, known_decline

    import aiter.fused_moe as fm

    try:
        from aiter.moev2 import paths
    except ImportError:
        paths = None

    path_name, case = request["path"], tuple(request["case"])
    env = case[10].get("env", {}) if len(case) > 10 else {}
    own_root = str(pathlib.Path(fm.__file__).resolve().parents[1])
    # A checkout running another's compiled modules reports that one's source paths.
    roots = {
        own_root,
        *filter(None, os.environ.get(_ROOTS_VARIABLE, "").split(os.pathsep)),
    }
    reply = {"id": request["id"]}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    def digests(call):
        return {k: _digest(v) for k, v in call.items() if isinstance(v, torch.Tensor)}

    try:
        call = gd._make_call(path_name, case)
        reply["inputs"] = digests(call)
        _clear_dispatch_caches()
        if paths is not None:
            paths.reset_outcomes()
        with captured_logs() as logs:
            error = gd._raised(lambda: fm.fused_moe(**call))
        reply["error"] = None if error is None else _without_roots(error, roots)
        reply["logs"] = [
            (level, _without_roots(msg, roots))
            for level, msg in logs
            if _dispatch_line(msg)
        ]
        if reply["error"] is None:
            reply["runs"] = []
            for i in range(2 + (_MORE_REFERENCE_RUNS if label == "reference" else 0)):
                dirty_allocator()
                bench = []
                fm.kernel_bench_callable = bench
                try:
                    out = fm.fused_moe(**call)
                finally:
                    fm.kernel_bench_callable = None
                torch.cuda.synchronize()
                file = os.path.join(shm, f"{label}_{request['id']}_{i}.pt")
                torch.save(out.detach().cpu(), file)
                reply["runs"].append(
                    {"output": file, "launches": [_launch(s, c) for s, c in bench]}
                )
            reply["inputs_after"] = digests(call)
        if paths is not None:
            reply["outcomes"] = [[*k, n] for k, n in paths.outcomes().items()]
            declined = sorted({k[1] for k in paths.outcomes() if k[2] == "declined"})
            reply["declines"] = []
            for name in declined:
                reason, known = known_decline(name, call) or (None, None)
                reply["declines"].append([name, reason, known and known.name])
    except torch.OutOfMemoryError:
        reply = {"id": request["id"], "oom": True}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        torch.cuda.empty_cache()
    return reply


def _fills(label):
    """`differential._FILLS` for a worker: the calls whose fill bytes the other
    checkout's worker never writes (`differential.fill_byte`)."""
    import itertools

    return itertools.count(int(label != "reference"), 2)


def _worker(reply_fd, shm, label):
    sys.path[:0] = [str(_HERE), str(_HERE.parent)]
    import differential
    import torch

    import aiter

    differential._FILLS = _fills(label)
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    replies = os.fdopen(reply_fd, "w")

    def send(message):
        replies.write(json.dumps(message) + "\n")
        replies.flush()

    send(
        {
            "aiter": str(pathlib.Path(aiter.__file__).resolve()),
            "gfx": get_gfx_runtime(),
            "cu_num": get_cu_num(),
            "device": torch.cuda.get_device_name(),
            "moev2": os.environ.get("AITER_MOEV2"),
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        try:
            reply = _worker_case(request, shm, label)
        except Exception:  # noqa: BLE001
            import traceback

            reply = {"id": request["id"], "harness": traceback.format_exc()}
        send(reply)
    return 0


# The driver.


class _Worker:
    def __init__(self, label, root, device, env, shm, log_dir, roots):
        self.label, self.root, self.device = label, root, device
        self.env, self.shm, self.log_dir = env, shm, log_dir
        self.roots = roots
        self.process = None

    def start(self):
        read_fd, write_fd = os.pipe()
        env = {k: v for k, v in os.environ.items() if k != "AITER_MOEV2"}
        env.update(self.env)
        env["HIP_VISIBLE_DEVICES"] = self.device
        env[_ROOTS_VARIABLE] = os.pathsep.join(str(r) for r in self.roots)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.root), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
        log = pathlib.Path(self.log_dir) / f"{self.label}.log"
        with open(log, "a") as out:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(_HERE / _NAME),
                    "--worker",
                    str(write_fd),
                    self.shm,
                    self.label,
                ],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=subprocess.STDOUT,
                pass_fds=(write_fd,),
                env=env,
                cwd=self.root,
                text=True,
            )
        os.close(write_fd)
        self.replies = os.fdopen(read_fd)
        self.info = self.read(1800)
        if self.info is None:
            raise SystemExit(f"{self.label} worker did not start: see {log}")
        return self.info

    def send(self, request):
        """False if the worker is gone."""
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def read(self, timeout, request_id=None):
        """The next reply, or None if the worker died, `timeout` s passed or the reply
        is not to `request_id`."""
        ready, _, _ = select.select([self.replies], [], [], timeout)
        line = self.replies.readline() if ready else ""
        reply = json.loads(line) if line else None
        if (
            reply is not None
            and request_id is not None
            and reply.get("id") != request_id
        ):
            return None
        return reply

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            with contextlib.suppress(OSError):
                self.process.stdin.close()
            with contextlib.suppress(OSError):
                self.replies.close()
        self.process = None


def _v2_selected(cand_env):
    return cand_env.get("AITER_MOEV2", "").strip().lower() not in _V2_OFF


def _as_v2_launches(launches):
    """The reference's launches with its stage 1 kernel name as v2 runs it."""
    from differential import legacy_stage1_as_v2

    kernel2 = next(
        (x["kw"].get("kernelName") for x in launches if x["stage"] == "stage2"), None
    )
    out = []
    for x in launches:
        name = x["kw"].get("kernelName")
        if (
            x["stage"] == "stage1"
            and isinstance(name, str)
            and isinstance(kernel2, str)
        ):
            x = {
                **x,
                "kw": {**x["kw"], "kernelName": legacy_stage1_as_v2(name, kernel2)},
            }
        out.append(x)
    return out


def _known(cand):
    """The `known_declines.py` entries of the candidate's declines when every call it
    counted was declined and each decline is known, else None."""
    outcomes, declines = cand.get("outcomes", []), cand.get("declines", [])
    if not declines or any(o[2] != "declined" for o in outcomes):
        return None
    if not all(known for _, _, known in declines):
        return None
    return [known for _, _, known in declines]


def compare(ref, cand, v2, info=None):
    """Problems (strings) between a case's reference and candidate replies. `info`
    gets `output`: "bitwise" or "tolerance" (the reference did not repeat itself),
    where the outputs were compared; `known`: the entries of a v2 candidate's known
    declines (legacy ran the call on both sides, so it is compared as legacy); and,
    when there is no problem, `undecided`: why the output comparison does not decide (a
    candidate run fails the check of the reference's first two runs, but is no farther
    from the first than the reference's six runs vary)."""
    import torch
    from differential import _as_v2_logs, _check_output, output_outside

    for label, reply in (("reference", ref), ("candidate", cand)):
        if reply is None:
            return [f"{label}: worker crashed or timed out"]
        if reply.get("oom"):
            return [f"{label}: out of GPU memory (not compared)"]
        if "harness" in reply:
            return [f"{label}: harness error\n{reply['harness']}"]
    problems = []
    for key, when in (("inputs", "before"), ("inputs_after", "after the runs")):
        if key not in ref or key not in cand:
            continue
        a, b = ref[key], cand[key]
        if a != b:
            differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            problems.append(f"inputs differ {when}: {differ}")
    if ref["error"] != cand["error"]:
        problems.append(
            f"errors: reference {ref['error']!r}, candidate {cand['error']!r}"
        )
    known = _known(cand) if v2 else None
    if known and info is not None:
        info["known"] = known
    served = v2 and known is None
    ref_logs = [tuple(x) for x in ref["logs"] if not _TABLE_LOAD.match(x[1])]
    cand_logs = [tuple(x) for x in cand["logs"] if not _TABLE_LOAD.match(x[1])]
    if served:
        ref_logs = _as_v2_logs(ref_logs)
    if ref_logs != cand_logs:
        problems.append(
            f"log lines differ\n      ref:  {ref_logs}\n      cand: {cand_logs}"
        )
    if ref["error"] is not None or cand["error"] is not None:
        return problems
    outcomes = cand.get("outcomes", [])
    if served:
        if not outcomes or any(o[2] != "served" for o in outcomes):
            reasons = [d for d in cand.get("declines", []) if d[1]]
            problems.append(
                f"v2 did not serve every call: {outcomes}"
                + (f"; declines {reasons}" if reasons else "")
            )
    elif outcomes and not v2:
        problems.append(f"v2 routed calls with AITER_MOEV2 off or unset: {outcomes}")
    for i in range(2):
        ref_launches = ref["runs"][i]["launches"]
        if served:
            ref_launches = _as_v2_launches(ref_launches)
        if ref_launches != cand["runs"][i]["launches"]:
            problems.append(
                f"run {i + 1} launches differ\n      ref:  {ref_launches}\n"
                f"      cand: {cand['runs'][i]['launches']}"
            )
            break
    files = [run["output"] for reply in (ref, cand) for run in reply["runs"]]
    if len(set(files)) != len(files):
        return problems + [f"harness: output files collide: {files}"]
    r1, r2, *more = (torch.load(r["output"], map_location="cpu") for r in ref["runs"])
    c1, c2 = (torch.load(r["output"], map_location="cpu") for r in cand["runs"][:2])
    undecided = None
    other = [c for c in (c1, c2) if (c.dtype, c.shape) != (r1.dtype, r1.shape)]
    if other:
        problems.append(
            f"output {r1.dtype}{tuple(r1.shape)} vs {other[0].dtype}"
            f"{tuple(other[0].shape)}"
        )
    elif _same(r1, r2):
        # The reference repeats itself bitwise: so must the candidate, at its value.
        if info is not None:
            info["output"] = "bitwise"
        clean = all(_same(r1, r) for r in more)
        for i, c in enumerate((c1, c2), 1):
            if _same(r1, c):
                continue
            if clean:
                problems.append(
                    f"candidate run {i} output differs from the reference, which "
                    f"repeats bitwise (max diff {_max_diff(r1, c)}; candidate runs "
                    f"{'equal' if _same(c1, c2) else 'differ'})"
                )
                continue
            far = output_outside(r1, [r2, *more], c)
            if far is not None:
                problems.append(f"candidate run {i} output: {far}")
            elif undecided is None:
                undecided = (
                    f"candidate run {i} differs from the reference's first, which its "
                    f"runs 3 to {len(more) + 2} do not repeat, and is no farther from "
                    f"it than its {len(more) + 2} runs vary"
                )
    else:
        if info is not None:
            info["output"] = "tolerance"
        for i, c in enumerate((c1, c2), 1):
            try:
                _check_output(
                    f"candidate run {i}", r1, r2, c, ("reference", "candidate")
                )
                failed = output_outside(r1, [r2], c)
                if failed is None:
                    continue
                failed = f"candidate run {i} against 2 reference runs: {failed}"
            except AssertionError as exc:
                failed = str(exc)
            far = output_outside(r1, [r2, *more], c)
            if far is not None:
                problems.append(f"{failed}; {far}")
            elif undecided is None:
                undecided = (
                    f"{failed}; no farther from the reference's first run than its "
                    f"{len(more) + 2} runs vary"
                )
    if undecided is not None and not problems and info is not None:
        info["undecided"] = undecided
    return problems


def _same(a, b):
    """Bitwise equal (NaN payloads included)."""
    import torch

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _max_diff(a, b):
    return (a.float() - b.float()).abs().nan_to_num(float("inf")).max().item()


def _notes(ref):
    """What a passing case did not compare."""
    runs = (ref or {}).get("runs") or []
    if runs and not runs[0]["launches"]:
        return ["no kernel launches recorded (outputs and log lines compared)"]
    return []


def _cleanup(*replies):
    for reply in replies:
        for run in (reply or {}).get("runs", []):
            with contextlib.suppress(FileNotFoundError):
                os.remove(run["output"])


def _cases(args):
    import gpu_differential as gd

    names = PATHS if args.path == "all" else tuple(args.path.split(","))
    unknown = sorted(set(names) - set(PATHS))
    if unknown:
        raise SystemExit(
            f"no single-GPU cases for {unknown}; known: {', '.join(PATHS)}"
        )
    if args.case:
        if len(names) != 1:
            raise SystemExit("--case needs one --path")
        return [(names[0], tuple(json.loads(args.case)), "--case")]
    todo = []
    for name in names:
        if args.model_table:
            if name == "fhmoe":
                continue  # FHMoE calls are built from a profile, not a table row
            todo += [(name, c, s) for c, s in gd.selected_model_cases(args, name)]
        else:
            todo += [
                (name, c, f"{name} case {i}") for i, c in enumerate(gd.cases(name))
            ]
    return todo


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return _worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    # The driver compares with this checkout's aiter (v2 names, table cases).
    sys.path[:0] = [str(_HERE), str(_REPO)]
    import gpu_differential as gd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="the reference checkout")
    parser.add_argument("--path", required=True, help="paths (comma-separated) or all")
    gd.add_model_arguments(parser)
    parser.add_argument("--case", help="run only this case (JSON, one --path)")
    parser.add_argument(
        "--cand-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set for the candidate only (e.g. AITER_MOEV2=a4w4)",
    )
    parser.add_argument("--devices", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    parser.add_argument("--timeout", type=float, default=1800, help="s per case")
    parser.add_argument(
        "--log-dir", help="worker logs and results.jsonl (default: tmp)"
    )
    args = parser.parse_args(argv)

    ref_root = pathlib.Path(args.ref).resolve()
    if not (ref_root / "aiter" / "__init__.py").is_file():
        raise SystemExit(f"{ref_root} is not an aiter checkout")
    if ref_root == _REPO:
        raise SystemExit("--ref is this checkout")
    cand_env = dict(e.split("=", 1) for e in args.cand_env)
    v2 = _v2_selected(cand_env)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(devices) not in (1, 2):
        raise SystemExit("--devices: one or two GPUs")
    if not args.arch and args.model_table:
        os.environ["HIP_VISIBLE_DEVICES"] = devices[0]
    todo = _cases(args)
    if args.list:
        for name, case, source in todo:
            print(f"{name} {source} {json.dumps(case)}")
        print(f"{len(todo)} cases")
        return 0
    log_dir = pathlib.Path(args.log_dir or tempfile.mkdtemp(prefix="moev2_xc_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    shm = tempfile.mkdtemp(
        prefix="moev2_xc_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None
    )

    roots = (ref_root, _REPO)
    workers = [
        _Worker("reference", ref_root, devices[0], {}, shm, log_dir, roots),
        _Worker("candidate", _REPO, devices[-1], cand_env, shm, log_dir, roots),
    ]
    try:
        infos = [w.start() for w in workers]
        for w, info in zip(workers, infos):
            print(f"{w.label}: {info}")
            if not info["aiter"].startswith(str(w.root) + os.sep):
                raise SystemExit(
                    f"{w.label} imports aiter from {info['aiter']}, not {w.root}"
                )
        if [(i["gfx"], i["cu_num"], i["device"]) for i in infos].count(
            (infos[0]["gfx"], infos[0]["cu_num"], infos[0]["device"])
        ) != 2:
            raise SystemExit(
                "the two devices differ: use one device or two of one model"
            )
        print(f"{len(todo)} cases; worker logs and results.jsonl in {log_dir}")
        with open(log_dir / "results.jsonl", "a") as results:
            counts, compared = _run(todo, workers, args, v2, results)
        print(
            f"outputs compared: {compared['bitwise']} bitwise, "
            f"{compared['tolerance']} within tolerance (the reference not repeatable)"
        )
        print(f"{counts['known']} known declines (known_declines.py)")
        print(f"{counts['undecided']} cases undecided (the reference not repeatable)")
        print(f"{counts['failed']} of {len(todo)} cases differ")
        return 1 if counts["failed"] else 2 if counts["undecided"] else 0
    finally:
        for w in workers:
            w.stop()
        shutil.rmtree(shm, ignore_errors=True)


def _run(todo, workers, args, v2, results):
    """Runs every case on both workers; returns how many cases differ, are undecided and
    are known declines (`counts`: `failed`, `undecided`, `known`), and how many outputs
    were compared each way (`compare`)."""
    counts = collections.Counter()
    compared = collections.Counter()
    together = workers[0].device != workers[1].device
    for n, (name, case, source) in enumerate(todo):
        request = {"id": n, "path": name, "case": case}
        start = time.monotonic()
        replies = []
        sent = []
        for w in workers:
            sent.append(w.send(request))
            if not together:
                replies.append(w.read(args.timeout, n) if sent[-1] else None)
        if together:
            replies = [
                w.read(args.timeout, n) if ok else None for w, ok in zip(workers, sent)
            ]
        for w, reply in zip(workers, replies):
            if reply is None or "harness" in reply:
                w.stop()
                w.start()
        info = {}
        problems = compare(*replies, v2, info)
        compared[info.get("output")] += 1
        notes = [] if problems else _notes(replies[0])
        if info.get("known"):
            notes.append(f"known decline: {', '.join(info['known'])}")
        if info.get("undecided") and not problems:
            notes.append(f"undecided: {info['undecided']}")
        _cleanup(*replies)
        status = "FAIL" if problems else "LEGACY" if info.get("undecided") else "ok"
        if status == "ok" and info.get("known"):
            status = "KNOWN"
        counts["failed"] += status == "FAIL"
        counts["undecided"] += status == "LEGACY"
        counts["known"] += status == "KNOWN"
        seconds = time.monotonic() - start
        print(
            f"{status:<6} {name} {source} {json.dumps(case)} ({seconds:.0f} s)",
            flush=True,
        )
        for p in problems + notes:
            print(f"    {p}")
        if problems or status == "LEGACY":
            print(
                f"    repro: python {(_HERE / _NAME).relative_to(_REPO)} "
                f"--ref {args.ref} --path {name} --case '{json.dumps(case)}'"
                + "".join(f" --cand-env {e}" for e in args.cand_env)
            )
        record = {"path": name, "source": source, "case": case, "problems": problems}
        record["notes"] = notes
        record["output"] = info.get("output")
        record["status"] = status
        results.write(json.dumps(record) + "\n")
        results.flush()
    return counts, compared


if __name__ == "__main__":
    sys.exit(main())
