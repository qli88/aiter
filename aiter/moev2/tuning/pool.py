# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Worker processes for tuning: one per device, one task at a time.

`Pool(handler, devices)` starts a spawned process per device (`HIP_VISIBLE_DEVICES` set
to it when the process starts) running `handler(payload)` for each task it is
sent. A task taking longer than `timeout`, or whose worker dies (a crash, a device
fault), comes back as `Failed` with the reason; the worker is then killed if needed and
restarted, and the remaining tasks go on. A handler raising `Restart(result)` returns
`result` and has its worker restarted (a device left unusable).
"""

import multiprocessing
import os
import pickle
import time
from dataclasses import dataclass
from multiprocessing.connection import wait

# s for a killed process to exit: one stuck in the driver (after a device fault) may
# never exit, and is then left to die with the parent (daemon).
KILL_WAIT = 10.0


@dataclass(frozen=True)
class Failed:
    reason: str

    @property
    def ok(self) -> bool:
        return False


class Restart(Exception):
    """Raised by a handler: return `result`, then restart the worker."""

    def __init__(self, result):
        super().__init__(result)
        self.result = result


def start_spawned(target, args, env):
    """Start a spawned process of `target(*args)` whose environment has `env` from its
    first instruction: a spawned child imports the parent's main module before
    `target` runs, and that import may initialize the device runtime, which reads
    `HIP_VISIBLE_DEVICES` once."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=args, daemon=True)
    saved = {name: os.environ.get(name) for name in env}
    try:
        os.environ.update(env)
        process.start()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return process


def _worker(pickled_handler, connection):
    handler = pickle.loads(pickled_handler)
    connection.send(("ready", None, False))
    while True:
        item = connection.recv()
        if item is None:
            return
        index, payload = item
        try:
            connection.send((index, handler(payload), False))
        except Restart as restart:
            connection.send((index, restart.result, True))
            return
        except Exception as error:  # noqa: BLE001 - reported as the task's failure
            reason = f"{type(error).__name__}: {str(error)[:500]}"
            connection.send((index, Failed(reason), False))


class _Slot:
    def __init__(self, pool, device):
        self.pool, self.device = pool, device
        self.task = self.deadline = None
        self._start()

    def _start(self, waiting_since=None):
        parent, child = multiprocessing.get_context("spawn").Pipe()
        env = {} if self.device is None else {"HIP_VISIBLE_DEVICES": str(self.device)}
        self.process = start_spawned(_worker, (self.pool.pickled_handler, child), env)
        child.close()
        self.connection = parent
        self.ready = False
        self.started = time.monotonic() if waiting_since is None else waiting_since

    def restart(self, grace=0.0):
        """Kill (after `grace` s) and respawn; a worker that never became ready keeps
        its start time, so workers dying at start still hit the start timeout."""
        self.process.join(grace)
        if self.process.is_alive():
            self.process.kill()
        self.process.join(KILL_WAIT)
        self.connection.close()
        self.task = self.deadline = None
        self._start(None if self.ready else self.started)

    def send(self, index, payload, timeout):
        self.task = index
        self.deadline = time.monotonic() + timeout
        self.connection.send((index, payload))

    def stop(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except OSError:
                pass
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(KILL_WAIT)
        self.connection.close()


class Pool:
    """`devices`: the `HIP_VISIBLE_DEVICES` value of each worker (None: inherit)."""

    def __init__(self, handler, devices=(None,), timeout=300.0, start_timeout=300.0):
        self.pickled_handler = pickle.dumps(handler)
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.slots = [_Slot(self, device) for device in devices]

    def map(self, payloads):
        """Yield `(index, result)` for each payload as it finishes: the handler's
        result, or `Failed`. Closed early, it restarts the workers still running a
        task, so their results cannot reach a later `map`."""
        try:
            yield from self._map(payloads)
        finally:
            for slot in self.slots:
                if slot.task is not None:
                    slot.restart()

    def _map(self, payloads):
        pending = list(enumerate(payloads))[::-1]
        busy = 0
        while pending or busy:
            for slot in self.slots:
                if slot.ready and slot.task is None and pending:
                    try:
                        slot.send(*pending[-1], self.timeout)
                    except OSError:
                        slot.restart()
                        continue
                    pending.pop()
                    busy += 1
            waiting = {slot.connection: slot for slot in self.slots}
            waiting.update({slot.process.sentinel: slot for slot in self.slots})
            deadlines = [s.deadline for s in self.slots if s.deadline is not None]
            now = time.monotonic()
            wait_for = min(deadlines) - now if deadlines else self.start_timeout
            for slot in {waiting[r] for r in wait(list(waiting), max(0.0, wait_for))}:
                message = None
                try:
                    if slot.connection.poll():
                        message = slot.connection.recv()
                except (EOFError, OSError):
                    pass
                if message is not None:
                    index, result, exiting = message
                    if index == "ready":
                        slot.ready = True
                        continue
                    slot.task = slot.deadline = None
                    if exiting:
                        slot.restart(grace=5.0)
                    busy -= 1
                    yield index, result
                    continue
                slot.process.join(1)
                if slot.process.is_alive():
                    continue
                task, code = slot.task, slot.process.exitcode
                slot.restart()
                if task is not None:
                    busy -= 1
                    yield task, Failed(f"worker died (exit code {code})")
            now = time.monotonic()
            for slot in self.slots:
                if slot.deadline is not None and now > slot.deadline:
                    task = slot.task
                    slot.restart()
                    busy -= 1
                    yield task, Failed(f"timed out after {self.timeout:g} s")
            now = time.monotonic()
            if all(
                not s.ready and now - s.started > self.start_timeout for s in self.slots
            ):
                raise RuntimeError(f"no worker started in {self.start_timeout:g} s")

    def close(self):
        for slot in self.slots:
            slot.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
