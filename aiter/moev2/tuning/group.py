# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rank processes for tuning a collective: one group, every task on every rank.

`Group(handler, devices)` spawns a process per device, rank r of `len(devices)`: each
sees every device of the group (`HIP_VISIBLE_DEVICES`, as torchrun's ranks do) with
the env:// rendezvous variables set (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
`MASTER_ADDR`, `MASTER_PORT`) from its start (`pool.start_spawned`), then runs `setup()` (when
given: the distributed init), `handler(payload)` for each task, and `teardown()` when
the group closes, on every rank. A task's result is rank 0's, unless a rank returned a
failed result (`ok` false) or raised: then the first such rank's (lists elementwise,
for tasks with a result per plan). A rank dying, or a task taking longer than `timeout`
on any rank, fails the task and restarts the whole group (a collective cannot go on
without a rank; a restart that does not start is tried once more), as does a handler
raising `pool.Restart` on any rank. `map` runs the tasks one at a time and has
`Pool.map`'s form, so the search's `evaluate` takes either.
"""

import multiprocessing
import pickle
import socket
import time
from multiprocessing.connection import wait

from .pool import KILL_WAIT, Failed, Restart, start_spawned

CLOSE_WAIT = 30.0  # s for the ranks' teardown


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_env(devices, rank, port) -> dict:
    return {
        "HIP_VISIBLE_DEVICES": ",".join(devices),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(len(devices)),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    }


def _rank_main(pickled_handler, pickled_setup, pickled_teardown, connection):
    try:
        handler = pickle.loads(pickled_handler)
        setup = pickle.loads(pickled_setup)
        teardown = pickle.loads(pickled_teardown)
        if setup is not None:
            setup()
    except Exception as error:  # noqa: BLE001 - the group does not start
        connection.send(("setup", f"{type(error).__name__}: {str(error)[:500]}", True))
        return
    connection.send(("ready", None, False))
    while True:
        item = connection.recv()
        if item is None:
            if teardown is not None:
                teardown()
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


def _failed(result) -> bool:
    return getattr(result, "ok", True) is False


def _answered(replies) -> str:
    """The failures the ranks that replied gave (a rank's own reason for a task the
    others then could not finish)."""
    failed = [
        f"rank {rank}: {getattr(result, 'reason', result)}"
        for rank, result in sorted(replies.items())
        if _failed(result)
    ]
    return "" if not failed else "; " + "; ".join(failed)


def combine(results):
    """The task's result from each rank's (`results`, rank order)."""
    if (
        all(isinstance(r, list) for r in results)
        and len({len(r) for r in results}) == 1
    ):
        return [combine(list(column)) for column in zip(*results)]
    for rank, result in enumerate(results):
        if _failed(result):
            reason = getattr(result, "reason", None)
            if rank and isinstance(reason, str) and not reason.startswith("rank "):
                try:
                    return type(result)(f"rank {rank}: {reason}")
                except TypeError:
                    return result
            return result
    for rank, result in enumerate(results):
        if isinstance(result, list):
            return Failed(f"rank {rank} returned {len(result)} results, others differ")
    return results[0]


class Group:
    """`devices`: the group's `HIP_VISIBLE_DEVICES` ids, rank order."""

    def __init__(
        self,
        handler,
        devices,
        timeout=600.0,
        start_timeout=600.0,
        setup=None,
        teardown=None,
        restart_wait=10.0,
    ):
        self.pickled_handler = pickle.dumps(handler)
        self.pickled_setup = pickle.dumps(setup)
        self.pickled_teardown = pickle.dumps(teardown)
        self.devices = [str(d) for d in devices]
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.restart_wait = restart_wait
        self.processes = []
        self.connections = []
        self.starts = 0
        self._start()

    def _start(self):
        context = multiprocessing.get_context("spawn")
        port = free_port()
        self.processes, self.connections = [], []
        pickled = (self.pickled_handler, self.pickled_setup, self.pickled_teardown)
        try:
            for rank in range(len(self.devices)):
                parent, child = context.Pipe()
                self.connections.append(parent)
                try:
                    process = start_spawned(
                        _rank_main,
                        (*pickled, child),
                        _rank_env(self.devices, rank, port),
                    )
                finally:
                    child.close()
                self.processes.append(process)
        except BaseException:
            self._kill()
            raise
        self.starts += 1
        _, _, failure = self._collect("ready", self.start_timeout)
        if failure is not None:
            self._kill()
            raise RuntimeError(f"the rank group did not start: {failure}")

    def _kill(self):
        for process in self.processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + KILL_WAIT
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for connection in self.connections:
            connection.close()

    def restart(self):
        """Start the group again; a start that fails (the killed ranks' devices still
        being released) is tried once more after `restart_wait` s."""
        self._kill()
        try:
            self._start()
        except RuntimeError:
            time.sleep(self.restart_wait)
            self._start()

    def _collect(self, index, timeout):
        """`(each rank's reply to `index`, restart, None)`, or `(None, False, the
        reason a rank did not reply)` (it died, or the deadline passed)."""
        deadline = time.monotonic() + timeout
        replies, restart = {}, False
        while len(replies) < len(self.connections):
            waiting = {}
            for rank, (connection, process) in enumerate(
                zip(self.connections, self.processes)
            ):
                if rank not in replies:
                    waiting[connection] = rank
                    waiting[process.sentinel] = rank
            left = deadline - time.monotonic()
            if left <= 0:
                late = sorted(set(waiting.values()))
                return (
                    None,
                    False,
                    f"timed out after {timeout:g} s (ranks {late})"
                    + _answered(replies),
                )
            for ready in wait(list(waiting), left):
                rank = waiting[ready]
                if rank in replies:
                    continue
                connection = self.connections[rank]
                message = None
                try:
                    if connection.poll():
                        message = connection.recv()
                except (EOFError, OSError):
                    pass
                if message is None:
                    process = self.processes[rank]
                    process.join(1)
                    if process.is_alive():
                        continue
                    return (
                        None,
                        False,
                        f"rank {rank} died (exit code {process.exitcode})"
                        + _answered(replies),
                    )
                got, result, exiting = message
                if got == "setup":
                    return None, False, f"rank {rank}: {result}"
                if got != index:
                    return None, False, f"rank {rank} replied to {got!r}, not {index!r}"
                replies[rank] = result
                restart |= exiting
        return [replies[r] for r in range(len(self.connections))], restart, None

    def run(self, index, payload, timeout=None):
        """The combined result of `payload` on every rank, or `Failed` (`timeout`, s:
        this task's, else the group's)."""
        for rank, connection in enumerate(self.connections):
            try:
                connection.send((index, payload))
            except OSError:
                self.restart()
                return Failed(f"rank {rank} is gone")
        results, restart, failure = self._collect(index, timeout or self.timeout)
        if failure is not None:
            self.restart()
            return Failed(failure)
        if restart:
            self.restart()
        return combine(results)

    def map(self, payloads):
        """Yield `(index, result)` for each payload, in order."""
        for index, payload in enumerate(payloads):
            yield index, self.run(index, payload)

    def close(self):
        for connection in self.connections:
            try:
                connection.send(None)
            except OSError:
                pass
        deadline = time.monotonic() + CLOSE_WAIT
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
        self._kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
