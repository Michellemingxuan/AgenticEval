"""Binding each concurrent worker to its own server instance.

Concurrency here is NOT a matter of firing more requests at one server. The
system under test keeps its data gateway and catalog as process-global objects
and re-scopes them to a case at the start of every turn, so two cases running
at once on one process interleave on those globals and a turn can execute
against the other case's tables. `server.py` says so itself:

    turns on ONE case are serialized by `sess.turn_lock`, but two different
    cases' turns can still interleave on these shared globals. That residual
    cross-case concurrency race is a separate, pre-existing limitation

The same applies within a case: `/rewind` clears the whole session, and the
Q→A cache is per session, so two repeats sharing a process would wipe each
other's history and serve each other's cached answers.

So a worker gets its OWN process, on its own port, its own log, and its own
trace DB. Anything shared between workers is a channel for them to corrupt each
other: `/rewind` deletes trace rows by CASE ID across every process holding the
file, so a shared DB means one worker's session-open wipes another's in-flight
evidence.

N workers means N server instances per system — real memory and real model
clients, which is why the default is 1 and the ceiling is small.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from agentic_eval.process import expand

#: The port in a `http://host:PORT` base URL. Matched on the URL rather than
#: taken from `env.PORT` alone because the adapter talks to the URL — if the
#: two disagree the worker would start a server nobody addresses.
_PORT_IN_URL = re.compile(r"^(?P<head>\w+://[^/:]+:)(?P<port>\d{2,5})(?P<tail>.*)$")


def _worker_path(path: str, worker: int) -> str:
    """`traces/current.db` -> `traces/current.w1.db`, keeping the suffix."""
    head, dot, tail = str(path).rpartition(".")
    return f"{head}.w{worker}.{tail}" if dot else f"{path}.w{worker}"


def bind_to_worker(
    name: str, target: dict[str, Any], worker: int,
) -> dict[str, Any]:
    """A copy of one system's config addressing worker `worker`'s instance.

    Worker 0 is the config as written, so a serial run is byte-for-byte what it
    always was. Each later worker shifts the port by its index, and shifts the
    server's own `PORT` to match — a base URL moved without the process moving
    would health-check against another worker's server and quietly measure it.
    """
    if worker == 0:
        return target
    out = copy.deepcopy(target)
    config = out.setdefault("config", {})
    base_url = str(config.get("base_url") or "")
    match = _PORT_IN_URL.match(base_url)
    if not match:
        raise ValueError(
            f"systems.{name}.config.base_url ({base_url!r}) has no port, so "
            "concurrent workers cannot be given separate servers; set "
            "experiment.workers: 1 or give the URL an explicit port"
        )
    port = int(match.group("port")) + worker
    config["base_url"] = f"{match.group('head')}{port}{match.group('tail')}"
    process = out.get("process")
    if process:
        env = dict(process.get("env") or {})
        if not env.get("PORT"):
            raise ValueError(
                f"systems.{name}.process.env.PORT is not set, so worker "
                f"{worker} cannot be started on its own port; set it, or use "
                "experiment.workers: 1"
            )
        if int(expand(env["PORT"])) != int(match.group("port")):
            # Silently shifting a mismatched pair would start a server on one
            # port and talk to another — the run would fail at healthcheck with
            # nothing pointing at the cause.
            raise ValueError(
                f"systems.{name}: process.env.PORT ({env['PORT']}) and "
                f"config.base_url port ({match.group('port')}) disagree; they "
                "must match before workers can be offset from them"
            )
        env["PORT"] = str(port)
        process["env"] = env
        # The trace DB must move with the worker. `/rewind` deletes trace rows
        # BY CASE ID, across every process sharing the file — deliberately, so
        # "clear means clear" for a reviewer. Two workers on one DB therefore
        # wipe each other: worker 1 opening a session on a case deletes the
        # rows worker 0 is still writing for it, and the evidence ledger for
        # that turn comes back empty. Measured on a real 2-worker run: 27 of 32
        # turns lost their traces.
        if env.get("NODE_TRACE_DB"):
            env["NODE_TRACE_DB"] = _worker_path(env["NODE_TRACE_DB"], worker)
        if config.get("trace_db"):
            # The evaluator READS this one; it has to be the same file the
            # server writes, or the turn reads another worker's trace.
            config["trace_db"] = _worker_path(str(config["trace_db"]), worker)
            env["NODE_TRACE_DB"] = config["trace_db"]
        process["env"] = env
        if process.get("stdout"):
            # One log per worker, or N servers append to one file and no line
            # says which instance wrote it.
            stdout = str(process["stdout"])
            head, dot, tail = stdout.rpartition(".")
            process["stdout"] = (
                f"{head}.w{worker}.{tail}" if dot else f"{stdout}.w{worker}"
            )
        out["process"] = process
    return out


def assert_workers_are_isolated(pool: list[dict[str, dict[str, Any]]]) -> None:
    """No two (worker, system) pairs may share a port or a trace DB.

    Ports: two systems configured close together (49102 and 49103) collide as
    soon as there are two workers, and it shows up as one server failing to
    bind — or worse, as both halves of the comparison measuring one checkout.

    Trace DBs: `/rewind` deletes rows by case id across every process holding
    the file, so sharing one means workers wipe each other's evidence. That is
    not a crash — the turn simply records no tool calls, and the consistency
    metric reports the system as inconsistent.
    """
    for field, label, hint in (
        ("base_url", "port", "Space the systems' ports at least `workers` apart"),
        ("trace_db", "trace DB",
         "Give each system its own trace_db; workers are offset from it"),
    ):
        seen: dict[str, str] = {}
        for worker, systems in enumerate(pool):
            for name, target in systems.items():
                value = (target.get("config") or {}).get(field)
                if not value:
                    continue
                owner = f"worker {worker} / {name}"
                if str(value) in seen:
                    raise ValueError(
                        f"{label} collision: {owner} and {seen[str(value)]} "
                        f"both use {value}. {hint}"
                    )
                seen[str(value)] = owner
