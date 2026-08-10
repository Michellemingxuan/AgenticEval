"""Reading a `.env` into the process, when one is asked for.

The judge needs credentials, and in the private environment it needs more than
that: SafeChain's own config layer reads `CONFIG_PATH`, and the transport is
chosen by `LLM_BACKEND`. All of it lives in the same file the system under test
uses, and `bin/compare` sources it — but a bare `agentic-eval evaluate-content`
sourced nothing, so re-judging a run by hand failed on configuration that was
sitting in a file three directories away.

Two rules keep this from becoming spooky action:

  * **The environment wins.** A variable already exported is left alone. The
    file fills gaps; it never silently replaces something a caller set on
    purpose, which is the opposite of `set -a; . file` and the safer default
    when one of those values decides where judging traffic goes.

  * **Loading is announced.** The path and the count are printed. A file that
    quietly changes the backend, the model, or the endpoint is exactly the kind
    of thing that should leave a line in the log.

A path asked for and missing is an error, not a shrug: `--env-file` names a
file the caller believes in, and continuing without it would fail later,
somewhere less obvious.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repo root — this file is `<root>/agentic_eval/envfile.py`.
_ROOT = Path(__file__).resolve().parents[1]


def _parse(text: str) -> dict[str, str]:
    """`KEY=value` lines, the subset a shell-sourced `.env` actually uses."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def resolve(explicit: str | None = None) -> Path | None:
    """Which file to load: the flag, then `AGENTIC_EVAL_ENV`, then `<root>/.env`.

    Only the last is optional. The first two were named by someone, so a
    missing file is their mistake surfaced now rather than a judge failure
    later with a message about something else.
    """
    for candidate, source in (
        (explicit, "--env-file"),
        (os.environ.get("AGENTIC_EVAL_ENV"), "AGENTIC_EVAL_ENV"),
    ):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{source}: no such file: {path}")
            return path
    default = _ROOT / ".env"
    return default if default.is_file() else None


def load(explicit: str | None = None) -> tuple[Path, int] | None:
    """Fill gaps in `os.environ` from the resolved file. Returns what it did."""
    path = resolve(explicit)
    if path is None:
        return None
    applied = 0
    for key, value in _parse(path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            applied += 1
    return path, applied
