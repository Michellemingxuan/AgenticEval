"""Lifecycle management for evaluator-launched target processes."""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable



#: `${VAR}` and `${VAR:-fallback}`, so a config can name an interpreter without
#: hardcoding one machine's path. Without this every config pinned
#: `/Users/<someone>/.pyenv/versions/.../bin/python`, which is correct on
#: exactly one laptop and a silent "command not found" everywhere else.
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}")


def expand(value: Any) -> str:
    """Resolve environment references in a config string.

    An unset variable with no fallback is an error rather than an empty
    string: a command that silently loses its interpreter fails later and
    further away, with a message about the wrong thing.
    """
    def sub(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        resolved = os.environ.get(name) or fallback
        if resolved is None:
            raise KeyError(
                f"{name} is not set and has no fallback; write "
                f"${{{name}:-<default>}} or export it"
            )
        return resolved
    return os.path.expanduser(_VAR.sub(sub, str(value)))


class ManagedProcess:
    def __init__(self, name: str, config: dict[str, Any], output_dir: Path) -> None:
        self.name = name
        self.config = config
        self.output_dir = output_dir
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def start(self, healthcheck: Callable[[], None]) -> None:
        command = [expand(item) for item in self.config["command"]]
        cwd = expand(self.config.get("cwd") or ".")
        log_path = Path(
            self.config.get("stdout")
            or self.output_dir / "logs" / f"{self.name}.server.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        env = os.environ.copy()
        env.update({
            str(k): expand(v) for k, v in (self.config.get("env") or {}).items()
        })
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        budget = float(self.config.get("startup_timeout_s", 180))
        started_at = time.monotonic()
        deadline = started_at + budget
        # Say what is being waited for. A server can legitimately take minutes
        # to boot — the private environment acquires a gateway token at import
        # — and the wait used to be entirely silent, so "still starting" and
        # "wedged" looked identical for up to `startup_timeout_s`.
        print(f"  starting {self.name} (up to {budget:.0f}s) — {log_path}")
        last_error: Exception | None = None
        announced = 0.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited during startup with code "
                    f"{self.process.returncode}; see {log_path}"
                )
            try:
                healthcheck()
                print(
                    f"  {self.name} answered after "
                    f"{time.monotonic() - started_at:.0f}s"
                )
                return
            except Exception as exc:  # target is still booting
                last_error = exc
                waited = time.monotonic() - started_at
                if waited - announced >= 15:
                    announced = waited
                    print(
                        f"  {self.name} not answering yet ({waited:.0f}s of "
                        f"{budget:.0f}s) — {type(exc).__name__}"
                    )
                time.sleep(1)
        raise RuntimeError(
            f"{self.name} did not become healthy; see {log_path}. Last error: "
            f"{last_error}"
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
        self.process = None
        self._log_handle = None

