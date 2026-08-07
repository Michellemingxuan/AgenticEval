"""Lifecycle management for evaluator-launched target processes."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


class ManagedProcess:
    def __init__(self, name: str, config: dict[str, Any], output_dir: Path) -> None:
        self.name = name
        self.config = config
        self.output_dir = output_dir
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def start(self, healthcheck: Callable[[], None]) -> None:
        command = [str(item) for item in self.config["command"]]
        cwd = str(self.config.get("cwd") or ".")
        log_path = Path(
            self.config.get("stdout")
            or self.output_dir / "logs" / f"{self.name}.server.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (self.config.get("env") or {}).items()})
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + float(self.config.get("startup_timeout_s", 180))
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited during startup with code "
                    f"{self.process.returncode}; see {log_path}"
                )
            try:
                healthcheck()
                return
            except Exception as exc:  # target is still booting
                last_error = exc
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

