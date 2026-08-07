"""Black-box system adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic_eval.models import AdapterResult, RunRequest


class SystemAdapter(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def healthcheck(self) -> None:
        """Raise when the target is not ready."""

    @abstractmethod
    def reset(self) -> None:
        """Clear target conversation state before a cold run/sequence."""

    @abstractmethod
    def run(self, request: RunRequest, timeout_s: float) -> AdapterResult:
        """Execute one question and return the normalized result."""

