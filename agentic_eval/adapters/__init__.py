"""Adapter registry.

Adapters are loaded lazily so third-party adapters can be added without
changing the runner.
"""
from __future__ import annotations

import importlib
from typing import Any

from agentic_eval.adapters.base import SystemAdapter


_BUILT_INS = {
    "agenticsys_sse": "agentic_eval.adapters.agenticsys_sse:AgenticSysSSEAdapter",
}


def build_adapter(name: str, config: dict[str, Any]) -> SystemAdapter:
    spec = _BUILT_INS.get(name, name)
    if ":" not in spec:
        raise ValueError(
            f"unknown adapter {name!r}; use a built-in name or module:Class"
        )
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    adapter = cls(config)
    if not isinstance(adapter, SystemAdapter):
        raise TypeError(f"{spec} must implement SystemAdapter")
    return adapter

