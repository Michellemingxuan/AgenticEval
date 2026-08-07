"""Pinned JSON-mode LLM judge client, independent of either target system."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class JudgeClient(Protocol):
    calls: list[dict[str, Any]]

    def complete_json(
        self, *, task: str, system_prompt: str, payload: dict[str, Any],
    ) -> dict[str, Any]: ...

    def run_tools(
        self, *, task: str, system_prompt: str, payload: dict[str, Any],
        tools: list[dict[str, Any]], dispatch: Callable[[str, dict[str, Any]], Any],
        finish_tool: str, max_steps: int = 12, model: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass
class OpenAIJudgeClient:
    """OpenAI chat-completions JSON client with the same retry posture as AgenticSys."""

    config: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = str(self.config.get("model") or "gpt-4.1")
        self.temperature = float(self.config.get("temperature", 0))
        self.timeout_s = float(self.config.get("timeout_s", 180))
        backend = str(self.config.get("backend") or os.environ.get("LLM_BACKEND", "openai"))
        if backend not in {"openai", "openai_compatible"}:
            raise ValueError(
                "AgenticEval supports openai/openai_compatible judges without "
                "importing AgenticSys. Configure a compatible base_url for private gateways."
            )
        from openai import OpenAI

        api_key_env = str(self.config.get("api_key_env") or "OPENAI_API_KEY")
        api_key = self.config.get("api_key") or os.environ.get(api_key_env)
        kwargs: dict[str, Any] = {
            "max_retries": int(self.config.get("max_retries", 8)),
            "timeout": self.timeout_s,
        }
        if api_key:
            kwargs["api_key"] = str(api_key)
        if self.config.get("base_url"):
            kwargs["base_url"] = str(self.config["base_url"])
        self._client = OpenAI(**kwargs)

    def complete_json(
        self, *, task: str, system_prompt: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"judge task {task} returned non-object JSON")
        self._record(task, self.model, started, getattr(response, "usage", None))
        return parsed

    def _record(self, task: str, model: str, started: float, usage: Any) -> None:
        self.calls.append({
            "task": task,
            "model": model,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })

    def run_tools(
        self, *, task: str, system_prompt: str, payload: dict[str, Any],
        tools: list[dict[str, Any]], dispatch: Callable[[str, dict[str, Any]], Any],
        finish_tool: str, max_steps: int = 12, model: str | None = None,
    ) -> dict[str, Any]:
        """Let the model search the evidence itself, then submit its finding.

        The single-shot judge has to name a `json_path` blind, and most of what
        we classify as judge error is a bad pointer rather than bad reading.
        Here it looks first and cites afterwards, which is also the only way to
        address a number inside a prose tool result: no path exists for
        "…= $174,897.36 (over 1 non-null value(s) in 1 matching row(s))", but a
        reader has no trouble saying which figure is the sum.

        Ends when the model calls `finish_tool`. Exhausting `max_steps` returns
        `{}` rather than raising: an unfinished search must read as "unknown",
        never as a verdict against the answer.
        """
        used = model or str(self.config.get("adjudication_model") or self.model)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for step in range(max_steps):
            # Left to itself the verifier keeps browsing after it already has
            # its citation — observed reading nine more payloads having found
            # the number on step three, and returning nothing when the cap hit.
            # An unfinished search is indistinguishable from an unverifiable
            # claim, so spend the last step submitting rather than looking.
            forced = step >= max_steps - 1
            started = time.perf_counter()
            response = self._client.chat.completions.create(
                model=used, messages=messages, tools=tools,
                tool_choice=(
                    {"type": "function", "function": {"name": finish_tool}}
                    if forced else "required"
                ),
                temperature=self.temperature,
            )
            self._record(task, used, started, getattr(response, "usage", None))
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            if not calls:
                break
            messages.append({
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": call.id, "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in calls
                ],
            })
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if call.function.name == finish_tool:
                    return arguments if isinstance(arguments, dict) else {}
                try:
                    result = dispatch(call.function.name, arguments)
                except Exception as error:  # a tool error is information, not a crash
                    result = {"error": f"{type(error).__name__}: {error}"}
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
        return {}
