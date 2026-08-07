"""Pinned JSON-mode LLM judge client, independent of either target system."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol


class JudgeClient(Protocol):
    calls: list[dict[str, Any]]

    def complete_json(
        self, *, task: str, system_prompt: str, payload: dict[str, Any],
    ) -> dict[str, Any]: ...



def build_client(config: dict[str, Any], backend: str, timeout_s: float) -> Any:
    """A client exposing `.chat.completions.create`, whichever transport it is.

    Both backends present that one method, so the judge is written against it
    and knows nothing else about either.

    `safechain` is the private environment's gateway. It is constructed HERE
    rather than borrowed from the system under test: an evaluator that imports
    its subject inherits that subject's configuration and its bugs, and a judge
    is not independent if a change to the thing being judged can change the
    judging. The package is imported lazily, so the dev environment — which
    does not have it — fails only if it is actually asked for, and says why.
    """
    if backend in {"openai", "openai_compatible"}:
        from openai import OpenAI

        api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
        api_key = config.get("api_key") or os.environ.get(api_key_env)
        kwargs: dict[str, Any] = {
            "max_retries": int(config.get("max_retries", 8)),
            "timeout": timeout_s,
        }
        if api_key:
            kwargs["api_key"] = str(api_key)
        if config.get("base_url"):
            kwargs["base_url"] = str(config["base_url"])
        return OpenAI(**kwargs)

    if backend == "safechain":
        try:
            from safechain.lc_factory import get_model  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - private environment only
            raise RuntimeError(
                "backend: safechain needs the `safechain` package, which only "
                "exists in the private environment. Use backend: openai here, "
                "or set LLM_BACKEND=openai."
            ) from exc
        return SafeChainClient(get_model(str(config.get("model") or "gpt-4.1")))

    raise ValueError(
        f"unknown judge backend {backend!r}; use 'openai', 'openai_compatible' "
        "or 'safechain'"
    )


class SafeChainClient:
    """Adapts a SafeChain LCEL model to the one call this judge makes.

    Only `chat.completions.create` is supported, because that is all the judge
    uses — anything else should fail loudly rather than appear to work.

    JSON mode is real here, not hoped for: SafeChain binds `response_format`
    onto the LCEL model and forwards it unchanged to the endpoint, the same
    way the OpenAI SDK does. Dropping it — as an earlier version of this
    adapter did by swallowing kwargs — would have left the judge relying on
    "Return JSON only" in the prompt and raising on the first prose reply.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self.chat = self

    @property
    def completions(self) -> "SafeChainClient":
        return self

    def create(
        self, *, model: str, messages: list[dict[str, Any]],
        response_format: Any = None, **kwargs: Any,
    ) -> Any:
        bound = (
            self._model.bind(response_format=response_format)
            if response_format is not None else self._model
        )
        reply = bound.invoke([
            (str(m.get("role")), str(m.get("content"))) for m in messages
        ])
        usage = getattr(reply, "usage_metadata", None) or {}
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=str(getattr(reply, "content", reply))),
            )],
            usage=SimpleNamespace(
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
        )


@dataclass
class OpenAIJudgeClient:
    """OpenAI chat-completions JSON client with the same retry posture as AgenticSys."""

    config: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = str(self.config.get("model") or "gpt-4.1")
        self.temperature = float(self.config.get("temperature", 0))
        self.timeout_s = float(self.config.get("timeout_s", 180))
        self.backend = str(
            self.config.get("backend") or os.environ.get("LLM_BACKEND", "openai")
        )
        self._client = build_client(self.config, self.backend, self.timeout_s)

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
