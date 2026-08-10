"""Pinned JSON-mode LLM judge client, independent of either target system."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

# SafeChain's token helper bridges sync->async with `asyncio.run(...)` on its
# own path, which raises inside a running loop. Patching here means the judge
# can stay synchronous. Absent in dev, where it is not needed.
try:  # pragma: no cover - private environment only
    import nest_asyncio as _nest_asyncio

    _nest_asyncio.apply()
except ImportError:
    _nest_asyncio = None
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
        return SafeChainClient(
            str(config.get("model") or "gpt-4.1"), timeout_s=timeout_s,
            # Same knob as the OpenAI branch above. It used to be read only
            # there, so a config asking for eight retries got none here and a
            # single slow call ended a run that had already paid for every
            # answer before it.
            max_retries=int(config.get("max_retries", 8)),
        )

    raise ValueError(
        f"unknown judge backend {backend!r}; use 'openai', 'openai_compatible' "
        "or 'safechain'"
    )


class SafeChainClient:
    """Adapts SafeChain to the one call this judge makes.

    Only `chat.completions.create` is supported, because that is all the judge
    uses — anything else should fail loudly rather than appear to work.

    Everything underneath is ASYNC, mirroring AgenticSys_v2's client:

      * `safechain.core.model.amodel()` is an async factory — it acquires a
        token over the network — so the build is awaited and bounded. An
        unbounded build hangs the whole run with no timeout, since a per-call
        timeout only wraps the invoke.
      * the chain is `ValidChatPromptTemplate | model.bind(...)` and runs via
        `ainvoke`. The template stays in the chain for COMPLIANCE: its
        `format_prompt` override is template-time and a bare `ainvoke(messages)`
        would skip it.
      * `response_format` is bound, so JSON mode reaches the endpoint exactly
        as it does on the OpenAI SDK path.

    The judge itself is synchronous, so this owns the bridge: one `asyncio.run`
    per call, with `nest_asyncio` applied at import because SafeChain's own
    token helper bridges sync->async the same way and would otherwise raise
    "asyncio.run() cannot be called from a running event loop".
    """

    def __init__(
        self, model_name: str, *, timeout_s: float, max_retries: int = 8,
    ) -> None:
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._llm: Any = None            # built on first use, then cached
        self.chat = self

    @property
    def completions(self) -> "SafeChainClient":
        return self

    def create(
        self, *, model: str, messages: list[dict[str, Any]],
        response_format: Any = None, **kwargs: Any,
    ) -> Any:
        reply = self._with_retries(messages, response_format)
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

    def _with_retries(
        self, messages: list[dict[str, Any]], response_format: Any,
    ) -> Any:
        """Retry a timed-out call; surface anything else immediately.

        ONLY timeouts. A `FirewallRejection`, a schema error or a bad request
        will fail the same way eight times over, and retrying them turns a
        clear error into a slow one.

        The cached model is dropped between attempts. A judging pass runs for
        an hour or more, and the failure mode this exists for — a call that
        hangs until the deadline — is as likely to be a stale connection or an
        expired token as a busy gateway. Rebuilding costs one token
        acquisition and removes that whole class from the retry.
        """
        delay = 2.0
        for attempt in range(self._max_retries + 1):
            try:
                return asyncio.run(self._acreate(messages, response_format))
            except (TimeoutError, asyncio.TimeoutError):
                if attempt >= self._max_retries:
                    raise
                self._llm = None
                print(
                    f"  judge: no reply within {self._timeout_s:.0f}s; "
                    f"retry {attempt + 1}/{self._max_retries} in {delay:.0f}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _amodel(self) -> Any:
        """Build once, awaited and bounded. `amodel()` does token acquisition."""
        if self._llm is not None:
            return self._llm
        try:
            from safechain.core.model import amodel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - private environment only
            raise RuntimeError(
                "backend: safechain needs the `safechain` package, which only "
                "exists in the private environment. Use backend: openai here, "
                "or set LLM_BACKEND=openai."
            ) from exc
        model_id = os.environ.get("SAFECHAIN_MODEL", self._model_name)
        try:
            self._llm = await asyncio.wait_for(
                amodel(model_id), timeout=self._timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"safechain amodel() build did not return within {self._timeout_s:.0f}s"
            ) from exc
        return self._llm

    async def _acreate(
        self, messages: list[dict[str, Any]], response_format: Any,
    ) -> Any:
        model = await self._amodel()
        bound = (
            model.bind(response_format=response_format)
            if response_format is not None else model
        )
        try:
            from safechain.prompts import ValidChatPromptTemplate  # type: ignore[import-not-found]
            from langchain_core.prompts import MessagesPlaceholder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - private environment only
            # No silent fallback to invoking the model directly. The template
            # is in the chain for COMPLIANCE — its `format_prompt` override is
            # template-time, and a bare `ainvoke` skips it. Quietly bypassing
            # it would keep the run working while dropping the one thing this
            # backend exists to guarantee, and nothing downstream could tell.
            raise RuntimeError(
                "backend: safechain requires safechain.prompts."
                "ValidChatPromptTemplate and langchain_core.prompts."
                "MessagesPlaceholder. The template carries compliance and is "
                "not optional; refusing to run without it."
            ) from exc
        chain = ValidChatPromptTemplate.from_messages(
            [MessagesPlaceholder("messages")],
        ) | bound
        payload = [(str(m.get("role")), str(m.get("content"))) for m in messages]
        return await asyncio.wait_for(
            chain.ainvoke({"messages": payload}), timeout=self._timeout_s,
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
        # Same precedence as `config._content_config`: the environment names
        # the transport, because the transport is a fact about where the run
        # is happening. This path matters on its own — a client built from a
        # hand-made dict never passes through config loading.
        self.backend = str(
            os.environ.get("LLM_BACKEND") or self.config.get("backend") or "openai"
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
