"""A timed-out judge call must not end a run that has already paid for itself.

`max_retries` was read only on the OpenAI branch, so a config asking for eight
got zero here — and the eligibility call for one answer, arriving one second
past the deadline, discarded every answer judged before it.
"""
from __future__ import annotations

import asyncio

import pytest

from agentic_eval.llm_judge import SafeChainClient, build_client


def _client(monkeypatch, outcomes, **kwargs):
    """A client whose transport yields `outcomes` in order.

    Each outcome is either an exception to raise or a value to return.
    """
    client = SafeChainClient("gpt-4.1", timeout_s=1.0, **kwargs)
    calls = {"n": 0}

    async def fake_acreate(messages, response_format):
        index = calls["n"]
        calls["n"] += 1
        outcome = outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client, "_acreate", fake_acreate)
    monkeypatch.setattr("agentic_eval.llm_judge.time.sleep", lambda _s: None)
    return client, calls


class _Reply:
    content = '{"ok": true}'
    usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


def test_a_timeout_is_retried_and_the_answer_survives(monkeypatch):
    client, calls = _client(
        monkeypatch, [TimeoutError(), TimeoutError(), _Reply()], max_retries=8,
    )
    reply = client.chat.completions.create(model="gpt-4.1", messages=[])
    assert reply.choices[0].message.content == '{"ok": true}'
    assert calls["n"] == 3


def test_the_model_is_rebuilt_between_attempts(monkeypatch):
    """A hung call is as likely to be a stale connection as a busy gateway."""
    client, _ = _client(monkeypatch, [TimeoutError(), _Reply()], max_retries=2)
    client._llm = object()                      # pretend one was cached
    client.chat.completions.create(model="gpt-4.1", messages=[])
    assert client._llm is None                  # dropped, so the next build is fresh


def test_retries_are_finite_and_the_timeout_still_surfaces(monkeypatch):
    client, calls = _client(
        monkeypatch, [TimeoutError()] * 5, max_retries=2,
    )
    with pytest.raises(TimeoutError):
        client.chat.completions.create(model="gpt-4.1", messages=[])
    assert calls["n"] == 3                      # the first try plus two retries


def test_only_timeouts_are_retried(monkeypatch):
    """A rejection or a schema error fails identically eight times over.

    Retrying those turns a clear error into a slow one and hides the cause.
    """
    client, calls = _client(
        monkeypatch, [ValueError("firewall rejected the payload")], max_retries=8,
    )
    with pytest.raises(ValueError, match="firewall"):
        client.chat.completions.create(model="gpt-4.1", messages=[])
    assert calls["n"] == 1


def test_the_configured_retry_count_reaches_the_client():
    """The gap itself: config said 8, the safechain path used none."""
    client = build_client(
        {"model": "gpt-4.1", "max_retries": 5}, "safechain", 60.0,
    )
    assert client._max_retries == 5
    assert build_client({}, "safechain", 60.0)._max_retries == 8


def test_asyncio_timeout_is_the_same_failure(monkeypatch):
    """3.11 aliases them, but the client must not depend on that."""
    client, calls = _client(
        monkeypatch, [asyncio.TimeoutError(), _Reply()], max_retries=1,
    )
    client.chat.completions.create(model="gpt-4.1", messages=[])
    assert calls["n"] == 2
