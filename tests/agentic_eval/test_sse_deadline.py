"""The turn deadline has to survive a stream that never yields an event.

Observed in the private environment: `b4_abnormal_transactions` ran for
34626 seconds — 9.6 hours — against `timeout_s: 600`. Nothing raised, because
every individual socket read completed well inside its own timeout. The stream
was alive; it just never finished a turn.
"""
from __future__ import annotations

import time

import pytest

from agentic_eval.adapters.agenticsys_sse import iter_sse


class _Heartbeats:
    """An endless SSE comment stream — exactly what a keepalive looks like."""

    def __init__(self, limit=100_000):
        self.reads, self.limit = 0, limit

    def __iter__(self):
        return self

    def __next__(self):
        self.reads += 1
        if self.reads > self.limit:      # a real socket would never stop
            raise AssertionError("iter_sse did not honour the deadline")
        return b": ping\n"


def test_a_heartbeat_only_stream_hits_the_deadline():
    """The bug: comments are skipped, so no event is ever yielded.

    A caller checking the clock once per yielded event therefore never checks
    it, and the turn runs unbounded while the socket stays happy.
    """
    stream = _Heartbeats()
    with pytest.raises(TimeoutError, match="deadline"):
        list(iter_sse(stream, deadline=time.monotonic() - 1))


def test_the_deadline_is_checked_per_line_not_per_event():
    """Data lines that never complete an event must not buy unlimited time."""
    class _Dribble:
        def __init__(self):
            self.reads = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.reads += 1
            if self.reads > 50_000:
                raise AssertionError("never timed out")
            return b"data: {}\n"        # no blank line, so nothing is yielded

    with pytest.raises(TimeoutError):
        list(iter_sse(_Dribble(), deadline=time.monotonic() - 1))


def test_events_still_parse_when_the_deadline_is_generous():
    stream = iter(
        [b"event: final\n", b'data: {"answer": "hi"}\n', b"\n",
         b"event: turn_done\n", b'data: {"outcome": "ok"}\n', b"\n"]
    )
    events = list(iter_sse(stream, deadline=time.monotonic() + 60))
    assert [name for name, _ in events] == ["final", "turn_done"]
    assert events[0][1]["answer"] == "hi"


def test_no_deadline_keeps_the_old_behaviour():
    """`iter_sse` is used elsewhere without a clock; it must stay usable."""
    stream = iter([b"event: ping\n", b"data: {}\n", b"\n"])
    assert [name for name, _ in iter_sse(stream)] == ["ping"]
