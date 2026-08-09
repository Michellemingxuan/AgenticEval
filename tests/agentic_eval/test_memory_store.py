"""The run must leave the system's memory store as it found it."""
from __future__ import annotations

import json

import pytest

from agentic_eval import memory_store


class _FakeStore:
    """A Qdrant-shaped stand-in: scroll returns pages, delete/upsert mutate."""

    def __init__(self, points):
        self.points = {p["id"]: p for p in points}
        self.calls = []

    def post(self, url, body, timeout):
        self.calls.append(url)
        if url.endswith("/points/scroll"):
            return {"result": {"points": list(self.points.values()),
                               "next_page_offset": None}}
        if "/points/delete" in url:
            for point_id in body["points"]:
                self.points.pop(point_id, None)
            return {"status": "ok"}
        if "/points" in url:
            for point in body["points"]:
                self.points[point["id"]] = point
            return {"status": "ok"}
        raise AssertionError(url)


@pytest.fixture
def store(monkeypatch):
    fake = _FakeStore([
        {"id": 1, "vector": [0.1], "payload": {"kind": "qa_turn"}},
        {"id": 2, "vector": [0.2], "payload": {"kind": "qa_turn"}},
    ])
    monkeypatch.setattr(memory_store, "_post", fake.post)
    return fake


def test_a_snapshot_carries_vectors(store):
    """Payload-only would leave memories that exist and can never be
    retrieved — worse than deleting them, because the store looks intact."""
    points = memory_store.snapshot("http://x", "c")
    assert [p["id"] for p in points] == [1, 2]
    assert all(p.get("vector") for p in points)


def test_restore_removes_what_the_run_added(store):
    before = memory_store.snapshot("http://x", "c")
    # The run writes two more.
    store.points[3] = {"id": 3, "vector": [0.3], "payload": {}}
    store.points[4] = {"id": 4, "vector": [0.4], "payload": {}}

    moved = memory_store.restore("http://x", "c", before)
    assert moved == {"removed": 2, "reinstated": 0}
    assert sorted(store.points) == [1, 2]


def test_restore_reinstates_what_the_run_deleted(store):
    """`/rewind` purges by case, so a run can DELETE pre-existing memories
    as well as add its own."""
    before = memory_store.snapshot("http://x", "c")
    store.points.clear()

    moved = memory_store.restore("http://x", "c", before)
    assert moved == {"removed": 0, "reinstated": 2}
    assert sorted(store.points) == [1, 2]
    assert store.points[1]["vector"] == [0.1]


def test_restoring_a_snapshot_over_itself_changes_nothing(store):
    before = memory_store.snapshot("http://x", "c")
    assert memory_store.restore("http://x", "c", before) == {
        "removed": 0, "reinstated": 0,
    }
    assert sorted(store.points) == [1, 2]
