"""Restoring the system's memory store must not be able to lose memories.

Written after a restore destroyed 22 of them: the re-insert used POST, which
is Qdrant's RETRIEVE endpoint, and the delete had already run. The bug was
invisible for weeks because every run began with an empty store — there was
nothing to put back, so the broken call never executed.
"""
from __future__ import annotations

import pytest

from agentic_eval import memory_store


class _Store:
    """A recording double: remembers calls, and can fail a chosen one."""

    def __init__(self, existing, fail_on=None):
        self.points = {p["id"]: p for p in existing}
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on

    def send(self, method, url, body, timeout):
        path = url.split("/collections/")[-1]
        self.calls.append((method, path.split("?")[0]))
        if self.fail_on and self.fail_on in path and method == self.fail_on_method:
            raise RuntimeError("HTTP Error 400: Bad Request")
        if path.endswith("points/scroll"):
            return {"result": {"points": list(self.points.values()),
                               "next_page_offset": None}}
        if "points/delete" in path:
            for point_id in body["points"]:
                self.points.pop(point_id, None)
            return {"result": {}}
        if method == "PUT":
            for point in body["points"]:
                self.points[point["id"]] = point
            return {"result": {}}
        raise AssertionError(f"unexpected {method} {path}")

    fail_on_method = "PUT"


def _point(i):
    return {"id": i, "vector": [0.1, 0.2], "payload": {"note": f"m{i}"}}


def _install(monkeypatch, store):
    monkeypatch.setattr(memory_store, "_send", store.send)


def test_an_upsert_uses_put_not_post(monkeypatch):
    """POST /points is the retrieve endpoint; it 400s on an upsert body."""
    store = _Store(existing=[_point(9)])          # run wrote 9; 1 and 2 are ours
    _install(monkeypatch, store)

    memory_store.restore("http://x", "c", [_point(1), _point(2)])

    methods = {method for method, path in store.calls if path.endswith("points")}
    assert methods == {"PUT"}, store.calls


def test_the_snapshot_is_reinstated_before_anything_is_deleted(monkeypatch):
    """Order is the safety property, not a detail.

    Delete-then-insert loses data when the insert fails; insert-then-delete
    leaves a superset, which a second attempt tidies.
    """
    store = _Store(existing=[_point(9)])
    _install(monkeypatch, store)

    memory_store.restore("http://x", "c", [_point(1)])

    ordered = [path for _method, path in store.calls if "scroll" not in path]
    assert ordered.index("c/points") < ordered.index("c/points/delete")


def test_a_failed_reinstate_deletes_nothing(monkeypatch):
    """The failure that cost 22 memories, pinned."""
    store = _Store(existing=[_point(9)], fail_on="points")
    _install(monkeypatch, store)

    with pytest.raises(RuntimeError, match="400"):
        memory_store.restore("http://x", "c", [_point(1), _point(2)])

    # The run's own memory is still there — untidy, and recoverable.
    assert set(store.points) == {9}
    assert not any("delete" in path for _m, path in store.calls)


def test_restore_puts_the_store_back_exactly(monkeypatch):
    before = [_point(1), _point(2)]
    store = _Store(existing=[_point(1), _point(2), _point(8), _point(9)])
    _install(monkeypatch, store)

    moved = memory_store.restore("http://x", "c", before)

    assert set(store.points) == {1, 2}
    assert moved == {"removed": 2, "reinstated": 0}


def test_a_restore_that_did_not_land_is_an_error(monkeypatch):
    """Silence here is what let a broken restore look successful."""
    class _Deaf(_Store):
        def send(self, method, url, body, timeout):
            if method == "PUT":            # accepts, stores nothing
                self.calls.append((method, "c/points"))
                return {"result": {}}
            return super().send(method, url, body, timeout)

    store = _Deaf(existing=[_point(9)])
    _install(monkeypatch, store)

    with pytest.raises(RuntimeError, match="did not land"):
        memory_store.restore("http://x", "c", [_point(1)])
