"""Leaving the system's memory store as the run found it.

Evaluation writes to a REAL memory store. Every turn deposits a `qa_turn`
memory keyed by case, and `/rewind` purges only the case it is clearing — so
whatever the last session of each case wrote survives the run and is still
there for the next one, and for whatever else uses that store.

Two problems follow, and this module addresses the second:

  * Repeats stop being independent. Fixed in the runner instead, by giving one
    worker ownership of a whole case: rewind then clears the case's memories
    at the start of every session, with no concurrent writer to race.

  * The store accumulates. Measured on one afternoon of small runs: thirteen
    memories, from two systems, both readable by the other for the same case.
    A test that changes the environment it measures is not repeatable, and the
    residue is indistinguishable from real operating history.

So: snapshot before, restore after. The store ends the run holding exactly the
points it held at the start — same ids, same payloads, same vectors.

Deliberately thin and optional. Nothing here is imported unless a config asks
for it, and a store that cannot be reached is reported rather than treated as
empty: "no snapshot" and "snapshot of nothing" must not look alike, or a failed
read would silently authorise deleting everything.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

#: Read in pages; a store with more memories than this per page still gets all
#: of them, the scroll just runs more than once.
_PAGE = 256


def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or "{}")


def snapshot(
    url: str, collection: str, *, timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Every point in the collection, with payload AND vector.

    The vector is what makes a restore faithful: re-inserting payloads alone
    would leave memories that exist but can never be retrieved, which is worse
    than deleting them outright because it looks like the store is intact.
    """
    points: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": _PAGE, "with_payload": True, "with_vector": True,
        }
        if offset is not None:
            body["offset"] = offset
        result = _post(
            f"{url.rstrip('/')}/collections/{collection}/points/scroll",
            body, timeout,
        ).get("result") or {}
        points.extend(result.get("points") or [])
        offset = result.get("next_page_offset")
        if offset is None:
            return points


def restore(
    url: str, collection: str, points: list[dict[str, Any]], *,
    timeout: float = 30.0,
) -> dict[str, int]:
    """Put the collection back to exactly `points`.

    Deletes what is there now, then re-inserts the snapshot. Returns what was
    removed and what was put back, so the caller can report the run's footprint
    rather than assert it was zero.
    """
    base = f"{url.rstrip('/')}/collections/{collection}"
    current = [
        point["id"] for point in snapshot(url, collection, timeout=timeout)
    ]
    keep = {point["id"] for point in points}
    remove = [point_id for point_id in current if point_id not in keep]
    if remove:
        _post(f"{base}/points/delete?wait=true", {"points": remove}, timeout)
    missing = [point for point in points if point["id"] not in set(current)]
    if missing:
        _post(f"{base}/points?wait=true", {"points": [
            {
                "id": point["id"],
                "vector": point.get("vector"),
                "payload": point.get("payload") or {},
            }
            for point in missing
        ]}, timeout)
    return {"removed": len(remove), "reinstated": len(missing)}
