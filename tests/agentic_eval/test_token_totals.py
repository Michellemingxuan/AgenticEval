"""Totalling tokens when a system fills `total_tokens` only sometimes.

The baseline does exactly that on the safechain path: its orchestrator rows
carry completions and no prompt estimate, and no `total_tokens` at all. Summing
the column with `or 0` therefore reported the completions alone — 1028 against
the candidate's 293172, a 140x "improvement" that was entirely this.
"""
from __future__ import annotations

import json
import sqlite3

from agentic_eval.adapters.agenticsys_sse import _trace_fields


def _db(tmp_path, rows):
    path = tmp_path / "trace.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE node_trace (id INTEGER PRIMARY KEY, turn_id TEXT,
        node TEXT, depth INTEGER, parent_id INTEGER, tags TEXT,
        prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
        messages_json TEXT)""")
    conn.executemany(
        "INSERT INTO node_trace VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(i, "t1", node, 1, None, json.dumps([]), p, c, tot, None)
         for i, (node, p, c, tot) in enumerate(rows, 1)],
    )
    conn.commit(); conn.close()
    return path


def test_a_partly_filled_total_column_is_reconstructed(tmp_path):
    """Rows with a total keep it; rows without get prompt + completion."""
    path = _db(tmp_path, [
        ("specialist.round_1", 20000, 500, None),   # no total of its own
        ("orchestrator.round_1", None, 300, 300),   # completion only, has total
    ])
    out = _trace_fields(str(path), "t1")
    assert out["prompt_tokens"] == 20000
    assert out["completion_tokens"] == 800
    assert out["total_tokens"] == 20500 + 300


def test_a_fully_filled_total_column_is_used_as_given(tmp_path):
    """Where the system reports a real total, it wins over reconstruction —
    cached and reasoning tokens do not appear in prompt + completion."""
    path = _db(tmp_path, [
        ("a", 100, 10, 999),
        ("b", 200, 20, 888),
    ])
    out = _trace_fields(str(path), "t1")
    assert out["total_tokens"] == 999 + 888


def test_a_column_nobody_filled_reads_unmeasured_not_zero(tmp_path):
    """`or 0` reported "nobody measured this" as a confident zero."""
    path = _db(tmp_path, [("a", None, None, None)])
    out = _trace_fields(str(path), "t1")
    assert out["prompt_tokens"] is None
    assert out["completion_tokens"] is None
    assert out["total_tokens"] is None


def test_the_two_systems_become_comparable(tmp_path):
    """The shape that produced 712 against 100,870."""
    baseline = _db(tmp_path / "b", [
        ("specialist.round_1", 37000, 700, None),
        ("orchestrator.round_1", None, 328, 328),
    ]) if (tmp_path / "b").mkdir() is None else None
    out = _trace_fields(str(baseline), "t1")
    # Not 1028: the prompt tokens it had already read are counted.
    assert out["total_tokens"] == 37700 + 328
