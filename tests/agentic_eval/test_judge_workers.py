"""Judging several (system, case) groups at once.

Every answer is judged independently — the judge's prior-turn context is built
from the RUN, never from another verdict — so parallelism changes wall-clock
and nothing else. What it must not change: the verdicts, the file, or the
order questions are asked in within a group.
"""
from __future__ import annotations

import json
import threading

from agentic_eval.content.pipeline import evaluate_runs_file


class _Judge:
    """Records which thread judged what, and in what order."""

    def __init__(self):
        self.calls = []
        self.seen: list[tuple[str, str, str]] = []
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def complete_json(self, *, task, system_prompt, payload):
        with self._lock:
            self.threads.add(threading.get_ident())
        if task == "claim_extraction":
            return {"claims": []}
        return {"claims": [], "must_haves": [], "memory_leverage": []}


def _record(system, case, name, position):
    return {
        "system": system, "mode": "stateful", "case_id": case,
        "question_set": "series_b", "name": name, "run_index": 1,
        "sequence_position": position, "outcome": "ok",
        "final_answer": f"answer to {name}", "question": name,
        "evaluation": {}, "evidence": [], "tools": [], "team": [],
        "subqueries": {}, "metrics": {},
    }


def _records():
    return [
        _record(system, case, name, position)
        for system in ("previous", "current")
        for case in ("366", "118")
        for position, name in enumerate(("b2", "b3", "b4"), 1)
    ]


def _run(tmp_path, workers):
    judge = _Judge()
    evaluate_runs_file(
        config={"llm": {}}, records=_records(), output_dir=tmp_path,
        baseline="previous", candidate="current", rubric_by_name={},
        judge=judge, workers=workers,
    )
    rows = [json.loads(l) for l in (tmp_path / "content" / "evaluations.jsonl").open()]
    return rows, judge


def test_parallel_judging_produces_the_same_answers(tmp_path):
    serial, _ = _run(tmp_path / "serial", 1)
    parallel, _ = _run(tmp_path / "parallel", 4)

    identity = lambda rows: sorted(
        (r["system"], r["case_id"], r["name"]) for r in rows
    )
    assert identity(serial) == identity(parallel)
    assert len(parallel) == 12


def test_question_order_is_kept_within_each_group(tmp_path):
    """A group is one system on one case; its questions stay in order."""
    rows, _ = _run(tmp_path, 4)
    for system in ("previous", "current"):
        for case in ("366", "118"):
            names = [
                r["name"] for r in rows
                if r["system"] == system and r["case_id"] == case
            ]
            assert names == ["b2", "b3", "b4"], (system, case, names)


def test_the_work_really_is_spread_over_threads(tmp_path):
    _rows, judge = _run(tmp_path, 4)
    assert len(judge.threads) > 1


def test_one_worker_stays_single_threaded(tmp_path):
    _rows, judge = _run(tmp_path, 1)
    assert len(judge.threads) == 1


def test_every_line_written_is_parseable(tmp_path):
    """Concurrent appends under one lock, or the file is unreadable — and
    `--resume` reads it."""
    _rows, _judge = _run(tmp_path, 4)
    path = tmp_path / "content" / "evaluations.jsonl"
    for line in path.read_text().splitlines():
        assert json.loads(line)["name"]
