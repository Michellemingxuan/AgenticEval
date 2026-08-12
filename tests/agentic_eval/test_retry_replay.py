"""The replay itself: what lands in runs.jsonl after a turn times out."""
from __future__ import annotations

import json

from agentic_eval.models import AdapterResult
from agentic_eval.runner import ComparisonRunner, RetryPolicy, Session


class _Question:
    def __init__(self, name):
        self.name, self.text, self.evaluation = name, f"{name}?", {}
        self.question_set = "s"


class _Adapter:
    """Times out on `fail_on` turns of the first pass, then answers."""

    def __init__(self, fail_on):
        self.fail_on, self.passes, self.asked, self.rewinds = fail_on, 0, [], 0

    def reset(self, case_id):
        self.rewinds += 1
        self.passes += 1

    def run(self, request, timeout_s):
        self.asked.append((self.passes, request.question.name))
        if self.passes == 1 and request.question.name in self.fail_on:
            return AdapterResult(outcome="timeout", error="busy", elapsed_seconds=1.0)
        return AdapterResult(
            outcome="ok", final_answer=f"answer to {request.question.name}",
            elapsed_seconds=1.0,
        )


def _runner(tmp_path, adapter, attempts=2):
    runner = ComparisonRunner.__new__(ComparisonRunner)
    runner.pool = [{"solo": adapter}]
    runner.workers = 1
    runner.config = type("C", (), {"experiment": {"timeout_s": 10, "cases": ["c1"]}})()
    runner.raw_path = tmp_path / "runs.jsonl"
    runner.raw_path.write_text("", encoding="utf-8")
    import threading
    runner._write_lock, runner._print_lock = threading.Lock(), threading.Lock()
    runner._retry = RetryPolicy(
        outcomes=frozenset({"timeout"}), attempts=attempts, backoff_s=0.0,
    )
    runner._order = lambda key: ["solo"]
    return runner


def _session():
    return Session("c1", "s", 1, "stateful",
                   [_Question("b2"), _Question("b3"), _Question("b4")])


def test_the_whole_pass_replays_from_the_first_question(tmp_path):
    """Not just the failed turn: /rewind emptied the conversation."""
    adapter = _Adapter(fail_on={"b3"})
    runner = _runner(tmp_path, adapter)

    records = runner._run_session(_session(), 0)

    second_pass = [name for p, name in adapter.asked if p == 2]
    assert second_pass == ["b2", "b3", "b4"], second_pass
    assert adapter.rewinds == 2                      # one per attempt
    assert [r["outcome"] for r in records] == ["ok"] * 3


def test_the_abandoned_attempt_is_not_written(tmp_path):
    """A duplicate row would be averaged in as a second, empty answer."""
    adapter = _Adapter(fail_on={"b3"})
    runner = _runner(tmp_path, adapter)

    runner._run_session(_session(), 0)

    rows = [json.loads(l) for l in runner.raw_path.open()]
    assert len(rows) == 3                            # not 6
    assert all(r["outcome"] == "ok" for r in rows)
    assert [r["name"] for r in rows] == ["b2", "b3", "b4"]


def test_the_replay_is_disclosed_on_every_kept_record(tmp_path):
    """A run that limped must not read like one that did not."""
    adapter = _Adapter(fail_on={"b3"})
    runner = _runner(tmp_path, adapter)

    rows = runner._run_session(_session(), 0)

    assert all(r["harness_attempts"] == 2 for r in rows)
    assert all(r["harness_retry_outcomes"] == ["timeout"] for r in rows)


def test_a_clean_pass_says_one_attempt_and_claims_no_retry(tmp_path):
    adapter = _Adapter(fail_on=set())
    rows = _runner(tmp_path, adapter)._run_session(_session(), 0)
    assert all(r["harness_attempts"] == 1 for r in rows)
    assert all("harness_retry_outcomes" not in r for r in rows)


def test_exhausted_attempts_keep_the_timeout_rather_than_hiding_it(tmp_path):
    """When it never succeeds, the failure is the result — recorded, not lost."""
    class _AlwaysBusy(_Adapter):
        def run(self, request, timeout_s):
            self.asked.append((self.passes, request.question.name))
            return AdapterResult(outcome="timeout", error="busy", elapsed_seconds=1.0)

    adapter = _AlwaysBusy(fail_on={"b3"})
    runner = _runner(tmp_path, adapter, attempts=1)

    rows = runner._run_session(_session(), 0)

    assert [r["outcome"] for r in rows] == ["timeout"] * 3
    assert all(r["harness_attempts"] == 2 for r in rows)   # tried twice, then kept
    assert len([json.loads(l) for l in runner.raw_path.open()]) == 3


def test_retries_are_off_by_default(tmp_path):
    adapter = _Adapter(fail_on={"b3"})
    runner = _runner(tmp_path, adapter, attempts=0)

    rows = runner._run_session(_session(), 0)

    assert [r["outcome"] for r in rows] == ["ok", "timeout", "ok"]
    assert adapter.rewinds == 1
