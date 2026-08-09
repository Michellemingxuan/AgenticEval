"""Concurrency: one server instance per worker, questions serial within one."""
from __future__ import annotations

import threading
import time

import pytest

from agentic_eval.config import load_config
from agentic_eval.models import AdapterResult, RunRequest
from agentic_eval.workers import assert_workers_are_isolated, bind_to_worker


def _system(port: int = 49102) -> dict:
    return {
        "adapter": "agenticsys_sse",
        "config": {"base_url": f"http://127.0.0.1:{port}", "case_id": "c1"},
        "process": {
            "command": ["python3", "server.py"],
            "env": {"PORT": str(port)},
            "stdout": "/logs/prev.log",
        },
    }


def test_worker_zero_is_the_config_exactly_as_written():
    """A serial run must be byte-for-byte what it always was."""
    target = _system()
    assert bind_to_worker("previous", target, 0) is target


def test_each_worker_gets_its_own_port_on_both_sides():
    """The URL and the server's own PORT must move together.

    Moving one without the other starts a server nobody addresses, or points
    the adapter at another worker's server and quietly measures that instead.
    """
    bound = bind_to_worker("previous", _system(49102), 2)
    assert bound["config"]["base_url"] == "http://127.0.0.1:49104"
    assert bound["process"]["env"]["PORT"] == "49104"
    # One log per worker, or N servers append to one file with no line saying
    # which instance wrote it.
    assert bound["process"]["stdout"] == "/logs/prev.w2.log"


def test_a_mismatched_port_pair_fails_loudly():
    target = _system(49102)
    target["process"]["env"]["PORT"] = "49999"
    with pytest.raises(ValueError, match="disagree"):
        bind_to_worker("previous", target, 1)


def test_a_url_without_a_port_fails_loudly():
    target = _system()
    target["config"]["base_url"] = "http://127.0.0.1"
    with pytest.raises(ValueError, match="no port"):
        bind_to_worker("previous", target, 1)


def test_colliding_ports_are_caught_before_anything_starts():
    """Systems one apart collide as soon as there are two workers."""
    pool = [
        {"previous": _system(49102), "current": _system(49103)},
        {"previous": _system(49103), "current": _system(49104)},
    ]
    with pytest.raises(ValueError, match="port collision"):
        assert_workers_are_isolated(pool)


def test_a_shared_trace_db_is_caught_before_anything_starts():
    """`/rewind` deletes trace rows by case id across every process.

    Two workers on one DB therefore wipe each other's in-flight evidence — not
    a crash, just turns that record no tool calls and a consistency metric that
    reports the system as inconsistent.
    """
    a, b = _system(49102), _system(49103)
    a["config"]["trace_db"] = b["config"]["trace_db"] = "/traces/current.db"
    with pytest.raises(ValueError, match="trace DB collision"):
        assert_workers_are_isolated([{"previous": a}, {"previous": b}])


def test_each_worker_gets_its_own_trace_db():
    target = _system(49102)
    target["config"]["trace_db"] = "/traces/current.db"
    target["process"]["env"]["NODE_TRACE_DB"] = "/traces/current.db"
    bound = bind_to_worker("current", target, 2)
    assert bound["config"]["trace_db"] == "/traces/current.w2.db"
    # The server must WRITE the file the evaluator READS.
    assert bound["process"]["env"]["NODE_TRACE_DB"] == bound["config"]["trace_db"]


def test_workers_above_the_ceiling_are_refused(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\n"
        "experiment: {baseline: old, candidate: new, workers: 20}\n"
        "questions:\n  - {name: q1, question: 'ask?'}\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o:1', case_id: c1}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n:2', case_id: c1}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Keep it <= 8"):
        load_config(str(path))


class _TrackingAdapter:
    """Records which adapter instance served each turn, and concurrency."""

    live = 0
    live_lock = threading.Lock()
    max_live = 0

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.asked: list[tuple] = []

    def healthcheck(self) -> None:
        pass

    def reset(self, case_id=None) -> None:
        pass

    def run(self, request: RunRequest, timeout_s: float) -> AdapterResult:
        with _TrackingAdapter.live_lock:
            _TrackingAdapter.live += 1
            _TrackingAdapter.max_live = max(
                _TrackingAdapter.max_live, _TrackingAdapter.live
            )
        # A turn takes non-zero time in reality; without a dwell here the
        # increment and decrement are adjacent and overlap is unobservable.
        time.sleep(0.02)
        self.asked.append(
            (request.case_id, request.question.name, request.run_index)
        )
        with _TrackingAdapter.live_lock:
            _TrackingAdapter.live -= 1
        return AdapterResult(
            turn_id="t", final_answer="a", outcome="ok", elapsed_seconds=0.01,
        )


def _runner(tmp_path, monkeypatch, workers: int, slot: str = "a"):
    from agentic_eval import runner as runner_module

    root = tmp_path / slot
    root.mkdir(exist_ok=True)
    (root / "series_b.yaml").write_text(
        "questions:\n"
        "  - {name: b1, question: 'first'}\n"
        "  - {name: b2, question: 'second'}\n"
        "  - {name: b3, question: 'third'}\n",
        encoding="utf-8",
    )
    path = root / "config.yaml"
    path.write_text(
        "version: 1\n"
        "experiment:\n"
        "  baseline: old\n  candidate: new\n  repeats: 2\n  mode: stateful\n"
        f"  workers: {workers}\n  cases: ['case_a', 'case_b']\n"
        f"  output_dir: {root / 'out'}\n"
        "questions_file: [series_b.yaml]\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o:100'}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n:200'}}\n",
        encoding="utf-8",
    )
    built: list[_TrackingAdapter] = []

    def fake_build(name, config):
        built.append(_TrackingAdapter(str(config.get("base_url"))))
        return built[-1]

    monkeypatch.setattr(runner_module, "build_adapter", fake_build)
    _TrackingAdapter.max_live = 0
    return runner_module.ComparisonRunner(load_config(str(path))), built


def test_questions_keep_their_order_inside_every_session(tmp_path, monkeypatch):
    """Parallelism is across sessions; a conversation is never reordered."""
    runner, built = _runner(tmp_path, monkeypatch, workers=3)
    records = runner._stateful()
    assert len(records) == 2 * 2 * 3 * 2  # cases x repeats x questions x systems
    for adapter in built:
        # Every session this adapter served asked b1, b2, b3 in that order.
        names = [name for _case, name, _run in adapter.asked]
        for start in range(0, len(names), 3):
            assert names[start:start + 3] == ["b1", "b2", "b3"]
        # A session never straddles two cases or two repeats.
        for start in range(0, len(adapter.asked), 3):
            chunk = adapter.asked[start:start + 3]
            assert len({(case, run) for case, _n, run in chunk}) == 1


def test_sessions_actually_run_concurrently(tmp_path, monkeypatch):
    runner, _ = _runner(tmp_path, monkeypatch, workers=3)
    runner._stateful()
    assert _TrackingAdapter.max_live > 1


def test_each_worker_talks_only_to_its_own_server(tmp_path, monkeypatch):
    """Two workers sharing a server would interleave on its process globals."""
    runner, built = _runner(tmp_path, monkeypatch, workers=3)
    urls = [adapter.base_url for adapter in built]
    assert sorted(urls) == [
        "http://n:200", "http://n:201", "http://n:202",
        "http://o:100", "http://o:101", "http://o:102",
    ]
    # One adapter per (worker, system) — no instance is shared between workers.
    assert len(set(urls)) == len(urls)


def test_the_record_order_does_not_depend_on_worker_timing(tmp_path, monkeypatch):
    """runs.jsonl must read the same however the workers finished."""
    serial, _ = _runner(tmp_path, monkeypatch, workers=1, slot="serial")
    serial_records = serial._stateful()
    parallel, _ = _runner(tmp_path, monkeypatch, workers=3, slot="parallel")
    parallel_records = parallel._stateful()
    identity = lambda rows: [
        (r["case_id"], r["question_set"], r["run_index"], r["system"],
         r["name"], r["sequence_position"])
        for r in rows
    ]
    assert identity(serial_records) == identity(parallel_records)
