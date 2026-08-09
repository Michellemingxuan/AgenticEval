"""Multi-case runs: discovery, config resolution, and the run loop."""
from __future__ import annotations

import pytest

from agentic_eval.cases import (
    describe_case, discover_case_ids, whitespace_padded_case_ids,
)
from agentic_eval.config import load_config
from agentic_eval.models import AdapterResult, RunRequest


def test_discovery_lists_case_directories_and_ignores_files(tmp_path):
    (tmp_path / "366132845011").mkdir()
    (tmp_path / "11854808010").mkdir()
    (tmp_path / "README.md").write_text("not a case", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    assert discover_case_ids(tmp_path) == ["11854808010", "366132845011"]


def test_discovery_preserves_a_trailing_space_in_a_case_name(tmp_path):
    """The real data has one of these, and the system keys on the raw name.

    `LocalDataGateway.from_case_folders` stores `case_dir.name` verbatim, so
    stripping here would yield an id matching no case — the run would complete
    with every answer empty, which reads as a system failure rather than ours.
    """
    (tmp_path / "11854808010 ").mkdir()
    assert discover_case_ids(tmp_path) == ["11854808010 "]
    assert whitespace_padded_case_ids(["11854808010 ", "366"]) == ["11854808010 "]
    # Quoted so the padding is visible in console output and page labels.
    assert describe_case("11854808010 ") == "'11854808010 '"
    assert describe_case("366132845011") == "366132845011"


def test_discovery_fails_loudly_on_a_missing_or_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_case_ids(tmp_path / "nope")
    with pytest.raises(ValueError, match="no case directories"):
        discover_case_ids(tmp_path)


def _config(tmp_path, experiment: str, systems: str | None = None) -> str:
    body = systems or (
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o', case_id: c1}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n', case_id: c1}}\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\n"
        f"experiment:\n{experiment}"
        "questions:\n  - {name: q1, question: 'ask?'}\n"
        f"systems:\n{body}",
        encoding="utf-8",
    )
    return str(path)


def test_a_single_case_config_still_resolves_to_one_case(tmp_path):
    config = load_config(_config(tmp_path, "  baseline: old\n  candidate: new\n"))
    assert config.experiment["cases"] == ["c1"]


def test_experiment_cases_lists_several(tmp_path):
    config = load_config(_config(
        tmp_path,
        "  baseline: old\n  candidate: new\n  cases: ['case_a', 'case_b']\n",
    ))
    assert config.experiment["cases"] == ["case_a", "case_b"]


def test_cases_from_discovers_them(tmp_path):
    (tmp_path / "real" / "case_a").mkdir(parents=True)
    (tmp_path / "real" / "case_b").mkdir()
    config = load_config(_config(
        tmp_path, "  baseline: old\n  candidate: new\n  cases_from: real\n",
    ))
    assert config.experiment["cases"] == ["case_a", "case_b"]


def test_systems_disagreeing_on_case_id_is_an_error(tmp_path):
    """Two systems asked about different customers is not a comparison."""
    with pytest.raises(ValueError, match="disagree on case_id"):
        load_config(_config(
            tmp_path, "  baseline: old\n  candidate: new\n",
            systems=(
                "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o', case_id: c1}}\n"
                "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n', case_id: c2}}\n"
            ),
        ))


def test_a_repeated_case_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="repeats"):
        load_config(_config(
            tmp_path,
            "  baseline: old\n  candidate: new\n  cases: ['case_a', 'case_a']\n",
        ))


def test_cases_and_cases_from_together_is_an_error(tmp_path):
    (tmp_path / "real").mkdir()
    with pytest.raises(ValueError, match="alternatives"):
        load_config(_config(
            tmp_path,
            "  baseline: old\n  candidate: new\n"
            "  cases: ['case_a']\n  cases_from: real\n",
        ))


class _RecordingAdapter:
    """Captures the (case, question) sequence and every reset."""

    def __init__(self) -> None:
        self.asked: list[tuple[str | None, str, int]] = []
        self.resets: list[str | None] = []

    def healthcheck(self) -> None:
        pass

    def reset(self, case_id=None) -> None:
        self.resets.append(case_id)

    def run(self, request: RunRequest, timeout_s: float) -> AdapterResult:
        self.asked.append(
            (request.case_id, request.question.name, request.run_index)
        )
        return AdapterResult(
            turn_id="t", final_answer="an answer", outcome="ok",
            elapsed_seconds=0.1,
        )


def _runner(tmp_path, experiment_extra: str, monkeypatch):
    from agentic_eval import runner as runner_module

    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\n"
        "experiment:\n"
        "  baseline: old\n  candidate: new\n  repeats: 1\n"
        f"  output_dir: {tmp_path / 'out'}\n"
        f"{experiment_extra}"
        "questions:\n"
        "  - {name: q1, question: 'first?'}\n"
        "  - {name: q2, question: 'second?'}\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o'}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n'}}\n",
        encoding="utf-8",
    )
    adapters: dict[str, _RecordingAdapter] = {}

    def fake_build(name, config):
        adapter = _RecordingAdapter()
        adapters[str(len(adapters))] = adapter
        return adapter

    monkeypatch.setattr(runner_module, "build_adapter", fake_build)
    return runner_module.ComparisonRunner(load_config(str(path))), adapters


def test_a_cold_run_asks_every_question_about_every_case(tmp_path, monkeypatch):
    runner, adapters = _runner(
        tmp_path, "  mode: cold\n  cases: ['case_a', 'case_b']\n", monkeypatch,
    )
    records = runner._cold()
    assert len(records) == 2 * 2 * 2  # cases x questions x systems
    assert {record["case_id"] for record in records} == {"case_a", "case_b"}
    for adapter in adapters.values():
        assert [case for case, _, _ in adapter.asked] == [
            "case_a", "case_a", "case_b", "case_b",
        ]
        # Each turn is reset against the case it is about; resetting the
        # previous case would leave the next one carrying its history.
        assert adapter.resets == [case for case, _, _ in adapter.asked]


def test_a_stateful_run_resets_between_cases(tmp_path, monkeypatch):
    """Case B must start empty, or it answers with case A's conversation."""
    runner, adapters = _runner(
        tmp_path, "  mode: stateful\n  cases: ['case_a', 'case_b']\n", monkeypatch,
    )
    records = runner._stateful()
    assert len(records) == 2 * 2 * 2
    for adapter in adapters.values():
        assert adapter.resets == ["case_a", "case_b"]
        assert adapter.asked == [
            ("case_a", "q1", 1), ("case_a", "q2", 1),
            ("case_b", "q1", 1), ("case_b", "q2", 1),
        ]


def test_the_adapter_prefers_the_requests_case_over_the_configured_one():
    from agentic_eval.adapters.agenticsys_sse import AgenticSysSSEAdapter

    adapter = AgenticSysSSEAdapter({"base_url": "http://x", "case_id": "configured"})
    assert adapter._quoted_case("asked") == "asked"
    assert adapter._quoted_case(None) == "configured"
    # A space survives the round trip rather than truncating the id.
    assert adapter._quoted_case("11854808010 ") == "11854808010%20"


def test_the_adapter_fails_loudly_when_no_case_is_named():
    from agentic_eval.adapters.agenticsys_sse import AgenticSysSSEAdapter

    adapter = AgenticSysSSEAdapter({"base_url": "http://x"})
    with pytest.raises(RuntimeError, match="no case id"):
        adapter._quoted_case(None)


def test_the_whitespace_note_goes_to_stderr_not_stdout(tmp_path, capsys):
    """`validate` writes JSON that `bin/compare` parses.

    The note printed alongside it on stdout made the document unparseable, so
    a multi-case plan crashed the driver script before anything ran.
    """
    import argparse

    from agentic_eval.cli import _apply_overrides

    config = load_config(_config(tmp_path, "  baseline: old\n  candidate: new\n"))
    args = argparse.Namespace(case_ids=["11854808010 "], cases_from=None)
    _apply_overrides(config, args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "leading/trailing whitespace" in captured.err
    assert config.experiment["cases"] == ["11854808010 "]


def _question_files(tmp_path):
    (tmp_path / "series_b.yaml").write_text(
        "questions:\n"
        "  - {name: b1, question: 'set the scene'}\n"
        "  - {name: b9, question: 'Any model opportunities?'}\n",
        encoding="utf-8",
    )
    (tmp_path / "series_d.yaml").write_text(
        "questions:\n"
        "  - {name: d1, question: 'Any model opportunities?'}\n",
        encoding="utf-8",
    )


def test_each_questions_file_becomes_its_own_set(tmp_path):
    _question_files(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\n"
        "experiment: {baseline: old, candidate: new}\n"
        "questions_file: [series_b.yaml, series_d.yaml]\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o', case_id: c1}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n', case_id: c1}}\n",
        encoding="utf-8",
    )
    config = load_config(str(path))
    assert [(q.name, q.question_set) for q in config.questions] == [
        ("b1", "series_b"), ("b9", "series_b"), ("d1", "series_d"),
    ]


def _set_runner(tmp_path, monkeypatch, mode="stateful"):
    from agentic_eval import runner as runner_module

    _question_files(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\n"
        "experiment:\n"
        f"  baseline: old\n  candidate: new\n  repeats: 1\n  mode: {mode}\n"
        f"  output_dir: {tmp_path / 'out'}\n"
        "questions_file: [series_b.yaml, series_d.yaml]\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o', case_id: c1}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n', case_id: c1}}\n",
        encoding="utf-8",
    )
    adapters: list[_RecordingAdapter] = []

    def fake_build(name, config):
        adapters.append(_RecordingAdapter())
        return adapters[-1]

    monkeypatch.setattr(runner_module, "build_adapter", fake_build)
    return runner_module.ComparisonRunner(load_config(str(path))), adapters


def test_each_set_is_its_own_stateful_session(tmp_path, monkeypatch):
    """Series D asks series B's last question with none of B's context.

    Run in one session, D1 followed B9 — the identical question — two turns
    later, so it measured the QA cache rather than cold discovery. A set is a
    conversation, so each one gets a reset and restarts at position 1.
    """
    runner, adapters = _set_runner(tmp_path, monkeypatch)
    records = runner._stateful()
    for adapter in adapters:
        # One reset per set, not one per repeat.
        assert adapter.resets == ["c1", "c1"]
    by_name = {r["name"]: r for r in records if r["system"] == "old"}
    assert by_name["b1"]["sequence_position"] == 1
    assert by_name["b9"]["sequence_position"] == 2
    # D1 is turn ONE of its own conversation, not turn three of B's.
    assert by_name["d1"]["sequence_position"] == 1
    assert by_name["d1"]["question_set"] == "series_d"
    assert by_name["b9"]["question_set"] == "series_b"


def test_the_judge_context_stops_at_the_set_boundary(tmp_path, monkeypatch):
    """`prior_turns` must match the session the system actually had.

    Without the set in the key, D1's cold answer would be judged against
    series B's turns — the very context the run withheld from the system.
    """
    from agentic_eval.content.pipeline import evaluate_runs_file

    runner, _ = _set_runner(tmp_path, monkeypatch)
    records = runner._stateful()
    for record in records:
        record["final_answer"] = f"answer for {record['name']}"
        record["outcome"] = "ok"

    seen: dict[str, list] = {}

    class _Spy:
        def evaluate(self, record, rubric, prior_turns):
            seen[str(record["name"])] = list(prior_turns)
            return {"system": record["system"], "name": record["name"],
                    "metrics": {}, "case_id": record.get("case_id"),
                    "question_set": record.get("question_set"),
                    "run_index": record.get("run_index")}

    import agentic_eval.content.pipeline as pipeline_module
    monkeypatch.setattr(
        pipeline_module, "ContentEvaluator", lambda *a, **k: _Spy(),
    )
    evaluate_runs_file(
        config={"expected_repeats": 1}, records=records,
        output_dir=tmp_path / "out2", baseline="old", candidate="new",
        rubric_by_name={},
    )
    assert seen["b1"] == []
    assert [t["question"] for t in seen["b9"]] == ["set the scene"]
    # The cold ask sees nothing, though B9 asked the same words before it.
    assert seen["d1"] == []
