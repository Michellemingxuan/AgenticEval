import pathlib
import pytest
from pathlib import Path

from agentic_eval.config import load_config


def test_config_resolves_paths_and_injects_trace_db(tmp_path: Path):
    questions = tmp_path / "questions.json"
    questions.write_text(
        '{"test_cases":[{"name":"q1","question":"Question?"}]}',
        encoding="utf-8",
    )
    cfg = tmp_path / "compare.yaml"
    cfg.write_text(
        """
version: 1
experiment:
  baseline: old
  candidate: new
  repeats: 2
  output_dir: ./out
questions_file: ./questions.json
systems:
  old:
    adapter: agenticsys_sse
    process:
      cwd: ./old
      command: [python, server.py]
    config:
      base_url: http://127.0.0.1:1
      case_id: c1
      trace_db: ./traces/old.db
  new:
    adapter: agenticsys_sse
    config:
      base_url: http://127.0.0.1:2
      case_id: c1
""",
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    old = loaded.systems["old"]
    assert loaded.questions[0].text == "Question?"
    assert old["process"]["cwd"] == str((tmp_path / "old").resolve())
    assert old["config"]["trace_db"] == str((tmp_path / "traces/old.db").resolve())
    assert old["process"]["env"]["NODE_TRACE_DB"] == old["config"]["trace_db"]
    assert loaded.experiment["output_dir"] == str((tmp_path / "out").resolve())


def test_content_rubric_is_merged_by_question_name(tmp_path: Path):
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "questions:\n  - name: q1\n    text: Question?\n", encoding="utf-8",
    )
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        "questions:\n  - name: q1\n    text: Question?\n    evaluation:\n"
        "      must_have_points:\n        - id: mh1\n          description: Answer it\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "compare.yaml"
    cfg.write_text(
        """
version: 1
experiment: {baseline: old, candidate: new}
questions_file: ./questions.yaml
content_evaluation:
  enabled: true
  rubric_file: ./rubric.yaml
systems:
  old: {adapter: agenticsys_sse, config: {base_url: 'http://old', case_id: c1}}
  new: {adapter: agenticsys_sse, config: {base_url: 'http://new', case_id: c1}}
""",
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.content_evaluation["enabled"] is True
    assert loaded.questions[0].evaluation["must_have_points"][0]["id"] == "mh1"


def test_memory_requirement_is_normalized(tmp_path: Path):
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "questions:\n"
        "  - name: seed\n    text: Did payments return?\n"
        "    memory_required: false\n"
        "  - name: followup\n    text: When did those occur?\n"
        "    memory_required: true\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "compare.yaml"
    cfg.write_text(
        "version: 1\n"
        "experiment: {baseline: old, candidate: new, mode: stateful}\n"
        "questions_file: ./questions.yaml\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://old', case_id: c1}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://new', case_id: c1}}\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.questions[0].evaluation["memory_required"] is False
    assert loaded.questions[1].evaluation["memory_required"] is True


def test_shipped_simple_question_suite_is_fully_wired():
    """Every simple question must carry its oracle.

    The rubric merges onto the question set BY NAME, so a rename on either
    side unwires the check silently: the question still runs, the oracle just
    never fires, and the report shows a blank rather than a failure.
    """
    from agentic_eval.config import load_config

    config = load_config(
        Path(__file__).resolve().parents[2]
        / "experiments" / "configs" / "simple_questions.yaml"
    )
    known = {
        # Simple: settled by a script.
        "payment_returns_occurrence", "commercial_card_count",
        "commercial_card_balance", "external_delinquency_coverage",
        "latest_fico_score", "transactions_last_month",
        # Complex: settled by must-haves and logic expectations.
        "case_overview", "spending_spikes_drivers", "tsr_cdss_bureau_reaction",
        "tsr_cdss_before_after", "abnormal_transactions_during_reaction",
        # The interpretation chain: read the pattern, then argue both sides.
        "connected_transaction_pattern", "evidence_supporting_pattern",
        "evidence_contradicting_pattern",
    }
    names = [question.name for question in config.questions]
    # Questions get commented out to keep a calibration run cheap, so the
    # guard is on WIRING, not on which subset is enabled: an unknown name
    # means a rename that would leave its oracle attached to nothing.
    assert names, "the suite has no enabled questions"
    assert set(names) <= known, f"unknown question(s): {set(names) - known}"
    for question in config.questions:
        # Every question is scored by SOMETHING. A complex question has no
        # script to settle it, so its rubric is the wiring; a question with
        # neither runs and is never judged, which reads as a blank rather
        # than a failure.
        assert (
            question.evaluation.get("expected_answers")
            or question.evaluation.get("must_have_points")
        ), question.name
    # "these cards" only has a referent in warm mode.
    balance = next(
        (q for q in config.questions if q.name == "commercial_card_balance"), None,
    )
    if balance is not None:
        assert balance.evaluation["memory_required"] is True
        assert balance.evaluation["relation"]["parent"] == "commercial_card_count"


def test_orphan_rubric_entry_is_an_error(tmp_path: Path):
    """A rubric entry matching no question is a dead check that looks configured.

    Without this the suite runs, the expectation never fires, and the report
    shows a blank rather than a failure — which is how a full set of oracles
    sat inert while appearing to be wired up.
    """
    (tmp_path / "questions.yaml").write_text(
        "questions:\n  - name: asked\n    question: what happened?\n",
        encoding="utf-8",
    )
    (tmp_path / "rubric.yaml").write_text(
        "questions:\n"
        "  - name: asked\n    evaluation: {must_have_points: [{id: a, description: d}]}\n"
        "  - name: renamed_away\n    evaluation: {must_have_points: [{id: b, description: d}]}\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.yaml").write_text(
        "version: 1\n"
        "experiment:\n  name: t\n  baseline: a\n  candidate: b\n  mode: cold\n"
        "  repeats: 1\n  timeout_s: 10\n  seed: 1\n  output_dir: ./out\n"
        "questions_file: ./questions.yaml\n"
        "content_evaluation:\n  enabled: true\n  rubric_file: ./rubric.yaml\n"
        "systems:\n"
        "  a: {adapter: agenticsys_sse, config: {base_url: 'http://x', case_id: '1'}}\n"
        "  b: {adapter: agenticsys_sse, config: {base_url: 'http://y', case_id: '1'}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="renamed_away"):
        load_config(tmp_path / "eval.yaml")


def test_inline_evaluation_keeps_question_and_oracle_together():
    """The shipped suite carries its expectations in the question set itself."""
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "experiments" / "configs" / "simple_questions.yaml"
    )
    assert config.content_evaluation.get("rubric_file") is None
    for question in config.questions:
        assert (
            question.evaluation.get("expected_answers")
            or question.evaluation.get("must_have_points")
        ), question.name


def test_run_layout_places_every_artifact_in_a_named_folder():
    """One owner for the run folder shape.

    Four writers contribute to a run; when each picks its own filenames the
    result is a flat pile where nothing says which artifact came from where.
    """
    from agentic_eval.layout import RunLayout

    layout = RunLayout(Path("/runs/exp_1"))
    assert layout.manifest.name == "manifest.json"
    assert layout.runs.name == "runs.jsonl"
    # Aggregates, content, review, and logs each get a folder.
    assert layout.summary.parent.name == "metrics"
    assert layout.evaluations.parent.name == "content"
    assert layout.answer_review.parent.name == "review"
    assert layout.evidence_review_key.parent.name == "review"
    assert layout.server_log("current").parent.name == "logs"
    # No `content_` prefix inside content/: the folder already says it.
    assert layout.content_summary.name == "summary.json"
    assert layout.evaluations.name == "evaluations.jsonl"
    # Both review keys sit side by side; they join on turn_id.
    assert layout.answer_review_key.parent == layout.evidence_review_key.parent


def test_run_layout_is_found_from_any_artifact(tmp_path: Path):
    from agentic_eval.layout import RunLayout

    run = tmp_path / "exp_1"
    (run / "content").mkdir(parents=True)
    (run / "runs.jsonl").write_text("", encoding="utf-8")
    found = RunLayout.find(run / "content" / "evaluations.jsonl")
    assert found is not None and found.root == run
    assert RunLayout.find(tmp_path / "elsewhere.txt") is None


def test_every_follow_up_names_a_question_that_precedes_it():
    """A follow-up asked before its parent has no referent.

    The chain is asked in file order in one stateful session, so a parent
    listed after its child would be measured as a memory failure that is
    really a question-set ordering bug.
    """
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "experiments" / "configs" / "simple_questions.yaml"
    )
    seen: set[str] = set()
    for question in config.questions:
        relation = question.evaluation.get("relation") or {}
        parent = relation.get("parent")
        if parent:
            assert parent in seen, f"{question.name} follows {parent}, unasked"
            assert question.evaluation["memory_required"] is True, question.name
        seen.add(question.name)


def test_must_haves_carry_no_hand_set_weights():
    """Every must-have counts the same; importance carries the emphasis."""
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "experiments" / "configs" / "simple_questions.yaml"
    )
    for question in config.questions:
        for point in question.evaluation.get("must_have_points") or []:
            assert "weight" not in point, f"{question.name}:{point.get('id')}"


def test_config_command_resolves_an_interpreter_from_the_environment(monkeypatch):
    """A config must not pin one machine's Python.

    Every shipped config hardcoded `/Users/<someone>/.pyenv/.../bin/python`,
    which is correct on one laptop and a "command not found" everywhere else.
    """
    from agentic_eval.process import expand
    monkeypatch.delenv("AGENTIC_SYS_PYTHON", raising=False)
    assert expand("${AGENTIC_SYS_PYTHON:-python3}") == "python3"
    monkeypatch.setenv("AGENTIC_SYS_PYTHON", "/opt/venv/bin/python")
    assert expand("${AGENTIC_SYS_PYTHON:-python3}") == "/opt/venv/bin/python"


def test_an_unset_variable_with_no_fallback_is_an_error(monkeypatch):
    """Silently dropping it would fail later, further away, about the wrong thing."""
    import pytest
    from agentic_eval.process import expand
    monkeypatch.delenv("NO_SUCH_VAR", raising=False)
    with pytest.raises(KeyError, match="NO_SUCH_VAR"):
        expand("${NO_SUCH_VAR}")


def test_no_shipped_config_pins_a_machine_specific_path():
    import pathlib
    offenders = [
        str(p) for p in pathlib.Path("experiments").rglob("*.yaml")
        if "/Users/" in p.read_text() or "/home/" in p.read_text()
    ]
    assert not offenders, offenders


def test_safechain_adapts_to_the_one_call_the_judge_makes():
    """The private gateway is a transport swap, not a second code path.

    Both backends present `.chat.completions.create`, so `complete_json` is
    written against that and knows nothing else about either.
    """
    from types import SimpleNamespace
    from agentic_eval.llm_judge import SafeChainClient

    class FakeModel:
        def invoke(self, messages):
            assert messages[0][0] == "system" and messages[1][0] == "user"
            return SimpleNamespace(
                content='{"claims": []}',
                usage_metadata={"input_tokens": 11, "output_tokens": 2,
                                "total_tokens": 13},
            )

    reply = SafeChainClient(FakeModel()).chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response_format={"type": "json_object"},
    )
    assert reply.choices[0].message.content == '{"claims": []}'
    assert reply.usage.prompt_tokens == 11 and reply.usage.total_tokens == 13


def test_an_absent_safechain_says_which_environment_it_needs(monkeypatch):
    import pytest
    from agentic_eval.llm_judge import build_client
    monkeypatch.setitem(__import__("sys").modules, "safechain.lc_factory", None)
    with pytest.raises(RuntimeError, match="private environment"):
        build_client({"model": "gpt-4.1"}, "safechain", 60.0)


def test_an_unknown_backend_is_rejected_by_name():
    import pytest
    from agentic_eval.llm_judge import build_client
    with pytest.raises(ValueError, match="unknown judge backend"):
        build_client({}, "anthropic", 60.0)


def test_run_tools_is_gone():
    """It backed the adjudication tier, which was removed; nothing called it."""
    import agentic_eval.llm_judge as j
    assert not hasattr(j.OpenAIJudgeClient, "run_tools")
    assert "run_tools" not in pathlib.Path("agentic_eval/llm_judge.py").read_text()
