"""Named scopes and question selection, for cheap smoke runs."""
from __future__ import annotations

import argparse

import pytest

from agentic_eval.cli import (
    _apply_overrides, _apply_question_scope, _apply_scope, _select_questions,
)
from agentic_eval.config import load_config


def _config(tmp_path, scopes: str = "") -> str:
    (tmp_path / "series_a.yaml").write_text(
        "questions:\n"
        "  - {name: a1, question: 'standalone'}\n"
        "  - {name: a2, question: 'how many cards?'}\n"
        "  - name: a3\n"
        "    question: 'balance of these cards?'\n"
        "    relation: {type: follow_up, parent: a2}\n",
        encoding="utf-8",
    )
    (tmp_path / "series_b.yaml").write_text(
        "questions:\n"
        "  - {name: b2, question: 'the spike?'}\n"
        "  - name: b3\n"
        "    question: 'during the reacting period?'\n"
        "    relation: {type: follow_up, parent: b2}\n"
        "  - name: b4\n"
        "    question: 'and the transactions?'\n"
        "    relation: {type: follow_up, parent: b3}\n",
        encoding="utf-8",
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\n"
        "experiment:\n"
        f"  baseline: old\n  candidate: new\n  repeats: 3\n{scopes}"
        "questions_file: [series_a.yaml, series_b.yaml]\n"
        "systems:\n"
        "  old: {adapter: agenticsys_sse, config: {base_url: 'http://o:1', case_id: c1}}\n"
        "  new: {adapter: agenticsys_sse, config: {base_url: 'http://n:2', case_id: c1}}\n",
        encoding="utf-8",
    )
    return str(path)


_SMOKE = (
    "  scopes:\n"
    "    smoke:\n"
    "      questions: [a1, b2, b3]\n"
    "      repeats: 2\n"
)


def test_a_scope_narrows_questions_and_repeats(tmp_path):
    config = load_config(_config(tmp_path, _SMOKE))
    _apply_scope(config, "smoke")
    assert [q.name for q in config.questions] == ["a1", "b2", "b3"]
    assert config.experiment["repeats"] == 2


def test_a_scope_keeps_the_question_sets_it_spans(tmp_path):
    """The smoke run must still exercise the per-set tables."""
    config = load_config(_config(tmp_path, _SMOKE))
    _apply_scope(config, "smoke")
    assert {q.question_set for q in config.questions} == {"series_a", "series_b"}


def test_selecting_a_follow_up_without_its_parent_is_refused(tmp_path):
    """A follow-up with no referent measures the selection, not the system."""
    config = load_config(_config(tmp_path))
    with pytest.raises(ValueError, match="without the turn they refer to"):
        _select_questions(config, {"a3"})


def test_the_whole_parent_chain_is_walked(tmp_path):
    """b4 needs b3, and b3 needs b2 — both must be named, not just the first."""
    config = load_config(_config(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        _select_questions(config, {"b4"})
    message = str(excinfo.value)
    assert "b4 needs b3" in message and "b3 needs b2" in message
    assert "--question b2" in message and "--question b3" in message


def test_a_complete_chain_is_allowed(tmp_path):
    config = load_config(_config(tmp_path))
    _select_questions(config, {"b2", "b3", "b4"})
    assert [q.name for q in config.questions] == ["b2", "b3", "b4"]


def test_a_standalone_question_needs_no_parent(tmp_path):
    config = load_config(_config(tmp_path))
    _select_questions(config, {"a1"})
    assert [q.name for q in config.questions] == ["a1"]


def test_an_unknown_scope_names_the_ones_that_exist(tmp_path):
    config = load_config(_config(tmp_path, _SMOKE))
    with pytest.raises(ValueError, match=r"unknown scope 'nope'.*\['smoke'\]"):
        _apply_scope(config, "nope")


def _args(**kwargs):
    return argparse.Namespace(**{
        "scope": None, "question_scope": None, "case_ids": None,
        "cases_from": None, "mode": None,
        "workers": None, "repeats": None, "questions": None,
        "eval_modules": None, "baseline_cwd": None, "candidate_cwd": None,
        **kwargs,
    })


def test_an_explicit_flag_beats_the_scope(tmp_path):
    config = load_config(_config(tmp_path, _SMOKE))
    _apply_overrides(config, _args(scope="smoke", repeats=1))
    assert config.experiment["repeats"] == 1
    assert [q.name for q in config.questions] == ["a1", "b2", "b3"]


def test_expected_repeats_follows_an_override(tmp_path):
    """A scoped run must not declare itself a partial evaluation.

    `expected_repeats` is captured at load time from the config's k. Left
    behind, every `--scope`/`--repeats` run reported
    `repetitions_complete: false` while being perfectly complete.
    """
    config = load_config(_config(tmp_path, _SMOKE))
    assert config.content_evaluation["expected_repeats"] == 3
    _apply_overrides(config, _args(scope="smoke"))
    assert config.content_evaluation["expected_repeats"] == 2


_QSCOPES = (
    "  question_scopes:\n"
    "    just_b: [b2, b3]\n"
    "    spelled_out:\n"
    "      questions: [a1]\n"
)


def test_a_question_scope_leaves_k_to_the_config(tmp_path):
    """The whole point of the distinction: rates rest on the config's k.

    A run scope pins k because a smoke run is a check, not a measurement. A
    question scope must not, or it hands back rates that look like the
    config's and are not.
    """
    config = load_config(_config(tmp_path, _QSCOPES))
    _apply_question_scope(config, "just_b")
    assert [q.name for q in config.questions] == ["b2", "b3"]
    assert config.experiment["repeats"] == 3          # untouched
    assert config.experiment["cases"] == ["c1"]       # untouched


def test_a_question_scope_accepts_a_bare_list_or_a_mapping(tmp_path):
    config = load_config(_config(tmp_path, _QSCOPES))
    _apply_question_scope(config, "spelled_out")
    assert [q.name for q in config.questions] == ["a1"]


def test_a_question_scope_pinning_k_is_refused(tmp_path):
    """Honouring it silently would change what every rate rests on."""
    scopes = (
        "  question_scopes:\n"
        "    sneaky:\n"
        "      questions: [a1]\n"
        "      repeats: 1\n"
    )
    config = load_config(_config(tmp_path, scopes))
    with pytest.raises(ValueError, match="selects questions only"):
        _apply_question_scope(config, "sneaky")


def test_a_question_scope_pinning_cases_is_refused(tmp_path):
    scopes = (
        "  question_scopes:\n"
        "    sneaky:\n"
        "      questions: [a1]\n"
        "      cases: [c9]\n"
    )
    config = load_config(_config(tmp_path, scopes))
    with pytest.raises(ValueError, match="selects questions only"):
        _apply_question_scope(config, "sneaky")


def test_a_question_scope_still_checks_the_parent_chain(tmp_path):
    scopes = (
        "  question_scopes:\n"
        "    broken: [b4]\n"
    )
    config = load_config(_config(tmp_path, scopes))
    with pytest.raises(ValueError, match="without the turn they refer to"):
        _apply_question_scope(config, "broken")


def test_an_unknown_key_in_a_run_scope_is_refused(tmp_path):
    scopes = (
        "  scopes:\n"
        "    typo:\n"
        "      question: [a1]\n"
    )
    config = load_config(_config(tmp_path, scopes))
    with pytest.raises(ValueError, match="unknown key"):
        _apply_scope(config, "typo")


def test_the_two_scope_flags_are_alternatives(tmp_path):
    """A run scope already pins the questions; combining them is ambiguous."""
    config = load_config(_config(tmp_path, _SMOKE + _QSCOPES))
    with pytest.raises(ValueError, match="alternatives"):
        _apply_overrides(config, _args(scope="smoke", question_scope="just_b"))


def test_a_question_scope_does_not_disturb_expected_repeats(tmp_path):
    config = load_config(_config(tmp_path, _QSCOPES))
    _apply_overrides(config, _args(question_scope="just_b"))
    assert config.experiment["repeats"] == 3
    assert config.content_evaluation["expected_repeats"] == 3


def test_completeness_counts_the_repeats_the_runs_file_holds():
    """Scoring is a separate command; its config need not match the run.

    `bin/compare` calls `evaluate-content --config <cfg>` without the scope
    flags, so a `--scope smoke` run at k=2 was scored against a config still
    saying k=3 — and every group in a complete evaluation reported
    `repetitions_complete: false`.
    """
    from agentic_eval.content.pipeline import _repeats_in

    records = [
        {"run_index": index, "name": "q1", "system": system}
        for index in (1, 2) for system in ("old", "new")
    ]
    assert _repeats_in(records) == 2
    # No run_index at all (an older or hand-built file): fall back, don't zero.
    assert _repeats_in([{"name": "q1"}]) is None
