"""Re-asking a turn the system never answered, without inventing evidence."""
from __future__ import annotations

import pytest

from agentic_eval.runner import RetryPolicy


def _record(outcome, **extra):
    return {"outcome": outcome, **extra}


def test_only_no_answer_outcomes_are_retried():
    """A timeout is the machine being busy; everything else is the system."""
    policy = RetryPolicy.from_config({"attempts": 2})
    assert policy.triggered_by([_record("timeout")]) == ["timeout"]
    assert policy.triggered_by([_record("screen_timeout")]) == ["screen_timeout"]
    # These ARE the system's behaviour and must be recorded as they happened.
    for outcome in ("ok", "error", "out_of_scope", "qa_cache_hit"):
        assert policy.triggered_by([_record(outcome)]) == []


def test_the_retryable_set_is_configurable():
    policy = RetryPolicy.from_config({"outcomes": ["gateway_busy"], "attempts": 1})
    assert policy.triggered_by([_record("gateway_busy")]) == ["gateway_busy"]
    assert policy.triggered_by([_record("timeout")]) == []


def test_absent_config_disables_it():
    """Retrying changes what a run measures, so it is opt-in."""
    assert RetryPolicy.from_config(None).attempts == 0
    assert RetryPolicy.from_config({}).attempts == 0


def test_a_misspelled_key_is_refused():
    """A silently ignored `attempt:` would read as configured and do nothing."""
    with pytest.raises(ValueError, match="unknown key"):
        RetryPolicy.from_config({"attempt": 3})
    with pytest.raises(ValueError, match="must be a mapping"):
        RetryPolicy.from_config(["timeout"])


def test_any_timed_out_turn_triggers_the_replay_not_just_the_last():
    """The pass is the unit: one dead turn compromises the conversation.

    In stateful mode recovery starts with /rewind, which clears the case — so
    re-asking only the failed question would put a follow-up in an empty
    session, where it answers something vague that every metric then charges
    to the system.
    """
    policy = RetryPolicy.from_config({"attempts": 2})
    passing = [_record("ok"), _record("timeout"), _record("ok")]
    assert policy.triggered_by(passing) == ["timeout"]


def test_backoff_and_attempts_are_clamped_not_negated():
    policy = RetryPolicy.from_config({"attempts": -5, "backoff_s": -1})
    assert policy.attempts == 0 and policy.backoff_s == 0.0
