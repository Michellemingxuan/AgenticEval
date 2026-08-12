"""Who did the fixing: the system itself, or us.

SELF-RECOVERY is the system correcting a problem without being asked again,
at two levels — a re-issued tool call, or the whole plan re-run for the same
turn. Both are the design working, so neither is a fault; they are split
because they cost differently and say different things about where the happy
path broke.

An EVALUATOR REPLAY is the other thing entirely: the system produced nothing,
and the harness asked again. Summed together as one `retry_count`, a system
with a busy safety net scored the same as one that answered nothing twice.
"""
from __future__ import annotations

from agentic_eval.dimensions.latency import section
from agentic_eval.models import AdapterResult, RECORD_SCHEMA, RunRequest
from agentic_eval.render.page import _delta_cell


def _request():
    question = type("Q", (), {"name": "q", "text": "q?", "evaluation": {},
                              "question_set": "s"})()
    return RunRequest(question, 1, "stateful", 1, case_id="c1")


def _record(**fields):
    base = {
        "outcome": "ok", "elapsed_seconds": 1.0, "system": "s",
        "name": "q", "run_index": 1, "tools": [], "evidence": [],
    }
    return {**base, **fields}


def test_the_two_kinds_are_reported_separately():
    rows = [
        _record(self_recovered_tool=True, self_recovered_orchestration=False,
                self_recovery_tool_count=3, self_recovery_orchestration_count=0,
                self_recovery_count=3),
        _record(self_recovered_tool=True, self_recovered_orchestration=False,
                self_recovery_tool_count=1, self_recovery_orchestration_count=0,
                self_recovery_count=1),
    ]
    out = section(rows)
    assert out["self_recovery_tool_rate"] == 1.0        # every run used the mechanism
    assert out["self_recovery_orchestration_rate"] == 0.0    # none restarted
    # One total, not one per level: the rates already say WHICH fired.
    assert out["self_recovery_attempts"] == 4


def test_the_two_levels_are_distinguishable():
    """The distinction the old single rate could not show."""
    safety_net = [_record(self_recovered_tool=True, self_recovered_orchestration=False,
                          self_recovery_tool_count=2, self_recovery_orchestration_count=0)]
    restarts = [_record(self_recovered_tool=False, self_recovered_orchestration=True,
                        self_recovery_tool_count=0, self_recovery_orchestration_count=2)]
    # Identical under the old metric: both have retry_count 2, retried True.
    assert section(safety_net)["self_recovery_orchestration_rate"] == 0.0
    assert section(restarts)["self_recovery_orchestration_rate"] == 1.0
    assert section(safety_net)["self_recovery_tool_rate"] == 1.0
    assert section(restarts)["self_recovery_tool_rate"] == 0.0


def test_runs_recorded_before_the_split_read_as_unmeasured():
    """Absent must not average as zero: it would claim a clean record."""
    rows = [_record(retried=True, retry_count=2)]      # no split fields
    out = section(rows)
    assert out["self_recovery_tool_rate"] is None
    assert out["self_recovery_orchestration_rate"] is None
    assert out["self_recovery_rate"] == 1.0                    # the old one still reads


def test_the_record_schema_was_bumped_so_old_runs_are_flagged():
    """The page warns on records older than the current schema."""
    assert RECORD_SCHEMA >= 5


def test_the_adapter_result_splits_and_still_sums():
    result = AdapterResult(
        outcome="ok", self_recovery_tool_count=3,
        self_recovery_orchestration_count=1, self_recovery_count=4,
    )
    record = result.to_record(system="s", request=_request())
    assert record["self_recovery_tool_count"] == 3 and record["self_recovered_tool"] is True
    assert record["self_recovery_orchestration_count"] == 1 and record["self_recovered_orchestration"] is True
    assert record["self_recovery_count"] == 4 and record["self_recovered"] is True


def test_an_absent_count_stays_none_rather_than_false():
    record = AdapterResult(outcome="ok").to_record(
        system="s", request=_request(),
    )
    assert record["self_recovered_tool"] is None
    assert record["self_recovered_orchestration"] is None


def test_a_neutral_metric_delta_is_not_painted_as_a_regression():
    """`higher_is_better=None` means the direction carries no verdict.

    `(change > 0) == None` is always False, so without an explicit branch every
    tool-retry delta would render red — editorialising a mechanism working.
    """
    neutral = _delta_cell(0.2, 0.5, None, "pct")
    assert "up" not in neutral and "down" not in neutral
    assert "+30%" in neutral
    assert "down" in _delta_cell(0.2, 0.5, False, "pct")
    assert "up" in _delta_cell(0.2, 0.5, True, "pct")


def test_self_recovery_rolls_up_both_levels():
    """One headline number, so a reader need not add two rows in their head."""
    rows = [
        _record(self_recovered=True, self_recovered_tool=True,
                self_recovered_orchestration=False),
        _record(self_recovered=True, self_recovered_tool=False,
                self_recovered_orchestration=True),
        _record(self_recovered=False, self_recovered_tool=False,
                self_recovered_orchestration=False),
    ]
    out = section(rows)
    assert out["self_recovery_rate"] == 2 / 3
    assert out["self_recovery_tool_rate"] == 1 / 3
    assert out["self_recovery_orchestration_rate"] == 1 / 3


def test_the_rollup_is_set_from_either_level():
    result = AdapterResult(
        outcome="ok", self_recovery_tool_count=0,
        self_recovery_orchestration_count=2,
    )
    assert result.to_record(system="s", request=_request())["self_recovered"] is True

    quiet = AdapterResult(
        outcome="ok", self_recovery_tool_count=0,
        self_recovery_orchestration_count=0,
    )
    assert quiet.to_record(system="s", request=_request())["self_recovered"] is False

    unmeasured = AdapterResult(outcome="ok")
    assert unmeasured.to_record(system="s", request=_request())["self_recovered"] is None


def test_an_evaluator_replay_is_never_counted_as_self_recovery():
    """The system answered nothing; crediting it with recovering would invert
    the finding."""
    rows = [_record(evaluator_replayed=True, self_recovered=False,
                    self_recovered_tool=False, self_recovered_orchestration=False)]
    out = section(rows)
    assert out["evaluator_replay_rate"] == 1.0
    assert out["self_recovery_rate"] == 0.0


def test_the_total_is_derived_when_only_the_levels_are_given():
    """An adapter that fills the levels but not the total must still be read.

    The alternative reading — "never measured" — is the one that looks like a
    clean record and is not.
    """
    result = AdapterResult(
        outcome="ok", self_recovery_tool_count=2,
        self_recovery_orchestration_count=1,
    )
    record = result.to_record(system="s", request=_request())
    assert record["self_recovery_count"] == 3
    assert record["self_recovered"] is True


def test_the_old_field_names_are_read_under_the_new_ones():
    """The rename is a rename: runs recorded before it still score.

    `retried` was always `retry_count > 0` over tool plus orchestration —
    exactly what self-recovery means — so no re-run is needed to read them.
    """
    rows = [_record(retried=True, retry_count=2), _record(retried=False, retry_count=0)]
    out = section(rows)
    assert out["self_recovery_rate"] == 0.5
    assert out["self_recovery_attempts"] == 2
    # But the split it never captured stays unmeasured rather than zero.
    assert out["self_recovery_tool_rate"] is None
    assert out["self_recovery_orchestration_rate"] is None


def test_completion_leads_the_system_block():
    """Effort only means something once you know an answer arrived."""
    from agentic_eval.render.page import _MODULE_METRICS
    spec = [key for key, *_ in _MODULE_METRICS["latency"]]
    assert spec[0] == "completion_rate"
    assert spec.index("self_recovery_rate") < spec.index("evaluator_replay_rate")
    assert spec.index("evaluator_replay_rate") < spec.index("tool_call_success_rate")


def test_an_aggregate_rate_names_what_it_averaged_over():
    """"100% (3 questions)" — the denominator is the unit.

    A bare percentage beside a per-call rate hides that one is a mean over
    questions and the other a share of calls.
    """
    from agentic_eval.render import answer_comparison_html
    rows = []
    for question in ("q1", "q2", "q3"):
        for system in ("old", "new"):
            rows.append({
                "system": system, "mode": "cold", "name": question, "run_index": 1,
                "question": question, "answer": "a", "claims": [], "fact_results": [],
                "metrics": {},
            })
    page = answer_comparison_html(
        rows, baseline="old", candidate="new",
        summary={"groups": [
            {"system": s, "mode": "cold", "name": q, "completion_rate": 1.0}
            for q in ("q1", "q2", "q3") for s in ("old", "new")
        ]},
    )
    assert "100% (3 questions)" in page


def test_every_system_metric_key_exists_in_the_module_that_computes_it():
    """A typo renders a permanently blank row rather than failing.

    `evaluator_replays_rate` for `evaluator_replay_rate` cost nothing at import
    and would have shown "—" forever, which reads as "never happened".
    """
    from agentic_eval.dimensions.latency import section
    from agentic_eval.render.page import _MODULE_METRICS

    produced = set(section([{
        "outcome": "ok", "elapsed_seconds": 1.0, "tools": [], "evidence": [],
    }]))
    produced |= {"completion_rate"}          # added by scoring.aggregate
    for key, *_ in _MODULE_METRICS["latency"]:
        root = key.split(".")[0]
        assert root in produced, f"{key} is not produced by latency.section"


def test_a_delta_keeps_one_decimal_only_when_it_says_something():
    """Rounding to whole units invented a difference that was not there.

    A mean of 7 -> 7.6 rendered "+1", which reads as a whole extra LLM call
    per answer when the gap is 0.6.
    """
    from agentic_eval.render.page import _delta_cell
    import re
    plain = lambda cell: re.sub(r"<[^>]+>", "", cell)

    assert plain(_delta_cell(7.0, 7.6, False, "num")) == "+0.6"
    assert plain(_delta_cell(0.5757575, 1.0, True, "pct")) == "+42.4%"
    # A whole number keeps no pointless ".0", and thousands stay grouped.
    assert plain(_delta_cell(0.5, 1.0, True, "pct")) == "+50%"
    assert plain(_delta_cell(14, 0, False, "count")) == "-14"
    # Past three digits the decimal is noise, so it is dropped.
    assert plain(_delta_cell(38601, 87605.3, False, "num")) == "+49,004"
    assert plain(_delta_cell(11.3, 15.5, False, "sec")) == "+4.2"
