"""A refusal is an answer.

`q0_off_domain_rejection` asks something outside the case, and the RIGHT
behaviour is to decline. The adapter records that as `out_of_scope`, which was
treated as a failure everywhere it mattered: the content pipeline judged only
`ok`, so the oracle that checks "did it decline" never ran and the one
question whose correct answer is a refusal could not be scored at all.
"""
from __future__ import annotations

from agentic_eval.models import ANSWERED_OUTCOMES


def _record(outcome, answer="[rejected] That is outside this case."):
    return {
        "system": "previous", "mode": "stateful", "case_id": "c1",
        "question_set": "series_a", "name": "q0_off_domain_rejection",
        "run_index": 1, "outcome": outcome, "final_answer": answer,
        "elapsed_seconds": 1.0, "tools": [], "evidence": [], "metrics": {},
    }


def test_a_refusal_counts_as_an_answer():
    assert "out_of_scope" in ANSWERED_OUTCOMES
    assert "ok" in ANSWERED_OUTCOMES
    # A turn that produced nothing does not.
    for outcome in ("timeout", "error", "screen_timeout"):
        assert outcome not in ANSWERED_OUTCOMES


def test_a_refusal_is_judged_rather_than_skipped():
    """The bug: q0's oracle never ran, so the question was unscoreable."""
    records = [_record("out_of_scope"), _record("timeout", answer="")]
    eligible = [
        r for r in records
        if r.get("outcome") in ANSWERED_OUTCOMES and r.get("final_answer")
    ]
    assert [r["outcome"] for r in eligible] == ["out_of_scope"]


def test_completion_counts_a_refusal_as_complete():
    from agentic_eval.scoring import aggregate
    summary = aggregate([_record("out_of_scope")], modules=["latency"])
    assert summary["groups"][0]["completion_rate"] == 1.0


def test_a_refusal_is_not_red_in_the_progress_grid():
    from agentic_eval.render import progress
    rows = [dict(_record("out_of_scope"), system=s) for s in ("previous", "current")]
    plan = {"systems": 2, "repeats": 1,
            "question_order": ["q0_off_domain_rejection"]}
    text = progress.grid(rows, plan=plan, colourise=True)
    assert "\033[31m" not in text          # no red
    assert "✗" not in text                 # and no failure count
    assert "1/1" in text                   # counted as a completed repeat


def test_a_timeout_is_still_red():
    from agentic_eval.render import progress
    rows = [dict(_record("timeout"), system=s) for s in ("previous", "current")]
    plan = {"systems": 2, "repeats": 1,
            "question_order": ["q0_off_domain_rejection"]}
    text = progress.grid(rows, plan=plan, colourise=True)
    assert "\033[31m" in text and "✗2" in text


def test_a_refusal_is_not_an_error_on_the_progress_page():
    """The HTML has its own outcome rendering, and it had its own copy of "ok".

    Fixing the terminal grid alone left the page still colouring a correct
    refusal red and counting it against every row.
    """
    from agentic_eval.render import progress
    rows = [dict(_record("out_of_scope"), system=s, case_id="c1")
            for s in ("previous", "current")]
    plan = {"questions": 1, "cases": 1, "repeats": 1, "systems": 2,
            "expected_records": 2, "question_order": ["q0_off_domain_rejection"]}
    html = progress.render(progress.summarize(rows, plan=plan))

    # The headline lists it as good, not bad.
    assert 'class="good">out_of_scope' in html
    assert 'class="bad">out_of_scope' not in html
    # And every row counts it as answered.
    assert "2/2 answered" in html
    assert "0/2 answered" not in html


def test_a_timeout_is_still_an_error_on_the_page():
    from agentic_eval.render import progress
    rows = [dict(_record("timeout"), system=s, case_id="c1")
            for s in ("previous", "current")]
    plan = {"questions": 1, "cases": 1, "repeats": 1, "systems": 2,
            "expected_records": 2}
    html = progress.render(progress.summarize(rows, plan=plan))
    assert 'class="bad">timeout' in html
    assert "0/2 answered" in html
