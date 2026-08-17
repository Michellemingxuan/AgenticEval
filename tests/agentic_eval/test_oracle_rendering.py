"""An oracle that could not run is not a failure.

`unavailable` drew the same red ✗ as `fail`, so a failed oracle COMMAND —
no ground truth computed, nothing compared — read as the system getting the
answer wrong. The metric already excluded it from both sides of the accuracy
rate, so the page and the number told different stories.
"""
from __future__ import annotations

# --- an oracle that could not run is not a failure ---------------------------

def _oracle_row(verdict, expected, reason=""):
    from agentic_eval.render.page import _expectations_block
    return _expectations_block({"expected_answer_results": [{
        "expected_answer_id": "commercial_card_count", "verdict": verdict,
        "expected": expected, "reason": reason,
    }]})


def test_an_oracle_that_could_not_run_is_not_marked_wrong():
    """`unavailable` drew the same red ✗ as `fail`.

    It is excluded from the accuracy rate on BOTH sides, so the page said the
    system got two answers wrong while the metric counted neither — and the
    real cause, a failed oracle command, was invisible.
    """
    html = _oracle_row("unavailable", None,
                       "Oracle command exited 2: required: --case")

    assert "✗" not in html and "✓" not in html
    assert 'class="skip"' in html
    assert "not computed" in html          # expected column
    assert "not checked" in html           # in-answer column
    # The cause is on the row, for hovering.
    assert "required: --case" in html


def test_a_real_failure_still_reads_as_one():
    html = _oracle_row("fail", 2,
                       "No material number in the answer equals 2 (tolerance 0).")

    assert "✗" in html
    assert 'class="skip"' not in html
    assert "no such figure in the answer" in html


def test_a_pass_is_unchanged():
    html = _oracle_row("pass", 2, "2 matches the computed answer.")

    assert "✓" in html and "2" in html
