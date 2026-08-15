"""A denial decides the verdict only where it covers the claim.

Negation used to win outright whenever both polarities matched, for a real
reason: on "had NO returned payments" the affirmative pattern matches INSIDE
the sentence that denies it, so treating a double match as ambiguous made
every correct negative answer unscoreable.

It also swallowed answers that affirm one thing and deny another, and graded
them the opposite of what they said.
"""
from __future__ import annotations

from agentic_eval.content.oracles import _boolean_answer_check

_ITEM = {
    "affirmative_patterns": [
        r"\b(had|has|have|were|was|are|is)\b[^.]{0,60}\breturn(ed|s)?\b",
        r"\b\d[\d,]*\s+return(ed)?\s+payments?\b",
    ],
    "negative_patterns": [
        r"\b(no|zero|none|not any)\b[^.]{0,40}\breturn",
        r"\breturn(ed|s)?\b[^.]{0,40}\b(none|zero|0)\b",
    ],
}


def _check(truth, answer):
    return _boolean_answer_check(_ITEM, truth, " ".join(answer.split()).lower())


def test_a_count_stands_even_beside_a_denial_of_something_else():
    """The answer that was graded false while reporting 24 returns."""
    result = _check(True, """
        The customer had **24 returned payments** recorded, with returns spread
        across 12 separate dates. The largest returns clustered on three dates
        but overall concentration is moderate, with no extreme single-date
        return spikes.
        - No evidence of extreme single-date concentration in returned payments
    """)
    assert result["verdict"] == "pass"
    assert "outside every denial" in result["reason"]


def test_a_real_denial_still_wins():
    """The case negation-precedence existed for: the affirmative pattern
    matches inside the sentence doing the denying."""
    result = _check(False, "The customer had no returned payments in the period.")
    assert result["verdict"] == "pass"
    assert "negation took precedence" in result["reason"]


def test_a_denial_beside_an_unrelated_affirmative_is_not_overridden():
    """Only an affirmative about THIS claim should stand."""
    result = _check(False, "There were zero returned payments; 32 attempts were made.")
    assert result["verdict"] == "pass"


def test_an_answer_with_no_denial_is_affirmative():
    assert _check(True, "The customer had 1 returned payment.")["verdict"] == "pass"


def test_neither_polarity_is_reported_rather_than_guessed():
    result = _check(True, "The payment history was reviewed in full.")
    assert result["verdict"] == "unavailable"
