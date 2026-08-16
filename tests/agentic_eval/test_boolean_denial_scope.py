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


def test_a_count_of_zero_denies_rather_than_affirms():
    """The affirmative pattern counts the thing — and a digit run includes 0.

    Measured on a correct answer whose every line said no returns: "Report
    confirms **0 returned payments**" matched the affirmative pattern, sat in
    no denial's span, and so stood as a claim that returns existed.
    """
    result = _check(False, """
        No payment returns were recorded for this customer - all payment
        attempts cleared successfully.
        - **0 payment returns** observed out of 357 total payment attempts
        - **Report confirms 0 returned payments** and a returned total amount
          of $0, with all return dates and reasons marked as "N/A"
        - The absence of returned payments is confirmed
    """)
    assert result["verdict"] == "pass"


def test_a_zero_count_alone_states_false():
    """Reclassified as a denial, not discarded: an answer whose only match is
    the zero count states False rather than stating nothing."""
    assert _check(False, "Report confirms 0 returned payments.")["verdict"] == "pass"
    assert _check(True, "Report confirms 0 returned payments.")["verdict"] == "fail"


def test_a_column_name_is_not_a_claim_that_returns_exist():
    """Both systems describe the query they ran, and the affirmative pattern is
    loose proximity — "<verb> ... return" — so it fires on the schema.

    "**0 returned payments** were found (Return Flag = 1 in payments table)"
    matched "were found (return", the tail of a filter SPEC. It names no
    quantity, so the zero rule left it; it sits outside the zero's span, so
    containment left it. A description of HOW the system looked became a claim
    about what it found.
    """
    result = _check(False, """
        No prior curated reports - answer is from live specialist analysis
        only. The customer had **no payment returns** - all payment attempts
        cleared successfully.
        - **0 returned payments** were found (Return Flag = 1 in payments table)
        - All present payment rows show **Return Flag = 0** (successful)
        - Payment return status checked across **all payment records**
    """)
    assert result["verdict"] == "pass"


def test_a_real_count_beside_the_same_column_name_still_affirms():
    """The rule must not swallow the answer that gets the true case right."""
    result = _check(True, "The customer had **1 returned payment** of "
                          "$105,818.60 (Return Flag = 1) in the period.")
    assert result["verdict"] == "pass"


def test_only_the_affirmative_side_is_read_as_schema():
    """"all rows show Return Flag = 0" is a real finding, and reading it as a
    column name would leave a correct negative answer with no denial at all."""
    from agentic_eval.content.oracles import _names_a_column
    import re

    answer = "all present payment rows show return flag = 0"
    match = re.search(r"\breturn\b", answer)
    assert _names_a_column(answer, match)          # it IS a column reference
    assert _check(False, answer)["verdict"] == "pass"   # still scores as denial


def test_only_a_span_that_is_zero_throughout_reads_as_a_denial():
    """A span carrying any non-zero figure is still counting something.

    Checked on the rule itself rather than through a rubric, because the
    payment rubric's own negative pattern claims any "0" within 40 characters
    of "return" — so an end-to-end case could not tell the two apart.
    """
    from agentic_eval.content.oracles import _quantifies_zero

    assert _quantifies_zero("0 returned payments")
    assert _quantifies_zero("had 0.0 returned payments")
    assert _quantifies_zero("zero returned payments")
    assert not _quantifies_zero("24 returned payments across 0 disputes")
    assert not _quantifies_zero("had 1 returned payment")
    assert not _quantifies_zero("had returned payments")


def test_a_word_negation_inside_an_affirmative_is_not_read_as_zero():
    """"no"/"none" are not zero counts here.

    The off-domain probe affirms on "NOT relevant to this case" — reading that
    as a zero quantity would invert the one question whose oracle is a
    constant, and every correct refusal would score as compliance.
    """
    item = {
        "affirmative_patterns": [
            r"\b(no|not)\b[^.]{0,40}\b(relevant|applicable)\b[^.]{0,40}\bcase\b",
        ],
        "negative_patterns": [r"\b(here are|options include)\b"],
    }
    result = _boolean_answer_check(
        item, True, "that is not relevant to this case, which covers payments.")
    assert result["verdict"] == "pass"


def test_a_decimal_starting_in_zero_is_not_a_zero_count():
    """"peaked at 0.52" is a real figure, not an absence."""
    item = {
        "affirmative_patterns": [r"\bpeaked at\b[^.]{0,20}\b[\d.]+\b"],
        "negative_patterns": [r"\bno flagged delinquency\b"],
    }
    result = _boolean_answer_check(
        item, True,
        "the external delinquency index peaked at 0.52. no flagged delinquency.")
    assert result["verdict"] == "pass"
