"""Readings of a figure that mean the same thing — and ones that do not.

Every case here is taken from a real run where the evaluator called a correct
figure wrong, or from the inverse it must keep catching.
"""
from __future__ import annotations

import pytest

from agentic_eval.content.numeric import (
    _satisfies, _written_tolerance, comparison_variants, infer_comparator,
)


def _agrees(written, expected, computed, tolerance=None):
    """Exactly what `verify` does: the answer's own precision is the window.

    Tests used to hand-pick a tolerance, and a value like 0.5 for a percentage
    claim is one production never computes — big enough to swallow the very
    disagreements the case existed to catch.
    """
    if tolerance is None:
        tolerance = max(_written_tolerance(written), 1e-12)
    comparator = infer_comparator(written, "==")
    return _satisfies(comparator, expected, computed, tolerance) or any(
        _satisfies(comparator, want, got, tol)
        for want, got, tol in comparison_variants(
            written, expected, computed, tolerance,
        )
    )


@pytest.mark.parametrize("written,expected,computed", [
    # SCALE. The judge reports "36%" as 36.0, the tool as 0.3603. Only one
    # direction of the conversion was tried, so which way round they landed
    # decided whether a correct figure was called wrong.
    ("36%", 36.0, 0.3603319502074689),
    ("36%", 0.36, 0.3603319502074689),
    # SIGN. "declined by 2.2%" is a magnitude; the tool reports -0.022.
    ("2.2%", 2.2, -0.022),
    ("96.6%", 96.6, -0.966),
    # The sign is in the text but the judge dropped it from the value.
    ("-2.2%", 2.2, -0.022),
    ("declined 2.2%", 2.2, -0.022),
    # BOUND. "all above 720" is satisfied by 721; read as `==` it was a
    # mismatch against the very evidence that proves it.
    ("all above 720", 720.0, 721.0),
    ("below 700", 700.0, 690.0),
    ("at least 10", 10.0, 10.0),
])
def test_the_same_figure_written_differently_agrees(written, expected, computed):
    assert _agrees(written, expected, computed)


@pytest.mark.parametrize("written,expected,computed,why", [
    ("38%", 38.0, 0.3603319502074689, "a genuine 2pp gap"),
    ("38%", 0.38, 0.3603319502074689, "the same gap, other scale"),
    ("+5%", 5.0, -0.05, "the answer got the direction wrong"),
    ("rose 2.2%", 2.2, -0.022, "it fell"),
    ("all above 720", 720.0, 719.0, "the bound is violated"),
    ("below 700", 700.0, 710.0, "the bound is violated"),
])
def test_a_real_disagreement_survives_every_variant(
    written, expected, computed, why,
):
    """The variants must not become a way to pass anything."""
    assert not _agrees(written, expected, computed), why


def test_a_bare_number_is_still_an_equality():
    assert infer_comparator("26.4", "==") == "=="
    # An explicit comparator from the judge is never overridden.
    assert infer_comparator("above 720", "<=") == "<="


@pytest.mark.parametrize("written,measures", [
    ("mid-2024", "period of spike for TSR and CDSS"),
    ("2025-02 to 2025-05", "period TSR exceeded threshold"),
    ("May 2025", None),
    ("Q2 2024", None),
    ("early 2025", None),
    ("2024-2025", None),
    ("2025-06", None),
])
def test_a_period_is_not_a_quantity(written, measures):
    """The judge lists these among the numbers and the parser mangles them.

    "mid-2024" became -2024.0, then failed to locate, then counted against the
    answer as an unsupported figure. Whether the period is right is a fair
    question — the NUMERIC trace is just the wrong instrument for it.
    """
    from agentic_eval.content.numeric import is_period_expression

    assert is_period_expression(written, measures)


@pytest.mark.parametrize("written,measures", [
    ("26.4", "TSR peak"),
    ("20", "risky threshold for TSR"),          # a threshold, not a year
    ("17", "consecutive months below 681"),     # a count of months
    ("$1,200,700", "total spend"),
    ("36%", "share of spend"),
    ("721", "minimum FICO"),
])
def test_a_real_figure_is_still_traced(written, measures):
    """The exclusion must not swallow quantities that merely mention time."""
    from agentic_eval.content.numeric import is_period_expression

    assert not is_period_expression(written, measures)


def test_a_period_mention_is_excluded_from_scoring_not_charged():
    from agentic_eval.content.verify import JUDGE_ERROR_FAILURES, SYSTEM_FAILURES

    assert "not_a_quantity" in JUDGE_ERROR_FAILURES
    assert "not_a_quantity" not in SYSTEM_FAILURES


@pytest.mark.parametrize("measures", [
    "risky threshold for TSR",
    "high-risk threshold for Paydex score",
    "policy limit for exposure",
    "target utilisation",
])
def test_a_constant_the_answer_supplied_is_not_traced(measures):
    """"exceeding the risky threshold of 20" states a benchmark, not a reading.

    Tool output holds measurements, not the thresholds they are judged
    against, so the numeric trace can never locate one — and reported that as
    the answer inventing a figure.
    """
    from agentic_eval.content.numeric import is_stated_constant

    assert is_stated_constant(measures)


@pytest.mark.parametrize("measures", [
    "TSR peak", "average FICO score for 2025", "total spend",
    "count of returned payments",
])
def test_a_measurement_is_still_traced(measures):
    from agentic_eval.content.numeric import is_stated_constant

    assert not is_stated_constant(measures)


def test_a_count_of_months_is_not_a_period():
    """"19" measured as "period of spending" is a COUNT, not a date.

    The prose signal alone excluded it; the written form has to look date-ish
    before the description is trusted.
    """
    from agentic_eval.content.numeric import is_period_expression

    assert not is_period_expression("19", "period of spending")
    assert is_period_expression("2024-06", "period of spending")


def test_stated_constants_are_excluded_from_scoring():
    from agentic_eval.content.verify import JUDGE_ERROR_FAILURES, SYSTEM_FAILURES

    assert "stated_constant" in JUDGE_ERROR_FAILURES
    assert "stated_constant" not in SYSTEM_FAILURES


@pytest.mark.parametrize("written,expected,computed,agree,why", [
    # `_written_tolerance("32%")` is already on the FRACTION scale (0.005).
    # Dividing it again when converting the value made the window 0.00005 and
    # called a 0.3pp difference a disagreement.
    ("32%", 32.0, 0.317, True, "0.3pp apart, inside the written precision"),
    ("14%", 14.0, 0.13722784400301974, True, "0.28pp apart"),
    ("36%", 36.0, 0.3603319502074689, True, "0.03pp apart"),
    ("38%", 38.0, 0.3603319502074689, False, "2pp — a real disagreement"),
    ("50%", 50.0, 0.62, False, "12pp — a real disagreement"),
])
def test_percent_tolerance_is_not_scaled_twice(
    written, expected, computed, agree, why,
):
    assert _agrees(written, expected, computed) is agree, why
