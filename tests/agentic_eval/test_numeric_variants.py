"""Readings of a figure that mean the same thing — and ones that do not.

Every case here is taken from a real run where the evaluator called a correct
figure wrong, or from the inverse it must keep catching.
"""
from __future__ import annotations

import pytest

from agentic_eval.content.numeric import (
    _satisfies, comparison_variants, infer_comparator,
)


def _agrees(written, expected, computed, tolerance):
    comparator = infer_comparator(written, "==")
    return _satisfies(comparator, expected, computed, tolerance) or any(
        _satisfies(comparator, want, got, tol)
        for want, got, tol in comparison_variants(
            written, expected, computed, tolerance,
        )
    )


@pytest.mark.parametrize("written,expected,computed,tolerance", [
    # SCALE. The judge reports "36%" as 36.0, the tool as 0.3603. Only one
    # direction of the conversion was tried, so which way round they landed
    # decided whether a correct figure was called wrong.
    ("36%", 36.0, 0.3603319502074689, 1.0),
    ("36%", 0.36, 0.3603319502074689, 0.005),
    # SIGN. "declined by 2.2%" is a magnitude; the tool reports -0.022.
    ("2.2%", 2.2, -0.022, 0.1),
    ("96.6%", 96.6, -0.966, 0.1),
    # The sign is in the text but the judge dropped it from the value.
    ("-2.2%", 2.2, -0.022, 0.1),
    ("declined 2.2%", 2.2, -0.022, 0.1),
    # BOUND. "all above 720" is satisfied by 721; read as `==` it was a
    # mismatch against the very evidence that proves it.
    ("all above 720", 720.0, 721.0, 0.5),
    ("below 700", 700.0, 690.0, 0.5),
    ("at least 10", 10.0, 10.0, 0.5),
])
def test_the_same_figure_written_differently_agrees(
    written, expected, computed, tolerance,
):
    assert _agrees(written, expected, computed, tolerance)


@pytest.mark.parametrize("written,expected,computed,tolerance,why", [
    ("38%", 38.0, 0.3603319502074689, 0.5, "a genuine 2pp gap"),
    ("38%", 0.38, 0.3603319502074689, 0.01, "the same gap, other scale"),
    ("+5%", 5.0, -0.05, 0.1, "the answer got the direction wrong"),
    ("rose 2.2%", 2.2, -0.022, 0.1, "it fell"),
    ("all above 720", 720.0, 719.0, 0.5, "the bound is violated"),
    ("below 700", 700.0, 710.0, 0.5, "the bound is violated"),
])
def test_a_real_disagreement_survives_every_variant(
    written, expected, computed, tolerance, why,
):
    """The variants must not become a way to pass anything."""
    assert not _agrees(written, expected, computed, tolerance), why


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
