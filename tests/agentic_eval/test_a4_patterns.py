"""a4 asks whether the MODEL HAS external-delinquency information.

A good answer proves it by reporting the feature's values — and often says in
the same breath that the feature shows no delinquency events. "no flagged
external delinquency" is about the EVENTS; "no external delinquency data" is
about the FEATURE. They are opposite verdicts and nearly identical strings, so
the denial pattern has to name what is absent.
"""
from __future__ import annotations

import pathlib
import re

import yaml


def _patterns():
    spec = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[2]
         / "experiments/questions/series_a.yaml").read_text()
    )
    question = next(
        q for q in spec["questions"]
        if q["name"] == "a4_external_delinquency_coverage"
    )
    expected = question["evaluation"]["expected_answers"][0]
    return expected["affirmative_patterns"], expected["negative_patterns"]


def _verdict(text):
    affirmative, negative = _patterns()
    if any(re.search(p, text, re.I) for p in negative):
        return False
    if any(re.search(p, text, re.I) for p in affirmative):
        return True
    return None


def test_reporting_the_index_with_no_events_is_a_yes():
    """The answer that was being marked wrong. It cites the index value, so
    the model plainly has the feature."""
    assert _verdict(
        "The internal model recorded persistent external credit risk signals, "
        "but no flagged external delinquency or excessive inquiry periods.\n"
        "- External delinquency index peaked at **0.52**, never breaching the "
        "risk threshold (>5)"
    ) is True


def test_denying_the_feature_is_still_a_no():
    for text in (
        "The model has no external delinquency information for this customer.",
        "There are no external delinquency columns in the modelling table.",
        "The modelling data lacks external delinquency features entirely.",
        "No data on external delinquency is available to the model.",
    ):
        assert _verdict(text) is False, text


def test_a_plain_yes_is_a_yes():
    assert _verdict("Yes, the model includes external delinquency features.") is True


def test_no_delinquency_events_does_not_read_as_no_feature():
    """The distinction the old pattern could not draw."""
    assert _verdict(
        "External delinquency index shows no delinquency events in 23 months."
    ) is True
