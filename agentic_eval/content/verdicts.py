"""Closed verdict vocabularies for the content cascade.

Every judge answer is coerced into one of these before it reaches a metric,
so an unexpected string degrades to the conservative option instead of
silently creating a new bucket.
"""
from __future__ import annotations


FACT_VERDICTS = {"supported", "contradicted", "unverifiable"}


NUMERIC_SUPPORT_VERDICTS = {"yes", "no", "not_applicable"}


TRACE_VERDICTS = {"yes", "no", "unavailable", "not_applicable"}


CORRECTNESS_VERDICTS = {"yes", "no", "unavailable", "not_applicable"}


MUST_HAVE_VERDICTS = {"full", "partial", "miss", "not_applicable"}


LOGIC_VERDICTS = {"valid", "weak", "invalid", "unverifiable"}


#: Whether the provenance a claim cited exists. Not a ranking of the
#: payload — ◆/◇ carries what the claim rests on.
EVIDENCE_RESOLUTIONS = ("resolved", "unresolved", "none")


CLAIM_STANCES = ("asserted", "attributed_unendorsed", "attributed_refuted")

