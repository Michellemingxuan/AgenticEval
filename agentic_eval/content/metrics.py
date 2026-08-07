"""Metric arithmetic. Denominators are explicit and match the cascade.

Every displayed metric is a numerator and a denominator, never a pre-averaged
rate: the viewer sums them across repeats, so a question answered three times
cannot outweigh one answered once by averaging averages.

Four things are measured, and nothing else is computed:

    Accuracy            did the oracle say the answer was right
    Orthogonal claims   how many distinct facts the answer asserted
    Grounded            factual (traced + eligible route) or report
    Reasoning eligible  of the routes we could assess, how many fit the question
    Must-have hit rate  per question, like accuracy

The rest of this module is diagnosis: what was excluded before verification,
and which failures were the evaluator's rather than the system's.
"""
from __future__ import annotations

from typing import Any

from agentic_eval.common.coerce import _slug
from agentic_eval.content.verdicts import (
    CORRECTNESS_VERDICTS, EVIDENCE_RESOLUTIONS, FACT_VERDICTS,
    MUST_HAVE_VERDICTS,
)

#: A claim scores as grounded on either route. The two are counted separately
#: as well, because they fail for different reasons: a factual claim fails when
#: its figures do not trace or its route answers another question, a report
#: claim when the report material it leans on does not resolve.
_GROUNDED_KINDS = ("factual", "report")


def calculate_content_metrics(
    claims: list[dict[str, Any]], fact_results: list[dict[str, Any]],
    must_have_results: list[dict[str, Any]],
    *, table_coverage: float | None,
    memory_leverage: dict[str, Any] | None = None, tool_provenance_available: bool = True,
    oracle_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def rate(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    # ---- what got excluded before anything was verified --------------------
    # Two filters run before verification, and BOTH are judgement calls that
    # can be wrong. A claim either removes is never scored either way, so it
    # can never appear as ungrounded — it simply leaves. Counting both is what
    # makes those calls auditable instead of silent.
    all_factual = [claim for claim in claims if claim.get("is_factual")]
    non_factual_count = len(claims) - len(all_factual)
    # One fact, counted once. Scoring a restatement again multiplies credit and
    # blame by how often the answer repeated itself, which rewards terseness
    # and punishes self-corroboration — neither is what any of this measures.
    factual = [claim for claim in all_factual if not claim.get("restates_claim_id")]
    restatement_count = len(all_factual) - len(factual)

    # ---- the claims that were actually verified ----------------------------
    facts = {row["claim_id"]: row for row in fact_results}
    verdict_counts = {name: 0 for name in FACT_VERDICTS}
    grounding_counts = {name: 0 for name in (*_GROUNDED_KINDS, "none")}
    eligibility_counts = {name: 0 for name in CORRECTNESS_VERDICTS}
    resolution_counts = {name: 0 for name in EVIDENCE_RESOLUTIONS}
    failure_counts: dict[str, int] = {}
    judge_error_claims = 0
    for claim in factual:
        result = facts.get(claim["claim_id"], {})
        verdict_counts[result.get("verdict", "unverifiable")] = (
            verdict_counts.get(result.get("verdict", "unverifiable"), 0) + 1
        )
        kind = result.get("grounding_kind", "none")
        grounding_counts[kind] = grounding_counts.get(kind, 0) + 1
        eligible = result.get("eligible", "unavailable")
        eligibility_counts[eligible] = eligibility_counts.get(eligible, 0) + 1
        resolution = result.get("evidence_resolution", "none")
        resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        if result.get("judge_error"):
            judge_error_claims += 1
        for failure in result.get("failures") or []:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1

    n_factual = len(factual)
    grounded_count = sum(grounding_counts[kind] for kind in _GROUNDED_KINDS)
    # Ungated by grounding, deliberately. A claim can be perfectly grounded and
    # still not answer the question — figures that trace, over the wrong
    # window. `unavailable` stays out of the denominator: "we could not tell"
    # and "this answers the wrong question" are different findings, and only
    # one of them is about the answer.
    assessable_traces = eligibility_counts["yes"] + eligibility_counts["no"]

    # ---- extractor quality, not answer quality -----------------------------
    # Whether each claim is anchored to text the answer actually contains.
    # Claims Python derived from parsed table cells are excluded — nothing
    # asserted them.
    extracted = [
        claim for claim in claims if claim.get("origin", "extracted") == "extracted"
    ]
    span_failure_counts: dict[str, int] = {}
    for claim in extracted:
        failure = claim.get("span_failure")
        if failure:
            span_failure_counts[failure] = span_failure_counts.get(failure, 0) + 1
    unanchored = sum(span_failure_counts.values())

    # ---- must-haves --------------------------------------------------------
    # Per QUESTION, like accuracy: one answer contributes one vote, so a rubric
    # with five must-haves cannot outweigh one with two. Pooling the weights
    # instead made the row read "8/8" for three must-haves.
    #
    # `partial` is worth half. Scored strictly (hit or nothing) the metric read
    # cleaner but discarded a real distinction: an answer that named the
    # required merchants without linking them is further along than one that
    # ignored the question, and collapsing both to zero makes the rubric blind
    # to the difference it exists to measure. The cost is that a question can
    # read 50% while satisfying nothing outright — `must_have_counts` carries
    # the verdict breakdown so that case stays visible rather than hidden
    # inside the rate.
    #
    # `not_applicable` is excluded from BOTH sides — a point the question does
    # not raise must not count against the answer.
    credits = {"full": 1.0, "partial": 0.5, "miss": 0.0}
    mh_numerator = mh_denominator = 0.0
    mh_counts = {name: 0 for name in MUST_HAVE_VERDICTS}
    for row in must_have_results:
        verdict = row.get("verdict", "miss")
        mh_counts[verdict] = mh_counts.get(verdict, 0) + 1
        if verdict in credits:
            weight = float(row.get("weight") or 1)
            mh_numerator += credits[verdict] * weight
            mh_denominator += weight

    # ---- oracles -----------------------------------------------------------
    # One answer, one verdict: the question was answered correctly if a script
    # checked it and nothing failed. Per question over repeats this becomes
    # "how many times out of how many"; over the set, "how many questions" —
    # both are sums of these two, so the display cannot drift from the rate.
    oracle_counts = {"pass": 0, "fail": 0, "unavailable": 0}
    for row in oracle_results or []:
        verdict = _slug(row.get("verdict") or "unavailable")
        oracle_counts[verdict] = oracle_counts.get(verdict, 0) + 1
    oracle_denominator = oracle_counts["pass"] + oracle_counts["fail"]
    answer_checked = int(oracle_denominator > 0)
    answer_correct = int(answer_checked and oracle_counts["fail"] == 0)

    # ---- memory: arrival and use are different questions -------------------
    # Whether memory REACHED the run is settled deterministically from the
    # trace. This counts the other thing: of the sources it was offered, how
    # many did it actually draw on. Exposed-but-unused is a legible state, not
    # a miss — but leveraged memory is what makes a session cheaper than its
    # turns, so it earns a number of its own.
    leverage = memory_leverage or {}
    sources = leverage.get("sources") or []
    offered = leverage.get("offered") or {}
    memory_sources_offered = sum([
        bool(offered.get("kb_topics")), bool(offered.get("episodic_turns")),
        bool(offered.get("qa_cache")),
    ])
    memory_sources_leveraged = sum(1 for row in sources if row.get("leveraged"))

    return {
        # -- memory -----------------------------------------------------------
        "memory_sources_offered": memory_sources_offered,
        "memory_sources_leveraged": memory_sources_leveraged,
        "memory_leverage_rate": rate(memory_sources_leveraged, memory_sources_offered),
        "memory_leveraged_where": sorted({
            place for row in sources if row.get("leveraged")
            for place in row.get("where") or []
        }),
        "memory_leveraged_sources": sorted(
            row["source"] for row in sources if row.get("leveraged")
        ),

        # -- displayed: numerator and denominator for every content row ------
        "answer_correct": answer_correct,
        "answer_checked": answer_checked,
        # How many distinct things the answer asserted. A verbose answer
        # restating one finding five ways is not five claims.
        "orthogonal_claim_count": n_factual,
        "all_factual_claim_count": len(all_factual),
        "grounded_count": grounded_count,
        "factual_grounded_count": grounding_counts["factual"],
        "report_grounded_count": grounding_counts["report"],
        "eligible_trace_count": eligibility_counts["yes"],
        "assessable_trace_count": assessable_traces,
        "must_have_coverage": rate(mh_numerator, mh_denominator),
        "must_have_questions": 1 if mh_denominator else 0,

        # -- rates, for the per-question walkthrough -------------------------
        "grounded_rate": rate(grounded_count, n_factual),
        "factual_grounded_rate": rate(grounding_counts["factual"], n_factual),
        "report_grounded_rate": rate(grounding_counts["report"], n_factual),
        "reasoning_eligible_rate": rate(eligibility_counts["yes"], assessable_traces),
        "expected_answer_accuracy_rate": rate(oracle_counts["pass"], oracle_denominator),
        "judge_error_rate": rate(judge_error_claims, n_factual),

        # -- diagnosis: what was excluded, and whose fault a gap was ---------
        "extracted_claim_count": len(claims),
        "restated_claim_count": restatement_count,
        "restatement_rate": rate(restatement_count, len(all_factual)),
        "non_factual_claim_count": non_factual_count,
        "unanchored_claim_count": unanchored,
        "claim_span_verified_rate": rate(len(extracted) - unanchored, len(extracted)),
        "span_failure_counts": span_failure_counts,
        "judge_error_claim_count": judge_error_claims,
        "failure_counts": failure_counts,
        "grounding_counts": grounding_counts,
        "eligibility_counts": eligibility_counts,
        "evidence_resolution_counts": resolution_counts,
        "factual_counts": verdict_counts,
        "must_have_counts": mh_counts,
        "must_have_hit_weight": mh_numerator,
        "must_have_total_weight": mh_denominator,
        "expected_answer_counts": oracle_counts,
        # A rubric point marked critical that did not hold. Not a rate: one
        # critical failure is a finding on its own, and dividing it by however
        # many points the rubric happened to list buries it.
        "critical_expected_answer_failures": sum(
            1 for row in oracle_results or []
            if _slug(row.get("verdict")) == "fail" and row.get("critical")
        ),

        "table_cell_coverage": table_coverage,
        # False means the run captured no measured provenance at all. Every
        # grounding verdict below reads `unavailable` in that case, and calling
        # it ungrounded would blame the system for the harness.
        "tool_provenance_available": tool_provenance_available,
    }
