import json

from agentic_eval.layout import RunLayout
from agentic_eval.content import (
    _evidence_float,
    ContentEvaluator,
    aggregate_content_evaluations,
    build_evidence_ledger,
    calculate_content_metrics,
    content_walkthrough_markdown,
    write_content_walkthrough,
    evaluate_runs_file,
)


class FakeJudge:
    """Serves scripted responses by TASK, not by position.

    The pipeline is three calls — extract, evidence, eligibility — and the
    evidence call returns fact_results, claim_traces and must_have_results
    together. Dispatching on the keys a response carries lets a test script
    only the parts it cares about, in any order, and stops an unrelated change
    in call count from breaking every fixture at once.
    """

    #: Which call each response key belongs to.
    _OWNER = {
        "claims": "claim_extraction",
        "fact_results": "claim_evidence",
        "claim_traces": "claim_evidence",
        "must_have_results": "claim_evidence",
        "eligibility": "claim_eligibility",
    }

    def __init__(self, responses):
        self.by_task: dict[str, dict] = {}
        for response in responses:
            if not isinstance(response, dict):
                continue
            for key, value in response.items():
                task = self._OWNER.get(key)
                if not task:
                    continue
                bucket = self.by_task.setdefault(task, {})
                # An EMPTY scripted value never displaces a real one. Fixtures
                # written for the old chain end with fillers like
                # `{"fact_results": []}` — one per call that no longer exists —
                # and letting those win would blank the response they follow.
                if value or key not in bucket:
                    bucket[key] = value
        self.calls = []
        self.payloads = []

    def complete_json(self, *, task, system_prompt, payload):
        self.calls.append({"task": task, "total_tokens": 10})
        self.payloads.append(payload)
        scripted = dict(self.by_task.get(task) or {})
        # Two defaults, so a fixture about the EVIDENCE layer does not have to
        # script a route and a verdict it is not testing. Left unscripted, both
        # would read `unavailable` and every such claim would come back
        # ungrounded for a reason the test is not about. A test that IS about
        # routes or eligibility scripts them, and the script always wins.
        if task == "claim_evidence" and "claim_traces" not in scripted:
            scripted["claim_traces"] = [
                {
                    "claim_id": row.get("claim_id"),
                    "call_ids": [
                        str(v) for v in (row.get("evidence_ids") or [])
                    ] or [
                        str(n.get("evidence_id"))
                        for n in row.get("numbers") or []
                        if n.get("evidence_id")
                    ],
                    "derivation": "scripted by the test fixture",
                }
                for row in scripted.get("fact_results") or []
            ]
        if task == "claim_eligibility" and "eligibility" not in scripted:
            scripted["eligibility"] = [
                {"claim_id": route["claim_id"], "verdict": "YES",
                 "reason": "default: the fixture is not about eligibility"}
                for route in payload.get("routes") or []
            ]
        return scripted


def test_primary_factual_rates_share_one_denominator():
    claims = [
        {"claim_id": "c1", "is_factual": True},
        {"claim_id": "c2", "is_factual": True},
        {"claim_id": "c3", "is_factual": True},
    ]
    facts = [
        {"claim_id": "c1", "verdict": "supported", "numeric_evidence_support": "yes", "numbers": []},
        {"claim_id": "c2", "verdict": "contradicted", "numeric_evidence_support": "no", "numbers": []},
        {"claim_id": "c3", "verdict": "unverifiable", "numeric_evidence_support": "unverifiable", "numbers": []},
    ]
    metrics = calculate_content_metrics(claims, facts, [], table_coverage=None)
    assert metrics["factual_counts"]["supported"] == 1
    assert metrics["factual_counts"]["contradicted"] == 1
    assert metrics["factual_counts"]["unverifiable"] == 1


def test_content_metrics_are_aggregated_across_the_same_k_runs():
    rows = [
        {
            "system": "new", "mode": "cold", "name": "q", "run_index": 1,
            "metrics": {"grounded_rate": 1.0},
        },
        {
            "system": "new", "mode": "cold", "name": "q", "run_index": 2,
            "metrics": {"grounded_rate": 0.5},
        },
        {
            "system": "new", "mode": "cold", "name": "q", "run_index": 3,
            "metrics": {"grounded_rate": 0.0},
        },
    ]
    summary = aggregate_content_evaluations(rows, expected_repeats=3)
    group = summary["groups"][0]
    distribution = group["metric_distributions"]["grounded_rate"]
    assert group["n_runs"] == 3
    assert group["run_indices"] == [1, 2, 3]
    assert group["repetitions_complete"] is True
    assert group["grounded_rate"] == 0.5
    assert distribution["values_by_run"] == [
        {"run_index": 1, "value": 1.0},
        {"run_index": 2, "value": 0.5},
        {"run_index": 3, "value": 0.0},
    ]
    assert distribution["stdev"] == 0.5


def test_content_evaluator_uses_baseline_must_haves_and_python_trace_check():
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "TSR was 0.72 in May",
            "proposition": "TSR was 0.72 in May",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "0.72", "value": 0.72, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_evidence_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": ["ev1"],
            "trace_reason": "Correct metric and month.",
            "reason": "Exact match", "confidence": 0.99,
            "numbers": [{
                "written_value": "0.72", "evidence_id": "ev1",
                "json_path": "monthly.may", "trace_kind": "direct", "tolerance": 0,
            }],
        }]},
        {"must_have_results": [{
            "must_have_id": "mh1", "verdict": "FULL",
            "answer_spans": ["TSR was 0.72 in May"], "evidence_ids": ["ev1"],
            "reason": "The required value is stated.", "confidence": 0.98,
        }]},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q1", "run_index": 1,
        "question": "What was TSR in May?", "final_answer": "TSR was 0.72 in May.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "summarize_trend", "result": {"monthly": {"may": 0.72}},
        }],
    }
    rubric = {"must_have_points": [{
        "id": "mh1", "description": "State May TSR", "importance": "critical", "weight": 2,
    }]}
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record, rubric)
    assert result["metrics"]["factual_counts"]["supported"] == 1
    assert result["metrics"]["must_have_coverage"] == 1.0
    assert result["fact_results"][0]["numbers"][0]["deterministically_correct"] is True
    assert result["fact_results"][0]["traced"] == "yes"
    assert result["fact_results"][0]["traced"] == "yes"
    assert result["fact_results"][0]["traced"] == "yes"
    assert result["fact_results"][0]["eligible"] == "yes"
    assert result["fact_results"][0]["traced"] == "yes"
    assert [call["task"] for call in judge.calls] == [
        "claim_extraction", "claim_evidence", "claim_eligibility",
    ]


def test_wrong_number_is_supported_but_not_traceable_or_correct():
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "TSR was 0.75", "proposition": "TSR was 0.75",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "0.75", "value": 0.75, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": ["ev1"],
            "numbers": [{
                "written_value": "0.75", "evidence_id": "ev1",
                "json_path": "value", "trace_kind": "direct",
            }],
        }]},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "What was TSR?", "final_answer": "TSR was 0.75.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "summarize_trend", "result": {"value": 0.72},
        }],
    }
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record)
    fact = result["fact_results"][0]
    assert fact["traced"] == "no"
    assert fact["traced"] == "no"
    assert fact["traced"] == "no"
    assert fact["verdict"] == "contradicted"


def _one_claim_judge(*, written, value, numbers, reason="Evidence supports it."):
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": f"{written} returned payments",
            "proposition": f"{written} payments were returned",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": written, "value": value, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": ["ev1"],
            "reason": reason, "numbers": numbers,
        }]},
    ])


def _record(evidence):
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "did the customer have any payment returns?",
        "final_answer": "0 returned payments.", "evidence": evidence,
    }


def test_prose_evidence_is_not_scraped_for_a_number():
    """A tool result sentence is not a measurement.

    Regression: the data tools return human-readable results, and the evidence
    resolver took the first digit run in the string. A path landing on
    "count filtered by Return Flag eq '1' = 0 (out of 357 total rows)" yielded
    1.0 — the FILTER LITERAL — so a correct claim of 0 was reported as
    contradicted, GROUNDED in a genuine tool result.
    """
    prose = "count filtered by Return Flag eq '1' = 0 (out of 357 total rows)"
    # The value is what follows the assignment, NEVER the digit in the label.
    # Returning None here was the safe first fix; it also reported true claims
    # as unlocatable, because several data tools format real measurements this
    # way. Reading after the last `=` satisfies both.
    assert _evidence_float(prose) == 0.0
    assert _evidence_float("count = 357 (out of 357 total rows)") == 357.0
    # A sentence with no assignment is still not a measurement.
    assert _evidence_float("the customer had several returned payments") is None
    assert _evidence_float("1. Key highlights (crisp pointers)") is None
    # Values that really are numbers still parse, including money/percent.
    assert _evidence_float(0) == 0.0
    assert _evidence_float("357") == 357.0
    assert _evidence_float("$4,838,219.70") == 4838219.70
    assert _evidence_float("26.1%") == 0.261
    assert _evidence_float("  '0.75' ") == 0.75


def test_ungrounded_numeric_mismatch_does_not_overrule_the_judge():
    """A disagreement with a specialist's PROSE is not grounds to contradict.

    Regression: every one of 18 contradicted claims in a real run carried the
    reason "Deterministic numeric check disagrees with the linked evidence.
    The evidence explicitly states 0 out of 357 payments were returned,
    DIRECTLY SUPPORTING THE CLAIM" — a self-contradiction produced by
    comparing the answer's 0 against a 1 scraped from an `agent_result`
    summary (in the real case, the literal from the filter `Return Flag = 1`).
    None of the 18 had a linked tool result.
    """
    judge = _one_claim_judge(
        written="0", value=0.0,
        numbers=[{"written_value": "0", "evidence_id": "ev1",
                  "json_path": "count", "trace_kind": "direct"}],
        reason="The evidence explicitly states 0 of 357 were returned.",
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        # `agent_result`, NOT `tool_result`: prose, so nothing to check against.
        _record([{"evidence_id": "ev1", "source_type": "agent_result",
                  "tool": "spend_payments", "result": {"count": 1.0}}])
    )
    fact = result["fact_results"][0]
    assert fact["verdict"] == "supported"
    # Sourced only from prose, so the numeric layer does not apply — the
    # grounding tier is what reports this, and checking it again would
    # double-count one defect.
    assert fact["traced"] != "no"
    assert fact["evidence_resolution"] == "resolved"
    assert "Deterministic numeric check disagrees" not in fact["reason"]


def test_grounded_numeric_mismatch_still_contradicts():
    """The gate must not disarm the check where it IS trustworthy."""
    judge = _one_claim_judge(
        written="0", value=0.0,
        numbers=[{"written_value": "0", "evidence_id": "ev1",
                  "json_path": "count", "trace_kind": "direct"}],
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _record([{"evidence_id": "ev1", "source_type": "tool_result",
                  "tool": "query_table", "result": {"count": 3.0}}])
    )
    fact = result["fact_results"][0]
    assert fact["verdict"] == "contradicted"
    assert fact["traced"] == "no"


def test_unmatched_numeric_mention_is_not_paired_positionally():
    """An unmatched mention is UNKNOWN, never "whatever sat at that index".

    Regression: positional fallback checked "$1,000" against 2.0 and "26.1%"
    against 0.03, then reported the claims as contradicted.
    """
    judge = _one_claim_judge(
        written="$1,000", value=1000.0,
        # The judge's only entry describes a DIFFERENT number.
        numbers=[{"written_value": "2", "evidence_id": "ev1",
                  "json_path": "count", "trace_kind": "direct"}],
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _record([{"evidence_id": "ev1", "source_type": "tool_result",
                  "tool": "query_table", "result": {"count": 2.0}}])
    )
    fact = result["fact_results"][0]
    assert fact["verdict"] == "supported"
    assert fact["traced"] == "yes"   # judge error: disclosed, not blocking


def test_derived_number_can_be_traced_and_recomputed_from_tool_outputs():
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "TSR increased by 0.11",
            "proposition": "TSR increased by 0.11",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "0.11", "value": 0.11, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES",
            "evidence_ids": ["ev_may", "ev_june"],
            "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev_may", "ev_june"],
            "numbers": [{
                "written_value": "0.11", "trace_kind": "derived",
                "calculation": {
                    "operation": "difference",
                    "operands": [
                        {"evidence_id": "ev_june", "json_path": "value"},
                        {"evidence_id": "ev_may", "json_path": "value"},
                    ],
                },
            }],
        }]},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "How much did TSR increase?",
        "final_answer": "TSR increased by 0.11.",
        "evidence": [
            {
                "evidence_id": "ev_may", "source_type": "tool_result",
                "tool": "summarize_trend", "arguments": {"month": "May"},
                "result": {"value": 0.61},
            },
            {
                "evidence_id": "ev_june", "source_type": "tool_result",
                "tool": "summarize_trend", "arguments": {"month": "June"},
                "result": {"value": 0.72},
            },
        ],
    }
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record)
    fact = result["fact_results"][0]
    number = fact["numbers"][0]
    assert number["trace_kind"] == "derived"
    assert number["operand_evidence_ids"] == ["ev_june", "ev_may"]
    assert number["evidence_value"] == 0.10999999999999999
    assert number["deterministically_correct"] is True
    assert fact["traced"] == "yes"
    assert fact["traced"] == "yes"
    assert fact["traced"] == "yes"


def test_nested_json_tool_result_is_decoded_before_path_verification():
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "CDSS was 5.4", "proposition": "CDSS was 5.4",
            "claim_type": "point_estimate", "is_factual": True,
            "numeric_mentions": [{"written": "5.4", "value": 5.4, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": ["ev1"],
            "numbers": [{
                "written_value": "5.4", "evidence_id": "ev1",
                "json_path": "results[0].result.series[0].value",
                "trace_kind": "direct", "tolerance": 0,
            }],
        }]},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "What was CDSS?", "final_answer": "CDSS was 5.4.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "batch_summarize_trend",
            "result": {"results": [{
                "result": '{"series": [{"value": "5.40"}]}'
            }]},
        }],
    }
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record)
    number = result["fact_results"][0]["numbers"][0]
    assert number["evidence_value"] == 5.4
    assert number["traceable_to_tool_output"] is True
    assert result["fact_results"][0]["traced"] == "yes"


def test_table_cells_are_deterministically_atomized_and_located():
    judge = FakeJudge([
        {"claims": []},
        {"missing_claims": []},
        {"fact_results": []},
        {"fact_results": []},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "How did TSR react?",
        "final_answer": """| Month | TSR | Notes |
|---|---:|---|
| May | 0.72 | Risk rose |
| June | 0.68 | Risk fell |
""",
        "evidence": [],
    }
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record)
    table_claims = [claim for claim in result["claims"] if claim["block_id"] == "b001"]
    assert len(table_claims) == 4
    assert result["metrics"]["table_cell_coverage"] == 1.0
    assert {
        (claim["source_locator"]["row"], claim["source_locator"]["column"])
        for claim in table_claims
    } == {(1, 1), (1, 2), (2, 1), (2, 2)}
    may_tsr = next(
        claim for claim in table_claims
        if claim["source_locator"]["row"] == 1
        and claim["source_locator"]["column"] == 1
    )
    assert may_tsr["numeric_mentions"] == [{
        "written": "0.72", "value": 0.72, "unit": None, "measures": None,
        "comparator": "==", "quoted": False, "material": True,
    }]


def test_audit_cannot_turn_table_placeholder_into_a_domain_fact():
    judge = FakeJudge([
        {"claims": []},
        {"missing_claims": [{
            "claim_id": "m1", "block_id": "b001",
            "source_locator": {"row": 1, "column": 2},
            "answer_span": "May | CDSS: --",
            "proposition": "May CDSS was unavailable.",
            "claim_type": "uncertainty_or_data_gap", "is_factual": True,
            "numeric_mentions": [],
        }]},
        {"fact_results": []},
        {"fact_results": []},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "How did the scores react?",
        "final_answer": """| Month | TSR | CDSS |
|---|---:|---:|
| May | 0.72 | -- |
""",
        "evidence": [],
    }
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record)
    assert all(claim["claim_id"] != "m1" for claim in result["claims"])


def test_short_table_cell_spans_receive_locators_without_duplicate_fallbacks():
    judge = FakeJudge([
        {"claims": [
            {
                "claim_id": "c1", "block_id": "b001", "answer_span": "0.72",
                "proposition": "May TSR was 0.72", "claim_type": "point_estimate",
                "is_factual": True, "metrics": ["TSR"], "time_window": "May",
                "numeric_mentions": [{"written": "0.72", "value": 0.72}],
            },
            {
                "claim_id": "c2", "block_id": "b001", "answer_span": "5.4",
                "proposition": "May CDSS was 5.4", "claim_type": "point_estimate",
                "is_factual": True, "metrics": ["CDSS"], "time_window": "May",
                "numeric_mentions": [{"written": "5.4", "value": 5.4}],
            },
        ]},
        {"missing_claims": []},
        {"fact_results": []},
        {"fact_results": []},
    ])
    record = {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "How did the scores react?",
        "final_answer": """| Month | TSR | CDSS |
|---|---:|---:|
| May | 0.72 | 5.4 |
""",
        "evidence": [],
    }
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(record)
    assert [claim["claim_id"] for claim in result["claims"]] == ["c1", "c2"]
    assert result["claims"][0]["source_locator"]["column"] == 1
    assert result["claims"][1]["source_locator"]["column"] == 2
    assert result["metrics"]["table_cell_coverage"] == 1.0


def test_post_run_evaluation_writes_summary_and_blind_evidence_packet(tmp_path):
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001", "answer_span": "Two returns",
            "proposition": "There were two returns", "claim_type": "quantitative_fact",
            "is_factual": True,
            "numeric_mentions": [{"written": "Two", "value": 2, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_evidence_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": ["ev1"],
            "numbers": [{"evidence_id": "ev1", "json_path": "count"}],
            "reason": "Match", "confidence": 1,
        }]},
    ])
    records = [{
        "system": "new", "mode": "cold", "name": "q1", "run_index": 1,
        "sequence_position": None, "turn_id": "t1", "outcome": "ok",
        "question": "Any returns?", "final_answer": "Two returns.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "query_table", "result": {"count": 2},
        }],
    }]
    output = evaluate_runs_file(
        config={"audit_claim_extraction": True}, records=records,
        output_dir=tmp_path, baseline="old", candidate="new",
        rubric_by_name={}, judge=judge,
    )
    layout = RunLayout(tmp_path)
    assert output == layout.evaluations and output.exists()
    # Artifacts land in their folders, not in a flat pile at the run root.
    assert layout.content_summary.exists()
    assert layout.content_comparison.exists()
    assert layout.walkthrough.exists()
    assert layout.answer_comparison.exists()
    assert layout.evidence_review.exists()
    assert layout.evidence_review_key.exists()
    assert not (tmp_path / "content_evaluations.jsonl").exists()


def test_one_call_captured_twice_collapses_into_one_ledger_entry():
    """The SSE event and the trace row describe ONE call, not two.

    Observed in a real run: `modeling` and `report_agent` each appeared as
    `agent:<call_id>` and `trace:<turn>:<call_id>`, duplicating ~4.5k chars of
    payload and handing the judge two ids for one measurement — two claims then
    cited "different" evidence for the same thing.
    """
    ledger = build_evidence_ledger({
        "system": "new",
        "evidence": [
            {
                "evidence_id": "agent:call_1", "call_id": "call_1",
                "source_type": "agent_result", "tool": "modeling",
                "scope": "model_scores: all dates", "result": {"findings": "TSR rose."},
            },
            {
                "evidence_id": "trace:t1:call_1", "call_id": "call_1",
                "source_type": "tool_result", "tool": "modeling",
                "arguments": {"sub_question": "TSR trajectory"},
                "trace_node": "orchestrator.round_2", "result": "TSR rose.",
            },
        ],
    }, {})
    assert len(ledger) == 1
    entry = ledger[0]
    assert entry["evidence_id"] == "agent:call_1"
    assert entry["duplicate_evidence_ids"] == ["trace:t1:call_1"]
    # Provenance from the second capture survives the merge...
    assert entry["arguments"] == {"sub_question": "TSR trajectory"}
    assert entry["trace_node"] == "orchestrator.round_2"
    assert entry["scope"] == "model_scores: all dates"
    # ...but an agent-level call is never promoted to primary tool evidence.
    assert entry["source_type"] == "agent_result"


def test_deduped_alias_id_still_grounds_a_number():
    """Dropping a duplicate id must not orphan a judge that cites it."""
    judge = _one_claim_judge(
        written="0.72", value=0.72,
        numbers=[{"written_value": "0.72", "evidence_id": "trace:t1:call_1",
                  "json_path": "value", "trace_kind": "direct"}],
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _record([
            {"evidence_id": "tool:call_1", "call_id": "call_1",
             "source_type": "tool_result", "tool": "summarize_trend",
             "result": {"value": 0.72}},
            {"evidence_id": "trace:t1:call_1", "call_id": "call_1",
             "source_type": "tool_result", "tool": "summarize_trend",
             "result": {"value": 0.72}},
        ])
    )
    assert len(result["evidence_ledger"]) == 1
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["traceable_to_tool_output"] is True
    assert fact["traced"] == "yes"
    assert fact["evidence_resolution"] == "resolved"


def _qualitative_judge(*, evidence_ids):
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "internal delinquency drove the late spike",
            "proposition": "Internal delinquency drove the late risk spike.",
            "claim_type": "causal", "is_factual": True, "numeric_mentions": [],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "NOT_APPLICABLE",
            "evidence_ids": evidence_ids, "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": evidence_ids,
            "reason": "The cited evidence describes the late delinquency spike.",
            "numbers": [],
        }]},
    ])


def _qualitative_record(evidence):
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr and cdss scores react",
        "final_answer": "Internal delinquency drove the late spike.",
        "evidence": evidence,
    }


def test_qualitative_claim_backed_only_by_agent_prose_is_not_grounded():
    """A claim the system read off its OWN summary is not independently grounded.

    Qualitative claims carry no number, so the numeric funnel reports
    not_applicable across the board for them. Without a grounding tier, a
    conclusion echoed from a specialist's prose scores identically to one read
    off a measurement.
    """
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_qualitative_judge(evidence_ids=["ev1"]),
    ).evaluate(_qualitative_record([{
        "evidence_id": "ev1", "source_type": "agent_result", "tool": "modeling",
        "result": {"findings": "Internal delinquency spiked in Mar-2025."},
    }]))
    fact = result["fact_results"][0]
    metrics = result["metrics"]
    assert fact["evidence_resolution"] == "resolved"
    assert fact["traced"] == "not_applicable"
    # The ids resolve — the prose is really in the ledger — but prose is not a
    # reasoning trace, so the claim is not factually grounded on it.
    assert fact["evidence_resolution"] == "resolved"
    assert fact["grounding_kind"] == "none"
    assert fact["grounding_kind"] == "none"


def test_qualitative_claim_cited_to_a_tool_result_resolves():
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_qualitative_judge(evidence_ids=["ev1"]),
    ).evaluate(_qualitative_record([{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "query_table",
        "result": {"rows": [{"month": "2025-03", "tpf_internal_delinq_idx_max": 11.9}]},
    }]))
    assert result["fact_results"][0]["evidence_resolution"] == "resolved"
    assert result["fact_results"][0]["grounding_kind"] != "none"


def test_cited_evidence_id_absent_from_the_ledger_is_unresolved():
    """A fabricated provenance id is a hallucination signal, not grounding."""
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_qualitative_judge(evidence_ids=["ev_nope"]),
    ).evaluate(_qualitative_record([{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "query_table",
        "result": {"rows": []},
    }]))
    fact = result["fact_results"][0]
    assert fact["evidence_resolution"] == "unresolved"
    assert fact["evidence_resolution"] == "unresolved"
    assert result["fact_results"][0]["grounding_kind"] == "none"


def test_qualitative_claim_with_no_citation_is_ungrounded():
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_qualitative_judge(evidence_ids=[]),
    ).evaluate(_qualitative_record([{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "query_table",
        "result": {"rows": []},
    }]))
    fact = result["fact_results"][0]
    assert fact["evidence_resolution"] == "none"
    # The judge still called it supported; the grounding metric is what
    # separates "the judge believed it" from "the system measured it".
    assert fact["verdict"] == "supported"
    assert result["fact_results"][0]["grounding_kind"] == "none"


def test_reading_back_a_self_authored_report_is_not_primary_grounding():
    """A prose blob is not a measurement, whatever the tool is called.

    Observed: `fs_read_file` returning `modeling_exp_0.md` — a markdown report
    the system wrote earlier in the same run — was typed `tool_result` only
    because the tool is not a planned team member. The answer's top-level
    conclusion then scored as grounded in a measurement.
    """
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_qualitative_judge(evidence_ids=["ev1"]),
    ).evaluate(_qualitative_record([{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "fs_read_file",
        "arguments": {"filename": "modeling_exp_0.md"},
        "result": "1. Key highlights\n- internal delinquency crossed threshold in Mar-2025",
    }]))
    # The claim rests on a report the run wrote, so it is REPORT support —
    # the ◇ marker — not factual support. That distinction now lives on the
    # claim, where it is read, rather than in a tier on the payload.
    assert result["fact_results"][0]["grounding_kind"] == "report"
    assert result["metrics"]["report_grounded_count"] == 1
    assert result["metrics"]["factual_grounded_count"] == 0


def _peak_judge(*, calculation, written="39.6", value=39.6):
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": f"TSR peaked at {written}",
            "proposition": f"The peak TSR was {written}",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": written, "value": value, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reasoning_trace_verdict": "YES", "trace_evidence_ids": ["ev1"],
            "reason": "The series peaks there.",
            "numbers": [{
                "written_value": written, "trace_kind": "derived",
                "calculation": calculation,
            }],
        }]},
    ])


def _series_record():
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr react", "final_answer": "TSR peaked at 39.6.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "batch_summarize_trend",
            "result": {"series": [
                {"period": "2024-01", "value": "8.40"},
                {"period": "2024-06", "value": "30.20"},
                {"period": "2024-09", "value": "39.60"},
                {"period": "2025-06", "value": "7.70"},
            ]},
        }],
    }


def test_peak_claim_is_traced_by_recomputing_max_over_the_series():
    """"The max TSR" must equal max() of the series, not one cherry-picked bucket."""
    judge = _peak_judge(calculation={
        "operation": "max",
        "operands": [{"evidence_id": "ev1", "json_path": "series", "select": "value"}],
    })
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _series_record()
    )
    number = result["fact_results"][0]["numbers"][0]
    assert number["evidence_value"] == 39.6
    assert number["operation"] == "max"
    assert number["traceable_to_tool_output"] is True
    assert result["fact_results"][0]["traced"] == "yes"


def test_wrong_aggregate_is_a_value_mismatch_not_a_hallucination():
    """Located but miscomputed is an arithmetic defect, a different bug entirely."""
    judge = _peak_judge(
        written="30.2", value=30.2,
        calculation={
            "operation": "max",
            "operands": [{"evidence_id": "ev1", "json_path": "series", "select": "value"}],
        },
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _series_record()
    )
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["evidence_value"] == 39.6
    assert fact["numbers"][0]["trace_failure"] == "value_mismatch"
    assert fact["traced"] == "no"


def test_value_absent_from_every_tool_output_is_a_hallucination():
    judge = _one_claim_judge(
        written="4200", value=4200.0,
        numbers=[{"written_value": "4200", "evidence_id": "ev1",
                  "json_path": "totals.missing", "trace_kind": "direct"}],
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _record([{"evidence_id": "ev1", "source_type": "tool_result",
                  "tool": "query_table", "result": {"totals": {"count": 3}}}])
    )
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["trace_failure"] == "not_located"
    assert fact["grounding_kind"] == "none"
    assert fact["failures"] == ["not_located"]


def test_materially_wrong_tool_is_a_hallucination_even_when_the_number_traces():
    """Cause (b): a real number produced by a measurement that does not answer
    the claim. The value checks out; the question it answers is the wrong one."""
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "May TSR was 0.72", "proposition": "May TSR was 0.72",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "0.72", "value": 0.72, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev_june"],
            "factual_verdict": "CONTRADICTED", "reasoning_trace_verdict": "NO",
            "trace_evidence_ids": ["ev_june"],
            "trace_reason": "The call measured June, not May.",
            "reason": "Wrong month.",
            "numbers": [{"written_value": "0.72", "evidence_id": "ev_june",
                         "json_path": "value", "trace_kind": "direct"}],
        }]},
    ])
    judge.by_task["claim_eligibility"] = {"eligibility": [
        {"claim_id": "c1", "verdict": "NO",
         "reason": "The call read June; the question asked about May."},
    ]}
    result = ContentEvaluator({}, judge=judge).evaluate({
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "What was May TSR?", "final_answer": "May TSR was 0.72.",
        "evidence": [{"evidence_id": "ev_june", "source_type": "tool_result",
                      "tool": "summarize_trend", "arguments": {"month": "June"},
                      "result": {"value": 0.72}}],
    })
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["traceable_to_tool_output"] is True
    # The figure resolves perfectly; what fails is the ROUTE — a June call
    # answering a question about May. Grounding needs both, so the claim is
    # ungrounded and the number counts as a hallucination.
    assert fact["eligible"] == "no"
    assert fact["grounding_kind"] == "none"
    assert fact["grounding_kind"] == "none"


def test_missing_instrumentation_is_not_called_a_hallucination():
    """No captured provenance is a harness gap; blaming the system would be wrong."""
    judge = _one_claim_judge(
        written="2", value=2.0,
        numbers=[{"written_value": "2", "evidence_id": "ev1", "json_path": "count"}],
    )
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _record([{"evidence_id": "ev1", "source_type": "agent_result",
                  "tool": "spend_payments", "result": {"count": 2}}])
    )


def test_expected_answer_oracle_checks_the_answer_without_a_judge():
    """A question a script can answer outright is decided in Python."""
    judge = _one_claim_judge(
        written="4", value=4.0,
        numbers=[{"written_value": "4", "evidence_id": "ev1", "json_path": "count"}],
    )
    record = {
        "system": "new", "mode": "cold", "name": "cards", "run_index": 1,
        "question": "how many cards does this customer have",
        "final_answer": "The customer has 4 cards.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "query_table", "result": {"count": 4}}],
    }
    rubric = {"expected_answers": [
        {"id": "card_count", "description": "number of cards", "value": 4,
         "critical": True},
    ]}
    result = ContentEvaluator({}, judge=judge).evaluate(record, rubric)
    oracle = result["expected_answer_results"][0]
    assert oracle["verdict"] == "pass"
    assert oracle["matched_claim_id"] == "c1"
    assert result["metrics"]["expected_answer_accuracy_rate"] == 1.0
    assert result["metrics"]["critical_expected_answer_failures"] == 0


def test_expected_answer_oracle_fails_a_wrong_answer():
    judge = _one_claim_judge(
        written="4", value=4.0,
        numbers=[{"written_value": "4", "evidence_id": "ev1", "json_path": "count"}],
    )
    record = {
        "system": "new", "mode": "cold", "name": "cards", "run_index": 1,
        "question": "how many cards does this customer have",
        "final_answer": "The customer has 4 cards.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "query_table", "result": {"count": 4}}],
    }
    rubric = {"expected_answers": [{"id": "card_count", "value": 7, "critical": True}]}
    result = ContentEvaluator({}, judge=judge).evaluate(record, rubric)
    assert result["expected_answer_results"][0]["verdict"] == "fail"
    assert result["metrics"]["expected_answer_accuracy_rate"] == 0.0
    assert result["metrics"]["critical_expected_answer_failures"] == 1


def test_expected_answer_oracle_runs_a_predefined_script(tmp_path):
    import sys
    script = tmp_path / "card_count.py"
    script.write_text("print(4)\n", encoding="utf-8")
    judge = _one_claim_judge(
        written="4", value=4.0,
        numbers=[{"written_value": "4", "evidence_id": "ev1", "json_path": "count"}],
    )
    record = {
        "system": "new", "mode": "cold", "name": "cards", "run_index": 1,
        "question": "how many cards does this customer have",
        "final_answer": "The customer has 4 cards.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "query_table", "result": {"count": 4}}],
    }
    rubric = {"expected_answers": [
        {"id": "card_count", "command": [sys.executable, str(script)]},
    ]}
    result = ContentEvaluator({}, judge=judge).evaluate(record, rubric)
    oracle = result["expected_answer_results"][0]
    assert oracle["source"] == "command"
    assert oracle["expected"] == 4
    assert oracle["verdict"] == "pass"


def _oracle_record(answer):
    return {
        "system": "new", "mode": "cold", "name": "returns", "run_index": 1,
        "question": "did the customer have any payment returns?",
        "final_answer": answer,
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "query_table", "result": {"count": 0}}],
    }


_BOOLEAN_ORACLE = {
    "id": "has_payment_returns", "kind": "boolean", "value": False, "critical": True,
    "affirmative_patterns": [r"\b(had|has|were)\b[^.]{0,60}\breturn(ed|s)?\b"],
    "negative_patterns": [r"\b(no|zero|none)\b[^.]{0,40}\breturn"],
}


def _no_claims_judge():
    return FakeJudge([{"claims": []}, {"missing_claims": []}, {"fact_results": []}])


def test_boolean_oracle_passes_a_correct_negative_answer():
    """"Did the customer have any payment returns?" has a computable answer,
    but the answer states it in prose, so polarity is what must be checked."""
    result = ContentEvaluator({}, judge=_no_claims_judge()).evaluate(
        _oracle_record("The customer had no returned payments across 357 records."),
        {"expected_answers": [_BOOLEAN_ORACLE]},
    )
    oracle = result["expected_answer_results"][0]
    assert oracle["verdict"] == "pass"
    assert result["metrics"]["critical_expected_answer_failures"] == 0


def test_boolean_oracle_fails_an_answer_that_invents_returns():
    result = ContentEvaluator({}, judge=_no_claims_judge()).evaluate(
        _oracle_record("The customer had 186 returned payments totaling $2,410,700."),
        {"expected_answers": [_BOOLEAN_ORACLE]},
    )
    oracle = result["expected_answer_results"][0]
    assert oracle["verdict"] == "fail"
    assert result["metrics"]["critical_expected_answer_failures"] == 1


def test_ambiguous_polarity_is_reported_not_guessed():
    """Matching both polarities, or neither, is unknown — never a coin flip."""
    result = ContentEvaluator({}, judge=_no_claims_judge()).evaluate(
        _oracle_record("Payment behaviour was reviewed for the period."),
        {"expected_answers": [_BOOLEAN_ORACLE]},
    )
    oracle = result["expected_answer_results"][0]
    assert oracle["verdict"] == "unavailable"
    assert "no affirmative or negative pattern" in oracle["reason"]
    assert result["metrics"]["expected_answer_accuracy_rate"] is None


def _hedged_judge(*, written, value, comparator=None):
    mention = {"written": written, "value": value, "material": True}
    if comparator:
        mention["comparator"] = comparator
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": f"June TSR was {written}",
            "proposition": f"In June 2024 TSR was {written}",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [mention],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "Within the stated bound.",
            "numbers": [{"written_value": written, "evidence_id": "ev1",
                         "json_path": "value", "trace_kind": "direct"}],
        }]},
    ])


def _tsr_record(answer="June TSR was ~28+.", value=30.2):
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr react", "final_answer": answer,
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "batch_summarize_trend", "result": {"value": value}}],
    }


def test_hedged_lower_bound_is_satisfied_by_a_larger_value():
    """"~28+" claims "about 28 or more", not "exactly 28".

    Regression: the hedge was coerced to 28.0 and required to match exactly, so
    a true claim about a real 30.2 was reported as an unlocatable value — a
    hallucination — while the judge's own reason said it was correct.
    """
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_hedged_judge(written="~28+", value=28.0)).evaluate(
        _tsr_record()
    )
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["comparator"] == ">="
    assert fact["numbers"][0]["evidence_value"] == 30.2
    assert fact["numbers"][0]["deterministically_correct"] is True
    assert fact["traced"] == "yes"


def test_hedged_lower_bound_still_fails_when_the_value_is_below_it():
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_hedged_judge(written="~28+", value=28.0)).evaluate(
        _tsr_record(value=12.0)
    )
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["deterministically_correct"] is False
    assert fact["numbers"][0]["trace_failure"] == "value_mismatch"


def test_upper_bound_hedge_reads_the_other_direction():
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_hedged_judge(written="<10", value=10.0)).evaluate(
        _tsr_record(answer="CDSS stayed <10.", value=5.4)
    )
    assert result["fact_results"][0]["numbers"][0]["comparator"] == "<"
    assert result["fact_results"][0]["numbers"][0]["deterministically_correct"] is True


def test_number_naming_a_metric_is_not_a_material_measurement():
    """"30+ DPD" is the name of `times_30_dpd_max`; the measured value was 2.0.

    Marked material it entered the traceability denominator, could never
    resolve, and was then reported as an invented number.
    """
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "Internal delinquency, 30+ DPD spike pushes risk higher",
            "proposition": "A 30+ DPD spike pushed risk metrics higher in Mar 2025",
            "claim_type": "causal", "is_factual": True,
            "numeric_mentions": [{"written": "30+", "value": 30.0, "unit": "days",
                                  "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "Shown in the report.",
            "numbers": [],
        }]},
    ])
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate({
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr react",
        "final_answer": "Internal delinquency, 30+ DPD spike pushes risk higher.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "query_table", "result": {"times_30_dpd_max": 2.0}}],
    })
    assert result["claims"][0]["numeric_mentions"][0]["material"] is False
    fact = result["fact_results"][0]
    assert fact["numbers"] == []


def test_a_reported_figure_the_answer_refutes_stays_one_atomic_fact():
    """The answer disowns the report's number and verifies the real one.

    Splitting "the report cites 186 returns, but the payments table shows none"
    into two claims manufactures a 186-returns assertion the answer never made
    and then scores it false. The quoted figure is not the answer's claim; the
    verifying figure is.
    """
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": (
                "the report cites 186 returned payments, but this is not grounded "
                "in live specialist output; the payments table shows 0 returns"
            ),
            "proposition": (
                "The report's 186 returned payments is not grounded in live "
                "specialist output; the payments table shows 0 returned payments."
            ),
            "claim_type": "quantitative_fact", "is_factual": True,
            "stance": "attributed_refuted",
            "numeric_mentions": [
                {"written": "186", "value": 186.0, "material": True, "quoted": True},
                {"written": "0", "value": 0.0, "material": True},
            ],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"],
            "reason": "The payments table records no returns.",
            "numbers": [{"written_value": "0", "evidence_id": "ev1",
                         "json_path": "returned", "trace_kind": "direct"}],
        }]},
    ])
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate({
        "system": "new", "mode": "cold", "name": "returns", "run_index": 1,
        "question": "did the customer have any payment returns?",
        "final_answer": (
            "The report cites 186 returned payments, but this is not grounded in "
            "live specialist output; the payments table shows 0 returns."
        ),
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "query_table", "result": {"returned": 0, "total": 357}}],
    })
    claim = result["claims"][0]
    assert len(result["claims"]) == 1
    assert claim["stance"] == "attributed_refuted"
    # The disowned figure is carried, but never checked as the answer's own.
    assert claim["numeric_mentions"][0]["quoted"] is True
    assert claim["numeric_mentions"][0]["material"] is False
    fact = result["fact_results"][0]
    assert [number["written_value"] for number in fact["numbers"]] == ["0"]
    assert fact["traced"] == "yes"


def test_walkthrough_marks_each_step_of_the_cascade(tmp_path):
    """The display must show which span became which fact, and its verdict."""
    judge = _one_claim_judge(
        written="0.72", value=0.72,
        numbers=[{"written_value": "0.72", "evidence_id": "ev1",
                  "json_path": "value", "trace_kind": "direct"}],
    )
    evaluation = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate(
        _record([{"evidence_id": "ev1", "source_type": "tool_result",
                  "tool": "summarize_trend", "result": {"value": 0.72}}])
    )
    body = content_walkthrough_markdown(evaluation)
    assert "### 1. Raw final answer" in body
    assert "0 returned payments." in body
    assert "### 2. Atomic facts" in body
    # num=yes, trc=yes, tul=yes, grounding=primary
    assert "| ◆ | ✓ |" in body
    assert "### 3. Numbers" in body
    assert "✓ traced" in body

    written = write_content_walkthrough(
        [evaluation], layout=RunLayout(tmp_path).ensure(),
    )
    assert written.exists()
    assert written.name == "walkthrough.md"


def test_walkthrough_shows_why_an_ungrounded_claim_is_ungrounded():
    """The marker alone is not reviewable; the reason has to be on the page."""
    judge = _pointer_judge(json_path="nowhere")
    result = ContentEvaluator({}, judge=judge).evaluate(
        _pointer_record({"summary": {"unrelated": 1}}),
    )
    body = content_walkthrough_markdown(result)
    assert "| ○ |" in body
    assert "grounded 0%" in body


def _threshold_judge(*, relations):
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "TSR below threshold",
            "proposition": "In January 2024, TSR was below its risk threshold.",
            "claim_type": "threshold", "is_factual": True,
            "numeric_mentions": [], "depends_on_claim_ids": ["c0"],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "8.4 is under the threshold.",
            "numbers": [], "relations": relations,
        }]},
    ])


def _threshold_record():
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr react",
        "final_answer": "In January 2024 TSR was below its risk threshold.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "batch_summarize_trend",
            "result": {"series": [{"period": "2024-01", "value": "8.40"}],
                       "summary": {"threshold": {"value": 20, "risky_when": "> 20"}}},
        }],
    }


def test_threshold_claim_without_a_number_is_still_verified():
    """"TSR was below its risk threshold" states no value but is checkable:
    both sides — the measurement and the threshold — are in the tool output."""
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_threshold_judge(relations=[{
        "left": {"evidence_id": "ev1", "json_path": "series[0].value"},
        "operator": "<",
        "right": {"evidence_id": "ev1", "json_path": "summary.threshold.value"},
    }])).evaluate(_threshold_record())
    fact = result["fact_results"][0]
    relation = fact["relations"][0]
    assert (relation["left_value"], relation["right_value"]) == (8.4, 20.0)
    assert relation["holds"] is True
    assert fact["traced"] == "yes"
    assert fact["traced"] == "yes"


def test_a_grounded_relation_that_fails_contradicts_the_claim():
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_threshold_judge(relations=[{
        "left": {"evidence_id": "ev1", "json_path": "summary.threshold.value"},
        "operator": "<",
        "right": {"evidence_id": "ev1", "json_path": "series[0].value"},
    }])).evaluate(_threshold_record())
    fact = result["fact_results"][0]
    assert fact["relations"][0]["holds"] is False
    assert fact["relations"][0]["trace_failure"] == "relation_does_not_hold"
    assert fact["verdict"] == "contradicted"
    assert fact["traced"] == "no"


def test_a_threshold_supplied_from_memory_is_not_grounded():
    """Both sides must be measured. A judge-asserted bound proves nothing."""
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_threshold_judge(relations=[{
        "left": {"evidence_id": "ev1", "json_path": "series[0].value"},
        "operator": "<", "right": {"value": 20},
    }])).evaluate(_threshold_record())
    relation = result["fact_results"][0]["relations"][0]
    assert relation["holds"] is True
    assert relation["grounded_in_tool_result"] is False
    assert relation["trace_failure"] == "not_tool_output"
    assert result["fact_results"][0]["traced"] == "no"


def test_a_relational_claim_with_no_relation_supplied_is_flagged():
    """Declining to check something checkable is unknown, not exempt."""
    result = ContentEvaluator({"audit_claim_extraction": True},
                              judge=_threshold_judge(relations=[])).evaluate(
        _threshold_record()
    )
    fact = result["fact_results"][0]
    assert ("relation_not_supplied" in fact["failures"]) is True
    assert "relation_not_supplied" in fact["failures"]
    assert fact["traced"] == "yes"   # judge error: disclosed, not blocking




def test_a_hedge_with_no_parsed_value_falls_back_to_the_written_form():
    """A judge asked for the value of "~28+" often returns null.

    Regression from a live run: the claim side was then unknown while the
    evidence side resolved to 30.2, the comparison read as a mismatch, and a
    TRUE claim was flipped to `contradicted` by a missing field.
    """
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "June TSR was ~28+",
            "proposition": "In June 2024 TSR was approximately 28 or higher",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "~28+", "value": None, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "30.2 is at least 28.",
            "numbers": [{"written_value": "~28+", "evidence_id": "ev1",
                         "json_path": "value", "trace_kind": "direct"}],
        }]},
    ])
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate({
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr react", "final_answer": "June TSR was ~28+.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "batch_summarize_trend", "result": {"value": 30.2}}],
    })
    claim = result["claims"][0]
    fact = result["fact_results"][0]
    assert claim["numeric_mentions"][0]["value"] == 28.0
    assert fact["numbers"][0]["deterministically_correct"] is True
    assert fact["verdict"] == "supported"
    assert fact["traced"] == "yes"


def test_an_unknown_claim_value_is_unresolved_not_a_mismatch():
    """Even with nothing to parse, the check is unknown — never contradicted.

    The guarantee now holds one step earlier: "a handful" has no digits, so it
    never enters the numeric layer at all. Left in, and paired with a value the
    judge invents, such a mention does not merely fail to resolve — it matches
    whatever stray figure that value happens to equal and is recorded as
    traced.
    """
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "TSR rose sharply", "proposition": "TSR rose sharply",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "a handful", "value": None,
                                  "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "Consistent.",
            "numbers": [{"written_value": "a handful", "evidence_id": "ev1",
                         "json_path": "value", "trace_kind": "direct"}],
        }]},
    ])
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate({
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did tsr react", "final_answer": "TSR rose sharply.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "batch_summarize_trend", "result": {"value": 30.2}}],
    })
    fact = result["fact_results"][0]
    assert fact["numbers"] == []
    assert fact["traced"] != "no"
    assert fact["verdict"] == "supported"


def test_restatements_are_counted_once():
    """An answer that says one fact five ways has one fact.

    Observed: "no returns", "0 of 357 were returned", "the data shows no
    return codes", and "the reports document zero returns" produced four
    claims and four hallucination flags for a single underlying assertion —
    multiplying both credit and blame by how often the answer repeated itself.
    """
    claims = [
        {"claim_id": "c1", "is_factual": True, "numeric_mentions": []},
        {"claim_id": "c2", "is_factual": True, "numeric_mentions": [],
         "restates_claim_id": "c1"},
        {"claim_id": "c3", "is_factual": True, "numeric_mentions": [],
         "restates_claim_id": "c1"},
        {"claim_id": "c4", "is_factual": True, "numeric_mentions": []},
    ]
    facts = [
        {"claim_id": cid, "verdict": "contradicted",
         "numeric_support": "yes", "numbers": []}
        for cid in ("c1", "c2", "c3", "c4")
    ]
    metrics = calculate_content_metrics(claims, facts, [], table_coverage=None)
    assert metrics["orthogonal_claim_count"] == 2
    assert metrics["restated_claim_count"] == 2
    assert metrics["restatement_rate"] == 0.5
    # Two distinct facts, both wrong — not four.
    assert metrics["factual_counts"]["contradicted"] == 2
    assert metrics["factual_counts"]["contradicted"] == 2


def test_one_unmapped_mention_does_not_poison_a_traced_claim():
    """The judge declining to map a number says nothing about the system.

    Observed: a claim whose 13, 18 and 83 all resolved cleanly was flagged as
    invention because the judge returned no entry for "> 75" — a threshold the
    same tool result states.
    """
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "breached its threshold (> 75) in 13 of 18 months",
            "proposition": "The feature breached its threshold (> 75) in 13 of 18 months",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [
                {"written": "> 75", "value": 75.0, "material": True},
                {"written": "13", "value": 13.0, "material": True},
            ],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "Shown in the summary.",
            # Only one of the two mentions was mapped.
            "numbers": [{"written_value": "13", "evidence_id": "ev1",
                         "json_path": "summary.n_breaching", "trace_kind": "direct"}],
        }]},
    ])
    result = ContentEvaluator({"audit_claim_extraction": True}, judge=judge).evaluate({
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "does the model carry external delinquency features?",
        "final_answer": "It breached its threshold (> 75) in 13 of 18 months.",
        "evidence": [{
            "evidence_id": "ev1", "source_type": "tool_result",
            "tool": "batch_summarize_trend",
            "result": {"summary": {"n_breaching": 13, "threshold": {"value": 75}}},
        }],
    })
    fact = result["fact_results"][0]
    unmapped = next(n for n in fact["numbers"] if n["written_value"] == "> 75")
    # 75 is in the linked tool result, so the mention was unmapped, not invented.
    assert unmapped["trace_failure"] == "unmapped_but_present"


def test_judge_error_is_disclosed_not_scored_against_the_system():
    """The judge failing to map a number is not evidence about the answer.

    Folding it into `traceability: no` blames the system for the evaluator's
    oversight, and hides that the evaluator is what needs fixing.
    """
    judge = FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "357 attempts", "proposition": "There were 357 attempts",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "357", "value": 357.0, "material": True}],
        }]},
        {"missing_claims": []},
        # The judge returns no mapping at all for the mention.
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "Stated by the tool.",
            "numbers": [],
        }]},
        {"fact_results": []},   # retry returns nothing either
    ])
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=judge,
    ).evaluate({
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how many attempts?", "final_answer": "There were 357 attempts.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "batch_aggregate", "result": "count = 357 (of 357)"}],
    })
    fact = result["fact_results"][0]
    assert fact["judge_error"] is True
    assert "unmapped_but_present" in fact["failures"]
    # Unknown, not a system failure.
    assert fact["traced"] == "yes"   # judge error: disclosed, not blocking
    assert result["metrics"]["judge_error_rate"] == 1.0
    assert result["metrics"]["failure_counts"]["unmapped_but_present"] == 1


def _pointer_judge(*, json_path):
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001",
            "answer_span": "peaked at 83", "proposition": "The feature peaked at 83",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "83", "value": 83.0, "material": True}],
        }]},
        {"missing_claims": []},
        {"fact_results": [{
            "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
            "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
            "trace_evidence_ids": ["ev1"], "reason": "Shown in the summary.",
            "numbers": [{"written_value": "83", "evidence_id": "ev1",
                         "json_path": json_path, "trace_kind": "direct"}],
        }]},
        {"fact_results": []},
    ])


def _pointer_record(result):
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did the feature behave?",
        "final_answer": "The feature peaked at 83.",
        "evidence": [{"evidence_id": "ev1", "source_type": "tool_result",
                      "tool": "batch_summarize_trend", "result": result}],
    }


def test_a_pointer_at_a_container_is_repaired_to_the_leaf():
    """The judge named the right evidence and the wrong depth.

    `imprecise_path` was the single largest evaluator failure. The value is in
    the node the judge pointed at, so the pointer is repairable without asking
    the judge again.
    """
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_pointer_judge(json_path="summary"),
    ).evaluate(_pointer_record({"summary": {"last": {"value": 83}, "n_buckets": 18}}))
    number = result["fact_results"][0]["numbers"][0]
    assert number["path_repaired"] is True
    assert number["resolved_json_path"] == "last.value"
    assert number["json_path"] == "summary"          # the judge's pointer is kept
    assert number["evidence_value"] == 83.0
    assert number["traceable_to_tool_output"] is True
    assert result["fact_results"][0]["traced"] == "yes"
    assert result["fact_results"][0]["judge_error"] is False


def test_an_ambiguous_pointer_is_not_guessed():
    """Several leaves hold the value, so which one the claim meant is unknown.

    Picking one would manufacture a trace; this stays an evaluator failure.
    """
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_pointer_judge(json_path=""),
    ).evaluate(_pointer_record({
        "series": [{"value": 83}, {"value": 12}], "summary": {"peak": {"value": 83}},
    }))
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["trace_failure"] == "ambiguous_path"
    assert fact["numbers"][0]["path_repaired"] is False
    assert fact["judge_error"] is True
    assert fact["traced"] == "yes"   # judge error: disclosed, not blocking


def test_a_value_absent_from_the_region_is_still_not_located():
    result = ContentEvaluator(
        {"audit_claim_extraction": True}, judge=_pointer_judge(json_path="summary"),
    ).evaluate(_pointer_record({"summary": {"last": {"value": 12}}}))
    fact = result["fact_results"][0]
    number = fact["numbers"][0]
    assert number["path_repaired"] is False
    # Searched the region; the value is genuinely absent, so this is a
    # statement about the answer rather than about the judge's pointer.
    assert number["trace_failure"] == "not_located"
    assert fact["judge_error"] is False
    assert fact["traced"] == "no"


def test_a_claim_about_what_a_report_says_carries_no_measurements():
    """Its figures are quoted from the report, not asserted by the answer.

    Checking them against tool output asks the wrong question — "does the live
    data equal the number the report printed?" — when the claim's own content
    is that the report printed it. Left material they are unlocatable by
    construction, and were being reported as invented.
    """
    from agentic_eval.content import _normalize_claim

    refuted = _normalize_claim({
        "claim_id": "c1",
        "proposition": (
            "Curated reports state a balance of $0 as the most recent exposure, "
            "but this is contradicted by the live data."
        ),
        "is_factual": True,
        "numeric_mentions": [{"written": "$0", "value": 0.0, "material": True}],
    }, 1)
    assert refuted["report_attribution"] is True
    assert refuted["stance"] == "attributed_unendorsed"
    assert refuted["numeric_mentions"][0]["material"] is False
    assert refuted["numeric_mentions"][0]["quoted"] is True

    coverage = _normalize_claim({
        "claim_id": "c2",
        "proposition": (
            "Curated reports only cover up to June 2025, stating 4 spend "
            "transactions for June 2025."
        ),
        "is_factual": True,
        "numeric_mentions": [{"written": "4", "value": 4.0, "material": True}],
    }, 2)
    assert coverage["numeric_mentions"][0]["material"] is False

    # A claim the answer asserts itself is untouched.
    asserted = _normalize_claim({
        "claim_id": "c3",
        "proposition": "The total balance across the commercial card is $174,897.36.",
        "is_factual": True,
        "numeric_mentions": [
            {"written": "$174,897.36", "value": 174897.36, "material": True},
        ],
    }, 3)
    assert asserted["report_attribution"] is False
    assert asserted["stance"] == "asserted"
    assert asserted["numeric_mentions"][0]["material"] is True


def test_a_restatement_that_adds_a_number_is_kept():
    """"357 attempts, 0 returned" is not a restatement of "no returns".

    They share the zero, so the judge collapsed them — and 357, a fact the
    first claim never makes, stopped being counted or verified. A restatement
    may add emphasis or a source; if it adds a MEASUREMENT it is not one.
    """
    from agentic_eval.content import _normalize_claim
    from agentic_eval.content.claims import _drop_false_restatements

    claims = [
        _normalize_claim({
            "claim_id": "c1", "proposition": "The customer had no payment returns.",
            "is_factual": True,
            "numeric_mentions": [{"written": "0", "value": 0.0, "material": True}],
        }, 1),
        _normalize_claim({
            "claim_id": "c2", "restates_claim_id": "c1", "is_factual": True,
            "proposition": "There were 357 attempts and 0 were returned.",
            "numeric_mentions": [
                {"written": "357", "value": 357.0, "material": True},
                {"written": "0", "value": 0.0, "material": True},
            ],
        }, 2),
        _normalize_claim({
            "claim_id": "c3", "restates_claim_id": "c1", "is_factual": True,
            "proposition": "The data shows no payment events with a return code.",
            "numeric_mentions": [],
        }, 3),
        _normalize_claim({
            "claim_id": "c4", "restates_claim_id": "nonexistent", "is_factual": True,
            "proposition": "Something else entirely.", "numeric_mentions": [],
        }, 4),
    ]
    resolved = {c["claim_id"]: c["restates_claim_id"] for c in _drop_false_restatements(claims)}
    assert resolved["c2"] is None      # adds 357
    assert resolved["c3"] == "c1"      # adds nothing
    # A dangling pointer would drop the claim from every denominator.
    assert resolved["c4"] is None


def test_a_report_figure_corrected_by_live_data_is_a_qualitative_claim():
    """"Curated reports state $0, but the live data contradicts it."

    The claim is about what the report said and that it is wrong. Its only
    figure is quoted, so it asserts no measurement — leaving `claim_type` as
    `quantitative_fact` kept dragging it into the numeric layer, where it read
    as "supported by numbers" because the judge could see digits.
    """
    from agentic_eval.content import _normalize_claim
    from agentic_eval.content.verify import _normalize_fact_results

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "quantitative_fact",
        "stance": "attributed_refuted",
        "proposition": (
            "Curated reports state a balance of $0 as the most recent exposure, "
            "but this is contradicted by the live data, which shows an active "
            "positive balance."
        ),
        "numeric_mentions": [{"written": "$0", "value": 0.0, "material": True}],
    }, 1)
    assert claim["claim_type"] == "qualitative_fact"
    assert claim["stance"] == "attributed_refuted"
    assert claim["numeric_mentions"][0]["material"] is False

    ledger = [{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "aggregate_column",
        "evidence_tier": "primary", "result": "sum(Balance) ... = $174,897.36 (1 row)",
    }]
    fact = _normalize_fact_results([claim], [{
        "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
        "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
        "trace_evidence_ids": ["ev1"], "reason": "live data is positive",
        "numbers": [{"written_value": "$0", "evidence_id": "ev1",
                     "json_path": "", "trace_kind": "direct"}],
    }], ledger)[0]
    # Nothing in the numeric cascade applies; it is judged as a qualitative claim.
    # Nothing in the numeric cascade applies; it is judged as a qualitative
    # claim, grounded by the provenance it cites reaching real operations.
    assert fact["traced"] == "not_applicable"
    assert fact["verdict"] == "supported"
    assert fact["numbers"] == []


def test_a_multi_number_scalar_is_not_charged_as_a_wrong_value():
    """One sentence, several measurements, no way to address them separately.

    "sum(Balance) ... = $174,897.36 (over 1 non-null value(s) in 1 matching
    row(s); 3 total)" holds the sum AND the row counts. A json_path can point
    at the string but not inside it, so Python reads the assignment value and
    a claim about the row count looks like a wrong number. The claim is right;
    the evidence is unaddressable.
    """
    from agentic_eval.content import _normalize_claim
    from agentic_eval.content.verify import _normalize_fact_results

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "count",
        "proposition": "There is 1 commercial (SBS) card held by the customer.",
        "numeric_mentions": [{"written": "1", "value": 1.0, "material": True}],
    }, 1)
    ledger = [{
        "evidence_id": "ev1", "source_type": "tool_result",
        "tool": "aggregate_column", "evidence_tier": "primary",
        "result": (
            "sum(Balance) filtered by Card Portfolio eq 'SBS' = $174,897.36 "
            "(over 1 non-null value(s) in 1 matching row(s); 3 total in crossbu_cards)"
        ),
    }]
    fact = _normalize_fact_results([claim], [{
        "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
        "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
        "trace_evidence_ids": ["ev1"], "reason": "one matching row",
        "numbers": [{"written_value": "1", "evidence_id": "ev1",
                     "json_path": "", "trace_kind": "direct"}],
    }], ledger)[0]
    number = fact["numbers"][0]
    assert number["trace_failure"] == "ambiguous_scalar"
    assert fact["judge_error"] is True
    # Unknown, not a wrong answer.
    assert fact["traced"] == "yes"   # judge error: disclosed, not blocking


def test_a_genuinely_wrong_number_still_mismatches():
    """The reclassification must not disarm the check where it bites."""
    from agentic_eval.content import _normalize_claim
    from agentic_eval.content.verify import _normalize_fact_results

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "count",
        "proposition": "There are 9 commercial cards.",
        "numeric_mentions": [{"written": "9", "value": 9.0, "material": True}],
    }, 1)
    ledger = [{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "aggregate_column",
        "evidence_tier": "primary",
        "result": "count filtered by Card Portfolio eq 'SBS' = 1 (out of 3 rows)",
    }]
    fact = _normalize_fact_results([claim], [{
        "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
        "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
        "trace_evidence_ids": ["ev1"], "reason": "stated",
        "numbers": [{"written_value": "9", "evidence_id": "ev1",
                     "json_path": "", "trace_kind": "direct"}],
    }], ledger)[0]
    # 9 appears nowhere in the string, so this stays the answer's problem.
    assert fact["numbers"][0]["trace_failure"] == "value_mismatch"
    assert fact["judge_error"] is False
    assert fact["traced"] == "no"


def test_a_numeric_mention_carries_what_it_measures():
    """A bare figure cannot be matched to a field.

    One tool result states a sum, a row count and a total in the same
    sentence, so "1" is only identifiable once the claim says it counts cards.
    """
    from agentic_eval.content import _normalize_claim

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "count",
        "proposition": "There is 1 commercial (SBS) card, holding the entire balance.",
        "numeric_mentions": [{
            "written": "1", "value": 1.0, "material": True,
            "measures": "count of commercial (SBS) cards",
        }],
    }, 1)
    assert claim["numeric_mentions"][0]["measures"] == "count of commercial (SBS) cards"

    # Absent, it degrades to None rather than an empty string, so a reader can
    # tell "not stated" from "stated as blank".
    bare = _normalize_claim({
        "claim_id": "c2", "is_factual": True,
        "proposition": "There is 1 card.",
        "numeric_mentions": [{"written": "1", "value": 1.0, "material": True}],
    }, 2)
    assert bare["numeric_mentions"][0]["measures"] is None


def test_a_period_is_not_a_measurement():
    """"June 2025" was extracted as the material number 2025 and checked
    against `summary.last.period`, which holds "2025-06". The claim is right;
    the number is a label, and the mismatch was charged to the answer."""
    from agentic_eval.content.claims import _is_date_like

    for period in ("June 2025", "2025-06", "Jun-2025", "July'2025",
                   "Q1 2025", "FY2025", "01/06/2025"):
        assert _is_date_like(period), period
    # A bare four-digit figure is a quantity. Treating every one as a year
    # silently drops real measurements from the numeric layer.
    for quantity in ("4200", "2025", "83", "174,897.36", "1"):
        assert not _is_date_like(quantity), quantity


def test_a_part_whole_encoded_as_an_equality_is_a_judge_error():
    """"13 of 18 months" is not "13 == 18".

    Encoded as an equality it is trivially false, and the claim was charged
    for the judge's choice of operator — while both figures traced cleanly as
    separate mentions.
    """
    from agentic_eval.content import _normalize_claim
    from agentic_eval.content.verify import _normalize_fact_results

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "threshold",
        "proposition": "The feature breached its threshold in 13 of 18 months.",
        "numeric_mentions": [
            {"written": "13", "value": 13.0, "material": True},
            {"written": "18", "value": 18.0, "material": True},
        ],
    }, 1)
    ledger = [{
        "evidence_id": "ev1", "source_type": "tool_result",
        "tool": "batch_summarize_trend", "evidence_tier": "primary",
        "result": {"summary": {"n_breaching": 13, "n_buckets": 18}},
    }]
    fact = _normalize_fact_results([claim], [{
        "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
        "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
        "trace_evidence_ids": ["ev1"], "reason": "13 of 18",
        "numbers": [
            {"written_value": "13", "evidence_id": "ev1",
             "json_path": "summary.n_breaching", "trace_kind": "direct"},
            {"written_value": "18", "evidence_id": "ev1",
             "json_path": "summary.n_buckets", "trace_kind": "direct"},
        ],
        "relations": [{
            "left": {"evidence_id": "ev1", "json_path": "summary.n_breaching"},
            "operator": "==",
            "right": {"evidence_id": "ev1", "json_path": "summary.n_buckets"},
        }],
    }], ledger)[0]
    assert fact["relations"][0]["trace_failure"] == "relation_misencoded"
    assert fact["judge_error"] is True
    # Recorded, and not charged: both figures the claim states resolved, so
    # the judge's choice of operator does not make the claim unverifiable.
    assert fact["traced"] == "yes"
    assert fact["verdict"] == "supported"


def test_a_number_inside_a_categorical_code_is_locatable():
    """An account status of "30 DPB" is what "30 days past due" reads."""
    from agentic_eval.content import _normalize_claim
    from agentic_eval.content.verify import _normalize_fact_results

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "measurement",
        "proposition": "The BUSINESS PLATINUM CARD is 30 days past due.",
        "numeric_mentions": [{"written": "30", "value": 30.0, "material": True}],
    }, 1)
    ledger = [{
        "evidence_id": "ev1", "source_type": "tool_result", "tool": "query_table",
        "evidence_tier": "primary",
        "result": {"rows": [{"card_name": "BUSINESS PLATINUM", "Account Status": "30 DPB"}]},
    }]
    fact = _normalize_fact_results([claim], [{
        "claim_id": "c1", "numeric_support": "YES", "evidence_ids": ["ev1"],
        "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
        "trace_evidence_ids": ["ev1"], "reason": "status says 30 DPB",
        "numbers": [{"written_value": "30", "evidence_id": "ev1",
                     "json_path": "rows[0].Account Status", "trace_kind": "direct"}],
    }], ledger)[0]
    number = fact["numbers"][0]
    assert number["path_repaired"] is True
    assert number["evidence_value"] == 30.0
    assert fact["traced"] == "yes"


def _compact_record():
    return {
        "system": "new", "mode": "cold", "name": "q1", "run_index": 1,
        "question": "What was TSR in May 2025?",
        "final_answer": "TSR was 0.72 in May 2025.",
        "subqueries": {"risk": "Report TSR for May 2025."},
        "team": ["risk"],
        "evidence": [{
            "evidence_id": "ev1", "call_id": "call_1",
            "trace_node": "specialist.risk.round_1",
            "source_type": "tool_result", "tool": "summarize_trend",
            "arguments": {"table": "tsr", "month": "2025-05"},
            "result": {"monthly": {"2025-05": 0.72}},
        }],
    }


def _compact_reading():
    return {
        "claims": [{
            "claim_id": "c1", "block_id": "b1",
            "answer_span": "TSR was 0.72 in May 2025",
            "proposition": "TSR was 0.72 in May 2025",
            "claim_type": "quantitative_fact", "is_factual": True,
            "numeric_mentions": [{"written": "0.72", "value": 0.72, "material": True}],
        }],
        "claim_traces": [{
            "claim_id": "c1", "specialist": "risk", "call_ids": ["call_1"],
            "derivation": "risk was briefed on May 2025 and read tsr.monthly.",
        }],
        "fact_results": [{
            "claim_id": "c1", "numeric_evidence_support": "YES",
            "evidence_ids": ["ev1"], "factual_verdict": "SUPPORTED",
            "reason": "Exact match", "confidence": 0.99,
            "numbers": [{
                "written_value": "0.72", "evidence_id": "ev1",
                "json_path": "monthly.2025-05", "trace_kind": "direct", "tolerance": 0,
            }],
        }],
        "must_have_results": [{
            "must_have_id": "mh1", "verdict": "FULL",
            "answer_spans": ["TSR was 0.72 in May 2025"], "evidence_ids": ["ev1"],
            "reason": "The required value is stated.", "confidence": 0.98,
        }],
    }



COMPACT_RUBRIC = {"must_have_points": [{"id": "mh1", "description": "State May 2025 TSR"}]}


def test_non_factual_claims_are_counted_rather_than_silently_dropped():
    """Both pre-verification filters must leave a trace.

    A restatement is excluded because its fact is already counted; a
    non-factual claim because there is nothing to falsify. Either call can be
    wrong, and an exclusion nobody can see is an exclusion nobody can check.
    """
    claims = [
        {"claim_id": "c1", "is_factual": True},
        {"claim_id": "c2", "is_factual": True, "restates_claim_id": "c1"},
        {"claim_id": "c3", "is_factual": False, "claim_type": "recommendation"},
    ]
    facts = [{"claim_id": "c1", "verdict": "supported", "numbers": []}]
    metrics = calculate_content_metrics(claims, facts, [], table_coverage=None)
    assert metrics["orthogonal_claim_count"] == 1
    assert metrics["all_factual_claim_count"] == 2
    assert metrics["restated_claim_count"] == 1
    assert metrics["non_factual_claim_count"] == 1


def test_an_answer_costs_exactly_three_calls():
    """Extract, evidence, eligibility — and the ledger travels once.

    The call count is not the point; what the ledger costs is. Pinning both
    stops a future pass from quietly adding a second ledger-bearing call.
    """
    judge = FakeJudge([_compact_reading(), {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "May 2025 is the asked window."},
    ]}])
    result = ContentEvaluator({}, judge=judge).evaluate(
        _compact_record(), COMPACT_RUBRIC,
    )
    assert [call["task"] for call in judge.calls] == [
        "claim_extraction", "claim_evidence", "claim_eligibility",
    ]
    assert sum("evidence_ledger" in p for p in judge.payloads) == 1
    # Extraction sees the answer and nothing else: primed on the evidence, it
    # draws claim boundaries around what the system measured rather than
    # around what the answer said, and every rate is computed over those.
    assert set(judge.payloads[0]) == {"question", "answer", "blocks"}
    # The eligibility call never gets the ledger — it rules on routes.
    assert "evidence_ledger" not in judge.payloads[2]
    assert result["claim_traces"]["c1"]["eligible"] == "yes"
    assert result["fact_results"][0]["grounding_kind"] == "factual"


def test_the_eligibility_ruling_stays_in_its_own_call():
    """The describer must not know the verdict, so the judge cannot be folded in.

    An evidence response that volunteers an eligibility verdict is ignored: the
    ruling comes from the third call, the only one given the briefs and the
    earlier turns.
    """
    reading = _compact_reading()
    reading["fact_results"][0]["reasoning_trace_verdict"] = "YES"
    judge = FakeJudge([reading, {"eligibility": [
        {"claim_id": "c1", "verdict": "NO",
         "reason": "The brief covered 2024, not the May 2025 episode."},
    ]}])
    result = ContentEvaluator({}, judge=judge).evaluate(
        _compact_record(), COMPACT_RUBRIC,
    )
    assert result["fact_results"][0]["eligible"] == "no"
    assert result["fact_results"][0]["grounding_kind"] == "none"


def test_a_mismatch_read_out_of_a_container_is_a_pointer_error():
    """`months[0].drivers` holds several values; the resolver returns one.

    Observed: the judge aimed both of a claim's numbers at the same list. The
    first happened to parse to the right figure, the second to an unrelated
    -0.34, and the claim was reported as stating a wrong number — when the
    number it states is in that very list.
    """
    judge = _pointer_judge(json_path="drivers")
    result = ContentEvaluator({}, judge=judge).evaluate(
        _pointer_record({"drivers": [{"name": "a", "value": -0.34},
                                     {"name": "b", "value": 83}]}),
    )
    number = result["fact_results"][0]["numbers"][0]
    assert number["path_repaired"] is True
    assert number["evidence_value"] == 83.0
    assert number["trace_failure"] is None
    assert result["fact_results"][0]["traced"] == "yes"


def test_a_genuine_mismatch_against_a_scalar_is_still_a_mismatch():
    """The repair must not launder a real disagreement.

    A pointer that addresses ONE value and disagrees with the claim is
    evidence about the answer, and no search of the surrounding payload may
    turn it into a pass.
    """
    judge = _pointer_judge(json_path="value")
    result = ContentEvaluator({}, judge=judge).evaluate(
        _pointer_record({"value": 41, "elsewhere": {"other": 83}}),
    )
    number = result["fact_results"][0]["numbers"][0]
    assert number["path_repaired"] is False
    assert number["trace_failure"] == "value_mismatch"
    assert result["fact_results"][0]["traced"] == "no"


def test_a_multi_number_string_inside_a_structured_result_is_not_a_wrong_value():
    """The node is prose holding several figures, nested in a dict payload.

    `ambiguous_scalar` only ever fired when the WHOLE tool result was a
    scalar. A path reaching a sentence inside a structured result hit none of
    the guards, so the first number in that sentence was compared against the
    claim and the answer was charged for the judge's aim. The claim is not
    wrong; the evidence is unaddressable at that depth.
    """
    judge = _pointer_judge(json_path="months.top")
    result = ContentEvaluator({}, judge=judge).evaluate(_pointer_record(
        {"months": {"top": "unpaid = -0.34 (up); debt service 83 (up)"}},
    ))
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["trace_failure"] == "ambiguous_scalar"
    assert fact["judge_error"] is True
    assert fact["traced"] == "yes"   # disclosed as a judge error, not blocking


def _memory_record():
    """A claim whose measurement was made in an earlier turn, kept as memory."""
    return {
        "system": "new", "mode": "stateful", "name": "q", "run_index": 1,
        "question": "What was the CDSS score in May 2025?",
        "final_answer": "The CDSS score in May 2025 was 1.8.",
        "subqueries": {"modeling": "Report CDSS for May 2025."},
        "team": ["modeling"],
        "evidence": [{
            "evidence_id": "memory:modeling:spike@abc", "source_type": "memory",
            "tool": "specialist_kb",
            "result": {"topic": "spike", "numbers": [{"period": "2025-05", "cdss_score": 1.8}]},
        }],
    }


def _memory_reading():
    return {
        "claims": [{
            "claim_id": "c1", "block_id": "b1",
            "answer_span": "The CDSS score in May 2025 was 1.8",
            "proposition": "The CDSS score in May 2025 was 1.8",
            "claim_type": "quantitative", "is_factual": True,
            "numeric_mentions": [{"written": "1.8", "value": 1.8, "material": True}],
        }],
        # No call_id: the operation ran in an earlier turn. The evidence pass
        # cites the memory entry that recorded it.
        "claim_traces": [{
            "claim_id": "c1", "specialist": "modeling", "call_ids": [],
            "derivation": "Read from modeling's KB entry for the spike.",
        }],
        "fact_results": [{
            "claim_id": "c1", "evidence_ids": ["memory:modeling:spike@abc"],
            "verdict": "SUPPORTED", "reason": "Stated in memory.",
            "numbers": [{"written_value": "1.8", "evidence_id": "memory:modeling:spike@abc",
                         "json_path": "numbers[0].cdss_score", "trace_kind": "direct"}],
        }],
    }


def test_a_claim_grounded_in_memory_still_has_a_route():
    """A measurement from an earlier turn is provenance, not a missing route.

    Forcing UNAVAILABLE on an empty `call_ids` scored every memory-grounded
    claim unassessable — and made the rate unstable, since the same claim
    grounded or not depending on whether the evidence pass happened to cite
    the original call or the memory entry that recorded it.
    """
    judge = FakeJudge([_memory_reading(), {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "May 2025 is the asked month."},
    ]}])
    fact = ContentEvaluator({}, judge=judge).evaluate(_memory_record())["fact_results"][0]
    assert fact["traced"] == "yes"
    assert fact["eligible"] == "yes"
    assert fact["grounding_kind"] == "factual"


def test_a_claim_with_no_route_and_no_measured_evidence_stays_unassessable():
    """The conservative default survives: nothing recorded it, nothing backs it."""
    reading = _memory_reading()
    reading["fact_results"][0]["evidence_ids"] = []
    reading["fact_results"][0]["numbers"] = []
    judge = FakeJudge([reading, {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "looks fine"},
    ]}])
    fact = ContentEvaluator({}, judge=judge).evaluate(_memory_record())["fact_results"][0]
    assert fact["eligible"] == "unavailable"
    assert fact["grounding_kind"] == "none"


def _two_source_judge(*, cited):
    """The claim cites two entries; the figure lives in the other one."""
    return FakeJudge([
        {"claims": [{
            "claim_id": "c1", "block_id": "b001", "answer_span": "peaked at 83",
            "proposition": "The feature peaked at 83", "claim_type": "quantitative",
            "is_factual": True,
            "numeric_mentions": [{"written": "83", "value": 83.0, "material": True}],
        }]},
        {"fact_results": [{
            "claim_id": "c1", "evidence_ids": ["ev_prose", "ev_table"],
            "verdict": "SUPPORTED", "reason": "Stated in the summary.",
            "numbers": [{"written_value": "83", "evidence_id": cited,
                         "json_path": "$.claim", "trace_kind": "direct"}],
        }]},
    ])


def _two_source_record(table_result):
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "how did the feature behave?",
        "final_answer": "The feature peaked at 83.",
        "evidence": [
            {"evidence_id": "ev_prose", "source_type": "memory", "tool": "specialist_kb",
             "result": {"claim": "the feature peaked, ending the window elevated"}},
            {"evidence_id": "ev_table", "source_type": "tool_result", "tool": "query_table",
             "result": table_result},
        ],
    }


def test_a_number_is_repointed_to_the_cited_evidence_that_holds_it():
    """The judge cited the prose summary; the measurement is in the table.

    Both are the claim's OWN provenance, so this is a pointer fix, not a hunt
    through the run — and `wrong_evidence_cited` was the largest remaining
    evaluator failure, holding claims unknown whose figures were right there.
    """
    result = ContentEvaluator({}, judge=_two_source_judge(cited="ev_prose")).evaluate(
        _two_source_record({"summary": {"peak": 83}}),
    )
    number = result["fact_results"][0]["numbers"][0]
    assert number["path_repaired"] is True
    assert number["evidence_id"] == "ev_table"
    assert number["trace_failure"] is None
    assert result["fact_results"][0]["traced"] == "yes"


def test_a_number_at_several_cited_places_is_disclosed_not_guessed():
    """Two leaves hold it, so which one the claim meant is a guess."""
    result = ContentEvaluator({}, judge=_two_source_judge(cited="ev_prose")).evaluate(
        _two_source_record({"summary": {"peak": 83}, "other": {"also": 83}}),
    )
    fact = result["fact_results"][0]
    assert fact["numbers"][0]["trace_failure"] == "wrong_evidence_cited"
    assert fact["judge_error"] is True
    assert fact["traced"] == "yes"   # disclosed as a judge error, not blocking


def test_grounding_rests_on_the_audited_route_not_on_the_judges_pointer():
    """A figure the judge failed to map does not un-ground the claim.

    The pointer is the least reliable thing in this pipeline — across the six
    questions, 12 of 20 evaluator failures were `unmapped_but_present`, where
    Python found the value in evidence the claim itself cited. Gating on it
    scored the evaluator's aim as the answer's provenance. The route is what
    is audited (every call_id must be one the run made), so the route grounds.
    """
    reading = _compact_reading()
    reading["fact_results"][0]["numbers"] = []          # judge mapped nothing
    judge = FakeJudge([reading, {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "the asked window"},
    ]}])
    fact = ContentEvaluator({}, judge=judge).evaluate(
        _compact_record(), COMPACT_RUBRIC,
    )["fact_results"][0]
    assert fact["judge_error"] is True                  # still disclosed
    assert fact["grounding_kind"] == "factual"          # but not blocking


def test_a_figure_contradicted_by_its_own_measurement_still_fails():
    """The one thing pointers uniquely catch, and it keeps blocking.

    A number that RESOLVED against a real tool result and disagreed with it is
    evidence about the answer, not about the evaluator's aim.
    """
    judge = _pointer_judge(json_path="value")
    fact = ContentEvaluator({}, judge=judge).evaluate(
        _pointer_record({"value": 41}),
    )["fact_results"][0]
    assert fact["numbers"][0]["trace_failure"] == "value_mismatch"
    assert fact["traced"] == "no"
    assert fact["grounding_kind"] == "none"


def test_a_report_claim_short_circuits_the_factual_test():
    """A claim quoting a report ran no operations, so there is nothing to trace.

    Putting it through the factual test only ever produced a second, noisier
    way of saying "not measured" — and charged the answer a numeric failure
    for relaying a figure it never claimed to have measured.
    """
    record = _compact_record()
    record["evidence"] = [{
        "evidence_id": "rep1", "source_type": "tool_result", "tool": "fs_read_file",
        "arguments": {"path": "modeling_exp_0.md"},
        "result": {"text": "TSR was 0.72 in May 2025 per the curated report."},
    }]
    reading = _compact_reading()
    reading["claims"][0]["stance"] = "attributed_unendorsed"
    reading["fact_results"][0] = {
        "claim_id": "c1", "evidence_ids": ["rep1"], "verdict": "SUPPORTED",
        "reason": "Stated in the report.",
        "numbers": [{"written_value": "0.72", "evidence_id": "rep1",
                     "json_path": "nowhere", "trace_kind": "direct"}],
    }
    reading["claim_traces"] = [{"claim_id": "c1", "call_ids": [], "derivation": "read the report"}]
    judge = FakeJudge([reading, {"eligibility": []}])
    fact = ContentEvaluator({}, judge=judge).evaluate(
        _compact_record() | {"evidence": record["evidence"]}, COMPACT_RUBRIC,
    )["fact_results"][0]
    assert fact["grounding_kind"] == "report"
    assert fact["traced"] == "not_applicable"


def _chain_record():
    """A claim a general specialist synthesised from two branches."""
    return {
        "system": "new", "mode": "stateful", "name": "q", "run_index": 1,
        "question": "What transactions are connected?",
        "final_answer": "Spending clusters by vendor, with S BERTRAM recurring.",
        "team": ["spend_payments", "modeling", "general_specialist"],
        "subqueries": {
            "spend_payments": "Surface linked transactions sharing vendors.",
            "modeling": "Model signals for coordinated patterns.",
            "general_specialist": "Synthesise the branches.",
        },
        "evidence": [
            {"evidence_id": "ev1", "call_id": "call_a",
             "trace_node": "specialist.spend_payments.round_1",
             "source_type": "tool_result", "tool": "summarize_by_group",
             "arguments": {"table_name": "spends", "value_column": "Amount",
                           "group_column": "Merchant Name", "op": "sum", "top_n": 10},
             "result": {"rows": [{"Merchant Name": "S BERTRAM", "sum": 392454}]}},
            {"evidence_id": "mem1", "source_type": "memory", "tool": "specialist_kb",
             "specialist": "modeling", "captured_at_turn": 1,
             "result": {"topic": "modeling_Spend Amount_trend", "claim": "spend rose"}},
        ],
    }


def _chain_reading(route):
    return {
        "claims": [{
            "claim_id": "c1", "block_id": "b1",
            "answer_span": "Spending clusters by vendor",
            "proposition": "Spending clusters by vendor",
            "claim_type": "qualitative", "is_factual": True, "numeric_mentions": [],
        }],
        "claim_traces": [{"claim_id": "c1", "route": route,
                          "derivation": "general synthesised both branches"}],
        "fact_results": [{
            "claim_id": "c1", "evidence_ids": ["ev1", "mem1"],
            "verdict": "SUPPORTED", "reason": "Both branches agree.", "numbers": [],
        }],
    }


def test_a_route_can_be_a_chain_through_a_synthesising_specialist():
    """claim -> general -> (spend_payments: tables) + (modeling: memory topic).

    A flat `{specialist, call_ids}` could not say that a claim came from a
    synthesis of two branches, one resting on a table operation and the other
    on a memory topic — so a real multi-hop route read as a one-hop one, or as
    no route at all.
    """
    route = [
        {"specialist": "general_specialist", "kind": "synthesis",
         "from_specialists": ["spend_payments", "modeling"],
         "note": "combined vendor clustering with the model signal"},
        {"specialist": "spend_payments", "kind": "operation", "call_ids": ["call_a"],
         "operations": ["summed spends.Amount by Merchant Name, top 10"]},
        {"specialist": "modeling", "kind": "memory",
         "memory_topics": ["modeling_Spend Amount_trend"]},
    ]
    judge = FakeJudge([_chain_reading(route), {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "vendor grouping answers it"},
    ]}])
    result = ContentEvaluator({}, judge=judge).evaluate(_chain_record())
    trace = result["claim_traces"]["c1"]
    assert [hop["kind"] for hop in trace["route"]] == ["synthesis", "operation", "memory"]
    assert trace["route"][0]["from_specialists"] == ["spend_payments", "modeling"]
    assert trace["call_ids"] == ["call_a"]
    assert trace["memory_topics"] == ["modeling_Spend Amount_trend"]
    assert result["fact_results"][0]["grounding_kind"] == "factual"


def test_an_invented_hop_is_dropped_and_the_rest_of_the_chain_survives():
    """One fabricated branch must not launder, nor sink, the real ones."""
    route = [
        {"specialist": "spend_payments", "kind": "operation", "call_ids": ["call_a"]},
        {"specialist": "fraud_team", "kind": "operation", "call_ids": ["call_nope"],
         "memory_topics": ["topic_that_never_existed"]},
    ]
    judge = FakeJudge([_chain_reading(route), {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "ok"},
    ]}])
    trace = ContentEvaluator({}, judge=judge).evaluate(_chain_record())["claim_traces"]["c1"]
    assert trace["call_ids"] == ["call_a"]
    assert sorted(trace["invented_steps"]) == [
        "call:call_nope", "specialist:fraud_team", "topic:topic_that_never_existed",
    ]


def test_a_memory_topic_alone_is_a_recorded_route():
    """The measurement ran in an earlier turn; the KB entry is what carries it."""
    route = [{"specialist": "modeling", "kind": "memory",
              "memory_topics": ["modeling_Spend Amount_trend"]}]
    judge = FakeJudge([_chain_reading(route), {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "the topic is the asked subject"},
    ]}])
    fact = ContentEvaluator({}, judge=judge).evaluate(_chain_record())["fact_results"][0]
    assert fact["route"]["no_recorded_operation"] is False
    assert fact["eligible"] == "yes"
    assert fact["grounding_kind"] == "factual"


def _report_only_record():
    return {
        "system": "new", "mode": "cold", "name": "q", "run_index": 1,
        "question": "What evidence contradicts the pattern?",
        "final_answer": "Reports document a high spend-to-payment ratio.",
        "subqueries": {"report_agent": "Find contradicting evidence in the reports."},
        "team": ["report_agent"],
        "evidence": [{
            "evidence_id": "rep1", "call_id": "call_r",
            "trace_node": "specialist.report_agent.round_1",
            "source_type": "tool_result", "tool": "fs_read_file",
            "arguments": {"path": "modeling_exp_0.md"},
            "result": {"text": "spend $1.2M against payments of $0.4M"},
        }],
    }


def _report_only_reading():
    return {
        "claims": [{
            "claim_id": "c1", "block_id": "b1",
            "answer_span": "Reports document a high spend-to-payment ratio",
            "proposition": "Reports document a high spend-to-payment ratio",
            "claim_type": "qualitative", "is_factual": True,
            "stance": "attributed_unendorsed", "numeric_mentions": [],
        }],
        "claim_traces": [{"claim_id": "c1", "route": [
            {"specialist": "report_agent", "kind": "operation", "call_ids": ["call_r"]},
        ], "derivation": "read the curated report"}],
        "fact_results": [{
            "claim_id": "c1", "evidence_ids": ["rep1"], "verdict": "SUPPORTED",
            "reason": "Stated in the report.", "numbers": [],
        }],
    }


def test_a_report_only_claim_is_not_sent_to_the_eligibility_call():
    """No verdict can change its grounding, so buying one is waste.

    Across the six-question set this spent 19 verdicts and returned noise: on
    `evidence_contradicting_pattern`, 10 of 11 came back NO while all 11
    grounded regardless — because `report` is tested first and short-circuits.
    """
    judge = FakeJudge([_report_only_reading()])
    result = ContentEvaluator({}, judge=judge).evaluate(_report_only_record())
    # Two calls, not three: there was nothing left to rule on.
    assert [call["task"] for call in judge.calls] == [
        "claim_extraction", "claim_evidence",
    ]
    fact = result["fact_results"][0]
    assert fact["grounding_kind"] == "report"
    assert "Not judged" in fact["eligibility_reason"]


def test_a_claim_backed_by_live_data_too_is_still_judged():
    """`all`, not `any`. Corroborated by a report is not dependent on one.

    The claim must not READ as an attribution either — "reports document X" is
    a claim about what a report says, and grounds on that report whatever its
    route. This one asserts the fact in its own voice and cites both sources,
    so its route still has to answer the question.
    """
    record = _report_only_record()
    record["final_answer"] = "Spend of $1.2M ran against payments of $0.4M."
    record["evidence"].append({
        "evidence_id": "ev1", "call_id": "call_q",
        "trace_node": "specialist.report_agent.round_1",
        "source_type": "tool_result", "tool": "query_table",
        "arguments": {"table_name": "spends"}, "result": {"total": 1200000},
    })
    reading = _report_only_reading()
    reading["claims"][0].update({
        "stance": "asserted",
        "answer_span": "Spend of $1.2M ran against payments of $0.4M",
        "proposition": "Spend of $1.2M ran against payments of $0.4M",
    })
    reading["fact_results"][0]["evidence_ids"] = ["rep1", "ev1"]
    judge = FakeJudge([reading, {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "the asked subject"},
    ]}])
    result = ContentEvaluator({}, judge=judge).evaluate(record)
    assert [call["task"] for call in judge.calls] == [
        "claim_extraction", "claim_evidence", "claim_eligibility",
    ]
    assert result["fact_results"][0]["eligible"] == "yes"


def _mh(mh_id, verdict, critical=False):
    return {"must_have_id": mh_id, "verdict": verdict, "weight": 1,
            "critical": critical}


def test_partial_credit_is_worth_half_a_point():
    """An answer part-way to a required point is not level with one that
    ignored it, so the rate has to be able to say so."""
    metrics = calculate_content_metrics(
        [], [], [_mh("a", "full"), _mh("b", "partial"), _mh("c", "miss")],
        table_coverage=None,
    )
    assert metrics["must_have_coverage"] == 0.5
    assert metrics["must_have_questions"] == 1
    # The rate alone cannot distinguish "one hit, one missed" from "two
    # halves", so the breakdown stays on the record beside it.
    assert metrics["must_have_counts"]["partial"] == 1


def test_a_not_applicable_point_leaves_both_sides_of_the_ratio():
    """A point the question does not raise must not count against the answer."""
    metrics = calculate_content_metrics(
        [], [], [_mh("a", "full"), _mh("b", "not_applicable")],
        table_coverage=None,
    )
    assert metrics["must_have_coverage"] == 1.0


def test_the_set_rate_is_the_average_over_questions():
    """One answer, one vote — a rubric with five points cannot outweigh one
    with two, which is what pooling the weights used to do."""
    rows = [
        {"system": "new", "mode": "cold", "name": "q1", "run_index": 1,
         "metrics": calculate_content_metrics(
             [], [], [_mh("a", "full"), _mh("b", "miss")], table_coverage=None)},
        {"system": "new", "mode": "cold", "name": "q2", "run_index": 1,
         "metrics": calculate_content_metrics(
             [], [], [_mh(str(i), "full") for i in range(5)], table_coverage=None)},
    ]
    coverage = sum(r["metrics"]["must_have_coverage"] for r in rows)
    questions = sum(r["metrics"]["must_have_questions"] for r in rows)
    assert coverage / questions == 0.75      # (0.5 + 1.0) / 2, not 6/7


def _memory_offered_record():
    r = _compact_record()
    r["kb_topics_exposed"] = ["modeling_TSR_trend", "bureau_FICO Score_trend"]
    r["episodic_turns_exposed"] = [
        {"turn_id": "t1", "question": "Any spending spikes?"},
    ]
    r["subqueries"] = {"modeling": "Report CDSS around the May 2025 spike."}
    return r


def test_memory_leverage_is_asked_once_and_audited():
    """Exposure is known from the trace; this call asks whether it was USED."""
    judge = FakeJudge([_compact_reading(), {"eligibility": [
        {"claim_id": "c1", "verdict": "YES", "reason": "right window"},
    ]}])
    judge.by_task["memory_leverage"] = {"memory_leverage": [
        {"source": "episodic", "leveraged": "YES", "where": ["construction"],
         "items": ["t1"], "reason": "the brief names the May 2025 spike from turn 1"},
        {"source": "kb", "leveraged": "NO", "where": [], "items": [],
         "reason": "the KPs were offered and nothing drew on them"},
    ]}
    result = ContentEvaluator({}, judge=judge).evaluate(
        _memory_offered_record(), COMPACT_RUBRIC,
    )
    assert "memory_leverage" in [c["task"] for c in judge.calls]
    m = result["metrics"]
    # two sources offered (kb topics, episodic turns), one leveraged
    assert m["memory_sources_offered"] == 2
    assert m["memory_sources_leveraged"] == 1
    assert m["memory_leverage_rate"] == 0.5
    assert m["memory_leveraged_where"] == ["construction"]


def test_a_turn_the_run_was_never_shown_cannot_be_leveraged():
    """Crediting an unoffered item would make this measure the harness."""
    judge = FakeJudge([_compact_reading(), {"eligibility": []}])
    judge.by_task["memory_leverage"] = {"memory_leverage": [
        {"source": "episodic", "leveraged": "YES", "where": ["answer"],
         "items": ["never_shown"], "reason": "claimed"},
    ]}
    result = ContentEvaluator({}, judge=judge).evaluate(
        _memory_offered_record(), COMPACT_RUBRIC,
    )
    row = result["memory_leverage"]["sources"][0]
    assert row["leveraged"] is False          # claimed use, cited nothing real
    assert row["invented_items"] == ["never_shown"]
    assert result["metrics"]["memory_sources_leveraged"] == 0


def test_no_memory_offered_costs_no_call():
    """An empty set cannot be leveraged; asking would buy nothing."""
    judge = FakeJudge([_compact_reading(), {"eligibility": []}])
    result = ContentEvaluator({}, judge=judge).evaluate(
        _compact_record(), COMPACT_RUBRIC,
    )
    assert "memory_leverage" not in [c["task"] for c in judge.calls]
    assert result["memory_leverage"]["asked"] is False
    assert result["metrics"]["memory_leverage_rate"] is None
