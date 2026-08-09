"""Tool-call success: did the call come back with anything?"""
from __future__ import annotations

import json

from agentic_eval import toolcalls
from agentic_eval.dimensions.consistency import _tool_call_signature
from agentic_eval.scoring import aggregate


def _row(system, run_index, results):
    return {
        "system": system, "mode": "cold", "name": "q", "run_index": run_index,
        "outcome": "ok", "elapsed_seconds": 1.0, "team_unique": [], "tools": [],
        "subqueries": {}, "evidence": [
            {"source_type": "tool_result", "tool": "summarize_trend",
             "arguments": {"value_column": col}, "result": res}
            for col, res in results
        ],
    }


def test_a_structured_payload_means_the_call_ran():
    assert toolcalls.outcome({"rows_in_range": 26}) == "data"
    # JSON text, as the batch tools nest it.
    assert toolcalls.outcome(json.dumps({"rows_in_range": 3})) == "data"


def test_a_zero_result_is_a_measurement_not_a_failure():
    """`a1` asks "were there any payment returns?" and the answer is none.

    The query scanned 357 payments and matched 0 — it ran perfectly. Scoring
    that as a failure put BOTH systems at 0% success on that question.
    """
    assert toolcalls.outcome({"rows_in_range": 0}) == "data"
    assert toolcalls.outcome({
        "table": "payments_data", "filter": "Return Flag eq '1'",
        "total_rows_in_table": 357, "rows_matching_filter": 0,
        "rows_returned": 0, "rows": [],
    }) == "data"


def test_prose_that_says_it_found_nothing_counts_as_empty():
    """The real message blames the dates for a column that never resolved."""
    assert toolcalls.outcome(
        "trend(max(derog_count) by month on month) = (no parseable month "
        "values; 26 total in bureau_data; 0 row(s) had unrecognized month format)"
    ) == "empty"


def test_an_unreadable_result_is_unknown_not_success():
    """A rate padded with calls we could not judge is worth less than one
    that says how many it could."""
    assert toolcalls.outcome(None) == "unknown"
    assert toolcalls.outcome("some prose with no verdict in it") == "unknown"


def test_the_failures_the_tools_actually_emit_are_caught():
    """Verbatim from the run — all three are genuine, none is a zero result."""
    for text in (
        "trend(max(derog_count) by month on month) = (no parseable month "
        "values in date range 2025-01-01..2026-12-31; 26 total)",
        "trend(sum(derog_count) by month on month) = (COLUMN NOT FOUND: "
        "'derog_count' is not a column of 'bureau_data' for this case)",
        "File not found: payment_spend_exp_0.md",
    ):
        assert toolcalls.outcome(text) == "empty", text[:40]


def test_a_batch_is_counted_per_spec():
    """Counting a batch as one call would hide that 3 of its 4 came back
    empty — which is exactly what happened on the real run."""
    batch = {
        "source_type": "tool_result", "tool": "batch_summarize_trend",
        "arguments": {"specs_json": "[]"},
        "result": {"results": [
            {"index": 0, "value_column": "fico_score",
             "result": json.dumps({"rows_in_range": 26})},
            {"index": 1, "value_column": "derog_count",
             "result": "trend(...) = (no parseable month values; 26 total)"},
            {"index": 2, "value_column": "lien_org",
             "result": "trend(...) = (no parseable month values; 26 total)"},
        ]},
    }
    assert toolcalls.counts([{"evidence": [batch]}]) == {
        "data": 1, "empty": 2, "unknown": 0,
    }


def test_success_rate_reaches_the_system_section():
    rows = [
        _row("old", 1, [("a", {"rows_in_range": 5}),
                        ("b", "no parseable month values")]),
        _row("old", 2, [("a", {"rows_in_range": 5}),
                        ("b", "no parseable month values")]),
    ]
    group = aggregate(rows, modules=["latency"])["groups"][0]
    assert group["tool_call_success_rate"] == 0.5
    assert group["tool_calls_with_data"] == 2
    assert group["tool_calls_empty"] == 2
    assert group["tool_calls_unreadable"] == 0


def test_no_judgeable_call_is_undefined_not_zero():
    rows = [_row("old", 1, [("a", None)]), _row("old", 2, [("a", None)])]
    group = aggregate(rows, modules=["latency"])["groups"][0]
    assert group["tool_call_success_rate"] is None
    assert group["tool_calls_unreadable"] == 2


def _call(tool, args, result):
    return {"source_type": "tool_result", "tool": tool,
            "arguments": args, "result": result}


def _consistency_row(system, run_index, calls, case_id="c1"):
    return {
        "system": system, "mode": "cold", "name": "q", "run_index": run_index,
        "case_id": case_id, "outcome": "ok", "elapsed_seconds": 1.0,
        "team_unique": [], "tools": [], "subqueries": {}, "evidence": calls,
    }


def test_consistency_counts_every_call_working_or_not():
    """Consistency asks whether the system did the same thing twice.

    Whether the thing WORKED is a separate question, answered by
    `tool_call_success_rate` under System. Filtering failures out here would
    also let a system that resolved almost nothing score highly on the handful
    that did, so the two metrics stay independent and are read together.
    """
    ok = _call("summarize_trend", {"value_column": "fico"}, {"rows_in_range": 26})
    dud = _call("summarize_trend", {"value_column": "derog_count"},
                "trend(...) = (no parseable month values; 26 total)")
    group = aggregate(
        [_consistency_row("old", 1, [ok, dud]), _consistency_row("old", 2, [ok, dud])],
        modules=["consistency"],
    )["groups"][0]
    # Both calls repeated identically, so the runs agree completely...
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0
    # ...while success says half of that agreement achieved nothing.
    assert aggregate(
        [_consistency_row("old", 1, [ok, dud]), _consistency_row("old", 2, [ok, dud])],
        modules=["latency"],
    )["groups"][0]["tool_call_success_rate"] == 0.5


def test_a_dropped_call_still_lowers_consistency():
    """A failing call made in one run and not the other is a difference."""
    ok = _call("summarize_trend", {"value_column": "fico"}, {"rows_in_range": 26})
    dud = _call("summarize_trend", {"value_column": "derog_count"},
                "trend(...) = (no parseable month values; 26 total)")
    group = aggregate(
        [_consistency_row("old", 1, [ok, dud]), _consistency_row("old", 2, [ok])],
        modules=["consistency"],
    )["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 0.5


def test_two_runs_with_no_tool_calls_are_undefined_not_perfect():
    """Multiset Jaccard of two empty sets is 1.0 by definition.

    Two runs that called nothing are not evidence of consistency.
    """
    group = aggregate(
        [_consistency_row("old", 1, []), _consistency_row("old", 2, [])],
        modules=["consistency"],
    )["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] is None


def test_a_call_with_no_row_count_still_counts_as_work():
    """`kb_lookup` and `fs_list_files` report no rows; they still ran."""
    lookup = _call("kb_lookup", {"topic": "trajectory"}, "some prose")
    group = aggregate(
        [_consistency_row("old", 1, [lookup]), _consistency_row("old", 2, [lookup])],
        modules=["consistency"],
    )["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0


def test_a_failed_call_still_gets_its_alias_resolved():
    """Resolution reads the physical name from the result payload.

    A call that found nothing has no payload to read, so it kept the raw name
    the model wrote — and two runs that asked IDENTICALLY then differed on
    `table_name` purely because one succeeded. Aliases are a property of the
    catalog, so one seen resolving anywhere in the run applies everywhere.
    """
    good = _call("summarize_trend",
                 {"table_name": "bureau", "value_column": "fico_score"},
                 {"table": "bureau_data", "value_column": "FICO Score",
                  "rows_in_range": 26})
    # Same table alias, but this one resolved nothing and echoes no payload.
    dud = _call("summarize_trend",
                {"table_name": "bureau", "value_column": "derog_count"},
                "trend(...) = (no parseable month values; 26 total)")
    group = aggregate(
        [_consistency_row("old", 1, [good, dud]),
         _consistency_row("old", 2, [good, dud])],
        modules=["consistency"],
    )["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0

    from agentic_eval.dimensions.consistency import _calls, _learn_aliases
    row = _consistency_row("old", 1, [good, dud])
    assert _learn_aliases(list(_calls(row))) == {
        ("table_name", "bureau"): "bureau_data",
        ("value_column", "fico_score"): "FICO Score",
    }
    # The failed call's table is rewritten to the physical name too.
    tables = {
        json.loads(s.split(":", 1)[1])["table_name"]
        for s in _tool_call_signature(row)
    }
    assert tables == {"bureau_data"}


def test_an_ambiguous_alias_is_dropped_rather_than_guessed():
    """One alias resolving two ways is not something to pick a winner from."""
    from agentic_eval.dimensions.consistency import _calls, _learn_aliases

    a = _call("summarize_trend", {"table_name": "b"}, {"table": "bureau_data",
                                                       "rows_in_range": 1})
    b = _call("summarize_trend", {"table_name": "b"}, {"table": "other_data",
                                                       "rows_in_range": 1})
    row = _consistency_row("old", 1, [a, b])
    assert _learn_aliases(list(_calls(row))) == {}
