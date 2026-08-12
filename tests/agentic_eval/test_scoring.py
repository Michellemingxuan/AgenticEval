from agentic_eval.scoring import aggregate, compare, score_content, score_memory


def _row(system, run_index, latency, tokens):
    return {
        "system": system, "mode": "cold", "name": "q", "run_index": run_index,
        "outcome": "ok", "elapsed_seconds": latency,
        "team_unique": ["modeling"], "tools": ["summarize_trend"],
        "subqueries": {"modeling": "check tsr trend"},
        "total_tokens": tokens, "llm_call_count": 3, "self_recovered": False,
        "qa_cache_hit": False, "kb_context_exposures": 0,
        "kb_lookup_calls": None, "kb_lookup_hits": None,
        "provenance_completeness": 1.0, "automated_content_score": 90,
    }


def test_aggregate_and_candidate_minus_baseline():
    records = [
        _row("old", 1, 4.0, 100), _row("old", 2, 6.0, 120),
        _row("new", 1, 3.0, 80), _row("new", 2, 5.0, 100),
    ]
    summary = aggregate(records)
    comparison = compare(
        summary, baseline="old", candidate="new", records=records, seed=1,
    )[0]
    delta = comparison["candidate_minus_baseline"]
    assert delta["median_latency_seconds"] == -1.0
    assert delta["total_tokens_mean"] == -20.0
    assert delta["team_exact_consistency"] == 0.0
    paired = comparison["paired"]["elapsed_seconds"]
    assert paired["n_pairs"] == 2
    assert paired["mean_delta"] == -1.0
    assert paired["candidate_wins"] == 2


def test_aggregate_survives_a_single_repeat():
    """repeats=1 is a valid config (calibration/smoke), so it must not crash.

    Regression: the pairwise-similarity guard tested `subqueries` rather than
    the pair list. Every record carries subqueries, but one run per cell forms
    NO pair, so `statistics.mean([])` raised "mean requires at least one data
    point" — after all records had already been collected and written.
    """
    records = [_row("old", 1, 4.0, 100), _row("new", 1, 3.0, 80)]
    summary = aggregate(records)  # must not raise
    cells = summary.values() if isinstance(summary, dict) else summary
    assert all(
        cell.get("subquery_pairwise_similarity") is None
        for cell in cells if isinstance(cell, dict)
    )
    comparison = compare(
        summary, baseline="old", candidate="new", records=records, seed=1,
    )[0]
    assert comparison["candidate_minus_baseline"]["median_latency_seconds"] == -1.0


def test_repeated_run_distributions_outliers_retries_and_tool_arguments():
    records = []
    for run_index, latency in enumerate([1.0, 1.0, 1.0, 1.0, 10.0], 1):
        row = _row("new", run_index, latency, 100 + run_index)
        row.update({
            "prompt_tokens": 80 + run_index,
            "completion_tokens": 20,
            "self_recovery_count": 2 if run_index == 5 else 0,
            "self_recovered": run_index == 5,
            "evidence": [{
                "source_type": "tool_result", "tool": "summarize_trend",
                "arguments": {"period": "June" if run_index == 5 else "May"},
            }],
        })
        records.append(row)

    group = aggregate(records)["groups"][0]
    assert group["n_runs"] == 5
    assert group["latency_seconds"]["values"] == [1.0, 1.0, 1.0, 1.0, 10.0]
    assert group["latency_seconds"]["outlier_eligible"] is True
    assert group["latency_seconds"]["outlier_count"] == 1
    assert group["latency_seconds"]["outlier_rate"] == 0.2
    assert group["total_tokens"]["n"] == 5
    assert group["self_recovery_rate"] == 0.2
    assert group["self_recovery_runs"] == 1
    assert group["self_recovery_attempts"] == 2
    assert group["self_recovery_count"]["mean"] == 0.4
    # Tool-name usage is identical, but normalized tool-call arguments expose
    # the one run that queried the wrong period.
    assert group["tool_exact_consistency"] == 1.0
    assert group["tool_call_exact_consistency"] == 0.8


def test_memory_hit_rate_uses_only_memory_required_questions():
    records = []
    for run_index, exposed in [(1, True), (2, False)]:
        row = _row("new", run_index, 1.0, 100)
        row.update({
            "mode": "stateful", "name": "followup", "sequence_position": 2,
            "evaluation": {"memory_required": True},
            # The TURNS themselves, not a flag saying a block appeared.
            "episodic_turns_exposed": (
                [{"turn_id": "t1", "question": "prior"}] if exposed else []
            ),
            "memory_telemetry_complete": True,
        })
        row.update(score_memory(row))
        records.append(row)
    summary = aggregate(records)
    group = summary["groups"][0]
    assert group["memory_required"] is True
    assert group["memory_required_run_count"] == 2
    assert group["memory_used_count"] == 1
    assert group["memory_hit_rate"] == 0.5
    assert summary["memory_groups"][0]["memory_hit_rate"] == 0.5


def test_memory_usage_is_a_simple_boolean_signal():
    no_memory = {
        **_row("new", 1, 1.0, 100),
        "evaluation": {"memory_required": True},
    }
    used_memory = {
        **no_memory,
        "kb_topics_exposed": ["modeling_TSR_trend"],
    }
    not_required = {
        **used_memory, "evaluation": {"memory_required": False},
    }
    assert score_memory(no_memory)["memory_used"] is False
    assert score_memory(no_memory)["memory_hit"] is False
    assert score_memory(used_memory)["memory_used"] is True
    assert score_memory(used_memory)["memory_hit"] is True
    assert score_memory(not_required)["memory_hit"] is None
    # `memory_context_exposed` no longer qualifies on its own. It fires on a KB
    # header the system emits every turn, which scored a hit on the FIRST
    # question of a session — where there is nothing yet to remember — and
    # pinned the rate at 100% for both systems across a whole set.
    header_only = {**no_memory, "memory_context_exposed": True}
    assert score_memory(header_only)["memory_used"] is False


def test_content_contract_stays_system_agnostic():
    record = {
        "outcome": "ok", "final_answer": "Two payments were returned.",
        "team_unique": ["spend_payments"], "scopes": ["payments: 2025"],
        "measured_over": ["query_table(payments.status)"],
        "provenance_completeness": 1.0,
        "evaluation": {
            "expected_outcome": "ok",
            "required_specialists": ["spend_payments"],
            "required_scope_terms": ["payments", "2025"],
            "answer_must_include": ["returned"],
        },
    }
    assert score_content(record)["automated_content_score"] == 100.0


def test_module_selection_accepts_lists_commas_and_all():
    """A sweep script should not have to spell the registry out."""
    from agentic_eval.dimensions import EVAL_MODULES, resolve_modules
    every = list(EVAL_MODULES)
    assert resolve_modules("all") == every
    assert resolve_modules(None) == every
    assert resolve_modules([]) == every
    assert resolve_modules("content,latency") == ["content", "latency"]
    assert resolve_modules(["memory", "content"]) == ["content", "memory"]
    # Registry order, so summaries stay comparable across invocations.
    assert resolve_modules(["latency", "consistency"]) == ["consistency", "latency"]
    # `all` anywhere in the selection wins.
    assert resolve_modules(["memory", "all"]) == every


def test_content_is_a_selectable_module():
    rows = [_row("new", 1, 1.0, 100), _row("new", 2, 2.0, 200)]
    content_only = aggregate(rows, modules="content")
    group = content_only["groups"][0]
    assert content_only["modules"] == ["content"]
    assert group["automated_content_score"] == 90
    assert group["provenance_completeness"] == 1.0
    assert "latency_seconds" not in group


def test_eval_modules_can_be_selected_independently():
    """Each metric family stands alone, so a sweep can compute only what it needs."""
    rows = [_row("new", 1, 1.0, 100), _row("new", 2, 2.0, 200)]
    latency_only = aggregate(rows, modules=["latency"])
    group = latency_only["groups"][0]
    assert latency_only["modules"] == ["latency"]
    assert "latency_seconds" in group
    # Identity and completion survive so partial summaries stay joinable.
    assert group["system"] == "new" and group["n_runs"] == 2
    assert "team_exact_consistency" not in group
    assert "memory_hit_rate" not in group
    assert "automated_content_score" not in group
    assert latency_only["memory_groups"] == []

    consistency_only = aggregate(rows, modules=["consistency"])
    assert "team_exact_consistency" in consistency_only["groups"][0]
    assert "latency_seconds" not in consistency_only["groups"][0]


def test_unknown_eval_module_fails_loudly():
    """A typo in a sweep script must not silently skip a metric family."""
    import pytest
    with pytest.raises(ValueError, match="unknown eval module"):
        aggregate([_row("new", 1, 1.0, 100)], modules=["latancy"])
    with pytest.raises(ValueError, match="latency"):  # the error names the options
        aggregate([_row("new", 1, 1.0, 100)], modules="content,latancy")


def _case_row(system, case_id, run_index, team):
    row = _row(system, run_index, 4.0, 100)
    row["case_id"] = case_id
    row["team_unique"] = team
    return row


def test_consistency_compares_repeats_within_a_case_not_across_cases():
    """Two customers legitimately get different specialists.

    Pooling cases would score that correct behaviour as instability, and the
    more cases a run covered the less consistent every system would look.
    Within each case the team is identical here, so consistency is perfect.
    """
    records = [
        _case_row("old", "case_a", 1, ["modeling"]),
        _case_row("old", "case_a", 2, ["modeling"]),
        _case_row("old", "case_b", 1, ["spend_payments"]),
        _case_row("old", "case_b", 2, ["spend_payments"]),
    ]
    group = aggregate(records, modules=["consistency"])["groups"][0]
    assert group["n_runs"] == 4
    assert group["team_exact_consistency"] == 1.0
    assert group["team_pairwise_jaccard"] == 1.0


def test_consistency_still_catches_instability_inside_one_case():
    """The partition must not become a way to hide real variation."""
    records = [
        _case_row("old", "case_a", 1, ["modeling"]),
        _case_row("old", "case_a", 2, ["spend_payments"]),
        _case_row("old", "case_b", 1, ["modeling"]),
        _case_row("old", "case_b", 2, ["modeling"]),
    ]
    group = aggregate(records, modules=["consistency"])["groups"][0]
    # case_a is 1/2 on its modal team, case_b is 2/2 -> mean 0.75.
    assert group["team_exact_consistency"] == 0.75


def test_pairing_keeps_the_two_cases_apart():
    """Without case in the key the two cases' rows overwrite each other."""
    records = [
        _case_row("old", "case_a", 1, ["modeling"]),
        _case_row("new", "case_a", 1, ["modeling"]),
        _case_row("old", "case_b", 1, ["modeling"]),
        _case_row("new", "case_b", 1, ["modeling"]),
    ]
    summary = aggregate(records)
    comparison = compare(
        summary, baseline="old", candidate="new", records=records, seed=1,
    )[0]
    assert comparison["paired"]["elapsed_seconds"]["n_pairs"] == 2


def _evidence_row(system, run_index, tool, case_id="c1"):
    row = _case_row(system, case_id, run_index, ["modeling"])
    row["evidence"] = [{
        "source_type": "tool_result", "tool": tool,
        "arguments": {"table": "payments"},
    }]
    row["measured_over"] = []
    return row


def _measured_over_row(system, run_index, measured, case_id="c1"):
    row = _case_row(system, case_id, run_index, ["modeling"])
    row["evidence"] = []
    row["measured_over"] = [measured]
    return row


def test_tool_call_consistency_compares_only_like_with_like():
    """The three signature sources are different vocabularies.

    `fs_read_file:{...}` from the evidence ledger can never equal
    `summarize_by_group(...)` from `measured_over`, so mixing them scored a
    system at 0% for a reason with nothing to do with what it did.
    """
    records = [
        _evidence_row("old", 1, "aggregate_column"),
        _evidence_row("old", 2, "aggregate_column"),
        # A third repeat whose calls landed in a different field.
        _measured_over_row("old", 3, "summarize_by_group(payments.amount)"),
    ]
    group = aggregate(records, modules=["consistency"])["groups"][0]
    # The two comparable runs agree exactly; the odd one out is set aside.
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0
    assert group["tool_call_exact_consistency"] == 1.0
    assert group["tool_call_runs_not_comparable"] == 1
    assert group["tool_call_signature_sources"] == {"evidence": 2, "measured_over": 1}


def test_no_comparable_pair_is_undefined_not_zero():
    """One run per vocabulary supports no comparison at all.

    Reporting 0% there is a claim of total inconsistency drawn from no
    evidence; `None` renders as "—" and says the measure did not apply.
    """
    records = [
        _evidence_row("old", 1, "aggregate_column"),
        _measured_over_row("old", 2, "summarize_by_group(payments.amount)"),
    ]
    group = aggregate(records, modules=["consistency"])["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] is None
    assert group["tool_call_runs_not_comparable"] == 1


def test_a_real_tool_swap_still_scores_zero():
    """The filter must not become a way to hide genuine inconsistency.

    Two runs, same vocabulary, different tool — that is a real finding and
    must survive.
    """
    records = [
        _evidence_row("old", 1, "aggregate_column"),
        _evidence_row("old", 2, "batch_aggregate"),
    ]
    group = aggregate(records, modules=["consistency"])["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 0.0
    assert group["tool_call_runs_not_comparable"] == 0


def _batch_row(system, run_index, specs_json, case_id="c1"):
    row = _case_row(system, case_id, run_index, ["modeling"])
    row["evidence"] = [{
        "source_type": "tool_result", "tool": "batch_summarize_trend",
        "arguments": {"specs_json": specs_json},
    }]
    row["measured_over"] = []
    return row


def test_a_reformatted_json_argument_is_the_same_call():
    """`specs_json` is a JSON STRING inside the arguments.

    An LLM emits it pretty-printed one run and compact the next. `sort_keys`
    normalises the outer dict but not the nested string, so any question
    answered with a `batch_*` tool scored 0% consistent however steady the
    system was.
    """
    compact = '[{"table_name":"modelling_data","value_column":"credit_loss_prob_max"}]'
    pretty = '[\n  {"value_column": "credit_loss_prob_max", "table_name": "modelling_data"}\n]'
    group = aggregate(
        [_batch_row("old", 1, compact), _batch_row("old", 2, pretty)],
        modules=["consistency"],
    )["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0
    assert group["tool_call_exact_consistency"] == 1.0


def test_a_real_argument_change_still_registers():
    """Normalisation must not erase a genuine difference: a different column
    analysed is exactly the inconsistency this metric exists to catch."""
    a = '[{"table_name":"modelling_data","value_column":"credit_loss_prob_max"}]'
    b = '[{"table_name":"modelling_data","value_column":"tot_struct_risk_score_max"}]'
    group = aggregate(
        [_batch_row("old", 1, a), _batch_row("old", 2, b)], modules=["consistency"],
    )["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 0.0


def test_a_plain_string_argument_is_left_alone():
    """Only strings that parse as JSON are re-encoded."""
    from agentic_eval.dimensions.consistency import _canonical

    assert _canonical("credit_loss_prob_max") == "credit_loss_prob_max"
    assert _canonical("not json {") == "not json {"
    assert _canonical('{"b":1,"a":2}') == {"b": 1, "a": 2}


def test_rescore_recomputes_metrics_from_an_existing_runs_file(tmp_path, capsys):
    """A scoring fix must not require re-running both systems.

    The answers are still good; only the numbers derived from them are stale.
    Re-running to correct arithmetic wastes the run AND changes the sample,
    so the two readings would not be comparable.
    """
    import argparse
    import json

    from agentic_eval.cli import main

    run = tmp_path / "run_1"
    (run / "metrics").mkdir(parents=True)
    records = [
        _batch_row("old", 1, '[{"a":1}]'),
        _batch_row("old", 2, '[\n  {"a": 1}\n]'),
        _batch_row("new", 1, '[{"a":1}]'),
        _batch_row("new", 2, '[{"a":2}]'),
    ]
    for row in records:
        row["mode"] = "cold"
    (run / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps({"baseline": "old", "candidate": "new", "seed": 1}),
        encoding="utf-8",
    )
    # A stale summary, as `run` would have left it before the fix.
    (run / "metrics" / "summary.json").write_text("{}", encoding="utf-8")

    import sys
    argv = sys.argv
    sys.argv = ["agentic-eval", "rescore", "--runs", str(run / "runs.jsonl")]
    try:
        main()
    finally:
        sys.argv = argv
    summary = json.loads((run / "metrics" / "summary.json").read_text())
    by = {(g["system"]): g for g in summary["groups"]}
    # `old` reformatted the same spec — one call, consistently.
    assert by["old"]["tool_call_pairwise_multiset_jaccard"] == 1.0
    # `new` genuinely changed the spec.
    assert by["new"]["tool_call_pairwise_multiset_jaccard"] == 0.0
    assert (run / "metrics" / "comparison.md").exists()


def test_rescore_without_a_manifest_asks_which_side_is_which(tmp_path):
    import json
    import sys

    from agentic_eval.cli import main

    run = tmp_path / "run_2"
    run.mkdir()
    (run / "runs.jsonl").write_text(
        json.dumps({**_batch_row("old", 1, "[]"), "mode": "cold"}), encoding="utf-8",
    )
    argv = sys.argv
    sys.argv = ["agentic-eval", "rescore", "--runs", str(run / "runs.jsonl")]
    try:
        import pytest as _pytest
        with _pytest.raises(ValueError, match="which system is the baseline"):
            main()
    finally:
        sys.argv = argv


def _call_row(system, run_index, calls, case_id="c1"):
    row = _case_row(system, case_id, run_index, ["modeling"])
    row["evidence"] = [
        {"source_type": "tool_result", "tool": tool, "arguments": args}
        for tool, args in calls
    ]
    row["measured_over"] = []
    return row


def test_batching_is_invisible_to_tool_call_consistency():
    """`batch_summarize_trend([a, b])` does what two `summarize_trend`s do.

    Batching is how the work was ISSUED, not what was done. Unexpanded, a
    system that batched one run and not the next shared nothing with itself:
    on a real run two repeats overlapping on 6 of 7 trend columns scored zero
    matching calls.
    """
    batched = _call_row("old", 1, [(
        "batch_summarize_trend",
        {"specs_json": '[{"value_column":"fico"},{"value_column":"paydex"}]'},
    )])
    unbatched = _call_row("old", 2, [
        ("summarize_trend", {"value_column": "fico"}),
        ("summarize_trend", {"value_column": "paydex"}),
    ])
    group = aggregate([batched, unbatched], modules=["consistency"])["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0


def test_spec_order_inside_a_batch_does_not_matter():
    """The same list serialised in a different order is the same work."""
    a = _call_row("old", 1, [(
        "batch_summarize_trend",
        {"specs_json": '[{"value_column":"fico"},{"value_column":"paydex"}]'},
    )])
    b = _call_row("old", 2, [(
        "batch_summarize_trend",
        {"specs_json": '[{"value_column":"paydex"},{"value_column":"fico"}]'},
    )])
    group = aggregate([a, b], modules=["consistency"])["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0


def test_arguments_shared_by_a_batch_reach_every_spec():
    """A windowed batch must not look like an unwindowed single call."""
    from agentic_eval.dimensions.consistency import _expand_batch

    expanded = _expand_batch("batch_summarize_trend", {
        "start_date": "2025-01", "specs_json": [{"value_column": "fico"}],
    })
    assert expanded == [
        ("summarize_trend", {"start_date": "2025-01", "value_column": "fico"}),
    ]


def test_a_genuinely_larger_batch_still_differs():
    """Expansion must not erase a real difference in work done."""
    a = _call_row("old", 1, [(
        "batch_summarize_trend", {"specs_json": '[{"value_column":"fico"}]'},
    )])
    b = _call_row("old", 2, [(
        "batch_summarize_trend",
        {"specs_json": '[{"value_column":"fico"},{"value_column":"paydex"}]'},
    )])
    group = aggregate([a, b], modules=["consistency"])["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 0.5  # 1 shared of 2


def test_a_month_window_matches_its_day_spelling():
    """`2025-01`..`2026-12` covers what `2025-01-01`..`2026-12-31` covers.

    The models write it both ways between runs, and the same window read as
    two different calls — on a real run this alone halved `previous`'s b3
    consistency.
    """
    a = _call_row("old", 1, [("summarize_trend", {
        "value_column": "fico", "start_date": "2025-01-01",
        "end_date": "2026-12-31",
    })])
    b = _call_row("old", 2, [("summarize_trend", {
        "value_column": "fico", "start_date": "2025-01", "end_date": "2026-12",
    })])
    group = aggregate([a, b], modules=["consistency"])["groups"][0]
    assert group["tool_call_pairwise_multiset_jaccard"] == 1.0


def test_a_mid_month_bound_is_not_widened():
    """Only the two genuinely equivalent forms are touched."""
    from agentic_eval.dimensions.consistency import _canonical_bound

    assert _canonical_bound("start_date", "2025-01") == "2025-01-01"
    assert _canonical_bound("end_date", "2026-02") == "2026-02-28"
    assert _canonical_bound("end_date", "2024-02") == "2024-02-29"  # leap year
    # A real change of window survives.
    assert _canonical_bound("start_date", "2025-01-15") == "2025-01-15"
    # A non-bound key keeps its value verbatim.
    assert _canonical_bound("period", "2025-01") == "2025-01"
