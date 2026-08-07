from agentic_eval.scoring import aggregate, compare, score_content, score_memory


def _row(system, run_index, latency, tokens):
    return {
        "system": system, "mode": "cold", "name": "q", "run_index": run_index,
        "outcome": "ok", "elapsed_seconds": latency,
        "team_unique": ["modeling"], "tools": ["summarize_trend"],
        "subqueries": {"modeling": "check tsr trend"},
        "total_tokens": tokens, "llm_call_count": 3, "retried": False,
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
            "retry_count": 2 if run_index == 5 else 0,
            "retried": run_index == 5,
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
    assert group["retry_rate"] == 0.2
    assert group["retry_run_count"] == 1
    assert group["retry_attempt_count"] == 2
    assert group["retry_count"]["mean"] == 0.4
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
    from agentic_eval.modules import EVAL_MODULES, resolve_modules
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
