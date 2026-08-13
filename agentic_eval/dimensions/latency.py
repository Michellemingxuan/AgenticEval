"""System module: wall-clock, tokens, LLM calls, retries, tool-call success.

Every quantity is reported as a distribution, never a bare mean: with k
repeats the spread and the outliers are the finding, and a mean alone hides
both.
"""
from __future__ import annotations

from typing import Any

from agentic_eval import toolcalls
from agentic_eval.common.stats import (
    _distribution, _optional_distribution, _optional_mean, _optional_rate,
)


def _with_recovery_names(row: dict[str, Any]) -> dict[str, Any]:
    """Old `retry_*` fields read under the current names.

    `retried` was always "any self-recovery fired" — `retry_count > 0` over
    tool plus orchestration — so the rename is a rename, not a new
    measurement, and a run recorded before it can be read without re-running.
    Done once here so nothing downstream has to know two vocabularies.

    Only fills what is MISSING: a new record's own values always win.
    """
    if row.get("self_recovery_count") is None and row.get("retry_count") is not None:
        row = {**row, "self_recovery_count": row["retry_count"]}
    if row.get("self_recovered") is None and row.get("retried") is not None:
        row = {**row, "self_recovered": row["retried"]}
    return row


def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """System-level metrics for the k repeats of one cell."""
    rows = [_with_recovery_names(row) for row in rows]
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    # A call that resolves nothing still costs a round trip and still counts as
    # a call everywhere else. Judged only over calls whose payload says either
    # way, so the rate is not padded with ones we could not read.
    calls = toolcalls.counts(rows)
    judged = calls["data"] + calls["failed"]
    return {
        # The total first: a rate means little without the count it is over,
        # and `unknown` sits inside the total but outside the rate.
        "tool_calls_total": calls["data"] + calls["failed"] + calls["unknown"],
        "tool_call_success_rate": calls["data"] / judged if judged else None,
        "tool_calls_judged": judged,
        "tool_calls_with_data": calls["data"],
        "tool_calls_failed": calls["failed"],
        "tool_calls_unreadable": calls["unknown"],
        "latency_seconds": _distribution(latencies),
        "prompt_tokens": _optional_distribution(rows, "prompt_tokens"),
        "completion_tokens": _optional_distribution(rows, "completion_tokens"),
        "total_tokens": _optional_distribution(rows, "total_tokens"),
        "llm_call_count": _optional_distribution(rows, "llm_call_count"),
        "self_recovery_count": _optional_distribution(rows, "self_recovery_count"),
        "total_tokens_mean": _optional_mean(rows, "total_tokens"),
        "llm_call_count_mean": _optional_mean(rows, "llm_call_count"),
        # Sums beside the means. A mean per answer compares two systems on
        # equal footing; a total says what the run actually cost. The page
        # labelled the MEAN "Total tokens", which read as the latter.
        "total_tokens_sum": sum(
            int(row.get("total_tokens") or 0) for row in rows
            if row.get("total_tokens") is not None
        ),
        "llm_call_count_sum": sum(
            int(row.get("llm_call_count") or 0) for row in rows
            if row.get("llm_call_count") is not None
        ),
        # SELF-RECOVERY: the system noticing something went wrong and fixing it
        # WITHOUT being asked again. This is what `retry_rate` always measured
        # — `retried` was `retry_count > 0` over tool plus orchestration — so
        # the rename says what the number already meant.
        #
        # Benign, and reported without a verdict: a system that re-issues a
        # call and then answers correctly has succeeded. What it measures is
        # how often the happy path missed, and how much the answer cost.
        #
        #   call           the transport abandoned a stalled LLM call and
        #                  re-issued it — below the tool level, and the
        #                  specialist never knew
        #   tool           inside one attempt: a re-issued call, an ungrounded
        #                  answer sent back for evidence, a retaken plan step
        #   orchestration  the whole plan re-run for the same turn
        #
        # `None` on runs that never recorded it, rather than 0 — "not
        # measured" and "never happened" must not read alike.
        "self_recovery_rate": _optional_rate(rows, "self_recovered"),
        "self_recovery_call_rate": _optional_rate(rows, "self_recovered_call"),
        "self_recovery_tool_rate": _optional_rate(rows, "self_recovered_tool"),
        "self_recovery_orchestration_rate": _optional_rate(
            rows, "self_recovered_orchestration",
        ),
        "self_recovery_attempts": sum(
            int(row.get("self_recovery_count") or 0) for row in rows
            if row.get("self_recovery_count") is not None
        ),
        "self_recovery_runs": sum(
            bool(row.get("self_recovered")) for row in rows
            if row.get("self_recovered") is not None
        ),
        "self_recovery_eligible_runs": sum(
            row.get("self_recovered") is not None for row in rows
        ),
        # The one that is NOT benign: the system produced nothing at all and
        # the EVALUATOR asked again. Never mixed into the rates above.
        "evaluator_replay_rate": _optional_rate(rows, "evaluator_replayed"),
    }
