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


def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """System-level metrics for the k repeats of one cell."""
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    # A call that resolves nothing still costs a round trip and still counts as
    # a call everywhere else. Judged only over calls whose payload says either
    # way, so the rate is not padded with ones we could not read.
    calls = toolcalls.counts(rows)
    judged = calls["data"] + calls["empty"]
    return {
        "tool_call_success_rate": calls["data"] / judged if judged else None,
        "tool_calls_with_data": calls["data"],
        "tool_calls_empty": calls["empty"],
        "tool_calls_unreadable": calls["unknown"],
        "latency_seconds": _distribution(latencies),
        "prompt_tokens": _optional_distribution(rows, "prompt_tokens"),
        "completion_tokens": _optional_distribution(rows, "completion_tokens"),
        "total_tokens": _optional_distribution(rows, "total_tokens"),
        "llm_call_count": _optional_distribution(rows, "llm_call_count"),
        "retry_count": _optional_distribution(rows, "retry_count"),
        "total_tokens_mean": _optional_mean(rows, "total_tokens"),
        "llm_call_count_mean": _optional_mean(rows, "llm_call_count"),
        "retry_rate": _optional_rate(rows, "retried"),
        "retry_run_count": sum(
            bool(row.get("retried")) for row in rows
            if row.get("retried") is not None
        ),
        "retry_eligible_runs": sum(row.get("retried") is not None for row in rows),
        "retry_attempt_count": sum(
            int(row.get("retry_count") or 0) for row in rows
            if row.get("retry_count") is not None
        ),
    }
