"""Latency module: wall-clock, token consumption, LLM calls, retries.

Every quantity is reported as a distribution, never a bare mean: with k
repeats the spread and the outliers are the finding, and a mean alone hides
both.
"""
from __future__ import annotations

from typing import Any

from agentic_eval.common.stats import (
    _distribution, _optional_distribution, _optional_mean, _optional_rate,
)


def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency and resource metrics for the k repeats of one cell."""
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    return {
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
