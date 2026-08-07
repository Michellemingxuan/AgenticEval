"""Comparison artifacts and blinded content-review sheets."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


def _fmt(value: float | None, *, percent: bool = False, decimals: int = 2) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{100 * value:.1f}%"
    return f"{value:.{decimals}f}"


def comparison_markdown(
    comparisons: list[dict[str, Any]], *, baseline: str, candidate: str,
) -> str:
    lines = [
        "# Agentic Q&A version comparison",
        "",
        f"Baseline: `{baseline}` · Candidate: `{candidate}`",
        "",
        "Deltas are candidate minus baseline. Positive is desirable for quality/"
        "consistency; negative is desirable for latency, tokens, calls, and retry.",
        "",
        "## Consistency across repeated runs",
        "",
        "| Mode | Question | Δ team exact | Δ team Jaccard | Δ tool-name exact | "
        "Δ tool-name Jaccard | Δ tool-call exact | Δ tool-call Jaccard | "
        "Δ subquery similarity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        delta = item["candidate_minus_baseline"]
        lines.append(
            f"| {item['mode']} | {item['name']} | "
            f"{_fmt(delta['team_exact_consistency'], percent=True)} | "
            f"{_fmt(delta['team_pairwise_jaccard'], percent=True)} | "
            f"{_fmt(delta['tool_exact_consistency'], percent=True)} | "
            f"{_fmt(delta['tool_pairwise_jaccard'], percent=True)} | "
            f"{_fmt(delta['tool_call_exact_consistency'], percent=True)} | "
            f"{_fmt(delta['tool_call_pairwise_multiset_jaccard'], percent=True)} | "
            f"{_fmt(delta['subquery_pairwise_similarity'], percent=True)} |"
        )
    memory_items = [
        item for item in comparisons
        if item["baseline"].get("memory_required") is True
        or item["candidate"].get("memory_required") is True
    ]
    lines.extend([
        "",
        "## Memory utilization",
        "",
        "Only questions marked `memory_required: true` enter this metric.",
        "",
        "| Mode | Question | Required | Baseline memory-used rate | "
        "Candidate memory-used rate | Δ memory hit rate |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for item in memory_items:
        baseline_row, candidate_row = item["baseline"], item["candidate"]
        delta = item["candidate_minus_baseline"]
        required = baseline_row.get("memory_required")
        if required is None:
            required = candidate_row.get("memory_required")
        lines.append(
            f"| {item['mode']} | {item['name']} | "
            f"{'yes' if required else 'no'} | "
            f"{_fmt(baseline_row.get('memory_hit_rate'), percent=True)} | "
            f"{_fmt(candidate_row.get('memory_hit_rate'), percent=True)} | "
            f"{_fmt(delta['memory_hit_rate'], percent=True)} |"
        )
    if not memory_items:
        lines.append("| - | No questions have memory annotations | - | - | - | - |")
    lines.extend([
        "",
        "## Latency and resource use across repeated runs",
        "",
        "| Mode | Question | Δ mean latency | Δ median | Δ p95 | Δ max | "
        "Δ outlier rate | Paired mean 95% CI | W/T/L | Δ tokens mean | "
        "Δ tokens p95 | Δ LLM calls | Δ retry-run rate | Δ retries/run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in comparisons:
        delta = item["candidate_minus_baseline"]
        paired_latency = (item.get("paired") or {}).get("elapsed_seconds") or {}
        interval = paired_latency.get("bootstrap_95ci_mean")
        interval_text = (
            f"[{interval[0]:.2f}, {interval[1]:.2f}]s"
            if interval and None not in interval else "—"
        )
        wtl = (
            f"{paired_latency.get('candidate_wins', 0)}/"
            f"{paired_latency.get('ties', 0)}/"
            f"{paired_latency.get('candidate_losses', 0)}"
            if paired_latency.get("n_pairs") else "—"
        )
        lines.append(
            f"| {item['mode']} | {item['name']} | "
            f"{_fmt(delta['mean_latency_seconds'])}s | "
            f"{_fmt(delta['median_latency_seconds'])}s | "
            f"{_fmt(delta['p95_latency_seconds'])}s | "
            f"{_fmt(delta['max_latency_seconds'])}s | "
            f"{_fmt(delta['latency_outlier_rate'], percent=True)} | "
            f"{interval_text} | {wtl} | "
            f"{_fmt(delta['total_tokens_mean'], decimals=0)} | "
            f"{_fmt(delta['total_tokens_p95'], decimals=0)} | "
            f"{_fmt(delta['llm_call_count_mean'])} | "
            f"{_fmt(delta['retry_rate'], percent=True)} | "
            f"{_fmt(delta['retry_count_mean'])} |"
        )
    lines.extend([
        "",
        "## Other automated signals",
        "",
        "| Mode | Question | Δ completion | Δ provenance | Δ deterministic content contract |",
        "|---|---|---:|---:|---:|",
    ])
    for item in comparisons:
        delta = item["candidate_minus_baseline"]
        lines.append(
            f"| {item['mode']} | {item['name']} | "
            f"{_fmt(delta['completion_rate'], percent=True)} | "
            f"{_fmt(delta['provenance_completeness'], percent=True)} | "
            f"{_fmt(delta['automated_content_score'])} |"
        )
    lines.extend([
        "",
        "Outliers use Tukey's 1.5*IQR rule and are reported only for `k >= 4`; "
        "the raw per-run values, max, and p95 remain available for smaller `k`.",
        "Subquery similarity is deterministic per-specialist lexical-token Jaccard.",
        "Tool-call consistency includes normalized arguments and repeated calls; "
        "tool-name consistency considers only the set of tool names.",
        "",
        "`—` means the selected adapter/version did not expose that telemetry.",
        "",
    ])
    return "\n".join(lines)


def write_blind_review(
    records: list[dict[str, Any]], *, review_path: Path, key_path: Path, seed: int,
) -> None:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    review_fields = [
        "review_id", "question", "answer", "correctness_1_5",
        "completeness_1_5", "relevance_1_5", "clarity_1_5",
        "uncertainty_calibration_1_5", "scope_correct_yes_no",
        "unsupported_claims", "reviewer_notes",
    ]
    key_fields = [
        "review_id", "system", "mode", "name", "run_index", "turn_id",
        "team", "scopes", "measured_over", "memory_required", "memory_used",
        "automated_content_score",
    ]
    with (
        review_path.open("w", newline="", encoding="utf-8") as review_fh,
        key_path.open("w", newline="", encoding="utf-8") as key_fh,
    ):
        review_writer = csv.DictWriter(review_fh, fieldnames=review_fields)
        key_writer = csv.DictWriter(key_fh, fieldnames=key_fields)
        review_writer.writeheader()
        key_writer.writeheader()
        for index, row in enumerate(shuffled, 1):
            review_id = f"R{index:04d}"
            review_writer.writerow({
                "review_id": review_id,
                "question": row["question"],
                "answer": row["final_answer"],
            })
            key_writer.writerow({
                "review_id": review_id,
                "system": row["system"],
                "mode": row["mode"],
                "name": row["name"],
                "run_index": row["run_index"],
                "turn_id": row.get("turn_id"),
                "team": json.dumps(row.get("team") or []),
                "scopes": json.dumps(row.get("scopes") or []),
                "measured_over": json.dumps(row.get("measured_over") or []),
                "memory_required": row.get("memory_required"),
                "memory_used": row.get("memory_used"),
                "automated_content_score": row.get("automated_content_score"),
            })
