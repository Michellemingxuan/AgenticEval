"""Per-question aggregation across repeats, and baseline/candidate comparison.

This module owns only composition. Each metric family lives in its own module
under `agentic_eval.modules`, so a dimension can be read, tested, or selected
without the others.
"""
from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any

from agentic_eval.common.stats import _percentile
from agentic_eval.modules import EVAL_MODULES, memory, resolve_modules
from agentic_eval.modules.content import score_content
from agentic_eval.modules.memory import score_memory

__all__ = ["aggregate", "compare", "score_content", "score_memory",
           "SECTION_BUILDERS"]

#: Metric families composed into every aggregate group, in registry order.
SECTION_BUILDERS = {
    name: module.section for name, module in EVAL_MODULES.items()
}


def aggregate(
    records: list[dict[str, Any]], *, modules: list[str] | None = None,
) -> dict[str, Any]:
    """Group runs by system/mode/question and compose each module's metrics.

    `modules` selects which metric families to compute — a list of names, a
    comma-separated string, or 'all'. Identity fields and completion rate are
    always present so a partial summary stays joinable.
    """
    selected = resolve_modules(modules)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    normalized_records = []
    for raw in records:
        row = raw if "memory_required" in raw else {**raw, **score_memory(raw)}
        normalized_records.append(row)
        grouped[(row["system"], row["mode"], row["name"])].append(row)
    items = []
    for (system, mode, name), rows in sorted(grouped.items()):
        group = {
            "system": system,
            "mode": mode,
            "name": name,
            "n_runs": len(rows),
            "completion_rate": sum(
                row.get("outcome") in {"ok", "out_of_scope"} for row in rows
            ) / len(rows),
        }
        for module_name in selected:
            group.update(SECTION_BUILDERS[module_name](rows))
        items.append(group)
    return {
        "n_records": len(records),
        "groups": items,
        "memory_groups": (
            memory.groups(normalized_records) if "memory" in selected else []
        ),
        "modules": selected,
    }


def compare(
    summary: dict[str, Any], *, baseline: str, candidate: str,
    records: list[dict[str, Any]] | None = None, seed: int = 20260731,
) -> list[dict[str, Any]]:
    by_key = {
        (row["system"], row["mode"], row["name"]): row
        for row in summary["groups"]
    }
    keys = sorted({
        (row["mode"], row["name"]) for row in summary["groups"]
        if row["system"] in {baseline, candidate}
    })

    def delta(c_value, b_value):
        return (
            float(c_value) - float(b_value)
            if c_value is not None and b_value is not None else None
        )

    out = []
    for mode, name in keys:
        b = by_key.get((baseline, mode, name))
        c = by_key.get((candidate, mode, name))
        if not b or not c:
            continue
        item = {
            "mode": mode,
            "name": name,
            "baseline": b,
            "candidate": c,
            "candidate_minus_baseline": {
                "completion_rate": delta(c["completion_rate"], b["completion_rate"]),
                "team_exact_consistency": delta(
                    c["team_exact_consistency"], b["team_exact_consistency"]
                ),
                "team_pairwise_jaccard": delta(
                    c["team_pairwise_jaccard"], b["team_pairwise_jaccard"]
                ),
                "tool_exact_consistency": delta(
                    c["tool_exact_consistency"], b["tool_exact_consistency"]
                ),
                "tool_pairwise_jaccard": delta(
                    c["tool_pairwise_jaccard"], b["tool_pairwise_jaccard"]
                ),
                "tool_call_exact_consistency": delta(
                    c["tool_call_exact_consistency"], b["tool_call_exact_consistency"]
                ),
                "tool_call_pairwise_multiset_jaccard": delta(
                    c["tool_call_pairwise_multiset_jaccard"],
                    b["tool_call_pairwise_multiset_jaccard"],
                ),
                "subquery_pairwise_similarity": delta(
                    c["subquery_pairwise_similarity"],
                    b["subquery_pairwise_similarity"],
                ),
                "mean_latency_seconds": delta(
                    c["latency_seconds"]["mean"], b["latency_seconds"]["mean"]
                ),
                "median_latency_seconds": delta(
                    c["latency_seconds"]["median"], b["latency_seconds"]["median"]
                ),
                "p95_latency_seconds": delta(
                    c["latency_seconds"]["p95"], b["latency_seconds"]["p95"]
                ),
                "max_latency_seconds": delta(
                    c["latency_seconds"]["max"], b["latency_seconds"]["max"]
                ),
                "latency_outlier_rate": delta(
                    c["latency_seconds"]["outlier_rate"],
                    b["latency_seconds"]["outlier_rate"],
                ),
                "total_tokens_mean": delta(
                    c["total_tokens_mean"], b["total_tokens_mean"]
                ),
                "total_tokens_p95": delta(
                    c["total_tokens"]["p95"], b["total_tokens"]["p95"]
                ),
                "llm_call_count_mean": delta(
                    c["llm_call_count_mean"], b["llm_call_count_mean"]
                ),
                "retry_rate": delta(c["retry_rate"], b["retry_rate"]),
                "retry_count_mean": delta(
                    c["retry_count"]["mean"], b["retry_count"]["mean"]
                ),
                "memory_hit_rate": delta(
                    c["memory_hit_rate"], b["memory_hit_rate"]
                ),
                "provenance_completeness": delta(
                    c["provenance_completeness"], b["provenance_completeness"]
                ),
                "automated_content_score": delta(
                    c["automated_content_score"], b["automated_content_score"]
                ),
            },
        }
        if records is not None:
            item["paired"] = _paired_metrics(
                records, baseline=baseline, candidate=candidate,
                mode=mode, name=name, seed=seed,
            )
        out.append(item)
    return out


def _paired_metrics(
    records: list[dict[str, Any]], *, baseline: str, candidate: str,
    mode: str, name: str, seed: int,
) -> dict[str, Any]:
    pairs: dict[tuple[Any, Any], dict[str, dict]] = defaultdict(dict)
    for row in records:
        if row.get("mode") != mode or row.get("name") != name:
            continue
        if row.get("system") not in {baseline, candidate}:
            continue
        key = (row.get("run_index"), row.get("sequence_position"))
        pairs[key][row["system"]] = row
    complete = [
        pair for pair in pairs.values() if baseline in pair and candidate in pair
    ]

    def metric(field: str, *, lower_is_better: bool) -> dict[str, Any]:
        deltas = []
        for pair in complete:
            left, right = pair[baseline].get(field), pair[candidate].get(field)
            if left is not None and right is not None:
                deltas.append(float(right) - float(left))
        if not deltas:
            return {
                "n_pairs": 0, "mean_delta": None, "median_delta": None,
                "bootstrap_95ci_mean": None, "candidate_wins": 0,
                "ties": 0, "candidate_losses": 0,
            }
        wins = sum(
            delta < 0 if lower_is_better else delta > 0 for delta in deltas
        )
        losses = sum(
            delta > 0 if lower_is_better else delta < 0 for delta in deltas
        )
        stable_offset = sum(ord(char) for char in f"{mode}:{name}:{field}")
        rng = random.Random(seed + stable_offset)
        boot = [
            statistics.mean(rng.choice(deltas) for _ in deltas)
            for _ in range(2000)
        ]
        return {
            "n_pairs": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "bootstrap_95ci_mean": [
                _percentile(boot, 0.025), _percentile(boot, 0.975),
            ],
            "candidate_wins": wins,
            "ties": len(deltas) - wins - losses,
            "candidate_losses": losses,
        }

    return {
        "elapsed_seconds": metric("elapsed_seconds", lower_is_better=True),
        "total_tokens": metric("total_tokens", lower_is_better=True),
        "llm_call_count": metric("llm_call_count", lower_is_better=True),
        "retry_count": metric("retry_count", lower_is_better=True),
        "automated_content_score": metric(
            "automated_content_score", lower_is_better=False,
        ),
        "provenance_completeness": metric(
            "provenance_completeness", lower_is_better=False,
        ),
    }

