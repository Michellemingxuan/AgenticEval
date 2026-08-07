"""Roll per-answer metrics up across the k repeated runs."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


#: Rates averaged across repeats, for the per-question walkthrough. The viewer
#: sums numerators and denominators instead — averaging averages hides whether
#: a figure rests on one claim or nine — so these are for reading, not deciding.
_MACRO_METRICS = (
    "grounded_rate", "factual_grounded_rate", "report_grounded_rate",
    "reasoning_eligible_rate", "expected_answer_accuracy_rate",
    "must_have_coverage", "judge_error_rate", "table_cell_coverage",
)


def aggregate_content_evaluations(
    rows: list[dict[str, Any]], *, expected_repeats: int | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("system")), str(row.get("mode")), str(row.get("name")))].append(row)
    groups = []
    for (system, mode, name), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "system": system, "mode": mode, "name": name, "n_runs": len(items),
            "run_indices": sorted({
                int(item["run_index"]) for item in items
                if item.get("run_index") is not None
            }),
            "expected_repeats": expected_repeats,
            "repetitions_complete": (
                len(items) == expected_repeats if expected_repeats is not None else None
            ),
            "metric_distributions": {},
        }
        for metric in _MACRO_METRICS:
            values = [
                float(item["metrics"][metric]) for item in items
                if item.get("metrics", {}).get(metric) is not None
            ]
            result[metric] = statistics.mean(values) if values else None
            result["metric_distributions"][metric] = {
                "n": len(values),
                "values_by_run": [
                    {
                        "run_index": item.get("run_index"),
                        "value": item.get("metrics", {}).get(metric),
                    }
                    for item in sorted(
                        items, key=lambda value: int(value.get("run_index") or 0)
                    )
                    if item.get("metrics", {}).get(metric) is not None
                ],
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) >= 2 else (
                    0.0 if values else None
                ),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        result["critical_must_have_misses"] = sum(
            int(item.get("metrics", {}).get("critical_must_have_misses") or 0)
            for item in items
        )
        result["critical_invalid_inferences"] = sum(
            int(item.get("metrics", {}).get("critical_invalid_inferences") or 0)
            for item in items
        )
        result["runs_with_critical_must_have_miss"] = sum(
            int((item.get("metrics", {}).get("critical_must_have_misses") or 0) > 0)
            for item in items
        )
        result["runs_with_critical_invalid_inference"] = sum(
            int((item.get("metrics", {}).get("critical_invalid_inferences") or 0) > 0)
            for item in items
        )
        groups.append(result)
    return {
        "n_evaluations": len(rows),
        "expected_repeats": expected_repeats,
        "groups": groups,
    }

