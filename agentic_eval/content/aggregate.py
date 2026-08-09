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
        # Metrics pool over cases AND repeats — the question is how the system
        # does on this question, and one customer is one sample of that. The
        # completeness check must count the same way, or every multi-case run
        # reports itself as a partial evaluation.
        case_ids = sorted({
            str(item.get("case_id")) for item in items
            if item.get("case_id") is not None
        })
        expected_rows = (
            expected_repeats * max(1, len(case_ids))
            if expected_repeats is not None else None
        )
        result: dict[str, Any] = {
            "system": system, "mode": mode, "name": name, "n_runs": len(items),
            "case_ids": case_ids,
            "run_indices": sorted({
                int(item["run_index"]) for item in items
                if item.get("run_index") is not None
            }),
            "expected_repeats": expected_repeats,
            "expected_rows": expected_rows,
            "repetitions_complete": (
                len(items) == expected_rows if expected_rows is not None else None
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
                        "case_id": item.get("case_id"),
                        "run_index": item.get("run_index"),
                        "value": item.get("metrics", {}).get(metric),
                    }
                    # Case first, so a multi-case run reads as one case's
                    # repeats then the next rather than interleaving them.
                    for item in sorted(
                        items,
                        key=lambda value: (
                            str(value.get("case_id") or ""),
                            int(value.get("run_index") or 0),
                        ),
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
        "set_groups": _set_groups(rows),
    }


#: Count pairs summed, never averaged: a mean of per-answer percentages hides
#: whether a figure rests on one claim or nine.
#:
#: Memory leverage is NOT here: it is judged by this cascade but reported in
#: the viewer's Memory block, beside arrival, so it lives in
#: `render.page._CONTENT_MEMORY_METRICS` instead.
#:
#: These pairs MUST match `render.page._CONTENT_METRICS`, so the set table and
#: the overview cannot disagree about what a metric means. They are two lists
#: because `render` cannot import `content` at module level without a cycle;
#: a test asserts they agree, so drift fails loudly rather than showing two
#: different numbers for one metric.
SET_RATIOS = (
    ("expected_answer_accuracy_rate", "answer_correct", "answer_checked"),
    ("orthogonal_claim_count", "orthogonal_claim_count", "all_factual_claim_count"),
    ("grounded_rate", "grounded_count", "orthogonal_claim_count"),
    ("factual_grounded_rate", "factual_grounded_count", "orthogonal_claim_count"),
    ("report_grounded_rate", "report_grounded_count", "orthogonal_claim_count"),
    ("must_have_coverage", "must_have_coverage", "must_have_questions"),
)


def _set_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Metrics for each question SET, pooled over its questions and repeats.

    A set is the unit a reader compares: series A is "can it get computable
    facts right", series B "can it hold a line of reasoning". Pooling those
    into one number answers neither question, and reading question by question
    buries the answer in eighteen rows.

    Numerators and denominators are summed rather than averaged so each figure
    can be shown as `n/d` and says how much it rests on.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row.get("system")), str(row.get("mode")),
            str(row.get("question_set") or "questions"),
        )].append(row)
    out = []
    for (system, mode, question_set), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "system": system,
            "mode": mode,
            "question_set": question_set,
            "n_answers": len(items),
            "questions": sorted({str(item.get("name")) for item in items}),
            "case_ids": sorted({
                str(item.get("case_id")) for item in items
                if item.get("case_id") is not None
            }),
        }
        for label, numerator, denominator in SET_RATIOS:
            top = sum(
                float(item.get("metrics", {}).get(numerator) or 0)
                for item in items
            )
            bottom = sum(
                float(item.get("metrics", {}).get(denominator) or 0)
                for item in items
            )
            result[label] = (top / bottom) if bottom else None
            result[f"{label}_counts"] = {
                "numerator": top, "denominator": bottom,
            }
        out.append(result)
    return out

