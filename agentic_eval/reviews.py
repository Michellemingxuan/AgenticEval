"""Aggregate blinded human ratings without mixing target identities."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RATING_FIELDS = (
    "correctness_1_5",
    "completeness_1_5",
    "relevance_1_5",
    "clarity_1_5",
    "uncertainty_calibration_1_5",
)


def _rating(value: str, review_id: str, field: str) -> float | None:
    if not str(value or "").strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{review_id}: {field} must be numeric") from exc
    if not 1 <= result <= 5:
        raise ValueError(f"{review_id}: {field} must be within 1..5")
    return result


def aggregate_review_files(review_path: Path, key_path: Path) -> dict[str, Any]:
    with review_path.open(newline="", encoding="utf-8") as fh:
        reviews = list(csv.DictReader(fh))
    with key_path.open(newline="", encoding="utf-8") as fh:
        keys = {row["review_id"]: row for row in csv.DictReader(fh)}
    joined = []
    for review in reviews:
        review_id = review.get("review_id") or ""
        if review_id not in keys:
            raise ValueError(f"{review_id}: missing from review key")
        ratings = {
            field: _rating(review.get(field, ""), review_id, field)
            for field in RATING_FIELDS
        }
        if not any(value is not None for value in ratings.values()):
            continue
        scope = str(review.get("scope_correct_yes_no") or "").strip().lower()
        if scope and scope not in {"yes", "no", "y", "n"}:
            raise ValueError(f"{review_id}: scope correctness must be yes/no")
        joined.append({
            **keys[review_id],
            **ratings,
            "scope_correct": scope in {"yes", "y"} if scope else None,
            "has_unsupported_claims": bool(
                str(review.get("unsupported_claims") or "").strip()
            ),
        })

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in joined:
        grouped[(row["system"], row["mode"], row["name"])].append(row)

    def summarize(rows: list[dict]) -> dict[str, Any]:
        result: dict[str, Any] = {"n_reviewed": len(rows)}
        for field in RATING_FIELDS:
            values = [row[field] for row in rows if row[field] is not None]
            result[field] = statistics.mean(values) if values else None
        scopes = [
            row["scope_correct"] for row in rows if row["scope_correct"] is not None
        ]
        result["scope_correct_rate"] = (
            sum(scopes) / len(scopes) if scopes else None
        )
        result["unsupported_claim_rate"] = (
            sum(row["has_unsupported_claims"] for row in rows) / len(rows)
            if rows else None
        )
        return result

    return {
        "n_reviewed": len(joined),
        "groups": [
            {"system": system, "mode": mode, "name": name, **summarize(rows)}
            for (system, mode, name), rows in sorted(grouped.items())
        ],
    }


def write_review_summary(summary: dict[str, Any], output_stem: Path) -> None:
    Path(f"{output_stem}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    def rating(value):
        return "—" if value is None else f"{value:.2f}"

    def percent(value):
        return "—" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Human content-quality review",
        "",
        f"Completed reviews: {summary['n_reviewed']}",
        "",
        "| System | Mode | Question | n | Correct | Complete | Relevant | "
        "Clear | Uncertainty | Scope correct | Unsupported claim |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['system']} | {row['mode']} | {row['name']} | "
            f"{row['n_reviewed']} | {rating(row['correctness_1_5'])} | "
            f"{rating(row['completeness_1_5'])} | "
            f"{rating(row['relevance_1_5'])} | "
            f"{rating(row['clarity_1_5'])} | "
            f"{rating(row['uncertainty_calibration_1_5'])} | "
            f"{percent(row['scope_correct_rate'])} | "
            f"{percent(row['unsupported_claim_rate'])} |"
        )
    Path(f"{output_stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
