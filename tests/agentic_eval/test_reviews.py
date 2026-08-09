import csv
from pathlib import Path

from agentic_eval.review import aggregate_review_files


def test_review_scores_remain_separated_by_system(tmp_path: Path):
    review = tmp_path / "review.csv"
    key = tmp_path / "key.csv"
    review_fields = [
        "review_id", "correctness_1_5", "completeness_1_5", "relevance_1_5",
        "clarity_1_5", "uncertainty_calibration_1_5",
        "scope_correct_yes_no", "unsupported_claims",
    ]
    with review.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=review_fields)
        writer.writeheader()
        writer.writerow({
            "review_id": "R1", "correctness_1_5": "3",
            "completeness_1_5": "3", "relevance_1_5": "4",
            "clarity_1_5": "4", "uncertainty_calibration_1_5": "3",
            "scope_correct_yes_no": "no", "unsupported_claims": "one",
        })
        writer.writerow({
            "review_id": "R2", "correctness_1_5": "5",
            "completeness_1_5": "5", "relevance_1_5": "5",
            "clarity_1_5": "5", "uncertainty_calibration_1_5": "5",
            "scope_correct_yes_no": "yes", "unsupported_claims": "",
        })
    with key.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["review_id", "system", "mode", "name"],
        )
        writer.writeheader()
        writer.writerow({
            "review_id": "R1", "system": "old", "mode": "cold", "name": "q",
        })
        writer.writerow({
            "review_id": "R2", "system": "new", "mode": "cold", "name": "q",
        })
    summary = aggregate_review_files(review, key)
    groups = {row["system"]: row for row in summary["groups"]}
    assert groups["old"]["correctness_1_5"] == 3
    assert groups["new"]["correctness_1_5"] == 5
    assert groups["old"]["unsupported_claim_rate"] == 1.0
    assert groups["new"]["scope_correct_rate"] == 1.0

