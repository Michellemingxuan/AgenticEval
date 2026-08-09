"""Human-facing content artifacts: scorecard, walkthrough, review packets."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_eval.render.markers import (
    ELIGIBILITY_MARKER as _STEP_MARKER,
    GROUNDING_MARKER as _GROUNDING_MARKER,
)
from agentic_eval.layout import RunLayout


def _measured_sources() -> tuple[str, ...]:
    """Imported at call time, deliberately.

    `render` must not depend on `content` at module scope: content's package
    init loads the pipeline, which loads this module to write its artifacts,
    and the cycle only shows up once the two live in different packages.
    Presentation reading one vocabulary constant from the evaluator is fine;
    presentation being a build-time dependency of it is not.
    """
    from agentic_eval.content.evidence import MEASURED_SOURCES

    return MEASURED_SOURCES


def content_comparison_markdown(
    summary: dict[str, Any], *, baseline: str, candidate: str,
) -> str:
    by_key = {
        (row["system"], row["mode"], row["name"]): row
        for row in summary.get("groups") or []
    }
    keys = sorted({
        (row["mode"], row["name"]) for row in summary.get("groups") or []
        if row["system"] in {baseline, candidate}
    })

    def pct_with_sd(row: dict[str, Any], metric: str) -> str:
        distribution = (row.get("metric_distributions") or {}).get(metric) or {}
        mean, stdev = distribution.get("mean"), distribution.get("stdev")
        if mean is None:
            return "-"
        return f"{100 * float(mean):.1f}% ± {100 * float(stdev or 0):.1f}%"

    lines = [
        "# Content evaluation comparison", "",
        f"Baseline: `{baseline}` | Candidate: `{candidate}`", "",
        "A claim is GROUNDED when it traces back to operations on specific tables "
        "by a route that answers the question asked (factual), or when it relays "
        "curated report material that resolves (report).", "",
        "Accuracy is Python ground truth from the rubric's oracle, judged without "
        "an LLM.", "",
        "Each cell is the mean ± sample standard deviation across repeated runs.", "",
        "| Mode | Question | System | Runs | Accuracy | Orthogonal claims | "
        "Grounded | · factual | · report | Reasoning eligible | Must-have hit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, name in keys:
        for system in (baseline, candidate):
            row = by_key.get((system, mode, name))
            if not row:
                continue
            lines.append(
                f"| {mode} | {name} | {system} | {row['n_runs']} | "
                f"{pct_with_sd(row, 'expected_answer_accuracy_rate')} | "
                f"{row.get('orthogonal_claim_count', '-')} | "
                f"{pct_with_sd(row, 'grounded_rate')} | "
                f"{pct_with_sd(row, 'factual_grounded_rate')} | "
                f"{pct_with_sd(row, 'report_grounded_rate')} | "
                f"{pct_with_sd(row, 'reasoning_eligible_rate')} | "
                f"{pct_with_sd(row, 'must_have_coverage')} |"
            )
    lines.extend(["", "`-` means the metric was not applicable or evidence was unavailable.", ""])
    return "\n".join(lines)






WALKTHROUGH_LEGEND = (
    "`gnd` grounding: `◆` factual (traces to operations by an eligible route) · "
    "`◇` report (drawn from the curated report files) · `○` neither  \n"
    "`elg` reasoning trace eligible for the question asked: "
    "`✓` yes · `✗` no · `?` no recorded operation to rule on  \n"
    "A claim is grounded when BOTH hold: it reaches operations on specific "
    "tables, and the route that got there answers the question asked."
)


def _claim_marker_row(claim: dict[str, Any], fact: dict[str, Any]) -> str:
    return (
        f"{_GROUNDING_MARKER.get(fact.get('grounding_kind'), '?')}"
        f"  {_STEP_MARKER.get(fact.get('eligible'), '?')}"
    )


def content_walkthrough_markdown(evaluation: dict[str, Any]) -> str:
    """Render one answer as answer → claims → numeric verdicts.

    The point is to make a verdict inspectable in one screen: which span of the
    raw answer became which claim, and what each fact was checked against.
    A rate is not reviewable; a marked-up answer is.
    """
    facts = {row["claim_id"]: row for row in evaluation.get("fact_results") or []}
    claims = evaluation.get("claims") or []
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_block[str(claim.get("block_id") or "")].append(claim)
    metrics = evaluation.get("metrics") or {}

    lines = [
        f"## {evaluation.get('system')} · {evaluation.get('mode')} · "
        f"{evaluation.get('name')} · run {evaluation.get('run_index')}",
        "",
        f"**Question** — {evaluation.get('question')}",
        "",
        "### 1. Raw final answer",
        "",
        "```text",
        str(evaluation.get("answer") or "").rstrip(),
        "```",
        "",
        "### 2. Claims",
        "",
        WALKTHROUGH_LEGEND,
        "",
    ]

    for block in evaluation.get("blocks") or []:
        block_id = str(block.get("block_id") or "")
        block_claims = by_block.get(block_id) or []
        if not block_claims:
            continue
        kind = str(block.get("type") or "")
        lines.append(f"**`{block_id}` {kind}**")
        lines.append("")
        lines.append("| gnd | elg | claim | claim |")
        lines.append("|:-:|:-:|---|---|")
        for claim in block_claims:
            fact = facts.get(claim.get("claim_id"), {})
            if not fact:
                continue
            markers = _claim_marker_row(claim, fact).split()
            locator = claim.get("source_locator") or {}
            where = (
                f"r{locator['row']}c{locator['column']}"
                if locator.get("row") is not None else "¶"
            )
            stance = claim.get("stance") or "asserted"
            proposition = str(claim.get("proposition") or "").replace("|", "\\|")
            if stance != "asserted":
                proposition = f"_{stance}_ · {proposition}"
            lines.append(
                f"| {markers[-2]} | {markers[-1]} | "
                f"`{claim.get('claim_id')}` {where} | {proposition} |"
            )
        lines.append("")

    lines.extend(["### 3. Numbers", ""])
    number_rows = [
        (claim, number)
        for claim in claims
        for number in (facts.get(claim.get("claim_id"), {}).get("numbers") or [])
    ]
    if not number_rows:
        lines.append("_No material numbers were asserted._")
    else:
        lines.append(
            "| claim | written | measures | asserts | tool output | evidence path "
            "| verdict |",
        )
        lines.append("|---|---|---|:-:|---:|---|---|")
        for claim, number in number_rows:
            comparator = str(number.get("comparator") or "==")
            relation = (
                f"`{comparator} {number.get('answer_value')}`"
                if comparator != "==" else f"`= {number.get('answer_value')}`"
            )
            operation = number.get("operation")
            path = number.get("json_path") or (
                ", ".join(number.get("operand_evidence_ids") or []) or "—"
            )
            if operation:
                path = f"`{operation}()` over {path}"
            failure = number.get("trace_failure")
            verdict = "✓ traced" if not failure else f"✗ {failure}"
            escaped_path = str(path).replace("|", "\\|")
            lines.append(
                f"| `{claim.get('claim_id')}` | `{number.get('written_value')}` | "
                f"{str(number.get('measures') or '—')} | "
                f"{relation} | {number.get('evidence_value')} | "
                f"{escaped_path} | {verdict} |"
            )
    lines.append("")

    relation_rows = [
        (claim, relation)
        for claim in claims
        for relation in (facts.get(claim.get("claim_id"), {}).get("relations") or [])
    ]
    if relation_rows:
        lines.extend(["### 3b. Relations", ""])
        lines.append("| claim | left | op | right | holds | verdict |")
        lines.append("|---|---:|:-:|---:|:-:|---|")
        for claim, relation in relation_rows:
            failure = relation.get("trace_failure")
            lines.append(
                f"| `{claim.get('claim_id')}` | {relation.get('left_value')} | "
                f"`{relation.get('operator')}` | {relation.get('right_value')} | "
                f"{'✓' if relation.get('holds') else '✗'} | "
                f"{'✓ traced' if not failure else f'✗ {failure}'} |"
            )
        lines.append("")

    oracles = evaluation.get("expected_answer_results") or []
    if oracles:
        lines.extend(["### 4. Python ground truth", ""])
        lines.append("| expected answer | truth | verdict | why |")
        lines.append("|---|---|:-:|---|")
        for row in oracles:
            mark = {"pass": "✓", "fail": "✗"}.get(row.get("verdict"), "?")
            lines.append(
                f"| `{row.get('expected_answer_id')}` | {row.get('expected')} | "
                f"{mark} {row.get('verdict')} | {row.get('reason')} |"
            )
        lines.append("")

    def pct(name: str) -> str:
        value = metrics.get(name)
        return "—" if value is None else f"{100 * float(value):.0f}%"

    lines.extend([
        "### Totals",
        "",
        f"**grounded {pct('grounded_rate')}** "
        f"= factual {pct('factual_grounded_rate')} + report {pct('report_grounded_rate')} "
        f"· of {metrics.get('orthogonal_claim_count', '—')} orthogonal claims  ",
        f"reasoning eligible {pct('reasoning_eligible_rate')} · "
        f"judge error {pct('judge_error_rate')} · "
        f"table cells covered {pct('table_cell_coverage')}",
        "",
    ])
    return "\n".join(lines)


def write_content_walkthrough(
    evaluations: list[dict[str, Any]], *, layout: RunLayout,
) -> Path:
    path = layout.walkthrough
    body = "\n".join(
        content_walkthrough_markdown(evaluation) for evaluation in evaluations
    )
    path.write_text(
        "# Content walkthrough\n\nRaw answer → claims → numeric verdicts, "
        "one section per evaluated answer.\n\n" + body,
        encoding="utf-8",
    )
    return path


def write_evidence_review_packets(
    evaluations: list[dict[str, Any]], *, layout: RunLayout, seed: int,
) -> None:
    """Write blinded Phase-B packets; the identity key stays in a separate file."""
    shuffled = list(evaluations)
    random.Random(seed).shuffle(shuffled)
    packet_path = layout.evidence_review
    key_path = layout.evidence_review_key
    packet_path.write_text("", encoding="utf-8")
    with key_path.open("w", newline="", encoding="utf-8") as key_fh:
        key_writer = csv.DictWriter(
            key_fh,
            fieldnames=[
                "review_id", "system", "mode", "name", "run_index", "turn_id",
                "memory_required", "memory_used",
            ],
        )
        key_writer.writeheader()
        for index, evaluation in enumerate(shuffled, 1):
            review_id = f"E{index:04d}"
            ledger = evaluation.get("evidence_ledger") or []
            packet = {
                "review_id": review_id,
                "question": evaluation.get("question"),
                "answer": evaluation.get("answer"),
                "canonical_evidence": [
                    item for item in ledger if item.get("source_type") == "canonical_fact"
                ],
                "system_provenance": [
                    item for item in ledger if item.get("source_type") != "canonical_fact"
                ],
                "provenance_note": (
                    "System tool-result provenance supplied."
                    if any(item.get("source_type") in _measured_sources() for item in ledger)
                    else "System tool-result provenance was not captured by this version."
                ),
                "claim_reviews": [
                    {
                        "claim": claim,
                        "preliminary_result": next(
                            (row for row in evaluation.get("fact_results") or []
                             if row.get("claim_id") == claim.get("claim_id")),
                            None,
                        ),
                        "reviewer_verdict": None,
                        "reviewer_note": "",
                    }
                    for claim in evaluation.get("claims") or [] if claim.get("is_factual")
                ],
                "must_have_reviews": evaluation.get("must_have_results") or [],
                            }
            with packet_path.open("a", encoding="utf-8") as packet_fh:
                packet_fh.write(json.dumps(packet, ensure_ascii=False, default=str) + "\n")
            key_writer.writerow({
                "review_id": review_id,
                "system": evaluation.get("system"),
                "mode": evaluation.get("mode"),
                "name": evaluation.get("name"),
                "run_index": evaluation.get("run_index"),
                "turn_id": evaluation.get("turn_id"),
                "memory_required": evaluation.get("memory_required"),
                "memory_used": evaluation.get("memory_used"),
            })

