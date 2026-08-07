"""Content module: the answer-quality dimension.

Two layers, deliberately separate:

- `score_content` / `section` are FREE. They read fields already on the run
  record — provenance completeness and a structural score — and need no
  judge, so they are safe to compute on every aggregation.
- The evidence-bound CASCADE in `agentic_eval.content` is the real content
  evaluation, and it costs LLM calls. Selecting this module by name turns it
  on; see `runner` for when it runs.
"""
from __future__ import annotations

from typing import Any

from agentic_eval.common.stats import _optional_mean


def score_content(record: dict[str, Any]) -> dict[str, Any]:
    """Apply the optional per-question deterministic answer contract."""
    cfg = record.get("evaluation") or {}
    answer = str(record.get("final_answer") or "").lower()
    team = set(record.get("team_unique") or record.get("team") or [])
    scope_blob = " ".join(
        [*(record.get("scopes") or []), *(record.get("measured_over") or [])]
    ).lower()
    components: dict[str, dict[str, Any]] = {}

    if cfg.get("expected_outcome"):
        components["outcome"] = {
            "score": float(record.get("outcome") == cfg["expected_outcome"]),
            "weight": 10,
        }
    required = set(cfg.get("required_specialists") or [])
    allowed = set(cfg.get("allowed_specialists") or [])
    if required or allowed:
        recall = len(team & required) / len(required) if required else 1.0
        precision = (
            len(team & allowed) / len(team)
            if allowed and team else (1.0 if not allowed or not required else 0.0)
        )
        value = (
            2 * recall * precision / (recall + precision)
            if allowed and recall + precision else recall
        )
        components["team"] = {
            "score": value, "weight": 15,
            "missing": sorted(required - team),
            "unexpected": sorted(team - allowed) if allowed else [],
        }
    scope_terms = [str(v).lower() for v in cfg.get("required_scope_terms") or []]
    if scope_terms:
        missing = [term for term in scope_terms if term not in scope_blob]
        components["scope_alignment"] = {
            "score": (len(scope_terms) - len(missing)) / len(scope_terms),
            "weight": 25, "missing": missing,
        }
    must = [str(v).lower() for v in cfg.get("answer_must_include") or []]
    groups = cfg.get("answer_must_include_any") or []
    forbidden = [str(v).lower() for v in cfg.get("answer_must_not_include") or []]
    checks = [term in answer for term in must]
    checks.extend(
        any(str(option).lower() in answer for option in group) for group in groups
    )
    checks.extend(term not in answer for term in forbidden)
    if checks:
        components["answer_requirements"] = {
            "score": sum(checks) / len(checks), "weight": 30,
            "n_checks": len(checks), "n_passed": sum(checks),
        }
    provenance = record.get("provenance_completeness")
    if provenance is not None:
        components["provenance"] = {
            "score": float(provenance), "weight": 20,
        }
    denominator = sum(item["weight"] for item in components.values())
    score = (
        100 * sum(item["score"] * item["weight"] for item in components.values())
        / denominator
        if cfg and denominator else None
    )
    return {
        "automated_content_score": score,
        "content_components": components,
    }



def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge-free content signals for the k repeats of one cell."""
    return {
        "provenance_completeness": _optional_mean(rows, "provenance_completeness"),
        "automated_content_score": _optional_mean(rows, "automated_content_score"),
    }
