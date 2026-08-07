"""Consistency module: team construction, tool usage, subquery similarity.

Answers "does the system do the same thing twice?" — it says nothing about
whether the thing is correct.
"""
from __future__ import annotations

import json
import re
import statistics
from typing import Any

from agentic_eval.common.stats import (
    _distribution, _exact_consistency, _jaccard, _multiset_jaccard, _pairwise,
    _pairwise_values,
)


_WORD = re.compile(r"[a-z0-9_]+")


def _subquery_similarity(a: dict[str, str], b: dict[str, str]) -> float:
    tools = set(a) | set(b)
    if not tools:
        return 1.0
    scores = []
    for tool in tools:
        if tool not in a or tool not in b:
            scores.append(0.0)
            continue
        left = set(_WORD.findall(a[tool].lower()))
        right = set(_WORD.findall(b[tool].lower()))
        scores.append(_jaccard(left, right))
    return statistics.mean(scores)


def _tool_call_signature(row: dict[str, Any]) -> tuple[str, ...]:
    signatures = []
    for item in row.get("evidence") or []:
        if item.get("source_type") not in {"tool_result", "unclassified_tool_result"}:
            continue
        tool = str(item.get("tool") or "?")
        arguments = item.get("arguments")
        encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
        signatures.append(f"{tool}:{encoded}")
    if signatures:
        return tuple(sorted(signatures))
    measured = [" ".join(str(value).lower().split()) for value in row.get("measured_over") or []]
    if measured:
        return tuple(sorted(measured))
    return tuple(sorted(str(value) for value in row.get("tools") or []))



def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Consistency metrics for the k repeats of one system/mode/question."""
    teams = [tuple(row.get("team_unique") or []) for row in rows]
    tools = [tuple(row.get("tools") or []) for row in rows]
    tool_calls = [_tool_call_signature(row) for row in rows]
    subqueries = [row.get("subqueries") or {} for row in rows]
    team_exact, modal_team, team_variants = _exact_consistency(teams)
    tool_exact, modal_tools, tool_variants = _exact_consistency(tools)
    tool_call_exact, modal_tool_calls, tool_call_variants = _exact_consistency(tool_calls)
    subquery_pairwise = _pairwise_values(subqueries, _subquery_similarity)
    return {
        "team_exact_consistency": team_exact if any(teams) else None,
        "team_pairwise_jaccard": (
            _pairwise(teams, _jaccard) if any(teams) else None
        ),
        "team_modal": list(modal_team or ()),
        "team_unique_variants": team_variants if any(teams) else 0,
        "team_size": _distribution(len(team) for team in teams),
        "tool_exact_consistency": tool_exact if any(tools) else None,
        "tool_pairwise_jaccard": (
            _pairwise(tools, _jaccard) if any(tools) else None
        ),
        "tool_modal": list(modal_tools or ()),
        "tool_unique_variants": tool_variants if any(tools) else 0,
        "tool_call_exact_consistency": (
            tool_call_exact if any(tool_calls) else None
        ),
        "tool_call_pairwise_multiset_jaccard": (
            _pairwise(tool_calls, _multiset_jaccard)
            if any(tool_calls) else None
        ),
        "tool_call_modal": list(modal_tool_calls or ()),
        "tool_call_unique_variants": tool_call_variants if any(tool_calls) else 0,
        "subquery_pairwise_similarity": (
            # Guard the PAIRS, not the subqueries. At repeats=1 there is
            # exactly one run per cell, so no pair exists and the list is
            # empty even though every record carries subqueries — which
            # crashed aggregation (`mean requires at least one data
            # point`) after all records had already been collected.
            # Undefined similarity is None, reported as "—".
            statistics.mean(subquery_pairwise)
            if subquery_pairwise else None
        ),
        "subquery_similarity": {
            **_distribution(subquery_pairwise),
            "method": "per-specialist lexical-token Jaccard",
        },
    }
