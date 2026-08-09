"""Consistency module: team construction, tool usage, subquery similarity.

Answers "does the system do the same thing twice?" — it says nothing about
whether the thing is correct.
"""
from __future__ import annotations

import calendar
import collections
import json
import re
import statistics
from typing import Any

from agentic_eval import toolcalls
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


#: Which field a row's tool-call signature was drawn from. The three are
#: DIFFERENT VOCABULARIES — `fs_read_file:{"filename":…}` from the evidence
#: ledger can never equal `summarize_by_group(spends_data.amount, …)` from
#: `measured_over` — so two rows are only comparable when both came from the
#: same one. Mixing them scores a system as 0% consistent for a reason that has
#: nothing to do with what it did.
def _signature_source(row: dict[str, Any]) -> str:
    if any(
        item.get("source_type") in {"tool_result", "unclassified_tool_result"}
        for item in row.get("evidence") or []
    ):
        return "evidence"
    if row.get("measured_over"):
        return "measured_over"
    if row.get("tools"):
        return "tools"
    return "none"


_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{2})$")


def _canonical_bound(key: str, value: Any) -> Any:
    """Widen a bare `YYYY-MM` window bound to the day it denotes.

    A monthly trend asked for `2025-01`..`2026-12` covers exactly what one
    asked for `2025-01-01`..`2026-12-31` covers, and the models write it both
    ways between runs. Left alone, the same window read as two different calls.

    Only the two forms that are genuinely equivalent are touched: a start
    becomes the FIRST of its month, an end the LAST. `2025-01-15` is untouched
    and still differs from `2025-01`, so a real change of window survives.
    """
    if isinstance(value, str) and (match := _YEAR_MONTH.match(value.strip())):
        year, month = int(match.group(1)), int(match.group(2))
        if "start" in key.lower():
            return f"{year:04d}-{month:02d}-01"
        if "end" in key.lower():
            return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    return _canonical(value)


def _canonical(value: Any) -> Any:
    """Normalise arguments so equal calls encode identically.

    `sort_keys` fixes key order in the argument dict, but the batch tools take
    their real payload as a JSON STRING (`specs_json`), and an LLM emits that
    string pretty-printed one run and compact the next. The specs are identical
    and the signatures were not, so any question answered with a `batch_*` tool
    scored 0% consistent no matter how steady the system was.

    Only strings that actually parse as a JSON object or array are re-encoded;
    a plain value like `"credit_loss_prob_max"` is left exactly as it is, so a
    genuine change of column still registers as a difference.
    """
    if isinstance(value, str):
        if value.strip()[:1] in {"[", "{"}:
            try:
                return _canonical(json.loads(value))
            except (ValueError, TypeError):
                return value
        return value
    if isinstance(value, dict):
        return {
            key: _canonical_bound(key, item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


#: Batch tools and the single-call tool they are equivalent to. Batching is an
#: implementation detail of HOW the work was issued, not of what was done:
#: `batch_summarize_trend(specs=[a, b])` computes exactly what two
#: `summarize_trend` calls compute. Left unexpanded, a system that batched one
#: run and not the next shared nothing with itself — measured on a real run,
#: two repeats overlapping on 6 of 7 trend columns scored ZERO matching calls.
_BATCH_TOOLS = {
    "batch_summarize_trend": "summarize_trend",
    "batch_aggregate": "aggregate_column",
}

#: Where a batch tool keeps its list of operations.
_SPEC_KEYS = ("specs_json", "specs")


def _expand_batch(tool: str, arguments: Any) -> list[tuple[str, Any]]:
    """One (tool, arguments) pair per operation the call actually performed.

    Arguments beside the spec list (a shared `start_date`, say) apply to every
    spec, so they are folded into each — dropping them would make a windowed
    batch look like an unwindowed single call.
    """
    if not tool.startswith("batch_") or not isinstance(arguments, dict):
        return [(tool, arguments)]
    key = next((k for k in _SPEC_KEYS if isinstance(arguments.get(k), list)), None)
    if key is None:
        return [(tool, arguments)]
    # An unlisted `batch_*` still expands, on the prefix. If its single-call
    # name differs the parts match nothing — exactly as before — but where the
    # convention holds the comparison becomes right for free.
    single = _BATCH_TOOLS.get(tool, tool[len("batch_"):])
    shared = {k: v for k, v in arguments.items() if k != key}
    return [
        (single, {**shared, **spec} if isinstance(spec, dict) else spec)
        for spec in arguments[key]
    ]


#: What a data tool echoes back, and the argument it resolves. The catalog
#: accepts aliases — `lien_org_count` and `Lien Org Count` are one column,
#: `bureau` and `bureau_data` one table — and the result payload reports the
#: PHYSICAL name it settled on. Keying on that measures the work performed
#: rather than the phrasing used to ask for it, which is the question this
#: metric exists to answer: did the system do the same thing twice?
_RESOLVED_ARGUMENT = {
    "table": "table_name",
    "value_column": "value_column",
    "time_column": "time_column",
}


def _result_parts(result: Any, count: int) -> list[Any]:
    """The raw result of each expanded call, aligned by the batch's `index`.

    Raw, not parsed: a call that found nothing may say so in prose rather than
    in a row count, and `toolcalls.outcome` needs the text to see it.
    """
    payload = toolcalls.payload(result) or {}
    entries = payload.get("results")
    if isinstance(entries, list):
        by_index: dict[int, Any] = {}
        for position, entry in enumerate(entries):
            entry = entry if isinstance(entry, dict) else {}
            index = entry.get("index")
            by_index[int(index) if isinstance(index, int) else position] = entry.get(
                "result"
            )
        return [by_index.get(i) for i in range(count)]
    # A single call's result describes that one call; anything else resolves
    # nothing and leaves the arguments as the model wrote them.
    return [result] + [None] * (count - 1) if count == 1 else [None] * count


def _resolve(arguments: Any, payload: dict[str, Any]) -> Any:
    if not isinstance(arguments, dict) or not payload:
        return arguments
    resolved = dict(arguments)
    for echoed, argument in _RESOLVED_ARGUMENT.items():
        if payload.get(echoed) is not None:
            resolved[argument] = payload[echoed]
    return resolved


def _calls(row: dict[str, Any]):
    """Every tool call the run made, batches already expanded.

    Yields `(tool, arguments, raw_result)` so the caller can both resolve
    aliases from the payload and judge what came back.
    """
    for item in row.get("evidence") or []:
        if item.get("source_type") not in {"tool_result", "unclassified_tool_result"}:
            continue
        tool = str(item.get("tool") or "?")
        expanded = _expand_batch(tool, _canonical(item.get("arguments")))
        parts = _result_parts(item.get("result"), len(expanded))
        for (name, arguments), raw in zip(expanded, parts):
            yield name, arguments, raw


def _learn_aliases(calls: list[tuple[str, Any, Any]]) -> dict[tuple[str, str], Any]:
    """Alias -> physical name, learned from the calls that DID resolve.

    Resolution reads the physical name out of the result payload, but a call
    that found nothing has no payload to read — so it keeps the raw name the
    model wrote. Two runs that asked identically then differ on `table_name`
    purely because one succeeded: `bureau` in the failed run against
    `bureau_data` in the successful one.

    Aliases are a property of the catalog, not of a single call, so one that
    resolved anywhere in the run is applied everywhere in it. An alias seen
    resolving two different ways is dropped rather than guessed at.
    """
    learned: dict[tuple[str, str], Any] = {}
    ambiguous: set[tuple[str, str]] = set()
    for _name, arguments, raw in calls:
        if not isinstance(arguments, dict):
            continue
        payload = toolcalls.payload(raw) or {}
        for echoed, argument in _RESOLVED_ARGUMENT.items():
            asked, resolved = arguments.get(argument), payload.get(echoed)
            if not asked or resolved is None or asked == resolved:
                continue
            key = (argument, str(asked))
            if key in learned and learned[key] != resolved:
                ambiguous.add(key)
            learned[key] = resolved
    for key in ambiguous:
        learned.pop(key, None)
    return learned


def _apply_aliases(arguments: Any, aliases: dict[tuple[str, str], Any]) -> Any:
    if not isinstance(arguments, dict) or not aliases:
        return arguments
    out = dict(arguments)
    for argument in _RESOLVED_ARGUMENT.values():
        asked = out.get(argument)
        if asked is not None and (argument, str(asked)) in aliases:
            out[argument] = aliases[(argument, str(asked))]
    return out


def _tool_call_signature(row: dict[str, Any]) -> tuple[str, ...]:
    calls = list(_calls(row))
    aliases = _learn_aliases(calls)
    signatures = []
    for name, arguments, raw in calls:
        # EVERY call counts, including ones that came back empty. This
        # measures whether the system does the same thing twice, not whether
        # the thing worked — `tool_call_success_rate` in the System section
        # answers that, and the two are meant to be read side by side.
        #
        # Aliases first as a default, then the payload on top: the payload is
        # what this call actually resolved, so it wins wherever it speaks.
        arguments = _resolve(
            _apply_aliases(arguments, aliases), toolcalls.payload(raw) or {},
        )
        encoded = json.dumps(
            arguments, sort_keys=True, separators=(",", ":"), default=str,
        )
        signatures.append(f"{name}:{encoded}")
    if signatures:
        return tuple(sorted(signatures))
    measured = [" ".join(str(value).lower().split()) for value in row.get("measured_over") or []]
    if measured:
        return tuple(sorted(measured))
    return tuple(sorted(str(value) for value in row.get("tools") or []))


def _cases(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Rows split by case, preserving order; one bucket when no case is set.

    Consistency asks whether the system did the SAME thing twice, so a repeat
    only means anything against another repeat of the same case. Two customers
    legitimately get different specialists and different tables — comparing
    across them would score correct behaviour as instability, and the more
    cases a run covered the less consistent every system would look.
    """
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row.get("case_id"), []).append(row)
    return list(buckets.values())


def _dominant_source(case: list[dict[str, Any]]) -> str | None:
    """The vocabulary most of this case's runs recorded their calls in."""
    counts = collections.Counter(
        _signature_source(row) for row in case
        if _signature_source(row) != "none"
    )
    return counts.most_common(1)[0][0] if counts else None


def _comparable(cases: list[list[dict[str, Any]]], statistic) -> float | None:
    """Apply a statistic to the rows of each case that CAN be compared.

    Within a case, only the runs whose signatures came from the same field are
    scored against each other; the rest are set aside. A run recorded in a
    different field says nothing about whether the system did the same thing
    twice, and scoring it as a mismatch reported a perfectly steady system as
    0% consistent.

    None when no case has two comparable runs — undefined, not zero.
    """
    values = []
    for case in cases:
        source = _dominant_source(case)
        if source is None:
            continue
        # A run that made NO tool calls at all has an empty signature, and
        # multiset Jaccard of two empty sets is 1.0 by definition — which
        # would score two runs that did nothing as perfectly consistent.
        # Undefined is the honest answer, so they are set aside.
        signatures = [
            signature for row in case
            if _signature_source(row) == source
            and (signature := _tool_call_signature(row))
        ]
        if len(signatures) >= 2:
            values.append(statistic(signatures))
    return _mean_over_cases(values)


def _distribution_of_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Which field each run's calls came from, so a mix is visible."""
    return dict(sorted(collections.Counter(
        _signature_source(row) for row in rows
    ).items()))


def _set_aside(cases: list[list[dict[str, Any]]]) -> int:
    """Runs excluded from the tool-call rates because they were not comparable.

    Reported rather than dropped silently: a rate over 2 of 4 runs is a weaker
    statement than one over 4, and nothing else on the page would say so.
    """
    total = 0
    for case in cases:
        source = _dominant_source(case)
        if source is None:
            total += len(case)
            continue
        total += sum(1 for row in case if _signature_source(row) != source)
    return total


def _mean_over_cases(values: list[float | None]) -> float | None:
    """Mean of the per-case values that are defined, or None if none are."""
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Consistency metrics for the k repeats of one system/mode/question.

    Rates are computed WITHIN each case and then averaged over cases; modal
    values and distributions describe the whole set, since "which team is
    typical here" is still a fair question across cases.
    """
    cases = _cases(rows)
    teams = [tuple(row.get("team_unique") or []) for row in rows]
    tools = [tuple(row.get("tools") or []) for row in rows]
    tool_calls = [_tool_call_signature(row) for row in rows]
    subqueries = [row.get("subqueries") or {} for row in rows]
    # Modal value and variant count stay whole-set descriptions; the
    # consistency RATES below are the ones that must not span cases.
    _, modal_team, team_variants = _exact_consistency(teams)
    _, modal_tools, tool_variants = _exact_consistency(tools)
    _, modal_tool_calls, tool_call_variants = _exact_consistency(tool_calls)

    def per_case(extract, statistic):
        """Apply a within-case statistic to each case and average the results."""
        return _mean_over_cases([
            statistic([extract(row) for row in case]) for case in cases
        ])

    def exact(values: list[Any]) -> float | None:
        return _exact_consistency(values)[0]

    def team_of(row):
        return tuple(row.get("team_unique") or [])

    def tools_of(row):
        return tuple(row.get("tools") or [])

    def subqueries_of(row):
        return row.get("subqueries") or {}

    # Same-case pairs only, pooled across cases so the mean still reflects
    # every comparison the run actually supports.
    subquery_pairwise = [
        value for case in cases
        for value in _pairwise_values(
            [subqueries_of(row) for row in case], _subquery_similarity,
        )
    ]
    return {
        "team_exact_consistency": per_case(team_of, exact) if any(teams) else None,
        "team_pairwise_jaccard": (
            per_case(team_of, lambda values: _pairwise(values, _jaccard))
            if any(teams) else None
        ),
        "team_modal": list(modal_team or ()),
        "team_unique_variants": team_variants if any(teams) else 0,
        "team_size": _distribution(len(team) for team in teams),
        "tool_exact_consistency": per_case(tools_of, exact) if any(tools) else None,
        "tool_pairwise_jaccard": (
            per_case(tools_of, lambda values: _pairwise(values, _jaccard))
            if any(tools) else None
        ),
        "tool_modal": list(modal_tools or ()),
        "tool_unique_variants": tool_variants if any(tools) else 0,
        # Comparable rows only — see `_signature_source`. A run whose calls were
        # recorded in a different field is not evidence of inconsistency, so it
        # is set aside and counted rather than scored against the others.
        "tool_call_exact_consistency": _comparable(cases, exact),
        "tool_call_pairwise_multiset_jaccard": _comparable(
            cases, lambda values: _pairwise(values, _multiset_jaccard),
        ),
        "tool_call_modal": list(modal_tool_calls or ()),
        "tool_call_unique_variants": tool_call_variants if any(tool_calls) else 0,
        "tool_call_signature_sources": _distribution_of_sources(rows),
        "tool_call_runs_not_comparable": _set_aside(cases),
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
