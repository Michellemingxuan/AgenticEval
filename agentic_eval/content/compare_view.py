"""Side-by-side viewer for one sampled repeat: answers, facts, metrics.

A scorecard says which system scored better; it cannot say why. This puts the
two raw answers next to each other, then the atomic facts each was decomposed
into with the cascade markers, then the metrics that follow from them — so a
number can be walked back to the sentence that produced it.

One repeat, not an average. Averaging answers is meaningless, and reading a
single real pair is how a rate gets sanity-checked.
"""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_eval.common.coerce import _slug
from agentic_eval.layout import RunLayout
from agentic_eval.content.report import _GROUNDING_MARKER

def _dig(source: dict[str, Any], key: str) -> Any:
    """Read a possibly dotted key, e.g. `latency_seconds.mean`."""
    value: Any = source
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


#: One schema per module, used for the per-question tables AND the question-set
#: overview, so a reader compares like with like. Deliberately short: a row
#: earns its place by being able to move independently of the others. The
#: cascade steps that decompose `Grounded` are on the record and in the
#: walkthrough — repeating them here made a dozen rows that always agreed.
#:
#: (key, label, higher_is_better, format). Format "count" is summed across
#: questions rather than averaged: "how many runs needed memory" is a total,
#: and averaging it reported 1 required run out of 6 questions as 0.17.
#: Content is reported as (numerator, denominator, label, higher_is_better).
#: Both are SUMMED across repeats — and, in the overview, across questions —
#: rather than averaging per-repeat percentages, so every figure is shown as
#: the claims or runs it actually rests on. 100% of one claim and 100% of nine
#: are not the same result, and averaging averages hides which you have.
#:
#: `Accuracy` sums to exactly the definition wanted at both levels: per
#: question it is times-correct over times-run, and over the set it is
#: questions-correct over questions-run, averaged across repeats.
#:
#: `Orthogonal claims` is second because every row below it is computed over
#: the deduplicated claims — its denominator is their denominator.
#:
#: Reasoning-trace eligibility has no row. It is not dropped — it GATES
#: `factual grounded`, so an ineligible route already shows up there as a
#: claim that did not ground. A second row restated the same finding against a
#: different denominator, which read as two results when there was one.
#: A fifth field is the display style:
#:   "rate"   `71% (66/94)` with a percentage-point delta
#:   "count"  `94 (out of 103)` with an absolute delta — how much an answer
#:            asserted is a quantity, not a proportion
#:   "part"   a rate with no delta: the two halves of `Grounded` move
#:            against each other by construction, so a signed change on each
#:            reads as two findings when there is one
_CONTENT_METRICS = (
    ("answer_correct", "answer_checked", "Accuracy", True, "rate"),
    ("orthogonal_claim_count", "all_factual_claim_count", "Orthogonal claims", True, "count"),
    ("grounded_count", "orthogonal_claim_count", "Grounded", True, "rate"),
    ("factual_grounded_count", "orthogonal_claim_count", "· factual grounded", True, "part"),
    ("report_grounded_count", "orthogonal_claim_count", "· report grounded", True, "part"),
    ("must_have_coverage", "must_have_questions", "Must-have hit rate", True, "rate"),
    # Arrival is in the memory module; this is USE. A source offered and not
    # drawn on is not a failure — but a session that never leverages what it
    # remembers is paying for the memory and re-doing the work anyway.
    ("memory_sources_leveraged", "memory_sources_offered", "Memory leveraged", True, "rate"),
)

#: The other three eval modules, read from the run's `summary.json`.
_MODULE_METRICS = {
    "consistency": (
        ("team_pairwise_jaccard", "Team Jaccard", True, "pct"),
        ("tool_call_pairwise_multiset_jaccard", "Tool-call Jaccard", True, "pct"),
        ("subquery_pairwise_similarity", "Subquery similarity", True, "pct"),
    ),
    "memory": (
        ("memory_hit_rate", "Memory hit rate", True, "pct"),
        ("memory_required_run_count", "Memory-required runs", True, "count"),
    ),
    "latency": (
        ("latency_seconds.mean", "Latency mean", False, "sec"),
        ("total_tokens.mean", "Total tokens", False, "num"),
        ("llm_call_count.mean", "LLM calls", False, "num"),
        ("retry_rate", "Retry rate", False, "pct"),
    ),
}

#: Per-question overrides. Exposure rates describe how a system behaves across
#: a session, so at one question they are noise; what a single question can
#: answer is whether IT needed memory and whether memory arrived.
_QUESTION_MODULE_METRICS = {
    **_MODULE_METRICS,
    "memory": (
        ("memory_required_run_count", "Memory required", True, "count"),
        ("memory_hit_rate", "Memory hit", True, "pct"),
    ),
}


def _fmt_value(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{100 * float(value):.0f}%"
    if kind == "sec":
        return f"{float(value):.1f}s"
    number = float(value)
    if abs(number) >= 100 or number == int(number):
        return f"{number:,.0f}"
    # A mean over repeats is not exact: "6.66667 LLM calls" reads as precision
    # the measurement does not have.
    return f"{number:,.1f}"


def _fmt_rate(value: Any, kind: str, count: Any, denominator: Any) -> str:
    """A rate, with the claims it was computed over when those are known."""
    shown = _fmt_value(value, kind)
    if count is None or denominator in (None, 0):
        return shown
    return f"{_fmt_value(count, 'count')}/{_fmt_value(denominator, 'count')} ({shown})"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.0f}%"


def _delta_cell(
    baseline: Any, candidate: Any, higher_is_better: bool, kind: str = "pct",
) -> str:
    if baseline is None or candidate is None:
        return '<td class="delta"></td>'
    change = float(candidate) - float(baseline)
    if abs(change) < 1e-9:
        return '<td class="delta">·</td>'
    good = (change > 0) == higher_is_better
    shown = (
        f"{100 * change:+.0f}%" if kind == "pct"
        else f"{change:+.1f}" if kind == "sec"
        else f"{change:+,.0f}"
    )
    return f'<td class="delta {"up" if good else "down"}">{shown}</td>'


def _module_table(
    base_group: dict[str, Any] | None, cand_group: dict[str, Any] | None,
    *, spec, baseline: str, candidate: str, note: str,
) -> str:
    base_group, cand_group = base_group or {}, cand_group or {}
    rows = []
    for key, label, higher_is_better, kind in spec:
        base_value, cand_value = _dig(base_group, key), _dig(cand_group, key)
        if base_value is None and cand_value is None:
            continue
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f'<td class="num">{_fmt_value(base_value, kind)}</td>'
            f'<td class="num">{_fmt_value(cand_value, kind)}</td>'
            f"{_delta_cell(base_value, cand_value, higher_is_better, kind)}</tr>"
        )
    if not rows:
        return (
            '<p class="missing">Not recorded. This module needs the run\'s '
            "<code>summary.json</code>, written by <code>run</code>.</p>"
        )
    return (
        f'<p class="tnote">{html.escape(note)}</p>'
        '<table class="metrics"><thead><tr><th>Metric</th>'
        f"<th>{html.escape(baseline)}</th><th>{html.escape(candidate)}</th>"
        "<th>Δ</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


#: Fields a fact result only carries if it was produced by a recent evaluator.
_VERSIONED_FIELDS = ("grounding_kind", "evidence_resolution", "eligible")


def _marker(fact: dict[str, Any], key: str, table: dict[str, str]) -> str:
    """Render one marker, distinguishing ABSENT from unknown.

    An evaluation produced before a measure existed has no opinion about it;
    showing that as `?` is indistinguishable from "the measure ran and could
    not decide", which invites reading a missing feature as a bad result.
    """
    if key not in fact:
        return (
            '<td class="m na" title="not recorded by this evaluation">–</td>'
        )
    return f'<td class="m">{table.get(fact.get(key), "?")}</td>'


def _team_block(evaluation: dict[str, Any] | None) -> str:
    """The specialists and tools that produced the answer above it.

    Construction sits beside content because they fail differently: a wrong
    answer from the wrong specialists is a routing defect, the same answer from
    the right ones is a reasoning defect, and a scorecard cannot tell them
    apart.
    """
    if not evaluation:
        return ""
    team = evaluation.get("team") or []
    subqueries = evaluation.get("subqueries") or {}
    if not (team or subqueries):
        return ""
    chips = "".join(
        f'<span class="chip">{html.escape(str(name))}</span>' for name in team
    ) or '<span class="missing">not recorded</span>'
    rows = "".join(
        f"<dt>{html.escape(str(specialist))}</dt>"
        f"<dd>{html.escape(str(text))}</dd>"
        for specialist, text in subqueries.items()
    )
    return (
        '<details class="team"><summary>'
        '<span class="alabel">team construction</span>'
        f"{chips}</summary>"
        + (f"<dl class=\"subq\">{rows}</dl>" if rows else "")
        + "</details>"
    )


def _facts_table(evaluation: dict[str, Any] | None) -> str:
    if not evaluation:
        return '<p class="missing">No evaluation for this system.</p>'
    facts = {row["claim_id"]: row for row in evaluation.get("fact_results") or []}
    rows = []
    for claim in evaluation.get("claims") or []:
        fact = facts.get(claim.get("claim_id"))
        # Two filters remove a claim before it is ever verified, and both must
        # stay visible. A restatement is excluded because the fact is already
        # counted; a NON-FACTUAL claim because there is nothing to falsify.
        # The second is the more dangerous call — a methodology statement like
        # "commercial cards are identified by Card Portfolio = 'SBS'" is
        # perfectly checkable against the call's arguments — and until this it
        # vanished from the page and every metric with no trace at all, which
        # is the one thing an exclusion must never do.
        if fact is None and not claim.get("restates_claim_id") and claim.get("is_factual"):
            continue
        # A restatement is deliberately not re-verified: it asserts a fact an
        # earlier claim already asserted, and scoring it twice would multiply
        # one result. It still belongs on the page — hiding it is the toggle's
        # job, not the renderer's — so it shows with empty markers.
        fact = fact if fact is not None else {}
        locator = claim.get("source_locator") or {}
        where = (
            f"r{locator['row']}c{locator['column']}"
            if locator.get("row") is not None else "¶"
        )
        stance = claim.get("stance") or "asserted"
        proposition = html.escape(str(claim.get("proposition") or ""))
        measured = [
            str(mention["measures"])
            for mention in claim.get("numeric_mentions") or []
            if mention.get("measures") and mention.get("material")
        ]
        if measured:
            proposition += "".join(
                f'<span class="measures">{html.escape(item)}</span>'
                for item in dict.fromkeys(measured)
            )
        if stance != "asserted":
            proposition = f"<em>{html.escape(stance)}</em> · {proposition}"
        restates = claim.get("restates_claim_id")
        flag = ""
        if restates:
            proposition = (
                f'<span class="restate">restates '
                f"{html.escape(str(restates))}</span> {proposition}"
            )
        elif not claim.get("is_factual"):
            proposition = (
                f'<span class="restate">not falsifiable · '
                f'{html.escape(str(claim.get("claim_type") or "?"))}</span> '
                f"{proposition}"
            )
        classes = " ".join(filter(None, [
            "halluc" if flag else "",
            "restated" if restates or not claim.get("is_factual") else "",
        ]))
        # One marker, not four. The cascade steps that produce it — supported
        # by numbers, traceable, eligible trace — are each other's inputs and
        # almost always agree, so a row of them reads as noise; the step that
        # failed is on the record and in the walkthrough when a verdict needs
        # unpicking. What the reader needs here is what the claim rests on.
        # `trace_failures` stays on the record for diagnosis; it is not shown.
        # The column answers one question — what supports this claim — and a
        # list of pointer failures beside it answered a different one.
        detail = (
            [f"cited provenance: {fact['evidence_resolution']}"]
            if fact.get("evidence_resolution") else []
        )
        gnd_cell = _marker(fact, "grounding_kind", _GROUNDING_MARKER)
        if detail and "grounding_kind" in fact:
            gnd_cell = gnd_cell.replace(
                '<td class="m"',
                f'<td class="m" title="{html.escape(" · ".join(detail))}"',
            )
        rows.append(
            f'<tr class="{classes}">'
            f'<td class="flag">{flag}</td>'
            f"{gnd_cell}"
            f'<td class="cid">{html.escape(str(claim.get("claim_id")))} '
            f'<span class="loc">{where}</span></td>'
            f"<td>{proposition}</td></tr>"
        )
    if not rows:
        return '<p class="missing">No factual claims were extracted.</p>'
    return (
        '<table class="facts"><thead><tr><th></th><th>gnd</th>'
        "<th>claim</th><th>atomic fact</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _oracle_found(row: dict[str, Any]) -> str:
    """What the answer actually said, for one expected answer.

    A numeric oracle records the figure it matched. A BOOLEAN one matches
    affirmative/negative wording instead and records no value, so keying the
    cell on `matched_value` printed "not in answer" next to a ✓ — the answer
    did state it, just not as a number.
    """
    if row.get("matched_value") is not None:
        return html.escape(str(row["matched_value"]))
    if _slug(row.get("verdict")) == "pass":
        # Matched by wording; the answer agreed with the expected value.
        return (
            f'{html.escape(str(row.get("expected")))} '
            '<span class="loc">stated</span>'
        )
    return '<span class="missing">not in answer</span>'


def _expectations_block(evaluation: dict[str, Any] | None) -> str:
    """What the answer was measured against, beside the answer itself.

    A marker column says a claim was grounded; it does not say whether the
    QUESTION was answered. For a question a script can settle that is the
    extracted value against the expected one; for one it cannot, it is which
    must-haves the answer covered.
    """
    if not evaluation:
        return ""
    oracles = evaluation.get("expected_answer_results") or []
    if oracles:
        rows = "".join(
            f"<tr title=\"{html.escape(str(row.get('reason') or ''))}\">"
            f"<td>{'✓' if row.get('verdict') == 'pass' else '✗'}</td>"
            f"<td>{html.escape(str(row.get('expected_answer_id') or ''))}</td>"
            f'<td class="num">{html.escape(str(row.get("expected")))}</td>'
            f'<td class="num">{_oracle_found(row)}</td></tr>'
            for row in oracles
        )
        return (
            '<details class="expect" open><summary>'
            '<span class="alabel">expected answer</span></summary>'
            '<table class="expect"><thead><tr><th></th><th>oracle</th>'
            "<th>expected</th><th>in answer</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></details>"
        )
    must_haves = evaluation.get("must_have_results") or []
    if not must_haves:
        return ""
    marks = {"full": "✓", "partial": "~", "miss": "✗"}
    rows = "".join(
        f"<tr><td>{marks.get(row.get('verdict'), '?')}</td>"
        f"<td>{html.escape(str(row.get('description') or row.get('must_have_id')))}"
        "</td></tr>"
        for row in must_haves
    )
    hit = sum(1 for row in must_haves if row.get("verdict") == "full")
    return (
        '<details class="expect" open><summary>'
        f'<span class="alabel">must-haves {hit}/{len(must_haves)}</span>'
        "</summary>"
        f'<table class="expect"><tbody>{rows}</tbody></table></details>'
    )


def _totals(evaluations: list[dict[str, Any]], key: str) -> float | None:
    """Sum one metric across every repeat, or None if no repeat reported it."""
    values = [
        float(value) for row in evaluations
        if (value := (row.get("metrics") or {}).get(key)) is not None
    ]
    return sum(values) if values else None


def _content_rows(
    base_runs: list[dict[str, Any]], cand_runs: list[dict[str, Any]],
) -> list[str]:
    """One row per content metric, as summed numerator over denominator."""
    rows = []
    for numerator_key, denominator_key, label, higher_is_better, style in _CONTENT_METRICS:
        cells, values = [], []
        for runs in (base_runs, cand_runs):
            numerator = _totals(runs, numerator_key)
            denominator = _totals(runs, denominator_key)
            if numerator is None or not denominator:
                cells.append("—")
                values.append(None)
                continue
            shown_n = _fmt_value(numerator, "count")
            if style == "count":
                # A claim count is compared as a count: "+55 claims" is the
                # finding, and a percentage of a moving denominator hides it.
                values.append(numerator)
                cells.append(f"{shown_n} (out of {_fmt_value(denominator, 'count')})")
            else:
                values.append(numerator / denominator)
                cells.append(
                    f"{100 * numerator / denominator:.0f}% "
                    f"({shown_n}/{_fmt_value(denominator, 'count')})"
                )
        if values[0] is None and values[1] is None:
            continue
        delta = (
            '<td class="delta"></td>' if style == "part"
            else _delta_cell(
                values[0], values[1], higher_is_better,
                "num" if style == "count" else "pct",
            )
        )
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f'<td class="num">{cells[0]}</td><td class="num">{cells[1]}</td>'
            f"{delta}</tr>"
        )
    return rows


def _metrics_table(
    base_runs: list[dict[str, Any]], cand_runs: list[dict[str, Any]],
    *, baseline: str, candidate: str, note: str,
) -> str:
    rows = _content_rows(base_runs, cand_runs)
    if not rows:
        return '<p class="missing">No metrics recorded.</p>'
    return (
        f'<p class="tnote">{html.escape(note)}</p>'
        '<table class="metrics"><thead><tr><th>Metric</th>'
        f"<th>{html.escape(baseline)}</th><th>{html.escape(candidate)}</th>"
        "<th>Δ</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def find_run_manifest(start: Path, *, depth: int = 3) -> dict[str, Any]:
    """Read the run's `manifest.json`, resolved via the run root.

    It records what the run actually was, which is the only correct source for
    which system is the baseline: inferring from the systems present means
    sorting names, and `sorted(["current", "previous"])` assigns them the wrong
    way round.
    """
    layout = RunLayout.find(start, depth=depth)
    if layout is None or not layout.manifest.is_file():
        return {}
    try:
        data = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def find_run_summary(start: Path, *, depth: int = 3) -> dict[str, Any]:
    """Read the RUN summary — the one carrying consistency/latency/memory.

    Resolved via the run root, never by walking up looking for the first
    `summary.json`: since the content cascade's own summary is
    `content/summary.json`, an upward filename search finds THAT first and
    silently renders the module tables from a file that has none of their
    metrics.
    """
    layout = RunLayout.find(start, depth=depth)
    if layout is None:
        return {}
    for candidate in (layout.summary, layout.root / "summary.json"):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def resolve_view_defaults(
    evaluations: list[dict[str, Any]], *, manifest: dict[str, Any] | None = None,
    baseline: str | None = None, candidate: str | None = None,
    mode: str | None = None,
) -> tuple[str, str, str | None, str]:
    """Settle baseline, candidate, and mode, and say where they came from."""
    manifest = manifest or {}
    systems = list(dict.fromkeys(
        str(row.get("system")) for row in evaluations if row.get("system")
    ))
    source = "explicit"
    if baseline is None and candidate is None:
        if manifest.get("baseline") or manifest.get("candidate"):
            source = "manifest.json"
        elif systems:
            source = "inferred from record order"
    baseline = baseline or manifest.get("baseline") or (
        systems[0] if systems else "baseline"
    )
    candidate = candidate or manifest.get("candidate") or (
        next((name for name in systems if name != baseline), baseline)
    )
    modes = {str(row.get("mode")) for row in evaluations if row.get("mode")}
    if mode is None:
        # A run recorded as `both` contains two modes; it is not itself a
        # filter value, so let the repeat selector pick the first present.
        manifest_mode = manifest.get("mode")
        mode = manifest_mode if manifest_mode in modes else None
    return str(baseline), str(candidate), mode, source


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary_block(
    runs_by_system: dict[str, list[dict[str, Any]]],
    question_count: int,
    groups: dict[tuple[str, str, str], dict[str, Any]],
    chosen_mode: str | None, *, baseline: str, candidate: str,
) -> str:
    """Set-level metrics over every question and every repeat.

    Content sums numerators and denominators rather than averaging per-repeat
    percentages: an average of averages cannot be shown as "n/d", and it hides
    how much each figure rests on. Module metrics stay macro-averaged over
    questions, since they are already aggregated per question by `run`.
    """
    def combine(values: list[float], kind: str) -> float | None:
        # A count is a total over the set; a rate is a macro average, so every
        # question counts once and a verbose answer cannot outvote a terse one.
        if not values:
            return None
        return sum(values) if kind == "count" else _mean(values)

    def module_value(system: str, key: str, kind: str) -> float | None:
        return combine([
            float(value)
            for (group_system, group_mode, _name), group in groups.items()
            if group_system == system
            and (chosen_mode is None or group_mode == chosen_mode)
            and (value := _dig(group, key)) is not None
        ], kind)

    blocks = []
    # A module with nothing to report still gets a block explaining why — but
    # an overview where NOTHING was measured is not an overview.
    measured = False
    content_rows = _content_rows(
        runs_by_system.get(baseline) or [], runs_by_system.get(candidate) or [],
    )
    if content_rows:
        measured = True
        blocks.append(
            '<div class="mblock"><h4>Content</h4>'
            '<p class="tnote">totalled over questions and repeats</p>'
            '<table class="metrics"><thead><tr><th>Metric</th>'
            f"<th>{html.escape(baseline)}</th><th>{html.escape(candidate)}</th>"
            "<th>Δ</th></tr></thead>"
            f"<tbody>{''.join(content_rows)}</tbody></table></div>"
        )
    for title, spec, reader, note in (
        ("Consistency", _MODULE_METRICS["consistency"], module_value,
         "macro average over questions, all repeats"),
        ("Memory", _MODULE_METRICS["memory"], module_value,
         "macro average over questions, all repeats"),
        ("Latency", _MODULE_METRICS["latency"], module_value,
         "macro average over questions, all repeats"),
    ):
        rows = []
        for key, label, higher_is_better, kind in spec:
            base_value = reader(baseline, key, kind)
            cand_value = reader(candidate, key, kind)
            if base_value is None and cand_value is None:
                continue
            rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f'<td class="num">{_fmt_value(base_value, kind)}</td>'
                f'<td class="num">{_fmt_value(cand_value, kind)}</td>'
                f"{_delta_cell(base_value, cand_value, higher_is_better, kind)}</tr>"
            )
        measured = measured or bool(rows)
        if not rows:
            # Dropping the block entirely reads as "this module was not part of
            # the evaluation". It was; it had nothing to say, and why is worth
            # a line — pairwise consistency is undefined until there are two
            # runs to compare.
            rows = [
                f"<tr><td>{html.escape(label)}</td>"
                '<td class="num">—</td><td class="num">—</td><td></td></tr>'
                for label in [item[1] for item in spec]
            ]
            note = (
                "not computed for this run"
                + (" — pairwise measures need at least 2 repeats"
                   if title == "Consistency" else "")
            )
        blocks.append(
            f'<div class="mblock"><h4>{title}</h4>'
            f'<p class="tnote">{note}</p>'
            '<table class="metrics"><thead><tr><th>Metric</th>'
            f"<th>{html.escape(baseline)}</th><th>{html.escape(candidate)}</th>"
            "<th>Δ</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    if not blocks or not measured:
        return ""
    return (
        '<section class="summary" id="overview" data-only-tab="metrics" hidden>'
        "<h3>Question set — overall</h3>"
        f'<p class="qtext">{question_count} questions</p>'
        f'<div class="tabpanel" data-tab="metrics">'
        f'<div class="mgrid">{"".join(blocks)}</div></div>'
        "</section>"
    )


def _metrics_block(
    base_runs: list[dict[str, Any]], cand_runs: list[dict[str, Any]],
    groups: dict[tuple[str, str, str], dict[str, Any]], name: str,
    chosen_mode: str | None, *, baseline: str, candidate: str,
) -> str:
    def group_for(system: str) -> dict[str, Any] | None:
        if chosen_mode is not None:
            return groups.get((system, chosen_mode, name))
        return next(
            (g for (s, _m, n), g in groups.items() if s == system and n == name),
            None,
        )

    base_group, cand_group = group_for(baseline), group_for(candidate)
    n_runs = (base_group or cand_group or {}).get("n_runs")
    over = f"across all {n_runs} repeats" if n_runs else "across all repeats"
    parts = [
        ("Content", _metrics_table(
            base_runs, cand_runs, baseline=baseline, candidate=candidate,
            note=over,
        )),
    ]
    for module in ("consistency", "memory", "latency"):
        parts.append((module.capitalize(), _module_table(
            base_group, cand_group, spec=_QUESTION_MODULE_METRICS[module],
            baseline=baseline, candidate=candidate, note=over,
        )))
    return (
        '<div class="mgrid">'
        + "".join(
            f'<div class="mblock"><h4>{html.escape(title)}</h4>{body}</div>'
            for title, body in parts
        )
        + "</div>"
    )


def select_repeat(
    evaluations: list[dict[str, Any]], *, mode: str | None, run_index: int | None,
) -> tuple[str | None, int | None]:
    """Pick the repeat to display, preferring an explicit request.

    Defaults to the lowest run index present so the same sample is shown every
    time the report is regenerated: a viewer that silently changes which run it
    displays makes two readings of "the same" report incomparable.
    """
    modes = sorted({str(row.get("mode")) for row in evaluations if row.get("mode")})
    chosen_mode = mode if mode is not None else (modes[0] if modes else None)
    indices = sorted({
        int(row["run_index"]) for row in evaluations
        if row.get("run_index") is not None
        and (chosen_mode is None or row.get("mode") == chosen_mode)
    })
    chosen_index = run_index if run_index is not None else (
        indices[0] if indices else None
    )
    return chosen_mode, chosen_index


def answer_comparison_html(
    evaluations: list[dict[str, Any]], *, baseline: str, candidate: str,
    mode: str | None = None, run_index: int | None = None,
    summary: dict[str, Any] | None = None,
) -> str:
    chosen_mode, chosen_index = select_repeat(
        evaluations, mode=mode, run_index=run_index,
    )
    # {question: {run_index: {system: evaluation}}} — every repeat is rendered
    # and switched in the browser. Sampling one made two readings of "the same"
    # report incomparable whenever the sample changed, and hid the thing k
    # repeats exist to show: whether a verdict is stable or a coin flip.
    by_repeat: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    # Every repeat, for the metrics. Content rates are summed over runs, so a
    # single sampled repeat would under-report what was actually measured.
    runs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in evaluations:
        if chosen_mode is not None and row.get("mode") != chosen_mode:
            continue
        runs[str(row.get("name"))][str(row.get("system"))].append(row)
        if row.get("run_index") is not None:
            by_repeat[str(row.get("name"))][int(row["run_index"])][
                str(row.get("system"))] = row
        if chosen_index is not None and row.get("run_index") != chosen_index:
            continue
        by_question[str(row.get("name"))][str(row.get("system"))] = row
    repeat_indices = sorted({
        int(row["run_index"]) for row in evaluations
        if row.get("run_index") is not None
        and (chosen_mode is None or row.get("mode") == chosen_mode)
    })
    # The order the questions were ASKED, not alphabetical. A stateful run is a
    # conversation: "what is the total balance of these cards?" only makes
    # sense after "how many commercial cards does the customer have?", and
    # sorting by name separates them and reverses the pair.
    def asked_at(name: str) -> tuple[float, str]:
        positions = [
            float(row["sequence_position"])
            for rows in runs[name].values() for row in rows
            if row.get("sequence_position") is not None
        ]
        # Name as tiebreak, so a run without positions still orders stably.
        return (min(positions) if positions else float("inf"), name)

    questions = sorted(by_question, key=asked_at)

    # summary.json groups are keyed by (system, mode, question) and are
    # aggregated over ALL repeats — unlike the content metrics, which
    # belong to the single sampled repeat. The tab says so.
    groups = {
        (str(g.get("system")), str(g.get("mode")), str(g.get("name"))): g
        for g in (summary or {}).get("groups") or []
    }
    sections = []
    toc = []
    for index, name in enumerate(questions):
        pair = by_question[name]
        base_eval, cand_eval = pair.get(baseline), pair.get(candidate)
        question_text = html.escape(str(
            (base_eval or cand_eval or {}).get("question") or ""
        ))
        # An ungrounded claim is what a reader needs flagged in the index:
        # either its figures reach no measurement, or the route that produced
        # them answers a different question.
        flagged = sum(
            1
            for evaluation in (base_eval, cand_eval) if evaluation
            for fact in evaluation.get("fact_results") or []
            if fact.get("grounding_kind") == "none"
        )
        toc.append(
            f'<li><a href="#q{index}" data-target="q{index}">'
            f'<span class="tlabel"><span class="tname">{html.escape(name)}</span>'
            f'<span class="tq">{question_text}</span></span>'
            + (f'<span class="tflag" title="{flagged} ungrounded claim(s)">○</span>'
               if flagged else "")
            + "</a></li>"
        )
        missing_tag = '<span class="missing">(no record)</span>'

        def panels_for(pair: dict[str, dict[str, Any]]) -> str:
            return "".join(
                f'<div class="panel"><h4>{html.escape(system)}'
                f'{"" if evaluation else " " + missing_tag}</h4>'
                '<div class="answer"><span class="alabel">raw answer</span>'
                f"<pre>{html.escape(str((evaluation or {}).get('answer') or ''))}</pre></div>"
                f"{_expectations_block(evaluation)}"
                f"{_team_block(evaluation)}"
                f"{_facts_table(evaluation)}</div>"
                for system, evaluation in (
                    (baseline, pair.get(baseline)), (candidate, pair.get(candidate)),
                )
            )

        # Every repeat, one hidden panel each. The metrics tab beside this is
        # totalled over ALL of them regardless of which repeat is on screen —
        # answers are read one at a time, rates are not.
        repeats = repeat_indices or [chosen_index]
        opened = chosen_index if chosen_index in repeats else repeats[0]
        repeat_panels = "".join(
            f'<div class="rpane" data-repeat="{idx}"'
            f'{"" if idx == opened else " hidden"}>'
            f'<div class="split">{panels_for(by_repeat[name].get(idx) or by_question[name])}</div>'
            "</div>"
            for idx in repeats
        )
        sections.append(
            f'<section class="question" id="q{index}" data-index="{index}">'
            f"<h3>{html.escape(name)}</h3>"
            f'<p class="qtext">{question_text}</p>'
            f'<div class="tabpanel" data-tab="facts">{repeat_panels}</div>'
            f'<div class="tabpanel" data-tab="metrics" hidden>'
            f"{_metrics_block(runs[name].get(baseline) or [], runs[name].get(candidate) or [], groups, name, chosen_mode, baseline=baseline, candidate=candidate)}"
            "</div></section>"
        )

    shown = [
        evaluation for pair in by_question.values() for evaluation in pair.values()
    ]
    stale = sorted({
        field
        for evaluation in shown
        for fact in evaluation.get("fact_results") or []
        for field in _VERSIONED_FIELDS if field not in fact
    })
    # No evaluator-health banner. An unmapped pointer no longer decides any
    # verdict on this page, so a warning about it described a risk the reader
    # cannot act on; `judge_error_rate` stays in `evaluations.jsonl`.
    judge_notice = ""

    notice = (
        '<div class="notice">These evaluations predate '
        + ", ".join(f"<code>{html.escape(field)}</code>" for field in stale)
        + ". Those columns read <code>–</code>: not recorded, rather than "
        "measured and unknown. Re-run <code>evaluate-content</code> to fill "
        "them in.</div>"
        if stale else ""
    )

    subtitle = " · ".join(filter(None, [
        f"baseline <code>{html.escape(baseline)}</code>",
        f"candidate <code>{html.escape(candidate)}</code>",
        f"mode <code>{html.escape(str(chosen_mode))}</code>" if chosen_mode else "",
        f"repeat <code>#{chosen_index}</code>" if chosen_index is not None else "",
        f"of {len(repeat_indices)}" if len(repeat_indices) > 1 else "",
        f"{len(questions)} questions",
    ]))
    runs_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for per_system in runs.values():
        for system, rows in per_system.items():
            runs_by_system[system].extend(rows)
    summary_block = _summary_block(
        runs_by_system, len(questions), groups, chosen_mode,
        baseline=baseline, candidate=candidate,
    )
    # The overview lives on the metrics tab, so its nav link carries the tab to
    # switch to — a link that scrolls to something hidden is a dead link.
    overview_nav = (
        '<ul class="toc toc-flat"><li>'
        '<a href="#overview" data-target="overview" data-tab="metrics">'
        '<span class="tlabel"><span class="tname">overview</span>'
        '<span class="tq">all questions and repeats</span></span>'
        "</a></li></ul>"
        if summary_block else ""
    )
    nav = (
        overview_nav + f'<ol class="toc">{"".join(toc)}</ol>' if toc
        else overview_nav or '<p class="missing">No questions.</p>'
    )
    repeat_bar = (
        '<div class="repeats" role="tablist" aria-label="repeat">'
        '<span class="rlabel">repeat</span>'
        + "".join(
            f'<button class="rtab" role="tab" data-repeat="{idx}" '
            f'aria-selected="{"true" if idx == chosen_index else "false"}">'
            f"#{idx}</button>"
            for idx in repeat_indices
        )
        + '<span class="rnote">answers only — metrics are totalled over all '
          f'{len(repeat_indices)} repeats</span></div>'
        if len(repeat_indices) > 1 else ""
    )
    return _PAGE.replace("{{REPEATS}}", repeat_bar).replace(
        "{{SUMMARY}}", summary_block).replace(
        "{{JUDGE_NOTICE}}", judge_notice,
    ).replace(
        "{{TOC}}", nav,
    ).replace(
        "{{NOTICE}}", notice,
    ).replace(
        "{{SUBTITLE}}", subtitle,
    ).replace(
        "{{SECTIONS}}", "".join(sections) or
        '<p class="missing">No evaluations matched the selected repeat.</p>'
    )


def write_answer_comparison(
    evaluations: list[dict[str, Any]], *, layout: RunLayout, baseline: str,
    candidate: str, mode: str | None = None, run_index: int | None = None,
    summary: dict[str, Any] | None = None,
) -> Path:
    layout.content_dir.mkdir(parents=True, exist_ok=True)
    path = layout.answer_comparison
    path.write_text(
        answer_comparison_html(
            evaluations, baseline=baseline, candidate=candidate,
            mode=mode, run_index=run_index, summary=summary,
        ),
        encoding="utf-8",
    )
    return path


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer comparison</title>
<style>
:root {
  --ink: #1a1a1a; --muted: #6b7280; --faint: #9ca3af;
  --line: #e5e7eb; --wash: #f9fafb; --wash2: #f6f8fa;
  --blue: #3b82f6; --blue-dark: #1d4ed8; --blue-wash: #eff6ff;
  --bad: #b91c1c; --bad-wash: #fee2e2; --good: #15803d;
  --ui: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono: "SF Mono", Menlo, Consolas, monospace;
}
html, body { margin: 0; padding: 0; }
body { font: 14px/1.45 var(--ui); color: var(--ink); background: #fff; }
header { padding: 22px 24px 0; }
header h1 { margin: 0 0 3px; font-size: 19px; font-weight: 600; }
header .sub { color: var(--muted); font-size: 12.5px; }
code {
  font-family: var(--mono); font-size: 11.5px; background: var(--wash2);
  border: 1px solid var(--line); border-radius: 3px; padding: 0 4px;
}
.toolbar {
  display: flex; justify-content: flex-end; gap: 14px;
  padding: 8px 24px 0; font-size: 12px; color: var(--muted);
}
.toolbar label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
body.hide-restated .facts tr.restated { display: none; }
.tabs { display: flex; gap: 2px; padding: 14px 24px 0; border-bottom: 1px solid var(--line); }
.tabs button {
  background: none; border: 0; border-bottom: 2px solid transparent;
  font: inherit; font-size: 13px; color: var(--muted);
  padding: 7px 14px; margin-bottom: -1px; cursor: pointer; border-radius: 4px 4px 0 0;
}
.tabs button:hover { background: var(--wash); color: var(--ink); }
.tabs button[aria-selected="true"] { color: var(--blue-dark); border-bottom-color: var(--blue); }
.notice, .legend {
  margin: 0; padding: 9px 24px; font-size: 12px; color: var(--muted);
  background: var(--wash); border-bottom: 1px solid var(--line);
}
.legend b { color: var(--ink); font-weight: 600; }
.warn-notice { background: #fffbeb; border-bottom-color: #fde68a; color: #78350f; }
.warn-notice b { color: #78350f; }
.content-grid { display: grid; grid-template-columns: 236px 1fr; gap: 26px; padding: 20px 24px 60px; }
aside { position: sticky; top: 12px; align-self: start; max-height: calc(100vh - 24px); overflow-y: auto; }
aside h2 {
  margin: 0 0 8px 8px; font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--faint);
}
.toc { list-style: none; margin: 0; padding: 0; counter-reset: q; }
.toc-flat { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
.toc-flat a::before { content: "◈"; color: var(--blue); }
.toc a {
  display: flex; gap: 8px; align-items: baseline; padding: 6px 9px;
  border-radius: 5px; text-decoration: none; color: var(--ink);
}
.toc a::before {
  counter-increment: q; content: counter(q); color: var(--faint);
  font-size: 11px; font-variant-numeric: tabular-nums; min-width: 11px;
}
.toc a:hover { background: var(--wash); }
.toc a.active { background: var(--blue-wash); }
.toc .tlabel { display: flex; flex-direction: column; min-width: 0; }
.toc .tname { font-family: var(--mono); font-size: 11.5px; color: var(--blue-dark); }
.toc .tq { font-size: 11.5px; color: var(--muted); line-height: 1.3; }
.toc .tflag { margin-left: auto; color: var(--bad); font-weight: 700; font-size: 10.5px; }
main { min-width: 0; }
.question, .summary { margin-bottom: 30px; }
.question h3, .summary h3 { font-size: 14px; font-weight: 600; margin: 0 0 2px; font-family: var(--mono); }
.qtext { margin: 0 0 12px; color: var(--muted); font-size: 13px; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 1100px) {
  .split { grid-template-columns: 1fr; }
  .content-grid { grid-template-columns: 1fr; }
}
.panel {
  background: #fff; border: 1px solid var(--line); border-radius: 6px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04); padding: 12px 14px; min-width: 0;
}
.panel h4 { margin: 0 0 9px; font-size: 12px; font-weight: 600; color: var(--blue-dark); }
.answer { background: var(--wash2); border: 1px solid var(--line); border-radius: 5px; padding: 9px 11px; margin-bottom: 12px; }
.alabel {
  display: block; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--faint); margin-bottom: 5px;
}
.answer pre { margin: 0; max-height: 320px; overflow: auto; }
.team { margin-bottom: 12px; font-size: 12px; }
.team summary { cursor: pointer; list-style: none; display: flex;
  flex-wrap: wrap; align-items: center; gap: 5px; }
.team summary::-webkit-details-marker { display: none; }
.team .alabel { margin: 0 4px 0 0; }
.chip {
  background: var(--blue-wash); color: var(--blue-dark); border-radius: 3px;
  padding: 1px 7px; font-family: var(--mono); font-size: 11px;
}
.chip.tool { background: var(--wash2); color: var(--muted); }
.subq { margin: 8px 0 0; font-size: 11.5px; }
.subq dt { font-family: var(--mono); color: var(--blue-dark); margin-top: 5px; }
.subq dd { margin: 1px 0 0 12px; color: var(--muted); }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--mono); font-size: 11.5px; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--line); }
th { color: var(--faint); font-weight: 500; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; }
.facts td.m, .facts th:nth-child(-n+5) { text-align: center; width: 30px; }
.facts .flag { color: var(--bad); font-weight: 700; width: 24px; }
.facts tr.halluc td { background: var(--bad-wash); }
.facts .cid { font-family: var(--mono); font-size: 11px; white-space: nowrap; }
.facts .loc, .facts .na { color: var(--faint); }
.facts tr.restated td { color: var(--faint); background: var(--wash); }
details.expect { margin: 6px 0 10px; }
table.expect { border-collapse: collapse; margin-top: 6px; width: 100%; }
table.expect td, table.expect th {
  padding: 3px 8px; text-align: left; font-size: 12px;
  border-bottom: 1px solid var(--line);
}
table.expect td:first-child { width: 18px; text-align: center; }
.facts tr.judge-error td { background: #fffbeb; }
.facts .warn { color: #b45309; cursor: help; }
.facts .measures {
  display: inline-block; margin-left: 6px; font-size: 10.5px;
  color: var(--blue-dark); background: var(--blue-wash);
  border-radius: 3px; padding: 0 5px; font-family: var(--mono);
}
.facts .restate {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--faint); border: 1px solid var(--line); border-radius: 3px;
  padding: 0 4px; margin-right: 5px;
}
.mblock {
  background: #fff; border: 1px solid var(--line); border-radius: 6px;
  padding: 12px 14px; margin-bottom: 12px; max-width: 640px;
}
/* Four modules read as one board rather than a column to scroll. */
.mgrid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px; max-width: 1180px;
}
.mgrid .mblock { margin-bottom: 0; max-width: none; }
@media (max-width: 1000px) { .mgrid { grid-template-columns: 1fr; } }
.mblock h4 { margin: 0 0 1px; font-size: 12px; font-weight: 600; color: var(--blue-dark); }
.tnote { margin: 0 0 8px; color: var(--faint); font-size: 11px; }
.metrics .num, .metrics .delta { text-align: right; font-variant-numeric: tabular-nums; }
.delta.up { color: var(--good); } .delta.down { color: var(--bad); }
.missing { color: var(--faint); font-style: italic; font-size: 12.5px; }
.repeats { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:10px 96px; border-bottom:1px solid #E2E8F0; background:#F4F6F9; }
.repeats .rlabel { font-size:11px; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; color:#4A5568; }
.repeats .rtab { font:inherit; font-size:13px; font-weight:600; cursor:pointer;
  padding:4px 14px; border:1px solid #E2E8F0; background:#FFFFFF; color:#4A5568;
  border-radius:2px; }
.repeats .rtab[aria-selected="true"] { background:#006FCF; border-color:#006FCF;
  color:#FFFFFF; }
.repeats .rnote { font-size:12px; color:#4A5568; margin-left:auto; }
</style></head><body>
<header>
  <h1>Answer comparison</h1>
  <div class="sub">{{SUBTITLE}}</div>
</header>
<div class="toolbar">
  <label><input type="checkbox" id="hide-restated"> hide restated claims</label>
</div>
<div class="tabs" role="tablist">
  <button role="tab" data-tab="facts" aria-selected="true">Answers &amp; atomic facts</button>
  <button role="tab" data-tab="metrics" aria-selected="false">Metrics</button>
</div>
{{REPEATS}}
{{NOTICE}}{{JUDGE_NOTICE}}
<div class="legend">
  <b>gnd</b> <code>◆</code> factual — the run recorded a route to operations
  on specific tables, and that route answers the question asked
  <code>◇</code> report — drawn from the curated report files, which resolve
  <code>○</code> neither; hover for the cited provenance and any evaluator
  failure &nbsp;·&nbsp;
  <b>elg</b> <code>✓</code> the route answers the question asked
  <code>✗</code> it answers a different one <code>?</code> no recorded
  operation to rule on &nbsp;·&nbsp;
  a row with no markers is a restatement, counted once and not re-verified
</div>
<div class="content-grid">
  <aside><h2>Questions</h2>{{TOC}}</aside>
  <main>{{SUMMARY}}{{SECTIONS}}</main>
</div>
<script>
// Highlight the question currently in view, so the panel says where you are.
const links = new Map(
  [...document.querySelectorAll('.toc a')].map(a => [a.dataset.target, a])
);
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    links.forEach(a => a.classList.remove('active'));
    links.get(entry.target.id)?.classList.add('active');
  }
}, { rootMargin: '-15% 0px -75% 0px' });
document.querySelectorAll('section.question, section.summary')
  .forEach(s => observer.observe(s));

function showTab(name) {
  tabs.forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  document.querySelectorAll('.tabpanel').forEach(panel => {
    panel.hidden = panel.dataset.tab !== name;
  });
  document.querySelectorAll('[data-only-tab]').forEach(section => {
    section.hidden = section.dataset.onlyTab !== name;
  });
}

// Repeats switch the ANSWERS only. Metrics stay totalled over every repeat —
// a rate that changed with the panel on screen would be a different metric,
// not a different view of one.
function showRepeat(index) {
  document.querySelectorAll('.repeats .rtab').forEach(b => {
    b.setAttribute('aria-selected', String(b.dataset.repeat === index));
  });
  document.querySelectorAll('.rpane').forEach(panel => {
    panel.hidden = panel.dataset.repeat !== index;
  });
}
document.querySelectorAll('.repeats .rtab').forEach(button => {
  button.addEventListener('click', () => showRepeat(button.dataset.repeat));
});

const tabs = document.querySelectorAll('.tabs button');
tabs.forEach(button => button.addEventListener('click', () => showTab(button.dataset.tab)));

// Restated claims are shown by default: hiding redundancy by default
// would hide a finding. The toggle is for reading, not for scoring —
// the metrics already count each fact once.
const hideRestated = document.getElementById('hide-restated');
hideRestated.addEventListener('change', () => {
  document.body.classList.toggle('hide-restated', hideRestated.checked);
});

// A nav link into a tab-scoped section switches tabs, then scrolls.
document.querySelectorAll('.toc a[data-tab]').forEach(link => {
  link.addEventListener('click', () => showTab(link.dataset.tab));
});
</script>
</body></html>
"""
