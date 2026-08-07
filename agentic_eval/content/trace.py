"""Per-claim reasoning routes, read off the run's own trace rather than guessed.

A claim is produced by a route, and the route has HOPS. The orchestrator briefs
a specialist; that specialist may synthesise what other specialists returned;
each of those rests on operations against specific tables, or on a memory topic
recorded when the measurement was made. Written out:

    claim -> general_specialist   synthesis of spend_payments + modeling
          -> spend_payments       summarize_by_group(spends.Amount,
                                  by Merchant Name, sum, top 10)
          -> modeling             memory topic `modeling_Spend Amount_trend`

or, with no synthesis hop:

    claim -> modeling             score_driver_values(month=2025-05)
                                  + memory topic `spend_tsr_cdss_spike_analysis`

The run records every link — `subqueries` for the brief, `trace_node`
(`specialist.<name>.round_N`) for who made each call, `specialist` and
`result.topic` on each memory entry, and a specialist's own `agent_result` for
what it returned — so the route is read, not inferred from prose.

Two passes over it, deliberately separate:

    extract   which route produced THIS claim, described from the trace
    judge     is that route eligible for the question that was asked

Splitting them keeps the description honest. Asked in one breath, a judge that
has already decided a claim is fine will describe a route that justifies it.

Every hop is audited: call ids must be ones the run made, memory topics must be
ones the ledger holds, and specialists must be ones the orchestrator dispatched.
A hop that fails is dropped rather than believed, and recorded in
`invented_steps` so the drop stays inspectable.
"""
from __future__ import annotations

import re
from typing import Any

from agentic_eval.common.coerce import _as_list, _slug

#: `specialist.modeling.round_3` — the middle segment is the specialist that
#: made the call. The orchestrator and the report agent use their own prefixes.
_SPECIALIST_NODE = re.compile(r"^specialist\.([^.]+)\.", re.IGNORECASE)

#: Argument names that say WHICH DATA an operation ran over. Surfaced beside
#: the tool name so a hop reads as "grouped spends by merchant" rather than
#: "called a tool" — which is what makes a route judgeable for eligibility at
#: all, since the wrong window and the wrong population live in these fields.
_SHAPE_ARGUMENTS = (
    "table_name", "table", "tables", "value_column", "group_column", "by",
    "columns", "op", "operation", "where", "filter", "filters", "month",
    "months", "start", "end", "period", "top_n", "limit",
)


def _operation(evidence: dict[str, Any]) -> str:
    """One call, written as the operation it performed."""
    arguments = evidence.get("arguments")
    shape = (
        {
            key: value for key, value in arguments.items()
            if key in _SHAPE_ARGUMENTS and value not in (None, "", [], {})
        }
        if isinstance(arguments, dict) else {}
    )
    inside = ", ".join(f"{key}={value!r}" for key, value in shape.items())
    return f"{evidence.get('tool')}({inside})"


def _memory_topic(evidence: dict[str, Any]) -> str | None:
    """The KB topic this memory entry recorded, if it names one."""
    result = evidence.get("result")
    topic = result.get("topic") if isinstance(result, dict) else None
    return str(topic) if topic else None


def _owner(evidence: dict[str, Any]) -> str | None:
    """Which specialist made this call, if the trace says."""
    match = _SPECIALIST_NODE.match(str(evidence.get("trace_node") or ""))
    if match:
        return match.group(1)
    node = str(evidence.get("trace_node") or "")
    return node.split(".")[0] or None


def build_trace_view(record: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """The route the run actually took, arranged for reading.

    Grouped by specialist because that is the unit a claim comes from: the
    brief it was given, the operations it ran, the memory topics it held, and
    what it reported back — the last being the hop a synthesising specialist
    reads FROM.
    """
    subqueries = dict(record.get("subqueries") or {})

    def blank(name: str) -> dict[str, Any]:
        return {
            "brief": subqueries.get(name), "calls": [], "memory": [],
            "findings": None,
        }

    specialists: dict[str, dict[str, Any]] = {name: blank(name) for name in subqueries}
    unattributed: list[dict[str, Any]] = []
    for item in ledger:
        if item.get("source_type") == "agent_result":
            # The specialist's own return: its findings, not a call it made.
            name = str(item.get("tool") or "")
            specialists.setdefault(name, blank(name))["findings"] = item.get("result")
            continue
        if item.get("source_type") == "memory":
            # Recorded when the measurement was made, possibly turns ago, so
            # its owner is stamped on the entry rather than on a trace node.
            owner = str(item.get("specialist") or "") or None
            topic = _memory_topic(item)
            if owner and topic:
                specialists.setdefault(owner, blank(owner))["memory"].append({
                    "topic": topic,
                    "evidence_id": item.get("evidence_id"),
                    "captured_at_turn": item.get("captured_at_turn"),
                })
            continue
        owner = _owner(item)
        entry = {
            "call_id": item.get("call_id") or item.get("evidence_id"),
            "evidence_id": item.get("evidence_id"),
            "operation": _operation(item),
        }
        if owner in specialists:
            specialists[owner]["calls"].append(entry)
        else:
            unattributed.append({**entry, "ran_by": owner})
    return {
        "question": record.get("question"),
        "team": list(record.get("team") or []),
        "specialists": specialists,
        "other_calls": unattributed,
    }


def known_call_ids(trace_view: dict[str, Any]) -> set[str]:
    """Every call the run actually made, for auditing a cited route."""
    ids = {
        str(call.get("call_id"))
        for entry in trace_view["specialists"].values()
        for call in entry["calls"]
    }
    ids |= {str(call.get("call_id")) for call in trace_view["other_calls"]}
    return {value for value in ids if value and value != "None"}


def known_memory_topics(trace_view: dict[str, Any]) -> set[str]:
    """Every KB topic the run actually carried."""
    return {
        str(entry["topic"])
        for spec in trace_view["specialists"].values()
        for entry in spec["memory"] if entry.get("topic")
    }


def audit_claim_traces(
    raw: Any, trace_view: dict[str, Any], claim_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Keep only hops the run really took.

    A call id absent from the trace, a memory topic the ledger does not hold,
    or a specialist the orchestrator never dispatched is a hop the model
    composed. Dropping the HOP rather than the whole route keeps a
    partially-correct description usable, and `invented_steps` records what
    was removed.
    """
    known_calls = known_call_ids(trace_view)
    known_topics = known_memory_topics(trace_view)
    known_specialists = set(trace_view["specialists"])
    traces: dict[str, dict[str, Any]] = {}
    for row in _as_list(raw):
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id") or "")
        if claim_id not in claim_ids:
            continue
        # A flat `{specialist, call_ids}` is still read, as a one-hop route, so
        # an older evaluation and a judge answering in the simpler shape both
        # keep working.
        hops = _as_list(row.get("route")) or [{
            "specialist": row.get("specialist"),
            "call_ids": row.get("call_ids"),
            "memory_topics": row.get("memory_topics"),
            "note": row.get("derivation"),
        }]
        kept: list[dict[str, Any]] = []
        invented: list[str] = []
        for hop in hops:
            if not isinstance(hop, dict):
                continue
            specialist = str(hop.get("specialist") or "") or None
            if specialist and specialist not in known_specialists:
                invented.append(f"specialist:{specialist}")
                specialist = None
            calls = [str(v) for v in _as_list(hop.get("call_ids"))]
            topics = [str(v) for v in _as_list(hop.get("memory_topics"))]
            sources = [
                str(v) for v in _as_list(hop.get("from_specialists"))
                if str(v) in known_specialists
            ]
            invented += [f"call:{v}" for v in calls if v not in known_calls]
            invented += [f"topic:{v}" for v in topics if v not in known_topics]
            kept_calls = [v for v in calls if v in known_calls]
            kept_topics = [v for v in topics if v in known_topics]
            if not (specialist or kept_calls or kept_topics or sources):
                continue
            kept.append({
                "specialist": specialist,
                "kind": _slug(hop.get("kind")) or (
                    "synthesis" if sources
                    else "memory" if kept_topics and not kept_calls
                    else "operation"
                ),
                "from_specialists": sources,
                "call_ids": kept_calls,
                "memory_topics": kept_topics,
                "operations": [str(v) for v in _as_list(hop.get("operations"))],
                "note": str(hop.get("note") or ""),
            })
        traces[claim_id] = {
            "route": kept,
            # Flattened, because everything downstream asks a route the same
            # two questions: did the run record it, and is it eligible.
            "call_ids": [v for hop in kept for v in hop["call_ids"]],
            "memory_topics": [v for hop in kept for v in hop["memory_topics"]],
            "specialist": next(
                (hop["specialist"] for hop in kept if hop["specialist"]), None,
            ),
            "invented_steps": invented,
            "derivation": str(row.get("derivation") or ""),
        }
    return traces


def apply_trace_eligibility(
    raw: Any, traces: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fold the eligibility verdict onto each claim's route."""
    for row in _as_list(raw):
        if not isinstance(row, dict):
            continue
        trace = traces.get(str(row.get("claim_id") or ""))
        if trace is None:
            continue
        verdict = _slug(row.get("verdict"))
        # What the judge actually said, kept whatever happens below. A claim
        # whose measurement was made in an EARLIER turn and carried forward as
        # specialist memory has no call_id in this turn's trace, yet it has
        # real provenance and the judge can rule on it perfectly well. Only
        # the caller — which can see the evidence — knows which case this is.
        trace["ruled_eligible"] = verdict if verdict in {"yes", "no"} else "unavailable"
        recorded = bool(trace.get("call_ids") or trace.get("memory_topics"))
        if not recorded:
            # Nothing in the run produced the claim — no operation this turn,
            # no memory topic from an earlier one. Default to "we cannot
            # tell": the prompt says return UNAVAILABLE and the judge sometimes
            # returns NO anyway, and "we cannot tell" and "this answers the
            # wrong question" are different findings, only one of which is
            # about the answer.
            verdict = "unavailable"
        trace["eligible"] = (
            verdict if verdict in {"yes", "no"} else "unavailable"
        )
        trace["eligibility_reason"] = str(row.get("reason") or "")
        trace["no_recorded_operation"] = not recorded
    for trace in traces.values():
        # A claim the judge skipped is unknown, never ineligible.
        trace.setdefault("eligible", "unavailable")
        trace.setdefault("eligibility_reason", "")
        trace.setdefault(
            "no_recorded_operation",
            not (trace.get("call_ids") or trace.get("memory_topics")),
        )
    return traces
