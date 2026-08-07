"""The evidence ledger: what the system actually measured, and how good it is."""
from __future__ import annotations

import json
from typing import Any

from agentic_eval.common.coerce import (
    _as_list, _deep_decode_json, _labelled_float, _slug,
)


def _merge_duplicate_evidence(kept: dict[str, Any], duplicate: dict[str, Any]) -> None:
    """Fold a second capture of one call into the ledger entry already kept.

    A single call reaches the ledger twice: once from the SSE `agent_completed`
    event (`agent:<call_id>`, carrying scope/measured_over) and once from the
    trace database (`trace:<turn>:<call_id>`, carrying arguments/trace_node).
    Two entries for one call inflate the judge payload and let two claims cite
    "different" evidence for one measurement, so keep one entry, union the
    provenance, and record the alias so either id still resolves.
    """
    aliases = kept.setdefault("duplicate_evidence_ids", [])
    alias = str(duplicate.get("evidence_id") or "")
    if alias and alias != kept.get("evidence_id") and alias not in aliases:
        aliases.append(alias)
    for key, value in duplicate.items():
        if key in {"evidence_id", "duplicate_evidence_ids", "source_type"}:
            continue
        if kept.get(key) in (None, "", [], {}):
            kept[key] = value
    # An agent-level call is never primary tool evidence, whichever capture
    # happened to arrive first, so the weaker classification wins.
    if "agent_result" in {kept.get("source_type"), duplicate.get("source_type")}:
        kept["source_type"] = "agent_result"


#: Sources that carry a MEASUREMENT. A specialist KB entry is a figure the
#: system measured in an earlier turn and remembered — recalling your own work
#: is not the same as asserting from nowhere, and excluding it left every
#: memory-backed claim with no route at all.
MEASURED_SOURCES = ("tool_result", "memory")


#: Tool names that hand back curated report material rather than a fresh
#: measurement: the report-writing specialist, and any file reader replaying a
#: report the run wrote earlier. Matched as substrings so `fs_read_file`,
#: `read_file` and `report_agent` are all caught without a per-system list.
_REPORT_TOOLS = ("report", "fs_read", "read_file", "file_read")


def _is_report_evidence(item: dict[str, Any]) -> bool:
    """Is this entry curated report material rather than a live measurement?

    Needed because "does the claim rest on a report?" is a question about
    PROVENANCE, and the only test we had was about phrasing — whether the
    sentence said "the report states…". The two come apart in both directions:
    a claim reading "the other cards are in the CPS portfolio" mentions no
    report yet cites nothing but one, while "no further commercial cards …
    according to both live data and curated reports" names reports and cites
    `query_table`.
    """
    if str(item.get("source_type") or "") == "report":
        return True
    tool = str(item.get("tool") or "").lower()
    return any(token in tool for token in _REPORT_TOOLS)


def claim_is_report_only(
    claim: dict[str, Any], raw_fact: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Does this claim rest on curated report material and nothing else?

    `all`, not `any`: a claim the live data also confirms is corroborated by
    the report, not dependent on it, and keeps counting as factual support.

    Used to decide, before the eligibility call is made, whether asking about
    this claim's route is worth a verdict. It is not — a report claim grounds
    on the report resolving, and its eligibility cannot change that.
    """
    if claim.get("report_attribution"):
        return True
    ids = {str(v) for v in _as_list(raw_fact.get("evidence_ids"))}
    ids |= {
        str(number.get("evidence_id"))
        for number in _as_list(raw_fact.get("numbers"))
        if isinstance(number, dict) and number.get("evidence_id")
    }
    cited = [evidence_by_id[value] for value in ids if value in evidence_by_id]
    return bool(cited) and all(_is_report_evidence(item) for item in cited)


def _reaches_operations(item: dict[str, Any]) -> bool:
    """Does the chain from this evidence reach specific tables and operations?

    Grounding is a walk, not a lookup: a claim rests on a specialist's
    findings, and those findings rest on the operations the specialist ran.
    Two links are recorded, so two ways to land:

      * a tool result that names its own call — the walk is already there
      * a specialist's findings carrying `measured_over`, which lists the
        operations behind them ("summarize_by_group(spends.Amount, by Merchant
        Name, where Date contains '2025-05')")

    A specialist's findings with no `measured_over` state a conclusion without
    saying what produced it, and the walk stops. That is an instrumentation
    gap upstream, not proof the answer invented anything — which is why it
    reads `unavailable` rather than counting against the system.
    """
    if item.get("source_type") in MEASURED_SOURCES:
        # The call itself is the operation. Requiring captured `arguments`
        # would fail a real measurement for a gap in the harness rather than
        # anything about the answer.
        return not _is_report_evidence(item)
    return bool(item.get("measured_over"))


def _evidence_tier(item: dict[str, Any]) -> str:
    """Rank one ledger entry as independent evidence for a qualitative claim.

    PRIMARY is curated ground truth, or a tool result that is STRUCTURED — a
    measurement with addressable fields. SECONDARY is everything else the run
    produced about itself: another agent's findings, and equally a tool result
    that is just a prose blob.

    The prose rule is not a technicality, it is the same principle
    `_evidence_float` already enforces one layer down ("a prose sentence is not
    a measurement"), and it is what keeps this tier honest without a per-system
    tool list. Observed: `fs_read_file` reading back `modeling_exp_0.md` — a
    markdown report THE SYSTEM ITSELF WROTE earlier in the run — was typed
    `tool_result` merely because the tool is not a planned team member, so the
    answer's top-level conclusion scored as grounded in a measurement when it
    was grounded in the system restating itself.
    """
    source_type = item.get("source_type")
    if source_type in {"canonical_fact", "memory"}:
        return "primary"
    if source_type != "tool_result":
        return "secondary"
    result = item.get("result")
    if isinstance(result, (dict, list)):
        return "primary"
    # A scalar-returning data tool formats its answer for a human:
    # "sum(Balance) filtered by ... = $174,897.36 (over 1 row)". That is a
    # measurement with a value in it, and demoting every such tool alongside a
    # file-reader replaying the system's own report was too blunt — it made
    # `aggregate_column` and a curated markdown blob indistinguishable.
    return "primary" if _labelled_float(str(result or "")) is not None else "secondary"


def build_evidence_ledger(record: dict[str, Any], rubric: dict[str, Any]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    by_call: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_as_list(record.get("evidence")), 1):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        if "result" in item:
            item["result"] = _deep_decode_json(item["result"])
        item.setdefault("evidence_id", f"system_ev_{index:03d}")
        item.setdefault("source_system", record.get("system"))
        call_id = str(item.get("call_id") or "").strip()
        kept = by_call.get(call_id) if call_id else None
        if kept is not None:
            _merge_duplicate_evidence(kept, item)
            continue
        if call_id:
            by_call[call_id] = item
        ledger.append(item)
    facts = rubric.get("reference_facts") or rubric.get("canonical_facts") or []
    for index, fact in enumerate(_as_list(facts), 1):
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("id") or f"canonical_{index:03d}")
        ledger.append({
            "evidence_id": f"canonical:{fact_id}",
            "source_type": "canonical_fact",
            "source_system": "benchmark",
            "proposition": fact.get("proposition"),
            "scope": fact.get("scope"),
            "result": {
                "expected": fact.get("expected"),
                "proposition": fact.get("proposition"),
                "scope": fact.get("scope"),
            },
            "importance": fact.get("importance"),
        })
    # After merging, so a duplicate that demotes source_type is reflected.
    for item in ledger:
        item["evidence_tier"] = _evidence_tier(item)
    return ledger


def _compact_for_judge(item: dict[str, Any]) -> dict[str, Any] | None:
    """Shrink an entry for the first-pass judge, or drop it.

    The judge needs to know a knowledge point EXISTS and be able to cite it;
    it does not need eighteen monthly rows inline. Verification reads the full
    entry from the ledger afterwards, so nothing is lost by summarising here —
    and memory was a quarter to a half of the payload.

    An earlier turn's own answer is dropped outright: it is recalled prose,
    excluded from grounding by design, and the conversation is already passed
    separately to the eligibility judge.
    """
    source = item.get("source_type")
    if source == "memory_recall":
        return None
    if source != "memory":
        return item
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    numbers = result.get("numbers") if isinstance(result.get("numbers"), list) else []
    periods = [
        str(row.get("period")) for row in numbers
        if isinstance(row, dict) and row.get("period")
    ]
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": source,
        "tool": item.get("tool"),
        "specialist": item.get("specialist"),
        # The knowledge point itself: what was remembered, over what span.
        "topic": result.get("topic"),
        "claim": result.get("claim"),
        "covers": f"{periods[0]}..{periods[-1]}" if periods else None,
        "n_numbers": len(numbers),
        "note": "summarised; cite this id and the full figures are read from the ledger",
    }


def _bounded_ledger(ledger: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    out = []
    used = 0
    for item in ledger:
        item = _compact_for_judge(item)
        if item is None:
            continue
        encoded = json.dumps(item, default=str, ensure_ascii=False)
        if used >= max_chars:
            break
        if len(encoded) + used > max_chars:
            keep = max(500, max_chars - used)
            item = {
                "evidence_id": item.get("evidence_id"),
                "source_type": item.get("source_type"),
                "tool": item.get("tool"),
                "scope": item.get("scope"),
                "measured_over": item.get("measured_over"),
                "result_truncated": encoded[:keep],
            }
            encoded = json.dumps(item, default=str, ensure_ascii=False)
        out.append(item)
        used += len(encoded)
    return out


def _call_signature(evidence: dict[str, Any]) -> str:
    """What was ASKED of the tool, never what came back.

    Matching a rubric constraint against the whole evidence blob passes on any
    coincidence — a table name mentioned inside a prose report satisfies a
    check about which table was queried. Only the tool name, its (deeply
    decoded) arguments, and the declared provenance describe the call itself.
    """
    return json.dumps({
        "tool": evidence.get("tool"),
        "arguments": _deep_decode_json(evidence.get("arguments")),
        "scope": evidence.get("scope"),
        "measured_over": evidence.get("measured_over"),
    }, default=str).lower()
