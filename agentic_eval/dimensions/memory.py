"""Memory module: whether a memory-requiring question actually used it."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from agentic_eval.common.stats import _optional_rate


def score_memory(record: dict[str, Any]) -> dict[str, Any]:
    """Record whether a question that requires prior information used memory."""
    evaluation = record.get("evaluation") or {}
    required = evaluation.get("memory_required")
    if not isinstance(required, bool):
        required = None
    # ARRIVAL, not use. Whether the run drew on any of this is a separate
    # judged measurement (`memory_leverage_rate` in the content metrics), and
    # conflating the two is what pinned this at 100%: `memory_context_exposed`
    # fires on a KB header string the system emits every turn, so a session's
    # FIRST question — with nothing yet to remember — scored a hit.
    #
    # Counted from the concrete memory the trace shows was offered: named KP
    # topics, decoded prior turns, a replayed cached answer. NOT from the
    # `kb_digest_present` tag, which one system sets on turns where the digest
    # is absent entirely — the tag says the hook ran, not that anything
    # arrived.
    used = any((
        record.get("qa_cache_hit") is True,
        bool(record.get("kb_topics_exposed")),
        bool(record.get("episodic_turns_exposed")),
    ))
    return {
        "memory_required": required,
        "memory_used": used,
        "memory_hit": used if required is True else None,
    }



def section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Memory metrics for the k repeats of one cell.

    `memory_hit_rate` counts only questions explicitly annotated
    `memory_required: true`. Unannotated questions are excluded rather than
    scored as misses, so the rate never depends on how many questions happen
    not to need memory.
    """
    kb_calls = sum(
        int(row["kb_lookup_calls"]) for row in rows
        if row.get("kb_lookup_calls") is not None
    )
    kb_hits = sum(
        int(row["kb_lookup_hits"]) for row in rows
        if row.get("kb_lookup_hits") is not None
    )
    annotated_memory = [row for row in rows if row.get("memory_required") is not None]
    memory_annotations = {bool(row.get("memory_required")) for row in annotated_memory}
    required_memory = [row for row in rows if row.get("memory_required") is True]
    return {
        "qa_cache_hit_rate": _optional_rate(rows, "qa_cache_hit"),
        "episodic_context_exposure_rate": _optional_rate(
            rows, "episodic_context_exposed"
        ),
        "memory_context_exposure_rate": _optional_rate(
            rows, "memory_context_exposed"
        ),
        "memory_required": (
            next(iter(memory_annotations))
            if len(memory_annotations) == 1 else None
        ),
        # A property of the QUESTION, so it counts once however many cases and
        # repeats asked it. Summed over a system's groups this is "how many
        # questions in the set need memory" — a fact about the suite, which is
        # the only reading that stays stable as k or the case list changes.
        # `memory_required_run_count` is the denominator behind `memory_hit_rate`
        # and is kept for that, not for display.
        "memory_required_question_count": (
            1 if next(iter(memory_annotations), None) is True else 0
        ),
        "memory_required_run_count": len(required_memory),
        "memory_used_count": sum(
            bool(row.get("memory_used")) for row in required_memory
        ),
        "memory_hit_rate": (
            sum(bool(row.get("memory_used")) for row in required_memory)
            / len(required_memory)
            if required_memory else None
        ),
        "kb_context_exposure_rate": (
            _optional_rate([
                {**row, "_exposed": bool(row.get("kb_context_exposures"))}
                for row in rows if row.get("kb_context_exposures") is not None
            ], "_exposed")
        ),
        "kb_lookup_hit_rate": kb_hits / kb_calls if kb_calls else None,
    }


def groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Memory hit rate pooled per system/mode, across all questions."""
    by_system_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_system_mode[(row["system"], row["mode"])].append(row)
    out = []
    for (system, mode), rows in sorted(by_system_mode.items()):
        required = [row for row in rows if row.get("memory_required") is True]
        out.append({
            "system": system,
            "mode": mode,
            "required_run_count": len(required),
            "memory_used_count": sum(bool(row.get("memory_used")) for row in required),
            "memory_hit_rate": (
                sum(bool(row.get("memory_used")) for row in required) / len(required)
                if required else None
            ),
        })
    return out
