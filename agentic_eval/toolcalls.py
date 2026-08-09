"""Reading tool calls, and whether they came back with anything.

A tool call that could not run is still a completed call: the turn succeeds,
the answer gets written, and every count of "calls made" includes it. On a real
run `previous` issued 26 trend calls and 13 of them resolved no column at all
— while reporting

    trend(max(derog_count) by month on month) = (no parseable month values;
    26 total in bureau_data; 0 row(s) had unrecognized month format)

which blames the dates for a column that never resolved. Nothing downstream
noticed: the answer was built from the half that worked, and consistency
counted two identically-empty repeats as agreement.

Success means the call RAN, not that it found rows. A query that scanned 357
payments and matched none ran perfectly — "no returned payments" is the true
answer to a1, and treating that zero as a failure scored both systems at 0% on
it. A zero is a measurement.

Where the result says neither the call is `unknown` rather than assumed good —
a rate over calls we could actually judge is worth more than one padded with
guesses.
"""
from __future__ import annotations

import json
from typing import Any

#: Prose a computing tool emits when it could NOT run. Every observed failure
#: arrives this way — a structured payload always means the call executed:
#:
#:   trend(max(derog_count) by month on month) = (no parseable month values…)
#:   trend(sum(derog_count) …) = (COLUMN NOT FOUND: 'derog_count' is not a
#:                                 column of 'bureau_data' for this case)
#:   File not found: payment_spend_exp_0.md
#:
#: Checked across a whole run's payloads: none carried an `error` field, so
#: structure is a reliable signal of execution.
_FAILURE_PROSE = (
    "no parseable", "column not found", "not found", "no such column",
    "no such table", "unknown column", "unknown table",
)


def payload(value: Any) -> dict[str, Any] | None:
    """A tool result as a mapping, parsing JSON text; None if it is neither."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def outcome(result: Any) -> str:
    """`data`, `empty`, or `unknown` — did the call RUN?

    Not "did it find rows". A query that scanned 357 payments and matched none
    ran perfectly: zero returned payments is the true answer to "did the
    customer have any payment returns?", and scoring it as a failure marked
    both systems 0% on that question. A zero is a measurement.

    So a structured payload means the tool executed, whatever the counts in it.
    A failure cannot produce one — it comes back as prose saying the column or
    file was not there.
    """
    if payload(result) is not None:
        return "data"
    text = str(result or "").strip().lower()
    if not text:
        return "unknown"
    if any(mark in text for mark in _FAILURE_PROSE):
        return "empty"
    # Prose that reports something (a file's contents, a KB topic) — it ran,
    # but nothing here counts what it found.
    return "unknown"


def iter_results(record: dict[str, Any]):
    """Every tool result in a record, batch calls yielded one spec at a time.

    A batch reports per-spec outcomes in `results`, so counting the batch as
    one call would hide that three of its four specs came back empty.
    """
    for item in record.get("evidence") or []:
        if item.get("source_type") not in {"tool_result", "unclassified_tool_result"}:
            continue
        parsed = payload(item.get("result"))
        entries = (parsed or {}).get("results")
        if isinstance(entries, list) and entries:
            for entry in entries:
                entry = entry if isinstance(entry, dict) else {}
                yield item.get("tool"), entry.get("value_column"), entry.get("result")
        else:
            arguments = item.get("arguments") or {}
            yield (
                item.get("tool"),
                arguments.get("value_column") if isinstance(arguments, dict) else None,
                item.get("result"),
            )


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many tool calls across these runs returned data, nothing, or unknown."""
    tally = {"data": 0, "empty": 0, "unknown": 0}
    for record in rows:
        for _tool, _column, result in iter_results(record):
            tally[outcome(result)] += 1
    return tally
