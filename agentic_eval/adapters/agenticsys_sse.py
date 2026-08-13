"""Black-box adapter for AgenticSys's POST-turn + SSE protocol."""
from __future__ import annotations

import json
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from agentic_eval.adapters.base import SystemAdapter
from agentic_eval.models import AdapterResult, RunRequest


_MEASURED_TOOL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_AUXILIARY = {"report_agent", "general_specialist"}


def iter_sse(
    stream: BinaryIO, *, deadline: float | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse an SSE byte stream into ``(event, JSON payload)`` tuples.

    The deadline is enforced HERE, per line, not by the caller per event.
    Heartbeats arrive as SSE comments (`: ping`) and are skipped below, so they
    never reach a consumer — a caller checking the clock once per yielded event
    therefore never checks it at all while a server dribbles keepalives. Each
    read stays under the socket timeout, nothing raises, and the turn runs
    unbounded: measured once at 34626s against a 600s limit.
    """
    event = "message"
    data_lines: list[str] = []
    for raw in stream:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("SSE stream exceeded the turn deadline")
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                data_text = "\n".join(data_lines)
                try:
                    payload = json.loads(data_text)
                except json.JSONDecodeError:
                    payload = {"_raw": data_text}
                yield event, payload if isinstance(payload, dict) else {"data": payload}
            event, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        data_text = "\n".join(data_lines)
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            payload = {"_raw": data_text}
        yield event, payload if isinstance(payload, dict) else {"data": payload}


def _json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if value else fallback
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _tags(row: dict[str, Any]) -> set[str]:
    value = _json(row.get("tags"), [])
    return {str(item) for item in value} if isinstance(value, list) else set()


def _trace_evidence(rows: list[dict[str, Any]], turn_id: str) -> list[dict[str, Any]]:
    """Recover paired function calls/results from stored LLM input messages."""
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    for row in rows:
        messages = _json(row.get("messages_json"), [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            kind = str(message.get("type") or "")
            # Two shapes coexist in one trace. The orchestrator's calls to its
            # specialists are stored Responses-style (`function_call` /
            # `function_call_output` items keyed by `call_id`). The
            # specialists' own DATA-tool calls, one level down, are stored
            # chat-completions-style: an `assistant` message carrying
            # `tool_calls`, answered by a `tool` message keyed by
            # `tool_call_id`. Reading only the first shape meant the evidence
            # ledger contained nothing but agent-to-agent calls — every entry
            # typed `agent_result`, no number ever grounded in a tool output,
            # and numeric grading permanently "unavailable".
            call_id = str(
                message.get("call_id") or message.get("tool_call_id")
                or message.get("id") or ""
            )
            if kind == "function_call" and call_id:
                calls[call_id] = {
                    "tool": str(message.get("name") or "?"),
                    "arguments": _json(message.get("arguments"), message.get("arguments")),
                    "node": row.get("node"),
                }
            elif kind == "function_call_output" and call_id:
                output = message.get("output")
                results[call_id] = _json(output, output)
            elif message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    nested = str(call.get("id") or "")
                    function = call.get("function") or {}
                    if not nested or not isinstance(function, dict):
                        continue
                    calls[nested] = {
                        "tool": str(function.get("name") or "?"),
                        "arguments": _json(
                            function.get("arguments"), function.get("arguments"),
                        ),
                        "node": row.get("node"),
                    }
            elif message.get("role") == "tool" and call_id:
                output = message.get("content")
                results[call_id] = _json(output, output)
    evidence = []
    for call_id, result in results.items():
        call = calls.get(call_id, {})
        evidence.append({
            "evidence_id": f"trace:{turn_id}:{call_id}",
            "source_type": "unclassified_tool_result",
            "call_id": call_id,
            "tool": call.get("tool", "?"),
            "arguments": call.get("arguments"),
            "result": result,
            "trace_node": call.get("node"),
        })
    return evidence


#: `- bureau_FICO Score_trend: 2023-07 to 2025-06: FICO 654.0 -> 764.0`
_KB_TOPIC_LINE = re.compile(r"^\s*-\s+([A-Za-z0-9_][^:\n]{2,80}?):\s", re.M)


def _unescaped(message_blob: str) -> str:
    """The prompt text as the model saw it.

    `messages_json` stores the conversation JSON-encoded, so the episodic block
    arrives with escaped quotes and newlines — not parseable, and not matchable
    by a pattern written against the rendered text. Decoding once here means
    both readers work on what was actually in the prompt.
    """
    try:
        return message_blob.encode("utf-8", "ignore").decode("unicode_escape")
    except (UnicodeDecodeError, ValueError):
        return message_blob


#: Marks the start of the episodic digest; the JSON array follows on the next
#: line. Matched by a DECODER rather than a bracket pattern — the array holds
#: nested `sub_answers` lists, so `\[.*?\]` stops at the first inner `]` and
#: yields a fragment that will not parse.
_EPISODIC_MARKER = "[EPISODIC"


def _episodic_turns(message_blob: str) -> list[dict[str, Any]]:
    """The prior turns the run was shown, as {turn_id, question}.

    Decoded from the prompt rather than re-derived from the session, because
    what matters is what this turn was actually offered — a turn the system was
    never shown cannot have been leveraged, and crediting it would make the
    metric measure the harness rather than the system.
    """
    blob = _unescaped(message_blob)
    decoder = json.JSONDecoder()
    turns: dict[str, dict[str, Any]] = {}
    start = blob.find(_EPISODIC_MARKER)
    while start >= 0:
        opening = blob.find("[", blob.find("\n", start))
        if opening > 0:
            try:
                entries, _ = decoder.raw_decode(blob, opening)
            except ValueError:
                entries = []
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, dict) and entry.get("turn_id"):
                    turns.setdefault(str(entry["turn_id"]), {
                        "turn_id": str(entry["turn_id"]),
                        "question": str(entry.get("question") or "")[:300],
                    })
        start = blob.find(_EPISODIC_MARKER, start + 1)
    return list(turns.values())


def _kb_topics(message_blob: str) -> list[str]:
    """The KP topics offered in the KB-warmth digest."""
    blob = _unescaped(message_blob)
    start = blob.find("[KB-warmth")
    if start < 0:
        start = blob.find("[KB \u2014")
    if start < 0:
        return []
    window = blob[start:start + 12000]
    seen: dict[str, None] = {}
    for match in _KB_TOPIC_LINE.finditer(window):
        seen.setdefault(match.group(1).strip(), None)
    return list(seen)


def _memory_evidence(conn: Any, turn_id: str) -> list[dict[str, Any]]:
    """The memory the run had in hand, as citable evidence.

    A specialist KB entry is a measurement the system made in an EARLIER turn
    and carried forward: `{"topic": "modeling_credit_loss_prob_trend", "claim":
    "...", "numbers": [{"period": "2024-01", "credit_loss_prob": 0.8,
    "threshold": 10}, ...]}`. Without it in the ledger a claim resting on
    remembered data has no route at all, and the trace pass can only report
    "no recorded operation" — which reads as an answer asserting from nowhere
    when the system was in fact recalling its own work.

    Read from the PREVIOUS turn's snapshot, never this turn's. A snapshot is
    taken after its turn completes, so the row keyed to this turn already
    contains what this turn just concluded — grounding turn 2's "TSR peaked at
    39.6 … above the risky threshold" in a KB entry turn 2 itself wrote. That
    is the system citing itself, and it would make every answer its own
    evidence. What was available TO a turn is what existed before it ran.
    """
    try:
        columns = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(session_snapshot)",
            ).fetchall()
        }
    except sqlite3.Error:
        return []
    if not {"turn_id", "specialist_kb_json"} <= columns:
        return []
    try:
        current = conn.execute(
            "SELECT id, chat_id FROM session_snapshot WHERE turn_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (turn_id,),
        ).fetchone()
        if current is None:
            return []
        row = conn.execute(
            "SELECT specialist_kb_json, qa_cache_json FROM session_snapshot "
            "WHERE id < ? AND chat_id IS ? ORDER BY id DESC LIMIT 1",
            (current["id"], current["chat_id"]),
        ).fetchone()
    except sqlite3.Error:
        return []
    # The first turn of a session had no memory to draw on.
    if row is None:
        return []
    evidence: list[dict[str, Any]] = []
    # One knowledge point, one entry. The KB records the same measurement more
    # than once — two specialists both storing the TSR trend, or one storing
    # its own topic twice across rounds — and two evidence ids for one figure
    # inflate the payload and let two claims cite "different" provenance for
    # the same measurement. Same rule as `_merge_duplicate_evidence` applies to
    # tool calls; the other recorders are kept so attribution is not lost.
    seen: dict[str, dict[str, Any]] = {}
    knowledge = _json(row["specialist_kb_json"], {})
    if isinstance(knowledge, dict):
        for specialist, entries in knowledge.items():
            for index, entry in enumerate(entries if isinstance(entries, list) else []):
                if not isinstance(entry, dict):
                    continue
                topic = str(entry.get("topic") or f"{specialist}_{index}")
                fingerprint = json.dumps(
                    {"claim": entry.get("claim"), "numbers": entry.get("numbers")},
                    sort_keys=True, default=str,
                )
                kept = seen.get(fingerprint)
                if kept is not None:
                    kept.setdefault("also_recorded_by", []).append(
                        f"{specialist}:{topic}",
                    )
                    continue
                # One topic can hold SEVERAL measurements — the same analysis
                # re-run on a wider scope in a later turn, so
                # `Amount_by_Merchant Name` appears with 6 figures and again
                # with 11. Keying the id on the topic alone collided them, and
                # `evidence_by_id` kept whichever came last: a claim citing the
                # early reading silently resolved against the late one.
                captured = str(entry.get("captured_at_turn") or index)
                item = {
                    "evidence_id": f"memory:{specialist}:{topic}@{captured}",
                    "captured_at_turn": entry.get("captured_at_turn"),
                    "source_call": entry.get("source_call"),
                    "source_type": "memory",
                    "tool": "specialist_kb",
                    "specialist": specialist,
                    "arguments": {"topic": topic},
                    "result": entry,
                }
                seen[fingerprint] = item
                evidence.append(item)
    cache = _json(row["qa_cache_json"], []) if "qa_cache_json" in columns else []
    for index, entry in enumerate(cache if isinstance(cache, list) else []):
        if not isinstance(entry, dict):
            continue
        # An earlier turn's own answer. Recalled prose, not a measurement, so
        # it is typed apart from the KB.
        evidence.append({
            "evidence_id": f"memory:qa:{entry.get('turn_id') or index}",
            "source_type": "memory_recall",
            "tool": "qa_cache",
            "arguments": {"question": entry.get("question")},
            "result": entry,
        })
    return evidence


def _event_fields(events: list[tuple[str, dict]]) -> dict[str, Any]:
    plans = [
        payload.get("tool_calls") or []
        for name, payload in events if name == "team_plan"
    ]
    plan = plans[-1] if plans else []
    team = [str(call.get("tool") or "?") for call in plan]
    subqueries = {
        str(call.get("tool") or "?"): str(call.get("sub_question") or "")
        for call in plan
    }
    completed: dict[str, tuple[str, dict]] = {}
    for name, payload in events:
        if name != "agent_completed":
            continue
        body = payload.get("payload")
        body = body if isinstance(body, dict) else _json(body, {})
        completed[str(payload.get("call_id") or len(completed))] = (
            str(payload.get("tool") or "?"), body,
        )

    scopes: list[str] = []
    measured: list[str] = []
    tools: set[str] = set()
    evidence: list[dict[str, Any]] = []
    eligible = complete = 0
    for call_id, (tool, body) in completed.items():
        evidence.append({
            "evidence_id": f"agent:{call_id}",
            "source_type": "agent_result",
            "call_id": call_id,
            "tool": tool,
            "scope": body.get("scope"),
            "measured_over": body.get("measured_over") or [],
            "result": body,
        })
        if tool in _AUXILIARY or tool == "?":
            continue
        eligible += 1
        scope = str(body.get("scope") or "").strip()
        lines = body.get("measured_over") or []
        if isinstance(lines, str):
            lines = [lines]
        lines = [str(line).strip() for line in lines if str(line).strip()]
        if scope:
            scopes.append(scope)
        measured.extend(lines)
        if scope and lines:
            complete += 1
        for line in lines:
            match = _MEASURED_TOOL.match(line)
            if match:
                tools.add(match.group(1))
    return {
        "team": team,
        "subqueries": subqueries,
        "tools": sorted(tools),
        "scopes": scopes,
        "measured_over": measured,
        "evidence": evidence,
        "provenance_completeness": complete / eligible if eligible else None,
    }


def _trace_fields(path: str | None, turn_id: str) -> dict[str, Any]:
    empty = {
        "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
        "llm_call_count": None, "self_recovery_count": None,
        "self_recovery_call_count": None,
        "self_recovery_tool_count": None, "self_recovery_orchestration_count": None,
        "qa_cache_hit": None,
        "episodic_context_exposed": None, "case_summary_exposed": None,
        "memory_context_exposed": None, "memory_telemetry_complete": None,
        "memory_sources": [],
        "kb_context_exposures": None, "kb_lookup_calls": None,
        "kb_lookup_hits": None,
        "evidence": [],
    }
    if not path or not Path(path).exists():
        return empty
    memory: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(node_trace)").fetchall()
        }
        if not {"id", "turn_id", "node"} <= columns:
            return empty
        wanted = [
            name for name in (
                "id", "parent_id", "node", "depth", "prompt_tokens",
                "completion_tokens", "total_tokens", "tags", "outcome",
                "messages_json",
            ) if name in columns
        ]
        rows = [
            dict(row) for row in conn.execute(
                f"SELECT {', '.join(wanted)} FROM node_trace "
                "WHERE turn_id = ? ORDER BY id",
                (turn_id,),
            ).fetchall()
        ]
        memory = _memory_evidence(conn, turn_id)
    except (OSError, sqlite3.Error):
        return empty
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass

    if not rows:
        return empty
    child_ids = {
        int(row["parent_id"]) for row in rows if row.get("parent_id") is not None
    }
    leaves = [row for row in rows if int(row["id"]) not in child_ids]
    token_columns = {"prompt_tokens", "completion_tokens", "total_tokens"} & columns
    if token_columns:
        llm_rows = [
            row for row in leaves
            if any(row.get(field) is not None for field in token_columns)
        ]
    else:
        llm_rows = [row for row in leaves if ".round_" in str(row.get("node") or "")]

    def summed(field: str) -> int | None:
        if field not in columns or not llm_rows:
            return None
        return sum(int(row.get(field) or 0) for row in llm_rows)

    prompt = summed("prompt_tokens")
    completion = summed("completion_tokens")
    total = summed("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    # SELF-RECOVERY, counted at three levels. All are the system noticing a
    # problem and fixing it WITHOUT being asked again, so all are benign —
    # what distinguishes them is cost and where the happy path broke:
    #
    #   call           the TRANSPORT abandoned a stalled LLM call and
    #                  re-issued it. Tagged `stall_retry`, or
    #                  `stall_retry_failed` when the re-issue also timed out —
    #                  which still fired, so it still counts as an attempt.
    #   tool           a re-issued call, an ungrounded answer sent back for
    #                  evidence, a retaken planning step — inside one attempt
    #   orchestration  the whole plan re-run for the same turn
    #
    # Exact tag matching, so `stall_retry` is NOT `retry`: they are different
    # mechanisms at different layers and pooling them would hide which one a
    # version actually changed.
    #
    # Counted per ROW, and the tag lands on the round, so two stalls inside
    # one round read as one. That under-reports rather than inventing, which
    # is the right direction for a count nobody can re-derive later.
    #
    # Neither is the evaluator re-asking a question the system never answered;
    # that is recorded by the runner, not read from here.
    self_recovery_call_count = len({
        int(row["id"]) for row in rows
        if _tags(row).intersection({"stall_retry", "stall_retry_failed"})
    })

    self_recovery_tool_count = len({
        int(row["id"]) for row in rows
        if str(row.get("node") or "").endswith(".retry")
        or _tags(row).intersection({"retry", "ungrounded_retry", "planning_timeout"})
    })
    orch_attempts = sum(
        1 for row in rows
        if row.get("node") == "orchestrator" and int(row.get("depth") or 0) == 0
    )
    self_recovery_orchestration_count = max(0, orch_attempts - 1)
    # Kept as the sum so runs recorded before the split still compare.
    self_recovery_count = (
        self_recovery_call_count
        + self_recovery_tool_count
        + self_recovery_orchestration_count
    )
    qa_cache_hit = any(
        row.get("node") == "cache_replay"
        and ("tags" not in columns or "cache_hit" in _tags(row))
        for row in rows
    )
    captured_messages = [
        str(row.get("messages_json")) for row in rows
        if row.get("messages_json")
        and '"_elided"' not in str(row.get("messages_json"))
    ]
    message_capture_available = bool(captured_messages)
    message_blob = "\n".join(captured_messages)
    episodic_context_exposed = (
        "[EPISODIC" in message_blob if message_capture_available else None
    )
    case_summary_exposed = (
        "[CASE SUMMARY" in message_blob if message_capture_available else None
    )
    kb_context_exposures = sum(
        1 for row in rows if "kb_digest_present" in _tags(row)
    ) if "tags" in columns else None
    kb_context_exposed = (
        bool(kb_context_exposures)
        or ("[KB-warmth" in message_blob or "[KB —" in message_blob)
        if message_capture_available or kb_context_exposures is not None else None
    )
    memory_telemetry_complete = message_capture_available and "tags" in columns
    context_signals = [
        episodic_context_exposed, case_summary_exposed, kb_context_exposed,
    ]
    memory_context_exposed = (
        True if any(value is True for value in context_signals)
        else False if memory_telemetry_complete
        else None
    )
    # The episodic block's CONTENT, not just its presence. Exposure is what the
    # old signals measured, and a header the system injects on every turn made
    # `memory_used` read True on all 12 runs of the last set — including the
    # first turn of a session, where there is nothing to remember. Carrying the
    # turns themselves lets a later pass ask the question that matters: was any
    # of this actually leveraged.
    episodic_turns = _episodic_turns(message_blob)
    kb_topics = _kb_topics(message_blob)

    memory_sources = []
    if qa_cache_hit:
        memory_sources.append("qa_cache")
    if episodic_context_exposed:
        memory_sources.append("episodic_context")
    if case_summary_exposed:
        memory_sources.append("case_summary")
    if kb_context_exposed:
        memory_sources.append("specialist_kb")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "llm_call_count": len(llm_rows),
        "self_recovery_count": self_recovery_count,
        "self_recovery_call_count": self_recovery_call_count,
        "self_recovery_tool_count": self_recovery_tool_count,
        "self_recovery_orchestration_count": self_recovery_orchestration_count,
        "qa_cache_hit": qa_cache_hit,
        "episodic_context_exposed": episodic_context_exposed,
        "case_summary_exposed": case_summary_exposed,
        "memory_context_exposed": memory_context_exposed,
        "memory_telemetry_complete": memory_telemetry_complete,
        "memory_sources": memory_sources,
        "kb_context_exposures": kb_context_exposures,
        # What was ON OFFER this turn, for the leverage judgement downstream.
        "episodic_turns_exposed": episodic_turns,
        "kb_topics_exposed": kb_topics,
        # Existing trace rows do not expose a stable tool-result join on every
        # historical version. Keep these unknown instead of manufacturing 0%.
        "kb_lookup_calls": None,
        "kb_lookup_hits": None,
        "evidence": _trace_evidence(rows, turn_id) + memory,
    }


class AgenticSysSSEAdapter(SystemAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.base_url = str(config["base_url"]).rstrip("/")
        # Optional: a multi-case run carries the case on each request instead,
        # and configuring one there would silently pin every turn to it.
        case_id = config.get("case_id")
        self.case_id = str(case_id) if case_id is not None else None
        self.trace_db = config.get("trace_db")
        self.headers = {
            str(k): str(v) for k, v in (config.get("headers") or {}).items()
        }
        self.healthcheck_path = str(config.get("healthcheck_path") or "/api/cases")
        self.turn_path = str(
            config.get("turn_path") or "/api/cases/{case_id}/turn"
        )
        self.stream_path = str(
            config.get("stream_path") or "/api/cases/{case_id}/stream"
        )
        self.reset_path = str(
            config.get("reset_path") or "/api/cases/{case_id}/rewind"
        )
        self.question_field = str(config.get("question_field") or "text")

    def _url(self, suffix: str) -> str:
        return f"{self.base_url}{suffix}"

    def _request_json(
        self, method: str, suffix: str, payload: dict | None, timeout_s: float,
    ) -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Accept": "application/json", **self.headers}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._url(suffix), data=body, headers=headers, method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
                parsed = json.loads(raw) if raw else {}
                return int(response.status), parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            raise RuntimeError(f"{method} {suffix} returned {exc.code}: {raw[:500]}") from exc

    def healthcheck(self) -> None:
        status, _ = self._request_json("GET", self.healthcheck_path, None, 10)
        if status != 200:
            raise RuntimeError(f"healthcheck returned HTTP {status}")

    def _quoted_case(self, case_id: str | None) -> str:
        """The case to address, request first, configuration second.

        Raising here rather than defaulting: a turn sent to the wrong case
        returns a well-formed answer about a different customer, which no
        downstream check can detect.
        """
        resolved = case_id if case_id is not None else self.case_id
        if resolved is None:
            raise RuntimeError(
                "no case id: set systems.<name>.config.case_id or "
                "experiment.cases"
            )
        # `safe=""` also encodes the trailing space that one real case folder
        # carries in its name, so the id survives the round trip intact.
        return urllib.parse.quote(resolved, safe="")

    def reset(self, case_id: str | None = None) -> None:
        path = self.reset_path.format(case_id=self._quoted_case(case_id))
        status, _ = self._request_json(
            "POST", path, {}, 30,
        )
        if status not in {200, 204}:
            raise RuntimeError(f"rewind returned HTTP {status}")

    def run(self, request: RunRequest, timeout_s: float) -> AdapterResult:
        quoted = self._quoted_case(request.case_id)
        turn_path = self.turn_path.format(case_id=quoted)
        stream_path = self.stream_path.format(case_id=quoted)
        started = time.perf_counter()
        status, payload = self._request_json(
            "POST", turn_path,
            {self.question_field: request.question.text}, min(timeout_s, 30),
        )
        if status not in {200, 202} or not payload.get("turn_id"):
            raise RuntimeError(f"turn start failed: HTTP {status}, {payload}")
        turn_id = str(payload["turn_id"])
        events: list[tuple[str, dict]] = []
        deadline = time.monotonic() + timeout_s
        stream_request = urllib.request.Request(
            self._url(stream_path),
            headers={"Accept": "text/event-stream", **self.headers},
        )
        try:
            with urllib.request.urlopen(
                stream_request, timeout=min(timeout_s, 30),
            ) as response:
                # Belt and braces on top of urlopen's own timeout: bounds a
                # single silent read. It does NOT bound the turn — a stream
                # that keeps producing bytes satisfies it forever — which is
                # why the deadline goes into `iter_sse` and is checked per
                # line, heartbeat or not.
                try:
                    response.fp.raw._sock.settimeout(min(30.0, timeout_s))  # type: ignore[attr-defined]
                except (AttributeError, OSError):
                    pass
                for event_name, event_payload in iter_sse(response, deadline=deadline):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"turn {turn_id} exceeded {timeout_s}s")
                    if event_payload.get("turn_id") != turn_id:
                        continue
                    events.append((event_name, event_payload))
                    if event_name == "turn_done":
                        break
                else:
                    raise RuntimeError(f"SSE stream ended before turn_done for {turn_id}")
        except (socket.timeout, TimeoutError) as exc:
            return AdapterResult(
                turn_id=turn_id, outcome="timeout", error=str(exc),
                elapsed_seconds=round(time.perf_counter() - started, 3),
                raw={"events": events},
            )

        finals = [p for name, p in events if name == "final"]
        errors = [p for name, p in events if name in {"turn_error", "error"}]
        dones = [p for name, p in events if name == "turn_done"]
        answer = str(finals[-1].get("answer") or "") if finals else ""
        done_outcome = str(dones[-1].get("outcome") or "ok") if dones else "error"
        outcome = (
            "out_of_scope" if answer.startswith("[rejected]")
            else ("ok" if finals and done_outcome in {"ok", "qa_cache_hit"} else done_outcome)
        )
        error = str(errors[-1].get("message") or "") if errors else ""
        normalized = _event_fields(events)
        trace = _trace_fields(self.trace_db, turn_id)
        trace_evidence = trace.pop("evidence", [])
        # Older AgenticSys versions may not emit measured_over, so classify
        # trace calls structurally as well: planned team members are agent
        # calls; other named calls inside their traces are data/tool results.
        agent_tools = set(normalized.get("team") or []) | _AUXILIARY
        data_tools = set(normalized.get("tools") or [])
        for item in trace_evidence:
            item["source_type"] = (
                "tool_result"
                if item.get("tool") in data_tools
                or item.get("tool") not in agent_tools | {"?"}
                else "agent_result"
            )
        normalized["evidence"] = [
            *(normalized.get("evidence") or []), *trace_evidence,
        ]
        normalized.update(trace)
        return AdapterResult(
            turn_id=turn_id,
            final_answer=answer,
            outcome=outcome,
            error=error,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            raw={"event_names": [name for name, _ in events]},
            **normalized,
        )
