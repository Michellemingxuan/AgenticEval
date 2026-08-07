import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from agentic_eval.adapters.agenticsys_sse import (
    AgenticSysSSEAdapter,
    _trace_fields,
    iter_sse,
)
from agentic_eval.models import Question, RunRequest


def _frame(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def test_iter_sse_parses_named_json_events():
    stream = BytesIO(
        b": connected\n\n"
        b"event: final\n"
        b'data: {"turn_id":"t1","answer":"ok"}\n\n'
    )
    assert list(iter_sse(stream)) == [
        ("final", {"turn_id": "t1", "answer": "ok"})
    ]


def test_trace_fields_extracts_optional_telemetry(tmp_path: Path):
    db = tmp_path / "trace.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE node_trace (id INTEGER, turn_id TEXT, node TEXT, "
        "parent_id INTEGER, depth INTEGER, prompt_tokens INTEGER, "
        "completion_tokens INTEGER, total_tokens INTEGER, tags TEXT, outcome TEXT)"
    )
    conn.executemany(
        "INSERT INTO node_trace VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "t1", "orchestrator", None, 0, None, None, None, None, "ok"),
            (2, "t1", "orchestrator.round_1", 1, 1, 10, 2, 12, None, "ok"),
            (3, "t1", "specialist.modeling.retry", None, 0, None, None, None,
             '["retry", "kb_digest_present"]', "ok"),
            (4, "t1", "specialist.modeling.retry.round_1", 3, 1, 5, 1, 6,
             None, "ok"),
        ],
    )
    conn.commit()
    conn.close()
    out = _trace_fields(str(db), "t1")
    assert out["total_tokens"] == 18
    assert out["llm_call_count"] == 2
    assert out["retry_count"] == 1
    assert out["kb_context_exposures"] == 1


def test_trace_fields_recovers_tool_result_evidence(tmp_path: Path):
    db = tmp_path / "trace-evidence.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE node_trace (id INTEGER, turn_id TEXT, node TEXT, "
        "parent_id INTEGER, depth INTEGER, messages_json TEXT)"
    )
    messages = json.dumps([
        {"type": "function_call", "call_id": "d1", "name": "query_table",
         "arguments": '{"table_name":"payments"}'},
        {"type": "function_call_output", "call_id": "d1",
         "output": '{"count":2}'},
    ])
    conn.execute(
        "INSERT INTO node_trace VALUES (?,?,?,?,?,?)",
        (1, "t1", "specialist.round_2", None, 1, messages),
    )
    conn.commit()
    conn.close()
    evidence = _trace_fields(str(db), "t1")["evidence"]
    assert evidence[0]["tool"] == "query_table"
    assert evidence[0]["arguments"] == {"table_name": "payments"}
    assert evidence[0]["result"] == {"count": 2}


def test_trace_fields_recovers_chat_completions_tool_calls(tmp_path: Path):
    """A specialist's DATA-tool calls are stored chat-completions-style.

    Regression: only the Responses shape (`function_call`/`function_call_output`,
    keyed by `call_id`) was parsed, which is how the orchestrator records its
    calls to SPECIALISTS. One level down, the specialists' own data-tool calls
    live in an `assistant.tool_calls` array answered by a `tool` message keyed
    by `tool_call_id` — neither of which the parser looked at. The ledger
    therefore held nothing but agent-to-agent calls, so no number was ever
    grounded in a tool output and numeric grading stayed "unavailable".
    """
    db = tmp_path / "chat-shape.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE node_trace (id INTEGER, turn_id TEXT, node TEXT, "
        "parent_id INTEGER, depth INTEGER, messages_json TEXT)"
    )
    messages = json.dumps([
        {"role": "system", "content": "..."},
        {"role": "assistant", "tool_calls": [{
            "id": "call_abc", "type": "function",
            "function": {"name": "batch_summarize_trend",
                         "arguments": '{"specs_json":"[]"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_abc",
         "content": '{"results":[{"value":0.75}]}'},
    ])
    conn.execute(
        "INSERT INTO node_trace VALUES (?,?,?,?,?,?)",
        (1, "t1", "specialist.modeling.round_2", None, 1, messages),
    )
    conn.commit()
    conn.close()
    evidence = _trace_fields(str(db), "t1")["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["tool"] == "batch_summarize_trend"
    assert evidence[0]["arguments"] == {"specs_json": "[]"}
    assert evidence[0]["result"] == {"results": [{"value": 0.75}]}


def test_trace_fields_detects_episodic_memory_context(tmp_path: Path):
    db = tmp_path / "trace-memory.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE node_trace (id INTEGER, turn_id TEXT, node TEXT, "
        "parent_id INTEGER, depth INTEGER, tags TEXT, messages_json TEXT)"
    )
    messages = json.dumps([
        {"role": "user", "content": "[EPISODIC - recent turns] prior answer"},
    ])
    conn.execute(
        "INSERT INTO node_trace VALUES (?,?,?,?,?,?,?)",
        (1, "t1", "orchestrator.round_1", None, 1, "[]", messages),
    )
    conn.commit()
    conn.close()
    out = _trace_fields(str(db), "t1")
    assert out["episodic_context_exposed"] is True
    assert out["memory_context_exposed"] is True
    assert out["memory_telemetry_complete"] is True
    assert out["memory_sources"] == ["episodic_context"]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/api/cases":
            body = b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.endswith("/stream"):
            frames = "".join([
                _frame("team_plan", {"turn_id": "t1", "tool_calls": [{
                    "call_id": "c1", "tool": "modeling",
                    "sub_question": "Check TSR in 2025",
                }]}),
                _frame("agent_completed", {
                    "turn_id": "t1", "call_id": "c1", "tool": "modeling",
                    "payload": {
                        "scope": "scores: 2025",
                        "measured_over": ["summarize_trend(scores.tsr, filters=2025)"],
                    },
                }),
                _frame("final", {"turn_id": "t1", "answer": "TSR increased."}),
                _frame("turn_done", {"turn_id": "t1", "outcome": "ok"}),
            ]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path.endswith("/turn"):
            body = b'{"turn_id":"t1"}'
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.endswith("/rewind"):
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)


def test_adapter_normalizes_agenticsys_http_and_sse():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = AgenticSysSSEAdapter({
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "case_id": "c1",
        })
        adapter.healthcheck()
        adapter.reset()
        result = adapter.run(
            RunRequest(Question("q", "How did TSR react?"), 1, "cold"),
            timeout_s=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.outcome == "ok"
    assert result.final_answer == "TSR increased."
    assert result.team == ["modeling"]
    assert result.tools == ["summarize_trend"]
    assert result.provenance_completeness == 1.0
    assert result.evidence[0]["source_type"] == "agent_result"


def test_memory_evidence_excludes_the_turn_that_produced_it(tmp_path):
    """A snapshot is taken AFTER its turn, so its own row already holds what
    the turn concluded. Using it would make every answer its own evidence."""
    import sqlite3
    from agentic_eval.adapters.agenticsys_sse import _memory_evidence

    db = tmp_path / "trace.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE session_snapshot (id INTEGER PRIMARY KEY, chat_id TEXT, "
        "turn_id TEXT, specialist_kb_json TEXT, qa_cache_json TEXT)"
    )
    con.execute(
        "INSERT INTO session_snapshot VALUES (1, 'c', 'turn_one', ?, ?)",
        (json.dumps({"modeling": [{"topic": "tsr_trend", "numbers": [1]}]}),
         json.dumps([{"turn_id": "turn_one", "question": "first?"}])),
    )
    con.execute(
        "INSERT INTO session_snapshot VALUES (2, 'c', 'turn_two', ?, ?)",
        (json.dumps({"modeling": [
            {"topic": "tsr_trend", "numbers": [1]},
            {"topic": "conclusion_turn_two", "numbers": [2]},
        ]}),
         json.dumps([{"turn_id": "turn_two"}, {"turn_id": "turn_one"}])),
    )
    con.commit()
    con.row_factory = sqlite3.Row

    first = _memory_evidence(con, "turn_one")
    assert first == [], "the first turn of a session had no memory to draw on"

    second = _memory_evidence(con, "turn_two")
    topics = {e["arguments"]["topic"] for e in second if e["source_type"] == "memory"}
    assert topics == {"tsr_trend"}
    assert "conclusion_turn_two" not in topics
    recalled = {e["evidence_id"] for e in second if e["source_type"] == "memory_recall"}
    assert recalled == {"memory:qa:turn_one"}
    con.close()


def test_a_knowledge_point_recorded_twice_becomes_one_entry(tmp_path):
    """The KB stores one measurement more than once — two specialists holding
    the same trend, or one specialist writing its topic again on a later round.
    Two ids for one figure let two claims cite "different" provenance for it."""
    import sqlite3
    from agentic_eval.adapters.agenticsys_sse import _memory_evidence

    trend = {"topic": "tsr_trend", "claim": "TSR 8.4 -> 7.7",
             "numbers": [{"period": "2024-01", "tsr": 8.4}]}
    kb = {
        "modeling": [trend],
        # Same measurement, recorded by a second specialist and again by the
        # first on a later round.
        "spend_payments": [dict(trend, topic="spend_tsr_trend"), dict(trend)],
        }
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE session_snapshot (id INTEGER PRIMARY KEY, chat_id TEXT, "
        "turn_id TEXT, specialist_kb_json TEXT, qa_cache_json TEXT)"
    )
    con.execute("INSERT INTO session_snapshot VALUES (1,'c','t1',?,'[]')",
                (json.dumps(kb),))
    con.execute("INSERT INTO session_snapshot VALUES (2,'c','t2',?,'[]')",
                (json.dumps(kb),))
    con.commit()
    con.row_factory = sqlite3.Row

    memory = [e for e in _memory_evidence(con, "t2") if e["source_type"] == "memory"]
    assert len(memory) == 1, "three recordings of one measurement, one entry"
    assert memory[0]["evidence_id"].startswith("memory:modeling:tsr_trend@")
    # The other recorders are kept, so attribution is not lost.
    assert memory[0]["also_recorded_by"] == [
        "spend_payments:spend_tsr_trend", "spend_payments:tsr_trend",
    ]
    con.close()


def test_one_topic_holding_several_measurements_keeps_them_apart(tmp_path):
    """`Amount_by_Merchant Name` is recorded with 6 figures and later, on a
    wider scope, with 11. Keying the id on the topic collided them and
    `evidence_by_id` kept only the last, so a claim citing the early reading
    silently resolved against the late one."""
    import sqlite3
    from agentic_eval.adapters.agenticsys_sse import _memory_evidence

    kb = {"spend_payments": [
        {"topic": "by_merchant", "captured_at_turn": "t1",
         "claim": "6 groups", "numbers": [1] * 6},
        {"topic": "by_merchant", "captured_at_turn": "t3",
         "claim": "11 groups", "numbers": [1] * 11},
    ]}
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE session_snapshot (id INTEGER PRIMARY KEY, chat_id TEXT, "
        "turn_id TEXT, specialist_kb_json TEXT, qa_cache_json TEXT)"
    )
    con.execute("INSERT INTO session_snapshot VALUES (1,'c','t3',?,'[]')", (json.dumps(kb),))
    con.execute("INSERT INTO session_snapshot VALUES (2,'c','t4','{}','[]')")
    con.commit()
    con.row_factory = sqlite3.Row

    memory = [e for e in _memory_evidence(con, "t4") if e["source_type"] == "memory"]
    assert len(memory) == 2, "two different measurements, not one"
    assert len({e["evidence_id"] for e in memory}) == 2
    assert {e["captured_at_turn"] for e in memory} == {"t1", "t3"}
    con.close()
