"""Orchestration: one answer through three judge calls, and a whole runs.jsonl.

    extract      the question and the answer      -> claims
    evidence     claims + ledger + the run trace  -> pointers, routes, must-haves
    eligibility  routes + briefs + earlier turns  -> one verdict per claim

Python sits between and after: spans are checked against the answer, evidence
ids and call ids against the run, figures against the payload they came from,
derivations are recomputed, and oracles settle correctness outright wherever a
script can. The LLM proposes; nothing it returns is believed unchecked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_eval.common.coerce import _as_list, _safe_float, _slug
from agentic_eval.common.io import read_jsonl
from agentic_eval.content.aggregate import aggregate_content_evaluations
from agentic_eval.content.claims import (
    _drop_false_restatements, _drop_placeholder_table_claims,
    _ensure_table_cell_claims, _normalize_claim, mark_component_restatements,
    validate_claim_spans,
)
from agentic_eval.content.document import parse_answer_document, table_cell_coverage
from agentic_eval.content.evidence import (
    MEASURED_SOURCES, _bounded_ledger, build_evidence_ledger,
    claim_is_report_only,
)
from agentic_eval.content.metrics import calculate_content_metrics
from agentic_eval.content.oracles import evaluate_expected_answers
from agentic_eval.content.prompts import (
    ELIGIBILITY_PROMPT, EVIDENCE_PROMPT, EXTRACT_PROMPT,
    MEMORY_LEVERAGE_PROMPT,
)
from agentic_eval.content.trace import (
    apply_trace_eligibility, audit_claim_traces, build_trace_view,
)
from agentic_eval.render.page import (
    find_run_summary, write_answer_comparison,
)
from agentic_eval.render.markdown import (
    content_comparison_markdown, write_content_walkthrough,
    write_evidence_review_packets,
)
from agentic_eval.content.verdicts import MUST_HAVE_VERDICTS
from agentic_eval.content.verify import _normalize_fact_results
from agentic_eval.llm_judge import JudgeClient, OpenAIJudgeClient
from agentic_eval.models import RECORD_SCHEMA
from agentic_eval.layout import RunLayout


def _collect_must_haves(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    """The rubric's baseline points, as dicts, whichever key holds them."""
    raw = rubric.get("must_have_points") or rubric.get("must_haves") or []
    return [
        item if isinstance(item, dict) else {
            "id": f"mh_{index:02d}", "description": str(item),
        }
        for index, item in enumerate(_as_list(raw), 1)
    ]


def _normalize_must_haves(
    raw: Any, baseline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score the baseline, defaulting anything the judge skipped to a miss."""
    by_id = {str(item.get("id")): item for item in baseline if isinstance(item, dict)}
    results: list[dict[str, Any]] = []
    for row in _as_list(raw):
        if not isinstance(row, dict):
            continue
        mh_id = str(row.get("must_have_id") or row.get("id") or "")
        verdict = _slug(row.get("verdict"))
        definition = by_id.get(mh_id, {})
        results.append({
            "must_have_id": mh_id,
            "description": definition.get("description"),
            "verdict": verdict if verdict in MUST_HAVE_VERDICTS else "miss",
            "answer_spans": [str(v) for v in _as_list(row.get("answer_spans"))],
            "evidence_ids": [str(v) for v in _as_list(row.get("evidence_ids"))],
            "reason": str(row.get("reason") or ""),
            "confidence": _safe_float(row.get("confidence")),
            "weight": float(definition.get("weight") or 1),
        })
    returned = {row["must_have_id"] for row in results}
    for mh_id, definition in by_id.items():
        if mh_id in returned:
            continue
        results.append({
            "must_have_id": mh_id, "description": definition.get("description"),
            "verdict": "miss", "answer_spans": [], "evidence_ids": [],
            "reason": "Judge omitted this baseline item.", "confidence": None,
            "weight": float(definition.get("weight") or 1),
        })
    return results


class ContentEvaluator:
    def __init__(self, config: dict[str, Any], judge: JudgeClient | None = None) -> None:
        self.config = config
        self.judge = judge or OpenAIJudgeClient(config.get("llm") or {})

    def _judge_memory_leverage(
        self, record: dict[str, Any], *, claims: list[dict[str, Any]],
        trace_view: dict[str, Any],
    ) -> dict[str, Any]:
        """One call: of the memory offered this turn, what did the run use?

        Skipped entirely when nothing was offered — there is no question to
        ask, and asking anyway would spend a call to learn that an empty set
        was not leveraged.
        """
        topics = [str(v) for v in _as_list(record.get("kb_topics_exposed"))]
        turns = [
            row for row in _as_list(record.get("episodic_turns_exposed"))
            if isinstance(row, dict)
        ]
        cached = bool(record.get("qa_cache_hit"))
        offered = {
            "kb_topics": topics, "episodic_turns": turns, "qa_cache": cached,
        }
        if not (topics or turns or cached):
            return {"offered": offered, "sources": [], "asked": False}
        raw = self.judge.complete_json(
            task="memory_leverage", system_prompt=MEMORY_LEVERAGE_PROMPT,
            payload={
                "question": record.get("question"),
                "offered": offered,
                "produced": {
                    "team": list(record.get("team") or []),
                    "subqueries": dict(record.get("subqueries") or {}),
                    "findings": {
                        name: entry["findings"]
                        for name, entry in trace_view["specialists"].items()
                        if entry.get("findings")
                    },
                    "answer": str(record.get("final_answer") or ""),
                },
            },
        )
        known_topics, known_turns = set(topics), {
            str(row.get("turn_id")) for row in turns
        }
        sources: list[dict[str, Any]] = []
        for row in _as_list(raw.get("memory_leverage")):
            if not isinstance(row, dict):
                continue
            source = _slug(row.get("source"))
            if source not in {"kb", "episodic", "qa_cache"}:
                continue
            cited = [str(v) for v in _as_list(row.get("items"))]
            # A topic or turn the run was never shown cannot have been used;
            # crediting it would make this measure the harness.
            allowed = known_topics if source == "kb" else known_turns
            kept = [v for v in cited if v in allowed] if source != "qa_cache" else cited
            leveraged = _slug(row.get("leveraged")) == "yes"
            if source != "qa_cache" and leveraged and not kept:
                # Claimed use, cited nothing real: unproven, so not counted.
                leveraged = False
            sources.append({
                "source": source,
                "leveraged": leveraged,
                "where": [_slug(v) for v in _as_list(row.get("where"))],
                "items": kept,
                "invented_items": [v for v in cited if v not in allowed]
                if source != "qa_cache" else [],
                "reason": str(row.get("reason") or ""),
            })
        return {"offered": offered, "sources": sources, "asked": True}

    def evaluate(
        self, record: dict[str, Any], rubric: dict[str, Any] | None = None,
        prior_turns: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rubric = rubric or record.get("evaluation") or {}
        answer = str(record.get("final_answer") or "")
        blocks = parse_answer_document(answer)
        ledger = build_evidence_ledger(record, rubric)
        must_haves = _collect_must_haves(rubric)
        trace_view = build_trace_view(record, ledger)
        call_start = len(self.judge.calls)

        # ---- 1. what does the answer claim? -------------------------------
        # The question and the answer, nothing else. Extraction primed on the
        # evidence pulls claim boundaries toward what the system measured, and
        # every rate below is computed over these boundaries.
        extraction = self.judge.complete_json(
            task="claim_extraction", system_prompt=EXTRACT_PROMPT,
            payload={
                "question": record.get("question"), "answer": answer,
                "blocks": blocks,
            },
        )
        claims = [
            _normalize_claim(raw, index)
            for index, raw in enumerate(_as_list(extraction.get("claims")), 1)
            if isinstance(raw, dict)
        ]
        claims = _ensure_table_cell_claims(blocks, claims)
        claims = _drop_placeholder_table_claims(blocks, claims)
        claims = _drop_false_restatements(claims)
        # The extractor emits a compound claim and then its parts, marking
        # none of them; caught here so one finding counts once.
        claims = mark_component_restatements(claims)
        # Anchor every claim to text the answer contains, before anything
        # downstream treats it as something the answer asserted.
        claims = validate_claim_spans(claims, answer)
        # A restatement asserts a fact an earlier claim already asserted, and
        # every metric counts it once. Verifying it twice buys nothing and,
        # when the two passes disagree, contradicts itself inside one answer.
        verifiable = [
            claim for claim in claims
            if claim.get("is_factual") and not claim.get("restates_claim_id")
        ]

        # ---- 2. what does each claim rest on? -----------------------------
        # The only call carrying the ledger. Descriptive throughout: what
        # measured the figure, and how the run reached it. No verdict on
        # whether that route answers the question — see step 3.
        evidence = self.judge.complete_json(
            task="claim_evidence", system_prompt=EVIDENCE_PROMPT,
            payload={
                "question": record.get("question"), "answer": answer,
                "claims": verifiable, "trace": trace_view,
                "must_haves": must_haves,
                "evidence_ledger": _bounded_ledger(
                    ledger, int(self.config.get("max_evidence_chars", 60000)),
                ),
            },
        )
        claim_traces = audit_claim_traces(
            evidence.get("claim_traces"), trace_view,
            {claim["claim_id"] for claim in verifiable},
        )

        # A claim resting on curated report material grounds on that report
        # resolving, and no eligibility verdict can change it — `report` is
        # tested first and short-circuits. Asking anyway spent a verdict per
        # such claim (19 across the six-question set) and returned noise: on
        # `evidence_contradicting_pattern`, 10 of 11 came back NO while all 11
        # grounded regardless. Decided in Python from the cited provenance, so
        # it costs nothing to know.
        evidence_by_id = {str(item["evidence_id"]): item for item in ledger}
        raw_facts = {
            str(row.get("claim_id")): row
            for row in _as_list(evidence.get("fact_results"))
            if isinstance(row, dict)
        }
        by_claim_id = {claim["claim_id"]: claim for claim in verifiable}
        report_only = {
            claim_id for claim_id in claim_traces
            if claim_is_report_only(
                by_claim_id.get(claim_id) or {},
                raw_facts.get(claim_id) or {}, evidence_by_id,
            )
        }
        for claim_id in report_only:
            claim_traces[claim_id]["eligibility_reason"] = (
                "Not judged: the claim relays curated report material, which "
                "grounds on that report resolving rather than on a route."
            )

        # ---- 3. does that route answer the question? ----------------------
        # Its own call, deliberately. A judge that has already decided a claim
        # is sound describes a route that justifies it, so the description in
        # step 2 has to be written before any verdict exists. Small payload:
        # the routes, the briefs, and what earlier turns established — never
        # the ledger.
        judgeable = {
            cid: trace for cid, trace in claim_traces.items()
            if cid not in report_only
        }
        if judgeable:
            ruled = self.judge.complete_json(
                task="claim_eligibility", system_prompt=ELIGIBILITY_PROMPT,
                payload={
                    "question": record.get("question"),
                    "routes": [
                        {"claim_id": cid, **{
                            key: value for key, value in trace.items()
                            if key in {"specialist", "call_ids", "derivation"}
                        }}
                        for cid, trace in judgeable.items()
                    ],
                    "briefs": {
                        name: entry["brief"]
                        for name, entry in trace_view["specialists"].items()
                    },
                    # A stateful question inherits its subject: "how did TSR
                    # react" means the reaction to the spike the previous turn
                    # identified, and without the conversation a route over the
                    # whole window looks eligible for it.
                    "earlier_turns": prior_turns or [],
                },
            )
            claim_traces = apply_trace_eligibility(
                ruled.get("eligibility"), claim_traces,
            )

        # ---- 4. was the memory it was given actually USED? ----------------
        # Arrival is settled deterministically from the trace; this asks the
        # other question. The two come apart in both directions, and the old
        # rule conflated them — a KB header the system injects every turn made
        # `memory_used` read True on all twelve runs of a set, including the
        # first turn of a session, where there is nothing to remember.
        #
        # Not leveraging memory is not scored as a failure. It is recorded
        # because leveraged memory is what makes a session cheaper than its
        # turns and what lets knowledge accumulate across them.
        memory_leverage = self._judge_memory_leverage(
            record, claims=claims, trace_view=trace_view,
        )

        # ---- everything below is Python -----------------------------------
        fact_results = _normalize_fact_results(
            verifiable, _as_list(evidence.get("fact_results")), ledger,
            claim_traces=claim_traces,
        )
        must_have_results = _normalize_must_haves(
            evidence.get("must_have_results"), must_haves,
        )
        # Deterministic ground truth, run before any of this is reported: for a
        # question a script can answer outright, the judge's opinion is not
        # evidence of anything.
        oracle_results = evaluate_expected_answers(
            claims, rubric, answer=answer,
            cwd=self.config.get("oracle_cwd"),
            timeout=float(self.config.get("oracle_timeout_s", 60)),
        )
        metrics = calculate_content_metrics(
            claims, fact_results, must_have_results,
            memory_leverage=memory_leverage,
            table_coverage=table_cell_coverage(blocks, claims),
            tool_provenance_available=any(
                item.get("source_type") in MEASURED_SOURCES for item in ledger
            ),
            oracle_results=oracle_results,
        )
        return {
            "system": record.get("system"),
            "mode": record.get("mode"),
            "case_id": record.get("case_id"),
            "question_set": record.get("question_set"),
            "name": record.get("name"),
            "run_index": record.get("run_index"),
            "sequence_position": record.get("sequence_position"),
            "memory_required": record.get("memory_required"),
            "memory_used": record.get("memory_used"),
            "turn_id": record.get("turn_id"),
            "question": record.get("question"),
            "answer": answer,
            # The team that produced this answer, carried through so the viewer
            # can show construction beside content: a wrong answer from the
            # wrong specialists is a different defect from a wrong answer from
            # the right ones.
            "team": list(record.get("team") or []),
            "tools": list(record.get("tools") or []),
            "subqueries": dict(record.get("subqueries") or {}),
            "blocks": blocks,
            "claims": claims,
            "claim_traces": claim_traces,
            "evidence_ledger": ledger,
            "fact_results": fact_results,
            "expected_answer_results": oracle_results,
            "must_have_results": must_have_results,
            "memory_leverage": memory_leverage,
            "metrics": metrics,
            "judge_calls": self.judge.calls[call_start:],
        }


def _repeats_in(records: list[dict[str, Any]]) -> int | None:
    """How many repeats this runs.jsonl actually holds.

    Distinct `run_index` values, so `repetitions_complete` asks the question
    worth asking — was every answer that was RUN also scored — rather than
    comparing against a config that may describe a different run entirely.
    """
    indices = {
        row.get("run_index") for row in records
        if row.get("run_index") is not None
    }
    return len(indices) or None


def evaluate_runs_file(
    *, config: dict[str, Any], records: list[dict[str, Any]], output_dir: Path,
    baseline: str, candidate: str, rubric_by_name: dict[str, dict[str, Any]],
    judge: JudgeClient | None = None, limit: int | None = None,
    questions: list[str] | None = None, resume: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = RunLayout(output_dir).ensure()
    out_path = layout.evaluations
    evaluations = read_jsonl(out_path) if resume and out_path.exists() else []
    if not resume:
        out_path.write_text("", encoding="utf-8")
    # `case_id` is part of the identity: without it a multi-case run's second
    # case looks already-evaluated to `--resume` and is silently skipped.
    def identity(row: dict[str, Any]) -> tuple:
        return (
            row.get("system"), row.get("mode"), row.get("case_id"),
            row.get("name"), row.get("run_index"), row.get("sequence_position"),
        )

    completed = {identity(row) for row in evaluations}
    # A run captured before the adapter learned a field cannot be scored for
    # it. Say so once, plainly, rather than letting the metric read zero.
    stale = sorted({
        int(row.get("record_schema") or 1) for row in records
        if int(row.get("record_schema") or 1) < RECORD_SCHEMA
    })
    if stale:
        print(
            f"  note: {len(stale)} schema version(s) older than {RECORD_SCHEMA} "
            f"in this runs.jsonl (found {stale}). Fields added since then were "
            "never captured, so metrics over them read empty rather than zero "
            "— re-run `run` to measure those."
        )
    evaluator = ContentEvaluator(config, judge=judge)
    eligible = [row for row in records if row.get("outcome") == "ok" and row.get("final_answer")]
    if questions:
        wanted = set(questions)
        eligible = [row for row in eligible if str(row.get("name")) in wanted]
        if not eligible:
            raise ValueError(
                f"no answers for {sorted(wanted)}; runs.jsonl has "
                f"{sorted({str(r.get('name')) for r in records})}"
            )
    eligible = [row for row in eligible if identity(row) not in completed]
    if limit is not None:
        eligible = eligible[:limit]
    # The conversation each answer sits in, so a follow-up is judged against
    # what was already established rather than against its sentence alone.
    #
    # This is the JUDGE's context, not the system's — the system is reset per
    # session and never sees another case. But the key must match the session
    # exactly: `case_id` because two cases share a run_index, and
    # `question_set` because each set is its own session and its
    # `sequence_position` restarts at 1. Without the set, series D's cold
    # question would be judged against series B's turns — the very context the
    # run went to the trouble of withholding from the system.
    def session_key(row: dict[str, Any]) -> tuple:
        return (
            row.get("system"), row.get("mode"), row.get("case_id"),
            row.get("question_set"), row.get("run_index"),
        )

    by_session: dict[tuple, list[dict[str, Any]]] = {}
    for row in records:
        by_session.setdefault(session_key(row), []).append(row)
    for turns in by_session.values():
        turns.sort(key=lambda row: row.get("sequence_position") or 0)

    for index, record in enumerate(eligible, 1):
        rubric = rubric_by_name.get(str(record.get("name"))) or record.get("evaluation") or {}
        position = record.get("sequence_position") or 0
        prior_turns = [
            {"question": row.get("question"), "answer": str(row.get("final_answer") or "")[:1200]}
            for row in by_session.get(session_key(record), [])
            if (row.get("sequence_position") or 0) < position
        ]
        result = evaluator.evaluate(record, rubric, prior_turns=prior_turns)
        evaluations.append(result)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        print(
            f"  content {index:>4}/{len(eligible)}  {record.get('system')}  "
            f"{record.get('name')} #{record.get('run_index')}"
        )
    summary = aggregate_content_evaluations(
        evaluations,
        # Counted from the RECORDS, not from the config's k. Scoring runs as a
        # separate command against a runs.jsonl, and the config it reloads need
        # not describe that file: a scoped run asks k=2 while the config still
        # says 3, and the completeness check then failed a run that was whole.
        # What was actually run is in the file; the config is a guess about it.
        expected_repeats=_repeats_in(records) or (
            int(config["expected_repeats"])
            if config.get("expected_repeats") is not None else None
        ),
    )
    layout.content_summary.write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    layout.content_comparison.write_text(
        content_comparison_markdown(summary, baseline=baseline, candidate=candidate),
        encoding="utf-8",
    )
    write_content_walkthrough(evaluations, layout=layout)
    write_answer_comparison(
        evaluations, layout=layout, baseline=baseline, candidate=candidate,
        summary=find_run_summary(output_dir),
    )
    write_evidence_review_packets(
        evaluations, layout=layout, seed=int(config.get("seed", 20260731)),
    )
    return out_path

