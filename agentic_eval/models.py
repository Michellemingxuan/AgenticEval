"""Small dependency-free data contracts shared by evaluator adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Question:
    name: str
    text: str
    evaluation: dict[str, Any] = field(default_factory=dict)
    #: Which set this question belongs to — normally the questions file it came
    #: from. A set is a CONVERSATION: in stateful mode it gets its own session,
    #: so questions in different sets cannot see each other's turns. Series D
    #: exists to ask series B's last question with none of B's context, and
    #: sharing one session made it a repeat of B9 rather than a cold ask.
    question_set: str = ""


@dataclass(frozen=True)
class RunRequest:
    question: Question
    run_index: int
    mode: str
    sequence_position: int | None = None
    #: Which case this turn asked about. Carried on every record so a run over
    #: several cases can be read per case — without it the answers pool into
    #: one undifferentiated set and a per-case view is impossible after the
    #: fact, since nothing else in the record identifies the subject.
    case_id: str | None = None


#: Bumped when the adapter starts capturing a field the evaluator needs. A run
#: recorded before the bump cannot be re-scored for what it never captured, and
#: the failure is silent otherwise: the field reads empty and the metric reads
#: zero, which is indistinguishable from a real zero. `evaluate-content` warns
#: rather than guessing.
#:
#:   2  episodic_turns_exposed / kb_topics_exposed — memory as content, not a
#:      flag, so leverage can be judged and audited against what was offered
#:   3  case_id — which case a turn asked about, so a multi-case run can be
#:      read per case
#:   4  question_set — which set the turn belongs to. Sets are separate
#:      sessions, so a run before this bump asked every set in one
#:      conversation and its cold-ask questions were not cold
RECORD_SCHEMA = 5


@dataclass
class AdapterResult:
    """Normalized adapter result.

    Adapter-specific raw material stays in ``raw``. Everything else is the
    stable cross-system schema consumed by scoring and reporting.
    """

    turn_id: str = ""
    final_answer: str = ""
    outcome: str = "error"
    error: str = ""
    elapsed_seconds: float = 0.0
    team: list[str] = field(default_factory=list)
    subqueries: dict[str, str] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    measured_over: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    provenance_completeness: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    llm_call_count: int | None = None
    #: Total self-recovery for the turn: tool level plus orchestration
    #: level. Was `retry_count`.
    self_recovery_count: int | None = None
    #: Recovery INSIDE one attempt at the question — a re-issued tool call, an
    #: ungrounded answer sent back for evidence, a retaken planning step. A
    #: designed mechanism, not a fault.
    self_recovery_tool_count: int | None = None
    #: The whole orchestration running again for one turn: the system starting
    #: the question over. This is the one nobody wants.
    self_recovery_orchestration_count: int | None = None
    qa_cache_hit: bool | None = None
    episodic_context_exposed: bool | None = None
    # What memory was OFFERED this turn, as content rather than a flag. The
    # booleans beside these say a block appeared; these say which prior turns
    # and which knowledge points, so a later pass can ask whether any of it was
    # actually used and Python can check that a cited item was really shown.
    episodic_turns_exposed: list[dict[str, Any]] = field(default_factory=list)
    kb_topics_exposed: list[str] = field(default_factory=list)
    case_summary_exposed: bool | None = None
    memory_context_exposed: bool | None = None
    memory_telemetry_complete: bool | None = None
    memory_sources: list[str] = field(default_factory=list)
    kb_context_exposures: int | None = None
    kb_lookup_calls: int | None = None
    kb_lookup_hits: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def _self_recovery_total(self) -> int | None:
        """The total, or the two levels added when only they were supplied.

        An adapter that fills the levels but forgets the total should not make
        the headline rate read "never measured" — that is the one reading that
        looks like a clean record and is not.
        """
        if self.self_recovery_count is not None:
            return self.self_recovery_count
        levels = [
            self.self_recovery_tool_count, self.self_recovery_orchestration_count,
        ]
        if all(level is None for level in levels):
            return None
        return sum(level or 0 for level in levels)

    def to_record(self, *, system: str, request: RunRequest) -> dict[str, Any]:
        return {
            "record_schema": RECORD_SCHEMA,
            "system": system,
            "case_id": request.case_id,
            "mode": request.mode,
            "question_set": request.question.question_set,
            "name": request.question.name,
            "question": request.question.text,
            "evaluation": request.question.evaluation,
            "run_index": request.run_index,
            "sequence_position": request.sequence_position,
            "turn_id": self.turn_id,
            "final_answer": self.final_answer,
            "outcome": self.outcome,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "team": self.team,
            "team_unique": sorted(set(self.team)),
            "subqueries": self.subqueries,
            "tools": sorted(set(self.tools)),
            "scopes": self.scopes,
            "measured_over": self.measured_over,
            "evidence": self.evidence,
            "provenance_completeness": self.provenance_completeness,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.llm_call_count,
            # SELF-RECOVERY: the system fixing its own problem without being
            # asked again. This is what `retry_count`/`retried` always held —
            # tool plus orchestration, summed — so the new names say what the
            # numbers already meant rather than measuring anything new. The
            # two levels are split because they cost differently and point at
            # different parts of the happy path.
            #
            # Contrast `evaluator_replayed`, set by the runner when the system
            # produced NOTHING and we asked again. `None` where a run never
            # recorded it: absent must not read as zero.
            "self_recovery_count": self._self_recovery_total(),
            "self_recovered": (
                total > 0
                if (total := self._self_recovery_total()) is not None else None
            ),
            "self_recovery_tool_count": self.self_recovery_tool_count,
            "self_recovered_tool": (
                self.self_recovery_tool_count > 0
                if self.self_recovery_tool_count is not None else None
            ),
            "self_recovery_orchestration_count": self.self_recovery_orchestration_count,
            "self_recovered_orchestration": (
                self.self_recovery_orchestration_count > 0
                if self.self_recovery_orchestration_count is not None else None
            ),
            "qa_cache_hit": self.qa_cache_hit,
            "episodic_context_exposed": self.episodic_context_exposed,
            "case_summary_exposed": self.case_summary_exposed,
            "memory_context_exposed": self.memory_context_exposed,
            "memory_telemetry_complete": self.memory_telemetry_complete,
            "memory_sources": self.memory_sources,
            "kb_context_exposures": self.kb_context_exposures,
            "episodic_turns_exposed": self.episodic_turns_exposed,
            "kb_topics_exposed": self.kb_topics_exposed,
            "kb_lookup_calls": self.kb_lookup_calls,
            "kb_lookup_hits": self.kb_lookup_hits,
            "raw": self.raw,
        }
