"""Paired, randomized baseline/candidate experiment runner."""
from __future__ import annotations

import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_eval.adapters import build_adapter
from agentic_eval.cases import describe_case
from agentic_eval.workers import assert_workers_are_isolated, bind_to_worker
from agentic_eval.config import EvalConfig
from agentic_eval import memory_store
from agentic_eval.content import evaluate_runs_file
from agentic_eval.models import AdapterResult, RunRequest
from agentic_eval.process import ManagedProcess, expand
from agentic_eval.render import progress
from agentic_eval.render.run_summary import comparison_markdown, write_blind_review
from agentic_eval.layout import RunLayout
from agentic_eval.scoring import aggregate, compare, score_content, score_memory


#: Outcomes worth asking again. A timeout under load says nothing about the
#: answer — the system never produced one — so recording it as the system's
#: verdict measures the machine's traffic, not its reasoning. Everything else
#: (`error`, `out_of_scope`, a wrong answer) IS the system's behaviour and is
#: kept exactly as it happened.
_DEFAULT_RETRY_OUTCOMES = ("timeout", "screen_timeout")


@dataclass(frozen=True)
class RetryPolicy:
    """When to ask again after a turn that produced nothing.

    Deliberately NOT the same thing as SELF-RECOVERY on a record. That is the
    system fixing its own problem — a re-issued tool call, a re-run plan — read
    from its trace and reported as `self_recovery_rate`, a measurement of the
    subject. This is the evaluator asking again because the system produced
    nothing at all. Mixing them would let our recovery inflate a number meant
    to describe the system.
    """

    outcomes: frozenset[str] = frozenset(_DEFAULT_RETRY_OUTCOMES)
    attempts: int = 0          # extra attempts after the first; 0 disables
    backoff_s: float = 30.0

    @classmethod
    def from_config(cls, raw: Any) -> "RetryPolicy":
        if not raw:
            return cls(attempts=0)
        if not isinstance(raw, dict):
            raise ValueError("experiment.retry must be a mapping")
        unknown = set(raw) - {"outcomes", "attempts", "backoff_s"}
        if unknown:
            raise ValueError(
                f"experiment.retry sets unknown key(s) {sorted(unknown)}; "
                "it may set outcomes, attempts, backoff_s"
            )
        return cls(
            outcomes=frozenset(
                str(name) for name in (raw.get("outcomes") or _DEFAULT_RETRY_OUTCOMES)
            ),
            attempts=max(0, int(raw.get("attempts", 0))),
            backoff_s=max(0.0, float(raw.get("backoff_s", 30.0))),
        )

    def triggered_by(self, records: list[dict[str, Any]]) -> list[str]:
        """Which of these records asked to be retried, in order."""
        return [
            str(record.get("outcome")) for record in records
            if str(record.get("outcome")) in self.outcomes
        ]


@dataclass(frozen=True)
class Session:
    """One conversation: a case, a question set, and a repeat of it.

    The unit of parallelism. Its questions are asked in order on one server
    instance, so a follow-up still lands after the turn it refers to.
    """

    case_id: str | None
    question_set: str
    run_index: int
    mode: str
    questions: list

    @property
    def key(self) -> tuple:
        """Identity used to seed this session's system ordering."""
        return (self.mode, self.case_id, self.question_set, self.run_index)


class ComparisonRunner:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        experiment = config.experiment
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = str(experiment.get("name") or "agentic_compare")
        self.output_dir = Path(experiment["output_dir"]) / f"{run_name}_{stamp}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.layout = RunLayout(self.output_dir).ensure()
        self.raw_path = self.layout.runs
        self.raw_path.write_text("", encoding="utf-8")
        self.workers = max(1, int(experiment.get("workers", 1) or 1))
        # One server instance per worker. Sharing a process would let two
        # sessions interleave on the system's process-global gateway — see
        # `workers.py`. Worker 0 is the config exactly as written.
        pool = [
            {
                name: bind_to_worker(name, target, worker)
                for name, target in config.systems.items()
            }
            for worker in range(self.workers)
        ]
        assert_workers_are_isolated(pool)
        self.pool = [
            {
                name: build_adapter(target["adapter"], target.get("config") or {})
                for name, target in systems.items()
            }
            for systems in pool
        ]
        self.processes = [
            ManagedProcess(
                f"{name}.w{worker}" if self.workers > 1 else name,
                target["process"], self.output_dir,
            )
            for worker, systems in enumerate(pool)
            for name, target in systems.items() if target.get("process")
        ]
        self.seed = int(experiment["seed"])
        self._retry = RetryPolicy.from_config(experiment.get("retry"))
        # Everything written so far, so the progress page is a view of the
        # run rather than a re-read of the file it is racing.
        self._done: list[dict[str, Any]] = []
        self._started_at = time.time()
        # Both guard shared state that concurrent workers touch. Interleaved
        # writes would corrupt runs.jsonl into unparseable half-lines.
        # Set on interrupt; workers check it between turns.
        self._stopping = threading.Event()
        self._write_lock = threading.Lock()
        self._print_lock = threading.Lock()

    def _persist(self, record: dict[str, Any]) -> None:
        with self._write_lock:
            with self.raw_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            # Under the same lock, so the page can never describe a half-written
            # runs.jsonl. `write` swallows its own errors: a progress page must
            # not be able to fail a run.
            self._done.append(record)
            progress.write(
                self.layout.progress, self._done,
                plan=self._plan(), started_at=self._started_at,
            )

    def _plan(self) -> dict[str, Any]:
        """The shape this run is aiming at, from the run itself.

        Not from the config's raw numbers: a scoped run asks fewer questions
        than the file lists, and a progress bar measuring against the config
        would sit at 40% on a complete run.
        """
        questions = len(self.config.questions)
        cases = len(self._cases)
        repeats = int(self.config.experiment["repeats"])
        systems = len(self.config.systems)
        modes = 2 if self.config.experiment["mode"] == "both" else 1
        return {
            "questions": questions, "cases": cases, "repeats": repeats,
            "systems": systems,
            # Per set, because sets differ in length: series_b has eight
            # questions and series_c one, so a shared denominator would put
            # one of them at 800%.
            "set_sizes": {
                name: len(items) for name, items in self._question_sets()
            },
            # Series order, then position inside the series: the order the
            # questions are ASKED. Without it a reader watching the grid sees
            # rows reshuffle as workers finish.
            "question_order": [
                question.name
                for _set, items in self._question_sets() for question in items
            ],
            "set_of": {
                question.name: name
                for name, items in self._question_sets() for question in items
            },
            "case_ids": [case for case in self._cases if case is not None],
            "expected_records": questions * cases * repeats * systems * modes,
        }

    def _order(self, key: tuple) -> list[str]:
        """System order for one session, shuffled deterministically.

        Derived from the seed and the session's own identity rather than drawn
        from one shared Random: with several workers the draw order depends on
        which thread got there first, so a shared generator makes the run
        unreproducible even at a fixed seed.
        """
        systems = list(self.config.systems)
        # Seeded from the repr, not `hash()`: string hashing is salted per
        # process, so a hash-seeded shuffle would differ between two runs of
        # the same config and seed.
        random.Random(repr((self.seed, key))).shuffle(systems)
        return systems

    def _one(
        self, system: str, request: RunRequest, worker: int = 0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = self.pool[worker][system].run(
                request, float(self.config.experiment["timeout_s"])
            )
        except Exception as exc:  # one target failure must not erase the trial
            result = AdapterResult(
                outcome="error",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=round(time.perf_counter() - started, 3),
            )
        record = result.to_record(system=system, request=request)
        record.update(score_memory(record))
        record.update(score_content(record))
        # NOT persisted here. A system pass that gets retried must not leave
        # its abandoned attempt in runs.jsonl: every consumer assumes one row
        # per (system, case, set, repeat, question), and a duplicate would be
        # averaged into latency and consistency as though the system had
        # answered twice. `_run_session` writes the attempt it keeps.
        case = (
            f"  [{describe_case(request.case_id)}]"
            if len(self._cases) > 1 else ""
        )
        tag = f"w{worker} " if self.workers > 1 else ""
        with self._print_lock:
            print(
                f"  {tag}{system:<14} {record['outcome']:<12} "
                f"{record['elapsed_seconds']:>7.2f}s  "
                f"{request.question.name} #{request.run_index}{case}"
            )
        return record

    def _reset(self, system: str, case_id: str | None, worker: int) -> None:
        self.pool[worker][system].reset(case_id)

    @property
    def _cases(self) -> list[str | None]:
        """The cases to ask, resolved by `load_config`; `[None]` if unset."""
        return self.config.experiment.get("cases") or [None]

    def _question_sets(self) -> list[tuple[str, list]]:
        """Questions grouped into sets, in the order the config lists them.

        A set is a conversation. In stateful mode each one gets its own
        session, so a question that must be asked cold really is asked cold —
        series D asks series B's final question with none of B's context, and
        running both in one session made it a restatement of B9 two turns
        later rather than the discovery probe it exists to be.
        """
        ordered: dict[str, list] = {}
        for question in self.config.questions:
            ordered.setdefault(question.question_set or "questions", []).append(
                question
            )
        return list(ordered.items())

    def _sessions(self, mode: str) -> list[Session]:
        """Every session this mode runs, as independent units of work.

        A SESSION is the unit — never a question. The questions inside one are
        a conversation and must be asked in the configured order on one server,
        so parallelism is across sessions and strictly serial within them.

        Cold gives each question its own session, which is what "reset before
        every turn" means. Stateful gives each question SET one, so a set's
        turns share history and no other set's turns are in scope.
        """
        repeats = range(1, self.config.experiment["repeats"] + 1)
        if mode == "cold":
            return [
                Session(case_id, "", run_index, "cold", [question])
                for case_id in self._cases
                for question in self.config.questions
                for run_index in repeats
            ]
        return [
            Session(case_id, set_name, run_index, "stateful", questions)
            for case_id in self._cases
            for set_name, questions in self._question_sets()
            for run_index in repeats
        ]

    def _system_pass(
        self, session: Session, system: str, worker: int,
    ) -> list[dict[str, Any]]:
        """One system's whole turn of a session: rewind, then every question."""
        self._reset(system, session.case_id, worker)
        records = []
        for position, question in enumerate(session.questions, 1):
            if self._stopping.is_set():
                break
            records.append(self._one(
                system,
                RunRequest(
                    question, session.run_index, session.mode,
                    # Cold turns are not a conversation, so they carry no
                    # position — the same as before sessions existed.
                    position if session.mode == "stateful" else None,
                    case_id=session.case_id,
                ),
                worker,
            ))
        return records

    def _run_session(self, session: Session, worker: int) -> list[dict[str, Any]]:
        """One session, both systems, questions in the configured order.

        A turn that timed out is retried by REPLAYING THE WHOLE PASS, not by
        asking that one question again. Recovery starts with `/rewind`, which
        clears the case — memory, trace rows, conversation — so a re-ask on its
        own would put a follow-up in an empty session: b3 would ask about "the
        reacting period" that b2 never established, and answer something vague
        that every metric then reads as the system's fault. In cold mode a pass
        is a single turn anyway, so the two are the same thing there.

        The abandoned attempt is dropped rather than recorded. A timeout is the
        machine being busy, not the system being wrong, and leaving both rows
        in runs.jsonl would average an empty answer into latency and
        consistency as if the system had answered twice. What DID happen is
        disclosed on the kept record — `evaluator_attempts` and the outcomes that
        forced them — so a run that limped is never silently indistinguishable
        from one that did not.
        """
        records = []
        for system in self._order(session.key):
            attempts, forced = 0, []
            while True:
                attempts += 1
                pass_records = self._system_pass(session, system, worker)
                triggered = self._retry.triggered_by(pass_records)
                if not triggered or attempts > self._retry.attempts:
                    break
                forced.extend(triggered)
                with self._print_lock:
                    print(
                        f"  {'w' + str(worker) + ' ' if self.workers > 1 else ''}"
                        f"{system:<14} {'/'.join(sorted(set(triggered))):<12} "
                        f"replaying {describe_case(session.case_id)} "
                        f"{session.question_set} #{session.run_index} "
                        f"({attempts}/{self._retry.attempts + 1}) "
                        f"after {self._retry.backoff_s:.0f}s"
                    )
                # Traffic, most likely. Waiting is the point: an immediate
                # retry re-enters the same congestion that caused the timeout.
                time.sleep(self._retry.backoff_s)
            for record in pass_records:
                record["evaluator_attempts"] = attempts
                # A boolean beside the count, so the rate can be read the same
                # way as the system's own two — and so "no replay" is False
                # rather than a missing key that averages as nothing.
                record["evaluator_replayed"] = attempts > 1
                if forced:
                    record["evaluator_replay_reasons"] = forced
                self._persist(record)
            records.extend(pass_records)
        # One grid per finished session: often enough to watch a long run,
        # rare enough not to bury the per-answer lines. Under the print lock,
        # so two workers cannot interleave halves of two grids.
        with self._print_lock:
            print(progress.terminal_report(
                list(self._done), plan=self._plan(),
                started_at=self._started_at,
                order=[q.name for q in self.config.questions],
            ))
        return records

    def _execute(self, mode: str) -> list[dict[str, Any]]:
        sessions = self._sessions(mode)
        if self.workers == 1:
            return [
                record for session in sessions
                for record in self._run_session(session, 0)
            ]
        # A worker owns CASES, not an arbitrary slice of sessions. Two workers
        # on the same case would still collide even with separate servers,
        # because the memory store is shared and `/rewind` purges it BY CASE:
        # one worker opening a session would delete the memories the other is
        # mid-way through writing. Owning the case end to end removes the
        # window entirely, and repeats of a case stay strictly sequential —
        # which is what makes them independent.
        cases = self._cases
        owner = {case: index % self.workers for index, case in enumerate(cases)}
        idle = self.workers - len({owner[case] for case in cases})
        print(
            f"  {len(sessions)} sessions over {self.workers} workers "
            f"({mode}); each worker owns whole cases, questions in order"
            + (f"; {idle} worker(s) idle — only {len(cases)} case(s) to own"
               if idle > 0 else "")
        )
        results: list[list[dict[str, Any]]] = [[] for _ in sessions]

        def run_slot(worker: int) -> None:
            for index, session in enumerate(sessions):
                if self._stopping.is_set():
                    return
                if owner[session.case_id] == worker:
                    results[index] = self._run_session(session, worker)

        # Not `with`: its __exit__ joins BEFORE any except clause runs, so a
        # Ctrl-C would wait for every worker to finish its whole slice before
        # the interrupt was even acknowledged. Setting the flag first lets each
        # worker stop at its next turn boundary.
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            list(pool.map(run_slot, range(self.workers)))
        except BaseException:
            self._stopping.set()
            print(
                "  interrupted — finishing the turn in flight, then stopping. "
                "runs.jsonl keeps every answer already recorded."
            )
            raise
        finally:
            pool.shutdown(wait=True)
        # Reassembled in session order, so runs.jsonl reads the same whatever
        # order the workers happened to finish in.
        return [record for group in results for record in group]

    def _cold(self) -> list[dict[str, Any]]:
        return self._execute("cold")

    def _stateful(self) -> list[dict[str, Any]]:
        return self._execute("stateful")

    def _start(self) -> None:
        started = {process.name for process in self.processes}
        for worker, adapters in enumerate(self.pool):
            for name, adapter in adapters.items():
                tag = f"{name}.w{worker}" if self.workers > 1 else name
                process = next(
                    (p for p in self.processes if p.name == tag), None,
                )
                if process is not None:
                    process.start(adapter.healthcheck)
                elif tag not in started:
                    # No process to manage: the server is already running, so
                    # just confirm the worker's own port answers.
                    adapter.healthcheck()

    def _stop(self) -> None:
        for process in reversed(self.processes):
            process.stop()

    def _memory_config(self) -> dict[str, Any] | None:
        """Where the system's memory store lives, if the run should restore it."""
        config = dict(self.config.experiment.get("memory_store") or {})
        if not config.get("restore_after_run"):
            return None
        # `${AMEM_STORE_URL:-...}` is written the same way as every other
        # environment reference in the config, so it resolves the same way.
        return {
            **config,
            "url": expand(config.get("url") or "http://127.0.0.1:6333"),
            "collection": expand(config.get("collection") or "amem_memories"),
        }

    def _manifest(self, n_records: int | None) -> dict[str, Any]:
        """What the run WAS ASKED to do, plus how much of it landed.

        Written before the first turn and again at the end. Everything here
        except `n_records` is known up front, and a run that dies partway
        used to leave answers with no manifest beside them — so nothing
        downstream could say which system was the baseline, and the records
        were unreadable by `rescore`, `merge` and the viewer alike.
        """
        return {
            "config": str(self.config.path),
            "baseline": self.config.experiment["baseline"],
            "candidate": self.config.experiment["candidate"],
            "seed": self.config.experiment["seed"],
            "mode": self.config.experiment["mode"],
            "repeats": self.config.experiment["repeats"],
            "cases": self._cases,
            "workers": self.workers,
            # Latency was measured under contention when this is > 1: N servers
            # per system shared one machine, so the numbers are comparable
            # within the run but not against a serial one.
            "latency_measured_concurrently": self.workers > 1,
            # What the harness was allowed to re-ask, and on what. A reader
            # comparing two runs needs to know whether one of them was given
            # second chances the other was not.
            "retry_policy": {
                "outcomes": sorted(self._retry.outcomes),
                "attempts": self._retry.attempts,
                "backoff_s": self._retry.backoff_s,
            },
            # Each set is a separate session in stateful mode, so this records
            # how the conversation was actually cut up.
            "question_sets": {
                name: [question.name for question in questions]
                for name, questions in self._question_sets()
            },
            "question_count": len(self.config.questions),
            "n_records": n_records,
            "content_evaluation": {
                "enabled": self.config.content_evaluation.get("enabled"),
                "auto_run": self.config.content_evaluation.get("auto_run"),
                "judge_model": (self.config.content_evaluation.get("llm") or {}).get("model"),
            },
            "memory_evaluation": {
                "annotated_questions": [
                    question.name for question in self.config.questions
                    if question.evaluation.get("memory_required") is not None
                ],
                "required_questions": [
                    question.name for question in self.config.questions
                    if question.evaluation.get("memory_required") is True
                ],
            },
            "systems": {
                name: self._system_manifest(name, target)
                for name, target in self.config.systems.items()
            },
        }

    def run(self) -> Path:
        records: list[dict[str, Any]] = []
        # Up front, so a run that dies partway leaves answers that can still be
        # read: `rescore` needs the baseline/candidate roles, `merge` needs
        # them to refuse joining two different experiments, and the viewer
        # needs them or it infers the roles backwards. Rewritten at the end
        # with the real count; `n_records: null` means it never finished.
        self.layout.manifest.write_text(
            json.dumps(self._manifest(None), indent=2), encoding="utf-8",
        )
        # Snapshot BEFORE anything starts. A store that cannot be read is
        # reported and left alone — "no snapshot" must never be mistaken for
        # "snapshot of nothing", or a failed read would authorise wiping it.
        memory = self._memory_config()
        before = None
        if memory:
            try:
                before = memory_store.snapshot(
                    memory["url"], memory["collection"],
                )
                print(f"  memory store: {len(before)} memories before the run")
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                print(
                    f"  memory store: could not snapshot ({exc}); it will be "
                    "left exactly as the run leaves it"
                )
        try:
            self._start()
            mode = self.config.experiment["mode"]
            if mode in {"cold", "both"}:
                records.extend(self._cold())
            if mode in {"stateful", "both"}:
                records.extend(self._stateful())
        finally:
            self._stop()
            if memory and before is not None:
                try:
                    moved = memory_store.restore(
                        memory["url"], memory["collection"], before,
                    )
                    print(
                        f"  memory store: removed {moved['removed']} written "
                        f"by this run, reinstated {moved['reinstated']}; "
                        f"back to {len(before)}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  memory store: RESTORE FAILED ({exc}). It still "
                        f"holds this run's memories; {len(before)} were there "
                        "before."
                    )

        summary = aggregate(
            records, modules=self.config.experiment.get("eval_modules"),
        )
        comparisons = compare(
            summary,
            baseline=self.config.experiment["baseline"],
            candidate=self.config.experiment["candidate"],
            records=records,
            seed=self.config.experiment["seed"],
        )
        self.layout.summary.write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
        self.layout.comparison_json.write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8",
        )
        self.layout.comparison_md.write_text(
            comparison_markdown(
                comparisons,
                baseline=self.config.experiment["baseline"],
                candidate=self.config.experiment["candidate"],
            ),
            encoding="utf-8",
        )
        write_blind_review(
            records,
            review_path=self.layout.answer_review,
            key_path=self.layout.answer_review_key,
            seed=self.config.experiment["seed"],
        )
        # The judge-based cascade costs LLM calls, so it runs only when asked
        # for. Naming `content` explicitly is such a request; an unfiltered
        # "all" is not, and still defers to `content_evaluation.auto_run` so a
        # broad sweep cannot start spending by omission.
        selected = self.config.experiment.get("eval_modules")
        explicit = self.config.experiment.get("eval_modules_explicit", True)
        content_requested = (
            "content" in selected if selected is not None and explicit
            else bool(self.config.content_evaluation.get("auto_run"))
        )
        if self.config.content_evaluation.get("enabled") and content_requested:
            evaluate_runs_file(
                config=self.config.content_evaluation,
                records=records,
                output_dir=self.output_dir,
                baseline=self.config.experiment["baseline"],
                candidate=self.config.experiment["candidate"],
                rubric_by_name={
                    question.name: question.evaluation
                    for question in self.config.questions
                },
            )
        manifest = self._manifest(len(records))
        self.layout.manifest.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        return self.output_dir

    @staticmethod
    def _system_manifest(name: str, target: dict[str, Any]) -> dict[str, Any]:
        process = target.get("process") or {}
        cwd = process.get("cwd")
        revision = None
        dirty = None
        if cwd:
            try:
                revision = subprocess.check_output(
                    ["git", "-C", str(cwd), "rev-parse", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                ).strip()
                dirty = bool(subprocess.check_output(
                    ["git", "-C", str(cwd), "status", "--porcelain"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                ).strip())
            except (OSError, subprocess.SubprocessError):
                pass
        return {
            "adapter": target["adapter"],
            "version_label": target.get("version_label") or name,
            "cwd": cwd,
            "git_revision": revision,
            "git_dirty": dirty,
        }
