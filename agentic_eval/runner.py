"""Paired, randomized baseline/candidate experiment runner."""
from __future__ import annotations

import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_eval.adapters import build_adapter
from agentic_eval.config import EvalConfig
from agentic_eval.content import evaluate_runs_file
from agentic_eval.models import AdapterResult, RunRequest
from agentic_eval.process import ManagedProcess
from agentic_eval.report import comparison_markdown, write_blind_review
from agentic_eval.layout import RunLayout
from agentic_eval.scoring import aggregate, compare, score_content, score_memory


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
        self.adapters = {
            name: build_adapter(target["adapter"], target.get("config") or {})
            for name, target in config.systems.items()
        }
        self.processes = {
            name: ManagedProcess(name, target["process"], self.output_dir)
            for name, target in config.systems.items() if target.get("process")
        }
        self.random = random.Random(experiment["seed"])

    def _persist(self, record: dict[str, Any]) -> None:
        with self.raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _one(self, system: str, request: RunRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = self.adapters[system].run(
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
        self._persist(record)
        print(
            f"  {system:<14} {record['outcome']:<12} "
            f"{record['elapsed_seconds']:>7.2f}s  "
            f"{request.question.name} #{request.run_index}"
        )
        return record

    def _reset(self, system: str) -> None:
        self.adapters[system].reset()

    def _cold(self) -> list[dict[str, Any]]:
        systems = list(self.config.systems)
        records = []
        for question in self.config.questions:
            for run_index in range(1, self.config.experiment["repeats"] + 1):
                order = list(systems)
                self.random.shuffle(order)
                for system in order:
                    self._reset(system)
                    records.append(self._one(
                        system,
                        RunRequest(question, run_index, "cold"),
                    ))
        return records

    def _stateful(self) -> list[dict[str, Any]]:
        systems = list(self.config.systems)
        records = []
        for run_index in range(1, self.config.experiment["repeats"] + 1):
            order = list(systems)
            self.random.shuffle(order)
            for system in order:
                self._reset(system)
                for position, question in enumerate(self.config.questions, 1):
                    records.append(self._one(
                        system,
                        RunRequest(question, run_index, "stateful", position),
                    ))
        return records

    def _start(self) -> None:
        for name, adapter in self.adapters.items():
            if name in self.processes:
                self.processes[name].start(adapter.healthcheck)
            else:
                adapter.healthcheck()

    def _stop(self) -> None:
        for process in reversed(list(self.processes.values())):
            process.stop()

    def run(self) -> Path:
        records: list[dict[str, Any]] = []
        try:
            self._start()
            mode = self.config.experiment["mode"]
            if mode in {"cold", "both"}:
                records.extend(self._cold())
            if mode in {"stateful", "both"}:
                records.extend(self._stateful())
        finally:
            self._stop()

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
        manifest = {
            "config": str(self.config.path),
            "baseline": self.config.experiment["baseline"],
            "candidate": self.config.experiment["candidate"],
            "seed": self.config.experiment["seed"],
            "mode": self.config.experiment["mode"],
            "repeats": self.config.experiment["repeats"],
            "question_count": len(self.config.questions),
            "n_records": len(records),
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
