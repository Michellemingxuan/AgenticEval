"""YAML loading, path resolution, and validation."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentic_eval.cases import describe_case, discover_case_ids
from agentic_eval.scoring import resolve_modules
from agentic_eval.models import Question


@dataclass(frozen=True)
class EvalConfig:
    path: Path
    experiment: dict[str, Any]
    systems: dict[str, dict[str, Any]]
    questions: list[Question]
    content_evaluation: dict[str, Any]


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _resolve_path(base: Path, value: str | None) -> str | None:
    if not value:
        return value
    expanded = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return str(expanded if expanded.is_absolute() else (base / expanded).resolve())


def _normalize_memory_evaluation(
    evaluation: dict[str, Any], row: dict[str, Any], *, question_name: str,
) -> dict[str, Any]:
    evaluation = dict(evaluation)
    if row.get("memory_required") is not None:
        evaluation.setdefault("memory_required", row["memory_required"])
    memory = evaluation.get("memory")
    if memory is not None:
        if not isinstance(memory, dict):
            raise ValueError(f"question {question_name}: evaluation.memory must be a mapping")
        if memory.get("required") is not None:
            evaluation.setdefault("memory_required", memory["required"])
    if "memory_required" in evaluation and not isinstance(
        evaluation["memory_required"], bool
    ):
        raise ValueError(f"question {question_name}: memory_required must be true or false")
    return evaluation


def _questions(data: dict[str, Any], base: Path) -> list[Question]:
    if data.get("questions_file"):
        # One file or several, concatenated in the order given. A suite mixes
        # simple questions a script can settle with complex ones only a rubric
        # can, and they are maintained separately — but they are asked in one
        # session, so the run must see a single ordered list.
        sources = data["questions_file"]
        rows = []
        for source in sources if isinstance(sources, list) else [sources]:
            path = Path(_resolve_path(base, source) or "")
            loaded = _load_mapping(path)
            # The file is the set unless it names one. Each set becomes its own
            # stateful session, so this is what keeps series D's cold ask cold.
            set_name = str(loaded.get("question_set") or path.stem)
            for row in loaded.get("questions") or loaded.get("test_cases") or []:
                if isinstance(row, dict):
                    row = {"question_set": set_name, **row}
                rows.append(row)
    else:
        q_data = data
        rows = q_data.get("questions") or q_data.get("test_cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("questions/questions_file must provide a non-empty list")
    out: list[Question] = []
    names: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"question #{index} must be a mapping")
        name = str(row.get("name") or "").strip()
        text = str(row.get("question") or row.get("text") or "").strip()
        if not name or not text:
            raise ValueError(f"question #{index} needs name and question/text")
        if name in names:
            raise ValueError(f"duplicate question name: {name}")
        names.add(name)
        evaluation = _normalize_memory_evaluation(
            dict(row.get("evaluation") or {}), row, question_name=name,
        )
        if row.get("relation") is not None:
            evaluation.setdefault("relation", row["relation"])
        out.append(Question(
            name, text, evaluation,
            # Inline questions are one unnamed set: a single config-level list
            # is one conversation, which is what it was before sets existed.
            question_set=str(row.get("question_set") or "questions"),
        ))
    return out


def _content_config(data: dict[str, Any], base: Path) -> dict[str, Any]:
    content = dict(data.get("content_evaluation") or {})
    if content.get("rubric_file"):
        content["rubric_file"] = _resolve_path(base, str(content["rubric_file"]))
    content["enabled"] = bool(content.get("enabled", bool(content)))
    content["auto_run"] = bool(content.get("auto_run", False))
    content["audit_claim_extraction"] = bool(
        content.get("audit_claim_extraction", True)
    )
    content["max_evidence_chars"] = int(content.get("max_evidence_chars", 60000))
    # Expected-answer oracle scripts resolve from the config file, like every
    # other path, so a rubric is portable between checkouts.
    content["oracle_cwd"] = _resolve_path(base, str(content.get("oracle_cwd") or "."))
    content["oracle_timeout_s"] = float(content.get("oracle_timeout_s", 60))
    llm = dict(content.get("llm") or {})
    llm.setdefault("backend", "openai")
    llm.setdefault("model", "gpt-4.1")
    llm.setdefault("temperature", 0)
    llm.setdefault("max_retries", 8)
    content["llm"] = llm
    return content


def _merge_content_rubric(
    questions: list[Question], content: dict[str, Any],
) -> list[Question]:
    path = content.get("rubric_file")
    if not path:
        return questions
    data = _load_mapping(Path(path))
    rows = data.get("questions") or data.get("test_cases") or []
    if not isinstance(rows, list):
        raise ValueError("content_evaluation.rubric_file must contain questions")
    rubric_by_name = {
        str(row.get("name")): row for row in rows
        if isinstance(row, dict) and row.get("name")
    }
    orphans = sorted(set(rubric_by_name) - {question.name for question in questions})
    if orphans:
        # A rubric entry matching no question is a dead check that still looks
        # configured: the suite runs, the expectation never fires, and the
        # report shows a blank rather than a failure.
        raise ValueError(
            f"{path}: rubric entries match no question: {', '.join(orphans)}; "
            f"questions are: {', '.join(question.name for question in questions)}"
        )
    merged = []
    for question in questions:
        row = rubric_by_name.get(question.name)
        if not row:
            merged.append(question)
            continue
        evaluation = dict(question.evaluation)
        evaluation.update(dict(row.get("evaluation") or {}))
        if row.get("relation") is not None:
            evaluation["relation"] = row["relation"]
        evaluation = _normalize_memory_evaluation(
            evaluation, row, question_name=question.name,
        )
        merged.append(Question(
            question.name, question.text, evaluation,
            question_set=question.question_set,
        ))
    return merged


def _cases(
    experiment: dict[str, Any], systems: dict[str, dict[str, Any]], base: Path,
) -> list[str | None]:
    """The cases this run covers, in the order they will be asked.

    Three sources, most explicit first: `experiment.cases` (a literal list),
    `experiment.cases_from` (a data directory to discover), and the per-system
    `config.case_id` that single-case configs already carry. The last keeps
    every existing config working unchanged — it resolves to a one-case run.

    Returns `[None]` when nothing names a case, so the runner always loops over
    at least one case and an adapter that does not take a case id still runs.
    """
    listed = experiment.get("cases")
    discovered = experiment.get("cases_from")
    if listed is not None and discovered is not None:
        raise ValueError(
            "experiment.cases and experiment.cases_from are alternatives; "
            "give one, not both"
        )
    if listed is not None:
        if not isinstance(listed, list) or not listed:
            raise ValueError("experiment.cases must be a non-empty list")
        cases = [str(case) for case in listed]
    elif discovered is not None:
        cases = discover_case_ids(_resolve_path(base, str(discovered)) or "")
    else:
        # Per-system case ids must agree: two systems asked about different
        # customers are not a comparison, and nothing downstream would say so.
        configured = {
            str(case_id)
            for target in systems.values()
            if (case_id := (target.get("config") or {}).get("case_id"))
        }
        if len(configured) > 1:
            raise ValueError(
                "systems disagree on case_id "
                f"({', '.join(sorted(describe_case(c) for c in configured))}); "
                "use experiment.cases to run several cases against both"
            )
        if not configured:
            return [None]
        cases = sorted(configured)
    duplicates = sorted({case for case in cases if cases.count(case) > 1})
    if duplicates:
        raise ValueError(
            f"experiment.cases repeats {', '.join(map(describe_case, duplicates))}; "
            "a case asked twice inflates every rate it contributes to"
        )
    return list(cases)


def _system_config(name: str, raw: dict[str, Any], base: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"systems.{name} must be a mapping")
    out = dict(raw)
    if not out.get("adapter"):
        raise ValueError(f"systems.{name}.adapter is required")
    process = dict(out.get("process") or {})
    if process:
        process["cwd"] = _resolve_path(base, process.get("cwd") or ".")
        command = process.get("command")
        if isinstance(command, str):
            process["command"] = shlex.split(command)
        if not isinstance(process.get("command"), list) or not process["command"]:
            raise ValueError(f"systems.{name}.process.command must be a list/string")
        process["env"] = {
            str(k): os.path.expanduser(os.path.expandvars(str(v)))
            for k, v in (process.get("env") or {}).items()
        }
        if process.get("stdout"):
            process["stdout"] = _resolve_path(base, process["stdout"])
        out["process"] = process
    adapter_cfg = dict(out.get("config") or {})
    if adapter_cfg.get("trace_db"):
        adapter_cfg["trace_db"] = _resolve_path(base, adapter_cfg["trace_db"])
    out["config"] = adapter_cfg
    if process and adapter_cfg.get("trace_db"):
        # Keep the server writer and evaluator reader on exactly the same file.
        process.setdefault("env", {}).setdefault(
            "NODE_TRACE_DB", adapter_cfg["trace_db"]
        )
        out["process"] = process
    return out


def load_config(path: str | Path) -> EvalConfig:
    config_path = Path(path).expanduser().resolve()
    data = _load_mapping(config_path)
    if int(data.get("version") or 1) != 1:
        raise ValueError("only config version 1 is supported")
    base = config_path.parent
    experiment = dict(data.get("experiment") or {})
    mode = experiment.get("mode", "cold")
    if mode not in {"cold", "stateful", "both"}:
        raise ValueError("experiment.mode must be cold, stateful, or both")
    repeats = int(experiment.get("repeats", 10))
    if repeats < 1:
        raise ValueError("experiment.repeats must be >= 1")
    experiment["mode"] = mode
    experiment["repeats"] = repeats
    # Each worker starts its own server instance per system, because the system
    # under test keeps its data gateway process-global and re-scopes it per
    # turn — two sessions on one process can execute against each other's case.
    # So this is bounded low: it costs real processes, not just threads.
    workers = int(experiment.get("workers", 1) or 1)
    if workers < 1:
        raise ValueError("experiment.workers must be >= 1")
    if workers > 8:
        raise ValueError(
            f"experiment.workers is {workers}: each one runs a full server per "
            "system, so this would start "
            f"{workers * len(data.get('systems') or {})} processes. Keep it <= 8"
        )
    experiment["workers"] = workers
    experiment["timeout_s"] = float(experiment.get("timeout_s", 600))
    experiment["seed"] = int(experiment.get("seed", 20260731))
    if experiment.get("eval_modules") is not None:
        experiment["eval_modules"] = resolve_modules(experiment["eval_modules"])
    experiment["output_dir"] = _resolve_path(
        base, experiment.get("output_dir") or "results"
    )

    raw_systems = data.get("systems")
    if not isinstance(raw_systems, dict) or len(raw_systems) != 2:
        raise ValueError("systems must contain exactly two named targets")
    systems = {
        str(name): _system_config(str(name), raw, base)
        for name, raw in raw_systems.items()
    }
    baseline = experiment.get("baseline") or next(iter(systems))
    candidate = experiment.get("candidate") or list(systems)[1]
    if baseline == candidate or baseline not in systems or candidate not in systems:
        raise ValueError("baseline and candidate must name two different systems")
    experiment["baseline"] = baseline
    experiment["candidate"] = candidate
    experiment["cases"] = _cases(experiment, systems, base)
    experiment.pop("cases_from", None)
    content = _content_config(data, base)
    # The same physical k runs feed latency, consistency, and content scoring.
    # Keeping this in the resolved content config lets post-processing detect a
    # partial/limited content evaluation without creating a second repeat knob.
    content["expected_repeats"] = repeats
    questions = _merge_content_rubric(_questions(data, base), content)
    return EvalConfig(config_path, experiment, systems, questions, content)
