"""Command-line entry points for the decoupled comparison framework."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_eval.cases import (
    describe_case, discover_case_ids, whitespace_padded_case_ids,
)
from agentic_eval.config import load_config
from agentic_eval.layout import RunLayout
from agentic_eval.render import find_run_manifest, find_run_summary, write_answer_comparison, write_content_walkthrough
from agentic_eval.render import resolve_view_defaults
from agentic_eval.content import (
    evaluate_runs_file, read_jsonl,
    )
from agentic_eval.review import aggregate_review_files, write_review_summary
from agentic_eval.runner import ComparisonRunner
from agentic_eval.dimensions import ALL_MODULES, EVAL_MODULES, resolve_modules


def _add_override_flags(parser: argparse.ArgumentParser) -> None:
    """Per-invocation overrides, so one YAML serves a whole sweep."""
    parser.add_argument(
        "--scope", default=None, metavar="NAME",
        help=(
            "a full run preset from experiment.scopes: questions, k, cases "
            "and workers together, so 'the smoke run' is one word. For "
            "spending little, not for measuring. Explicit flags still win"
        ),
    )
    parser.add_argument(
        "--question-scope", default=None, metavar="NAME",
        dest="question_scope",
        help=(
            "a named question selection from experiment.question_scopes. "
            "Narrows WHAT is asked and nothing else — k, cases and workers "
            "stay as the config has them, so the rates mean what they "
            "usually mean"
        ),
    )
    parser.add_argument(
        "--case-id", action="append", default=None, dest="case_ids",
        metavar="ID",
        help=(
            "case to ask about, overriding the config; repeat the flag to "
            "cover several cases in one run. Every question is asked about "
            "every case"
        ),
    )
    parser.add_argument(
        "--cases-from", default=None, metavar="DIR",
        help=(
            "read the case ID LIST from a data directory (e.g. "
            "AgenticSys_v2/data_tables/real) and run all of them. Only the "
            "ids are taken: each system still serves the case from its own "
            "checkout's data tables"
        ),
    )
    parser.add_argument(
        "--mode", default=None, choices=["cold", "stateful", "both"],
        help=(
            "cold: reset before every turn, questions independent; "
            "stateful: one session per repeat, questions share history; "
            "both: cold pass then stateful pass"
        ),
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="override experiment.repeats, the shared k",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "run this many sessions at once (default 1). Each worker starts "
            "its OWN server per system, because the system keeps its data "
            "gateway process-global; sharing one would let two cases execute "
            "against each other's tables. Latency is then measured under "
            "contention"
        ),
    )
    parser.add_argument(
        "--baseline-cwd", default=None,
        help="checkout to run as the baseline, overriding systems.<baseline>.process.cwd",
    )
    parser.add_argument(
        "--candidate-cwd", default=None,
        help="checkout to run as the candidate, overriding systems.<candidate>.process.cwd",
    )
    parser.add_argument(
        "--question", action="append", default=None, dest="questions",
        metavar="NAME",
        help=(
            "run only these questions; repeat for several. Note this changes "
            "a stateful run: the selected questions become the whole session, "
            "so a follow-up asked without its parent has no referent."
        ),
    )
    parser.add_argument(
        "--eval-module", action="append", default=None, dest="eval_modules",
        metavar="MODULE",
        help=(
            "evaluation dimension(s) to run: "
            + ", ".join(sorted(EVAL_MODULES)) + f", or '{ALL_MODULES}'. "
            "Repeat the flag or pass a comma-separated list. Default: all."
        ),
    )


def _select_questions(config, wanted: set[str]) -> None:
    """Narrow the run to `wanted`, refusing a follow-up without its parent.

    A follow-up asked with no parent has no referent — "what is the total
    balance of these cards?" with no prior turn naming any cards. The system
    answers something vague, every metric on it reads badly, and the conclusion
    drawn is about the system rather than about the selection. That is exactly
    the trap a small smoke subset walks into, so it is refused rather than run.
    """
    known = {question.name for question in config.questions}
    unknown = wanted - known
    if unknown:
        raise ValueError(
            f"unknown question(s) {sorted(unknown)}; "
            f"the set defines {sorted(known)}"
        )
    parents = {
        question.name: str(
            ((question.evaluation.get("relation") or {}).get("parent") or "")
        )
        for question in config.questions
    }
    # Walk the whole chain: b4 needs b3, and b3 needs b2.
    missing: dict[str, str] = {}
    for name in sorted(wanted):
        step = name
        while (parent := parents.get(step)):
            if parent not in wanted:
                missing[step] = parent
            step = parent
    if missing:
        needed = sorted(set(missing.values()) - wanted)
        raise ValueError(
            "follow-up question(s) selected without the turn they refer to: "
            + ", ".join(f"{child} needs {parent}" for child, parent in sorted(missing.items()))
            + f". Add {' '.join('--question ' + name for name in needed)}, "
            "or select a question that stands on its own"
        )
    # The config is frozen; `questions` is a list, so filter in place.
    config.questions[:] = [q for q in config.questions if q.name in wanted]


#: What a full run scope may pin. A question scope may pin only `questions` —
#: everything else is deliberately left to the config, which is the whole point
#: of the distinction below.
_SCOPE_KEYS = {"questions", "repeats", "cases", "workers"}


def _apply_scope(config, name: str) -> None:
    """Apply a full RUN scope from `experiment.scopes`.

    A run scope pins the whole shape of a run — questions, k, cases, workers —
    so `--scope smoke` is one word rather than a line of flags remembered
    correctly. It is the right tool when the point is to spend little: a smoke
    run is not a measurement, it is a check that the chain works end to end.

    Distinct from a question scope (below), which narrows WHAT is asked and
    leaves HOW MANY TIMES to the config. Use that one when the numbers matter.
    """
    scopes = config.experiment.get("scopes") or {}
    scope = scopes.get(name)
    if scope is None:
        raise ValueError(
            f"unknown scope {name!r}; the config defines {sorted(scopes) or 'none'}"
        )
    if not isinstance(scope, dict):
        raise ValueError(f"experiment.scopes.{name} must be a mapping")
    unknown = set(scope) - _SCOPE_KEYS
    if unknown:
        raise ValueError(
            f"experiment.scopes.{name} sets unknown key(s) {sorted(unknown)}; "
            f"a scope may set {sorted(_SCOPE_KEYS)}"
        )
    if scope.get("repeats") is not None:
        config.experiment["repeats"] = int(scope["repeats"])
    if scope.get("workers") is not None:
        config.experiment["workers"] = int(scope["workers"])
    if scope.get("cases") is not None:
        config.experiment["cases"] = [str(case) for case in scope["cases"]]
    if scope.get("questions") is not None:
        _select_questions(config, {str(q) for q in scope["questions"]})


def _apply_question_scope(config, name: str) -> None:
    """Apply a named QUESTION scope from `experiment.question_scopes`.

    Narrows what is asked and nothing else: k, cases and workers stay exactly
    as the config has them. That is the difference from a run scope — this one
    is for asking a real question of a subset ("how does it do on series B?"),
    so the repeats behind every rate must be the ones the config intends.

    A question scope carrying `repeats` or `cases` is refused rather than
    honoured. Silently changing k here would hand back rates that look like the
    config's and are not, and nothing downstream would say so.
    """
    scopes = config.experiment.get("question_scopes") or {}
    scope = scopes.get(name)
    if scope is None:
        raise ValueError(
            f"unknown question scope {name!r}; the config defines "
            f"{sorted(scopes) or 'none'}"
        )
    # A bare list is the natural way to write "just these questions".
    questions = scope.get("questions") if isinstance(scope, dict) else scope
    if isinstance(scope, dict):
        pinned = sorted(set(scope) - {"questions", "description"})
        if pinned:
            raise ValueError(
                f"experiment.question_scopes.{name} sets {pinned}; a question "
                "scope selects questions only and takes k, cases and workers "
                f"from the config. Use experiment.scopes.{name} for a scope "
                "that pins those too"
            )
    if not isinstance(questions, list) or not questions:
        raise ValueError(
            f"experiment.question_scopes.{name} must be a non-empty list of "
            "question names, or a mapping with a `questions:` list"
        )
    _select_questions(config, {str(q) for q in questions})


def _apply_overrides(config, args):
    """Apply CLI overrides in place, so one YAML serves many cases/runs.

    A bash driver sweeping cases should not have to template a config file
    per case, and a selected module list must reach `aggregate` unchanged.
    """
    scope = getattr(args, "scope", None)
    question_scope = getattr(args, "question_scope", None)
    if scope and question_scope:
        raise ValueError(
            "--scope and --question-scope are alternatives: a run scope "
            "already pins the questions, so combining them would leave it "
            "unclear which selection ran"
        )
    if scope:
        _apply_scope(config, scope)
    if question_scope:
        _apply_question_scope(config, question_scope)
    case_ids = getattr(args, "case_ids", None)
    cases_from = getattr(args, "cases_from", None)
    if case_ids and cases_from:
        raise ValueError("--case-id and --cases-from are alternatives; give one")
    if cases_from:
        case_ids = discover_case_ids(cases_from)
    if case_ids:
        config.experiment["cases"] = list(case_ids)
        # The per-system id would otherwise still pin the adapter, and a run
        # asking about the config's case while reporting the flag's would be
        # wrong in a way no output shows.
        for system in config.systems.values():
            system.setdefault("config", {}).pop("case_id", None)
        if len(case_ids) == 1:
            for system in config.systems.values():
                system["config"]["case_id"] = case_ids[0]
    padded = whitespace_padded_case_ids([
        case for case in config.experiment.get("cases") or [] if case
    ])
    if padded:
        # Real data has one of these. Silence here would look like a run that
        # simply found nothing for that case.
        #
        # stderr, not stdout: `validate` writes JSON that `bin/compare` parses,
        # and a note printed alongside it makes the document unreadable.
        print(
            "note: case id(s) carry leading/trailing whitespace, matching the "
            "directory names on disk: "
            + ", ".join(describe_case(case) for case in padded),
            file=sys.stderr,
        )
    mode = getattr(args, "mode", None)
    if mode:
        config.experiment["mode"] = mode
    workers = getattr(args, "workers", None)
    if workers is not None:
        if workers < 1:
            raise ValueError("--workers must be at least 1")
        config.experiment["workers"] = workers
    repeats = getattr(args, "repeats", None)
    if repeats is not None:
        if repeats < 1:
            raise ValueError("--repeats must be at least 1")
        config.experiment["repeats"] = repeats
    for flag, role in (("baseline_cwd", "baseline"), ("candidate_cwd", "candidate")):
        cwd = getattr(args, flag, None)
        if not cwd:
            continue
        name = config.experiment.get(role)
        if name not in config.systems:
            raise ValueError(
                f"--{flag.replace('_', '-')} given, but experiment.{role} "
                f"({name!r}) is not one of {sorted(config.systems)}"
            )
        config.systems[name].setdefault("process", {})["cwd"] = cwd
    questions = getattr(args, "questions", None)
    if questions:
        _select_questions(config, {
            part.strip() for value in questions for part in str(value).split(",")
        })
    # `expected_repeats` was captured at load time from the config's k. Any
    # override above changes how many repeats actually run, and leaving the
    # expectation behind made every scoped run report `repetitions_complete:
    # false` — a complete evaluation declaring itself partial.
    config.content_evaluation["expected_repeats"] = config.experiment["repeats"]
    modules = getattr(args, "eval_modules", None)
    if modules:
        # Resolve here so a bad name fails before either system is started.
        resolved = resolve_modules(modules)
        config.experiment["eval_modules"] = resolved
        named = {
            part.strip() for value in modules for part in str(value).split(",")
        }
        if ALL_MODULES in named:
            # `all` selects every metric family, but leaves the paid content
            # cascade under the config's `auto_run`, per runner.
            config.experiment["eval_modules_explicit"] = False
        else:
            config.experiment["eval_modules_explicit"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentic-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run baseline vs candidate")
    run_parser.add_argument("--config", required=True)
    _add_override_flags(run_parser)
    validate_parser = subparsers.add_parser(
        "validate", help="validate YAML without starting either system",
    )
    validate_parser.add_argument("--config", required=True)
    _add_override_flags(validate_parser)
    review_parser = subparsers.add_parser(
        "score-reviews", help="aggregate a completed blind-review sheet",
    )
    review_parser.add_argument("--review", required=True, type=Path)
    review_parser.add_argument("--key", required=True, type=Path)
    review_parser.add_argument("--out-stem", type=Path, default=None)
    rescore_parser = subparsers.add_parser(
        "rescore",
        help="recompute the metric artifacts from an existing runs.jsonl",
    )
    rescore_parser.add_argument("--runs", required=True, type=Path)
    rescore_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="run folder to write into; default: the runs.jsonl's folder",
    )
    rescore_parser.add_argument("--baseline", default=None)
    rescore_parser.add_argument("--candidate", default=None)
    content_parser = subparsers.add_parser(
        "evaluate-content",
        help="score answers on claims, grounding and must-haves over runs.jsonl",
    )
    content_parser.add_argument("--config", required=True)
    content_parser.add_argument("--runs", required=True, type=Path)
    content_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="run folder to write into; default: the runs.jsonl's folder",
    )
    content_parser.add_argument(
        "--limit", type=int, default=None,
        help="evaluate only the first N eligible answers (useful for calibration)",
    )
    content_parser.add_argument(
        "--question", action="append", default=None, metavar="NAME",
        help="evaluate only these question names; repeat for several. Calibrating "
             "on one question costs a fraction of a full pass",
    )
    content_parser.add_argument(
        "--resume", action="store_true",
        help="append only answers not already present in content_evaluations.jsonl",
    )
    walkthrough_parser = subparsers.add_parser(
        "walkthrough",
        help="render answer -> atomic facts -> numeric verdicts as markdown",
    )
    walkthrough_parser.add_argument("--evaluations", required=True, type=Path)
    walkthrough_parser.add_argument("--output-dir", type=Path, default=None)
    compare_parser = subparsers.add_parser(
        "compare-answers",
        help="side-by-side viewer for one repeat: answers, facts, metrics",
    )
    compare_parser.add_argument("--evaluations", required=True, type=Path)
    compare_parser.add_argument("--output-dir", type=Path, default=None)
    compare_parser.add_argument(
        "--baseline", default=None,
        help="default: the run manifest's baseline",
    )
    compare_parser.add_argument(
        "--candidate", default=None,
        help="default: the run manifest's candidate",
    )
    compare_parser.add_argument(
        "--mode", dest="view_mode", default=None,
        help="cold or stateful; default: the run manifest's mode, else the first present",
    )
    compare_parser.add_argument(
        "--run-index", type=int, default=None,
        help="which repeat to sample; default: the first repeat present",
    )
    args = parser.parse_args()

    if args.command == "compare-answers":
        path = args.evaluations.expanduser().resolve()
        layout = RunLayout(args.output_dir.expanduser().resolve()) if args.output_dir \
            else (RunLayout.find(path) or RunLayout(path.parent))
        rows = read_jsonl(path)
        baseline, candidate, mode, source = resolve_view_defaults(
            rows, manifest=find_run_manifest(path),
            baseline=args.baseline, candidate=args.candidate,
            mode=args.view_mode,
        )
        written = write_answer_comparison(
            rows, layout=layout, baseline=baseline,
            candidate=candidate, mode=mode, run_index=args.run_index,
            summary=find_run_summary(path),
        )
        print(
            f"Comparison written: {written}\n"
            f"  baseline={baseline} candidate={candidate} ({source})"
        )
        return

    if args.command == "rescore":
        # The metric artifacts are written once, by `run`. When a scoring bug
        # is fixed afterwards the answers are still good — only the numbers
        # derived from them are stale, and re-running both systems to correct
        # arithmetic wastes the run and changes the sample. This recomputes
        # them in place from the answers already on disk.
        from agentic_eval.scoring import aggregate, compare
        from agentic_eval.render.run_summary import comparison_markdown

        path = args.runs.expanduser().resolve()
        layout = RunLayout(args.output_dir.expanduser().resolve()) if args.output_dir \
            else (RunLayout.find(path) or RunLayout(path.parent))
        layout.ensure()
        records = read_jsonl(path)
        manifest = find_run_manifest(path)
        baseline = args.baseline or manifest.get("baseline")
        candidate = args.candidate or manifest.get("candidate")
        if not baseline or not candidate:
            raise ValueError(
                "cannot tell which system is the baseline: no manifest.json "
                "beside the runs file, so pass --baseline and --candidate"
            )
        summary = aggregate(records, modules=manifest.get("eval_modules"))
        comparisons = compare(
            summary, baseline=baseline, candidate=candidate,
            records=records, seed=int(manifest.get("seed") or 20260731),
        )
        layout.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        layout.comparison_json.write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8",
        )
        layout.comparison_md.write_text(
            comparison_markdown(
                comparisons, baseline=baseline, candidate=candidate,
            ),
            encoding="utf-8",
        )
        print(
            f"Rescored {len(records)} records: {layout.summary}\n"
            f"  baseline={baseline} candidate={candidate}"
        )
        return

    if args.command == "walkthrough":
        path = args.evaluations.expanduser().resolve()
        run = RunLayout(args.output_dir.expanduser().resolve()) if args.output_dir \
            else (RunLayout.find(path) or RunLayout(path.parent))
        output_dir = run.root
        written = write_content_walkthrough(
            read_jsonl(path), layout=RunLayout(output_dir).ensure(),
        )
        print(f"Walkthrough written: {written}")
        return

    if args.command == "score-reviews":
        summary = aggregate_review_files(args.review, args.key)
        stem = args.out_stem or args.review.with_name("human_quality")
        write_review_summary(summary, stem)
        print(f"Reviewed {summary['n_reviewed']} answers; summary: {stem}.md")
        return

    config = load_config(args.config)
    config = _apply_overrides(config, args)
    if args.command == "evaluate-content":
        if not config.content_evaluation.get("enabled"):
            raise ValueError("content_evaluation.enabled must be true")
        runs_path = args.runs.expanduser().resolve()
        # The RUN ROOT; `RunLayout` decides where each artifact lands.
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir else runs_path.parent
        )
        rubric_by_name = {
            question.name: question.evaluation for question in config.questions
        }
        path = evaluate_runs_file(
            config=config.content_evaluation,
            records=read_jsonl(runs_path),
            output_dir=output_dir,
            baseline=config.experiment["baseline"],
            candidate=config.experiment["candidate"],
            rubric_by_name=rubric_by_name,
            limit=args.limit,
            questions=args.question,
            resume=args.resume,
        )
        print(f"Content evaluation complete: {path}")
        return
    if args.command == "validate":
        print(json.dumps({
            "config": str(config.path),
            "mode": config.experiment["mode"],
            "repeats": config.experiment["repeats"],
            "workers": config.experiment.get("workers", 1),
            "scope": getattr(args, "scope", None),
            "question_scope": getattr(args, "question_scope", None),
            "available_scopes": sorted(config.experiment.get("scopes") or {}),
            "available_question_scopes": sorted(
                config.experiment.get("question_scopes") or {}
            ),
            "systems": list(config.systems),
            "questions": [question.name for question in config.questions],
            "cases": [
                case for case in config.experiment.get("cases") or [] if case
            ],
            # In stateful mode each set is its own session, so this is the
            # conversation layout the run will actually use.
            "question_sets": {
                name: [q.name for q in config.questions if q.question_set == name]
                for name in dict.fromkeys(
                    q.question_set or "questions" for q in config.questions
                )
            },
            "eval_modules": resolve_modules(
                config.experiment.get("eval_modules")
            ),
            "content_evaluation": {
                "enabled": config.content_evaluation["enabled"],
                "auto_run": config.content_evaluation["auto_run"],
                "model": config.content_evaluation["llm"]["model"],
            },
            "memory_evaluation": {
                "annotated_questions": [
                    question.name for question in config.questions
                    if question.evaluation.get("memory_required") is not None
                ],
                "required_questions": [
                    question.name for question in config.questions
                    if question.evaluation.get("memory_required") is True
                ],
            },
            "planned_records": (
                len(config.questions) * config.experiment["repeats"]
                * len(config.systems)
                * max(1, len(config.experiment.get("cases") or []))
                * (2 if config.experiment["mode"] == "both" else 1)
            ),
        }, indent=2))
        return
    output = ComparisonRunner(config).run()
    print(f"Comparison complete: {output}")
