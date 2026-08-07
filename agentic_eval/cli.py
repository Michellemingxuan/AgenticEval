"""Command-line entry points for the decoupled comparison framework."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_eval.config import load_config
from agentic_eval.layout import RunLayout
from agentic_eval.content import (
    evaluate_runs_file, find_run_manifest, find_run_summary, read_jsonl,
    resolve_view_defaults,
    write_answer_comparison, write_content_walkthrough,
)
from agentic_eval.reviews import aggregate_review_files, write_review_summary
from agentic_eval.runner import ComparisonRunner
from agentic_eval.modules import ALL_MODULES, EVAL_MODULES, resolve_modules


def _add_override_flags(parser: argparse.ArgumentParser) -> None:
    """Per-invocation overrides, so one YAML serves a whole sweep."""
    parser.add_argument(
        "--case-id", default=None,
        help="override the case id for every system (one case per run)",
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


def _apply_overrides(config, args):
    """Apply CLI overrides in place, so one YAML serves many cases/runs.

    A bash driver sweeping cases should not have to template a config file
    per case, and a selected module list must reach `aggregate` unchanged.
    """
    case_id = getattr(args, "case_id", None)
    if case_id:
        for system in config.systems.values():
            system.setdefault("config", {})["case_id"] = case_id
    mode = getattr(args, "mode", None)
    if mode:
        config.experiment["mode"] = mode
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
        wanted = {
            part.strip() for value in questions for part in str(value).split(",")
        }
        known = {question.name for question in config.questions}
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"unknown question(s) {sorted(unknown)}; "
                f"the set defines {sorted(known)}"
            )
        # The config is frozen; `questions` is a list, so filter in place.
        config.questions[:] = [q for q in config.questions if q.name in wanted]
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
            "systems": list(config.systems),
            "questions": [question.name for question in config.questions],
            "case_ids": sorted({
                str((system.get("config") or {}).get("case_id"))
                for system in config.systems.values()
                if (system.get("config") or {}).get("case_id")
            }),
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
                * (2 if config.experiment["mode"] == "both" else 1)
            ),
        }, indent=2))
        return
    output = ComparisonRunner(config).run()
    print(f"Comparison complete: {output}")
