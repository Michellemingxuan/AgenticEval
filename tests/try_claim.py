#!/usr/bin/env python3
"""Classify a single claim by hand, one layer at a time.

Both layers are pure functions over plain dicts — no LLM, no I/O — so a claim
can be driven through them directly. Use this to answer "why did the evaluator
say that?" without paying for a judge call or re-running a suite.

    # a proposition you type
    python tests/try_claim.py "Curated reports state a balance of $0 …"

    # with evidence, to reach the verification layer
    python tests/try_claim.py "The balance is $174,897.36." \\
        --evidence "sum(Balance) filtered by Card Portfolio eq 'SBS' = $174,897.36"

    # a real claim from a finished run
    python tests/try_claim.py --from experiments/results/<run>/content/evaluations.jsonl \\
        --question commercial_card_balance --system current --claim c004

Named for pytest's benefit: no `test_` prefix, so it is not collected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_eval.content.claims import (  # noqa: E402
    _CELL_NUMBER,
    _is_report_attribution,
    _normalize_claim,
)
from agentic_eval.content.evidence import _evidence_tier, build_evidence_ledger  # noqa: E402
from agentic_eval.content.verify import (  # noqa: E402
    JUDGE_ERROR_FAILURES,
    SYSTEM_FAILURES,
    _normalize_fact_results,
)
from agentic_eval.common.coerce import _safe_float  # noqa: E402


def _mentions_from_text(text: str) -> list[dict]:
    """Approximate what the extractor would emit, so numbers need not be typed."""
    out = []
    for match in _CELL_NUMBER.finditer(text):
        written = match.group(1).strip()
        value = _safe_float(written)
        if value is not None:
            out.append({"written": written, "value": value, "material": True})
    return out


def _from_run(path: Path, question: str | None, system: str | None, claim_id: str | None):
    """Pull a real claim, its fact result, and its ledger out of a finished run."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [
        row for row in rows
        if (question is None or row.get("name") == question)
        and (system is None or row.get("system") == system)
    ]
    if not matches:
        raise SystemExit(
            f"no evaluation matched question={question} system={system}. "
            f"available: {sorted({(r.get('system'), r.get('name')) for r in rows})}"
        )
    row = matches[0]
    claims = row.get("claims") or []
    if claim_id:
        claims = [c for c in claims if c.get("claim_id") == claim_id]
        if not claims:
            raise SystemExit(
                f"no claim {claim_id}; available: "
                f"{[c.get('claim_id') for c in (row.get('claims') or [])]}"
            )
    raw_facts = {f.get("claim_id"): f for f in row.get("fact_results") or []}
    return row, claims, raw_facts


def _print_claim(claim: dict) -> None:
    print(f"  claim_id            {claim['claim_id']}")
    print(f"  proposition         {claim['proposition']}")
    print(f"  claim_type          {claim['claim_type']}")
    print(f"  stance              {claim['stance']}")
    print(f"  report_attribution  {claim['report_attribution']}")
    if claim.get("restates_claim_id"):
        print(f"  restates            {claim['restates_claim_id']}")
    if not claim["numeric_mentions"]:
        print("  numeric_mentions    (none)")
    for mention in claim["numeric_mentions"]:
        flags = []
        if mention.get("quoted"):
            flags.append("quoted")
        if not mention.get("material"):
            flags.append("IMMATERIAL — not checked")
        print(
            f"  mention             {mention['written']!r} -> {mention['value']}"
            f"  comparator={mention['comparator']}"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )


def _print_fact(fact: dict) -> None:
    for key in (
        "numeric_support", "traceability", "number_correctness",
        "reasoning_trace_correctness", "actual_correctness", "hallucination",
        "evidence_grounding", "factual_verdict",
    ):
        print(f"  {key:<23} {fact.get(key)}")
    if fact.get("hallucination_causes"):
        print(f"  {'hallucination_causes':<23} {fact['hallucination_causes']}")
    if fact.get("judge_error"):
        print(f"  {'judge_error':<23} True  {fact.get('judge_error_failures')}")
    for number in fact.get("numbers") or []:
        failure = number.get("trace_failure")
        blame = (
            "" if not failure
            else "  [EVALUATOR failed]" if failure in JUDGE_ERROR_FAILURES
            else "  [system failure]" if failure in SYSTEM_FAILURES
            else ""
        )
        print(
            f"  number                {number['written_value']!r} "
            f"{number.get('comparator')} {number.get('answer_value')} "
            f"vs {number.get('evidence_value')}  "
            f"{failure or 'TRACED'}{blame}"
        )
        if number.get("path_repaired"):
            print(
                f"  {'':<22}  pointer repaired: {number['json_path']!r} "
                f"-> {number['resolved_json_path']!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("proposition", nargs="?", help="the claim text to classify")
    parser.add_argument(
        "--evidence", action="append", default=None,
        help="a tool result the claim cites; repeat for several. JSON is parsed, "
             "anything else is kept as the scalar string a data tool returned",
    )
    parser.add_argument("--json-path", default="", help="path the judge would supply")
    parser.add_argument(
        "--stance", default=None,
        help="asserted | attributed_unendorsed | attributed_refuted",
    )
    parser.add_argument("--type", dest="claim_type", default="quantitative_fact")
    parser.add_argument(
        "--number", action="append", default=None,
        help="override the auto-detected mentions, e.g. --number '$0=0'",
    )
    parser.add_argument("--from", dest="source", type=Path, default=None,
                        help="evaluations.jsonl from a finished run")
    parser.add_argument("--question", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--claim", default=None, help="claim_id within that answer")
    args = parser.parse_args()

    if args.source:
        row, raw_claims, raw_facts = _from_run(
            args.source, args.question, args.system, args.claim,
        )
        # An evaluation record already carries its built ledger; a raw run
        # record carries `evidence`. Rebuilding from the wrong field yields an
        # empty ledger and reports every number as citing an unknown id.
        ledger = row.get("evidence_ledger") or build_evidence_ledger(row, {})
        claims = [_normalize_claim(c, i) for i, c in enumerate(raw_claims, 1)]
        facts = _normalize_fact_results(
            claims, [raw_facts[c["claim_id"]] for c in claims if c["claim_id"] in raw_facts],
            ledger,
        )
        by_id = {f["claim_id"]: f for f in facts}
        print(f"run: {row.get('system')} / {row.get('name')} #{row.get('run_index')}")
        print(f"evidence: {[(e['tool'], e.get('evidence_tier')) for e in ledger]}\n")
        for claim in claims:
            print("-" * 78)
            _print_claim(claim)
            fact = by_id.get(claim["claim_id"])
            if fact:
                print("  --- verification ---")
                _print_fact(fact)
        return 0

    if not args.proposition:
        parser.error("give a proposition, or --from a run's evaluations.jsonl")

    mentions = (
        [
            {
                "written": item.split("=", 1)[0],
                "value": _safe_float(item.split("=", 1)[-1]),
                "material": True,
            }
            for item in args.number
        ]
        if args.number else _mentions_from_text(args.proposition)
    )
    raw_claim = {
        "claim_id": "c1", "proposition": args.proposition, "is_factual": True,
        "claim_type": args.claim_type, "numeric_mentions": mentions,
    }
    if args.stance:
        raw_claim["stance"] = args.stance

    print(f"_is_report_attribution -> {_is_report_attribution(args.proposition)}\n")
    print("--- claim extraction ---")
    claim = _normalize_claim(raw_claim, 1)
    _print_claim(claim)

    if not args.evidence:
        print("\n(no --evidence given, so the verification layer was not run)")
        return 0

    ledger = []
    for index, item in enumerate(args.evidence, 1):
        try:
            result = json.loads(item)
        except json.JSONDecodeError:
            result = item
        entry = {
            "evidence_id": f"ev{index}", "source_type": "tool_result",
            "tool": "manual", "result": result,
        }
        entry["evidence_tier"] = _evidence_tier(entry)
        ledger.append(entry)

    raw_fact = {
        "claim_id": "c1", "numeric_support": "YES",
        "evidence_ids": [e["evidence_id"] for e in ledger],
        "factual_verdict": "SUPPORTED", "reasoning_trace_verdict": "YES",
        "trace_evidence_ids": [e["evidence_id"] for e in ledger],
        "reason": "(supplied by try_claim)",
        "numbers": [
            {
                "written_value": mention["written"], "evidence_id": "ev1",
                "json_path": args.json_path, "trace_kind": "direct",
            }
            for mention in claim["numeric_mentions"] if mention["material"]
        ],
    }
    print("\n--- evidence ledger ---")
    for entry in ledger:
        print(f"  {entry['evidence_id']}  tier={entry['evidence_tier']}  "
              f"{str(entry['result'])[:88]!r}")
    print("\n--- verification ---")
    _print_fact(_normalize_fact_results([claim], [raw_fact], ledger)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
