"""Content-quality evaluation.

The module is a cascade, one step per question, each step's denominator being
the previous step's yes-branch:

    document  -> answer into blocks (Python)
    claims    -> blocks into atomic facts (LLM, with Python coverage guarantees)
    evidence  -> the ledger of what the system actually measured
    numeric   -> step 2: located AND correctly derived
    evidence  -> step 3: was the right tool used
    oracles   -> Python ground truth, no judge involved
    metrics   -> explicit denominators
    report    -> scorecard and walkthrough

This facade re-exports the public surface so callers need not know the layout.
"""
from agentic_eval.common.io import read_jsonl
from agentic_eval.content.aggregate import (
    _MACRO_METRICS,
    aggregate_content_evaluations,
)
from agentic_eval.content.claims import _normalize_claim
from agentic_eval.content.compare_view import (
    answer_comparison_html,
    find_run_manifest,
    find_run_summary,
    resolve_view_defaults,
    select_repeat,
    write_answer_comparison,
)
from agentic_eval.content.document import (
    parse_answer_document,
    table_cell_coverage,
)
from agentic_eval.content.evidence import (
    _call_signature,
    _evidence_tier,
    build_evidence_ledger,
)
from agentic_eval.content.metrics import calculate_content_metrics
from agentic_eval.content.numeric import _comparator, _satisfies
from agentic_eval.content.oracles import evaluate_expected_answers
from agentic_eval.content.pipeline import ContentEvaluator, evaluate_runs_file
from agentic_eval.content.prompts import (
    EXTRACT_PROMPT,
)
from agentic_eval.content.report import (
    WALKTHROUGH_LEGEND,
    content_comparison_markdown,
    content_walkthrough_markdown,
    write_content_walkthrough,
    write_evidence_review_packets,
)
from agentic_eval.content.verdicts import (
    CLAIM_STANCES,
    CORRECTNESS_VERDICTS,
    FACT_VERDICTS,
    EVIDENCE_RESOLUTIONS,
    LOGIC_VERDICTS,
    MUST_HAVE_VERDICTS,
    NUMERIC_SUPPORT_VERDICTS,
    TRACE_VERDICTS,
)
from agentic_eval.content.verify import _normalize_fact_results

# Re-exported: callers predating the package split import these from here.
from agentic_eval.common.coerce import (
    _evidence_float,
    _resolve_path,
    _safe_float,
)

__all__ = [
    "answer_comparison_html",
    "find_run_manifest",
    "find_run_summary",
    "resolve_view_defaults",
    "select_repeat",
    "write_answer_comparison",
    "CLAIM_STANCES",
    "CORRECTNESS_VERDICTS",
    "ContentEvaluator",
    "EXTRACT_PROMPT",
    "FACT_VERDICTS",
    "EVIDENCE_RESOLUTIONS",
    "LOGIC_VERDICTS",
    "MUST_HAVE_VERDICTS",
    "NUMERIC_SUPPORT_VERDICTS",
    "TRACE_VERDICTS",
    "WALKTHROUGH_LEGEND",
    "_MACRO_METRICS",
    "_call_signature",
    "_comparator",
    "_evidence_float",
    "_evidence_tier",
    "_normalize_claim",
    "_normalize_fact_results",
    "_resolve_path",
    "_safe_float",
    "_satisfies",
    "aggregate_content_evaluations",
    "build_evidence_ledger",
    "calculate_content_metrics",
    "content_comparison_markdown",
    "content_walkthrough_markdown",
    "evaluate_expected_answers",
    "evaluate_runs_file",
    "parse_answer_document",
    "read_jsonl",
    "table_cell_coverage",
    "write_content_walkthrough",
    "write_evidence_review_packets",
]
