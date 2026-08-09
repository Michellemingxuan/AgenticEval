"""Everything a human reads: the HTML viewer, the markdown, the scorecards.

Kept apart from the code that produces the numbers. `content/` decides what is
true about an answer; this decides how that is shown. The split matters because
the two change for different reasons — a rendering fix should never be able to
alter a verdict.

    page.py         the side-by-side HTML viewer, with repeat tabs
    markdown.py     content scorecard, per-answer walkthrough, review packets
    run_summary.py  the run-level comparison table and the blinded review CSV
"""
from __future__ import annotations

from agentic_eval.render.markdown import *  # noqa: F401,F403
from agentic_eval.render.markers import *  # noqa: F401,F403
from agentic_eval.render.page import *  # noqa: F401,F403
from agentic_eval.render.run_summary import *  # noqa: F401,F403

__all__ = [
    "ELIGIBILITY_MARKER",
    "GROUNDING_MARKER",
    "WALKTHROUGH_LEGEND",
    "answer_comparison_html",
    "comparison_markdown",
    "content_comparison_markdown",
    "content_walkthrough_markdown",
    "find_run_manifest",
    "find_run_summary",
    "resolve_view_defaults",
    "select_repeat",
    "write_answer_comparison",
    "write_blind_review",
    "write_content_walkthrough",
    "write_evidence_review_packets",
]
