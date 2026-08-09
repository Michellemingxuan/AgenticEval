"""The marker vocabulary, owned by neither renderer.

Both the HTML page and the markdown walkthrough show the same verdicts, and
they must agree about what a symbol means. Defining them here rather than in
one renderer and importing across removes the coupling — and says plainly that
the vocabulary belongs to the evaluation, not to a view of it.

Counted straight off `grounding_kind` and `eligible`, so a marker on the page
and the rate beside it cannot drift.
"""
from __future__ import annotations

#: What the claim rests on.
GROUNDING_MARKER = {"factual": "◆", "report": "◇", "none": "○"}

#: Whether the route that produced the claim answers the question asked.
ELIGIBILITY_MARKER = {
    "yes": "✓", "no": "✗", "unavailable": "?", "not_applicable": "–",
}
