"""One order for every report: the order the questions were asked.

The page, the scorecard and the walkthrough are three views of one run, and a
reader moves between them by question. When they disagree about order that
movement stops working — worse, silently, because each view is internally
consistent and looks right on its own.

`manifest.json` records `question_sets` — set name -> question names, both in
config order — at run time, and it survives `select` and `merge`. It is the
only source that matches what was asked. Everything derivable from the answers
themselves is wrong in some real case:

  * NAME order is right only by luck of naming. `q0_off_domain_rejection`
    opens series A and sorts after `a1`.
  * `sequence_position` restarts at 1 in every set AND is renumbered by a
    subset run, so re-asking a follow-up on its own floats it above the
    question it follows.
  * FILE order is arrival order. A merged run is trimmed-then-fresh, so a
    re-run question lands after everything that was kept.

So the plan decides, and what the plan does not name falls back — planned
always first, since a question the plan omits is one the run did not intend to
ask in that position.
"""
from __future__ import annotations

import re
from typing import Any, Callable


def natural_key(text: str) -> tuple:
    """Sort `a2` before `a10`, and `series_a` before `series_b`.

    Digit runs compare as numbers so question 10 does not land between 1 and 2.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", str(text)) if part
    )


def normalise_plan(question_sets: dict[str, Any] | None) -> dict[str, list[str]]:
    """The manifest's `question_sets`, with every name coerced to `str`.

    JSON round-trips keys as strings but the runner builds this from config
    objects, so a plan can arrive either way.
    """
    return {
        str(name): [str(question) for question in questions or ()]
        for name, questions in (question_sets or {}).items()
    }


def planned_order(question_sets: dict[str, Any] | None) -> list[str]:
    """Every planned question, flattened into the order it was asked."""
    return [
        question for questions in normalise_plan(question_sets).values()
        for question in questions
    ]


def question_sort_key(
    question_sets: dict[str, Any] | None = None, *,
    rows: list[dict[str, Any]] | None = None,
) -> Callable[[Any], tuple]:
    """A sort key over question NAMES, for the reports with no set sections.

    `rows` supplies the fallback: the earliest `sequence_position` a question
    was seen at. Reports built from `summary.json` have no positions to give —
    its groups carry `system`, `mode` and `name` and nothing else — so there
    the fallback is the natural name, which is what those reports already used.
    """
    rank = {name: index for index, name in enumerate(planned_order(question_sets))}
    positions: dict[str, float] = {}
    for row in rows or ():
        position = row.get("sequence_position")
        if position is not None:
            name = str(row.get("name"))
            positions[name] = min(positions.get(name, float("inf")), float(position))

    def key(name: Any) -> tuple:
        name = str(name)
        if name in rank:
            return (0, rank[name], 0.0, ())
        return (1, 0, positions.get(name, float("inf")), natural_key(name))

    return key
