"""The question set decides what order the page reads in.

`manifest.json` records `question_sets` — set name -> question names, both in
config order — at run time, and it survives `select` and `merge`. Everything
the page could derive from the answers themselves is wrong in some real case:

  * NAME order is right only by luck of naming;
  * `sequence_position` restarts at 1 per set and is RENUMBERED by a subset
    run, so re-asking one question reorders the set it came from.

The fallback still has to work, because a run captured before the manifest
carried a plan has nothing else.
"""
from __future__ import annotations

import re

from agentic_eval.render.page import answer_comparison_html


def _rows(specs):
    return [
        {
            "system": system, "mode": "stateful", "case_id": "366",
            "question_set": question_set, "name": name, "run_index": 1,
            "sequence_position": position, "question": name, "answer": "a",
            "claims": [], "metrics": {}, "expected_answer_results": [],
            "must_have_results": [], "fact_results": [], "evidence_ledger": [],
            "tools": [],
        }
        for question_set, name, position in specs
        for system in ("previous", "current")
    ]


def _order(evaluations, question_sets=None):
    html = answer_comparison_html(
        evaluations, baseline="previous", candidate="current",
        question_sets=question_sets,
    )
    seen = []
    for heading in re.findall(r"<h3[^>]*>([^<]{0,60})</h3>", html):
        if heading not in seen and not heading.startswith("All question"):
            seen.append(heading)
    return seen


def test_the_opening_question_stays_first_even_when_its_name_sorts_last():
    """`q0_off_domain_rejection` opens series A and sorts after `a1`.

    It carries no position either — the guardrail probe is asked cold — so
    nothing in the answers puts it back where it belongs.
    """
    rows = _rows([("series_a", "q0_off_domain_rejection", None),
                  ("series_a", "a1_payment_returns", None)])

    assert _order(rows) == ["a1_payment_returns", "q0_off_domain_rejection"]
    assert _order(rows, {"series_a": ["q0_off_domain_rejection",
                                      "a1_payment_returns"]}) == [
        "q0_off_domain_rejection", "a1_payment_returns"]


def test_a_question_re_asked_alone_does_not_reorder_its_set():
    """Re-run b3 by itself and it comes back as position 1 of its own session,
    ahead of the b2 it is a follow-up to."""
    rows = _rows([("series_b", "b2_tsr", 2), ("series_b", "b3_bureau", 1)])

    assert _order(rows) == ["b3_bureau", "b2_tsr"]
    assert _order(rows, {"series_b": ["b2_tsr", "b3_bureau"]}) == [
        "b2_tsr", "b3_bureau"]


def test_sets_follow_the_config_not_the_alphabet():
    rows = _rows([("overview", "o1", 1), ("deep_dive", "d1", 1)])

    assert _order(rows) == ["d1", "o1"]
    assert _order(rows, {"overview": ["o1"], "deep_dive": ["d1"]}) == ["o1", "d1"]


def test_a_question_the_plan_does_not_name_renders_after_its_set():
    """Planned precedes unplanned, so a question added after the run was
    captured lands at the end of its set rather than at the front."""
    rows = _rows([("series_a", "a1", 1), ("series_a", "a2_added_later", 2)])

    assert _order(rows, {"series_a": ["a1"]}) == ["a1", "a2_added_later"]


def test_a_plan_naming_absent_questions_grows_no_empty_sections():
    """`select --exclude-question` keeps the whole plan in the manifest. The
    page must order BY it, not iterate it."""
    rows = _rows([("series_a", "a1", 1)])

    assert _order(rows, {"series_a": ["a1", "a2_dropped", "a3_dropped"],
                         "series_b": ["b2_dropped"]}) == ["a1"]


def test_a_run_with_no_plan_still_orders_stably():
    """The pre-manifest fallback: set name, then position, then natural name."""
    rows = _rows([("series_b", "b2", 1), ("series_a", "a2", 2),
                  ("series_a", "a10", 3), ("series_a", "a1", 1)])

    assert _order(rows, None) == ["a1", "a2", "a10", "b2"]
    assert _order(rows, {}) == ["a1", "a2", "a10", "b2"]


# --- the other two views of the same run ------------------------------------
#
# A reader moves between the page, the scorecard and the walkthrough by
# question. Three different orders makes that guesswork, and each view looks
# right on its own.

_PLAN = {"series_a": ["q0_off_domain", "a1_returns"], "series_b": ["b2_tsr"]}


def test_the_scorecard_follows_the_plan_not_the_alphabet():
    from agentic_eval.render.markdown import content_comparison_markdown

    summary = {"groups": [
        {"system": system, "mode": "stateful", "name": name,
         "n_runs": 1, "metric_distributions": {}}
        for name in ("b2_tsr", "a1_returns", "q0_off_domain")
        for system in ("previous", "current")
    ]}

    def order(question_sets):
        table = content_comparison_markdown(
            summary, baseline="previous", candidate="current",
            question_sets=question_sets)
        return [name for name in ("q0_off_domain", "a1_returns", "b2_tsr")
                if name in table]

    rendered = content_comparison_markdown(
        summary, baseline="previous", candidate="current", question_sets=_PLAN)
    positions = {name: rendered.index(name)
                 for name in ("q0_off_domain", "a1_returns", "b2_tsr")}
    assert positions["q0_off_domain"] < positions["a1_returns"] < positions["b2_tsr"]

    # Without a plan it stays alphabetical, which is what it always did.
    plain = content_comparison_markdown(
        summary, baseline="previous", candidate="current")
    assert plain.index("a1_returns") < plain.index("q0_off_domain")
    assert order(_PLAN)  # the names really are in the table


def test_the_walkthrough_follows_the_plan_not_the_file(tmp_path):
    """File order is ARRIVAL order — a merged run is trimmed-then-fresh, so the
    re-run question's sections landed after everything that was kept."""
    from agentic_eval.layout import RunLayout
    from agentic_eval.render.markdown import write_content_walkthrough

    layout = RunLayout(tmp_path).ensure()
    # Deliberately the order a splice leaves behind: b2 first, a1 last.
    evaluations = _rows([("series_b", "b2_tsr", 1),
                         ("series_a", "a1_returns", 2),
                         ("series_a", "q0_off_domain", 1)])

    write_content_walkthrough(evaluations, layout=layout, question_sets=_PLAN)
    text = layout.walkthrough.read_text(encoding="utf-8")

    assert (text.index("q0_off_domain") < text.index("a1_returns")
            < text.index("b2_tsr"))


def test_the_walkthrough_keeps_the_baseline_ahead_of_the_candidate(tmp_path):
    """Sorting by system name would put "current" first and reverse the pairing
    every other view reads baseline-then-candidate."""
    from agentic_eval.layout import RunLayout
    from agentic_eval.render.markdown import write_content_walkthrough

    layout = RunLayout(tmp_path).ensure()
    evaluations = _rows([("series_a", "a1_returns", 1)])
    assert [row["system"] for row in evaluations] == ["previous", "current"]

    write_content_walkthrough(evaluations, layout=layout, question_sets=_PLAN)
    text = layout.walkthrough.read_text(encoding="utf-8")

    assert text.index("previous") < text.index("current")


def test_every_view_reads_the_plan_from_the_same_field(tmp_path):
    """The page and the walkthrough both default to the run's manifest, so a
    caller cannot wire one and forget the other."""
    import json

    from agentic_eval.layout import RunLayout
    from agentic_eval.render.markdown import write_content_walkthrough
    from agentic_eval.render.page import write_answer_comparison

    layout = RunLayout(tmp_path).ensure()
    layout.manifest.write_text(json.dumps({
        "baseline": "previous", "candidate": "current", "mode": "stateful",
        "question_sets": _PLAN,
    }), encoding="utf-8")
    evaluations = _rows([("series_a", "a1_returns", 2),
                         ("series_a", "q0_off_domain", 1)])

    write_content_walkthrough(evaluations, layout=layout)
    write_answer_comparison(evaluations, layout=layout,
                            baseline="previous", candidate="current")

    for path in (layout.walkthrough, layout.answer_comparison):
        text = path.read_text(encoding="utf-8")
        assert text.index("q0_off_domain") < text.index("a1_returns"), path.name
