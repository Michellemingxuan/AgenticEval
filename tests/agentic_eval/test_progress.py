"""The progress page: right denominators, and never able to fail a run."""
from __future__ import annotations

from agentic_eval.render import progress


def _rows(n, **fixed):
    return [
        {"system": "previous", "case_id": "c1", "run_index": 1, "name": "q1",
         "question_set": "series_b", "outcome": "ok", **fixed}
        for _ in range(n)
    ]


_PLAN = {
    "questions": 3, "cases": 2, "repeats": 3, "systems": 2,
    "set_sizes": {"series_b": 2, "series_c": 1},
    "expected_records": 36,
}


def test_progress_is_measured_against_the_run_not_the_config():
    """A scoped run asks fewer questions than the file lists.

    Measuring against the config's totals would leave a complete run showing
    40% — the reading that makes someone kill a finished job.
    """
    state = progress.summarize(_rows(18), plan=_PLAN)
    assert state["done"] == 18 and state["expected"] == 36
    assert state["percent"] == 50.0


def test_each_set_gets_its_own_denominator():
    """series_b has more questions than series_c; one total would misreport."""
    html = progress.render(progress.summarize([
        *[dict(r, question_set="series_b") for r in _rows(4)],
        *[dict(r, question_set="series_c", name="c1") for r in _rows(2)],
    ], plan=_PLAN))
    assert "/24" in html            # series_b: 2 q x 2 sys x 3 rep x 2 cases
    assert "/12" in html            # series_c: 1 q x ...


def test_a_padded_case_id_is_shown_quoted():
    """`11854808010 ` and `11854808010` print identically and are not the same."""
    html = progress.render(progress.summarize(
        [dict(r, case_id="11854808010 ") for r in _rows(1)], plan=_PLAN,
    ))
    assert "&#x27;11854808010 &#x27;" in html


def test_replayed_records_are_disclosed_on_the_page():
    state = progress.summarize(_rows(4, evaluator_attempts=2), plan=_PLAN)
    assert state["retried_records"] == 4
    assert "replayed pass" in progress.render(state)


def test_outcomes_other_than_ok_are_counted_and_marked():
    rows = _rows(3) + [dict(r, outcome="timeout") for r in _rows(2)]
    state = progress.summarize(rows, plan=_PLAN)
    assert state["outcomes"] == {"ok": 3, "timeout": 2}
    assert 'class="bad">timeout' in progress.render(state)


def test_an_empty_run_renders_rather_than_dividing_by_zero():
    html = progress.render(progress.summarize([], plan={}))
    assert "0%" in html and "nothing recorded yet" in html


def test_writing_never_raises(tmp_path):
    """It runs inside the record-writing lock; a raise would abort the run."""
    progress.write(tmp_path / "nested" / "p.html", _rows(2), plan=_PLAN)
    assert (tmp_path / "nested" / "p.html").is_file()

    class _Hostile:
        """Unwritable path and unserialisable plan, at once."""
        parent = property(lambda self: (_ for _ in ()).throw(OSError("nope")))

    progress.write(_Hostile(), _rows(1), plan=_PLAN)      # must not raise


def test_eta_is_extrapolated_from_what_has_finished():
    state = progress.summarize(_rows(9), plan=_PLAN, started_at=0.0)
    # 9 of 36 done => three times the elapsed remains
    assert state["eta_s"] is not None
    assert state["eta_s"] > state["elapsed_s"] * 2.9


def test_the_grid_puts_cases_across_and_questions_down():
    rows = [
        {"system": s, "case_id": c, "run_index": r, "name": q, "outcome": "ok"}
        for c in ("366132845011", "11854808010 ")
        for r in (1, 2)
        for q in ("a1", "b2")
        for s in ("previous", "current")
        # case 118 only got through repeat 1
        if not (c == "11854808010 " and r == 2)
    ]
    text = progress.grid(rows, plan={"systems": 2, "repeats": 2})
    lines = text.splitlines()
    assert "366132845011" in lines[0] and "11854808010 " in lines[0]
    assert lines[1].startswith("a1") and lines[2].startswith("b2")
    assert "2/2" in lines[1]        # case 366 complete
    assert "1/2" in lines[1]        # case 118 half way


def test_a_repeat_counts_only_when_every_system_answered_it():
    """Half a repeat is not a repeat — the count must not round up."""
    rows = [{"system": "previous", "case_id": "c", "run_index": 1,
             "name": "a1", "outcome": "ok"}]
    text = progress.grid(rows, plan={"systems": 2, "repeats": 1})
    assert "0/1" in text            # one of two systems in
    rows.append({"system": "current", "case_id": "c", "run_index": 1,
                 "name": "a1", "outcome": "ok"})
    assert "1/1" in progress.grid(rows, plan={"systems": 2, "repeats": 1})


def test_the_grid_follows_the_configured_question_order():
    rows = [{"system": "s", "case_id": "c", "run_index": 1, "name": n,
             "outcome": "ok"} for n in ("c1", "a1")]
    text = progress.grid(rows, plan={"systems": 1, "repeats": 1},
                         order=["a1", "c1"])
    lines = text.splitlines()
    assert lines[1].startswith("a1") and lines[2].startswith("c1")


def test_the_grid_says_so_when_nothing_has_landed():
    assert "nothing recorded yet" in progress.grid([], plan={})


def _grid_rows(outcomes, case="366132845011", name="b3", repeats=1):
    return [{"system": s, "case_id": case, "run_index": 1, "name": name,
             "outcome": o} for s, o in outcomes]


def test_a_failed_cell_is_red_and_says_how_many():
    rows = _grid_rows([("previous", "timeout"), ("current", "ok")])
    plan = {"systems": 2, "repeats": 1, "question_order": ["b3"]}
    text = progress.grid(rows, plan=plan, colourise=True)
    assert "\033[31m" in text and "\033[0m" in text
    assert "✗1" in text


def test_a_clean_cell_carries_no_escape_codes():
    rows = _grid_rows([("previous", "ok"), ("current", "ok")])
    plan = {"systems": 2, "repeats": 1, "question_order": ["b3"]}
    assert "\033[" not in progress.grid(rows, plan=plan, colourise=True)


def test_colour_never_disturbs_the_columns():
    """Padding happens before colouring; escape codes have no width."""
    import re
    rows = [{"system": s, "case_id": c, "run_index": 1, "name": q, "outcome": o}
            for c in ("caseA", "caseB")
            for q, o in (("b3", "timeout"), ("b4", "ok"))
            for s in ("previous", "current")]
    plan = {"systems": 2, "repeats": 1, "question_order": ["b3", "b4"]}
    stripped = re.sub(r"\033\[\d+m", "",
                      progress.grid(rows, plan=plan, colourise=True))
    plain = progress.grid(rows, plan=plan, colourise=False)
    assert [l.rstrip() for l in stripped.splitlines()] == \
           [l.rstrip() for l in plain.splitlines()]


def test_a_case_id_with_a_trailing_space_prints_plainly():
    """The padding is real, but quoting it is noise on a progress display."""
    assert progress.case_labels(["11854808010 ", "366132845011"]) == {
        "11854808010 ": "11854808010", "366132845011": "366132845011",
    }


def test_two_cases_differing_only_by_whitespace_keep_their_quotes():
    """Stripping would give both rows the same label and hide which is which."""
    labels = progress.case_labels(["11854808010 ", "11854808010"])
    assert labels["11854808010 "] == "'11854808010 '"
    assert labels["11854808010"] == "'11854808010'"


def test_rows_follow_series_order_then_position_in_the_series():
    """Not arrival order, which reshuffles as workers finish."""
    order = ["a1", "b2", "b3", "c1"]
    rows = [{"system": "s", "case_id": "c", "run_index": 1, "name": n,
             "outcome": "ok"} for n in reversed(order)]
    plan = {"systems": 1, "repeats": 1, "question_order": order,
            "set_of": {"a1": "series_a", "b2": "series_b", "b3": "series_b",
                       "c1": "series_c"}}
    lines = [l.strip() for l in progress.grid(rows, plan=plan, colourise=False).splitlines()]
    assert lines[1] == "series_a"
    assert lines[2].startswith("a1")
    assert lines[3] == "series_b"
    assert lines[4].startswith("b2") and lines[5].startswith("b3")
    assert lines[6] == "series_c"


def test_the_plan_is_recoverable_from_a_manifest():
    """A finished run predating this page has no plan file beside it.

    Without a fallback every denominator reads "?", which is the one thing a
    progress page exists to supply.
    """
    plan = progress.plan_from_manifest({
        "question_count": 4, "repeats": 2, "mode": "stateful",
        "cases": ["366132845011", "11854808010 "],
        "systems": {"previous": {}, "current": {}},
        "question_sets": {"series_a": ["a1"], "series_b": ["b2", "b3"],
                          "series_c": ["c1"]},
    })
    assert plan["expected_records"] == 4 * 2 * 2 * 2
    assert plan["set_sizes"] == {"series_a": 1, "series_b": 2, "series_c": 1}
    assert plan["question_order"] == ["a1", "b2", "b3", "c1"]
    assert plan["set_of"]["b3"] == "series_b"
    assert plan["case_ids"] == ["366132845011", "11854808010 "]


def test_a_both_mode_run_expects_two_passes():
    plan = progress.plan_from_manifest({
        "question_count": 2, "repeats": 1, "mode": "both", "cases": ["c"],
        "systems": {"a": {}, "b": {}}, "question_sets": {"s": ["q1", "q2"]},
    })
    assert plan["expected_records"] == 2 * 1 * 1 * 2 * 2


def test_an_empty_manifest_does_not_invent_a_denominator():
    plan = progress.plan_from_manifest({})
    assert plan["expected_records"] == 0
    assert progress.summarize([], plan=plan)["expected"] == 0
