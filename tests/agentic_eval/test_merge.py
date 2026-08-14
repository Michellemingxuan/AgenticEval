"""Joining runs that covered different cases.

`run` has no resume, so a long pass is split by CASE and the pieces joined.
Concatenating by hand works until it doesn't, and both failures are silent:
a duplicated case doubles every count, and two runs of different experiments
join without complaint.
"""
from __future__ import annotations

import pytest

from agentic_eval.merge import merge


def _record(case, name="q1", system="previous", run_index=1):
    return {
        "system": system, "mode": "stateful", "case_id": case,
        "question_set": "series_a", "name": name, "run_index": run_index,
        "outcome": "ok", "final_answer": "a",
    }


_MANIFEST = {"baseline": "previous", "candidate": "current", "mode": "stateful"}


def test_two_cases_join_into_one_run():
    a = ([_record("366"), _record("366", system="current")], _MANIFEST)
    b = ([_record("118"), _record("118", system="current")], _MANIFEST)

    records, manifest = merge([a, b])

    assert len(records) == 4
    assert manifest["cases"] == ["118", "366"]
    assert manifest["merged_from"] == 2
    # The comparison's identity is carried through unchanged.
    assert manifest["baseline"] == "previous" and manifest["candidate"] == "current"


def test_a_repeated_case_is_refused_rather_than_counted_twice():
    """The silent one. Appending re-run answers doubles every count over them,
    and consistency then compares the run against itself."""
    same = ([_record("366")], _MANIFEST)

    with pytest.raises(ValueError, match="twice"):
        merge([same, same])


def test_the_error_names_the_answer_and_both_sources():
    with pytest.raises(ValueError) as exc:
        merge([([_record("366")], _MANIFEST), ([_record("366")], _MANIFEST)])
    message = str(exc.value)
    assert "previous" in message and "q1" in message and "'366'" in message
    assert "source 1" in message and "source 2" in message


def test_runs_of_different_experiments_are_refused():
    a = ([_record("366")], _MANIFEST)
    b = ([_record("118")], {**_MANIFEST, "candidate": "something_else"})

    with pytest.raises(ValueError, match="disagree about candidate"):
        merge([a, b])


def test_swapped_baseline_and_candidate_is_refused():
    """Same two systems, opposite roles — every delta would flip mid-file."""
    a = ([_record("366")], _MANIFEST)
    b = ([_record("118")], {"baseline": "current", "candidate": "previous",
                            "mode": "stateful"})

    with pytest.raises(ValueError, match="disagree about baseline"):
        merge([a, b])


def test_the_same_case_may_appear_with_different_questions():
    """Splitting by QUESTION rather than case is legitimate too."""
    a = ([_record("366", name="a1")], _MANIFEST)
    b = ([_record("366", name="b2")], _MANIFEST)

    records, manifest = merge([a, b])

    assert len(records) == 2 and manifest["cases"] == ["366"]


def test_merging_nothing_is_an_error():
    with pytest.raises(ValueError, match="nothing to merge"):
        merge([])


def test_a_run_that_died_partway_is_still_readable(tmp_path):
    """The manifest is written BEFORE the first turn, not after the last.

    A crashed run used to leave answers with no manifest beside them, so
    nothing downstream could say which system was the baseline: `rescore`
    refused, `merge` could not check the two runs matched, and the viewer
    would have inferred the roles backwards.
    """
    import json
    from agentic_eval.layout import RunLayout
    from agentic_eval.merge import read_run

    # What a killed run leaves: a manifest with n_records null, and whatever
    # answers had been appended.
    layout = RunLayout(tmp_path).ensure()
    layout.manifest.write_text(json.dumps({
        "baseline": "previous", "candidate": "current", "mode": "stateful",
        "n_records": None,
    }), encoding="utf-8")
    layout.runs.write_text(json.dumps(_record("366")) + "\n", encoding="utf-8")

    records, manifest = read_run(tmp_path)

    assert len(records) == 1
    assert manifest["baseline"] == "previous"
    # And it can be joined with the run that finishes the job.
    merged, joined = merge([(records, manifest), ([_record("118")], manifest)])
    assert len(merged) == 2 and joined["cases"] == ["118", "366"]


def test_an_unfinished_run_says_so():
    """`n_records: null` distinguishes "died" from "ran and found nothing"."""
    unfinished = {"baseline": "a", "candidate": "b", "mode": "cold",
                  "n_records": None}
    finished = {**unfinished, "n_records": 32}
    assert unfinished["n_records"] is None
    assert finished["n_records"] == 32
