"""Oracles must answer for the case under test, not a default one."""
from __future__ import annotations

import json

from agentic_eval.content.oracles import _bind_case, evaluate_expected_answers


def test_the_case_is_substituted_into_the_command():
    bound, error = _bind_case(
        ["python3", "case_facts.py", "--case", "{case_id}", "--fact", "x"],
        "11854808010 ",
    )
    assert error is None
    # Exactly as recorded, trailing space and all — it is the folder name.
    assert bound == ["python3", "case_facts.py", "--case", "11854808010 ",
                     "--fact", "x"]


def test_a_command_that_wants_no_case_is_left_alone():
    """`python3 -c "print('true')"` takes no case; appending one breaks it."""
    command = ["python3", "-c", "print('true')"]
    assert _bind_case(command, "c1") == (command, None)


def test_a_case_blind_run_is_refused_not_guessed():
    """The worst failure mode: it runs, returns a number, and the number is
    for a different customer.

    Measured on a two-case run — the rubric never passed `--case`, so every
    answer about `11854808010 ` (1 payment return) was graded against
    `366132845011` (0), and three false answers scored correct.
    """
    _bound, error = _bind_case(["x", "--case", "{case_id}"], None)
    assert error is not None and "no case" in error


def test_the_oracle_result_reports_the_binding_failure(tmp_path):
    rubric = {"expected_answers": [{
        "id": "needs_case", "description": "d",
        "command": ["python3", "-c", "print(1)", "{case_id}"],
    }]}
    results = evaluate_expected_answers(
        [], rubric, answer="anything", cwd=str(tmp_path), case_id=None,
    )
    assert results and results[0]["verdict"] != "pass"
    assert "case" in json.dumps(results[0]).lower()


def _count_item(expected_value, gate=None):
    item = {
        "id": "payment_return_count", "description": "Number of returns.",
        "value": expected_value, "tolerance": 0,
        "accept_patterns": [r"\b(no|zero|none|not any)\b[^.]{0,40}\breturn"],
    }
    if gate is not None:
        item["accept_when_expected"] = gate
    return {"expected_answers": [item]}


ZERO_IN_WORDS = "The customer had no returned payments."


def test_words_for_zero_pass_when_zero_is_the_truth():
    """A count of zero is stated in words, never as the digit."""
    results = evaluate_expected_answers(
        [], _count_item(0, gate=0), answer=ZERO_IN_WORDS,
    )
    assert results[0]["verdict"] == "pass"


def test_words_for_zero_fail_when_the_truth_is_not_zero():
    """The pattern encodes what the ANSWER should say, which depends on what
    the truth IS.

    On the second case the count is 1, and "no returned payments" — plainly
    wrong — matched the pattern and passed, so the question reported both
    systems correct on a case they both got wrong.
    """
    results = evaluate_expected_answers(
        [], _count_item(1, gate=0), answer=ZERO_IN_WORDS,
    )
    assert results[0]["verdict"] == "fail"


def test_an_ungated_pattern_still_applies():
    """Existing rubrics without the gate keep their behaviour."""
    results = evaluate_expected_answers(
        [], _count_item(0), answer=ZERO_IN_WORDS,
    )
    assert results[0]["verdict"] == "pass"


def test_rejudging_one_question_keeps_the_others(tmp_path, monkeypatch):
    """`--question a1` without `--resume` truncated the whole file.

    Eight b2 evaluations went with it — judge calls already spent, and not
    recoverable from anything on disk.
    """
    import agentic_eval.content.pipeline as pipeline

    run = tmp_path / "run"
    (run / "content").mkdir(parents=True)
    out = run / "content" / "evaluations.jsonl"
    kept = {"system": "old", "mode": "cold", "name": "b2", "run_index": 1,
            "case_id": "c1", "metrics": {}}
    out.write_text(json.dumps(kept) + "\n", encoding="utf-8")

    records = [{
        "system": "old", "mode": "cold", "name": "a1", "run_index": 1,
        "case_id": "c1", "outcome": "ok", "final_answer": "an answer",
        "evidence": [],
    }]

    class _Spy:
        def evaluate(self, record, rubric, prior_turns):
            return {"system": record["system"], "mode": record["mode"],
                    "name": record["name"], "run_index": record["run_index"],
                    "case_id": record["case_id"], "metrics": {}}

    monkeypatch.setattr(pipeline, "ContentEvaluator", lambda *a, **k: _Spy())
    pipeline.evaluate_runs_file(
        config={"expected_repeats": 1}, records=records, output_dir=run,
        baseline="old", candidate="new", rubric_by_name={},
        questions=["a1"],
    )
    names = sorted(
        json.loads(line)["name"] for line in out.read_text().splitlines() if line.strip()
    )
    assert names == ["a1", "b2"], "the untouched question must survive"


_ANSWER_WITH_ONE_RETURN = (
    "no prior curated reports — answer is from live specialist analysis only. "
    "the customer had one returned (failed) payment in their history: "
    "$105,818.60 on 2025-04-28, returned for insufficient funds. "
    "only one payment return was found across all payment records. "
    "no other payment returns are recorded for this customer."
)

_BOOL_ITEM = {
    "affirmative_patterns": [r"\b(had|has|have|were|was|are|is)\b[^.]{0,60}\breturn(ed|s)?\b"],
    "negative_patterns": [r"\b(no|zero|none|not any)\b[^.]{0,40}\breturn"],
}


def test_no_other_x_is_not_a_denial_of_x():
    """"No OTHER payment returns are recorded" is only sayable when one exists.

    Read as a denial it inverted a correct answer: the same answer opened
    "the customer had one returned payment of $105,818.60" and was scored as
    stating False against a ground truth of True.
    """
    from agentic_eval.content.oracles import _boolean_answer_check

    result = _boolean_answer_check(_BOOL_ITEM, True, _ANSWER_WITH_ONE_RETURN)
    assert result["verdict"] == "pass", result["reason"]


def test_a_real_denial_still_wins_over_a_stray_affirmative():
    """The negation-precedence rule exists because "had NO returned payments"
    matches an affirmative pattern inside the sentence that denies it."""
    from agentic_eval.content.oracles import _boolean_answer_check

    denial = "the customer had no returned payments; all payments were successful."
    assert _boolean_answer_check(_BOOL_ITEM, False, denial)["verdict"] == "pass"
    assert _boolean_answer_check(_BOOL_ITEM, True, denial)["verdict"] == "fail"


def test_a_count_spelled_as_a_word_is_found():
    """"one returned payment" states the count exactly; a digits-only scan
    reported "no such figure in the answer" against a truth of 1."""
    rubric = {"expected_answers": [{
        "id": "payment_return_count", "description": "Number of returns.",
        "value": 1, "tolerance": 0,
    }]}
    result = evaluate_expected_answers(
        [], rubric, answer=_ANSWER_WITH_ONE_RETURN,
    )[0]
    assert result["verdict"] == "pass"
    assert "word" in result["reason"]


def test_a_word_for_a_different_number_does_not_pass():
    from agentic_eval.content.oracles import _spelled_number_in

    assert _spelled_number_in("had one returned payment", 2) is None
    # Beyond twelve prose uses digits; a word-match there would be reaching.
    assert _spelled_number_in("had twenty returns", 20) is None


def test_every_shipped_oracle_binds_the_case_it_asks_about():
    """The guard `_bind_case` cannot provide.

    It refuses a command whose `{case_id}` cannot be filled — but a rubric that
    never wrote the placeholder asks for nothing, so there is nothing to refuse
    and the oracle answers about whatever case it defaults to.

    Measured on a real run: `b1_case_overview` shipped without `--case`, so an
    answer about a customer with 2 SBS cards and 24 returned payments was
    graded against another case's 1 and 0. The card count read as wrong, and
    the return count read as RIGHT — the answer said "no consumer cards", which
    put a 0 in it, and 0 was the other case's return count.

    Only `case_facts.py` is checked: an oracle can legitimately take no case,
    like the off-domain probe's `print('true')`.
    """
    import glob

    import yaml

    blind = []
    for path in sorted(glob.glob("experiments/questions/*.yaml")):
        for question in (yaml.safe_load(open(path)) or {}).get("questions") or []:
            for item in (question.get("evaluation") or {}).get(
                "expected_answers"
            ) or []:
                command = [str(part) for part in item.get("command") or []]
                if not any("case_facts" in part for part in command):
                    continue
                if "{case_id}" not in " ".join(command):
                    blind.append(f"{path}::{question['name']}::{item['id']}")

    assert not blind, (
        "these oracles read case data but never receive the case, so they "
        f"answer about whichever one they default to: {blind}"
    )


def test_the_oracle_script_refuses_to_guess_a_case():
    """`--case` is required, with no default.

    A default is the worst possible behaviour: the script runs, prints a
    number, and the number is for a different customer. Exiting non-zero makes
    the evaluator report the oracle unavailable and say why, which is
    recoverable; a plausible wrong number is not.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path("experiments/oracles/case_facts.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--fact", "commercial_card_count"],
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode != 0
    assert "--case" in result.stderr
    assert not result.stdout.strip(), "a refusal must not also print a value"
