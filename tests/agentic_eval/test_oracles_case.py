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


def test_a_failing_oracle_reports_the_exception_not_the_traceback_header():
    """The first 200 characters of a traceback are the least useful 200.

    Real report from a private environment, in full:

        Oracle command exited 1: Traceback (most recent call last):
          File ".../case_facts.py", line 291, in <module>
            raise SystemExit(main())

    The exception — which named the missing file — fell off the end. It took
    three rounds of guessing before anyone saw the actual cause.
    """
    from agentic_eval.content.oracles import _failure_line

    traceback = (
        "Traceback (most recent call last):\n"
        '  File "/Users/x/_proj/AgenticEval-main/experiments/oracles/'
        'case_facts.py", line 291, in <module>\n'
        "    raise SystemExit(main())\n"
        '  File "case_facts.py", line 122, in _cards\n'
        '    return _rows(_table(case, "crossbu_cards.csv"))\n'
        "FileNotFoundError: case '1059922019' has none of: crossbu_cards.csv"
    )

    line = _failure_line(traceback)

    assert "FileNotFoundError" in line and "crossbu_cards.csv" in line
    assert "Traceback (most recent call last)" not in line


def test_a_terse_exception_keeps_the_frame_that_locates_it():
    """"KeyError: 'Balance'" says nothing about where."""
    from agentic_eval.content.oracles import _failure_line

    line = _failure_line(
        "Traceback:\n  File 'x.py', line 9, in commercial_card_balance\n"
        "KeyError: 'Balance'"
    )
    assert "KeyError: 'Balance'" in line and "commercial_card_balance" in line


def test_empty_stderr_says_so_rather_than_reporting_nothing():
    from agentic_eval.content.oracles import _failure_line

    assert _failure_line("") == "no output on stderr"
    assert _failure_line("   \n\n") == "no output on stderr"


def test_the_oracle_script_names_the_files_it_did_not_find(tmp_path):
    """A case whose CSVs are named differently is a one-line diagnosis, not a
    guess: say what was wanted AND what is actually there."""
    import subprocess
    import sys
    from pathlib import Path

    case = tmp_path / "1059922019"
    case.mkdir()
    (case / "crossbu_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    script = Path("experiments/oracles/case_facts.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--case", "1059922019",
         "--fact", "commercial_card_count", "--data-root", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "crossbu_cards.csv" in result.stderr        # what it wanted
    assert "crossbu_data.csv" in result.stderr         # what is there
    assert not result.stdout.strip()


def test_every_fact_takes_the_same_two_parameters():
    """A dispatch table whose entries disagree about their own signature.

    Six of these named the second parameter `_p` and one named it
    `profile_dir`, so calling the table BY KEYWORD worked for exactly one fact
    and raised "unexpected keyword argument 'profile_dir'" for the rest —
    including both of b1's. Reported from a private environment as an
    unexplained b1 failure, and the traceback that would have named it was
    truncated away before anyone saw it.

    Positional calls hid the disagreement completely, which is why it survived.
    """
    import importlib.util
    import inspect
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "case_facts", Path("experiments/oracles/case_facts.py").resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    signatures = {
        name: list(inspect.signature(fn).parameters)
        for name, fn in module.FACTS.items()
    }
    assert set(map(tuple, signatures.values())) == {("case", "profile_dir")}, (
        f"facts disagree about their parameters: {signatures}"
    )


def test_every_fact_is_callable_by_keyword(tmp_path):
    """The call style that broke. Reaching the body at all is the assertion —
    a missing CSV raises later, and from inside the fact, not at the call."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "case_facts", Path("experiments/oracles/case_facts.py").resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, fact in module.FACTS.items():
        try:
            fact(tmp_path, profile_dir=tmp_path)
        except TypeError as error:
            if "keyword argument" in str(error):
                raise AssertionError(f"{name} rejects a keyword call: {error}")
        except Exception:
            pass          # data missing is fine; the CALL is what is tested
