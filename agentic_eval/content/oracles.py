"""Python ground truth for questions a script can answer outright."""
from __future__ import annotations

import json
import math
import re
import subprocess
from typing import Any

from agentic_eval.common.coerce import (
    _as_list, _evidence_float, _safe_float, _slug,
)


#: Placeholder a rubric command uses to receive the case under test. Explicit
#: rather than automatic: an oracle like `python3 -c "print('true')"` takes no
#: case, and appending one would break it.
_CASE_PLACEHOLDER = "{case_id}"


def _bind_case(command: list[Any], case_id: str | None) -> tuple[list[str], str | None]:
    """Substitute the case into the command, or say why it cannot be.

    A case-blind oracle is the worst kind of wrong: it runs, it returns a
    number, and the number is for a different customer. Measured on a two-case
    run — the rubric never passed `--case`, so every answer about
    `11854808010 ` was graded against `366132845011`, whose payment-return
    count is 0 against the other's 1. Three answers that were false scored
    correct and the run reported 75%.
    """
    parts = [str(part) for part in command]
    if not any(_CASE_PLACEHOLDER in part for part in parts):
        return parts, None
    if not case_id:
        return parts, (
            "Oracle command asks for {case_id} but the record carries no case; "
            "re-run so the case is recorded, or drop the placeholder."
        )
    return [part.replace(_CASE_PLACEHOLDER, case_id) for part in parts], None


def _oracle_value(
    item: dict[str, Any], *, cwd: str | None, timeout: float,
    case_id: str | None = None,
) -> tuple[Any, str | None]:
    """Ground truth for a question a script can answer outright.

    "How many cards does this customer have" has one right answer that Python
    can compute from the case data, so no judge should be asked. Either the
    rubric states `value` directly, or `command` names a script whose stdout is
    the answer (bare number, or JSON with a `value` key).
    """
    if "value" in item:
        return item["value"], None
    command = item.get("command")
    if not isinstance(command, list) or not command:
        return None, "No `value` and no runnable `command` supplied."
    command, unbound = _bind_case(command, case_id)
    if unbound:
        return None, unbound
    try:
        completed = subprocess.run(  # noqa: S603 - operator-supplied oracle
            command,
            cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"Oracle command failed: {error}"
    if completed.returncode != 0:
        return None, (
            f"Oracle command exited {completed.returncode}: "
            f"{completed.stderr.strip()[:200]}"
        )
    text = completed.stdout.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, None
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"], None
    return payload, None


#: Small counts are written as words. "The customer had ONE returned payment"
#: states the count exactly, and a scan that only sees digits reported "no such
#: figure in the answer" against a ground truth of 1 — the answer was right and
#: named the figure, just not in digits. Only up to twelve: past that, prose
#: switches to digits and a word-match would be reaching.
_NUMBER_WORDS = {
    "zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def _figure_in_text(answer: str, wanted: float) -> str | None:
    """The expected value written as a standalone figure in the answer.

    The primary path matches against the claims the judge extracted, which is
    precise but depends on it having listed the figure as a material mention.
    It does not always: one answer said "The customer had **1 returned
    payment**" and the oracle reported "no such figure in the answer", because
    that 1 never reached the mention list.

    The question this oracle asks is whether the ANSWER states the value, so
    the answer's own text is the better authority. Used only as a fallback,
    and matched on token boundaries so 1 does not find itself inside 105,818.
    """
    for token in re.findall(r"-?\d[\d,]*\.?\d*", answer):
        try:
            if math.isclose(float(token.replace(",", "")), wanted,
                            rel_tol=1e-12, abs_tol=1e-9):
                return token
        except ValueError:
            continue
    return None


def _spelled_number_in(answer: str, wanted: float) -> str | None:
    """The word the answer used for `wanted`, if it used one."""
    if wanted != int(wanted) or not 0 <= wanted <= 12:
        return None
    for word, value in _NUMBER_WORDS.items():
        if value == int(wanted) and re.search(rf"\b{word}\b", answer, re.I):
            return word
    return None


#: A negation that PRESUPPOSES what it appears to deny. "No OTHER payment
#: returns are recorded" is only sayable when one exists, so reading it as a
#: denial inverts the answer. Measured: an answer that opened "The customer had
#: one returned payment of $105,818.60" and closed with this sentence was
#: scored as stating False, against a ground truth of True — correct answer,
#: failed oracle, and the fault was one word.
_QUALIFIED_NEGATION = re.compile(
    r"\b(?:no|not any)\s+(?:other|further|additional|more|second|subsequent)\b",
    re.I,
)


def _is_denial(pattern: str, answer: str) -> bool:
    """Does this negative pattern match a real denial, not a qualified one?

    Every place the pattern hits is checked: if all of them sit inside "no
    other ...", the answer is scoping its claim, not denying it.
    """
    hits = list(re.finditer(str(pattern), answer))
    if not hits:
        return False
    return any(not _QUALIFIED_NEGATION.match(hit.group(0)) for hit in hits)


def _boolean_answer_check(
    item: dict[str, Any], expected: Any, normalized_answer: str,
) -> dict[str, Any]:
    """Check a yes/no ground truth against the answer's stated polarity.

    An existence question ("did the customer have any payment returns?") has a
    right answer a script can compute, but the answer states it in prose, and
    no amount of number matching reads a "no" out of a sentence. The rubric
    supplies the patterns, so the check stays deterministic and auditable
    rather than smuggling a second judge into the Python path.

    Ambiguity is reported, never guessed: if the answer matches both polarities
    or neither, the verdict is `unavailable` and says so.
    """
    truth = expected if isinstance(expected, bool) else str(expected).strip().lower() in {
        "true", "yes", "1",
    }
    affirmative = [
        (pattern, match.span())
        for pattern in _as_list(item.get("affirmative_patterns"))
        for match in re.finditer(str(pattern), normalized_answer)
    ]
    negative = [
        (pattern, match.span())
        for pattern in _as_list(item.get("negative_patterns"))
        if _is_denial(pattern, normalized_answer)
        for match in re.finditer(str(pattern), normalized_answer)
    ]
    if not affirmative and not negative:
        return {
            "verdict": "unavailable",
            "reason": "The answer matches no affirmative or negative pattern.",
        }

    # A denial wins only where it actually COVERS the claim.
    #
    # Negation used to win outright, for a good reason: on "had NO returned
    # payments" the affirmative pattern (`had ... returns`) matches INSIDE the
    # sentence that denies it, so calling a double match ambiguous made every
    # correct negative answer unscoreable.
    #
    # But it also swallowed answers that affirm one thing and deny another:
    #
    #   "had 24 returned payments ... with no extreme single-date spikes"
    #   "external delinquency index peaked at 0.52 ... no flagged delinquency"
    #
    # Both were graded FALSE while stating the opposite. The difference is
    # containment: in the true denial the affirmative hit lies inside the
    # negated span; here it sits in another clause entirely. So an affirmative
    # match that no denial overlaps is a claim in its own right.
    def _overlaps(span, spans) -> bool:
        return any(span[0] < end and start < span[1] for start, end in spans)

    negative_spans = [span for _pattern, span in negative]
    standalone = [
        (pattern, span) for pattern, span in affirmative
        if not _overlaps(span, negative_spans)
    ]
    stated = bool(standalone) if negative else True
    matched = (standalone or negative or affirmative)[0][0]
    return {
        "verdict": "pass" if stated == truth else "fail",
        "reason": (
            f"Ground truth is {truth}; the answer states {stated} "
            f"(matched: {matched})."
            + (
                " Both polarities matched; the affirmative sits outside every "
                "denial, so it stands."
                if negative and standalone else
                " Both polarities matched; every affirmative falls inside a "
                "denial, so negation took precedence."
                if negative and affirmative else ""
            )
        ),
    }


def evaluate_expected_answers(
    claims: list[dict[str, Any]], rubric: dict[str, Any], *,
    answer: str = "", cwd: str | None = None, timeout: float = 60.0,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compare the answer against Python-computed ground truth, no LLM involved.

    This is the deterministic path for simple questions. It is deliberately
    separate from reasoning-trace judging: it does not ask HOW the system got there,
    only whether the answer states the right value.
    """
    mentions = [
        (claim["claim_id"], mention)
        for claim in claims if claim.get("is_factual")
        for mention in claim.get("numeric_mentions") or []
        if mention.get("material", True) and mention.get("value") is not None
    ]
    normalized_answer = " ".join(str(answer).split()).lower()
    results = []
    for index, item in enumerate(_as_list(rubric.get("expected_answers")), 1):
        if not isinstance(item, dict):
            continue
        oracle_id = str(item.get("id") or f"oracle_{index:02d}")
        expected, error = _oracle_value(item, cwd=cwd, timeout=timeout, case_id=case_id)
        tolerance = abs(_safe_float(item.get("tolerance")) or 0.0)
        row = {
            "expected_answer_id": oracle_id,
            "description": item.get("description"),
            "expected": expected,
            "source": "literal" if "value" in item else "command",
            "tolerance": tolerance,
            "critical": bool(item.get("critical")),
            "matched_claim_id": None,
            "matched_value": None,
        }
        expected_number = _evidence_float(expected)
        if not error and _slug(item.get("kind")) == "boolean":
            results.append({**row, **_boolean_answer_check(
                item, expected, normalized_answer,
            )})
            continue
        if error or expected is None:
            results.append({
                **row, "verdict": "unavailable",
                "reason": error or "Oracle produced no value.",
            })
            continue
        if expected_number is None:
            # A non-numeric ground truth is checked as an exact phrase.
            present = " ".join(str(expected).split()).lower() in normalized_answer
            results.append({
                **row, "verdict": "pass" if present else "fail",
                "reason": (
                    "The expected text appears in the answer." if present
                    else "The expected text does not appear in the answer."
                ),
            })
            continue
        match = next((
            (claim_id, mention) for claim_id, mention in mentions
            if math.isclose(
                float(mention["value"]), expected_number,
                rel_tol=1e-12, abs_tol=max(tolerance, 1e-12),
            )
        ), None)
        if match is None:
            # A count of zero is normally stated in words ("no returned
            # payments"), never as the digit. Without this the only way to
            # pass would be to write a number the question never asked for.
            #
            # But such a pattern encodes what the ANSWER should say, which
            # depends on what the truth IS. `accept_when_expected` pins it:
            # the words for zero must only pass when zero is the answer. On a
            # multi-case run the same rubric met a case whose count is 1, and
            # "no returned payments" — plainly wrong — matched and passed.
            # Before falling back to patterns: did the answer simply spell
            # the number out?
            spelled = (
                _spelled_number_in(normalized_answer, expected_number)
                if expected_number is not None else None
            )
            if spelled:
                results.append({
                    **row, "verdict": "pass",
                    "reason": (
                        f"The answer writes the value as a word "
                        f"(\"{spelled}\" = {expected_number:g})."
                    ),
                })
                continue
            literal = (
                _figure_in_text(normalized_answer, expected_number)
                if expected_number is not None else None
            )
            if literal:
                results.append({
                    **row, "verdict": "pass",
                    "reason": (
                        f"The answer states {literal}, though no extracted "
                        "claim carried it as a material figure."
                    ),
                })
                continue
            gate = item.get("accept_when_expected")
            gate_open = gate is None or (
                expected_number is not None
                and math.isclose(float(gate), expected_number, rel_tol=1e-12,
                                 abs_tol=max(tolerance, 1e-12))
            )
            accepted = next((
                pattern for pattern in _as_list(item.get("accept_patterns"))
                if gate_open and re.search(str(pattern), normalized_answer)
            ), None)
            if accepted:
                results.append({
                    **row, "verdict": "pass",
                    "reason": f"The answer states the value in words (matched: {accepted}).",
                })
                continue
            results.append({
                **row, "verdict": "fail",
                "reason": (
                    f"No material number in the answer equals {expected_number} "
                    f"(tolerance {tolerance})."
                ),
            })
            continue
        results.append({
            **row, "verdict": "pass",
            "matched_claim_id": match[0],
            "matched_value": match[1].get("value"),
            "reason": f"{match[1].get('written')} matches the computed answer.",
        })
    return results

