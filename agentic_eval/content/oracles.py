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


def _oracle_value(
    item: dict[str, Any], *, cwd: str | None, timeout: float,
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
    try:
        completed = subprocess.run(  # noqa: S603 - operator-supplied oracle
            [str(part) for part in command],
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
        pattern for pattern in _as_list(item.get("affirmative_patterns"))
        if re.search(str(pattern), normalized_answer)
    ]
    negative = [
        pattern for pattern in _as_list(item.get("negative_patterns"))
        if re.search(str(pattern), normalized_answer)
    ]
    if not affirmative and not negative:
        return {
            "verdict": "unavailable",
            "reason": "The answer matches no affirmative or negative pattern.",
        }
    # Negation wins when both fire. An affirmative pattern for an existence
    # question ("had ... returns") matches inside the negated sentence that
    # denies it ("had NO returned payments"), so treating a double match as
    # ambiguous would make every correct negative answer unscoreable.
    stated = not negative
    return {
        "verdict": "pass" if stated == truth else "fail",
        "reason": (
            f"Ground truth is {truth}; the answer states {stated} "
            f"(matched: {(negative or affirmative)[0]})."
            + (" Both polarities matched; negation took precedence."
               if negative and affirmative else "")
        ),
    }


def evaluate_expected_answers(
    claims: list[dict[str, Any]], rubric: dict[str, Any], *,
    answer: str = "", cwd: str | None = None, timeout: float = 60.0,
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
        expected, error = _oracle_value(item, cwd=cwd, timeout=timeout)
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
            accepted = next((
                pattern for pattern in _as_list(item.get("accept_patterns"))
                if re.search(str(pattern), normalized_answer)
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

