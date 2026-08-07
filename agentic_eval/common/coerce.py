"""Value coercion and JSON-path resolution shared by every eval module."""
from __future__ import annotations

import json
import math
import re
from typing import Any


_PATH_PART = re.compile(r"([^.[\]]+)|\[(\d+)\]")


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def squash(text: Any) -> str:
    """Whitespace- and quote-insensitive form, for containment tests.

    Used wherever a model must anchor an assertion to source text it does not
    own — a citation into a tool payload, a claim into the answer. A model
    retyping a fragment gets the characters right and the escaping wrong, so
    requiring the literal byte sequence would reject true quotations; this
    still requires the full character sequence, which no fabrication satisfies.
    """
    return re.sub(r"[\s\"'`]+", "", str(text or ""))


def squash_prose(text: Any) -> str:
    """`squash` for markdown prose, additionally blind to emphasis markers.

    An answer writes "has **1 commercial (SBS) card**"; an extractor quoting it
    returns "has 1 commercial (SBS) card". The span is a true quotation and the
    asterisks are formatting, so a containment test that sees them rejects
    every emphasised sentence — which is most of the load-bearing ones.

    Kept separate from `squash`, which guards evidence citations: there the
    payload is data rather than prose, and permissiveness costs strictness
    exactly where fabrication has to be caught.
    """
    return re.sub(r"[\s\"'`*_~]+", "", str(text or ""))


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        number = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return number / 100 if "%" in str(value) else number


def _deep_decode_json(value: Any) -> Any:
    """Decode object/array JSON strings nested inside captured tool output.

    Some AgenticSys tools return a structured outer object whose ``result``
    fields are themselves JSON strings. Keeping that inner layer encoded makes
    a valid path such as ``results[0].result.series[0].value`` impossible for
    the deterministic verifier to resolve.
    """
    if isinstance(value, dict):
        return {key: _deep_decode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_decode_json(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    ):
        return value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return value
    return _deep_decode_json(parsed)


def _evidence_float(value: Any) -> float | None:
    """Strict parse for a value pulled OUT OF EVIDENCE via a json_path.

    `_safe_float` scrapes the first digit run anywhere in a string, which is
    right for reading a number a judge wrote in prose but catastrophic here:
    the data tools return human-readable results, so a path that lands on
    `"count filtered by Return Flag eq '1' = 0 (out of 357 total rows)"`
    yielded 1.0 — the FILTER LITERAL — and the claim was then reported as
    contradicted. Same origin as `$1,000` checked against 2.0 and `26.1%`
    against 0.03.

    A prose sentence is not a measurement. Accept a real number, or a string
    that is ENTIRELY one, and otherwise return None so the comparison stays
    unresolved and surfaces as "unavailable".
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None
    text = value.strip().strip("'\"").strip()
    if "=" in text:
        labelled = _labelled_float(text)
        if labelled is not None:
            return labelled
    percent = text.endswith("%")
    text = text.rstrip("%").strip().lstrip("$£€").strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number / 100 if percent else number


_LABELLED_VALUE = re.compile(
    # "<label> = <number>", optionally followed by a parenthetical aside.
    # Greedy up to the LAST `=`, so a label containing digits or its own `=`
    # cannot be mistaken for the measurement. The aside is matched loosely
    # because it nests parentheses: "(over 1 non-null value(s) in 1 row(s))".
    r"^.*=\s*(?P<value>[-+]?[$£€]?\s*\d[\d,]*(?:\.\d+)?\s*%?)\s*"
    r"(?:\(.*\))?\s*$"
)


def _labelled_float(value: str) -> float | None:
    """Read `count = 357 (out of 357 total rows)` as 357.

    Several data tools return a measurement already formatted for a human. The
    number is genuinely there, so refusing the whole string reports a true
    claim as unlocatable — but scraping the FIRST digit run is what once turned
    "count filtered by Return Flag eq '1' = 0" into 1, the filter literal,
    contradicting a correct answer of 0.

    Taking the value after the LAST `=` reads both correctly: the label may
    contain digits, the measurement is what follows the assignment.
    """
    match = _LABELLED_VALUE.match(value.strip())
    if not match:
        return None
    return _evidence_float(match.group("value"))


def _resolve_path(value: Any, path: str) -> Any:
    path = str(path or "").strip().lstrip("$").lstrip(".")
    if path.startswith("result."):
        path = path[7:]
    current = value
    for match in _PATH_PART.finditer(path):
        key, index = match.groups()
        if key is not None:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(path)
            current = current[key]
        else:
            if not isinstance(current, list) or int(index) >= len(current):
                raise KeyError(path)
            current = current[int(index)]
    return current

