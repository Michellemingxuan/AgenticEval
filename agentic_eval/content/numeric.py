"""Deterministic numeric verification: derivations, hedges, relations."""
from __future__ import annotations

import math
import re
import statistics
from typing import Any

from agentic_eval.common.coerce import _evidence_float, _resolve_path, _safe_float


def _operand_values(node: Any, select: str | None) -> list[float] | None:
    """Expand one resolved operand into the numbers it contributes.

    A scalar contributes itself. A LIST contributes every element, which is what
    makes an aggregate claim checkable: "the peak TSR was 39.6" is only traced
    when 39.6 is max() over the actual series, not when it happens to equal the
    one bucket the judge pointed at. `select` names the field to read from each
    element of a list of records.

    Strict on purpose: if any element fails to parse, the whole operand is
    unresolved. A partly-read series would silently change what max/mean mean.
    """
    if isinstance(node, dict) and select:
        node = node.get(select)
    if isinstance(node, list):
        values = []
        for element in node:
            if isinstance(element, dict):
                if not select or select not in element:
                    return None
                element = element[select]
            value = _evidence_float(element)
            if value is None:
                return None
            values.append(value)
        return values
    value = _evidence_float(node)
    return None if value is None else [value]


def _operation(name: str, operands: list[float]) -> float | None:
    if not operands:
        return None
    if name == "sum":
        return sum(operands)
    if name == "count":
        return float(len(operands))
    if name == "max":
        return max(operands)
    if name == "min":
        return min(operands)
    if name in {"mean", "average"}:
        return sum(operands) / len(operands)
    # A claim about "typical months" is usually a median over the series, and
    # without this the verifier can only answer "absent" — the figure is in no
    # single cell, so a true summary statistic read as an invented number.
    if name == "median":
        return statistics.median(operands)
    if name == "difference" and len(operands) == 2:
        return operands[0] - operands[1]
    if name == "product":
        return math.prod(operands)
    if name == "ratio" and len(operands) == 2 and operands[1] != 0:
        return operands[0] / operands[1]
    if name == "percent_change" and len(operands) == 2 and operands[0] != 0:
        return (operands[1] - operands[0]) / abs(operands[0])
    return None


def _comparator(written: str) -> str:
    """Read the relation a hedged figure actually asserts.

    "~28+" does not claim the value IS 28; it claims the value is about 28 or
    more. Coercing it to 28.0 and demanding an exact match makes a true claim
    permanently uncheckable — observed: the answer wrote "~28+" for a June-2024
    TSR whose real value is 30.2, and it was reported as a hallucination.
    """
    text = str(written or "").strip()
    lowered = text.lower()
    # Words carry the same relation as symbols. "over 100× the typical size"
    # was read as an equality and compared against an actual 113.6, so a claim
    # its own evidence satisfies was scored a mismatch.
    for prefix, comparator in (
        ("at least", ">="), ("no less than", ">="), ("more than", ">"),
        ("greater than", ">"), ("over", ">"), ("above", ">"),
        ("at most", "<="), ("no more than", "<="), ("up to", "<="),
        ("less than", "<"), ("fewer than", "<"), ("under", "<"), ("below", "<"),
    ):
        if lowered.startswith(prefix):
            return comparator
    if text.endswith("+"):
        return ">="
    if text.startswith((">=", "≥")):
        return ">="
    if text.startswith(("<=", "≤")):
        return "<="
    if text.startswith(">"):
        return ">"
    if text.startswith("<"):
        return "<"
    return "=="


_PRECISION = re.compile(
    r"(?P<int>\d[\d,]*)(?:\.(?P<dec>\d+))?\s*(?P<scale>[KMB])?",
    re.IGNORECASE,
)
_SCALES = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}


def _written_tolerance(written: Any) -> float:
    """How precise the answer's own figure claims to be.

    "$404K" does not assert 404000 exactly; it asserts a total rounded to the
    nearest thousand, and the tool's $404,151.99 is that number. Demanding
    equality made every abbreviated or rounded figure a mismatch — ten of them
    in one run, all correct — so the tolerance is half the last unit the answer
    actually wrote.
    """
    match = _PRECISION.search(str(written or ""))
    if not match:
        return 0.0
    scale = _SCALES.get((match.group("scale") or "").lower(), 1.0)
    decimals = len(match.group("dec") or "")
    # "16%" is parsed to 0.16, so its tolerance belongs on that scale too;
    # leaving it at 0.5 would have let 0.42 satisfy a claim of 16%.
    if "%" in str(written):
        scale /= 100.0
    return (10.0 ** -decimals) * scale / 2.0


def _satisfies(comparator: str, expected: float, computed: float, tolerance: float) -> bool:
    if comparator == ">=":
        return computed >= expected - tolerance
    if comparator == ">":
        return computed > expected - tolerance
    if comparator == "<=":
        return computed <= expected + tolerance
    if comparator == "<":
        return computed < expected + tolerance
    return math.isclose(expected, computed, rel_tol=1e-12, abs_tol=tolerance)


_RELATION_OPERATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9),
    "!=": lambda a, b: not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9),
}


_RELATIONAL_CLAIM_TYPES = {"threshold", "comparison", "ranking"}


def _resolve_relation_side(
    spec: Any, evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[float | None, str | None, str | None, bool]:
    """Resolve one side of a relation.

    Returns (value, source_type, evidence_id, imprecise). `imprecise` is set
    when the path RESOLVED but not to a number — the judge addressed a region
    instead of a value. "22 is lower than typical months" was encoded with the
    right side pointing at `series`, the whole monthly list; that side cannot
    become a scalar, and reporting it as `not_located` charged the answer with
    inventing a figure it never stated.
    """
    if isinstance(spec, dict) and spec.get("json_path"):
        evidence_id = str(spec.get("evidence_id") or "")
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            return None, None, evidence_id or None, False
        try:
            node = _resolve_path(evidence.get("result"), str(spec["json_path"]))
        except (KeyError, TypeError, ValueError):
            return None, evidence.get("source_type"), evidence_id, False
        value = _evidence_float(node)
        imprecise = value is None and node is not None
        return value, evidence.get("source_type"), evidence_id, imprecise
    literal = spec.get("value") if isinstance(spec, dict) else spec
    return _safe_float(literal), "literal", None, False



#: Words that make a figure a BOUND rather than an equality. "all above 720" is
#: satisfied by 721; read as `==` it was a mismatch against the very evidence
#: that proves it.
_LOWER_BOUND = re.compile(
    r"\b(above|over|exceed\w*|greater than|more than|at least|no less than)\b",
    re.I,
)
_UPPER_BOUND = re.compile(
    r"\b(below|under|less than|fewer than|at most|no more than)\b", re.I,
)


def infer_comparator(written: Any, comparator: str) -> str:
    """Upgrade `==` to a bound when the answer's own words state one."""
    if comparator not in {"==", ""}:
        return comparator
    text = str(written or "")
    if _LOWER_BOUND.search(text):
        return ">="
    if _UPPER_BOUND.search(text):
        return "<="
    return comparator or "=="


#: Direction the answer committed to, if any. Only the figure's own text is
#: available here, so this is a leading sign or an explicit word — absent
#: either, the answer stated a size and not a direction.
_FALLING = re.compile(r"^\s*-|\b(decline\w*|decrease\w*|fell|fall\w*|drop\w*|down)\b", re.I)
_RISING = re.compile(r"^\s*\+|\b(rose|rise\w*|increase\w*|grew|grow\w*|up)\b", re.I)


def _stated_direction(text: str) -> int | None:
    if _FALLING.search(text):
        return -1
    if _RISING.search(text):
        return 1
    return None


def comparison_variants(
    written: Any, expected: float, computed: float, tolerance: float,
):
    """Readings of the same figure that mean the same thing.

    Tried only after an exact comparison fails, so a genuine disagreement is
    still a disagreement — "38%" against 36.03% fails every variant.

    * SCALE. A share is written "36%" and the judge reports it as either 36.0
      or 0.36, while the tool reports 0.3603. Only one direction was tried, so
      whichever way round the two landed decided whether a correct figure was
      called wrong.
    * SIGN. "declined by 2.2%" is a magnitude; the tool reports -0.022. Their
      difference is notation, not arithmetic — but only when the answer did not
      write a sign itself.
    """
    text = str(written or "")
    variants = []
    if "%" in text:
        variants.append((expected * 100.0, computed, tolerance * 100.0))
        variants.append((expected / 100.0, computed, tolerance / 100.0))
    # Magnitudes are comparable unless the answer stated a direction that the
    # evidence contradicts. "declined by 2.2%" against -0.022 is notation; "+5%"
    # against -0.05 is the answer getting the direction wrong, which is exactly
    # what this check exists to catch.
    stated = _stated_direction(text)
    if stated is None or computed == 0 or stated == (1 if computed > 0 else -1):
        variants.extend([
            (abs(want), abs(got), tol) for want, got, tol in
            [(expected, computed, tolerance), *variants]
        ])
    return variants


#: A written value that names a PERIOD rather than a quantity. The judge emits
#: these among the numbers — `written_value: "mid-2024"`, `measures: "period of
#: spike for TSR"` — and the numeric parser turns them into nonsense (-2024.0
#: from "mid-2024", 2025.0 from "2025-02 to 2025-05"), which then fails to
#: locate and is charged to the answer as an unsupported figure.
#:
#: A period is a real part of a claim and worth checking; it is simply not a
#: number, and the numeric trace is the wrong instrument for it.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october"
    "|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_PERIOD_FORMS = re.compile(
    r"^\s*(?:"
    r"(?:mid|early|late|end of|start of|beginning of)[\s-]+\d{4}"      # mid-2024
    r"|q[1-4]\s*'?\d{2,4}"                                             # Q2 2024
    r"|\d{4}-\d{2}(?:-\d{2})?(?:\s*(?:to|-|–|through|until)\s*"
    r"\d{4}-\d{2}(?:-\d{2})?)?"                                       # 2025-02 to 2025-05
    r"|(?:" + _MONTHS + r")[\s-]*\d{4}"                                # May 2025
    r"|\d{4}\s*(?:to|-|–|through)\s*\d{4}"                           # 2024-2025
    r")\s*$",
    re.I,
)


def is_period_expression(written: Any, measures: Any = None) -> bool:
    """Does this mention name a period rather than a quantity?

    Two independent signals, either is enough: the written form parses as a
    date or span, or the judge's own `measures` says it is a period. The second
    catches phrasings the pattern misses, and costs nothing — a mention
    described as a period was never a figure to trace.
    """
    if _PERIOD_FORMS.match(str(written or "")):
        return True
    described = str(measures or "").strip().lower()
    return described.startswith(("period ", "period of", "month ", "months ",
                                 "timeframe", "time period", "date range"))
