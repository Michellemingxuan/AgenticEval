"""Distribution and set-similarity helpers shared by the eval modules."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from itertools import combinations
from typing import Any, Iterable


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / len(aa | bb)


def _pairwise(values: list[Any], fn) -> float | None:
    if len(values) < 2:
        return None
    return statistics.mean(fn(a, b) for a, b in combinations(values, 2))


def _pairwise_values(values: list[Any], fn) -> list[float]:
    return [fn(a, b) for a, b in combinations(values, 2)]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _outlier_count(values: list[float]) -> int:
    if len(values) < 4:
        return 0
    q1, q3 = _percentile(values, 0.25), _percentile(values, 0.75)
    assert q1 is not None and q3 is not None
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return sum(value < lo or value > hi for value in values)


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    """Return an auditable repeated-run distribution.

    Tukey-IQR outlier detection is exposed only when at least four observations
    exist. For smaller k, the raw values and max remain visible, but an outlier
    rate would imply more statistical confidence than the sample supports.
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "n": 0, "values": [], "mean": None, "median": None,
            "stdev": None, "min": None, "p95": None, "max": None,
            "outlier_method": "tukey_iqr_1.5", "outlier_eligible": False,
            "outlier_count": None, "outlier_rate": None,
            "outlier_values": [],
        }
    outlier_eligible = len(finite) >= 4
    outliers: list[float] = []
    if outlier_eligible:
        q1, q3 = _percentile(finite, 0.25), _percentile(finite, 0.75)
        assert q1 is not None and q3 is not None
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = [value for value in finite if value < lo or value > hi]
    return {
        "n": len(finite),
        "values": finite,
        "mean": statistics.mean(finite),
        "median": statistics.median(finite),
        "stdev": statistics.stdev(finite) if len(finite) >= 2 else 0.0,
        "min": min(finite),
        "p95": _percentile(finite, 0.95),
        "max": max(finite),
        "outlier_method": "tukey_iqr_1.5",
        "outlier_eligible": outlier_eligible,
        "outlier_count": len(outliers) if outlier_eligible else None,
        "outlier_rate": len(outliers) / len(finite) if outlier_eligible else None,
        "outlier_values": outliers,
    }


def _optional_mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.mean(values) if values else None


def _optional_rate(rows: list[dict], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _optional_distribution(rows: list[dict], field: str) -> dict[str, Any]:
    return _distribution(
        float(row[field]) for row in rows if row.get(field) is not None
    )


def _exact_consistency(values: list[Any]) -> tuple[float | None, Any, int]:
    """Share of runs matching the modal value; undefined for a single run.

    A lone run trivially agrees with itself, and reporting that as 100%
    consistency claims a property of the system that was never observed —
    beside a pairwise measure that honestly reads "—" for the same data.
    """
    if not values:
        return None, None, 0
    modal, modal_n = Counter(values).most_common(1)[0]
    if len(values) < 2:
        return None, modal, len(set(values))
    return modal_n / len(values), modal, len(set(values))


def _multiset_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = Counter(a), Counter(b)
    keys = set(aa) | set(bb)
    if not keys:
        return 1.0
    return sum(min(aa[key], bb[key]) for key in keys) / sum(
        max(aa[key], bb[key]) for key in keys
    )

