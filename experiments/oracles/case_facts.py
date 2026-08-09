#!/usr/bin/env python3
"""Ground truth for questions a script can answer outright.

Reads the case CSVs directly. This is data access, not a system import: the
evaluator still never loads either AgenticSys checkout, so a wrong answer here
cannot be produced by the same bug that produced a wrong answer there.

    python case_facts.py --fact latest_fico_score

Prints JSON `{"value": ..., "detail": {...}}` on stdout. `value` is what the
evaluator compares against the answer; `detail` is for the reviewer.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import sys
from pathlib import Path


# Anchored to this file, not to the caller's cwd: the script is invoked both
# from an experiment config (`oracle_cwd`) and by hand from the repo root, and
# a cwd-relative default silently resolves to a missing directory in one of
# those two cases.
_PROJECTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = _PROJECTS_ROOT / "AgenticSys_v2" / "data_tables" / "real"
DEFAULT_CASE = "366132845011"
# The catalog tags model features by concept; these are the profile files that
# describe the modelling tables.
CONCEPT_PROFILES = ("model_scores.yaml", "model_scores_transaction.yaml")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _month_key(value: str) -> datetime.datetime:
    """Order months written as `July'2025`."""
    return datetime.datetime.strptime(value.replace("'", " ").strip(), "%B %Y")


def _number(value: str) -> float:
    return float(str(value).replace(",", "").replace("$", "").strip() or 0)


def _table(case: Path, *names: str) -> Path:
    """The first of these filenames that this case actually has.

    Filenames are NOT uniform across cases: one ships `crossbu_cards.csv`, the
    other `crossbu_data_cards.csv`. An oracle that hard-codes one name works on
    the case it was written against and raises on the next — and a raising
    oracle is worse than a missing one, because the run keeps going and the
    metric quietly describes whichever case still parses.
    """
    for name in names:
        candidate = case / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"case {case.name!r} has none of: {', '.join(names)}"
    )


def _payment_rows(case: Path) -> list[dict[str, str]]:
    """Every payment for the case, whichever layout it is stored in.

    Two layouts exist in the real data. One case ships a single merged
    `payments_data.csv`; the other ships `payments_success.csv` beside
    `payments_returns_data.csv`, and the system's own gateway rbinds those two
    into one `payments` table. The oracle has to read the case as the system
    does, or it answers about a table the system never saw — here it simply
    crashed on the split layout, which is why the rubric was silently grading
    every answer against the other case.
    """
    merged = case / "payments_data.csv"
    if merged.exists():
        return _rows(merged)
    rows: list[dict[str, str]] = []
    for name in ("payments_success.csv", "payments_returns_data.csv"):
        path = case / name
        if path.exists():
            rows.extend(_rows(path))
    if not rows:
        raise FileNotFoundError(
            f"no payments table for {case.name!r}: expected payments_data.csv, "
            "or payments_success.csv beside payments_returns_data.csv"
        )
    return rows


def has_payment_returns(case: Path) -> dict:
    rows = _payment_rows(case)
    returned = [
        row for row in rows
        if str(row.get("Return Flag", "")).strip() not in {"", "0"}
        or str(row.get("Return Reason", "")).strip()
    ]
    return {
        "value": bool(returned),
        "detail": {
            "returned_payments": len(returned),
            "total_payments": len(rows),
            "total_paid": round(sum(_number(r["Payment Amount"]) for r in rows), 2),
            "date_range": [
                min(r["Payment Date"] for r in rows),
                max(r["Payment Date"] for r in rows),
            ],
        },
    }


def payment_return_count(case: Path) -> dict:
    result = has_payment_returns(case)
    return {"value": result["detail"]["returned_payments"], "detail": result["detail"]}


def _cards(case: Path) -> list[dict[str, str]]:
    return _rows(_table(case, "crossbu_cards.csv", "crossbu_data_cards.csv"))


def commercial_card_count(case: Path) -> dict:
    """Commercial = the small-business portfolio, not the consumer cards.

    `Card Portfolio` separates SBS (small business) from CPS (consumer). Card
    Type `LOC`/`Lending` describes the lending product, not who holds it, so it
    is reported but not used as the discriminator.
    """
    cards = _cards(case)
    commercial = [row for row in cards if row.get("Card Portfolio") == "SBS"]
    return {
        "value": len(commercial),
        "detail": {
            "commercial_cards": [row["Card Name"] for row in commercial],
            "all_cards": [
                {"name": row["Card Name"], "portfolio": row["Card Portfolio"],
                 "type": row["Card Type"], "balance": _number(row["Balance"])}
                for row in cards
            ],
            "all_card_count": len(cards),
        },
    }


def commercial_card_balance(case: Path) -> dict:
    cards = _cards(case)
    commercial = [row for row in cards if row.get("Card Portfolio") == "SBS"]
    return {
        "value": round(sum(_number(row["Balance"]) for row in commercial), 2),
        "detail": {
            "per_card": {row["Card Name"]: _number(row["Balance"]) for row in commercial},
            "all_cards_balance": round(
                sum(_number(row["Balance"]) for row in cards), 2,
            ),
        },
    }


def latest_fico_score(case: Path) -> dict:
    rows = [row for row in _rows(case / "bureau_data.csv") if row.get("FICO Score")]
    latest = max(rows, key=lambda row: _month_key(row["month"]))
    return {
        "value": _number(latest["FICO Score"]),
        "detail": {"month": latest["month"]},
    }


def has_external_delinquency_features(case: Path, *, profile_dir: Path | None) -> dict:
    """Does the modelling data actually carry external-delinquency features?

    Two conditions, both required: the catalog declares columns under the
    `external_delinquency` concept, AND this case's modelling table contains
    them with values. A declared-but-absent feature is not information the
    model has about this customer.
    """
    declared: list[str] = []
    if profile_dir and profile_dir.is_dir():
        for name in CONCEPT_PROFILES:
            path = profile_dir / name
            if not path.is_file():
                continue
            # Read the YAML textually; parsing it needs no dependency and the
            # evaluator must not import the system to get its own ground truth.
            text = path.read_text(encoding="utf-8")
            for block in re.finditer(
                r"\n  ([A-Za-z0-9_]+):\n(.*?)(?=\n  [A-Za-z0-9_]+:\n|\Z)",
                text, re.DOTALL,
            ):
                body = block.group(2)
                if "external_delinquency" not in body:
                    continue
                aliases = re.findall(r"aliases:\n((?:\s*-\s*\S+\n?)+)", body)
                names = [block.group(1)]
                if aliases:
                    names += re.findall(r"-\s*(\S+)", aliases[0])
                declared.extend(names)
    rows = _rows(case / "modelling_data.csv")
    columns = set(rows[0].keys()) if rows else set()
    present = sorted({name for name in declared if name in columns})
    populated = sorted({
        name for name in present
        if any(str(row.get(name, "")).strip() not in {"", "nan"} for row in rows)
    })
    return {
        "value": bool(populated),
        "detail": {
            "declared_concept_columns": sorted(set(declared)),
            "present_in_case": present,
            "populated_in_case": populated,
        },
    }


def transactions_last_month(case: Path) -> dict:
    """Transactions in the latest month available in the spends table.

    "Last month" means the most recent month the data actually has, which is
    the last month a reader of the table would see. It is a partial capture —
    the extract ends mid-month — so the preceding complete month is reported
    alongside it rather than substituted for it.
    """
    rows = _rows(case / "spends_data.csv")
    months: dict[str, int] = {}
    for row in rows:
        months[row["Month"]] = months.get(row["Month"], 0) + 1
    ordered = sorted(months.items(), key=lambda item: _month_key(item[0]))
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    return {
        "value": latest[1],
        "detail": {
            "month": latest[0],
            "note": "latest month in the table; the extract ends mid-month",
            "previous_complete_month": (
                {"month": previous[0], "transactions": previous[1]}
                if previous else None
            ),
        },
    }


FACTS = {
    "has_payment_returns": lambda case, _p: has_payment_returns(case),
    "payment_return_count": lambda case, _p: payment_return_count(case),
    "commercial_card_count": lambda case, _p: commercial_card_count(case),
    "commercial_card_balance": lambda case, _p: commercial_card_balance(case),
    "latest_fico_score": lambda case, _p: latest_fico_score(case),
    "has_external_delinquency_features": (
        lambda case, profile_dir: has_external_delinquency_features(
            case, profile_dir=profile_dir,
        )
    ),
    "transactions_last_month": lambda case, _p: transactions_last_month(case),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact", required=True, choices=sorted(FACTS))
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--profile-dir",
        default=str(_PROJECTS_ROOT / "AgenticSys_v2" / "config" / "data_profiles"),
    )
    args = parser.parse_args()

    case_dir = Path(args.data_root).expanduser() / args.case
    if not case_dir.is_dir():
        print(f"case directory not found: {case_dir}", file=sys.stderr)
        return 2
    result = FACTS[args.fact](case_dir, Path(args.profile_dir).expanduser())
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
