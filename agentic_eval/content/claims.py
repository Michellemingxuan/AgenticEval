"""Atomic claims: normalization, hedges, label numbers, table-cell coverage."""
from __future__ import annotations

import re
from typing import Any

from agentic_eval.common.coerce import _as_list, _safe_float, _slug, squash_prose
from agentic_eval.content.numeric import _comparator
from agentic_eval.content.verdicts import CLAIM_STANCES


_CELL_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"((?:[~\u2248]|[<>]=?|[\u2264\u2265])?\s*[$\u00a3\u20ac]?\s*[-+]?\d[\d,]*(?:\.\d+)?%?\+?)"
)


_DATE_LIKE = re.compile(
    r"^\s*(?:"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[-\s]*'?\d{2,4}"
    r"|\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|q[1-4]\s*'?\d{2,4}"
    r"|(?:fy|cy)\s*'?\d{2,4}"
    # No bare four-digit branch: "4200" is a quantity, not a year, and
    # treating every 4-digit figure as a period silently drops real
    # measurements from the numeric layer.
    r")\s*$",
    re.IGNORECASE,
)


def _is_date_like(written: Any) -> bool:
    """Is this mention a period rather than a quantity?

    "June 2025" was being extracted as the material number 2025 and checked
    against `summary.last.period` — which holds "2025-06", a period string. The
    claim is right, the number is not a measurement, and the mismatch was
    charged to the answer.
    """
    return bool(_DATE_LIKE.match(str(written or "").strip()))


def _is_metric_label_number(written: Any, span: Any) -> bool:
    """Is this number naming a metric rather than measuring anything?

    "30+ DPD" is the name of the bucket `times_30_dpd_max`, not a measurement
    — the measured value there was 2.0. Marked material, such a label enters
    the traceability denominator, can never resolve to a tool output, and is
    then reported as an invented number.
    """
    core = re.sub(r"[^\d.]", "", str(written or ""))
    if not core:
        return False
    pattern = re.compile(
        rf"(?<![\d.]){re.escape(core)}\s*\+?\s*[-\s_]*(?:dpd\b|days?\b)", re.IGNORECASE,
    )
    return bool(pattern.search(str(span or "")))


#: Words a judge sometimes returns as `written` for a quantity — "one",
#: "several", "none". Kept for documentation; the check below is the general
#: form, because the judge also returns whole phrases ("no observable risk
#: signal", "no breaches below bureau risk thresholds").
_WORD_QUANTIFIERS = {
    "one", "two", "three", "single", "a", "an", "none", "zero", "no",
    "several", "many", "few", "some", "both", "all",
}

_HAS_DIGIT = re.compile(r"\d")


def _is_digitless_quantity(written: Any) -> bool:
    """Is this "number" a word or phrase with no figure in it?

    Nothing can be located for it, so it should never have entered the numeric
    layer — but the judge pairs such a mention with `value: 0`, which then
    matches any zero anywhere in the evidence and is recorded as TRACED. That
    is worse than a miss: a phrase counts as a verified number and inflates the
    traceability numerator. Observed on "There were no observable risk signals
    or breaches below bureau risk thresholds", which contributed two.
    """
    return not _HAS_DIGIT.search(str(written or ""))


_DATE_FRAGMENT = re.compile(r"(?<![\d/])\d{1,2}/\d{1,2}(?:/\d{2,4})?(?![\d/])")


def _is_date_fragment(written: Any, span: Any) -> bool:
    """Is this number one half of a slashed date the extractor split apart?

    "S BERTRAM ($49,400 6/13)" yields mentions for 6 and 13 — the month and
    day of the transaction, extracted as if they were quantities. `_is_date_like`
    cannot catch them: by the time they are mentions they are bare integers,
    and the date only exists in the span they came from.
    """
    core = str(written or "").strip()
    if not core.isdigit() or len(core) > 2:
        return False
    return any(
        core in match.split("/")
        for match in _DATE_FRAGMENT.findall(str(span or ""))
    )


#: Two endpoints joined by a dash or "to". A range characterises a series —
#: "typical months show 400–700 transactions" — rather than asserting either
#: endpoint as a measurement, so locating one of them is the wrong test, and
#: failing to locate it was reported as an invented number.
_RANGE = re.compile(
    r"^\s*[~\u2248<>]?\s*[$\u00a3\u20ac]?\s*\d[\d,]*(?:\.\d+)?\s*[KMB]?\s*"
    r"(?:[-\u2013\u2014]|to)\s*"
    r"[$\u00a3\u20ac]?\s*\d[\d,]*(?:\.\d+)?\s*[KMB]?\s*$",
    re.IGNORECASE,
)


def _is_range(written: Any) -> bool:
    # "2025-06" is a period, not 2025 through 6. Dates are excluded here too so
    # the test holds whichever order a caller applies the filters in.
    return bool(_RANGE.match(str(written or "").strip())) and not _is_date_like(written)


#: A claim ABOUT what a curated report says. The figures in it are quoted from
#: that report, not measurements the answer asserts, so checking them against
#: tool output asks the wrong question: "does the live data equal the number
#: the report printed?" — when the claim's own content is that the report
#: printed it. Left material they are unlocatable by construction.
_REPORT_ATTRIBUTION = re.compile(
    r"\b(?:curated\s+)?(?:report|summary|file)s?\b[^.]{0,60}?"
    r"\b(?:state|states|stated|say|says|document|documents|documented|"
    r"cover|covers|covered|confirm|confirms|list|lists|note|notes|"
    r"mention|mentions|indicate|indicates|report|reports|show|shows)\b",
    re.IGNORECASE,
)


def _is_report_attribution(*texts: Any) -> bool:
    return any(_REPORT_ATTRIBUTION.search(str(text or "")) for text in texts)


def _normalize_claim(raw: dict[str, Any], index: int) -> dict[str, Any]:
    claim_type = _slug(raw.get("claim_type") or "qualitative_fact")
    is_factual = raw.get("is_factual")
    if is_factual is None:
        is_factual = claim_type not in {"recommendation", "uncertainty_or_data_gap", "opinion"}
    stance = _slug(raw.get("stance") or "asserted")
    if stance not in CLAIM_STANCES:
        stance = "asserted"
    answer_span = str(raw.get("answer_span") or "")
    attribution = _is_report_attribution(
        raw.get("proposition"), raw.get("claim"), answer_span,
    )
    if attribution and stance == "asserted":
        stance = "attributed_unendorsed"
    mentions = []
    for mention in _as_list(raw.get("numeric_mentions")):
        if not isinstance(mention, dict):
            continue
        written = str(mention.get("written") or mention.get("text") or "")
        # A figure the answer quotes in order to disown it is not the answer's
        # own assertion, so it must not be checked as one.
        quoted = bool(mention.get("quoted")) or stance == "attributed_unendorsed"
        material = bool(mention.get("material", True)) and not quoted
        if material and _is_metric_label_number(written, answer_span):
            material = False
        # A spelled-out quantifier has no digits to locate. Marked material it
        # either reports a correct answer as an invented number, or — paired
        # with `value: 0` — matches a stray zero and counts as verified.
        if material and _is_digitless_quantity(written):
            material = False
        if material and (
            _is_date_like(written) or _is_date_fragment(written, answer_span)
        ):
            material = False
        if material and _is_range(written):
            material = False
        # A figure the claim attributes to a report is quoted, whoever the
        # judge said asserted it.
        if material and attribution:
            material = False
            quoted = True
        mentions.append({
            "written": written,
            # What this number quantifies. Without it a mention is a bare
            # figure: "1" cannot be told from the sum or the row count in
            # "sum(Balance) ... = $174,897.36 (over 1 ... row(s); 3 total)",
            # so neither the judge nor Python can pick the right field.
            "measures": (
                str(mention.get("measures") or mention.get("quantity") or "").strip()
                or None
            ),
            # Fall back to the written form. A judge asked for the value of a
            # hedge often returns null — it cannot pick one number for "~28+" —
            # and without this the claim side is unknown while the evidence
            # side resolves, which reads downstream as a mismatch and
            # contradicts a true claim.
            "value": (
                _safe_float(mention.get("value"))
                if mention.get("value") is not None else _safe_float(written)
            ),
            "unit": mention.get("unit"),
            "comparator": _slug(mention.get("comparator")) and str(
                mention.get("comparator")
            ) or _comparator(written),
            "quoted": quoted,
            "material": material,
        })
    # A report's figure that the answer relays — and here corrects against live
    # data — is a claim ABOUT the report, not a measurement. Once every figure
    # in it is quoted, the claim is qualitative, and leaving `claim_type` as
    # `quantitative_fact` keeps dragging it back into the numeric layer.
    if attribution and not any(
        mention["material"] for mention in mentions
    ):
        claim_type = "qualitative_fact"
    return {
        "stance": stance,
        "report_attribution": attribution,
        # Set when this claim says something an earlier claim already
        # said. Counting restatements multiplies one fact — and one
        # defect — by however many times the answer repeated itself.
        "restates_claim_id": (
            str(raw["restates_claim_id"])
            if raw.get("restates_claim_id") else None
        ),
        "claim_id": str(raw.get("claim_id") or f"c{index:03d}"),
        # Who produced this claim. `extracted` came from a model and must be
        # anchored to the answer; `table_cell` was derived by Python from a
        # parsed table, so there is nothing to anchor.
        "origin": _slug(raw.get("origin") or "extracted"),
        "block_id": str(raw.get("block_id") or ""),
        "source_locator": raw.get("source_locator") if isinstance(raw.get("source_locator"), dict) else {},
        "answer_span": str(raw.get("answer_span") or ""),
        "proposition": str(raw.get("proposition") or raw.get("claim") or "").strip(),
        "claim_type": claim_type,
        "is_factual": bool(is_factual),
        "numeric_mentions": mentions,
    }


#: A claim whose span the answer does not contain. Both are evaluator failures:
#: they say the extractor did not anchor its output, not that the system said
#: anything wrong.
SPAN_FAILURES = {
    "span_missing",         # no answer_span returned at all
    "span_not_in_answer",   # a span the answer does not contain
}


def _row_cells_present(span: str, haystack: str, *, floor: int = 3) -> bool:
    """Is every cell of a reconstructed table row present in the answer?

    An extractor quoting a markdown table does not copy the row; it rebuilds it
    as "cell | column header: cell", so the span is not a substring even though
    every part of it is in the answer. Splitting on the pipe and requiring each
    substantial piece keeps the check honest — a fabricated cell still fails —
    while not calling an entire table fabricated.
    """
    def present(cell: str) -> bool:
        squashed = squash_prose(cell)
        if len(squashed) < floor:
            return True          # too short to prove anything either way
        if squashed in haystack:
            return True
        # The extractor labels a cell with its column header — "Specialist
        # view: All large transactions" — so the header is not in the row.
        _, _, remainder = cell.partition(":")
        remainder = squash_prose(remainder)
        return len(remainder) >= floor and remainder in haystack

    # Table cells are short — a period and a score, not sentences — so the
    # floor is low and the strength comes from requiring EVERY cell, with at
    # least two to check. Telling the extractor to quote rows verbatim did not
    # work: it still emits "2024-09 | TSR Score: 39.6" for a row reading
    # "| 2024-09 | 39.6 | <10 | 703 |".
    cells = span.split("|")
    substantial = [c for c in cells if len(squash_prose(c)) >= floor]
    return len(substantial) >= 2 and all(present(cell) for cell in cells)


def validate_claim_spans(
    claims: list[dict[str, Any]], answer: str,
) -> list[dict[str, Any]]:
    """Check every extracted claim points at text the answer actually contains.

    A claim carries the `answer_span` the extractor says it came from, and
    nothing ever checked it — the same hole the numeric layer had before
    citations were audited. Unanchored, a claim the answer never made is scored
    exactly like one it did, and a hallucinated CLAIM is invisible to a cascade
    whose whole job is finding hallucinated numbers.

    Recorded, not enforced: a claim keeps its verdicts either way, so the rate
    can be read before anything is dropped on the strength of it.
    """
    haystack = squash_prose(answer)
    for claim in claims:
        if claim.get("origin") != "extracted":
            claim["span_verified"] = True
            claim["span_failure"] = None
            continue
        raw = str(claim.get("answer_span") or "")
        span = squash_prose(raw)
        failure = (
            "span_missing" if not span
            else "span_not_in_answer"
            if not (span in haystack or _row_cells_present(raw, haystack))
            else None
        )
        claim["span_verified"] = failure is None
        claim["span_failure"] = failure
    return claims


#: Words that carry no subject matter, so overlap between two propositions is
#: measured on what they are ABOUT rather than on English.
_STOPWORDS = {
    "a", "an", "the", "was", "were", "is", "are", "be", "been", "in", "of",
    "and", "or", "to", "for", "on", "at", "by", "with", "its", "it", "this",
    "that", "these", "those", "there", "as", "from", "had", "has", "have",
    "both", "also", "while", "which", "when", "during", "into", "over",
}


def _content_tokens(text: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z][a-z_]+", str(text or "").lower())
        if token not in _STOPWORDS
    }


#: Month names and years, so two claims about different periods are never
#: collapsed. Without this, "In May 2025, CDSS was under 10" was marked as
#: restating "In September 2024, CDSS was under 10" — same words, same figure,
#: different fact — because the token comparison drops digits.
_PERIOD_TOKEN = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(19|20)\d{2}\b",
    re.IGNORECASE,
)


def _period_tokens(text: Any) -> set[str]:
    return {
        match.group(0).lower()[:3] if match.group(1) else match.group(0)
        for match in _PERIOD_TOKEN.finditer(str(text or ""))
    }


def _material_values(claim: dict[str, Any]) -> set[float]:
    return {
        mention["value"] for mention in claim.get("numeric_mentions") or []
        if mention.get("material") and mention.get("value") is not None
    }


def mark_component_restatements(
    claims: list[dict[str, Any]], *, overlap: float = 0.6,
) -> list[dict[str, Any]]:
    """Mark a claim that only restates figures an earlier claim already stated.

    The extractor emits a compound claim and then its parts:

        c7   "key internal drivers were high unpaid amount (0.491) and
              debt service ratio 8.14"
        c17  "the weighted unpaid amount was 0.491"      <- same fact
        c18  "the debt service ratio was 8.14"           <- same fact

    and marks none of them `restates_claim_id`, so one finding is counted three
    times and every rate computed over the claim set is diluted.

    Two conditions, because either alone is wrong. The later claim must add no
    NEW figure — its material values are a subset of the earlier claim's — and
    it must be about the same thing, measured as overlap of subject words. The
    subset test alone would collapse "there is 1 commercial card" and "the
    entire balance sits on that single card", which share the figure 1 and
    assert different facts.
    """
    for index, claim in enumerate(claims):
        if claim.get("restates_claim_id"):
            continue
        values = _material_values(claim)
        if not values:
            continue
        tokens = _content_tokens(claim.get("proposition"))
        if not tokens:
            continue
        periods = _period_tokens(claim.get("proposition"))
        for earlier in claims[:index]:
            if earlier.get("restates_claim_id") or earlier is claim:
                continue
            earlier_values = _material_values(earlier)
            if not earlier_values or not values <= earlier_values:
                continue
            # A compound claim covering several months subsumes a claim about
            # one of them; a claim about a DIFFERENT month is a different fact.
            if periods and not periods <= _period_tokens(earlier.get("proposition")):
                continue
            shared = tokens & _content_tokens(earlier.get("proposition"))
            if len(shared) / len(tokens) >= overlap:
                claim["restates_claim_id"] = earlier["claim_id"]
                break
    return claims


def _table_numeric_mentions(value: str) -> list[dict[str, Any]]:
    mentions = []
    for match in _CELL_NUMBER.finditer(value):
        written = match.group(1).strip()
        number = _safe_float(written)
        if number is None:
            continue
        mentions.append({
            "written": written,
            "value": number,
            "unit": "percent" if "%" in written else None,
            "material": True,
        })
    return mentions


def _ensure_table_cell_claims(
    blocks: list[dict[str, Any]], claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Guarantee one location-preserving atomic claim per eligible table cell.

    The LLM still supplies semantic claim wording when it extracted the cell.
    Python supplies the locator and fills any omitted cell deterministically,
    so table completeness is not dependent on an instruction-following rate.
    """
    for block in blocks:
        if block.get("type") != "table":
            continue
        block_id = str(block.get("block_id") or "")
        section = str(block.get("section") or "")
        for cell in block.get("data_cells") or []:
            row = int(cell["row"])
            column = int(cell["column"])
            locator = {
                "row": row,
                "column": column,
                "row_header": cell.get("row_header"),
                "column_header": cell.get("column_header"),
                "header_path": [
                    value for value in (section, cell.get("column_header")) if value
                ],
            }
            covered = next((
                claim for claim in claims
                if claim.get("block_id") == block_id
                and (claim.get("source_locator") or {}).get("row") == row
                and (claim.get("source_locator") or {}).get("column") == column
            ), None)
            if covered is not None:
                covered["source_locator"] = {
                    **locator, **(covered.get("source_locator") or {}),
                }
                continue

            claim_text = " ".join(str(cell.get("claim_text") or "").split())
            cell_value = " ".join(str(cell.get("value") or "").split())
            row_header = str(cell.get("row_header") or "").strip()
            column_header = str(cell.get("column_header") or "").strip()

            def match_score(claim: dict[str, Any]) -> int:
                if claim.get("block_id") != block_id or claim.get("source_locator"):
                    return -1
                span = " ".join(str(claim.get("answer_span") or "").split())
                if span not in {claim_text, cell_value}:
                    return -1
                score = 100 if span == claim_text else 50
                # Tie-breakers, for the case where two claims quote the same
                # span in one block. `source_locator` carries what
                # `time_window`/`entities`/`metrics` used to: those were five
                # extra fields per claim read by this function alone.
                locator = claim.get("source_locator") or {}
                if str(locator.get("row_header") or "").lower() == row_header.lower():
                    score += 20
                if _slug(str(locator.get("column_header") or "")) == _slug(column_header):
                    score += 10
                return score

            candidates = [
                (match_score(claim), claim) for claim in claims
                if match_score(claim) >= 0
            ]
            extracted = max(candidates, key=lambda item: item[0])[1] if candidates else None
            if extracted is not None:
                extracted["source_locator"] = locator
                continue

            value = str(cell.get("value") or "").strip()
            proposition = (
                f"For {row_header}: {value}."
                if column_header.lower() in {"note", "notes", "comment", "comments"}
                else f"For {row_header}, {column_header} was {value}."
            )
            mentions = _table_numeric_mentions(value)
            claims.append(_normalize_claim({
                "claim_id": f"tc_{block_id}_r{row}_c{column}",
                "origin": "table_cell",
                "block_id": block_id,
                "source_locator": locator,
                "answer_span": claim_text,
                "proposition": proposition,
                "claim_type": "point_estimate" if mentions else "qualitative_fact",
                "is_factual": True,
                "metrics": [] if column_header.lower() in {"note", "notes"} else [column_header],
                "numeric_mentions": mentions,
            }, len(claims) + 1))
    return claims


def _drop_placeholder_table_claims(
    blocks: list[dict[str, Any]], claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove audit-added claims for cells that display only a placeholder.

    A bare ``--``/``N/A`` says that the answer did not display a value; it does
    not by itself assert that the underlying data is unavailable. Treating it
    as a domain fact creates tautological or misleading fact judgments.
    """
    placeholders = {"", "-", "--", "—", "n/a", "na", "null", "none"}
    missing_locations = set()
    for block in blocks:
        if block.get("type") != "table":
            continue
        for row_index, row in enumerate(block.get("rows") or [], 1):
            for column_index, value in enumerate(row[1:], 1):
                if str(value).strip().lower() in placeholders:
                    missing_locations.add((block.get("block_id"), row_index, column_index))
    return [
        claim for claim in claims
        if (
            claim.get("block_id"),
            (claim.get("source_locator") or {}).get("row"),
            (claim.get("source_locator") or {}).get("column"),
        ) not in missing_locations
    ]



def _drop_false_restatements(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a claim that adds a number the one it "restates" never stated.

    Observed: "there were 357 attempts, and 0 were returned" was marked as
    restating "the customer had no payment returns". They share the zero, so
    the judge collapsed them — and 357, a fact the first claim never makes,
    stopped being counted or verified. A restatement may add emphasis or a
    source; if it adds a MEASUREMENT it is not one.
    """
    by_id = {claim["claim_id"]: claim for claim in claims}

    def material_values(claim: dict[str, Any]) -> set[float]:
        return {
            mention["value"]
            for mention in claim.get("numeric_mentions") or []
            if mention.get("material") and mention.get("value") is not None
        }

    for claim in claims:
        target_id = claim.get("restates_claim_id")
        if not target_id:
            continue
        target = by_id.get(target_id)
        if target is None or target is claim:
            # A dangling pointer would silently drop the claim from every
            # denominator, so treat it as no pointer at all.
            claim["restates_claim_id"] = None
            continue
        if material_values(claim) - material_values(target):
            claim["restates_claim_id"] = None
    return claims
