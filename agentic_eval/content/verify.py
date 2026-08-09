"""Does each claim trace back to operations, and does that route answer the question?

Two questions, and only two. A claim is GROUNDED when both hold:

    it reaches operations   its figures resolve inside a measured source, or —
                            carrying no figures — its evidence names the
                            operations behind it
    the route is eligible   the brief and the calls that produced it answer the
                            question that was asked, not a different one

There is no separate invention verdict. "Did the system make this up" is a
second reading of evidence these two have already settled, and every false
positive this evaluator produced under that label came from its own pointer
failures rather than from the answer.
"""
from __future__ import annotations

import math
import re
from typing import Any

from agentic_eval.common.coerce import (
    _as_list, _evidence_float, _resolve_path, _safe_float, _slug, squash,
)
from agentic_eval.content.evidence import (
    MEASURED_SOURCES, _is_report_evidence,
    _reaches_operations,
)
from agentic_eval.content.numeric import (
    _RELATIONAL_CLAIM_TYPES, _RELATION_OPERATORS, _operand_values, _operation,
    _resolve_relation_side, _satisfies, _written_tolerance,
    comparison_variants as _comparison_variants,
    is_period_expression as _is_period_expression,
    is_stated_constant as _is_stated_constant,
    infer_comparator as _infer_comparator,
)
from agentic_eval.content.verdicts import (
    FACT_VERDICTS,
)


#: Failures caused by the JUDGE, not the system under test. The judge did not
#: return a usable mapping — it says nothing about whether the answer's number
#: is real. Counting these against the system blames the wrong party, and
#: silently reclassifying them hides that the evaluator needs fixing.
JUDGE_ERROR_FAILURES = {
    "unmapped_mention",       # no `numbers` entry returned for the mention
    "unmapped_but_present",   # not mapped, yet the value is in the evidence
    "imprecise_path",         # mapped to a container instead of the value
    "not_a_measurement",      # path landed on prose rather than a number
    "evidence_id_unknown",    # cited an evidence id that does not exist
    "no_evidence_cited",      # no provenance offered at all
    "unknown_operator",       # derivation named an operation we cannot apply
    "ambiguous_path",         # the value occurs at several leaves; cannot pick
    "ambiguous_scalar",       # tool returned several numbers in one unaddressable string
    "relation_misencoded",    # a part-whole encoded as an equality
    "wrong_evidence_cited",   # value is in another tool result the claim cited
    "relation_not_supplied",  # a checkable relation the judge did not check
    "relation_not_located",   # a relation SIDE did not resolve — the judge's
                              # pointers, not the claim's figures. Kept apart
                              # from `not_located` so a fumbled relation cannot
                              # block a claim whose every number resolved.
    # An agentic verifier that searched the evidence and still could not
    # produce a citation Python can re-check. Its own failure to conclude, and
    # the same rule applies: unknown, never charged to the answer.
    "agent_unverifiable",     # searched, and would not commit to a reading
    "agent_citation_unfound", # cited a snippet that is not in that evidence
    "agent_value_absent",     # the figure is not in the snippet it cited
    # A period, not a quantity: "mid-2024", "2025-02 to 2025-05". The numeric
    # parser turns these into nonsense (-2024.0) and then reports the nonsense
    # as unlocatable. Whether the period is right is a real question, but not
    # one the NUMERIC trace can answer, so it is excluded rather than charged.
    "not_a_quantity",
    # A constant the ANSWER supplied — "the risky threshold of 20". Tool output
    # holds measurements, not the thresholds they are judged against, so this
    # trace can never locate one and reported that as an invented figure.
    "stated_constant",
}

#: Failures that ARE evidence about the system.
SYSTEM_FAILURES = {
    "not_located",            # value in no tool output, derivable from none
    "value_mismatch",         # located, and it disagrees
    "relation_does_not_hold",
    "not_tool_output",
    "secondary_source",
}


def _find_value_paths(
    node: Any, expected: float, tolerance: float, *, limit: int = 4,
) -> list[str]:
    """Every leaf path under `node` whose value equals `expected`.

    Used to repair a pointer, not to search for evidence: the judge already
    named the evidence and the region: this only walks the node it pointed at.
    Stops early once ambiguity is proven, because a repair is only safe when
    exactly one leaf matches — picking among several would be guessing which
    number the claim meant.
    """
    matches: list[str] = []
    stack: list[tuple[Any, str]] = [(node, "")]
    seen = 0
    while stack and seen < 5000 and len(matches) < limit:
        current, path = stack.pop()
        seen += 1
        if isinstance(current, dict):
            for key, value in current.items():
                stack.append((value, f"{path}.{key}" if path else str(key)))
            continue
        if isinstance(current, list):
            for index, value in enumerate(current):
                stack.append((value, f"{path}[{index}]"))
            continue
        value = _evidence_float(current)
        if value is not None and math.isclose(
            value, expected, rel_tol=1e-12, abs_tol=max(tolerance, 1e-9),
        ):
            matches.append(path)
            continue
        if isinstance(current, str):
            # A categorical code carries a real figure: an account status of
            # "30 DPB" is what "30 days past due" reads. It is not a
            # measurement `_evidence_float` will parse, but the value is there
            # and the claim is right.
            for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", current):
                parsed = _evidence_float(token)
                if parsed is not None and math.isclose(
                    parsed, expected, rel_tol=1e-12, abs_tol=max(tolerance, 1e-9),
                ):
                    matches.append(path)
                    break
    return matches


def _value_is_present(node: Any, expected: float, tolerance: float) -> bool:
    """Does `expected` occur anywhere inside this tool result?

    Used only to answer "did the system invent this number?", never to verify
    it. A judge that points at `$.result` instead of `$.result.count`, or at
    the container holding the series, has addressed the evidence imprecisely —
    the measurement is still in the tool output, and calling that a
    hallucination sends the reader hunting for a fabrication that is not there.
    """
    stack = [node]
    seen = 0
    while stack and seen < 5000:
        current = stack.pop()
        seen += 1
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        else:
            value = _evidence_float(current)
            if value is not None and math.isclose(
                value, expected, rel_tol=1e-12, abs_tol=max(tolerance, 1e-9),
            ):
                return True
            if isinstance(current, str):
                # Token-match EVEN WHEN the string parsed: a tool sentence
                # states one value after the `=` and others in its aside, and
                # stopping at the parsed one misses every number but the first.
                # Matched as whole tokens so "357" does not match "3570".
                for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", current):
                    parsed = _evidence_float(token)
                    if parsed is not None and math.isclose(
                        parsed, expected, rel_tol=1e-12,
                        abs_tol=max(tolerance, 1e-9),
                    ):
                        return True
    return False


def _normalize_fact_results(
    claims: list[dict[str, Any]], raw_results: list[Any], ledger: list[dict[str, Any]],
    claim_traces: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_claim = {
        str(item.get("claim_id")): item for item in raw_results if isinstance(item, dict)
    }
    evidence_by_id = {str(item.get("evidence_id")): item for item in ledger}
    # A judge working from an older ledger, or from the alias printed inside a
    # merged entry, must still resolve to the surviving entry.
    for item in ledger:
        for alias in _as_list(item.get("duplicate_evidence_ids")):
            evidence_by_id.setdefault(str(alias), item)
    tool_provenance_available = any(
        item.get("source_type") in MEASURED_SOURCES for item in ledger
    )
    out = []
    for claim in claims:
        if not claim["is_factual"]:
            continue
        raw = by_claim.get(claim["claim_id"], {})
        # Only MATERIAL numbers make a claim quantitative. A claim whose sole
        # figure is a metric label ("30+ DPD") or a quoted figure the answer
        # disowns asserts no measurement of its own, and pulling it into the
        # numeric funnel leaves it permanently "unavailable" there.
        material_mentions = [
            mention for mention in claim.get("numeric_mentions") or []
            if mention.get("material", True)
        ]
        # A claim about what a report says asserts no measurement of its own —
        # every figure in it is quoted — so `claim_type: quantitative_fact`
        # must not drag it into the numeric denominators as `unavailable`.
        # Driven by the CLAIM, never by what the judge returned. Keying off a
        # `relations` array the judge volunteered let it decide which funnel a
        # claim entered: a purely qualitative claim ("CDSS stayed well below
        # its trigger") acquired a relation, failed to resolve it, and was
        # reported as an invented number.
        is_quantitative = not claim.get("report_attribution") and (
            bool(material_mentions) or claim.get("claim_type") in {
                "quantitative_fact", "comparison",
            } or claim.get("claim_type") in _RELATIONAL_CLAIM_TYPES
        )
        verdict = _slug(raw.get("verdict") or raw.get("factual_verdict"))
        if verdict not in FACT_VERDICTS:
            verdict = "unverifiable"
        claim_trace = (claim_traces or {}).get(claim["claim_id"]) or {}

        number_results = []
        raw_numbers = _as_list(raw.get("numbers"))
        # Every tool result this claim cited, for widening a pointer search.
        # Still the claim's own provenance — never a free hunt through the run.
        claim_tool_payloads = [
            (str(eid), evidence_by_id[str(eid)].get("result"))
            for eid in {
                *(str(v) for v in _as_list(raw.get("evidence_ids"))),
                *(str(n.get("evidence_id")) for n in raw_numbers
                  if isinstance(n, dict) and n.get("evidence_id")),
            }
            if str(eid) in evidence_by_id
            and evidence_by_id[str(eid)].get("source_type") in MEASURED_SOURCES
        ]
        for mention_index, mention in enumerate(claim["numeric_mentions"]):
            if not mention.get("material", True):
                continue
            link = next((
                item for item in raw_numbers if isinstance(item, dict)
                and str(item.get("written_value") or item.get("written") or "")
                == str(mention.get("written") or "")
            ), None)
            if link is None:
                # No positional fallback. Matching mention #n to raw_numbers[n]
                # assumes the judge emitted its numbers in answer order and
                # skipped none — when that assumption breaks, a number gets
                # compared against an unrelated evidence value and the claim is
                # reported as contradicted on the strength of a coincidence
                # (observed: "$1,000" checked against 2.0, "26.1%" against
                # 0.03). An unmatched mention is UNKNOWN, so leave it
                # unresolved and let it surface as "unavailable".
                link = {}
            evidence_id = str(link.get("evidence_id") or "")
            evidence = evidence_by_id.get(evidence_id)
            trace_kind = _slug(link.get("trace_kind") or "direct")
            expected = mention.get("value")
            computed = None
            resolved = False
            source_type = evidence.get("source_type") if evidence else None
            trace_source_types: list[str | None] = []
            operand_evidence_ids: list[str] = []
            operation = _slug((link.get("calculation") or {}).get("operation"))
            located = False
            unparseable = False
            node_is_container = False
            resolved_node = None
            try:
                if trace_kind == "derived":
                    calculation = link.get("calculation") or {}
                    operands: list[float] | None = []
                    for operand in _as_list(calculation.get("operands")):
                        operand_id = str(operand["evidence_id"])
                        operand_evidence_ids.append(operand_id)
                        operand_ev = evidence_by_id[operand_id]
                        trace_source_types.append(operand_ev.get("source_type"))
                        values = _operand_values(
                            _resolve_path(
                                operand_ev.get("result"), str(operand["json_path"]),
                            ),
                            operand.get("select") or operand.get("field"),
                        )
                        located = True
                        if values is None or operands is None:
                            operands = None
                            unparseable = True
                            continue
                        operands.extend(values)
                    if operands:
                        computed = _operation(operation, operands)
                        resolved = computed is not None
                elif evidence is not None and link.get("json_path") is not None:
                    # `""` addresses the root: correct when the tool's entire
                    # result IS the value. Testing truthiness skipped the
                    # resolver entirely and reported the number as unlocatable.
                    trace_source_types.append(source_type)
                    payload = evidence.get("result")
                    if isinstance(payload, (dict, list)):
                        node = _resolve_path(payload, str(link["json_path"]))
                    else:
                        # A scalar result holds exactly one value, so any path
                        # into it is a judge artifact ("$.result" on a string).
                        # Honouring it raised KeyError and reported a number
                        # that IS the tool's answer as unlocatable.
                        node = payload
                    located = True
                    # A pointer that lands on anything other than a bare
                    # number has not addressed a value. A dict or a list is
                    # obvious; so is a STRING holding several figures
                    # ("top: unpaid 0.81, debt service -0.34, …"), where the
                    # resolver reads whichever number comes first. In both
                    # cases the figure read is the judge's aim, not the
                    # claim's, and a disagreement says nothing about the answer.
                    node_is_container = isinstance(node, (dict, list))
                    resolved_node = node
                    computed = _evidence_float(node)
                    resolved = computed is not None
                    unparseable = not resolved
            except (KeyError, TypeError, ValueError):
                resolved = False
            # Preserve an explicit domain/rounding tolerance, while always
            # allowing the tiny representation error introduced by binary
            # floating-point arithmetic (for example, 0.72 - 0.61).
            requested_tolerance = _safe_float(link.get("tolerance"))
            # The answer's own precision is a floor: "$404K" is satisfied by
            # $404,151.99, and demanding equality reported ten correct rounded
            # figures as mismatches in one run.
            tolerance = max(
                abs(requested_tolerance or 0.0),
                _written_tolerance(mention.get("written")),
                1e-12,
            )
            written = mention.get("written")
            # "all above 720" is a bound, not an equality; read as `==` it was
            # a mismatch against the 721 that proves it.
            measures = mention.get("measures")
            if _is_period_expression(written, measures) or _is_stated_constant(measures):
                excluded = (
                    "not_a_quantity" if _is_period_expression(written, measures)
                    else "stated_constant"
                )
                # Decided before anything is parsed: reading "mid-2024" as a
                # number is what produced the bad verdict in the first place.
                number_results.append({
                    "written_value": mention.get("written"),
                    "measures": mention.get("measures"),
                    "json_path": link.get("json_path"),
                    "answer_value": None,
                    "evidence_value": None,
                    "trace_kind": trace_kind,
                    "located_in_tool_output": False,
                    "traceable_to_tool_output": False,
                    "grounded_in_tool_result": False,
                    "deterministically_correct": None,
                    "trace_failure": excluded,
                })
                continue
            comparator = _infer_comparator(
                written, str(mention.get("comparator") or "=="),
            )
            correct = (
                resolved and expected is not None and computed is not None
                and _satisfies(comparator, expected, computed, tolerance)
            )
            # Only after the exact reading fails: the same figure written on a
            # different scale ("36%" vs 0.3603) or without its sign ("declined
            # by 2.2%" vs -0.022). A real disagreement survives every variant.
            if not correct and resolved and expected is not None and computed is not None:
                correct = any(
                    _satisfies(comparator, want, got, tol)
                    for want, got, tol in _comparison_variants(
                        written, expected, computed, tolerance,
                    )
                )
            # "Traceability" means an actual system tool result, not a canonical
            # fact or a specialist's prose summary.
            traceable_to_tool = bool(
                resolved and correct and trace_source_types
                and all(kind in MEASURED_SOURCES for kind in trace_source_types)
            )
            # Whether the comparison ran against a real system tool result,
            # INDEPENDENT of whether it agreed. `traceable_to_tool_output`
            # folds in `correct`, so it cannot answer the question that
            # actually matters here: "is this MISMATCH trustworthy?" A
            # disagreement with a specialist's prose summary is not grounds to
            # overrule the semantic judge.
            grounded_in_tool_result = bool(
                resolved and trace_source_types
                and all(kind in MEASURED_SOURCES for kind in trace_source_types)
            )
            # Decompose the "not traceable" bucket. Only `not_located` is the
            # hallucination case: the value is in no tool output and derives
            # from none. A located-but-mismatched value is a derivation or
            # transcription error, and an unmapped mention is a judge failure —
            # three different defects with three different fixes.
            if traceable_to_tool:
                trace_failure = None
            elif not link:
                trace_failure = "unmapped_mention"
            elif not (evidence_id or operand_evidence_ids):
                trace_failure = "no_evidence_cited"
            elif evidence is None and not operand_evidence_ids:
                trace_failure = "evidence_id_unknown"
            elif not located:
                trace_failure = "not_located"
            elif unparseable:
                trace_failure = "not_a_measurement"
            # Re-read below: either may be a bad POINTER rather than invention.
            elif expected is None:
                trace_failure = "claim_value_missing"
            elif not correct:
                trace_failure = "value_mismatch"
            elif not grounded_in_tool_result:
                trace_failure = "not_tool_output"
            else:
                trace_failure = "unresolved"
            repaired_path = None
            if (
                # A mismatch read out of a CONTAINER is a pointer aimed at a
                # region, not a wrong number: "drivers" holds several values
                # and the resolver returned whichever came first. Search the
                # region for the figure the claim actually states before
                # charging the answer for the judge's aim.
                (trace_failure in {"not_located", "not_a_measurement"}
                 or (trace_failure == "value_mismatch" and node_is_container))
                and evidence is not None and expected is not None
            ):
                # The judge named the right evidence and the wrong depth. Walk
                # the node it pointed at: one match is a pointer we can fix,
                # several is a guess we should not make.
                try:
                    region = _resolve_path(
                        evidence.get("result"), str(link.get("json_path") or ""),
                    )
                except (KeyError, TypeError, ValueError):
                    region = evidence.get("result")
                found = _find_value_paths(region, expected, tolerance)
                if not found and region is not evidence.get("result"):
                    # The pointer resolved, but to a region without the value.
                    # Widen to the rest of the SAME evidence entry before
                    # concluding the number is absent.
                    found = _find_value_paths(
                        evidence.get("result"), expected, tolerance,
                    )
                if len(found) == 1 and evidence.get("source_type") in MEASURED_SOURCES:
                    repaired_path = found[0]
                    computed = expected
                    resolved = True
                    correct = True
                    located = True
                    trace_source_types = [evidence.get("source_type")]
                    trace_failure = None
                elif found and evidence.get("source_type") in MEASURED_SOURCES:
                    trace_failure = "ambiguous_path"
                elif (elsewhere := [
                    (eid, paths)
                    for eid, payload in claim_tool_payloads
                    if eid != evidence_id
                    and (paths := _find_value_paths(payload, expected, tolerance))
                ]):
                    # In a tool result the claim cited, just not the one this
                    # number was attached to. The judge picked the wrong
                    # evidence — the measurement exists. Checked BEFORE
                    # `secondary_source`: a value present in both a specialist's
                    # prose and a real tool result was being reported as resting
                    # on the prose, purely because the judge cited that one.
                    #
                    # Repoint when exactly ONE cited entry holds it at exactly
                    # one leaf. That is the same discipline as the repair
                    # above — unambiguous, and never outside the provenance the
                    # claim itself offered — and it is not a free hunt through
                    # the run. Anything less certain stays disclosed.
                    if len(elsewhere) == 1 and len(elsewhere[0][1]) == 1:
                        evidence_id, repaired_path = elsewhere[0][0], elsewhere[0][1][0]
                        evidence = evidence_by_id[evidence_id]
                        computed = expected
                        resolved = correct = located = True
                        trace_source_types = [evidence.get("source_type")]
                        trace_failure = None
                    else:
                        trace_failure = "wrong_evidence_cited"
                elif found:
                    # Present only in a non-tool source the claim cited.
                    trace_failure = "secondary_source"
                else:
                    # Searched every tool result the claim cited and the value
                    # is in none of them. That is a statement about the ANSWER,
                    # so it stops being an evaluator failure here.
                    trace_failure = "not_located"
            if (
                trace_failure == "value_mismatch"
                and evidence is not None and expected is not None
                and isinstance(resolved_node, str)
                and _value_is_present(resolved_node, expected, tolerance)
            ):
                # The tool packed several measurements into one sentence:
                # "sum(...) = $174,897.36 (over 1 non-null value(s) in 1
                # matching row(s); 3 total)". A path can address the string but
                # not a number inside it, so Python reads the assignment value
                # and any other figure the claim meant looks like a mismatch.
                # The claim is not wrong; the evidence is unaddressable.
                trace_failure = "ambiguous_scalar"
            if repaired_path is not None:
                traceable_to_tool = True
                grounded_in_tool_result = True
            number_results.append({
                "written_value": mention.get("written"),
                # What the answer says this number quantifies, carried through
                # so a mismatch can be read: "1" against a balance sum is a
                # mis-mapping, not a wrong count.
                "measures": mention.get("measures"),
                # Kept separately so a reviewer can audit the repair rather
                # than trust it: the judge's pointer stays in `json_path`.
                "resolved_json_path": repaired_path,
                "path_repaired": repaired_path is not None,
                "answer_value": expected,
                "comparator": comparator,
                "evidence_id": evidence_id or None,
                "operand_evidence_ids": operand_evidence_ids,
                "json_path": link.get("json_path"),
                "trace_kind": trace_kind,
                "operation": operation or None,
                "evidence_value": computed,
                "tolerance": tolerance,
                "located_in_tool_output": bool(located and grounded_in_tool_result),
                "traceable_to_tool_output": traceable_to_tool,
                "grounded_in_tool_result": grounded_in_tool_result,
                # Unknown on EITHER side is unresolved, never wrong. `resolved`
                # speaks only for the evidence side; if the claim side has no
                # value, there is nothing to disagree with, and reporting False
                # here contradicts a claim on the strength of a missing field.
                "deterministically_correct": (
                    correct if resolved and expected is not None else None
                ),
                "trace_failure": trace_failure,
            })

        # Relational claims: a threshold or comparison asserts something
        # checkable even with no number of its own, and skipping it means the
        # cascade silently declines to verify a statement it CAN verify.
        relation_results = []
        for relation in _as_list(raw.get("relations")):
            if not isinstance(relation, dict):
                continue
            operator = str(relation.get("operator") or "").strip()
            left, left_source, left_id, left_imprecise = _resolve_relation_side(
                relation.get("left"), evidence_by_id,
            )
            right, right_source, right_id, right_imprecise = _resolve_relation_side(
                relation.get("right"), evidence_by_id,
            )
            check = _RELATION_OPERATORS.get(operator)
            holds = (
                check(left, right)
                if check and left is not None and right is not None else None
            )
            # Both sides must come from real measurements. A threshold the
            # judge supplied from memory proves nothing about the system.
            grounded = {left_source, right_source} <= set(MEASURED_SOURCES)
            misencoded = (
                operator == "=="
                and left is not None and right is not None
                and not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)
                and {left, right} <= {
                    mention.get("value") for mention in claim["numeric_mentions"]
                    if mention.get("material")
                }
            )
            if holds is None:
                failure = (
                    "unknown_operator" if not check
                    # A side that resolved to a container is a pointer the
                    # judge aimed at a region, not a value the answer invented.
                    else "imprecise_path" if left_imprecise or right_imprecise
                    else "relation_not_located"
                )
            elif misencoded:
                # "13 of 18 months" is a part-whole. Encoded as `13 == 18` it
                # is trivially false, and the claim gets charged for the
                # judge's choice of operator. Both figures are separately
                # stated by the claim and separately traced.
                failure = "relation_misencoded"
            elif not grounded:
                failure = "not_tool_output"
            elif not holds:
                failure = "relation_does_not_hold"
            else:
                failure = None
            relation_results.append({
                "operator": operator,
                "left_value": left, "right_value": right,
                "left_evidence_id": left_id, "right_evidence_id": right_id,
                "left_json_path": (relation.get("left") or {}).get("json_path")
                if isinstance(relation.get("left"), dict) else None,
                "right_json_path": (relation.get("right") or {}).get("json_path")
                if isinstance(relation.get("right"), dict) else None,
                "holds": holds,
                "grounded_in_tool_result": grounded,
                "traceable_to_tool_output": bool(holds and grounded),
                "trace_failure": failure,
                "reason": str(relation.get("reason") or ""),
            })
        relation_evidence_ids = {
            str(value)
            for relation in relation_results
            for value in (
                relation.get("left_evidence_id"), relation.get("right_evidence_id"),
            ) if value
        }
        # A claim that asserts a relation but supplies none is UNKNOWN, not
        # exempt: the judge declined to check something it could have.
        relation_expected = (
            claim.get("claim_type") in _RELATIONAL_CLAIM_TYPES
            and not material_mentions and not relation_results
        )

        # A mention the judge never mapped says nothing about whether the value
        # exists: it says the judge did not answer. Search the evidence the
        # claim DID cite before calling it invention — one unmapped mention
        # otherwise poisons a claim whose every other number traced cleanly.
        cited_results = [
            evidence_by_id[str(number["evidence_id"])].get("result")
            for number in number_results
            if number.get("evidence_id")
            and str(number["evidence_id"]) in evidence_by_id
        ] + [
            evidence_by_id[str(value)].get("result")
            for value in _as_list(raw.get("evidence_ids"))
            if str(value) in evidence_by_id
            and evidence_by_id[str(value)].get("source_type") in MEASURED_SOURCES
        ]
        for number in number_results:
            if number.get("trace_failure") != "unmapped_mention":
                continue
            expected_value = number.get("answer_value")
            if expected_value is None:
                continue
            if any(
                _value_is_present(payload, expected_value, number.get("tolerance") or 0)
                for payload in cited_results
            ):
                number["trace_failure"] = "unmapped_but_present"

        # Only a mismatch against a genuine tool result may overrule the
        # semantic judge. Without that gate, an unlinked or mis-resolved
        # comparison flips a well-evidenced claim to "contradicted" — which
        # produced self-contradicting records like "Deterministic numeric check
        # disagrees … The evidence explicitly states 0 out of 357 payments were
        # returned, directly supporting the claim."
        deterministic_mismatch = any(
            number.get("deterministically_correct") is False
            and number.get("grounded_in_tool_result")
            for number in number_results
        ) or any(
            relation.get("holds") is False
            and relation.get("grounded_in_tool_result")
            # A part-whole the judge wrote as an equality is false by
            # construction; letting it contradict the claim charges the answer
            # for the judge's choice of operator.
            and relation.get("trace_failure") == "relation_does_not_hold"
            for relation in relation_results
        )
        if deterministic_mismatch:
            verdict = "contradicted"
            reason = (
                "Deterministic numeric check disagrees with the linked evidence. "
                + str(raw.get("reason") or "")
            ).strip()
        else:
            reason = str(raw.get("reason") or "No judge reason supplied.")
        # Does the claim's own evidence CONTRADICT it?
        #
        # Not "did every pointer resolve". The pointer is the judge's aim, and
        # it is the least reliable thing in this pipeline: across the six
        # questions, 12 of 20 evaluator failures were `unmapped_but_present` —
        # the judge returned no mapping for a figure that Python then found
        # sitting in evidence the claim itself cited. Gating grounding on that
        # scored the evaluator's aim as the answer's provenance.
        #
        # So judge-side pointer failures are DISCLOSED and do not block. What
        # still blocks is a figure that resolved against a real measurement and
        # disagreed with it, which is evidence about the answer and nothing
        # else. Grounding rests on the audited ROUTE below; this asks only
        # whether anything measured refutes the claim.
        number_failures = {
            item["trace_failure"] for item in number_results
            if item.get("trace_failure")
        }
        relation_failures = {
            item["trace_failure"] for item in relation_results
            if item.get("trace_failure")
        }
        if not is_quantitative:
            traced = "not_applicable"
        elif not tool_provenance_available:
            traced = "unavailable"
        elif (number_failures | relation_failures) & SYSTEM_FAILURES:
            traced = "no"
        else:
            traced = "yes"

        tool_evidence_ids = {
            str(v) for v in _as_list(
                raw.get("trace_evidence_ids")
                or raw.get("tool_usage_evidence_ids")  # pre-rename field name
                or raw.get("evidence_ids")
            )
        }
        tool_evidence_ids.update(
            str(number.get("evidence_id")) for number in number_results
            if number.get("evidence_id")
        )
        for number in number_results:
            tool_evidence_ids.update(number.get("operand_evidence_ids") or [])
        tool_evidence_ids.update(relation_evidence_ids)
        # Grounding for EVERY factual claim, numeric or not. A qualitative
        # claim ("delinquency drove the late spike") carries no number for the
        # numeric funnel to check, so without this it is scored on the judge's
        # verdict alone — and a claim lifted from a specialist's prose findings
        # is indistinguishable from one read off a real measurement.
        cited_evidence_ids = {
            str(v) for v in _as_list(raw.get("evidence_ids"))
        } | tool_evidence_ids
        grounding_evidence = [
            evidence_by_id[evidence_id] for evidence_id in sorted(cited_evidence_ids)
            if evidence_id in evidence_by_id
        ]
        # Report-supported when the claim RELAYS what a report said, or when
        # nothing but report material backs it. `all`, not `any`: a claim the
        # live data also confirms is corroborated by the report, not dependent
        # on it, and should keep counting as factual support.
        report_supported = bool(claim.get("report_attribution")) or bool(
            grounding_evidence
            and all(_is_report_evidence(item) for item in grounding_evidence)
        )
        # ------------------------------------------------------------------
        # THE THREE OUTCOMES.
        #
        #   report   the claim relays curated report material, and that
        #            material resolves. Checked FIRST and short-circuits: a
        #            claim quoting a report ran no operations, so putting it
        #            through the factual test only ever produced a second,
        #            noisier way of saying "not measured".
        #   factual  the run recorded a route that produced it — an operation
        #            this turn, or a measured source carried forward as memory
        #            — and that route answers the question asked.
        #   none     neither.
        # ------------------------------------------------------------------
        resolution = (
            "none" if not cited_evidence_ids
            # Every cited id is absent from the ledger: the judge pointed at
            # provenance that does not exist. Unknown, not ungrounded.
            else "unresolved" if not grounding_evidence
            else "resolved"
        )
        report_grounded = report_supported and resolution == "resolved"
        # The walk: claim -> (specialist findings ->) tables and operations. A
        # recorded call is the walk itself; evidence naming the operations
        # behind it completes the same walk one link further back.
        reaches_operations = any(
            _reaches_operations(item) for item in grounding_evidence
        )
        # A route the run recorded: an operation it ran this turn, a memory
        # topic it wrote when the measurement was made, or evidence that names
        # the operations behind it. Any of the three completes the walk
        # claim -> specialist -> tables and operations.
        has_route = bool(
            claim_trace.get("call_ids") or claim_trace.get("memory_topics")
        ) or reaches_operations
        eligible = _slug(claim_trace.get("eligible") or "unavailable")
        if eligible not in {"yes", "no"}:
            eligible = "unavailable"
        if (
            eligible == "unavailable"
            and claim_trace.get("no_recorded_operation")
            and reaches_operations
        ):
            # No call_id in THIS turn, but the claim rests on a measured source
            # — a specialist KB entry written when the measurement was made,
            # one or more turns ago. That is a route, and the judge ruled on
            # it; discarding its verdict scored a memory-grounded claim as
            # unassessable purely because the operation predates this turn.
            eligible = _slug(claim_trace.get("ruled_eligible") or "unavailable")
            if eligible not in {"yes", "no"}:
                eligible = "unavailable"
        if report_grounded:
            grounding_kind = "report"
            traced = "not_applicable"
        elif eligible == "yes" and has_route and traced != "no":
            grounding_kind = "factual"
        else:
            grounding_kind = "none"
        failures = sorted(
            (number_failures | relation_failures)
            | ({"relation_not_supplied"} if relation_expected else set())
        )
        out.append({
            "claim_id": claim["claim_id"],
            # The marker on the page and the grounding metrics are both counted
            # straight off this field, so the two cannot drift apart.
            "grounding_kind": grounding_kind,
            "eligible": eligible,
            "eligibility_reason": str(claim_trace.get("eligibility_reason") or ""),
            "verdict": verdict,
            "reason": reason,
            "traced": traced,
            "evidence_resolution": resolution,
            "evidence_ids": sorted(cited_evidence_ids),
            "numbers": number_results,
            "relations": relation_results,
            # How the claim was produced, read off the run's own trace.
            "route": claim_trace or None,
            "judge_error": bool(failures and set(failures) & JUDGE_ERROR_FAILURES),
            "failures": failures,
        })
    return out
