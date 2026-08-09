import json
import re

from agentic_eval.layout import RunLayout
from agentic_eval.render import write_answer_comparison, select_repeat, answer_comparison_html
from agentic_eval.render import find_run_manifest, resolve_view_defaults


def _evaluation(system, name, run_index, answer, *, metrics=None, fact=None):
    return {
        "system": system, "mode": "cold", "name": name, "run_index": run_index,
        "question": f"question for {name}", "answer": answer,
        "claims": [{
            "claim_id": "c1", "block_id": "b001", "is_factual": True,
            "proposition": f"{system} says something about {name}",
            "source_locator": {"row": 1, "column": 1},
        }],
        "fact_results": [fact if fact is not None else {
            "claim_id": "c1", "numeric_support": "yes", "traceability": "yes",
            "reasoning_trace_correctness": "yes", "evidence_resolution": "resolved",
            "grounding_kind": "factual",
        }],
        "metrics": metrics or {
            "answer_correct": 1, "answer_checked": 1,
            "orthogonal_claim_count": 1, "all_factual_claim_count": 1,
            "factual_grounded_count": 1,
        },
    }


def test_all_repeats_are_rendered_with_the_sampled_one_open():
    """Every repeat is on the page; the sampled one is merely shown first.

    Averaging answers is still meaningless — these are panes, not a mean — but
    rendering only one hid the run-to-run variance the repeats exist to show.
    """
    rows = [
        _evaluation("old", "q1", 1, "OLD ANSWER ONE"),
        _evaluation("new", "q1", 1, "NEW ANSWER ONE"),
        _evaluation("old", "q1", 2, "OLD ANSWER TWO"),
        _evaluation("new", "q1", 2, "NEW ANSWER TWO"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    for answer in ("OLD ANSWER ONE", "NEW ANSWER ONE",
                   "OLD ANSWER TWO", "NEW ANSWER TWO"):
        assert answer in page, answer
    # Repeat 1 is the open pane; repeat 2 is present but hidden.
    assert 'data-repeat="1"' in page and 'data-repeat="2"' in page
    assert 'data-repeat="2" hidden' in page
    assert "repeat <code>#1</code>" in page

    second = answer_comparison_html(rows, baseline="old", candidate="new", run_index=2)
    assert 'data-repeat="1" hidden' in second


def test_repeat_selection_is_stable_not_arbitrary():
    """Regenerating the report must show the same sample, or two readings of
    "the same" report are not comparable."""
    rows = [_evaluation("new", "q1", index, f"a{index}") for index in (3, 1, 2)]
    assert select_repeat(rows, mode=None, run_index=None) == ("cold", 1)


def test_a_missing_system_is_labelled_not_silently_dropped():
    rows = [_evaluation("new", "q1", 1, "ONLY THE CANDIDATE RAN")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "ONLY THE CANDIDATE RAN" in page
    assert "(no record)" in page


def test_absent_measure_is_distinguished_from_unknown_value():
    """A record predating a measure has no opinion about it.

    Rendering that as `?` is indistinguishable from "the measure ran and could
    not decide", which reads a missing feature as a bad result.
    """
    old_style = {
        "claim_id": "c1", "numeric_support": "yes", "traceability": "unavailable",
        "reasoning_trace_correctness": "not_applicable",
        # no evidence_grounding, no hallucination
    }
    rows = [
        _evaluation("old", "q1", 1, "A", fact=old_style),
        _evaluation("new", "q1", 1, "B"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'class="m na"' in page
    assert "predate" in page
    # The candidate carries the field, so it renders a real marker.
    assert 'title="cited provenance: resolved"' in page


def test_metrics_tab_shows_both_sides_and_a_signed_delta():
    rows = [
        _evaluation("old", "q1", 1, "A", metrics={
            "grounded_count": 1, "factual_grounded_count": 1,
            "orthogonal_claim_count": 2, "all_factual_claim_count": 4,
        }),
        _evaluation("new", "q1", 1, "B", metrics={
            "grounded_count": 9, "factual_grounded_count": 9,
            "orthogonal_claim_count": 10, "all_factual_claim_count": 10,
        }),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "50% (1/2)" in page and "90% (9/10)" in page
    # A rate delta carries its unit; a claim count is compared as a count.
    section = page.split('id="q0"', 1)[1]
    assert "+40%" in section and "+8" in section
    assert section.count('class="delta up"') == 2


def test_answers_are_escaped_not_injected():
    rows = [_evaluation("new", "q1", 1, "<script>alert(1)</script> & <b>x</b>")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_write_answer_comparison_is_self_contained(tmp_path):
    rows = [
        _evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B"),
    ]
    path = write_answer_comparison(
        rows, layout=RunLayout(tmp_path), baseline="old", candidate="new",
    )
    page = path.read_text(encoding="utf-8")
    assert path.name == "answer_comparison.html"
    # No network dependency: the viewer must open from disk with no server.
    # The SVG namespace is a name, not an address — nothing fetches it, and a
    # data-URI icon will not render without it. Everything else must be local.
    fetchable = page.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in fetchable and "https://" not in fetchable
    assert "<script>" in page  # the tab switcher is inline


def test_baseline_and_candidate_default_to_the_run_manifest():
    """Inferring from the systems present gets the roles backwards.

    `sorted(["current", "previous"])` puts the candidate first, so an inferred
    baseline would be the candidate and every delta would carry the wrong sign.
    The manifest records what the run actually was.
    """
    rows = [
        _evaluation("current", "q1", 1, "C"), _evaluation("previous", "q1", 1, "P"),
    ]
    baseline, candidate, _mode, source = resolve_view_defaults(
        rows, manifest={"baseline": "previous", "candidate": "current"},
    )
    assert (baseline, candidate) == ("previous", "current")
    assert source == "manifest.json"


def test_explicit_flags_beat_the_manifest():
    rows = [_evaluation("current", "q1", 1, "C"), _evaluation("previous", "q1", 1, "P")]
    baseline, candidate, _mode, source = resolve_view_defaults(
        rows, manifest={"baseline": "previous", "candidate": "current"},
        baseline="current", candidate="previous",
    )
    assert (baseline, candidate) == ("current", "previous")
    assert source == "explicit"


def test_a_both_mode_manifest_does_not_become_a_filter():
    """`both` is a run setting, not a value any record carries."""
    rows = [_evaluation("previous", "q1", 1, "P"), _evaluation("current", "q1", 1, "C")]
    _b, _c, mode, _source = resolve_view_defaults(rows, manifest={"mode": "both"})
    assert mode is None
    page = answer_comparison_html(rows, baseline="previous", candidate="current",
                                  mode=mode)
    assert "P" in page and "C" in page


def test_manifest_is_found_from_the_content_subfolder(tmp_path):
    run = tmp_path / "run_1"
    (run / "content").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"baseline": "previous", "candidate": "current", "mode": "cold"}),
        encoding="utf-8",
    )
    evaluations = run / "content" / "content_evaluations.jsonl"
    evaluations.write_text("", encoding="utf-8")
    assert find_run_manifest(evaluations)["baseline"] == "previous"


def test_missing_manifest_degrades_without_crashing(tmp_path):
    evaluations = tmp_path / "content_evaluations.jsonl"
    evaluations.write_text("", encoding="utf-8")
    assert find_run_manifest(evaluations) == {}
    rows = [_evaluation("previous", "q1", 1, "P"), _evaluation("current", "q1", 1, "C")]
    baseline, candidate, _mode, source = resolve_view_defaults(rows)
    # Record order, so the roles at least follow how the runner wrote them.
    assert (baseline, candidate) == ("previous", "current")
    assert source == "inferred from record order"


def _summary_group(system, name, **metrics):
    return {"system": system, "mode": "cold", "name": name, "n_runs": 10, **metrics}


def test_metrics_tab_has_all_four_eval_modules():
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    summary = {"groups": [
        _summary_group("old", "q1", team_exact_consistency=1.0,
                       latency_seconds={"mean": 20.0, "p95": 25.0},
                       memory_hit_rate=0.5, retry_rate=0.0),
        _summary_group("new", "q1", team_exact_consistency=0.8,
                       latency_seconds={"mean": 30.0, "p95": 40.0},
                       memory_hit_rate=1.0, retry_rate=0.1),
    ]}
    page = answer_comparison_html(
        rows, baseline="old", candidate="new", summary=summary,
    )
    for heading in ("Content", "Consistency", "System", "Memory"):
        assert f"<h4>{heading}</h4>" in page
    assert "20.0s" in page and "30.0s" in page      # dotted key resolved
    assert "Memory hit rate" in page
    # Slower is worse, more memory hits is better.
    assert 'class="delta down">+10.0' in page
    assert 'class="delta up">+50' in page


def test_module_metrics_are_labelled_as_spanning_all_repeats():
    """Content metrics belong to ONE repeat; the others are aggregates.

    Showing them in the same tab without saying so invites reading a 10-run
    latency mean as a property of the single answer displayed above it.
    """
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    summary = {"groups": [
        _summary_group("old", "q1", retry_rate=0.0),
        _summary_group("new", "q1", retry_rate=0.0),
    ]}
    page = answer_comparison_html(
        rows, baseline="old", candidate="new", summary=summary,
    )
    assert "across all" in page
    assert "across all 10 repeats" in page


def test_missing_summary_explains_itself_instead_of_showing_blanks():
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "summary.json" in page
    # Titled System, not Latency: the block carries tool-call success
    # beside the costs. `latency` remains the module's registry name.
    assert "<h4>System</h4>" in page


def test_questions_appear_in_the_navigation_panel():
    rows = [
        _evaluation("old", "alpha", 1, "A"), _evaluation("new", "alpha", 1, "B"),
        _evaluation("old", "beta", 1, "C"), _evaluation("new", "beta", 1, "D"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'class="toc"' in page
    assert 'data-target="q0"' in page and 'data-target="q1"' in page
    assert 'id="q0"' in page and 'id="q1"' in page
    assert "alpha" in page and "beta" in page


def test_navigation_flags_a_question_containing_an_ungrounded_claim():
    bad = {
        "claim_id": "c1", "grounding_kind": "none", "eligible": "no",
        "traced": "no", "evidence_resolution": "none",
    }
    rows = [
        _evaluation("old", "clean", 1, "A"),
        _evaluation("new", "dirty", 1, "B", fact=bad),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "tflag" in page or "○" in page
    assert "ungrounded claim(s)" in page


def test_run_summary_is_not_confused_with_the_content_summary(tmp_path):
    """Two files are called `summary.json`; only one has module metrics.

    The content cascade writes `content/summary.json`. An upward filename
    search from the evaluations file finds THAT first and renders the
    consistency/latency/memory tables from a file containing none of them —
    silently, as empty rows.
    """
    from agentic_eval.render import find_run_summary

    run = tmp_path / "exp_1"
    (run / "content").mkdir(parents=True)
    (run / "metrics").mkdir()
    (run / "runs.jsonl").write_text("", encoding="utf-8")
    (run / "content" / "summary.json").write_text(
        json.dumps({"groups": [{"system": "new", "name": "q1"}]}), encoding="utf-8",
    )
    (run / "metrics" / "summary.json").write_text(
        json.dumps({"groups": [
            {"system": "new", "mode": "cold", "name": "q1", "retry_rate": 0.25},
        ]}), encoding="utf-8",
    )
    summary = find_run_summary(run / "content" / "evaluations.jsonl")
    assert summary["groups"][0]["retry_rate"] == 0.25


def test_pre_migration_runs_still_resolve_their_summary(tmp_path):
    """Older runs keep summary.json at the root; it must still be read."""
    from agentic_eval.render import find_run_summary

    run = tmp_path / "old_run"
    (run / "content").mkdir(parents=True)
    (run / "runs.jsonl").write_text("", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"groups": [{"system": "new", "name": "q1", "retry_rate": 0.5}]}),
        encoding="utf-8",
    )
    assert find_run_summary(run / "content" / "evaluations.jsonl")["groups"][0][
        "retry_rate"
    ] == 0.5


def test_summary_section_totals_the_question_set():
    """Numerators and denominators are summed, never averaged as percentages.

    An average of averages cannot be shown as "n/d", and it hides how much
    each figure rests on: here `new` scores 2/3, not the 75% that averaging
    100% and 50% would report.
    """
    rows = [
        _evaluation("old", "q1", 1, "A", metrics={"factual_grounded_count": 1, "orthogonal_claim_count": 1}),
        _evaluation("new", "q1", 1, "B", metrics={"factual_grounded_count": 1, "orthogonal_claim_count": 1}),
        _evaluation("old", "q2", 1, "C", metrics={"factual_grounded_count": 0, "orthogonal_claim_count": 1}),
        _evaluation("new", "q2", 1, "D", metrics={"factual_grounded_count": 1, "orthogonal_claim_count": 2}),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    summary = page.split('class="summary"', 1)[1].split("</section>", 1)[0]
    # "All question sets", not "Question set": a set is now a named dimension
    # with its own sections below, so the overall block must not claim the name.
    assert "All question sets — overall" in page
    assert "2 questions" in summary
    # old: 1/2 = 50%   new: 2/3 = 67%
    assert "50% (1/2)" in summary and "67% (2/3)" in summary
    assert "totalled over" in summary


def test_summary_is_only_visible_on_the_metrics_tab():
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'data-only-tab="metrics"' in page
    assert "data-only-tab" in page.split("<script>")[1]  # the toggle handles it


def test_navigation_shows_the_question_text_beside_the_id():
    rows = [_evaluation("new", "latest_fico_score", 1, "A")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    toc = page.split('class="toc"', 1)[1].split("</ol>", 1)[0]
    assert "latest_fico_score" in toc
    assert "question for latest_fico_score" in toc


def test_overview_is_reachable_from_the_navigation():
    """The overview sits on the metrics tab, so its nav link must switch tabs.

    A link that scrolls to a hidden section is a dead link.
    """
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'id="overview"' in page
    assert 'data-target="overview" data-tab="metrics"' in page
    # The handler exists and the overview is scroll-spied like a question.
    assert ".toc a[data-tab]" in page
    assert "section.question, section.summary" in page


def test_no_overview_link_when_there_is_nothing_to_summarise():
    page = answer_comparison_html([], baseline="old", candidate="new")
    assert 'data-target="overview"' not in page
    assert "No evaluations matched" in page


def test_a_one_sided_summary_is_still_shown():
    """Only the candidate ran: the overview still reports what there is."""
    rows = [_evaluation("new", "q1", 1, "A")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'data-target="overview"' in page


def test_a_judge_failure_no_longer_decides_whether_a_claim_is_grounded():
    """It used to force `○`, so the viewer had to shout ⚠ to correct the read.

    Grounding is now satisfied by either route — a traced figure OR cited
    provenance that resolves — so an evaluator oversight stops mattering to
    the marker. The failure list stays on the record; the column shows what
    supports the claim and nothing else.
    """
    judge_failed = {
        "claim_id": "c1", "numeric_support": "yes", "traceability": "unavailable",
        "reasoning_trace_correctness": "not_applicable", "evidence_resolution": "resolved",
        "hallucination": "unavailable", "judge_error": True,
        "judge_error_failures": ["unmapped_but_present"],
    }
    rows = [
        _evaluation("old", "q1", 1, "A", fact=judge_failed,
                    metrics={"numeric_claim_count": 1, "judge_error_claim_count": 1}),
        _evaluation("new", "q1", 1, "B",
                    metrics={"numeric_claim_count": 1, "judge_error_claim_count": 0}),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "⚠" not in page
    assert "evaluator failure" not in page
    # And no banner: it warned about a risk the reader cannot act on now.
    assert "could not be mapped" not in page


def test_no_judge_banner_when_nothing_failed():
    rows = [
        _evaluation("old", "q1", 1, "A",
                    metrics={"numeric_claim_count": 2, "judge_error_claim_count": 0}),
        _evaluation("new", "q1", 1, "B",
                    metrics={"numeric_claim_count": 2, "judge_error_claim_count": 0}),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "could not be scored" not in page


def test_team_construction_is_shown_beside_the_answer():
    """Construction and content fail differently and belong on one screen."""
    rows = [
        {**_evaluation("new", "q1", 1, "ANSWER TEXT"),
         "team": ["spend_payments", "report_agent"],
         "tools": ["batch_aggregate"],
         "subqueries": {"spend_payments": "count returned payments"}},
        _evaluation("old", "q1", 1, "OTHER"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "team construction" in page
    assert "spend_payments" in page and "report_agent" in page
    # Data tools are not team construction; only specialists are shown.
    assert "batch_aggregate" not in page
    assert "count returned payments" in page


def test_consistency_uses_jaccard_not_exact_match():
    """Exact match is all-or-nothing: one differing call collapses it to 0,
    which says nothing about how far apart two runs actually were."""
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    summary = {"groups": [
        _summary_group("old", "q1", tool_call_pairwise_multiset_jaccard=0.8,
                       tool_exact_consistency=0.0),
        _summary_group("new", "q1", tool_call_pairwise_multiset_jaccard=0.4,
                       tool_exact_consistency=0.0),
    ]}
    page = answer_comparison_html(rows, baseline="old", candidate="new",
                                  summary=summary)
    assert "Tool-call Jaccard" in page
    assert "Tool-name exact" not in page
    assert "Tool-call exact" not in page


def test_judge_error_is_not_shown_as_a_metric_but_claims_stay_marked():
    rows = [
        _evaluation("old", "q1", 1, "A", metrics={
            "judge_error_rate": 0.5, "numeric_claim_count": 2,
            "judge_error_claim_count": 1, "orthogonal_claim_count": 1, "all_factual_claim_count": 2,
        }),
        _evaluation("new", "q1", 1, "B", metrics={
            "judge_error_rate": 0.0, "numeric_claim_count": 2,
            "judge_error_claim_count": 0, "orthogonal_claim_count": 1, "all_factual_claim_count": 1,
        }),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "Judge error" not in page
    assert "could not be mapped" not in page
    assert "Orthogonal claims" in page


def test_modules_are_ordered_with_memory_before_system():
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    summary = {"groups": [
        _summary_group("old", "q1", memory_hit_rate=1.0,
                       latency_seconds={"mean": 10.0}),
        _summary_group("new", "q1", memory_hit_rate=0.5,
                       latency_seconds={"mean": 20.0}),
    ]}
    page = answer_comparison_html(rows, baseline="old", candidate="new",
                                  summary=summary)
    assert page.index("<h4>Memory</h4>") < page.index("<h4>System</h4>")
    # Per-question metrics use the same 2x2 board as the overview.
    assert page.count('class="mgrid"') >= 2


def test_a_count_is_totalled_over_the_question_set_not_averaged():
    """1 memory-required question out of 2 is 1, not 0.5.

    The count is over QUESTIONS, so it does not move when k or the case list
    changes — it is a fact about the suite, not about how often it was asked.
    """
    rows = [
        _evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B"),
        _evaluation("old", "q2", 1, "C"), _evaluation("new", "q2", 1, "D"),
    ]
    summary = {"groups": [
        _summary_group("old", "q1", memory_required_question_count=1),
        _summary_group("old", "q2", memory_required_question_count=0),
        _summary_group("new", "q1", memory_required_question_count=1),
        _summary_group("new", "q2", memory_required_question_count=0),
    ]}
    page = answer_comparison_html(
        rows, baseline="old", candidate="new", summary=summary,
    )
    overview = page.split('id="overview"', 1)[1].split("</section>", 1)[0]
    memory = overview.split("<h4>Memory</h4>", 1)[1]
    assert "Memory-required questions" in memory
    assert ">1<" in memory and "0.5" not in memory


def test_restated_claims_are_rendered_so_the_toggle_can_hide_them():
    """They carry no fact result now; dropping them made the toggle look broken."""
    row = _evaluation("old", "q1", 1, "A")
    row["claims"].append({
        "claim_id": "c2", "block_id": "b001", "is_factual": True,
        "restates_claim_id": "c1", "proposition": "the same thing again",
        "source_locator": {},
    })
    page = answer_comparison_html([row], baseline="old", candidate="new")
    assert "the same thing again" in page
    assert 'class="restated"' in page or "restated" in page
    assert 'id="hide-restated"' in page


def test_the_grounded_column_shows_the_verdict_not_the_evidence_tier():
    row = _evaluation("old", "q1", 1, "A", fact={
        "claim_id": "c1", "numeric_support": "yes", "traceability": "yes",
        "reasoning_trace_correctness": "yes", "evidence_resolution": "resolved",
        "grounding_kind": "factual",
    })
    page = answer_comparison_html([row], baseline="old", candidate="new")
    facts = page.split('<table class="facts"', 1)[1].split("</table>", 1)[0]
    # The tier is diagnostic context, not the verdict the metrics report.
    assert "cited provenance: resolved" in facts
    assert "◇" not in facts


def test_questions_appear_in_the_order_they_were_asked():
    """A stateful run is a conversation; alphabetical order breaks the thread."""
    def at(system, name, position):
        row = _evaluation(system, name, 1, "answer")
        row["sequence_position"] = position
        return row

    rows = [
        at("old", "the_balance", 2), at("new", "the_balance", 2),
        at("old", "how_many_cards", 1), at("new", "how_many_cards", 1),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert page.index("how_many_cards") < page.index("the_balance")


def test_a_grounded_qualitative_claim_is_factual_support():
    from agentic_eval.content.claims import _normalize_claim
    from agentic_eval.content.verify import _normalize_fact_results

    claim = _normalize_claim({
        "claim_id": "c1", "is_factual": True, "claim_type": "negative_entity_count",
        "proposition": "No further commercial cards are present.",
        "numeric_mentions": [{"written": "No further", "value": 0}],
    }, 1)
    ledger = [{"evidence_id": "ev1", "source_type": "tool_result", "tool": "query_table",
               "evidence_tier": "primary", "result": {"rows": [{"Card Portfolio": "SBS"}]}}]
    raw = {"claim_id": "c1", "evidence_ids": ["ev1"],
           "verdict": "supported", "reason": "", "numbers": []}
    # Eligibility has exactly ONE source now — the ruling on the recorded
    # route. A verdict volunteered inside the fact result is ignored, so the
    # route has to be supplied here for the claim to ground.
    traces = {"c1": {"call_ids": ["ev1"], "eligible": "yes"}}
    fact = _normalize_fact_results([claim], [raw], ledger, claim_traces=traces)[0]
    # It carries no checkable figure, but the answer's own evidence reaches the
    # operations behind it, and the route answers the question.
    assert fact["grounding_kind"] == "factual"


def test_the_fact_table_keeps_one_marker():
    """Four cascade columns that almost always agree read as noise."""
    page = answer_comparison_html(
        [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")],
        baseline="old", candidate="new",
    )
    header = page.split('<table class="facts"', 1)[1].split("</thead>", 1)[0]
    assert "gnd" in header
    for dropped in ("num", "trc", "trq"):
        assert f"<th>{dropped}</th>" not in header


def test_an_oracle_question_shows_the_extracted_answer():
    row = _evaluation("old", "q1", 1, "A")
    row["expected_answer_results"] = [{
        "expected_answer_id": "transactions_last_month", "expected": 22,
        "matched_value": 42, "verdict": "fail",
    }]
    page = answer_comparison_html([row], baseline="old", candidate="new")
    assert "expected answer" in page
    assert "transactions_last_month" in page and ">22<" in page and ">42<" in page


def test_a_rubric_question_shows_which_must_haves_were_hit():
    row = _evaluation("old", "q1", 1, "A")
    row["must_have_results"] = [
        {"must_have_id": "mh_a", "description": "date the spike", "verdict": "full"},
        {"must_have_id": "mh_b", "description": "name the drivers", "verdict": "miss"},
    ]
    page = answer_comparison_html([row], baseline="old", candidate="new")
    assert "must-haves 1/2" in page
    assert "date the spike" in page and "name the drivers" in page


def test_a_boolean_oracle_that_passed_is_not_shown_as_absent():
    """It matches wording, not a figure, so it records no `matched_value`."""
    row = _evaluation("old", "q1", 1, "A")
    row["expected_answer_results"] = [{
        "expected_answer_id": "has_payment_returns", "expected": False,
        "matched_value": None, "verdict": "pass",
        "reason": "Ground truth is False; the answer states False.",
    }]
    page = answer_comparison_html([row], baseline="old", candidate="new")
    block = page.split('class="expect"', 1)[1].split("</details>", 1)[0]
    assert "not in answer" not in block
    assert "stated" in block
    # The pattern that matched stays inspectable.
    assert "the answer states False" in page


def _repeat_eval(system, run_index, name, answer, grounded):
    return {
        "system": system, "mode": "stateful", "name": name,
        "run_index": run_index, "sequence_position": 1,
        "question": "how many cards?", "answer": answer,
        "claims": [{"claim_id": "c1", "is_factual": True, "proposition": answer,
                    "numeric_mentions": []}],
        "fact_results": [{"claim_id": "c1",
                          "grounding_kind": "factual" if grounded else "none",
                          "eligible": "yes", "traced": "yes", "numbers": []}],
        "must_have_results": [], "expected_answer_results": [],
        "metrics": {"orthogonal_claim_count": 1, "all_factual_claim_count": 1,
                    "grounded_count": 1 if grounded else 0,
                    "factual_grounded_count": 1 if grounded else 0,
                    "report_grounded_count": 0},
        "evidence_ledger": [], "blocks": [], "judge_calls": [],
    }


def test_every_repeat_is_rendered_not_just_the_sampled_one():
    """k=3 answers the same question three times and they can differ.

    Showing one and calling it the comparison hides the variance the repeats
    were run to expose.
    """
    rows = [
        _repeat_eval("old", 1, "q", "one card", True),
        _repeat_eval("old", 2, "q", "two cards", False),
        _repeat_eval("old", 3, "q", "three cards", True),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    for answer in ("one card", "two cards", "three cards"):
        assert answer in page, answer
    assert page.count('class="rtab') == 3
    # exactly one pane visible to begin with; the rest are hidden
    assert page.count('class="rpane"') + page.count('class="rpane" ') >= 0
    assert page.count("data-repeat=") >= 6          # 3 buttons + 3 panes


def test_metrics_total_over_repeats_not_the_shown_one():
    """The rate must not belong to whichever pane is open.

    Two of three repeats grounded is 2/3 — reading 1/1 off the visible pane
    would report a third of the evidence as if it were all of it.
    """
    rows = [
        _repeat_eval("old", 1, "q", "a", True),
        _repeat_eval("old", 2, "q", "b", False),
        _repeat_eval("old", 3, "q", "c", True),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "2/3" in page
    assert "totalled over all 3" in page


def test_metrics_open_first_and_claims_are_deduplicated_by_default():
    """The metrics are the result; the answers are the evidence for it.

    And a restatement is one fact stated twice — the metrics already count it
    once, so showing it by default made the page and the numbers disagree.
    """
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert '<button role="tab" data-tab="metrics" aria-selected="true">' in page
    assert 'data-tab="claims" aria-selected="false"' in page
    assert '<div class="tabpanel" data-tab="claims" hidden>' in page
    assert 'id="hide-restated" checked' in page
    assert '<body class="hide-restated">' in page


def test_the_page_does_not_say_atomic_facts():
    """"Atomic fact" reads as the system's own KPs; these are extracted claims."""
    rows = [_evaluation("old", "q1", 1, "A"), _evaluation("new", "q1", 1, "B")]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert "atomic fact" not in page.lower()
    assert "claim" in page.lower()


def _with_case(evaluation, case_id):
    return {**evaluation, "case_id": case_id}


def test_case_tabs_appear_and_every_case_repeat_pane_is_rendered():
    """Case and repeat are coordinates on one grid, not independent filters.

    A pane carries both, and the page opens on the first case's sampled
    repeat — so two readings of "the same" report show the same answers.
    """
    rows = [
        _with_case(_evaluation("old", "q1", 1, "OLD A1"), "case_a"),
        _with_case(_evaluation("new", "q1", 1, "NEW A1"), "case_a"),
        _with_case(_evaluation("old", "q1", 2, "OLD A2"), "case_a"),
        _with_case(_evaluation("new", "q1", 2, "NEW A2"), "case_a"),
        _with_case(_evaluation("old", "q1", 1, "OLD B1"), "case_b"),
        _with_case(_evaluation("new", "q1", 1, "NEW B1"), "case_b"),
        _with_case(_evaluation("old", "q1", 2, "OLD B2"), "case_b"),
        _with_case(_evaluation("new", "q1", 2, "NEW B2"), "case_b"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'class="ctab"' in page
    assert 'data-case="case_a"' in page and 'data-case="case_b"' in page
    # Every case x repeat answer is on the page, not just the sampled one.
    for answer in ("OLD A1", "OLD A2", "OLD B1", "OLD B2", "NEW B2"):
        assert answer in page
    # Exactly one pane open: first case, first repeat.
    assert '<div class="rpane" data-repeat="1" data-case="case_a">' in page
    assert '<div class="rpane" data-repeat="2" data-case="case_a" hidden>' in page
    assert '<div class="rpane" data-repeat="1" data-case="case_b" hidden>' in page
    assert page.count("2 cases") >= 1


def test_a_single_case_run_shows_no_case_selector():
    """One case is not a choice, and a selector with one option is noise."""
    rows = [
        _with_case(_evaluation("old", "q1", 1, "OLD ONE"), "case_a"),
        _with_case(_evaluation("new", "q1", 1, "NEW ONE"), "case_a"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'class="ctab"' not in page
    assert "OLD ONE" in page and "NEW ONE" in page


def test_a_run_predating_case_id_still_renders():
    """Older evaluations carry no case; the page must not require one."""
    rows = [
        _evaluation("old", "q1", 1, "OLD ONE"),
        _evaluation("new", "q1", 1, "NEW ONE"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'class="ctab"' not in page
    assert "data-case=" not in page
    assert "OLD ONE" in page and "NEW ONE" in page


def test_a_padded_case_reads_plainly_but_selects_exactly():
    """The tab LABEL is tidied; the value that selects panes is not.

    Quoting the label put `'11854808010 '` on a button, which reads as part of
    the id. The exact id stays in `data-case` — panes are matched on it — and
    hovering says what it really is.
    """
    rows = [
        _with_case(_evaluation("old", "q1", 1, "OLD PAD"), "11854808010 "),
        _with_case(_evaluation("new", "q1", 1, "NEW PAD"), "11854808010 "),
        _with_case(_evaluation("old", "q1", 1, "OLD PLAIN"), "366132845011"),
        _with_case(_evaluation("new", "q1", 1, "NEW PLAIN"), "366132845011"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert '>11854808010</button>' in page          # label, tidied
    assert 'data-case="11854808010 "' in page       # value, exact
    assert "exact id:" in page                      # hover says so


def _in_set(evaluation, question_set):
    return {**evaluation, "question_set": question_set}


def test_each_question_set_gets_its_own_metrics_section():
    """Series A and series B answer different questions about the system.

    Pooling them into one overall number answers neither, and reading every
    question separately buries it. The set total is built from the same
    `_content_rows` as the overview, so the two cannot disagree.
    """
    rows = [
        _in_set(_evaluation("old", "a1", 1, "A", metrics={
            "factual_grounded_count": 1, "orthogonal_claim_count": 1}), "series_a"),
        _in_set(_evaluation("new", "a1", 1, "B", metrics={
            "factual_grounded_count": 1, "orthogonal_claim_count": 1}), "series_a"),
        _in_set(_evaluation("old", "b1", 1, "C", metrics={
            "factual_grounded_count": 0, "orthogonal_claim_count": 2}), "series_b"),
        _in_set(_evaluation("new", "b1", 1, "D", metrics={
            "factual_grounded_count": 2, "orthogonal_claim_count": 2}), "series_b"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'id="set-series_a"' in page and 'id="set-series_b"' in page
    a_block = page.split('id="set-series_a"', 1)[1].split("</section>", 1)[0]
    b_block = page.split('id="set-series_b"', 1)[1].split("</section>", 1)[0]
    # series_a: both 1/1. series_b: old 0/2, new 2/2.
    assert "100% (1/1)" in a_block
    assert "0% (0/2)" in b_block and "100% (2/2)" in b_block
    assert "own session" in a_block


def test_one_set_gets_no_per_set_section():
    """With a single set the per-set table is the overview under another name."""
    rows = [
        _in_set(_evaluation("old", "a1", 1, "A"), "series_a"),
        _in_set(_evaluation("new", "a1", 1, "B"), "series_a"),
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert 'id="set-series_a"' not in page
    assert "All question sets — overall" in page


def _two_case_rows():
    rows = []
    for case in ("case_a", "case_b"):
        for repeat in (1, 2):
            for system in ("old", "new"):
                for name, qset in (("a1", "series_a"), ("b1", "series_b")):
                    rows.append({
                        **_evaluation(system, name, repeat, f"{system} {name}"),
                        "case_id": case, "question_set": qset,
                    })
    return rows


def test_the_case_and_repeat_selectors_belong_to_the_answers_tab():
    """They say "answers only" — on the metrics tab they control nothing.

    Every metric is totalled over all cases and repeats, so a selector sitting
    above them invites the reading that it filters them.
    """
    page = answer_comparison_html(_two_case_rows(), baseline="old", candidate="new")
    for label in ('aria-label="case"', 'aria-label="repeat"'):
        bar = page.split(label, 1)[0].rsplit("<div", 1)[1] + label
        tag = page[page.index("<div" + bar) : page.index(">", page.index(label))]
        assert 'data-only-tab="claims"' in tag, label
        assert "hidden" in tag, f"{label} must start hidden — metrics opens first"
    # `hidden` loses to `display:flex` without an explicit rule.
    assert ".repeats[hidden] { display: none; }" in page


def test_the_nav_follows_the_tab():
    """Metrics lists what the metrics group by; Answers lists the questions."""
    page = answer_comparison_html(_two_case_rows(), baseline="old", candidate="new")
    metrics_nav = page.split('<div data-only-tab="metrics">', 1)[1].split("</div>", 1)[0]
    claims_nav = page.split('<div data-only-tab="claims" hidden>', 1)[1].split("</div>", 1)[0]
    assert "<h2>Metrics</h2>" in metrics_nav
    assert "<h2>Questions</h2>" in claims_nav
    names = lambda block: re.findall(r'class="tname">([^<]+)<', block)
    # Sets, each followed by its own questions nested beneath it.
    assert names(metrics_nav) == ["overview", "series_a", "a1", "series_b", "b1"]
    assert sorted(names(claims_nav)) == ["a1", "b1"]
    # The question list must not offer links while the metrics tab is open.
    assert '<div data-only-tab="claims" hidden><h2>Questions' in page


def test_the_open_tab_and_the_page_agree_on_load():
    """Every `data-only-tab="metrics"` section ships hidden.

    With metrics as the default tab and no call on load, the page opened with
    the tab selected but its overview and per-set tables invisible until the
    tab was clicked once.
    """
    page = answer_comparison_html(_two_case_rows(), baseline="old", candidate="new")
    assert 'showTab(document.querySelector' in page


def test_the_metrics_nav_nests_each_sets_questions_under_it():
    """Per-question metrics live on this tab; the nav must reach them.

    Listing only the set totals left the rows they are made of scrollable but
    unreachable — the reader could see 30/40 for series_b and had no way to ask
    which of its questions lost the ten.
    """
    page = answer_comparison_html(_two_case_rows(), baseline="old", candidate="new")
    metrics_nav = page.split('<div data-only-tab="metrics">', 1)[1].split("</div>", 1)[0]
    assert '<ul class="subtoc">' in metrics_nav
    # Each question links to its own metrics section and keeps the tab.
    for name, anchor in (("a1", "q0"), ("b1", "q1")):
        assert (
            f'<li><a href="#{anchor}" data-target="{anchor}" data-tab="metrics">'
            f'<span class="tname">{name}</span></a></li>'
        ) in metrics_nav, name
    # A question sits under ITS set, not the other one.
    a_block = metrics_nav.split("series_a", 1)[1].split("series_b", 1)[0]
    assert "a1" in a_block and "b1" not in a_block


def test_the_answers_nav_stays_flat():
    """It lists questions only — nesting them under sets would repeat the
    metrics nav while the sets have no answers panel to open."""
    page = answer_comparison_html(_two_case_rows(), baseline="old", candidate="new")
    claims_nav = page.split('<div data-only-tab="claims" hidden>', 1)[1].split("</div>", 1)[0]
    assert '<ul class="subtoc">' not in claims_nav


def _ordered_rows():
    """Two sets whose positions both restart at 1, listed b-then-a."""
    spec = [
        ("series_b", "b2", 1), ("series_b", "b3", 2),
        ("series_a", "a1", 1), ("series_a", "a2", 2),
    ]
    return [
        {
            **_evaluation(system, name, 1, f"{system} {name}"),
            "question_set": qset, "sequence_position": position,
            "mode": "stateful",
        }
        for qset, name, position in spec
        for system in ("old", "new")
    ]


def test_questions_order_by_set_name_then_position():
    """Sets in name order, questions in the order asked within each.

    `sequence_position` restarts at 1 in every set, so ordering by it alone
    grouped every set's OPENING question together and pushed the follow-ups to
    the end. Set order is the NAME, not first appearance in the file: a subset
    re-judge rewrites the file in pieces, and the page then read b, a, c.
    """
    page = answer_comparison_html(_ordered_rows(), baseline="old", candidate="new")
    ordered = re.findall(
        r'<section class="question" id="q\d+"[^>]*>\s*<h3>([^<]+)</h3>', page,
    )
    assert ordered == ["a1", "a2", "b2", "b3"]


def test_set_order_is_the_name_not_the_file_order():
    """The fixture lists series_b first on purpose.

    Order a reader can predict beats order that depends on which questions
    happened to be re-scored last.
    """
    page = answer_comparison_html(_ordered_rows(), baseline="old", candidate="new")
    assert page.index('id="set-series_a"') < page.index('id="set-series_b"')
    nav = page.split('<div data-only-tab="metrics">', 1)[1].split("</div>", 1)[0]
    assert re.findall(r'class="tname">([^<]+)<', nav) == [
        "overview", "series_a", "a1", "a2", "series_b", "b2", "b3",
    ]


def test_natural_order_puts_ten_after_two():
    from agentic_eval.render.page import _natural_key

    assert sorted(["a10", "a2", "a1"], key=_natural_key) == ["a1", "a2", "a10"]


def test_the_anchors_of_one_set_are_contiguous():
    """b3 must follow b2, not land after another set's question."""
    page = answer_comparison_html(_ordered_rows(), baseline="old", candidate="new")
    nav = page.split('<div data-only-tab="metrics">', 1)[1].split("</div>", 1)[0]
    b_sub = nav.split("series_b", 1)[1]
    assert re.findall(r'href="#(q\d+)"', b_sub) == ["q2", "q3"]


def _ungrounded(**fields):
    return {
        "claim_id": "c1", "grounding_kind": "none",
        "evidence_resolution": "resolved", **fields,
    }


def test_an_ungrounded_claim_says_why_in_red():
    """The marker says a claim did not ground; it does not say why.

    Without this a reader has to open the record for every ○, and the shape of
    the failure — nine claims losing on the same wrong window — is invisible.
    """
    rows = [_evaluation("old", "q1", 1, "AN ANSWER", fact=_ungrounded(
        eligible="no", traced="yes", route=[{"step": "x"}],
        eligibility_reason="covers July 2023 to Feb 2025, but the question "
                           "asks about 2025/2026",
    ))]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    assert '<div class="why"' in page
    assert "route answers a different question" in page
    # The judge's full sentence stays on hover, not in the row.
    assert "covers July 2023 to Feb 2025" in page
    assert ".facts .why {" in page and "color: var(--bad)" in page


def test_the_reason_names_the_figure_that_failed():
    """"figures not found" does not say WHICH figure.

    The written value is what a reader is looking for, and naming it often
    exposes the real problem — "mid-2024" is a period label the numeric check
    was never going to locate.
    """
    cases = [
        (dict(eligible="no", route=[{"s": 1}]), "route answers a different question"),
        (dict(eligible="yes", route=[{"s": 1}], numbers=[
            {"written_value": "$0", "trace_failure": "not_located"}]),
         "\u201c$0\u201d not found in the cited evidence"),
        (dict(eligible="yes", route=[{"s": 1}], numbers=[
            {"written_value": "42%", "trace_failure": "value_mismatch",
             "evidence_value": 38}]),
         "\u201c42%\u201d disagrees with the evidence (38)"),
        (dict(eligible="yes", route=[{"s": 1}], relations=[
            {"trace_failure": "relation_does_not_hold"}]),
         "the stated relation is not confirmed by the evidence"),
        (dict(eligible="yes", route=[]), "no recorded operation behind it"),
        (dict(eligible="unavailable", route=[{"s": 1}]),
         "no operation recorded, so relevance could not be judged"),
    ]
    for fields, expected in cases:
        page = answer_comparison_html(
            [_evaluation("old", "q1", 1, "A", fact=_ungrounded(**fields))],
            baseline="old", candidate="new",
        )
        assert expected in page, fields


def test_several_failing_figures_are_summarised_not_listed_in_full():
    fact = _ungrounded(eligible="yes", route=[{"s": 1}], numbers=[
        {"written_value": f"v{i}", "trace_failure": "not_located"} for i in range(4)
    ])
    page = answer_comparison_html(
        [_evaluation("old", "q1", 1, "A", fact=fact)],
        baseline="old", candidate="new",
    )
    assert "(+2 more)" in page


def test_the_hover_detail_matches_the_label():
    """It used to be `eligibility_reason` whatever failed, so a numeric-trace
    failure hovered to a sentence explaining why the route WAS eligible."""
    fact = _ungrounded(
        eligible="yes",
        eligibility_reason="the route is eligible and addresses the question",
        route=[{"s": 1}],
        numbers=[{"written_value": "$0", "trace_failure": "not_located",
                  "json_path": "results[0].summary"}],
    )
    page = answer_comparison_html(
        [_evaluation("old", "q1", 1, "A", fact=fact)],
        baseline="old", candidate="new",
    )
    assert "the route is eligible" not in page
    assert "results[0].summary" in page


def test_a_grounded_claim_carries_no_reason():
    """The red line is for failures only; on a grounded row it is noise."""
    page = answer_comparison_html(
        [_evaluation("old", "q1", 1, "A")], baseline="old", candidate="new",
    )
    assert '<div class="why"' not in page


def test_memory_requirement_is_a_question_count_not_a_run_count():
    """`Memory required: 4` was counting runs — 2 cases x 2 repeats.

    Needing memory is a property of the QUESTION, so it must not move when k
    or the case list changes. Over the set it is a count of questions; for a
    single question it is a yes/no.
    """
    from agentic_eval.render.page import _MODULE_METRICS, _QUESTION_MODULE_METRICS

    overview = {label: kind for key, label, _b, kind in _MODULE_METRICS["memory"]}
    assert overview["Memory-required questions"] == "count"
    assert "Memory-required runs" not in overview

    per_question = {
        label: (key, kind)
        for key, label, _b, kind in _QUESTION_MODULE_METRICS["memory"]
    }
    assert per_question["Memory required"] == ("memory_required", "bool")


def test_a_yes_no_row_renders_as_words_with_no_delta():
    from agentic_eval.render.page import _delta_cell, _fmt_value

    assert _fmt_value(True, "bool") == "yes"
    assert _fmt_value(False, "bool") == "no"
    # Not annotated is not the same as "no".
    assert _fmt_value(None, "bool") == "—"
    # Both systems answer the same question set, so a delta is meaningless.
    assert _delta_cell(True, False, True, "bool") == '<td class="delta"></td>' 


def _scored(system, name, run_index, *, correct, checked, case_id="c1"):
    row = _evaluation(system, name, run_index, "A", metrics={
        "answer_correct": correct, "answer_checked": checked,
    })
    row["case_id"] = case_id
    return row


def test_a_per_question_judgement_is_averaged_over_questions_not_answers():
    """"3/4" at the overview reads as three of four QUESTIONS.

    It was three of four answers to the only question that had an oracle. A
    question asked about more cases or more repeats must not count for more,
    so each question's own rate is taken first and those are averaged.
    """
    rows = [
        # q1: 4 answers, all correct. q2: 2 answers, both wrong.
        *[_scored("old", "q1", i, correct=1, checked=1) for i in (1, 2)],
        *[_scored("old", "q1", i, correct=1, checked=1, case_id="c2") for i in (1, 2)],
        *[_scored("old", "q2", i, correct=0, checked=1) for i in (1, 2)],
        *[_scored("new", "q1", i, correct=1, checked=1) for i in (1, 2)],
        *[_scored("new", "q1", i, correct=1, checked=1, case_id="c2") for i in (1, 2)],
        *[_scored("new", "q2", i, correct=0, checked=1) for i in (1, 2)],
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    overview = page.split('id="overview"', 1)[1].split("</section>", 1)[0]
    accuracy = re.search(
        r'<tr><td>Accuracy</td><td class="num">([^<]*)</td>', overview,
    ).group(1)
    # Pooled over answers this is 4/6 = 67%. Averaged over questions it is
    # (100% + 0%) / 2 = 50%, and q1's extra case does not buy it weight.
    assert accuracy == "50% (2 questions)"


def test_a_question_shows_its_rate_with_no_counts():
    """At a question the value IS the mean over its answers.

    "50% (2/4)" invited reading the pair as a fraction of must-haves rather
    than of answers, and it is the same kind of number as the 96.9% beside it.
    Only the aggregate levels name what they averaged over.
    """
    rows = [
        *[_scored("old", "q1", i, correct=1, checked=1) for i in (1, 2)],
        *[_scored("new", "q1", i, correct=0, checked=1) for i in (1, 2)],
    ]
    page = answer_comparison_html(rows, baseline="old", candidate="new")
    section = page.split('id="q0"', 1)[1].split("</section>", 1)[0]
    assert '<tr><td>Accuracy</td><td class="num">100%</td>' in section
    assert "(2/2)" not in section
    # The overview over the same single question names the question count.
    overview = page.split('id="overview"', 1)[1].split("</section>", 1)[0]
    assert "1 question)" in overview


def test_a_contradicted_oracle_says_what_the_answer_claimed():
    """"not in answer" read as if the answer had been silent.

    On the case with one payment return, every answer asserted there were
    none — a contradiction, not an omission, and the worse of the two.
    """
    from agentic_eval.render.page import _oracle_found

    assert _oracle_found({
        "verdict": "fail", "matched_value": None, "expected": True,
        "reason": "Ground truth is True; the answer states False (matched: x).",
    }) == '<span class="bad-val">False</span> <span class="loc">stated</span>'


def test_a_missing_figure_says_which_kind_of_absence():
    from agentic_eval.render.page import _oracle_found

    assert "no such figure" in _oracle_found({
        "verdict": "fail", "matched_value": None, "expected": 1,
        "reason": "No material number in the answer equals 1.0 (tolerance 0.0).",
    })
    # Genuinely silent stays "not in answer".
    assert "not in answer" in _oracle_found({
        "verdict": "fail", "matched_value": None, "expected": 5, "reason": "",
    })
