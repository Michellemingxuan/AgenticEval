"""A page that says how far a run has got, while it is still going.

The console prints one line per answer, which tells you what just happened and
nothing about what remains: not how many cases are in flight, not which
questions are done, not how many repeats of them. On a 200-record run against a
slow gateway that is the difference between "it is working" and "it has been
stuck for twenty minutes".

So: rewrite a small self-contained page every time a record lands. It refreshes
itself, needs no server, and can be opened mid-run.

Everything here is derived from the records already written — there is no
second source of truth to drift. The plan (how many are expected) comes from
the run's own shape, so a scoped run shows its own total rather than the
config's.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
from typing import Any

from agentic_eval.cases import describe_case
from agentic_eval.models import ANSWERED_OUTCOMES

#: Outcomes that are not a failure. `out_of_scope` is a correct answer to an
#: off-domain question, so a refusal must not show red in the grid.
_GOOD = set(ANSWERED_OUTCOMES) | {"qa_cache_hit"}

#: Seconds between browser refreshes. Short enough to feel live on a run whose
#: turns take a minute, long enough not to fight a reader scrolling the table.
_REFRESH_S = 10


def _rate(done: int, total: int) -> float:
    return (done / total * 100.0) if total else 0.0


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def summarize(
    records: list[dict[str, Any]], *, plan: dict[str, Any],
    started_at: float | None = None,
) -> dict[str, Any]:
    """Counts by every axis a reader asks about: case, set, question, repeat.

    Split out from rendering so the arithmetic is testable without parsing
    HTML, and so the same numbers could feed a different surface later.
    """
    expected = int(plan.get("expected_records") or 0)
    done = len(records)
    by_outcome: dict[str, int] = {}
    for record in records:
        outcome = str(record.get("outcome") or "unknown")
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

    def tally(key: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for record in records:
            name = record.get(key)
            label = "—" if name is None else str(name)
            slot = out.setdefault(label, {"done": 0, "answered": 0})
            slot["done"] += 1
            # ANSWERED, not "ok": a refusal to an off-domain question is a
            # correct answer, and counting it against the row made the system
            # look worse the better it behaved.
            if str(record.get("outcome")) in _GOOD:
                slot["answered"] += 1
        return out

    # `is not None`, not truthiness: a start time of 0.0 is a real timestamp
    # and would otherwise silently drop the elapsed clock and the ETA.
    elapsed = (time.time() - started_at) if started_at is not None else None
    remaining = None
    if elapsed and done and expected > done:
        remaining = elapsed / done * (expected - done)

    retried = sum(1 for r in records if int(r.get("evaluator_attempts") or 1) > 1)
    return {
        "done": done, "expected": expected,
        "percent": _rate(done, expected),
        "outcomes": by_outcome,
        "retried_records": retried,
        "elapsed_s": elapsed, "eta_s": remaining,
        "by_case": tally("case_id"),
        "by_set": tally("question_set"),
        "by_question": tally("name"),
        "by_system": tally("system"),
        "by_repeat": tally("run_index"),
        "plan": plan,
    }


def grid(
    records: list[dict[str, Any]], *, plan: dict[str, Any],
    order: list[str] | None = None, width: int = 10,
    colourise: bool | None = None,
) -> str:
    """Questions down, cases across, repeats in the cell — for a terminal.

    The console's one-line-per-answer says what just happened and nothing about
    the shape of what remains. This is the same run seen from above: which
    question, on which case, how many of its repeats are in.

    The BAR is answers (each repeat is one answer per system, so it advances
    twice per repeat and moves smoothly); the COUNT is whole repeats, since
    "2/3" meaning two complete repeats is what a reader actually wants. A
    repeat counts only when every system has answered it — a half-answered
    repeat is not a repeat.
    """
    cases: list[str] = []
    for record in records:
        case = record.get("case_id")
        case = "—" if case is None else str(case)
        if case not in cases:
            cases.append(case)
    for case in (plan.get("case_ids") or []):
        if str(case) not in cases:
            cases.append(str(case))

    # Series order, then position within the series — the order the questions
    # are ASKED, which is the order a reader is tracking. Arrival order would
    # reshuffle the rows as workers finish, so the same run would look
    # different every refresh.
    planned = list(order or plan.get("question_order") or [])
    seen_names = {str(r.get("name") or "—") for r in records}
    questions = [name for name in planned if name in seen_names or True]
    questions += [name for name in sorted(seen_names) if name not in questions]
    if not cases or not questions:
        return "  (nothing recorded yet)"

    systems = max(1, int(plan.get("systems") or 2))
    repeats = max(1, int(plan.get("repeats") or 1))
    set_of = dict(plan.get("set_of") or {})
    colour = _supports_colour() if colourise is None else colourise

    # (question, case) -> {repeat: [outcomes]}
    seen: dict[tuple[str, str], dict[Any, list[str]]] = {}
    for record in records:
        case = record.get("case_id")
        key = (str(record.get("name") or "—"), "—" if case is None else str(case))
        slot = seen.setdefault(key, {})
        slot.setdefault(record.get("run_index"), []).append(
            str(record.get("outcome") or "unknown")
        )

    shown = case_labels(cases)
    # +2 for the indent grouped rows carry, or a short name like "a1" would be
    # truncated to "… " by its own label column.
    indent = 2 if set_of else 0
    label_width = min(34, max(len(q) for q in questions) + indent + 1)
    cell_width = width + 10
    lines = [
        " " * label_width + "".join(_fit(shown[case], cell_width) for case in cases)
    ]
    current_set = None
    for question in questions:
        group = set_of.get(question)
        if group and group != current_set:
            lines.append(f"{group}")
            current_set = group
        row = _fit(("  " if group else "") + question, label_width)
        for case in cases:
            slot = seen.get((question, case), {})
            answers = sum(len(v) for v in slot.values())
            whole = sum(1 for outcomes in slot.values() if len(outcomes) >= systems)
            bad = sum(
                1 for outcomes in slot.values()
                for outcome in outcomes if outcome not in _GOOD
            )
            text = f"{_blocks(answers, repeats * systems, width)} {whole}/{repeats}"
            if bad:
                text += f" ✗{bad}"
            # Pad FIRST, colour after: escape codes have no width, and
            # colouring before padding would misalign every later column.
            row += _red(_fit(text, cell_width), enabled=colour) if bad \
                else _fit(text, cell_width)
        lines.append(row.rstrip())
    return "\n".join(lines)


def case_labels(cases: list[str]) -> dict[str, str]:
    """Display names for case ids: plain, unless plain would be ambiguous.

    The ids carry their directory names exactly, and one real folder ends in a
    space — `describe_case` quotes that so the padding is visible. For a
    progress display the quoting is noise, so the space is dropped. But if two
    cases differ ONLY by whitespace, stripping makes them the same label and
    the reader cannot tell which row is which — so there both keep the quotes.
    """
    stripped = [case.strip() for case in cases]
    ambiguous = {name for name in stripped if stripped.count(name) > 1}
    return {
        # `repr`, not `describe_case`: describe_case quotes only the padded
        # one, so the pair would print as `'11854808010 '` and `11854808010`
        # — different, but not obviously two forms of the same id. Quoting
        # both makes the difference the thing you read.
        case: (repr(case) if case.strip() in ambiguous else case.strip())
        for case in cases
    }


def _supports_colour() -> bool:
    """ANSI only for a real terminal, and never when NO_COLOR is set.

    `bin/compare` tees this output to a log; escape codes in a file are noise,
    and a reader grepping it should not have to strip them.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _red(text: str, *, enabled: bool) -> str:
    return f"\033[31m{text}\033[0m" if enabled else text


def _fit(text: str, width: int) -> str:
    """Pad or truncate to exactly `width`, keeping one space of separation."""
    if len(text) >= width:
        return text[: max(0, width - 2)] + "… "
    return text + " " * (width - len(text))


def _blocks(done: int, total: int, width: int) -> str:
    filled = 0 if not total else min(width, round(width * done / total))
    return "█" * filled + "░" * (width - filled)


def _bar(done: int, total: int) -> str:
    percent = _rate(done, total)
    return (
        f'<div class="bar"><span style="width:{min(percent, 100):.1f}%"></span></div>'
        f'<div class="n">{done}<span class="of">/{total or "?"}</span></div>'
    )


def _table(title: str, rows: dict[str, dict[str, int]], per: Any) -> str:
    """`per` is one total for every row, or a per-label mapping.

    Question sets need the mapping: series_b has eight questions and series_c
    has one, so a single denominator would show one of them at 800%.
    """
    if not rows:
        return ""
    total_for = per.get if isinstance(per, dict) else (lambda _label: per)
    body = "".join(
        f"<tr><td class=\"k\">{html.escape(describe_case(label))}</td>"
        f"<td class=\"p\">{_bar(slot['done'], int(total_for(label) or 0))}</td>"
        f"<td class=\"m\">{slot['answered']}/{slot['done']} answered</td></tr>"
        for label, slot in sorted(rows.items())
    )
    return f'<section><h2>{html.escape(title)}</h2><table>{body}</table></section>'


def render(state: dict[str, Any]) -> str:
    plan = state.get("plan") or {}
    expected = int(state.get("expected") or 0)
    cases = max(1, int(plan.get("cases") or 1))
    systems = max(1, int(plan.get("systems") or 2))
    repeats = max(1, int(plan.get("repeats") or 1))
    questions = max(1, int(plan.get("questions") or 1))

    outcomes = " · ".join(
        f'<b class="{"bad" if name not in _GOOD else "good"}">'
        f"{html.escape(name)}</b> {count}"
        for name, count in sorted(state.get("outcomes", {}).items())
    ) or "nothing recorded yet"

    note = ""
    if state.get("retried_records"):
        note = (
            f'<p class="note">{state["retried_records"]} record(s) came from a '
            "replayed pass — a turn timed out and its session was re-asked.</p>"
        )

    return _PAGE.replace("{{REFRESH}}", str(_REFRESH_S)).replace(
        "{{PERCENT}}", f"{state.get('percent', 0):.0f}"
    ).replace("{{DONE}}", str(state.get("done", 0))).replace(
        "{{EXPECTED}}", str(expected or "?")
    ).replace("{{OVERALL}}", _bar(state.get("done", 0), expected)).replace(
        "{{ELAPSED}}", _clock(state.get("elapsed_s"))
    ).replace("{{ETA}}", _clock(state.get("eta_s"))).replace(
        "{{OUTCOMES}}", outcomes
    ).replace("{{NOTE}}", note).replace(
        "{{SECTIONS}}",
        _table("By case", state.get("by_case", {}), questions * systems * repeats)
        + _table("By question set", state.get("by_set", {}), {
            name: size * systems * repeats * cases
            for name, size in (plan.get("set_sizes") or {}).items()
        })
        + _table("By repeat", state.get("by_repeat", {}), questions * systems * cases)
        + _table("By system", state.get("by_system", {}), questions * repeats * cases)
        + _table("By question", state.get("by_question", {}), systems * repeats * cases),
    ).replace("{{STAMP}}", time.strftime("%H:%M:%S"))


def plan_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """The run's shape, recovered from its manifest.

    `progress.json` is written as a run goes, so a run that finished before
    this module existed — or one killed before its first record landed — has
    no plan beside it and every denominator reads "?". The manifest holds the
    same facts, so a finished run can still be described exactly.
    """
    sets = dict(manifest.get("question_sets") or {})
    cases = [c for c in (manifest.get("cases") or []) if c is not None]
    questions = int(manifest.get("question_count") or sum(
        len(names) for names in sets.values()
    ) or 0)
    systems = len(manifest.get("systems") or {}) or 2
    repeats = int(manifest.get("repeats") or 1)
    modes = 2 if manifest.get("mode") == "both" else 1
    return {
        "questions": questions,
        "cases": len(cases) or 1,
        "repeats": repeats,
        "systems": systems,
        "case_ids": [str(case) for case in cases],
        "set_sizes": {name: len(names) for name, names in sets.items()},
        "question_order": [name for names in sets.values() for name in names],
        "set_of": {
            name: set_name for set_name, names in sets.items() for name in names
        },
        "expected_records": (
            questions * (len(cases) or 1) * repeats * systems * modes
        ),
    }


def terminal_report(
    records: list[dict[str, Any]], *, plan: dict[str, Any],
    started_at: float | None = None, order: list[str] | None = None,
) -> str:
    """Headline, then the grid. What `run` prints and `progress` re-prints."""
    state = summarize(records, plan=plan, started_at=started_at)
    colour = _supports_colour()
    outcomes = "  ".join(
        _red(f"{name} {count}", enabled=colour) if name not in _GOOD
        else f"{name} {count}"
        for name, count in sorted(state["outcomes"].items())
    )
    head = (
        f"  {state['percent']:.0f}%  {state['done']}/{state['expected'] or '?'} answers"
        f"   elapsed {_clock(state['elapsed_s'])}"
        f"   remaining ~{_clock(state['eta_s'])}"
    )
    lines = [head, f"  {outcomes}" if outcomes else "", ""]
    if state["retried_records"]:
        lines.insert(2, f"  {state['retried_records']} from a replayed pass")
    body = grid(records, plan=plan, order=order)
    lines.extend("  " + line for line in body.splitlines())
    lines.append("")
    lines.append("  bar: answers  ·  count: whole repeats, both systems in")
    return "\n".join(line for line in lines if line is not None)


def write(
    path, records: list[dict[str, Any]], *, plan: dict[str, Any],
    started_at: float | None = None,
) -> None:
    """Best effort, always. A progress page must never fail a run.

    It is written from inside the record-writing lock on every answer, so a
    raise here would abort a run that is otherwise fine — the one thing a
    read-only convenience must not do.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render(summarize(records, plan=plan, started_at=started_at)),
            encoding="utf-8",
        )
        # The plan beside the page, so `agentic-eval progress` in another shell
        # can show real denominators mid-run. manifest.json is written last, so
        # while a run is going it is not there to ask.
        path.with_suffix(".json").write_text(
            json.dumps({"plan": plan, "started_at": started_at}, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - never fatal
        pass


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{{REFRESH}}">
<title>{{PERCENT}}% · AgenticEval run</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2032%2032'%3E%3Crect%20width='32'%20height='32'%20rx='6'%20fill='%2300175a'/%3E%3Cpath%20d='M14.6,8L8,16L14.6,24Z'%20fill='%23fff'/%3E%3Cpath%20d='M17.4,8L24,16L17.4,24'%20fill='none'%20stroke='%234c9ae8'%20stroke-width='3'%20stroke-linejoin='round'/%3E%3C/svg%3E">
<style>
:root {
  --ink: #1a1a1a; --muted: #6b7280; --faint: #9ca3af; --line: #e5e7eb;
  --wash: #f9fafb; --blue: #006fcf; --navy: #00175a; --good: #15803d;
  --bad: #b91c1c;
  --ui: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
html, body { margin: 0; padding: 0; }
body { font: 14px/1.45 var(--ui); color: var(--ink); background: #fff; }
header { background: var(--navy); color: #fff; padding: 20px 32px; position: relative; }
header::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: var(--blue);
}
header h1 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
header .sub { color: #b9c6de; font-size: 12.5px; margin-top: 3px; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 32px 60px; }
.headline { display: flex; align-items: baseline; gap: 14px; margin-bottom: 6px; }
.headline .pct { font-size: 34px; font-weight: 700; color: var(--navy); }
.headline .of { color: var(--faint); }
.meta { color: var(--muted); font-size: 12.5px; margin: 10px 0 22px; }
.meta b { font-weight: 600; }
.meta b.good { color: var(--good); }
.meta b.bad { color: var(--bad); }
.note {
  background: #fffbeb; border: 1px solid #fde68a; color: #78350f;
  padding: 8px 12px; font-size: 12.5px; border-radius: 2px; margin: 0 0 20px;
}
section { margin-bottom: 22px; }
h2 {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 8px;
}
table { border-collapse: collapse; width: 100%; }
td { padding: 5px 0; vertical-align: middle; border-bottom: 1px solid var(--line); }
td.k { width: 260px; font-size: 13px; }
td.p { width: auto; }
td.m { width: 110px; text-align: right; font-size: 12px; color: var(--muted); }
.bar {
  display: inline-block; width: calc(100% - 78px); height: 9px;
  background: var(--wash); border: 1px solid var(--line); border-radius: 2px;
  overflow: hidden; vertical-align: middle;
}
.bar span { display: block; height: 100%; background: var(--blue); }
.n {
  display: inline-block; width: 70px; text-align: right; font-size: 12px;
  font-variant-numeric: tabular-nums; color: var(--ink);
}
.n .of { color: var(--faint); }
footer { color: var(--faint); font-size: 11.5px; padding-top: 8px; }
</style></head><body>
<header>
  <h1>Run in progress</h1>
  <div class="sub">this page reloads every {{REFRESH}}s · close it any time, the run does not depend on it</div>
</header>
<main>
  <div class="headline">
    <span class="pct">{{PERCENT}}%</span>
    <span>{{DONE}}<span class="of">/{{EXPECTED}} answers</span></span>
  </div>
  {{OVERALL}}
  <p class="meta">
    elapsed <b>{{ELAPSED}}</b> · remaining <b>{{ETA}}</b> at the current rate
    &nbsp;·&nbsp; {{OUTCOMES}}
  </p>
  {{NOTE}}
  {{SECTIONS}}
  <footer>last written {{STAMP}} — if this stamp stops advancing, the run is stuck, not finishing.</footer>
</main></body></html>
"""
