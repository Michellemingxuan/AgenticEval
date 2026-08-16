"""Re-judging one question through 401s, with the rest left alone.

After an oracle or rubric fix the answers are still good and only their
verdicts are stale, so the cheap move is to re-judge that one question. Two
things make it a script rather than a flag:

  * `--resume` is a NO-OP for a subset re-judge. Every one of the question's
    answers is still present, so the pass judges nothing — and reports success.
    Measured on a real run: byte-identical file, `Content evaluation complete`.
  * without `--resume` a 401 partway through leaves the question half judged,
    and the retry must pick up rather than start over.

So the first attempt discards and judges, and every later attempt resumes.
`agentic-eval` is stubbed here: what is under test is that sequencing, not the
judge.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

_JUDGE = Path(__file__).resolve().parents[2] / "bin" / "judge"

_STUB = '''#!/usr/bin/env python3
"""Mimics evaluate-content's file semantics, and dies partway like a 401."""
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
runs = Path(argv[argv.index("--runs") + 1])
questions = {argv[i + 1] for i, v in enumerate(argv) if v == "--question"}
resume = "--resume" in argv
evals = runs.parent / "content" / "evaluations.jsonl"
Path(os.environ["STUB_LOG"]).open("a").write(" ".join(argv) + "\\n")

def key(row):
    return (row.get("system"), row.get("case_id"), row.get("name"),
            row.get("run_index"))

rows = [json.loads(l) for l in runs.open() if l.strip()]
want = [r for r in rows if not questions or r.get("name") in questions]
have = [json.loads(l) for l in evals.open() if l.strip()] if evals.exists() else []
if not resume and questions:
    have = [r for r in have if r.get("name") not in questions]
done = {key(r) for r in have}
todo = [r for r in want if key(r) not in done]
budget = len(todo) if resume else 1
for row in todo[:budget]:
    have.append({**row, "verdict": "fresh", "resumed": resume})
evals.write_text("".join(json.dumps(r) + "\\n" for r in have), encoding="utf-8")
sys.exit(1 if budget < len(todo) else 0)
'''


def _record(name, system="previous", run_index=1):
    return {
        "system": system, "mode": "stateful", "case_id": "366",
        "question_set": "series_a", "name": name, "run_index": run_index,
        "outcome": "ok", "final_answer": f"answer to {name}",
    }


def _run_folder(tmp_path):
    from agentic_eval.layout import RunLayout

    layout = RunLayout(tmp_path / "run").ensure()
    records = [
        _record(name, system=system, run_index=k)
        for name in ("a1", "b2") for system in ("previous", "current")
        for k in (1, 2)
    ]
    layout.runs.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    # Every answer already judged — the state a re-judge actually starts from,
    # and the one where `--resume` silently does nothing.
    layout.evaluations.write_text(
        "".join(json.dumps({**r, "verdict": "stale"}) + "\n" for r in records),
        encoding="utf-8")
    return layout


def _judge(tmp_path, layout, *args):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "agentic-eval"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        ["bash", str(_JUDGE), "--config", "cfg.yaml",
         "--runs", str(layout.runs), "--backoff", "0", *args],
        capture_output=True, text=True, timeout=120,
        env={"PATH": f"{stub_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
             "STUB_LOG": str(log)},
    )
    calls = log.read_text().splitlines() if log.exists() else []
    rows = [json.loads(l) for l in layout.evaluations.open() if l.strip()]
    return result, calls, rows


def test_a_question_is_rejudged_and_the_rest_are_kept(tmp_path):
    layout = _run_folder(tmp_path)

    result, calls, rows = _judge(tmp_path, layout, "--question", "a1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [r["verdict"] for r in rows if r["name"] == "a1"] == ["fresh"] * 4
    # The other question keeps the verdicts it already had. An earlier version
    # of this path truncated the file to what it had just judged.
    assert [r["verdict"] for r in rows if r["name"] == "b2"] == ["stale"] * 4


def test_the_first_attempt_discards_and_the_retry_resumes(tmp_path):
    """The whole point. `--resume` on attempt 1 would judge nothing at all."""
    layout = _run_folder(tmp_path)

    _result, calls, rows = _judge(tmp_path, layout, "--question", "a1")

    assert "--resume" not in calls[0]
    assert all("--resume" in call for call in calls[1:])
    assert all("--question a1" in call for call in calls)
    # One judged per failing attempt, the rest once the retry resumed.
    assert sum(r.get("resumed") is False for r in rows) == 1
    assert sum(r.get("resumed") is True for r in rows) == 3


def test_a_question_that_matches_no_answer_stops_before_writing(tmp_path):
    """A typo must not clear the question it meant and judge nothing back."""
    layout = _run_folder(tmp_path)
    before = layout.evaluations.read_bytes()

    result, calls, _rows = _judge(tmp_path, layout, "--question", "a9")

    assert result.returncode == 2
    assert "no answers for a9" in result.stderr
    assert calls == []
    assert layout.evaluations.read_bytes() == before


def test_workers_reach_the_judge(tmp_path):
    layout = _run_folder(tmp_path)

    _result, calls, _rows = _judge(
        tmp_path, layout, "--question", "a1", "--workers", "3")

    assert all("--workers 3" in call for call in calls)


def test_a_whole_run_still_resumes_from_the_first_attempt(tmp_path):
    """Without --question the behaviour is unchanged: everything resumes, and
    a fully judged run exits done without spending a call."""
    layout = _run_folder(tmp_path)

    result, calls, rows = _judge(tmp_path, layout)

    assert result.returncode == 0
    assert calls == []
    assert "done: 8/8" in result.stdout
    assert all(r["verdict"] == "stale" for r in rows)
