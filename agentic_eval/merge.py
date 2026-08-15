"""Joining runs that covered different cases into one comparable run.

`run` has no resume: it writes a fresh folder each time. For a long pass that
is fine as long as the work can be split, and the natural seam is the CASE —
a worker already owns whole cases, metrics pool over them, and every record
carries its own `case_id`. So "do case A today, case B tomorrow" works: run
each separately, join the two `runs.jsonl` files, and score the result.

Concatenating them by hand also works, right up until it doesn't. This module
exists for the two ways that goes wrong silently:

  * DUPLICATES. Re-running a case that is already in the pile does not
    overwrite anything — it appends. The question then has twice the answers,
    every count doubles, and consistency compares a run against itself. The
    file looks fine and the page looks fine.

  * MISMATCHED RUNS. Two runs of different configs, or with baseline and
    candidate swapped, join without complaint and the comparison silently
    stops meaning anything.

Both are refused here rather than reported, because a merged file is an input
to everything downstream and nothing after this point can tell.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: What makes a record unique. Same shape as the content pipeline's `identity`,
#: for the same reason: `case_id` is part of it, or a second case looks like a
#: repeat of the first.
_IDENTITY = (
    "system", "mode", "case_id", "question_set", "name", "run_index",
)

#: Manifest fields that must agree, because a comparison built from runs that
#: disagreed on them is not a comparison.
_MUST_MATCH = ("baseline", "candidate", "mode")


def identity(record: dict[str, Any]) -> tuple:
    return tuple(record.get(field) for field in _IDENTITY)


def _describe(record: dict[str, Any]) -> str:
    return (
        f"{record.get('system')} · {record.get('name')} · "
        f"case {record.get('case_id')!r} · repeat {record.get('run_index')}"
    )


def merge(
    sources: list[tuple[list[dict[str, Any]], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Records and a manifest for the joined run, or an error saying why not.

    Takes already-read (records, manifest) pairs so the caller owns the I/O and
    this stays testable without a filesystem.
    """
    if not sources:
        raise ValueError("nothing to merge")

    base_manifest = sources[0][1]
    for _records, manifest in sources[1:]:
        for field in _MUST_MATCH:
            first, other = base_manifest.get(field), manifest.get(field)
            if first != other:
                raise ValueError(
                    f"these runs disagree about {field}: {first!r} vs {other!r}. "
                    "Merging them would compare two different experiments"
                )

    merged: list[dict[str, Any]] = []
    seen: dict[tuple, int] = {}
    for index, (records, _manifest) in enumerate(sources):
        for record in records:
            key = identity(record)
            if key in seen:
                raise ValueError(
                    f"{_describe(record)} appears in source {seen[key] + 1} "
                    f"and source {index + 1}. Merging would count that answer "
                    "twice — every rate over it, and consistency would compare "
                    "the run against itself"
                )
            seen[key] = index
            merged.append(record)

    cases = sorted({
        str(record.get("case_id")) for record in merged
        if record.get("case_id") is not None
    })
    manifest = {
        **base_manifest,
        "cases": cases,
        # What this was built from, so a reader can tell a merged run from one
        # that ran in a single pass — the latency numbers differ in kind.
        "merged_from": len(sources),
        "merged_records": len(merged),
    }
    return merged, manifest


def _filter(
    records: list[dict[str, Any]], field: str, label: str,
    include: list[str] | None, exclude: list[str] | None,
) -> list[dict[str, Any]]:
    """Keep or drop by one field, refusing a name that is not there.

    Silence would drop nothing and look exactly like success. The name most
    likely to be wrong is the case id whose real value ends in a space, so the
    message says to look for it.
    """
    if include and exclude:
        raise ValueError(
            f"give --{label} or --exclude-{label}, not both: two filters that "
            "disagree have no obvious answer"
        )
    if not include and not exclude:
        return records
    known = {str(record.get(field)) for record in records}
    for named in set(include or ()) | set(exclude or ()):
        if named not in known:
            raise ValueError(
                f"no {label} {named!r} in this run; it has {sorted(known)}. "
                "Check for a trailing space"
            )
    if include:
        wanted = set(include)
        return [r for r in records if str(r.get(field)) in wanted]
    unwanted = set(exclude)
    return [r for r in records if str(r.get(field)) not in unwanted]


def select(
    records: list[dict[str, Any]], *,
    cases: list[str] | None = None, exclude_cases: list[str] | None = None,
    questions: list[str] | None = None,
    exclude_questions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """The records worth keeping, by case and by question.

    Two uses, and they are different in kind.

    DROPPING A CASE whose data tables are incomplete: both systems answer it
    badly for a reason that is neither system's, and pooled with the rest it
    moves every rate, so a difference in the fixture reads as a difference in
    quality.

    DROPPING A QUESTION so its answers can be replaced by a fresh run. `merge`
    refuses duplicates, so the old answers have to go first. Note what that
    costs in `stateful` mode: a question re-run on its own is turn 1 of its own
    session, not turn N of the original conversation, so the spliced answers
    were produced under different conditions than the ones around them. For a
    question with no parent that is usually acceptable; for a follow-up it is
    not, and the chain guard on `run` will refuse the selection anyway.

    Either way this belongs in a COPY of the run rather than a flag on each
    reader: `rescore` and `compare-answers` would both have to be given the
    same filter every time, and forgetting one produces a page whose metrics
    describe a different set of answers than its own tables do.
    """
    kept = _filter(records, "case_id", "case-id", cases, exclude_cases)
    return _filter(kept, "name", "question", questions, exclude_questions)


def read_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A run's records and manifest, given either path."""
    runs = path if path.is_file() else path / "runs.jsonl"
    if not runs.is_file():
        raise FileNotFoundError(f"no runs.jsonl at {runs}")
    manifest_path = runs.parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    records = [
        json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, manifest
