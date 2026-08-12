"""The on-disk shape of a run folder, owned in one place.

Four writers contribute to a run — the runner, the content cascade, the review
packet builders, and the viewer — and when each picks its own filenames the
result is a flat pile where nothing says which artifact came from where, or
which are inputs to the next step. Every path lives here instead, so the layout
can be read (and changed) without grepping for string literals.

    <run>/
      manifest.json          what was run: systems, mode, repeats, seed
      runs.jsonl             raw records, the only irreplaceable artifact
      metrics/               aggregates over all repeats
      content/               the content cascade's output
      review/                blinded human-review packets and their keys
      logs/                  stdout of each launched system
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Every path in a run folder, derived from its root."""

    root: Path

    # -- index and raw data, kept at the root because everything else is
    #    derived from them and a reader should see them first.
    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs.jsonl"

    # -- aggregates over all repeats
    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def summary(self) -> Path:
        return self.metrics_dir / "summary.json"

    @property
    def comparison_json(self) -> Path:
        return self.metrics_dir / "comparison.json"

    @property
    def comparison_md(self) -> Path:
        return self.metrics_dir / "comparison.md"

    # -- content cascade. No `content_` prefixes: the folder already says it,
    #    and `content/content_summary.json` reads as a mistake.
    @property
    def content_dir(self) -> Path:
        return self.root / "content"

    @property
    def evaluations(self) -> Path:
        return self.content_dir / "evaluations.jsonl"

    @property
    def content_summary(self) -> Path:
        return self.content_dir / "summary.json"

    @property
    def content_comparison(self) -> Path:
        return self.content_dir / "comparison.md"

    @property
    def walkthrough(self) -> Path:
        return self.content_dir / "walkthrough.md"

    @property
    def answer_comparison(self) -> Path:
        return self.content_dir / "answer_comparison.html"

    @property
    def progress(self) -> Path:
        """Live progress, rewritten as each answer lands.

        In `content/` beside the viewer rather than at the root: it is a page
        you open in a browser, and it is the one artifact that exists mainly
        while the run is still going.
        """
        return self.content_dir / "progress.html"

    # -- human review. The two phases sit side by side rather than in separate
    #    folders: they are independently blinded but joined on `turn_id`, and
    #    keeping the keys adjacent is what makes that obvious.
    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def answer_review(self) -> Path:
        return self.review_dir / "answers.csv"

    @property
    def answer_review_key(self) -> Path:
        return self.review_dir / "answers.key.csv"

    @property
    def evidence_review(self) -> Path:
        return self.review_dir / "evidence.jsonl"

    @property
    def evidence_review_key(self) -> Path:
        return self.review_dir / "evidence.key.csv"

    # -- process output
    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def server_log(self, system: str) -> Path:
        return self.logs_dir / f"{system}.server.log"

    def ensure(self) -> RunLayout:
        """Create the folders a writer is about to use."""
        for directory in (
            self.root, self.metrics_dir, self.content_dir, self.review_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def find(cls, start: Path, *, depth: int = 3) -> RunLayout | None:
        """Locate the run containing `start`, by its manifest or its runs file.

        Artifacts live a level or two below the root, so a reader given any of
        them can still resolve its siblings.
        """
        directory = start if start.is_dir() else start.parent
        for _ in range(depth):
            if (directory / "manifest.json").is_file() or (
                directory / "runs.jsonl"
            ).is_file():
                return cls(directory)
            if directory.parent == directory:
                break
            directory = directory.parent
        return None
