"""Which cases a run covers, and where the list comes from.

A case is the subject of a run: one customer, one folder of tables. Asking the
same question set about several cases is how a difference between two systems
is separated from a quirk of one customer's data — a candidate that wins on one
case and loses on two has not won.

Discovery reads ONE directory and takes only the ID LIST from it. It is not a
data source: each system under test serves the case from its own checkout's
`data_tables/`, and AgenticEval never reads a table. Pointing `--cases-from` at
AgenticSys_v2 asks "which cases exist?", and both the baseline and the
candidate then answer about those ids using their own copies of the data.

That distinction matters when the two checkouts differ. If a case id exists in
the directory that was listed but not in a system's own tables, that system
fails on it — which is a finding about that checkout, not about discovery, and
is left to surface as a failed turn rather than filtered out here.
"""
from __future__ import annotations

from pathlib import Path


def discover_case_ids(data_dir: str | Path) -> list[str]:
    """Case ids under a `data_tables/real`-shaped directory, in sorted order.

    The id is the directory name EXACTLY as it appears on disk, because that is
    what the system under test uses as its key: `LocalDataGateway`'s
    `from_case_folders` stores `case_dir.name` verbatim. One of the real case
    folders has a trailing space in its name, and normalising it here would
    produce an id that matches no case — the run would complete with every
    answer empty, which reads as a system failure rather than our own edit.

    `whitespace_padded_case_ids` reports such names so they can be flagged,
    since a caller pasting one into a shell will lose the padding silently.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"case directory does not exist: {root}")
    ids = sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if not ids:
        raise ValueError(f"no case directories under {root}")
    return ids


def whitespace_padded_case_ids(case_ids: list[str]) -> list[str]:
    """Ids whose leading/trailing whitespace is real and load-bearing."""
    return [case_id for case_id in case_ids if case_id != case_id.strip()]


def describe_case(case_id: str | None) -> str:
    """A case id rendered so invisible padding is visible.

    Used in console output and page labels: `11854808010` and `11854808010 `
    are different cases and print identically, so the padded one is shown
    quoted.
    """
    if case_id is None:
        return "(unset)"
    return repr(case_id) if case_id != case_id.strip() else case_id
