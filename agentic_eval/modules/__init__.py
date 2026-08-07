"""One module per evaluation dimension.

Each exposes `section(rows)` returning that dimension's metrics for the k
repeats of a single system/mode/question, so `scoring.aggregate` composes them
rather than owning them, and a caller can select dimensions by name.
"""
from agentic_eval.modules import consistency, content, latency, memory

#: Dimensions selectable by name, e.g. via `--eval-module`.
EVAL_MODULES = {
    "consistency": consistency,
    "content": content,
    "latency": latency,
    "memory": memory,
}

#: Accepted in place of a module list to mean "every dimension".
ALL_MODULES = "all"


def resolve_modules(values: object) -> list[str]:
    """Normalize a module selection into an ordered, validated name list.

    Accepts a YAML list, repeated CLI flags, a comma-separated string, or the
    literal `all`. Unknown names raise rather than being skipped: a typo in a
    sweep script must not silently drop a dimension and leave a summary that
    looks complete.
    """
    if values is None:
        return list(EVAL_MODULES)
    if isinstance(values, str):
        values = [values]
    names: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if part == ALL_MODULES:
                return list(EVAL_MODULES)
            if part not in names:
                names.append(part)
    if not names:
        return list(EVAL_MODULES)
    unknown = [name for name in names if name not in EVAL_MODULES]
    if unknown:
        raise ValueError(
            f"unknown eval module(s): {', '.join(unknown)}; available: "
            f"{', '.join(sorted(EVAL_MODULES))}, or '{ALL_MODULES}'"
        )
    # Report in registry order so summaries are comparable across invocations.
    return [name for name in EVAL_MODULES if name in names]


__all__ = [
    "ALL_MODULES", "EVAL_MODULES", "resolve_modules",
    "consistency", "content", "latency", "memory",
]
