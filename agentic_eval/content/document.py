"""Deterministic answer parsing before LLM claim extraction."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any


_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MISSING_CELL_VALUES = {"", "-", "--", "—", "n/a", "na", "null", "none"}


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, _attrs) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_block(
    rows: list[list[str]], *, block_id: str, section: str, source_type: str,
) -> dict[str, Any]:
    headers = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    data_cells = []
    for row_index, row in enumerate(body, 1):
        row_header = row[0] if row else ""
        for column_index, value in enumerate(row[1:], 1):
            # Placeholder cells communicate no asserted value, so they are
            # not atomic facts and must not inflate table-cell coverage.
            if value.strip().lower() in _MISSING_CELL_VALUES:
                continue
            column_header = headers[column_index] if column_index < len(headers) else f"column_{column_index}"
            data_cells.append({
                "row": row_index,
                "column": column_index,
                "row_header": row_header,
                "column_header": column_header,
                "value": value,
                "claim_text": f"{row_header} | {column_header}: {value}",
            })
    return {
        "block_id": block_id,
        "type": "table",
        "source_type": source_type,
        "section": section,
        "headers": headers,
        "rows": body,
        "data_cells": data_cells,
        "text": "\n".join(cell["claim_text"] for cell in data_cells),
    }


def parse_answer_document(answer: str) -> list[dict[str, Any]]:
    """Parse prose/lists/tables into stable blocks without interpreting claims."""
    lines = answer.splitlines()
    blocks: list[dict[str, Any]] = []
    section = ""
    paragraph: list[str] = []
    block_counter = 0

    def add(kind: str, text: str, **extra: Any) -> None:
        nonlocal block_counter
        text = " ".join(text.split())
        if not text:
            return
        block_counter += 1
        blocks.append({
            "block_id": f"b{block_counter:03d}",
            "type": kind,
            "section": section,
            "text": text,
            **extra,
        })

    def flush() -> None:
        if paragraph:
            add("paragraph", " ".join(paragraph))
            paragraph.clear()

    i = 0
    in_fence = False
    fence_lines: list[str] = []
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush()
            if in_fence:
                add("code", "\n".join(fence_lines), factual_default=False)
                fence_lines.clear()
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            fence_lines.append(raw)
            i += 1
            continue

        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            flush()
            section = heading.group(1).strip()
            add("heading", section, factual_default=False)
            i += 1
            continue

        if (
            stripped.startswith("|") and i + 1 < len(lines)
            and _TABLE_DIVIDER.match(lines[i + 1])
        ):
            flush()
            table_rows = [_cells(raw)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_rows.append(_cells(lines[i]))
                i += 1
            block_counter += 1
            blocks.append(_table_block(
                table_rows, block_id=f"b{block_counter:03d}",
                section=section, source_type="markdown_table",
            ))
            continue

        list_item = re.match(r"^\s*(?:[-*+] |\d+[.)] )(.+)$", raw)
        if list_item:
            flush()
            add("list_item", list_item.group(1))
            i += 1
            continue
        if stripped.startswith(">"):
            flush()
            add("quote", stripped.lstrip("> "))
            i += 1
            continue
        if not stripped:
            flush()
        else:
            paragraph.append(stripped)
        i += 1
    flush()
    if fence_lines:
        add("code", "\n".join(fence_lines), factual_default=False)

    if "<table" in answer.lower():
        parser = _HTMLTableParser()
        parser.feed(answer)
        for rows in parser.tables:
            if not rows:
                continue
            block_counter += 1
            blocks.append(_table_block(
                rows, block_id=f"b{block_counter:03d}",
                section=section, source_type="html_table",
            ))
    return blocks


def table_cell_coverage(blocks: list[dict[str, Any]], claims: list[dict[str, Any]]) -> float | None:
    eligible = {
        (block["block_id"], cell["row"], cell["column"])
        for block in blocks if block.get("type") == "table"
        for cell in block.get("data_cells") or []
    }
    if not eligible:
        return None
    covered = set()
    for claim in claims:
        locator = claim.get("source_locator") or {}
        key = (claim.get("block_id"), locator.get("row"), locator.get("column"))
        if key in eligible:
            covered.add(key)
    return len(covered) / len(eligible)
