from agentic_eval.content.document import parse_answer_document, table_cell_coverage


def test_markdown_table_expands_cells_and_tracks_coverage():
    blocks = parse_answer_document(
        """## Results

| Month | TSR | Returns |
|---|---:|---:|
| May | 0.72 | 2 |
| June | 0.68 | 0 |
"""
    )
    table = next(block for block in blocks if block["type"] == "table")
    assert len(table["data_cells"]) == 4
    claims = [
        {
            "block_id": table["block_id"],
            "source_locator": {"row": cell["row"], "column": cell["column"]},
        }
        for cell in table["data_cells"]
    ]
    assert table_cell_coverage(blocks, claims) == 1.0


def test_long_sentence_stays_one_block_for_clause_level_llm_split():
    blocks = parse_answer_document(
        "TSR rose to 0.72 in May, returns increased to two, and risk increased."
    )
    assert len(blocks) == 1
    assert blocks[0]["type"] == "paragraph"


def test_table_placeholders_are_not_eligible_atomic_facts():
    blocks = parse_answer_document(
        """| Month | TSR | CDSS |
|---|---:|---:|
| May | 0.72 | -- |
| June | ↑ | N/A |
"""
    )
    table = next(block for block in blocks if block["type"] == "table")
    assert [(cell["row"], cell["column"], cell["value"]) for cell in table["data_cells"]] == [
        (1, 1, "0.72"),
        (2, 1, "↑"),
    ]
