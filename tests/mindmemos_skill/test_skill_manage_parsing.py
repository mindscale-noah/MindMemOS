"""AC-P2-001: parser/normalization contracts for ``skill_manage``."""

from __future__ import annotations

from pathlib import Path

import pytest
from mindmemos_skill.algos.skill_manage.errors import SkillManageError
from mindmemos_skill.algos.skill_manage.parsing import (
    normalize_content,
    parse_skill_document,
)

FIXTURE = Path(__file__).parent / "fixtures" / "skill_manage" / "gos_minimal" / "skills"

CSV_PARSER = """---
name: CSV Parser Skill
description: Parses raw CSV text into structured rows of records.
domain:
  - data-processing
  - text
inputs:
  - raw_csv_text
outputs:
  - parsed_records
---

# CSV Parser Skill

Parses raw CSV text into structured records ready for downstream processing.

## Inputs

- raw_csv_text

## Outputs

- parsed_records
"""


def test_parse_fixture_skill_matches_golden_fields() -> None:
    parsed = parse_skill_document(CSV_PARSER, source_path="skills/csv_parser.md", version_id="v1")

    assert parsed["name"] == "CSV Parser Skill"
    assert parsed["description"] == "Parses raw CSV text into structured rows of records."
    assert parsed["inputs"] == ["raw_csv_text"]
    assert parsed["outputs"] == ["parsed_records"]
    assert parsed["domain_tags"] == ["data-processing", "text"]
    assert parsed["source_path"] == "skills/csv_parser.md"
    assert "CSV Parser Skill" in parsed["rendered_snippet"]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("# No frontmatter here\n", "missing frontmatter"),
        ("---\nname: x\n---\n\n", "cannot derive description"),
        ("---\ndescription: only description\n---\n\nbody\n", "missing name"),
        ("---\nname: x\ndescription: y\nname: z\n---\n\nbody\n", "duplicate key"),
        ("---\n- a\n- b\n---\n\nbody\n", "frontmatter must be a mapping"),
    ],
)
def test_invalid_skill_document_raises_typed_error(content: str, reason: str) -> None:
    with pytest.raises(SkillManageError) as exc_info:
        parse_skill_document(content, version_id="v1")
    assert exc_info.value.error_code == "SKILL_MANAGE_INVALID_SKILL_DOCUMENT"
    assert reason in str(exc_info.value)


def test_input_too_large_raises_typed_error_not_truncation() -> None:
    content = "---\nname: x\ndescription: y\n---\n\n" + ("a" * 400)
    with pytest.raises(SkillManageError) as exc_info:
        parse_skill_document(content, version_id="v1", max_chars=100)
    assert exc_info.value.error_code == "SKILL_MANAGE_INPUT_TOO_LARGE"
    assert exc_info.value.details["actual"] > 100


def test_crlf_and_whitespace_normalization_is_stable() -> None:
    lf = parse_skill_document(CSV_PARSER, version_id="v1")
    crlf = parse_skill_document(CSV_PARSER.replace("\n", "\r\n"), version_id="v1")
    assert lf["description"] == crlf["description"]
    assert lf["rendered_snippet"] == crlf["rendered_snippet"]
    assert normalize_content("a\r\nb\rc") == "a\nb\nc"


def test_semantic_change_produces_different_rendered_snippet() -> None:
    first = parse_skill_document(CSV_PARSER, version_id="v1")
    second = parse_skill_document(CSV_PARSER.replace("structured records", "loose text"), version_id="v2")
    assert first["rendered_snippet"] != second["rendered_snippet"]


def test_empty_sections_normalize_to_empty_lists() -> None:
    content = "---\nname: X\ndescription: Does a thing\ninputs: []\noutputs: []\n---\n\nbody\n"
    parsed = parse_skill_document(content, version_id="v1")
    assert parsed["inputs"] == []
    assert parsed["outputs"] == []
