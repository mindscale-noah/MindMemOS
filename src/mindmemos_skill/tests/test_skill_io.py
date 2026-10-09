"""Exercise complete local Skill source capture and export."""

from pathlib import Path

import pytest
from mindmemos_skill.skill import Skill, SkillContent


def test_local_file_and_structured_source_are_equivalent(tmp_path: Path):
    """Preserve CRLF text and binary resources across both input paths."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    content = "---\r\nname: Example\r\ndescription: Example skill\r\n---\r\nBody\r\n"
    (source_dir / "SKILL.md").write_bytes(content.encode("utf-8"))
    (source_dir / "assets").mkdir()
    (source_dir / "assets" / "logo.bin").write_bytes(b"\x00\xff")
    from_file = Skill.from_path(source_dir)
    structured = Skill(SkillContent(content=content, resources={"assets/logo.bin": b"\x00\xff"}))
    assert from_file.snapshot() == structured.snapshot()
    restored = Skill.from_snapshot(from_file.snapshot())
    destination = restored.export(tmp_path / "export")
    assert (destination / "SKILL.md").read_bytes() == content.encode("utf-8")
    assert (destination / "assets" / "logo.bin").read_bytes() == b"\x00\xff"


def test_missing_or_invalid_local_skill_does_not_create_destination(tmp_path: Path):
    """Reject absent or malformed entrypoints before any export is attempted."""
    with pytest.raises((OSError, ValueError)):
        Skill.from_path(tmp_path / "missing")
    source_dir = tmp_path / "invalid"
    source_dir.mkdir()
    (source_dir / "SKILL.md").write_text("plain content")
    with pytest.raises(ValueError):
        Skill.from_path(source_dir)
    assert not (tmp_path / "export").exists()


def test_content_replacement_retains_resources_and_parent(tmp_path: Path):
    """Keep original Skill untouched while replacing candidate content."""
    first = "---\nname: Example\ndescription: First\n---\nA\n"
    second = "---\nname: Example\ndescription: Second\n---\nB\n"
    parent = Skill(SkillContent(content=first, resources={"x.bin": b"\0"}))
    candidate = parent.with_content(second)
    assert parent.source.content == first
    assert candidate.source.content == second
    assert parent.source.resources == candidate.source.resources
    assert parent.snapshot().content_hash != candidate.snapshot().content_hash
    assert parent.snapshot().skill_hash != candidate.snapshot().skill_hash
