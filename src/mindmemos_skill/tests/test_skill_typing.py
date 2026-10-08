"""Validate source metadata and snapshot integrity without rewriting SKILL.md."""

import pytest
from mindmemos_skill.skill.typing import SkillContent, SkillSnapshot
from pydantic import ValidationError


def test_standard_content_preserves_original_text_and_resources():
    """Parse display metadata while keeping CRLF text and binary assets intact."""
    content = "---\r\nname: Example\r\ndescription: Explain examples\r\nlicense: MIT\r\n---\r\nBody\r\n"
    source = SkillContent(content=content, resources={"assets/data.bin": b"\x00\xff"})
    snapshot = SkillSnapshot.from_content(source)
    assert source.name == "Example"
    assert source.description == "Explain examples"
    assert snapshot.content == content
    assert snapshot.resources == {"assets/data.bin": "AP8="}
    assert snapshot.to_content() == source


@pytest.mark.parametrize(
    "content",
    [
        "Body",
        "---\nname: Example\n---\nBody",
        "---\ndescription: Example\n---\n",
        "---\nname: ''\ndescription: Example\n---\n",
    ],
)
def test_standard_content_requires_valid_frontmatter(content):
    """Reject a standard Skill whose required YAML metadata is absent."""
    with pytest.raises(ValidationError):
        SkillContent(content=content)


def test_standard_content_rejects_conflicting_explicit_metadata():
    """Reject caller metadata that disagrees with the source document."""
    with pytest.raises(ValidationError, match="disagrees"):
        SkillContent(content="---\nname: A\ndescription: D\n---\n", name="B")


def test_nonstandard_content_requires_separate_metadata_and_stays_unchanged():
    """Preserve plain Markdown without adding synthesized frontmatter."""
    with pytest.raises(ValidationError):
        SkillContent(content="Plain content", standard_format=False, name="A")
    source = SkillContent(content="Plain content\r\n", standard_format=False, name="A", description="D")
    snapshot = SkillSnapshot.from_content(source)
    assert snapshot.content == "Plain content\r\n"
    assert snapshot.to_content() == source


def test_hashes_ignore_display_metadata_but_detect_content_changes():
    """Keep display fields outside both source hashes."""
    first = SkillContent(content="x", name="A", description="D", standard_format=False)
    renamed = SkillContent(content="x", name="B", description="E", standard_format=False)
    changed = SkillContent(content="y", name="A", description="D", standard_format=False)
    assert (first.content_hash, first.skill_hash) == (renamed.content_hash, renamed.skill_hash)
    assert (first.content_hash, first.skill_hash) != (changed.content_hash, changed.skill_hash)


@pytest.mark.parametrize("field", ["content", "resources", "content_hash", "skill_hash"])
def test_snapshot_rejects_tampering(field):
    """Reject source or digest changes before restoring a persisted snapshot."""
    snapshot = SkillSnapshot.from_content(
        SkillContent(content="body", name="A", description="D", standard_format=False)
    )
    raw = snapshot.model_dump()
    raw[field] = {"asset": "YQ=="} if field == "resources" else "changed"
    with pytest.raises(ValidationError):
        SkillSnapshot.model_validate(raw)
