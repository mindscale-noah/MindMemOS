"""Validate the new public Skill source request shape."""

import pytest
from mindmemos_skill.api.typing import RegisterRequest, StructuredSkillInfo
from pydantic import ValidationError


def test_structured_registration_uses_source_text_and_frontmatter():
    """Resolve standard metadata while preserving complete text and resources."""
    content = "---\r\nname: Example\r\ndescription: Example skill\r\nlicense: MIT\r\n---\r\nBody\r\n"
    request = RegisterRequest(
        project_id="project",
        skill=StructuredSkillInfo(content=content, resources={"asset.bin": "AP8="}),
    )
    assert request.skill.content == content
    assert request.skill.name == "Example"
    assert request.skill.description == "Example skill"
    assert request.skill.metadata == {"license": "MIT"}
    assert request.skill.resources == {"asset.bin": "AP8="}


def test_structured_registration_rejects_server_file_path():
    """Keep local paths out of the portable registration request."""
    with pytest.raises(ValidationError):
        RegisterRequest.model_validate({"project_id": "project", "skill": {"path": "/tmp/SKILL.md"}})


def test_structured_nonstandard_content_requires_explicit_display_metadata():
    """Never synthesize frontmatter for plain content."""
    with pytest.raises(ValidationError):
        StructuredSkillInfo(content="plain", standard_format=False, name="A")
    source = StructuredSkillInfo(content="plain", standard_format=False, name="A", description="D")
    assert source.content == "plain"
