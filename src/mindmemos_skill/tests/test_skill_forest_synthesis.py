"""Verify body-only synthesis without provider calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.algos.train.skill_forest.env_adapter import SkillForestEnvAdapter
from mindmemos_skill.algos.train.skill_forest.synthesis import split_merge_content
from mindmemos_skill.algos.train.skill_forest.trainer import SkillForestTrainer
from mindmemos_skill.algos.train.skill_forest.typing import Node
from mindmemos_skill.skill import Skill, SkillContent

from .test_skill_forest import config


@pytest.mark.parametrize("standard", [True, False])
@pytest.mark.parametrize("body", ["Plain instructions.", "- Follow this rule.", "# Instructions\nDo this."])
@pytest.mark.parametrize("response_header", ["", "---\nname: generated-private\ndescription: unwanted\n---\n"])
async def test_merge_only_exposes_bodies_and_preserves_initial_metadata(standard, body, response_header, tmp_path):
    header = "---\nname: initial-private\ndescription: original-private\ncustom: retained\n---\n"
    initial = Skill(
        source=SkillContent(
            content=header + "# Original" if standard else "# Original",
            standard_format=standard,
            name=None if standard else "nonstandard",
            description=None if standard else "External description",
        )
    )
    candidates = [
        SimpleNamespace(
            skill=Skill(
                source=SkillContent(content=f"---\nname: private-{side}\ndescription: secret-{side}\n---\nBody {side}")
            ),
            score=1.0,
            owner=Node(id="owner", tree_id=0, skill=initial, scores={}),
        )
        for side in ("left", "right")
    ]
    runtime = SimpleNamespace(
        chat=AsyncMock(return_value=response_header + body),
        env_adapter=SkillForestEnvAdapter.build("generic", {}),
    )
    trainer = SkillForestTrainer(config())
    trainer._score_ensemble_skill = AsyncMock(return_value=1.0)
    result = await trainer._merge_ensemble_pair(*candidates, "merge", initial, {}, None, runtime, tmp_path)
    messages = runtime.chat.call_args.kwargs["messages"]
    request = "\n".join(message["content"] for message in messages)
    for marker in (
        "private-left",
        "private-right",
        "secret-left",
        "secret-right",
        "initial-private",
        "original-private",
    ):
        assert marker not in request
    assert "Body left" in request and "Body right" in request
    assert result.skill.content == (header if standard else "") + body
    assert trainer._score_ensemble_skill.call_args.args[0].content == result.skill.content


@pytest.mark.parametrize(
    "text", ["---\nA horizontal rule followed by prose.", "---\n# Heading\nParagraph.", "Plain text", "- A list"]
)
def test_plain_markdown_is_preserved(text):
    assert split_merge_content(text) == ("", text)


@pytest.mark.parametrize("text", ["---\nname: broken\n", "---\nname: [broken\n---\nBody"])
def test_malformed_metadata_is_rejected(text):
    with pytest.raises(ValueError, match="frontmatter"):
        split_merge_content(text)


@pytest.mark.parametrize("response", [" ", "---\nname: generated\ndescription: metadata only\n---\n"])
async def test_empty_synthesis_body_is_rejected(response, tmp_path):
    initial = Skill(source=SkillContent(content="# Original", name="test", description="test", standard_format=False))
    candidate = SimpleNamespace(skill=initial, score=1, owner=None)
    runtime = SimpleNamespace(
        chat=AsyncMock(return_value=response), env_adapter=SkillForestEnvAdapter.build("generic", {})
    )
    with pytest.raises(ValueError, match="empty body"):
        await SkillForestTrainer(config())._merge_ensemble_pair(
            candidate, candidate, "merge", initial, {}, None, runtime, tmp_path
        )
