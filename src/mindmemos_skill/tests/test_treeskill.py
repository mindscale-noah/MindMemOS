"""TreeSkill subtree selection, full-Skill execution, and failure boundaries."""

import json
from unittest.mock import patch

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.algos.train.treeskill.construction.markdown import (
    ancestor_closure,
    collapse_ancestor_descendant_selections,
    parse_skill_markdown,
    render_selected_subtrees,
)
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillContent, SkillSet
from pydantic import BaseModel


class EnvironmentOptions(BaseModel):
    """Settings for the deterministic test environment."""


CONTENT = """---
name: xlsx
---
Preamble.

# Skill
Root intro.

## Formula
Use formulas.

### Verification
Read references/check.md.

## Formatting
Apply number formats.

```python
# Not a heading
```
"""


class Router(OpenAIChatAgent):
    """Provide deterministic responses without provider calls."""

    def __init__(self, output):
        super().__init__(OpenAIChatConfig(model="test-router", temperature=0, max_completion_tokens=512))
        self.output = output
        self.calls = []

    async def run(self, *, messages):
        self.calls.append((messages[0]["content"], messages[1]["content"]))
        if isinstance(self.output, Exception):
            raise self.output
        return AgentResult(output=self.output, trajectory=[], usage={"total_tokens": 17})


def make_skill(output):
    router = Router(output)
    skill = TreeSkill(
        TreeSkillConfig(router=router.config),
        source=SkillContent(
            content=CONTENT,
            resources={"references/check.md": b"Original reference"},
            name="test-skill",
            description="Task instructions",
            standard_format=False,
        ),
        router=router,
    )
    return skill, router


async def test_subtree_selection_preserves_ancestors_and_uses_public_task():
    skill, router = make_skill('{"selected_subtree_ids":["003"],"rationale":"verification"}')
    result = await skill.route(Task(id="one", instruction="Verify the formulas"))
    assert result.selected_subtree_ids == ["003"]
    assert result.ancestor_ids == ["001", "002"]
    text = result.content
    assert "Root intro." in text and "Use formulas." in text and "references/check.md" in text
    assert "Formatting" not in text and "Not a heading" not in text
    assert result.response.usage["total_tokens"] == 17
    task_text, tree_text = router.calls[0][1].split("Task:\n", 1)[1].split("\n\nSkill tree:\n", 1)
    assert json.loads(task_text) == {"id": "one", "instruction": "Verify the formulas"}
    assert json.loads(tree_text)[0]["local_content"] == "Root intro."
    assert skill.content == CONTENT


def test_parent_selection_is_not_repeated_and_no_top_k_limit():
    tree = parse_skill_markdown(CONTENT)
    selected = collapse_ancestor_descendant_selections(tree, ["003", "002", "003"])
    ancestors = ancestor_closure(tree, selected)
    text = render_selected_subtrees(tree, selected)
    assert selected == ("002",) and ancestors == ("001",)
    assert text.count("### Verification") == 1 and "Formatting" not in text
    content = "# Root\nIntro.\n" + "\n".join(f"## Part {i}\nBody {i}." for i in range(10))
    tree = parse_skill_markdown(content)
    ids = [f"{index:03d}" for index in range(2, 12)]
    assert collapse_ancestor_descendant_selections(tree, ids) == tuple(ids)


def test_fences_and_empty_headings_are_not_routing_nodes():
    tree = parse_skill_markdown(CONTENT + "\n## Empty\n\n## Code\n~~~~\n# Inside fence\n~~~~\n")
    assert [node.heading for node in tree.nodes] == ["Skill", "Formula", "Verification", "Formatting", "Code"]
    assert tree.root_nodes[0].node_id == "001"


@pytest.mark.parametrize("output", ['{"selected_subtree_ids":[]}', '```json\n{"selected_subtree_ids":[]}\n```'])
async def test_explicit_empty_selection_is_not_full_skill_fallback(output):
    skill, _ = make_skill(output)
    result = await skill.route(Task(id="one", instruction="No applicable skill"))
    assert result.content == ""
    assert result.fallback_reason is None


@pytest.mark.parametrize("output", ["invalid", '{"selected_subtree_ids":["999"]}', '{"selected_subtree_ids":[3]}'])
async def test_invalid_selection_falls_back_to_source_and_reports_reason(output):
    skill, _ = make_skill(output)
    result = await skill.route(Task(id="one", instruction="Verify"))
    assert result.content == skill.content and result.fallback_reason
    assert result.response.output == output


async def test_provider_errors_use_reference_fallback_and_no_heading_needs_no_model():
    skill, router = make_skill(RuntimeError("Provider unavailable"))
    result = await skill.route(Task(id="one", instruction="Verify"))
    assert result.content == skill.content
    assert result.fallback_reason == "router_failed: RuntimeError: Provider unavailable"
    assert result.response is None
    plain = skill.with_content("Plain instructions")
    assert (await plain.assemble(Task(id="two", instruction="Verify"))).content == plain.content
    assert len(router.calls) == 1


async def test_snapshot_and_environment_preserve_references_and_route_only_once(tmp_path):
    skill, router = make_skill('{"selected_subtree_ids":["003"]}')
    restored = TreeSkill.from_snapshot(skill.snapshot(), router=router)
    assert isinstance(restored, TreeSkill) and restored.resources == skill.resources

    class LocalEnvironment(Environment[EnvironmentOptions]):
        async def _run(self, *, task, reward_data, skills, workspace):
            assert isinstance(skills[0], Skill)
            destination = skills[0].export(workspace / "skill")
            assert (destination / "references/check.md").read_bytes() == b"Original reference"
            return RolloutResult(
                output=skills[0].content,
                trajectory=[],
                reward=Reward(score=1, success=True, detail="Passed"),
            )

    env = LocalEnvironment(EnvironmentOptions())
    task = Task(id="one", instruction="Verify")
    result = await env.rollout(
        task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": restored}, max_concurrency=1)
    )
    assert (
        await env.rollout(
            task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": restored}, max_concurrency=1)
        )
    ) == result
    assert len(router.calls) == 1 and "Formatting" not in result.output
    assert restored.content == skill.content


async def test_default_skill_is_static_and_restorable():
    skill = Skill(
        source=SkillContent(
            content="Same instructions", name="test-skill", description="Task instructions", standard_format=False
        )
    )
    assert (await skill.assemble(Task(id="one", instruction="First"))).snapshot() == (
        await skill.assemble(Task(id="two", instruction="Second"))
    ).snapshot()
    assert Skill.from_snapshot(skill.snapshot()).content == skill.content


async def test_assemble_returns_complete_skill_and_environment_uses_that_object(tmp_path):
    skill, router = make_skill('{"selected_subtree_ids":["003"]}')
    task = Task(id="one", instruction="Verify")
    prepared = await skill.assemble(task)
    assert isinstance(prepared, TreeSkill) and prepared is not skill
    assert prepared.resources == skill.resources and prepared.router is router
    assert "Formatting" not in prepared.content
    assert "Formatting" in skill.content

    class IdentityEnvironment(Environment[EnvironmentOptions]):
        async def _run(self, *, task, reward_data, skills, workspace):
            assert skills[0] is prepared
            destination = skills[0].export(workspace / "skill")
            assert (destination / "references/check.md").read_bytes() == b"Original reference"
            return RolloutResult(output="ok", trajectory=[], reward=Reward(score=1, success=True, detail="Passed"))

    from unittest.mock import patch

    with patch.object(skill, "assemble", return_value=prepared) as assemble:
        (
            await IdentityEnvironment(EnvironmentOptions()).rollout(
                task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1)
            )
        )
        assemble.assert_called_once_with(task)


async def test_restore_and_export_defer_router_until_assembly(tmp_path):
    source, router = make_skill('{"selected_subtree_ids":["003"]}')
    snapshot = source.snapshot()
    with patch("mindmemos_skill.agents.Agent.from_config", return_value=router) as factory:
        restored = TreeSkill.from_snapshot(snapshot)
        destination = restored.export(tmp_path / "exported")
        assert restored.snapshot() == snapshot
        assert (destination / "references/check.md").read_bytes() == b"Original reference"
        plain = restored.with_content("Plain instructions")
        assert (await plain.assemble(Task(id="plain", instruction="Read"))).content == plain.content
        factory.assert_not_called()
        prepared = await restored.assemble(Task(id="route", instruction="Verify"))
        factory.assert_called_once_with(restored.config.router)
        assert len(router.calls) == 1
        assert "Formatting" not in prepared.content
        assert restored.snapshot() == snapshot


async def test_snapshot_restoration_preserves_explicit_router_credentials():
    """Restore the complete router configuration without credential reinjection."""
    skill, router = make_skill('{"selected_subtree_ids":["003"],"rationale":"verification"}')
    router.config.api_key = "configured-secret"
    router.config.base_url = "https://configured.invalid/v1"
    skill.config.router = router.config
    snapshot = skill.snapshot()
    assert snapshot.config["router"]["api_key"] == "configured-secret"
    restored = TreeSkill.from_snapshot(snapshot)
    assert restored.config.router.api_key == "configured-secret"
    restored = TreeSkill.from_snapshot(snapshot, router=router)
    assert restored.config.router.api_key == "configured-secret"
    assert "Verification" in (await restored.assemble(Task(id="one", instruction="Verify"))).content
