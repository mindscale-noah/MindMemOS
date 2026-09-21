"""TreeSkill subtree selection, full-Skill execution, and failure boundaries."""

import json
from unittest.mock import patch

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.algos.train.treeskill.tree import parse_tree, render_selection
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillBundle, SkillResources, SkillSet
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

    def run(self, *, system, user):
        self.calls.append((system, user))
        if isinstance(self.output, Exception):
            raise self.output
        return AgentResult(output=self.output, trajectory=[], usage={"total_tokens": 17})


def make_skill(output):
    router = Router(output)
    skill = TreeSkill(
        TreeSkillConfig(router=router.config),
        bundle=SkillBundle(files={"SKILL.md": CONTENT}),
        resources=SkillResources(files={"references/check.md": b"Original reference"}),
        router=router,
    )
    return skill, router


def test_subtree_selection_preserves_ancestors_and_uses_public_task():
    skill, router = make_skill('{"selected_subtree_ids":["003"],"rationale":"verification"}')
    result = skill._route(Task(id="one", instruction="Verify the formulas"))
    assert result.selected_subtree_ids == ["003"]
    assert result.ancestor_ids == ["001", "002"]
    text = result.bundle.files["SKILL.md"]
    assert "Root intro." in text and "Use formulas." in text and "references/check.md" in text
    assert "Formatting" not in text and "Not a heading" not in text
    assert result.response.usage["total_tokens"] == 17
    payload = json.loads(router.calls[0][1].split("\n\n", 1)[1])
    assert payload["task"] == {"id": "one", "instruction": "Verify the formulas"}
    assert payload["skill_tree"][0]["local_content"] == "Root intro."
    assert skill.bundle.files["SKILL.md"] == CONTENT


def test_parent_selection_is_not_repeated_and_no_top_k_limit():
    tree = parse_tree(CONTENT)
    text, selected, ancestors = render_selection(tree, ["003", "002", "003"])
    assert selected == ["002"] and ancestors == ["001"]
    assert text.count("### Verification") == 1 and "Formatting" not in text
    content = "# Root\nIntro.\n" + "\n".join(f"## Part {i}\nBody {i}." for i in range(10))
    tree = parse_tree(content)
    ids = [f"{index:03d}" for index in range(2, 12)]
    assert render_selection(tree, ids)[1] == ids


def test_fences_and_empty_headings_are_not_routing_nodes():
    tree = parse_tree(CONTENT + "\n## Empty\n\n## Code\n~~~~\n# Inside fence\n~~~~\n")
    assert [node.heading for node in tree.nodes()] == ["Skill", "Formula", "Verification", "Formatting", "Code"]
    assert tree.roots[0].id == "001"


@pytest.mark.parametrize("output", ['{"selected_subtree_ids":[]}', '```json\n{"selected_node_ids":[]}\n```'])
def test_explicit_empty_selection_is_not_full_skill_fallback(output):
    skill, _ = make_skill(output)
    result = skill._route(Task(id="one", instruction="No applicable skill"))
    assert result.bundle.files["SKILL.md"] == ""
    assert result.fallback_reason is None


@pytest.mark.parametrize("output", ["invalid", '{"selected_subtree_ids":["999"]}', '{"selected_subtree_ids":[3]}'])
def test_invalid_selection_falls_back_to_source_and_reports_reason(output):
    skill, _ = make_skill(output)
    result = skill._route(Task(id="one", instruction="Verify"))
    assert result.bundle == skill.bundle and result.fallback_reason
    assert result.response.output == output


def test_provider_errors_propagate_and_no_heading_needs_no_model():
    skill, router = make_skill(RuntimeError("Provider unavailable"))
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        skill.assemble(Task(id="one", instruction="Verify"))
    plain = skill.with_bundle(SkillBundle(files={"SKILL.md": "Plain instructions"}))
    assert plain.assemble(Task(id="two", instruction="Verify")).bundle == plain.bundle
    assert len(router.calls) == 1


def test_snapshot_and_environment_preserve_references_and_route_only_once(tmp_path):
    skill, router = make_skill('{"selected_subtree_ids":["003"]}')
    restored = TreeSkill.from_snapshot(skill.snapshot(), router=router)
    assert isinstance(restored, TreeSkill) and restored.resources == skill.resources

    class LocalEnvironment(Environment[EnvironmentOptions]):
        def _run(self, *, task, reward_data, skills, workspace):
            assert isinstance(skills[0], Skill)
            skills[0].export(workspace / "skill")
            assert (workspace / "skill/references/check.md").read_bytes() == b"Original reference"
            return RolloutResult(
                output=skills[0].bundle.files["SKILL.md"],
                trajectory=[],
                reward=Reward(score=1, success=True, detail="Passed"),
            )

    env = LocalEnvironment(EnvironmentOptions())
    task = Task(id="one", instruction="Verify")
    result = env.rollout(
        task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": restored}, max_concurrency=1)
    )
    assert (
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": restored}, max_concurrency=1))
        == result
    )
    assert len(router.calls) == 1 and "Formatting" not in result.output
    assert restored.bundle == skill.bundle


def test_default_skill_is_static_and_restorable():
    skill = Skill(bundle=SkillBundle(files={"SKILL.md": "Same instructions"}))
    assert (
        skill.assemble(Task(id="one", instruction="First")).snapshot()
        == skill.assemble(Task(id="two", instruction="Second")).snapshot()
    )
    assert Skill.from_snapshot(skill.snapshot()).bundle == skill.bundle


def test_assemble_returns_complete_skill_and_environment_uses_that_object(tmp_path):
    skill, router = make_skill('{"selected_subtree_ids":["003"]}')
    task = Task(id="one", instruction="Verify")
    prepared = skill.assemble(task)
    assert isinstance(prepared, TreeSkill) and prepared is not skill
    assert prepared.resources == skill.resources and prepared.router is router
    assert "Formatting" not in prepared.bundle.files["SKILL.md"]
    assert "Formatting" in skill.bundle.files["SKILL.md"]

    class IdentityEnvironment(Environment[EnvironmentOptions]):
        def _run(self, *, task, reward_data, skills, workspace):
            assert skills[0] is prepared
            skills[0].export(workspace / "skill")
            assert (workspace / "skill/references/check.md").read_bytes() == b"Original reference"
            return RolloutResult(output="ok", trajectory=[], reward=Reward(score=1, success=True, detail="Passed"))

    from unittest.mock import patch

    with patch.object(skill, "assemble", return_value=prepared) as assemble:
        IdentityEnvironment(EnvironmentOptions()).rollout(
            task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1)
        )
        assemble.assert_called_once_with(task)


def test_restore_and_export_defer_router_until_assembly(tmp_path):
    source, router = make_skill('{"selected_subtree_ids":["003"]}')
    snapshot = source.snapshot()
    with patch("mindmemos_skill.algos.train.treeskill.skill.OpenAIChatAgent", return_value=router) as factory:
        restored = TreeSkill.from_snapshot(snapshot)
        restored.export(tmp_path / "exported")
        assert restored.snapshot() == snapshot
        assert (tmp_path / "exported/references/check.md").read_bytes() == b"Original reference"
        plain = restored.with_bundle(SkillBundle(files={"SKILL.md": "Plain instructions"}))
        assert plain.assemble(Task(id="plain", instruction="Read")).bundle == plain.bundle
        factory.assert_not_called()
        prepared = restored.assemble(Task(id="route", instruction="Verify"))
        factory.assert_called_once_with(restored.config.router)
        assert len(router.calls) == 1
        assert "Formatting" not in prepared.bundle.files["SKILL.md"]
        assert restored.snapshot() == snapshot
