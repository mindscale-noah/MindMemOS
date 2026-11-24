"""Trainer-owned item recovery, direct agents and honest usage coverage."""

import asyncio
import json
from collections import Counter

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer, TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.artifacts import model_usage
from mindmemos_skill.algos.train.treeskill.construction.markdown import parse_skill_markdown
from mindmemos_skill.algos.train.treeskill.typing import LocatedEvidence
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.skill import Skill, SkillContent

from . import treeskill_smoke as smoke
from .agent_testing import agent_config
from .test_treeskill_training import make_sample


def source_skill():
    return Skill(
        source=SkillContent(
            content="# Workflow\nSave the output.\n",
            name="recovery",
            description="Offline recovery",
            standard_format=False,
        )
    )


@pytest.mark.asyncio
async def test_interrupted_localization_reuses_structured_analysis_and_rollout(monkeypatch, tmp_path):
    calls = Counter()
    interrupted = False

    class Agent(smoke.SmokeAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            nonlocal interrupted
            phase = messages[-1]["content"].split(" ", 1)[0]
            calls[phase] += 1
            if phase == "Locate" and not interrupted:
                interrupted = True
                raise asyncio.CancelledError("interrupted localization")
            return await super().run_messages(messages=messages, tools=tools, response_format=response_format)

    config = OpenAIChatConfig(model="offline", api_key="recovery-private-key")
    trainer = TreeSkillTrainer(
        agent_config(monkeypatch, Agent, TreeSkillTrainerConfig(optimizer=config, router=config))
    )
    env = smoke.SmokeEnvironment(smoke.SmokeConfig())
    kwargs = dict(skill=source_skill(), env=env, train_samples=[make_sample("train")], work_dir=tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await trainer.train(**kwargs)
    assert read_json(tmp_path / "config.json")["algorithm"]["optimizer"]["api_key"] == "[REDACTED]"
    assert "recovery-private-key" not in (tmp_path / "config.json").read_text()
    assert trainer.config.optimizer.api_key == "recovery-private-key"
    assert len(list((tmp_path / "items/analysis").glob("*/completed.json"))) == 1
    result = await trainer.train(**kwargs)
    assert "Verify the saved output." in result.content
    assert calls == {"Analyze": 1, "Locate": 2, "Fuse": 1}
    assert len(list((tmp_path / "rollouts/train").glob("*/attempt-*"))) == 1
    usage = model_usage(tmp_path)
    assert usage["calls"] == 3 and usage["total_tokens"] == 60
    assert usage["interrupted_items"] == 1 and not usage["usage_complete"]
    before = dict(calls)
    restored = await trainer.train(**kwargs)
    assert restored.content == result.content and dict(calls) == before
    assert type(restored.router) is Agent


def lessons():
    return [
        LocatedEvidence(
            instance_id="t",
            evidence_id=node,
            record_source="success",
            reusable_lesson="Verify results.",
            target_node_id=node,
            rationale="Relevant",
        )
        for node in ("001", "002")
    ]


@pytest.mark.asyncio
async def test_fusion_resume_preserves_new_child_ids_and_never_applies_prefix_twice(monkeypatch, tmp_path):
    calls = []
    interrupted = False

    class Agent(OpenAIChatAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            nonlocal interrupted
            # The deeper original node is fused before the root.
            calls.append(messages)
            if len(calls) == 2 and not interrupted:
                interrupted = True
                raise asyncio.CancelledError()
            if len(calls) == 1:
                edits = [
                    {
                        "operation": "create_child",
                        "new_child": {"heading": "Saved output", "content": "Reload it.", "children": []},
                        "rationale": "Verify",
                    }
                ]
            else:
                edits = [{"operation": "update_node", "content": "Inspect and verify.", "rationale": "Improve"}]
            return AgentResult(output=json.dumps({"rationale": "Improve", "edits": edits}))

    config = OpenAIChatConfig(model="offline")
    trainer = TreeSkillTrainer(
        agent_config(monkeypatch, Agent, TreeSkillTrainerConfig(optimizer=config, router=config))
    )
    initial = parse_skill_markdown("# Workbook\nInspect.\n## Edit\nSave.\n")
    with pytest.raises(asyncio.CancelledError):
        await trainer._fuse(initial, lessons(), tmp_path)
    checkpoint = next((tmp_path / "items/fusion").glob("*/completed.json"))
    saved = checkpoint.read_bytes()
    final, edits, failures = await trainer._fuse(initial, lessons(), tmp_path)
    assert not failures and len(edits) == 2
    assert len(calls) == 3
    assert final.full_content.count("Saved output") == 1
    assert final.node_by_id["003"].heading == "Saved output"
    assert final.node_by_id["002"].heading == "Edit"
    assert checkpoint.read_bytes() == saved
    assert len(list((tmp_path / "patches").glob("*/patch.json"))) == 2
    again, _, _ = await trainer._fuse(initial, lessons(), tmp_path)
    assert again == final and len(calls) == 3


@pytest.mark.asyncio
async def test_failed_fusion_retries_item_and_recomputes_uncommitted_suffix(monkeypatch, tmp_path):
    calls = 0

    class Agent(OpenAIChatAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary provider failure")
            return AgentResult(
                output=json.dumps({"rationale": "Keep", "edits": [{"operation": "reject", "rationale": "No change"}]})
            )

    config = OpenAIChatConfig(model="offline")
    trainer = TreeSkillTrainer(
        agent_config(monkeypatch, Agent, TreeSkillTrainerConfig(optimizer=config, router=config))
    )
    tree = parse_skill_markdown("# Workbook\nInspect.\n## Edit\nSave.\n")
    _, _, failures = await trainer._fuse(tree, lessons(), tmp_path)
    assert len(failures) == 1 and calls == 2
    assert not list((tmp_path / "items/fusion").glob("*/completed.json"))
    _, _, failures = await trainer._fuse(tree, lessons(), tmp_path)
    assert not failures and calls == 4
    assert len(list((tmp_path / "items/fusion").glob("*/completed.json"))) == 2
    assert model_usage(tmp_path)["failed_calls"] == 1


@pytest.mark.asyncio
async def test_patch_write_failure_propagates_and_never_commits_node(tmp_path, monkeypatch):
    from mindmemos_skill.algos.train.treeskill import trainer as module

    class Agent(OpenAIChatAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            return AgentResult(
                output=json.dumps({"rationale": "Keep", "edits": [{"operation": "reject", "rationale": "No change"}]})
            )

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, "record_patch", fail_write)
    config = OpenAIChatConfig(model="offline")
    trainer = TreeSkillTrainer(
        agent_config(monkeypatch, Agent, TreeSkillTrainerConfig(optimizer=config, router=config))
    )
    with pytest.raises(OSError, match="disk full"):
        await trainer._fuse(parse_skill_markdown("# Workbook\nInspect.\n"), lessons()[:1], tmp_path)
    assert not list((tmp_path / "items/fusion").glob("*/completed.json"))


@pytest.mark.asyncio
async def test_corrupt_checkpoint_is_rejected_before_reusing_completed_training(monkeypatch, tmp_path):
    config = OpenAIChatConfig(model="offline")
    trainer = TreeSkillTrainer(
        agent_config(monkeypatch, smoke.SmokeAgent, TreeSkillTrainerConfig(optimizer=config, router=config))
    )
    kwargs = dict(
        skill=source_skill(),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=tmp_path,
    )
    await trainer.train(**kwargs)
    checkpoint = next((tmp_path / "items/analysis").glob("*/completed.json"))
    saved = read_json(checkpoint)
    saved["payload"]["record"]["task_id"] = "corrupt"
    write_json(checkpoint, saved)
    with pytest.raises(ValueError, match="Recovery checkpoint mismatch"):
        await trainer.train(**kwargs)


@pytest.mark.asyncio
async def test_localization_reuses_independent_success_and_invalidates_changed_inputs(monkeypatch, tmp_path):
    from mindmemos_skill.algos.train.treeskill.typing import AnalysisItem, TrajectoryAnalysisRecord

    calls = Counter()

    class Agent(OpenAIChatAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            record = json.loads(messages[-1]["content"].split("\n\n", 1)[1])["analysis_record"]
            identifier = record["instance_id"]
            calls[identifier] += 1
            if identifier == "a" and calls[identifier] <= 2:
                raise RuntimeError("retry later")
            return AgentResult(
                output=json.dumps(
                    {
                        "instance_id": identifier,
                        "evidence": [
                            {
                                "evidence_id": "e1",
                                "reusable_lesson": "Verify saved results.",
                                "target_node_id": "001",
                                "rationale": "Relevant",
                            }
                        ],
                    }
                )
            )

    config = OpenAIChatConfig(model="offline")
    trainer = TreeSkillTrainer(
        agent_config(
            monkeypatch, Agent, TreeSkillTrainerConfig(optimizer=config, router=config, localization_concurrency=1)
        )
    )
    tree = parse_skill_markdown("# Workbook\nInspect.\n")
    analyses = [
        TrajectoryAnalysisRecord(
            instance_id=name,
            task_id=name,
            record_source="success",
            items=(AnalysisItem(item_id="i", kind="success_memory", content="Verify."),),
        )
        for name in ("a", "b")
    ]
    located, failures = await trainer._localize(tree, analyses, tmp_path)
    assert [item.instance_id for item in located] == ["b"] and len(failures) == 1
    located, failures = await trainer._localize(tree, analyses, tmp_path)
    assert not failures and len(located) == 2
    assert calls == {"a": 3, "b": 1}
    analyses[1] = analyses[1].model_copy(
        update={"items": (AnalysisItem(item_id="new", kind="success_memory", content="Reopen after saving."),)}
    )
    await trainer._localize(tree, analyses, tmp_path)
    assert calls == {"a": 3, "b": 2}
    assert len(list((tmp_path / "items/localization/item-b").glob("superseded-*.json"))) == 1


@pytest.mark.asyncio
async def test_direct_agent_keeps_parser_retries_and_aggregates_usage_once(monkeypatch, tmp_path):
    analysis_calls = 0

    class Agent(smoke.SmokeAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            nonlocal analysis_calls
            if messages[1]["content"].startswith("Analyze"):
                analysis_calls += 1
                if analysis_calls == 1:
                    return AgentResult(output="invalid JSON", usage={"total_tokens": 5})
                assert messages[-2]["content"] == "invalid JSON"
                messages = messages[:2]
            return await super().run_messages(messages=messages, tools=tools, response_format=response_format)

    config = OpenAIChatConfig(model="offline", parse_max_attempts=2)
    trainer = TreeSkillTrainer(
        agent_config(monkeypatch, Agent, TreeSkillTrainerConfig(optimizer=config, router=config))
    )
    await trainer.train(
        skill=source_skill(),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=tmp_path,
    )
    usage = model_usage(tmp_path)
    assert analysis_calls == 2
    assert usage["calls"] == 3 and usage["total_tokens"] == 65
    assert usage["usage_complete"]
    outcome = read_json(next((tmp_path / "items/analysis").glob("*/attempt-*/outcome.json")))
    assert outcome["responses"][0]["usage"]["parse_attempts"] == 2
    assert not (tmp_path / "model_calls").exists()
