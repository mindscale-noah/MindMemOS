"""Exercise patch-level Skill-Forest recovery and workspace retention."""

import json
from collections import Counter

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer
from mindmemos_skill.algos.train.skill_forest import trainer as trainer_module
from mindmemos_skill.common.json import read_jsonl

from .agent_testing import agent_config
from .test_skill_forest import FakeEnvironment, FakeOptimizer, config, initial_skill, samples


async def execute(root, trainer, env):
    """Run one complete or interrupted local forest with stable inputs."""
    return await trainer.train(
        skill=initial_skill(), env=env, train_samples=samples(), validation_samples=[], work_dir=root
    )


def small_config(**updates):
    """Keep interruption tests focused on one root and one candidate."""
    return config(num_trees=1, expansion_budget=1, candidates_per_node=1, rollout_concurrency=1, **updates)


def candidate_dir(root):
    """Locate the only patch-applied candidate in a small run."""
    return root / "trees/tree-0/node-tree-000-root-00"


async def test_patch_applied_before_interruption_is_reused(tmp_path, monkeypatch):
    """A committed Skill snapshot survives while its validation restarts."""
    optimizer, env = FakeOptimizer(), FakeEnvironment()
    trainer = SkillForestTrainer(agent_config(monkeypatch, optimizer, small_config()))
    original = trainer_module.write_node
    interrupted = False

    def stop_after_generation(root, checkpoint):
        nonlocal interrupted
        original(root, checkpoint)
        if checkpoint.status == "generated" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(trainer_module, "write_node", stop_after_generation)
    with pytest.raises(ExceptionGroup):
        (await execute(tmp_path, trainer, env))
    assert interrupted
    saved = json.loads((candidate_dir(tmp_path) / "node.json").read_text())
    assert saved["status"] == "generated"
    model_calls = len(optimizer.messages)

    selected = await execute(tmp_path, trainer, env)
    assert "improved" in selected.content
    assert len(optimizer.messages) == model_calls
    assert json.loads((candidate_dir(tmp_path) / "node.json").read_text())["status"] == "accepted"
    assert (tmp_path / "result.json").exists()


async def test_partial_validation_restarts_from_first_task(monkeypatch, tmp_path):
    """A partial JSONL audit does not become a per-task recovery cache."""

    class InterruptedEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.candidate_calls = []
            self.failed = False

        async def rollout(self, **kwargs):
            directory = str(kwargs["work_dir"])
            if "node-tree-000-root-00/workspaces/eval" in directory and "selection-" not in directory:
                self.candidate_calls.append(kwargs["task"].id)
                if len(self.candidate_calls) == 2 and not self.failed:
                    self.failed = True
                    raise RuntimeError("interrupted candidate validation")
            return await super().rollout(**kwargs)

    env = InterruptedEnvironment()
    optimizer = FakeOptimizer()
    trainer = SkillForestTrainer(agent_config(monkeypatch, optimizer, small_config()))
    with pytest.raises(ExceptionGroup):
        (await execute(tmp_path, trainer, env))
    first_task = env.candidate_calls[0]
    assert len(list(read_jsonl(candidate_dir(tmp_path) / "evaluation.jsonl"))) == 1
    model_calls = len(optimizer.messages)

    (await execute(tmp_path, trainer, env))
    assert Counter(env.candidate_calls)[first_task] == 2
    assert len(optimizer.messages) == model_calls
    rows = list(read_jsonl(candidate_dir(tmp_path) / "evaluation.jsonl"))
    assert len(rows) == 3
    assert all("trajectory" in row["result"] for row in rows)


async def test_resume_rejects_changed_inputs_and_corrupt_nodes(monkeypatch, tmp_path):
    """A matching final result is reusable; mismatched inputs fail before calls."""
    env, optimizer = FakeEnvironment(), FakeOptimizer()
    trainer = SkillForestTrainer(agent_config(monkeypatch, optimizer, small_config()))
    selected = await execute(tmp_path, trainer, env)
    before = len(env.calls), len(optimizer.messages)
    assert (await execute(tmp_path, trainer, env)).source.content_hash == selected.source.content_hash
    assert (len(env.calls), len(optimizer.messages)) == before

    changed = SkillForestTrainer(agent_config(monkeypatch, optimizer, small_config(seed=99)))
    with pytest.raises(ValueError, match="identity mismatch"):
        (await execute(tmp_path, changed, env))
    assert (len(env.calls), len(optimizer.messages)) == before

    (tmp_path / "result.json").unlink()
    checkpoint_path = candidate_dir(tmp_path) / "node.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["scores"] = {}
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        (await execute(tmp_path, trainer, env))


async def test_config_snapshot_masks_credentials_and_still_supports_recovery(monkeypatch, tmp_path):
    """Credential rotation preserves recovery without exposing keys on disk."""
    settings = small_config()
    settings.optimizer.api_key = "forest-private-key"
    trainer = SkillForestTrainer(agent_config(monkeypatch, FakeOptimizer(), settings))
    env = FakeEnvironment()
    await execute(tmp_path, trainer, env)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["trainer"]["optimizer"]["api_key"] == "[REDACTED]"
    assert "forest-private-key" not in (tmp_path / "config.json").read_text()
    assert trainer.config.optimizer.api_key == "forest-private-key"
    before = len(env.calls)
    await execute(tmp_path, trainer, env)
    assert len(env.calls) == before
    changed = settings.model_copy(deep=True)
    changed.optimizer.api_key = "different-key"
    await execute(tmp_path, SkillForestTrainer(agent_config(monkeypatch, FakeOptimizer(), changed)), env)
    assert len(env.calls) == before
    changed.optimizer.model = "different-model"
    with pytest.raises(ValueError, match="identity mismatch"):
        await execute(tmp_path, SkillForestTrainer(agent_config(monkeypatch, FakeOptimizer(), changed)), env)


async def test_legacy_directory_is_not_silently_imported(tmp_path):
    """An old collection of audit files cannot masquerade as a new run."""
    (tmp_path / "forest.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        (await execute(tmp_path, SkillForestTrainer(small_config()), FakeEnvironment()))


@pytest.mark.parametrize("cleanup", [True, False])
async def test_workspace_cleanup_is_configurable(monkeypatch, tmp_path, cleanup):
    """Completed task workspaces are removed by default and retained on request."""

    class FileEnvironment(FakeEnvironment):
        async def rollout(self, **kwargs):
            directory = kwargs["work_dir"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "artifact.txt").write_text("task artifact")
            return await super().rollout(**kwargs)

    (
        await execute(
            tmp_path,
            SkillForestTrainer(agent_config(monkeypatch, FakeOptimizer(), small_config(cleanup_workspaces=cleanup))),
            FileEnvironment(),
        )
    )
    artifacts = list(tmp_path.rglob("artifact.txt"))
    assert bool(artifacts) is not cleanup


async def test_selection_failure_restarts_selection_without_reexpanding(monkeypatch, tmp_path):
    """Committed parent expansions survive a failed final selection stage."""
    env, optimizer = FakeEnvironment(), FakeOptimizer()
    trainer = SkillForestTrainer(agent_config(monkeypatch, optimizer, small_config()))
    original = trainer._ensemble
    interrupted = False

    async def stop_once(*args):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("selection unavailable")
        return await original(*args)

    trainer._ensemble = stop_once
    with pytest.raises(RuntimeError, match="selection unavailable"):
        (await execute(tmp_path, trainer, env))
    before = len(optimizer.messages)
    assert json.loads((tmp_path / "trees/tree-0/node-tree-000-root/node.json").read_text())["expanded"]
    (await execute(tmp_path, trainer, env))
    assert len(optimizer.messages) == before
    assert (tmp_path / "result.json").exists()
