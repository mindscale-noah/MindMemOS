"""Offline algorithm and concurrency verification without provider calls."""

import asyncio
import itertools
import json
from collections import Counter

import pytest
from mindmemos_skill.agents import AgentResult
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.algos.train.skill_forest import (
    ALGORITHM_VERSION,
    SkillForestEnvAdapter,
    SkillForestTrainer,
    SkillForestTrainerConfig,
)
from mindmemos_skill.algos.train.skill_forest.tree import (
    apply_patch,
    cannot_improve,
    expansion_wave,
    novelty,
    split_tasks,
    tree_winners,
)
from mindmemos_skill.algos.train.skill_forest.typing import Node, Patch
from mindmemos_skill.envs.data.typing import DataSample, RewardData
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import ValidationError

from .agent_testing import agent_config


class PrivateAnswer(RewardData):
    """Private input that must never reach the optimizer."""

    answer: str = "PRIVATE-ANSWER-DO-NOT-SEND"
    """Sentinel for detecting training prompt leakage."""


class FakeEnvironment:
    """Observe actual worker concurrency and deterministic Skill-dependent rewards."""

    def __init__(self, *, fail=False, candidate_pass=True, merge_pass=True):
        self.lock = asyncio.Lock()
        self.active = 0
        self.peak = 0
        self.calls = []
        self.fail = fail
        self.candidate_pass = candidate_pass
        self.merge_pass = merge_pass

    async def rollout(self, *, task, reward_data, skill_set, work_dir, resume):
        assert resume is False
        assert isinstance(reward_data, PrivateAnswer)
        text = skill_set.skills["candidate"].content
        async with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((task.id, text, str(work_dir)))
        try:
            await asyncio.sleep(0.002)
            if self.fail:
                raise RuntimeError("environment unavailable")
            passed = self.merge_pass if "synthesized" in text else self.candidate_pass and "improved" in text
            return RolloutResult(
                output="public output",
                trajectory=[{"role": "assistant", "content": "public reasoning"}],
                reward=Reward(score=float(passed), success=passed, detail="PRIVATE-DETAIL-DO-NOT-SEND"),
            )
        finally:
            async with self.lock:
                self.active -= 1


class FakeOptimizer:
    """Observe model concurrency, exact prompts and pairwise synthesis calls."""

    def __init__(self, *, malformed=False, fail=False):
        self.lock = asyncio.Lock()
        self.active = 0
        self.peak = 0
        self.messages = []
        self.merges = 0
        self.malformed = malformed
        self.fail = fail

    def __call__(self, config):
        assert config.max_retries == 3
        assert config.parse_max_attempts == 3
        return self

    async def run(self, *, messages):
        async with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.messages.append(messages)
        try:
            await asyncio.sleep(0.002)
            if self.fail:
                raise RuntimeError("provider unavailable")
            if self.malformed:
                return AgentResult(output="malformed")
            if messages[1]["content"].startswith("## Skill Candidate 1"):
                async with self.lock:
                    self.merges += 1
                return AgentResult(output="# Skill\nsynthesized\n")
            patch = {"edits": [{"op": "append", "content": "improved"}]}
            if (
                "## Failed Trajectories" in messages[1]["content"]
                or "## Successful Trajectories" in messages[1]["content"]
            ):
                return AgentResult(output=json.dumps({"batch_size": 3, "patch": patch}))
            return AgentResult(output=json.dumps(patch))
        finally:
            async with self.lock:
                self.active -= 1


def config(**updates):
    """Build a small offline forest configuration."""
    return SkillForestTrainerConfig.model_validate(
        {
            "optimizer": {"model": "offline"},
            "num_trees": 3,
            "expansion_budget": 3,
            "max_depth": 2,
            "train_ratio": 0.5,
            "candidates_per_node": 2,
            "rollout_concurrency": 3,
            "model_concurrency": 2,
            "expansion_concurrency": 3,
            **updates,
        }
    )


def initial_skill():
    """Provide multiple managed files to exercise non-entrypoint preservation."""
    return Skill(
        source=SkillContent(
            content="# Skill\nOriginal instructions.\n",
            resources={"helper.txt": "keep"},
            name="test-skill",
            description="Task instructions",
            standard_format=False,
        )
    )


def samples(count=6):
    """Use path-like task IDs to verify that directory names do not use raw IDs."""
    return [
        DataSample(task=Task(id=f"../task/{index}", instruction=f"Public task {index}"), reward_data=PrivateAnswer())
        for index in range(count)
    ]


async def run_training(tmp_path, settings=None, env=None, optimizer=None, *, monkeypatch):
    """Run a complete trainer with deterministic local doubles."""
    settings = settings or config()
    env = env or FakeEnvironment()
    optimizer = optimizer or FakeOptimizer()
    skill = initial_skill()
    before = skill.snapshot()
    result = await SkillForestTrainer(agent_config(monkeypatch, optimizer, settings)).train(
        skill=skill,
        env=env,
        train_samples=samples(),
        validation_samples=[DataSample(task=Task(id="held-out", instruction="Excluded"), reward_data=PrivateAnswer())],
        work_dir=tmp_path,
    )
    assert skill.snapshot() == before
    assert result.resources == skill.resources
    return result, env, optimizer


def test_registry_defaults_and_reject_unpublished_options():
    trainer = Algorithm.build("skill_forest", {"optimizer": {"model": "offline"}})
    assert isinstance(trainer, SkillForestTrainer)
    assert trainer.config.ensemble_strategy == "select"
    assert trainer.config.optimizer.timeout_s == 180
    assert trainer.config.optimizer.max_retries == 3
    assert config(optimizer={"model": "x", "max_retries": 1}).optimizer.max_retries == 1
    for updates in ({"bootstrap": True}, {"expansion_budget": 1}, {"optimizer": {"model": "x", "max_retries": -1}}):
        with pytest.raises(ValidationError):
            config(**updates)


def test_stateless_split_and_atomic_patch():
    ids = [str(index) for index in range(10)]
    split = split_tasks(ids, train_ratio=0.7, seed=42)
    assert split == split_tasks(ids, train_ratio=0.7, seed=42)
    assert split != split_tasks(ids, train_ratio=0.7, seed=43)
    assert set(split.train_ids).isdisjoint(split.validation_ids)
    assert set(split.train_ids + split.validation_ids) == set(ids)
    assert ids == [str(index) for index in range(10)]
    skill = initial_skill()
    patch = Patch.model_validate(
        {"edits": [{"op": "append", "content": "new"}, {"op": "replace", "target": "missing", "content": "bad"}]}
    )
    with pytest.raises(ValueError):
        apply_patch(skill, patch)
    assert "new" not in skill.content


def test_exact_bound_preserves_full_validation_acceptance():
    for parent in itertools.product((0.0, 0.5, 1.0), repeat=3):
        for child in itertools.product((0.0, 0.5, 1.0), repeat=3):
            parent_scores = dict(enumerate(parent))
            for delta in (0, 0.1, 0.5):
                accepted = sum(child) / 3 - sum(parent) / 3 > delta
                for count in range(4):
                    if cannot_improve(dict(enumerate(child[:count])), parent_scores, delta):
                        assert not accepted


def test_global_priority_and_tree_winners():
    skill = initial_skill()
    nodes = [
        Node(id=str(index), tree_id=index % 2, parent_id="root", depth=1, skill=skill, gain=gain, scores={"x": score})
        for index, (gain, score) in enumerate([(0.2, 0.9), (0.5, 0.7), (0.4, 0.8)])
    ]
    assert [node.id for node in expansion_wave(nodes, limit=2, max_depth=2)] == ["1", "2"]
    assert [node.id for node in tree_winners(nodes)] == ["0", "1"]
    assert [node.id for node in nodes] == ["0", "1", "2"]
    assert novelty("same words", "same words") == pytest.approx(0)
    assert novelty("same words", "different vocabulary") == pytest.approx(1)


@pytest.mark.parametrize("shared_train,shared_validation", [(True, True), (False, True), (False, False)])
async def test_complete_select_global_caps_and_root_sharing(monkeypatch, tmp_path, shared_train, shared_validation):
    prepared_seeds = []

    def prepare_sample(self, sample, *, seed):
        prepared_seeds.append(seed)
        return sample

    monkeypatch.setattr(SkillForestEnvAdapter, "prepare_sample", prepare_sample)
    settings = config(root_share_train_trajectory=shared_train, root_share_validation=shared_validation)
    result, env, optimizer = await run_training(tmp_path, settings, monkeypatch=monkeypatch)
    assert "improved" in result.content
    assert 1 < env.peak <= settings.rollout_concurrency
    assert 1 < optimizer.peak <= settings.model_concurrency
    assert env.active == optimizer.active == 0
    assert len({path for _, _, path in env.calls}) == len(env.calls)
    assert not any(key == "held-out" for key, _, _ in env.calls)
    root_calls = [(key, path) for key, text, path in env.calls if "improved" not in text]
    splits = json.loads((tmp_path / "config.json").read_text())["splits"]
    union = set(key for split in splits for key in split["validation_ids"])
    train_union = set(key for split in splits for key in split["train_ids"])
    if shared_train:
        assert len(root_calls) == 6
        assert set(Counter(key for key, _ in root_calls).values()) == {1}
    elif shared_validation:
        assert len(root_calls) == len(union) + 3 * 3
    else:
        assert len(root_calls) == sum(len(split["validation_ids"]) for split in splits) + 3 * 3
    first_root_evaluations = len([path for _, path in root_calls if "node-tree-000-root/workspaces/eval" in path])
    if shared_train:
        assert first_root_evaluations == len(union | train_union)
        assert prepared_seeds[:first_root_evaluations] == [42] * first_root_evaluations
    elif shared_validation:
        assert first_root_evaluations == len(union)
        assert prepared_seeds[:first_root_evaluations] == [10042] * first_root_evaluations
    else:
        assert first_root_evaluations == len(splits[0]["validation_ids"])
    serialized = json.dumps(optimizer.messages)
    assert "PRIVATE-ANSWER" not in serialized and "PRIVATE-DETAIL" not in serialized
    result_data = json.loads((tmp_path / "result.json").read_text())
    forest = result_data["training"]["forest"]
    assert forest["expansions"] == 3
    assert len(forest["nodes"]) == 9
    assert result_data["status"] == "completed"
    assert len(list((tmp_path / "skills").glob("skill-tree-*.md"))) == settings.num_trees
    assert not list(tmp_path.rglob("audit/result.json"))
    assert optimizer.merges == 0


async def test_merge_does_not_gate_low_scoring_synthesized_skills(monkeypatch, tmp_path):
    result, env, optimizer = await run_training(
        tmp_path,
        config(ensemble_strategy="hierarchical_merge"),
        env=FakeEnvironment(merge_pass=False),
        monkeypatch=monkeypatch,
    )
    assert "synthesized" in result.content
    assert optimizer.merges == 2
    synthesized_calls = [call for call in env.calls if "synthesized" in call[1]]
    assert len(synthesized_calls) == 12
    assert json.loads((tmp_path / "result.json").read_text())["training"]["winners"]


@pytest.mark.parametrize("malformed", [False, True])
async def test_no_winner_is_explicit_failure_not_root_fallback(monkeypatch, tmp_path, malformed):
    with pytest.raises(RuntimeError, match="No positive-gain"):
        (
            await run_training(
                tmp_path,
                env=FakeEnvironment(candidate_pass=False),
                optimizer=FakeOptimizer(malformed=malformed),
                monkeypatch=monkeypatch,
            )
        )
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "result.json").exists()
    assert not list((tmp_path / "skills").glob("skill-tree-*.md"))


@pytest.mark.parametrize("source", ["env", "model"])
async def test_failure_is_audited_without_retry_or_zero_score(monkeypatch, tmp_path, source):
    env = FakeEnvironment(fail=source == "env")
    optimizer = FakeOptimizer(fail=source == "model")
    with pytest.raises(RuntimeError if source == "env" else ExceptionGroup):
        (await run_training(tmp_path, env=env, optimizer=optimizer, monkeypatch=monkeypatch))
    assert env.active == optimizer.active == 0
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "result.json").exists()
    assert not list(tmp_path.rglob("call.json"))


async def test_reject_nonempty_run_directory(monkeypatch, tmp_path):
    (tmp_path / "existing.txt").write_text("preserve")
    with pytest.raises(ValueError, match="legacy"):
        (await run_training(tmp_path, monkeypatch=monkeypatch))
    assert (tmp_path / "existing.txt").read_text() == "preserve"


async def test_real_environment_rollout_contract(monkeypatch, tmp_path):
    from mindmemos_skill.envs.base import Environment
    from pydantic import BaseModel

    class LocalConfig(BaseModel):
        """No external dependencies for this environment."""

    class LocalEnvironment(Environment):
        supports_resume = False

        async def _run(self, *, task, reward_data, skills, workspace):
            assert len(skills) == 1
            assert workspace.is_dir()
            passed = "improved" in skills[0].content
            return RolloutResult(
                output="local", trajectory=[], reward=Reward(score=float(passed), success=passed, detail="")
            )

    (
        await run_training(
            tmp_path,
            config(num_trees=1, expansion_budget=1),
            env=LocalEnvironment(LocalConfig()),
            monkeypatch=monkeypatch,
        )
    )
    from mindmemos_skill.common.json import read_jsonl

    result_data = json.loads((tmp_path / "result.json").read_text())
    streams = [row for path in tmp_path.rglob("*.jsonl") for row in read_jsonl(path)]
    assert streams
    assert all(row["result"]["started_at"] for row in streams)
    assert result_data["training"]["summary"]["decisions"] == {"accepted": 2}
    assert not list(tmp_path.rglob("audit/result.json"))


async def test_multiple_depths_respect_global_expansion_budget(monkeypatch, tmp_path):
    class ProgressiveEnvironment(FakeEnvironment):
        async def rollout(self, **kwargs):
            result = await super().rollout(**kwargs)
            text = kwargs["skill_set"].skills["candidate"].content
            depth = text.count("improved")
            task_index = int(kwargs["task"].id.rsplit("/", 1)[1])
            passed = depth >= 2 or depth == 1 and task_index % 2 == 0
            return result.model_copy(update={"reward": Reward(score=float(passed), success=passed, detail="")})

    (
        await run_training(
            tmp_path, config(expansion_budget=4, max_depth=2), env=ProgressiveEnvironment(), monkeypatch=monkeypatch
        )
    )
    forest = json.loads((tmp_path / "result.json").read_text())["training"]["forest"]
    assert forest["expansions"] == 4
    assert max(node["depth"] for node in forest["nodes"]) == 2
    assert all(node["gain"] > 0 for node in forest["nodes"] if node["parent_id"] is not None)
    root_children = [node for node in forest["nodes"] if node["depth"] == 1]
    expanded = {node["parent_id"] for node in forest["nodes"] if node["depth"] == 2}
    assert len(expanded) == 1
    assert next(node["gain"] for node in root_children if node["id"] in expanded) == max(
        node["gain"] for node in root_children
    )


async def test_early_rejection_reduces_calls_without_changing_winner(monkeypatch, tmp_path):
    early, early_env, _ = await run_training(tmp_path / "early", config(expansion_budget=6), monkeypatch=monkeypatch)
    full, full_env, _ = await run_training(
        tmp_path / "full", config(expansion_budget=6, early_rejection=False), monkeypatch=monkeypatch
    )
    assert early.content == full.content
    assert len(early_env.calls) < len(full_env.calls)
    decisions = [json.loads(path.read_text()) for path in (tmp_path / "early").glob("trees/tree-*/node-*/node.json")]
    assert any(decision["status"] == "rejected" and len(decision["scores"]) < 3 for decision in decisions)


@pytest.mark.parametrize("concurrency", [1, 2, 8])
@pytest.mark.parametrize("passed,early_rejection", [(False, True), (False, False), (True, True)])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
async def test_candidate_validation_queue(tmp_path, monkeypatch, concurrency, passed, early_rejection, batch_size):
    from mindmemos_skill.algos.train.skill_forest import trainer as trainer_module
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime
    from mindmemos_skill.algos.train.skill_forest.typing import TreeSplit

    async def generate_patch(**kwargs):
        return Patch.model_validate({"edits": [{"op": "append", "content": "improved"}]})

    monkeypatch.setattr(trainer_module, "generate_patch", generate_patch)
    settings = config(
        rollout_concurrency=concurrency, early_rejection=early_rejection, validation_batch_size=batch_size
    )
    expected_peak = min(7, concurrency, batch_size)

    class ValidationEnvironment:
        def __init__(self):
            self.lock = asyncio.Lock()
            self.ready = asyncio.Event()
            self.active = 0
            self.peak = 0
            self.calls = []

        async def rollout(self, *, task, work_dir, **kwargs):
            async with self.lock:
                self.calls.append((task.id, str(work_dir)))
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == expected_peak:
                    self.ready.set()
            try:
                assert await asyncio.wait_for(self.ready.wait(), timeout=5), "Validation tasks did not run concurrently"
                return RolloutResult(
                    output="public",
                    trajectory=[],
                    reward=Reward(score=float(passed), success=passed, detail=""),
                )
            finally:
                async with self.lock:
                    self.active -= 1

    env = ValidationEnvironment()
    pool = {sample.task.id: sample for sample in samples(8)}
    ids = list(pool)
    split = TreeSplit(train_ids=tuple(ids[:1]), validation_ids=tuple(ids[1:]))
    parent = Node(id="root", tree_id=0, skill=initial_skill(), scores={key: 0.5 for key in ids[1:]})

    async def run():
        runtime = ForestRuntime(settings)
        try:
            return await SkillForestTrainer(settings)._candidate(
                parent, 0, split, pool, {ids[0]: None}, env, runtime, tmp_path
            )
        finally:
            await runtime.close()

    candidate = await run()
    expected_calls = len(env.calls)
    # Bounds are checked only after all calls in the reference microbatch finish.
    if early_rejection and not passed:
        assert expected_calls == min(7, ((4 + batch_size - 1) // batch_size) * batch_size)
    else:
        assert expected_calls == 7
    assert env.peak == expected_peak
    assert len({directory for _, directory in env.calls}) == expected_calls
    assert {key for key, _ in env.calls} == set(ids[1 : expected_calls + 1])
    assert (candidate is not None) == passed
    decision = json.loads((tmp_path / "trees/tree-0/node-root-00/node.json").read_text())
    assert len(decision["scores"]) == expected_calls
    assert (len(decision["scores"]) < 7) == (expected_calls < 7)


def test_obsolete_validation_microbatch_size_is_rejected():
    with pytest.raises(ValidationError):
        config(validation_microbatch_size=0)


@pytest.mark.parametrize("tiebreak,expected", [("novelty", 1), ("validation", 0)])
async def test_select_breaks_score_ties_by_configured_protocol(monkeypatch, tmp_path, tiebreak, expected):
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime

    initial = initial_skill()
    texts = [initial.content + "improved", "# Novel\nimproved unrelated vocabulary"]
    winners = [
        Node(
            id=f"winner-{index}",
            tree_id=index,
            parent_id="root",
            gain=1,
            skill=initial.with_content(text),
            scores={"x": 1 if index == 0 else 0.5},
        )
        for index, text in enumerate(texts)
    ]
    settings = config(selection_tiebreak=tiebreak)

    async def evaluate():
        runtime = ForestRuntime(agent_config(monkeypatch, FakeOptimizer(), settings))
        try:
            return await SkillForestTrainer(settings)._ensemble(
                initial,
                winners,
                {sample.task.id: sample for sample in samples()},
                FakeEnvironment(),
                runtime,
                tmp_path,
            )
        finally:
            await runtime.close()

    assert (await evaluate())[0].content == winners[expected].skill.content


async def test_merge_filters_only_initial_winners(monkeypatch, tmp_path):
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime

    initial = initial_skill()
    texts = ["improved first", "improved second", "weak"]
    winners = [
        Node(
            id=f"winner-{index}",
            tree_id=index,
            parent_id="root",
            gain=1,
            skill=initial.with_content(text),
            scores={"x": 1},
        )
        for index, text in enumerate(texts)
    ]
    settings = config(ensemble_strategy="hierarchical_merge")
    optimizer = FakeOptimizer()

    async def evaluate():
        runtime = ForestRuntime(agent_config(monkeypatch, optimizer, settings))
        try:
            return await SkillForestTrainer(settings)._ensemble(
                initial,
                winners,
                {sample.task.id: sample for sample in samples()},
                FakeEnvironment(merge_pass=False),
                runtime,
                tmp_path,
            )
        finally:
            await runtime.close()

    assert "synthesized" in (await evaluate())[0].content
    assert optimizer.merges == 1


async def test_cancellation_drains_started_rollouts(monkeypatch, tmp_path):
    env = FakeEnvironment()

    async def cancel():
        trainer = SkillForestTrainer(agent_config(monkeypatch, FakeOptimizer(), config()))
        task = asyncio.create_task(
            trainer.train(
                skill=initial_skill(), env=env, train_samples=samples(), validation_samples=[], work_dir=tmp_path
            )
        )
        while not env.active:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    (await cancel())
    assert env.active == 0
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_prompt_text_and_structured_requests():
    from hashlib import sha256
    from pathlib import Path

    from mindmemos_skill.algos.train.skill_forest.prompts import (
        final_patch_user,
        format_trajectory,
        load_prompt,
        patch_merge_user,
        reflection_user,
        skill_merge_user,
    )

    prompt_hashes = {
        "analyst_error": "6f08c7d85d928add077f710583d1b53272cf99c45ad290e2a8588e7a6572c19b",
        "analyst_success": "603940125a601d6a91f2e80d2bfc631ab017129b6f1f3508a8ae96e3c2025d5a",
        "merge_failure": "c2e4c64a95065022616cceb8b46c275b07b5e6dab6375fa551d4766ebbfa7767",
        "merge_success": "9f62da1aff25522d632a6936d71577a3d13bfa3fbd2c8ee35294ec87f2710d72",
        "merge_final": "fabbc8adbdbfc62577d9b09108c64782ee45589a0fd0a18a3dc4fee96b28d5b3",
        "hierarchical_merge_skills": "073838c5e2889300fba3ba306c933d28a28a7a90f6215d91c989723f3dcca1ce",
    }
    for name, expected_hash in prompt_hashes.items():
        assert sha256(load_prompt(name).encode()).hexdigest() == expected_hash
    golden = json.loads((Path(__file__).parent / "fixtures/skill_forest/prompt_golden.json").read_text())
    evidence = format_trajectory([{"type": "tool_call", "cmd": "inspect", "obs": "found"}])
    assert "inspect" in evidence and "found" in evidence
    for success in (False, True):
        outcome = "Successful" if success else "Failed"
        assert reflection_user("# Rules", evidence, count=1, success=success, edit_budget=4) == (
            "## Current Skill\n# Rules\n\n## Edits Budget\nProduce at most L=4 edits.\n\n"
            f"## {outcome} Trajectories (1 total)\n{evidence}"
        )
    patch = {"edits": [{"op": "append", "content": "Verify outputs."}]}
    assert json.loads(patch_merge_user("# Rules", [patch], level=2)) == {
        "skill": "# Rules",
        "patches": [patch],
        "merge_level": 2,
    }
    assert json.loads(final_patch_user("# Rules", [("failure", patch)])) == {
        "skill": "# Rules",
        "groups": [{"outcome": "failure", "patch": patch}],
    }
    assert skill_merge_user("# Left\nRules.\n", "# Right\nNew.\n", 0.75, 0.5) == golden["skill_merge_user"]
    assert skill_merge_user("# Left\nRules.\n", "# Left\nRules.\n", 0.75, 0.5) == golden["identical_skill_merge_user"]


async def test_minibatch_reflection_preserves_evidence_and_outcome_groups(monkeypatch, tmp_path):
    from mindmemos_skill.algos.train.skill_forest.generation import generate_patch
    from mindmemos_skill.algos.train.skill_forest.prompts import load_prompt
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime

    optimizer = FakeOptimizer()
    settings = config(reflection_minibatch_size=2, merge_batch_size=2)
    tasks = samples()
    results = [
        RolloutResult(
            output="done",
            trajectory=[
                {
                    "role": "assistant",
                    "content": "execute",
                    "tool_calls": [{"function": {"name": "search", "arguments": "needle"}}],
                },
                {"role": "system", "content": "PRIVATE-DETAIL-DO-NOT-SEND"},
            ],
            reward=Reward(score=0, success=index < 3, detail="PRIVATE-DETAIL-DO-NOT-SEND"),
        )
        for index in range(6)
    ]

    async def generate():
        runtime = ForestRuntime(agent_config(monkeypatch, optimizer, settings))
        try:
            return await generate_patch(
                skill=initial_skill(), samples=tasks, results=results, runtime=runtime, directory=tmp_path, seed=42
            )
        finally:
            await runtime.close()

    patch = await generate()
    assert patch.edits
    systems = Counter(message[0]["content"] for message in optimizer.messages)
    assert systems[load_prompt("analyst_error")] == 2
    assert systems[load_prompt("analyst_success")] == 2
    assert systems[load_prompt("merge_failure")] == 1
    assert systems[load_prompt("merge_success")] == 1
    assert systems[load_prompt("merge_final")] == 1
    assert "needle" in json.dumps(optimizer.messages)
    assert "PRIVATE-DETAIL" not in json.dumps(optimizer.messages)
    reflected = [
        messages[1]["content"]
        for messages in optimizer.messages
        if messages[1]["content"].startswith("## Current Skill")
    ]
    assert sum(request.count("### Trajectory ") for request in reflected) == 6
    assert any("Trajectories (1 total)" in request for request in reflected)
    merges = [
        json.loads(messages[1]["content"])
        for messages in optimizer.messages
        if messages[1]["content"].startswith('{"skill"') and '"patches"' in messages[1]["content"]
    ]
    for request in merges:
        assert sorted(patch["edits"][0]["support_count"] for patch in request["patches"]) == [3, 3]


async def test_reflection_rejects_unpaired_results_before_model_calls(tmp_path):
    from mindmemos_skill.algos.train.skill_forest.generation import generate_patch

    with pytest.raises(ValueError, match="one result for every sample"):
        (
            await generate_patch(
                skill=initial_skill(), samples=samples(), results=[], runtime=None, directory=tmp_path, seed=42
            )
        )


async def test_config_and_node_records_use_the_compact_layout(monkeypatch, tmp_path):
    """Run identity, Skill snapshots and trajectories use only their owners."""
    settings = config()
    settings.optimizer.api_key = "forest-test-key"
    (await run_training(tmp_path, settings=settings, monkeypatch=monkeypatch))
    configuration = json.loads((tmp_path / "config.json").read_text())
    assert configuration["schema_version"] == ALGORITHM_VERSION
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["schema_version"] == ALGORITHM_VERSION
    assert result["training"]["algorithm_report"]["schema_version"] == ALGORITHM_VERSION
    assert configuration["trainer"]["optimizer"]["api_key"] == "[REDACTED]"
    assert settings.optimizer.api_key == "forest-test-key"
    nodes = list(tmp_path.glob("trees/tree-*/node-*/node.json"))
    assert nodes
    assert all(json.loads(path.read_text())["skill"] for path in nodes)
    assert not list(tmp_path.rglob("call.json"))
    assert not list(tmp_path.rglob("completed.json"))
