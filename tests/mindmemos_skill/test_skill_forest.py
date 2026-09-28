"""Offline algorithm and concurrency verification without provider calls."""

import asyncio
import itertools
import json
import threading
import time
from collections import Counter

import pytest
from mindmemos_skill.agents import AgentResult
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer, SkillForestTrainerConfig
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
from mindmemos_skill.skill import Skill, SkillBundle, SkillResources
from pydantic import ValidationError


class PrivateAnswer(RewardData):
    """Private input that must never reach the optimizer."""

    answer: str = "PRIVATE-ANSWER-DO-NOT-SEND"
    """Sentinel for detecting training prompt leakage."""


class FakeEnvironment:
    """Observe actual worker concurrency and deterministic Skill-dependent rewards."""

    def __init__(self, *, fail=False, candidate_pass=True, merge_pass=True):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = []
        self.fail = fail
        self.candidate_pass = candidate_pass
        self.merge_pass = merge_pass

    def rollout(self, *, task, reward_data, skill_set, work_dir, resume):
        assert resume is False
        assert isinstance(reward_data, PrivateAnswer)
        text = skill_set.skills["candidate"].bundle.files["SKILL.md"]
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((task.id, text, str(work_dir)))
        try:
            time.sleep(0.002)
            if self.fail:
                raise RuntimeError("environment unavailable")
            passed = self.merge_pass if "synthesized" in text else self.candidate_pass and "improved" in text
            return RolloutResult(
                output="public output",
                trajectory=[{"role": "assistant", "content": "public reasoning"}],
                reward=Reward(score=float(passed), success=passed, detail="PRIVATE-DETAIL-DO-NOT-SEND"),
            )
        finally:
            with self.lock:
                self.active -= 1


class FakeOptimizer:
    """Observe model concurrency, exact prompts and pairwise synthesis calls."""

    def __init__(self, *, malformed=False, fail=False):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.messages = []
        self.merges = 0
        self.malformed = malformed
        self.fail = fail

    def __call__(self, config):
        assert config.max_retries == config.sdk_max_retries == 0
        assert config.parse_max_attempts == 1
        return self

    def run(self, *, messages):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.messages.append(messages)
        try:
            time.sleep(0.002)
            if self.fail:
                raise RuntimeError("provider unavailable")
            if self.malformed:
                return AgentResult(output="malformed")
            if messages[1]["content"].startswith("## Skill Candidate 1"):
                with self.lock:
                    self.merges += 1
                return AgentResult(output="# Skill\nsynthesized\n")
            patch = {"edits": [{"op": "append", "content": "improved"}]}
            if "trajectory_count" in json.loads(messages[1]["content"]):
                return AgentResult(output=json.dumps({"batch_size": 3, "patch": patch}))
            return AgentResult(output=json.dumps(patch))
        finally:
            with self.lock:
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
        bundle=SkillBundle(files={"SKILL.md": "# Skill\nOriginal instructions.\n"}),
        resources=SkillResources(files={"helper.txt": "keep"}),
    )


def samples(count=6):
    """Use path-like task IDs to verify that directory names do not use raw IDs."""
    return [
        DataSample(task=Task(id=f"../task/{index}", instruction=f"Public task {index}"), reward_data=PrivateAnswer())
        for index in range(count)
    ]


def run_training(tmp_path, settings=None, env=None, optimizer=None):
    """Run a complete trainer with deterministic local doubles."""
    settings = settings or config()
    env = env or FakeEnvironment()
    optimizer = optimizer or FakeOptimizer()
    skill = initial_skill()
    before = skill.snapshot()
    result = asyncio.run(
        SkillForestTrainer(settings, agent_factory=optimizer).train(
            skill=skill,
            env=env,
            train_samples=samples(),
            validation_samples=[
                DataSample(task=Task(id="held-out", instruction="Excluded"), reward_data=PrivateAnswer())
            ],
            work_dir=tmp_path,
        )
    )
    assert skill.snapshot() == before
    assert result.resources == skill.resources
    return result, env, optimizer


def test_registry_defaults_and_reject_unpublished_options():
    trainer = Algorithm.build("skill_forest", {"optimizer": {"model": "offline"}})
    assert isinstance(trainer, SkillForestTrainer)
    assert trainer.config.ensemble_strategy == "select"
    assert trainer.config.optimizer.timeout_s == 21600
    for updates in ({"bootstrap": True}, {"expansion_budget": 1}, {"optimizer": {"model": "x", "max_retries": 1}}):
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
    assert "new" not in skill.bundle.files["SKILL.md"]


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


@pytest.mark.parametrize("shared_train", [False, True])
def test_complete_select_global_caps_and_root_sharing(tmp_path, shared_train):
    settings = config(root_share_train_trajectory=shared_train)
    result, env, optimizer = run_training(tmp_path, settings)
    assert "improved" in result.bundle.files["SKILL.md"]
    assert 1 < env.peak <= settings.rollout_concurrency
    assert 1 < optimizer.peak <= settings.model_concurrency
    assert env.active == optimizer.active == 0
    assert len({path for _, _, path in env.calls}) == len(env.calls)
    assert not any(key == "held-out" for key, _, _ in env.calls)
    root_calls = [(key, path) for key, text, path in env.calls if "improved" not in text]
    splits = json.loads((tmp_path / "splits.json").read_text())
    union = set(key for split in splits for key in split["validation_ids"])
    if shared_train:
        assert len(root_calls) == 6
        assert set(Counter(key for key, _ in root_calls).values()) == {1}
    else:
        assert len(root_calls) == len(union) + 3 * 3
    assert len([path for _, path in root_calls if "root_validation" in path]) == len(union)
    serialized = json.dumps(optimizer.messages)
    assert "PRIVATE-ANSWER" not in serialized and "PRIVATE-DETAIL" not in serialized
    forest = json.loads((tmp_path / "forest.json").read_text())
    assert forest["expansions"] == 3
    assert len(forest["nodes"]) == 9
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"
    assert optimizer.merges == 0


def test_merge_does_not_gate_low_scoring_synthesized_skills(tmp_path):
    result, env, optimizer = run_training(
        tmp_path, config(ensemble_strategy="hierarchical_merge"), env=FakeEnvironment(merge_pass=False)
    )
    assert "synthesized" in result.bundle.files["SKILL.md"]
    assert optimizer.merges == 2
    synthesized_calls = [call for call in env.calls if "synthesized" in call[1]]
    assert len(synthesized_calls) == 12
    assert len(list((tmp_path / "ensemble").glob("merge-*/scores.json"))) == 2


@pytest.mark.parametrize("malformed", [False, True])
def test_no_winner_is_explicit_failure_not_root_fallback(tmp_path, malformed):
    with pytest.raises(RuntimeError, match="No positive-gain"):
        run_training(tmp_path, env=FakeEnvironment(candidate_pass=False), optimizer=FakeOptimizer(malformed=malformed))
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "selected_skill.json").exists()


@pytest.mark.parametrize("source", ["env", "model"])
def test_failure_is_audited_without_retry_or_zero_score(tmp_path, source):
    env = FakeEnvironment(fail=source == "env")
    optimizer = FakeOptimizer(fail=source == "model")
    with pytest.raises(RuntimeError if source == "env" else ExceptionGroup):
        run_training(tmp_path, env=env, optimizer=optimizer)
    assert env.active == optimizer.active == 0
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    calls = [json.loads(path.read_text()) for path in tmp_path.rglob("call.json")]
    assert any(call["status"] == "failed" for call in calls)
    assert not any(call["status"] == "running" for call in calls)


def test_reject_nonempty_run_directory(tmp_path):
    (tmp_path / "existing.txt").write_text("preserve")
    with pytest.raises(ValueError, match="legacy"):
        run_training(tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "preserve"


def test_real_environment_rollout_contract(tmp_path):
    from mindmemos_skill.envs.base import Environment
    from pydantic import BaseModel

    class LocalConfig(BaseModel):
        """No external dependencies for this environment."""

    class LocalEnvironment(Environment):
        supports_resume = False

        def _run(self, *, task, reward_data, skills, workspace):
            assert len(skills) == 1
            assert workspace.is_dir()
            passed = "improved" in skills[0].bundle.files["SKILL.md"]
            return RolloutResult(
                output="local", trajectory=[], reward=Reward(score=float(passed), success=passed, detail="")
            )

    run_training(tmp_path, config(num_trees=1, expansion_budget=1), env=LocalEnvironment(LocalConfig()))
    results = list(tmp_path.rglob("audit/result.json"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(results) == summary["rollout"]["calls"]
    assert all(json.loads(path.read_text())["started_at"] for path in results)
    assert summary["decisions"] == {"accepted": 2}


def test_multiple_depths_respect_global_expansion_budget(tmp_path):
    class ProgressiveEnvironment(FakeEnvironment):
        def rollout(self, **kwargs):
            result = super().rollout(**kwargs)
            text = kwargs["skill_set"].skills["candidate"].bundle.files["SKILL.md"]
            depth = text.count("improved")
            task_index = int(kwargs["task"].id.rsplit("/", 1)[1])
            passed = depth >= 2 or depth == 1 and task_index % 2 == 0
            return result.model_copy(update={"reward": Reward(score=float(passed), success=passed, detail="")})

    run_training(tmp_path, config(expansion_budget=4, max_depth=2), env=ProgressiveEnvironment())
    forest = json.loads((tmp_path / "forest.json").read_text())
    assert forest["expansions"] == 4
    assert max(node["depth"] for node in forest["nodes"]) == 2
    assert all(node["gain"] > 0 for node in forest["nodes"] if node["parent_id"] is not None)
    root_children = [node for node in forest["nodes"] if node["depth"] == 1]
    expanded = {node["parent_id"] for node in forest["nodes"] if node["depth"] == 2}
    assert len(expanded) == 1
    assert next(node["gain"] for node in root_children if node["id"] in expanded) == max(
        node["gain"] for node in root_children
    )


def test_early_rejection_reduces_calls_without_changing_winner(tmp_path):
    early, early_env, _ = run_training(tmp_path / "early", config(expansion_budget=6))
    full, full_env, _ = run_training(tmp_path / "full", config(expansion_budget=6, early_rejection=False))
    assert early.bundle == full.bundle
    assert len(early_env.calls) < len(full_env.calls)
    decisions = [json.loads(path.read_text()) for path in (tmp_path / "early").glob("nodes/*/decision.json")]
    assert any(decision.get("early_rejected") for decision in decisions)


@pytest.mark.parametrize("concurrency", [1, 2, 8])
@pytest.mark.parametrize("passed,early_rejection", [(False, True), (False, False), (True, True)])
def test_candidate_validation_queue(tmp_path, monkeypatch, concurrency, passed, early_rejection):
    from mindmemos_skill.algos.train.skill_forest import trainer as trainer_module
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime
    from mindmemos_skill.algos.train.skill_forest.typing import TreeSplit

    async def generate_patch(**kwargs):
        return Patch.model_validate({"edits": [{"op": "append", "content": "improved"}]})

    monkeypatch.setattr(trainer_module, "generate_patch", generate_patch)
    settings = config(rollout_concurrency=concurrency, early_rejection=early_rejection)
    expected_peak = min(7, concurrency)

    class ValidationEnvironment:
        def __init__(self):
            self.lock = threading.Lock()
            self.ready = threading.Event()
            self.active = 0
            self.peak = 0
            self.calls = []

        def rollout(self, *, task, work_dir, **kwargs):
            with self.lock:
                self.calls.append((task.id, str(work_dir)))
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == expected_peak:
                    self.ready.set()
            try:
                assert self.ready.wait(timeout=5), "Validation tasks did not run concurrently"
                return RolloutResult(
                    output="public",
                    trajectory=[],
                    reward=Reward(score=float(passed), success=passed, detail=""),
                )
            finally:
                with self.lock:
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

    candidate = asyncio.run(run())
    expected_calls = len(env.calls)
    if early_rejection and not passed:
        assert 4 <= expected_calls <= min(7, 4 + concurrency - 1)
    else:
        assert expected_calls == 7
    assert env.peak == expected_peak
    assert len({directory for _, directory in env.calls}) == expected_calls
    assert {key for key, _ in env.calls} == set(ids[1 : expected_calls + 1])
    assert (candidate is not None) == passed
    decision = json.loads((tmp_path / "nodes/root-00/decision.json").read_text())
    assert decision["evaluated"] == expected_calls
    assert decision["early_rejected"] == (expected_calls < 7)


def test_obsolete_validation_microbatch_size_is_rejected():
    with pytest.raises(ValidationError):
        config(validation_microbatch_size=0)


def test_select_breaks_score_ties_by_paper_novelty(tmp_path):
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime

    initial = initial_skill()
    texts = [initial.bundle.files["SKILL.md"] + "improved", "# Novel\nimproved unrelated vocabulary"]
    winners = [
        Node(
            id=f"winner-{index}",
            tree_id=index,
            parent_id="root",
            gain=1,
            skill=initial.with_bundle(SkillBundle(files={"SKILL.md": text})),
            scores={"x": 1},
        )
        for index, text in enumerate(texts)
    ]
    settings = config()

    async def evaluate():
        runtime = ForestRuntime(settings, agent_factory=FakeOptimizer())
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

    assert asyncio.run(evaluate()).bundle == winners[1].skill.bundle


def test_merge_filters_only_initial_winners(tmp_path):
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime

    initial = initial_skill()
    texts = ["improved first", "improved second", "weak"]
    winners = [
        Node(
            id=f"winner-{index}",
            tree_id=index,
            parent_id="root",
            gain=1,
            skill=initial.with_bundle(SkillBundle(files={"SKILL.md": text})),
            scores={"x": 1},
        )
        for index, text in enumerate(texts)
    ]
    settings = config(ensemble_strategy="hierarchical_merge")
    optimizer = FakeOptimizer()

    async def evaluate():
        runtime = ForestRuntime(settings, agent_factory=optimizer)
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

    assert "synthesized" in asyncio.run(evaluate()).bundle.files["SKILL.md"]
    assert optimizer.merges == 1
    pool = json.loads((tmp_path / "ensemble/merge_pool.json").read_text())
    assert set(pool["ids"]) == {"winner-0", "winner-1"}


def test_cancellation_drains_started_rollouts(tmp_path):
    env = FakeEnvironment()

    async def cancel():
        trainer = SkillForestTrainer(config(), agent_factory=FakeOptimizer())
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

    asyncio.run(cancel())
    assert env.active == 0
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    assert all(json.loads(path.read_text())["status"] != "running" for path in tmp_path.rglob("call.json"))


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
        "analyst_error": "999080aeb192068820de2e1d921a646feff00f154b48931a195fa124a26fc81d",
        "analyst_success": "8240f5b7a21785de819863aa6676ccd0b8723b16522350af6232bb181ea18cbf",
        "merge_failure": "c552073fee770dcd14ce4c2f9ea0c2648f5e058fb4ed26f9d8a3556a7db71d9d",
        "merge_success": "66849c0f229aea8074c4599d973b3ff99c826e200776c8389a062b8a0cf54d50",
        "merge_final": "5818a7c978d2a932c6fa9b6b5aa74c2d8bd2ec42c5f567c8ffc832a468d7c50a",
        "hierarchical_merge_skills": "8872756ea14c76ef5c3ee77ec2d00c66c2d6709228a5776849576e68aa4555e9",
    }
    for name, expected_hash in prompt_hashes.items():
        assert sha256(load_prompt(name).encode()).hexdigest() == expected_hash
    golden = json.loads((Path(__file__).parent / "fixtures/skill_forest/prompt_golden.json").read_text())
    evidence = format_trajectory([{"type": "tool_call", "cmd": "inspect", "obs": "found"}])
    assert "inspect" in evidence and "found" in evidence
    for success in (False, True):
        request = json.loads(reflection_user("# Rules", evidence, count=1, success=success, edit_budget=4))
        assert request == {
            "skill": "# Rules",
            "trajectories": evidence,
            "trajectory_count": 1,
            "edit_limit": 4,
            "outcome": "success" if success else "failure",
        }
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


def test_minibatch_reflection_preserves_evidence_and_outcome_groups(tmp_path):
    from mindmemos_skill.algos.train.skill_forest.generation import generate_patch
    from mindmemos_skill.algos.train.skill_forest.prompts import load_prompt
    from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime

    optimizer = FakeOptimizer()
    settings = config(reflection_minibatch_size=2, merge_batch_size=2, env_adapter={"class_name": "generic"})
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
        runtime = ForestRuntime(settings, agent_factory=optimizer)
        try:
            return await generate_patch(
                skill=initial_skill(), samples=tasks, results=results, runtime=runtime, directory=tmp_path, seed=42
            )
        finally:
            await runtime.close()

    patch = asyncio.run(generate())
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
        json.loads(messages[1]["content"])
        for messages in optimizer.messages
        if "trajectory_count" in json.loads(messages[1]["content"])
    ]
    assert sum(request["trajectory_count"] for request in reflected) == 6
    assert any(request["trajectory_count"] == 1 for request in reflected)
    assert sum(len(json.loads(request["trajectories"])) for request in reflected) == 6
    merges = [
        json.loads(messages[1]["content"])
        for messages in optimizer.messages
        if "patches" in json.loads(messages[1]["content"])
    ]
    for request in merges:
        assert sorted(patch["edits"][0]["support_count"] for patch in request["patches"]) == [1, 2]


def test_reflection_rejects_unpaired_results_before_model_calls(tmp_path):
    from mindmemos_skill.algos.train.skill_forest.generation import generate_patch

    with pytest.raises(ValueError, match="one result for every sample"):
        asyncio.run(
            generate_patch(
                skill=initial_skill(), samples=samples(), results=[], runtime=None, directory=tmp_path, seed=42
            )
        )


@pytest.mark.parametrize(
    "name,payload",
    [
        ("status.json", {"status": "completed", "duration_s": 1}),
        ("status.json", {"status": "failed", "duration_s": -1, "error": "failure"}),
        ("decision.json", {"status": "invalid_patch"}),
        (
            "decision.json",
            {"status": "accepted", "scores": {}, "gain": None, "evaluated": 0, "expected": 1, "early_rejected": True},
        ),
        ("call.json", {"status": "completed", "task_id": "a", "content_hash": "hash", "duration_s": 1}),
        ("parents.json", ["one"]),
        ("scores.json", {"scores": {}, "mean": float("nan")}),
        ("selection.json", {"selected": "a", "typo": True}),
        ("unknown.json", {}),
    ],
)
def test_artifact_validation_preserves_existing_file(tmp_path, name, payload):
    """Invalid artifacts must fail before replacing an existing destination."""
    from mindmemos_skill.algos.train.skill_forest.artifacts import record

    path = tmp_path / name
    path.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError):
        record(path, payload)
    assert path.read_text(encoding="utf-8") == "original"


def test_artifact_config_redacts_credentials_and_patch_preserves_omissions(tmp_path):
    """Central storage filters credentials and retains the original patch wire shape."""
    from mindmemos_skill.algos.train.skill_forest.artifacts import record

    settings = config().model_dump(mode="json")
    settings["optimizer"]["api_key"] = "test-secret"
    record(tmp_path / "config.json", settings)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "api_key" not in saved["optimizer"]
    patch = {"edits": [{"op": "append", "content": "new"}]}
    record(tmp_path / "patch.json", patch)
    assert json.loads((tmp_path / "patch.json").read_text()) == patch


def test_artifact_rollout_preserves_environment_extensions(tmp_path):
    """Audit validation keeps environment-specific result and reward data intact."""
    from mindmemos_skill.algos.train.skill_forest.artifacts import record

    payload = {
        "status": "completed",
        "task_id": "a",
        "content_hash": "hash",
        "duration_s": 1,
        "result": {
            "output": "ok",
            "trajectory": [],
            "reward": {"score": 1, "success": True, "detail": "", "custom_metric": 2},
            "environment_detail": {"files": ["result.txt"]},
        },
    }
    record(tmp_path / "call.json", payload)
    assert json.loads((tmp_path / "call.json").read_text()) == payload
