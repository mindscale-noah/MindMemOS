"""Exercise interruption recovery without repeating committed physical calls."""

import asyncio
import json
from collections import Counter

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer
from mindmemos_skill.algos.train.skill_forest import trainer as trainer_module
from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime
from test_skill_forest import FakeEnvironment, FakeOptimizer, config, initial_skill, samples


def execute(root, trainer, env):
    """Run the same bounded protocol before and after a simulated interruption."""
    return asyncio.run(
        trainer.train(skill=initial_skill(), env=env, train_samples=samples(), validation_samples=[], work_dir=root)
    )


@pytest.mark.parametrize("point", ["generated.json", "validated.json", "wave", "ensemble"])
def test_resume_committed_boundaries_and_budget(tmp_path, monkeypatch, point):
    """Interrupt after a durable boundary and reuse every completed physical call."""
    optimizer, env = FakeOptimizer(), FakeEnvironment()
    settings = config(expansion_budget=5)
    trainer = SkillForestTrainer(settings, agent_factory=optimizer)
    original = trainer_module.boundary
    failed = False

    def interrupt(path, identity, *, payload=None):
        nonlocal failed
        result = original(path, identity, payload=payload)
        hit = path.name == point or (point == "wave" and path.parent.name == "waves" and path.stem == "000001")
        if hit and payload is not None and not failed:
            failed = True
            raise RuntimeError("simulated process interruption")
        return result

    original_ensemble = trainer._ensemble

    async def stop_ensemble(*args):
        nonlocal failed
        if point == "ensemble" and not failed:
            failed = True
            raise RuntimeError("simulated Select interruption")
        return await original_ensemble(*args)

    monkeypatch.setattr(trainer_module, "boundary", interrupt)
    monkeypatch.setattr(trainer, "_ensemble", stop_ensemble)
    with pytest.raises((RuntimeError, ExceptionGroup)):
        execute(tmp_path, trainer, env)
    assert failed
    completed_before = {
        str(p.parent) for p in tmp_path.rglob("completed.json") if "result" in json.loads(p.read_text())["payload"]
    }
    calls_before = len(env.calls)
    restored = execute(tmp_path, trainer, env)
    assert "improved" in restored.bundle.files["SKILL.md"]
    assert not completed_before.intersection(path for _, _, path in env.calls[calls_before:])
    forest = json.loads((tmp_path / "forest.json").read_text())
    assert forest["expansions"] == 5
    assert len({node["id"] for node in forest["nodes"]}) == len(forest["nodes"])
    physical = len(env.calls), len(optimizer.messages)
    assert execute(tmp_path, trainer, env).bundle.content_hash == restored.bundle.content_hash
    assert (len(env.calls), len(optimizer.messages)) == physical


def test_partial_validation_reuses_stable_task_directories(tmp_path):
    """Restore a partial group without renumbering remaining tasks or rerunning completed ones."""
    env = FakeEnvironment()
    settings = config(rollout_concurrency=1)
    original = env.rollout
    failed = False

    def interrupted(**kwargs):
        nonlocal failed
        if kwargs["task"].id == "../task/2" and not failed:
            failed = True
            raise RuntimeError("interrupted third task")
        return original(**kwargs)

    env.rollout = interrupted

    async def collect(resume):
        runtime = ForestRuntime(settings, work_dir=tmp_path, resume=resume)
        try:
            return await runtime.scheduler.collect(
                env=env, samples=samples(), skill=initial_skill(), directory=tmp_path / "validation"
            )
        finally:
            await runtime.close()

    with pytest.raises(RuntimeError):
        asyncio.run(collect(False))
    result = asyncio.run(collect(True))
    assert len(result) == 6
    assert Counter(task for task, _, _ in env.calls) == Counter(sample.task.id for sample in samples())
    assert all(path.endswith(f"{int(task.rsplit('/', 1)[1]):06d}") for task, _, path in env.calls)


def test_restored_scores_can_reject_without_submitting_work(tmp_path):
    """Evaluate an early-rejection predicate against restored results before allocating calls."""
    env = FakeEnvironment()

    async def collect(resume, predicate):
        runtime = ForestRuntime(config(rollout_concurrency=1), work_dir=tmp_path, resume=resume)
        try:
            return await runtime.scheduler.collect(
                env=env,
                samples=samples(),
                skill=initial_skill(),
                directory=tmp_path / "validation",
                stop_when=predicate,
            )
        finally:
            await runtime.close()

    asyncio.run(collect(False, lambda results: len(results) >= 2))
    before = len(env.calls)
    result = asyncio.run(collect(True, lambda results: len(results) >= 2))
    assert len(result) == before == 2
    assert len(env.calls) == before


def test_resume_rejects_changed_inputs_and_corruption(tmp_path):
    """Reject incompatible protocols and modified committed outputs before any new calls."""
    env, optimizer = FakeEnvironment(), FakeOptimizer()
    trainer = SkillForestTrainer(config(), agent_factory=optimizer)
    execute(tmp_path, trainer, env)
    before = len(env.calls)
    changed = SkillForestTrainer(config(seed=99), agent_factory=optimizer)
    with pytest.raises(ValueError, match="mismatch"):
        execute(tmp_path, changed, env)
    path = tmp_path / "completed.json"
    value = json.loads(path.read_text())
    value["digest"] = "corrupt"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="mismatch"):
        execute(tmp_path, trainer, env)
    assert len(env.calls) == before


def test_legacy_output_is_not_silently_imported(tmp_path):
    """Old audit files alone must not be treated as validated recovery checkpoints."""
    (tmp_path / "forest.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        execute(tmp_path, SkillForestTrainer(config()), FakeEnvironment())


def test_invalid_candidates_remain_terminal_on_resume(tmp_path):
    """A rejected model format must not receive a fresh stochastic generation attempt."""
    env, optimizer = FakeEnvironment(), FakeOptimizer(malformed=True)
    trainer = SkillForestTrainer(config(), agent_factory=optimizer)
    with pytest.raises(RuntimeError, match="No positive-gain"):
        execute(tmp_path, trainer, env)
    before = len(env.calls), len(optimizer.messages)
    with pytest.raises(RuntimeError, match="No positive-gain"):
        execute(tmp_path, trainer, env)
    assert (len(env.calls), len(optimizer.messages)) == before


def test_select_failure_only_retries_missing_scores(tmp_path):
    """Keep the entire constructed forest and all completed Select requests on failure."""
    env, optimizer = FakeEnvironment(), FakeOptimizer()
    original = env.rollout
    failed = False

    def interrupted(**kwargs):
        nonlocal failed
        if "ensemble" in str(kwargs["work_dir"]) and not failed:
            failed = True
            raise RuntimeError("Select provider error")
        return original(**kwargs)

    env.rollout = interrupted
    trainer = SkillForestTrainer(config(), agent_factory=optimizer)
    with pytest.raises(ExceptionGroup):
        execute(tmp_path, trainer, env)
    before = len(env.calls), len(optimizer.messages)
    execute(tmp_path, trainer, env)
    assert len(optimizer.messages) == before[1]
    assert all("ensemble" in path for _, _, path in env.calls[before[0] :])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["rollout"]["statuses"]["failed"] == 1
    assert len(list(tmp_path.glob("ensemble/**/attempts/*/call.json"))) == 1
