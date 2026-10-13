"""Offline service training, automatic recovery, and durable publication checks."""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from mindmemos_skill.agents import Agent as RegisteredAgent
from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillTrainer
from mindmemos_skill.api.typing import CommitRequest, RegisterRequest, ShowRequest, StructuredSkillInfo, TrainingRequest
from mindmemos_skill.common.json import read_json
from mindmemos_skill.config.config import ServiceConfig
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.errors import SkillConflictError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.db_services.skills import SkillDataService
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import BaseModel, ValidationError

from mindmemos_skill import create_service

from .test_skill_forest import FakeOptimizer
from .test_skill_forest import config as forest_config
from .test_treeskill_training import make_trainer, smoke


class OfflineConfig(BaseModel):
    """No external dependencies."""


class OfflineEnvironment(Environment[OfflineConfig]):
    config_type = OfflineConfig
    supports_resume = False
    instances = []

    def __init__(self, config):
        super().__init__(config)
        self.calls = []
        self.instances.append(self)

    async def _run(self, *, task, reward_data, skills, workspace):
        self.calls.append(task.id)
        text = skills[0].content
        success = "improved" in text or "Verify" in text
        return RolloutResult(
            output="Saved",
            trajectory=[{"role": "assistant", "content": "Saved output"}],
            reward=Reward(score=float(success), success=success, detail="offline"),
        )


@pytest_asyncio.fixture
async def setup_service(tmp_path, monkeypatch):
    monkeypatch.setattr(OfflineEnvironment, "instances", [])
    monkeypatch.setattr(Environment, "_registry", dict(Environment._registry))
    Environment.register("training_offline")(OfflineEnvironment)
    model = make_trainer(monkeypatch=monkeypatch).config.model_dump(mode="json")
    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'existing.db'}"}},
            },
            "repository_pipeline": {"class_name": "repository"},
            "training_pipelines": {
                name: {
                    "class_name": "training",
                    "config": {"trainer": {"class_name": name, "config": options}},
                }
                for name, options in (("treeskill", model), ("skill_forest", forest_config().model_dump(mode="json")))
            },
        }
    )
    service = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    source = Skill(
        source=SkillContent(
            content="# Workflow\nSave the output.\n",
            resources={"helper.txt": "keep"},
            name="test-skill",
            description="Task instructions",
            standard_format=False,
        )
    )
    with service.request_principal("test-user"):
        initial = await service.register_skill(
            RegisterRequest(
                project_id="project",
                skill=StructuredSkillInfo(
                    **source.snapshot().model_dump(
                        include={"content", "resources", "name", "description", "standard_format"}
                    )
                ),
            )
        )
    from mindmemos_skill.algos.train.skill_forest.config import ForestOptimizerConfig

    forest_optimizer = FakeOptimizer()
    monkeypatch.setattr(
        RegisteredAgent,
        "from_config",
        lambda options: (
            forest_optimizer(options) if isinstance(options, ForestOptimizerConfig) else smoke.SmokeAgent(options)
        ),
    )
    with service.request_principal("test-user"):
        yield service, initial, source, config
    await service.close()


def request_for(tmp_path, initial, algorithm):
    ids = [f"train-{index}" for index in range(6)]
    return TrainingRequest(
        algo=algorithm,
        env={"class_name": "training_offline"},
        project_id="project",
        init_skill_version_id=initial.skill_version_id,
        work_dir=str(tmp_path / algorithm),
        tasks=[
            DataSample(task=Task(id=identifier, instruction="Save and verify"), reward_data=RewardData())
            for identifier in [*ids, "val", "test"]
        ],
        split={"train": ids, "val": ["val"], "test": ["test"]},
    )


@pytest.mark.asyncio
async def test_pipeline_uses_call_environment(setup_service, tmp_path):
    """Pass the selected environment to training and held-out evaluation."""
    service, initial, _, _ = setup_service
    pipeline = service.backends["training_pipelines.treeskill"]
    selected_environment = OfflineEnvironment(OfflineConfig())
    request = request_for(tmp_path, initial, "treeskill")
    assert "environment" not in type(pipeline.config).model_fields
    with pytest.raises(AttributeError):
        _ = pipeline.environment
    with pytest.raises(TypeError, match="env"):
        await pipeline.train(request)
    with pipeline.persistence.session(Scope(fields={"project_id": "project"})):
        await pipeline.train(request, env=selected_environment)
    assert selected_environment.calls[-2:] == ["val", "test"]
    assert OfflineEnvironment.instances == [selected_environment]


def test_training_config_rejects_initial_environment(
    monkeypatch,
):
    """The configured training pipeline never owns an environment."""
    from mindmemos_skill.pipelines.builtin_pipelines.training.config import TrainingPipelineConfig

    with pytest.raises(ValidationError, match="environment"):
        TrainingPipelineConfig.model_validate(
            {
                "trainer": {
                    "class_name": "treeskill",
                    "config": make_trainer(monkeypatch=monkeypatch).config.model_dump(mode="json"),
                },
                "environment": {"class_name": "training_offline"},
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["treeskill", "skill_forest"])
async def test_service_selects_trainer_and_publishes_once(setup_service, tmp_path, algorithm):
    service, initial, source, config = setup_service
    request = request_for(tmp_path, initial, algorithm)
    result = await service.run_training(request)
    pipeline = service.backends[f"training_pipelines.{algorithm}"]
    assert isinstance(pipeline.trainer, TreeSkillTrainer if algorithm == "treeskill" else SkillForestTrainer)
    version = (await service.show_skill(ShowRequest(project_id="project", skill_id=result.skill_id))).version
    assert version.parent_version_ids == [initial.skill_version_id]
    assert version.version_label.endswith(algorithm.replace("_", "-"))
    evaluation = version.metadata["algorithm_report"]["evaluations"]["test"]
    assert evaluation["count"] == 1 and evaluation["errors"] == 0
    assert evaluation["wall_seconds"] >= 0
    assert evaluation["total_tokens"] is None
    assert Path(evaluation["trajectory_file"]).is_file()
    if algorithm == "skill_forest":
        local_result = read_json(tmp_path / algorithm / "train" / "result.json")
        assert local_result["validation"]["count"] == 1
        assert local_result["test"]["count"] == 1
        assert local_result["test"]["tasks"][0]["task_id"] == "test"
        report = version.metadata["algorithm_report"]
        assert report["run_id"] == result.run_id
        assert report["algorithm"] == "skill_forest"
        assert report["node_count"] == sum(tree["node_count"] for tree in report["trees"])
        assert report["trees"]
        assert all(
            {"gain", "train_top_select_score", "validation_mean_score"} <= tree["winner"].keys()
            for tree in report["trees"]
            if tree["winner"] is not None
        )
    restored = Skill.from_snapshot(source.snapshot())
    assert restored.content == "# Workflow\nSave the output.\n"
    assert version.resources == source.snapshot().resources
    environment = OfflineEnvironment.instances[-1]
    assert environment.calls[-2:] == ["val", "test"]
    assert not {"val", "test"} & set(environment.calls[:-2])
    test_index = read_json(tmp_path / algorithm / "test" / "results.json")
    assert [task["task_id"] for task in test_index["tasks"]] == ["test"]
    assert test_index["tasks"][0]["result_path"] == "task-000000/completed.json"
    assert test_index["tasks"][0]["score"] in {0.0, 1.0}
    count = len(environment.calls)
    assert await service.run_training(request) == result
    assert len(OfflineEnvironment.instances[-1].calls) == 0
    assert len(environment.calls) == count
    # A new service over an existing populated database recognizes the same publication.
    reopened = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    try:
        with reopened.request_principal("test-user"):
            assert await reopened.run_training(request) == result
    finally:
        await reopened.close()
    with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
        assert len((await store.skills.list_versions(result.skill_id))) == 2
    assert "must-not-be-recorded" not in "\n".join(path.read_text() for path in (tmp_path / algorithm).rglob("*.json"))


@pytest.mark.asyncio
async def test_commit_failure_preserves_candidate_and_reuses_work(setup_service, tmp_path, monkeypatch):
    service, initial, _, _ = setup_service
    request = request_for(tmp_path, initial, "treeskill")
    pipeline = service.backends["training_pipelines.treeskill"]
    original = SkillDataService.commit_algorithm

    def unavailable(*args, **kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(SkillDataService, "commit_algorithm", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        await service.run_training(request)
    assert (tmp_path / "treeskill/selected_skill.json").exists()
    count = len(OfflineEnvironment.instances[-1].calls)
    monkeypatch.setattr(SkillDataService, "commit_algorithm", original)
    result = await service.run_training(request)
    assert result.version_id != initial.skill_version_id
    assert len(OfflineEnvironment.instances[-1].calls) == 0
    assert len(OfflineEnvironment.instances[-2].calls) == count


@pytest.mark.asyncio
async def test_training_can_publish_without_activating(setup_service, tmp_path):
    service, initial, _, _ = setup_service
    request = request_for(tmp_path, initial, "treeskill").model_copy(update={"auto_accept": False})
    result = await service.run_training(request)
    detail = await service.show_skill(
        ShowRequest(project_id="project", skill_id=initial.skill_id, version_id=result.version_id)
    )
    assert detail.version.parent_version_ids == [initial.skill_version_id]
    assert detail.skill.active_version_id == initial.skill_version_id
    assert await service.run_training(request) == result
    with pytest.raises(SkillConflictError, match="different inputs"):
        await service.run_training(request.model_copy(update={"auto_accept": True}))


@pytest.mark.asyncio
async def test_commit_acknowledgement_loss_and_concurrent_parent(setup_service, tmp_path, monkeypatch):
    service, initial, source, _ = setup_service
    request = request_for(tmp_path, initial, "treeskill")
    original = SkillDataService.commit_algorithm

    async def lose_ack(self, *args, **kwargs):
        await original(self, *args, **kwargs)
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(SkillDataService, "commit_algorithm", lose_ack)
    with pytest.raises(OSError, match="acknowledgement"):
        await service.run_training(request)
    monkeypatch.setattr(SkillDataService, "commit_algorithm", original)
    published = await service.run_training(request)
    changed = await service.commit_skill(
        CommitRequest(
            project_id="project",
            skill_id=initial.skill_id,
            base_version_id=published.version_id,
            skill=StructuredSkillInfo(
                **source.snapshot().model_dump(
                    include={"content", "resources", "name", "description", "standard_format"}
                )
            ),
        )
    )
    assert await service.run_training(request) == published
    assert (
        await service.show_skill(ShowRequest(project_id="project", skill_id=initial.skill_id))
    ).version.version_id == changed.version.version_id
    branch = await service.run_training(request.model_copy(update={"work_dir": str(tmp_path / "other")}))
    historical_child = await service.show_skill(
        ShowRequest(project_id="project", skill_id=initial.skill_id, version_id=branch.version_id)
    )
    assert historical_child.version.parent_version_ids == [initial.skill_version_id]
    assert historical_child.skill.active_version_id == branch.version_id
    with pytest.raises(SkillConflictError, match="different inputs"):
        await service.run_training(request.model_copy(update={"init_skill_version_id": changed.version.version_id}))


@pytest.mark.asyncio
@pytest.mark.parametrize("point", ["collection", "model", "final"])
async def test_treeskill_automatically_recovers_without_repeating_calls(tmp_path, monkeypatch, point):
    from mindmemos_skill.algos.train.treeskill import trainer as module

    trainer = make_trainer(monkeypatch=monkeypatch)
    env = OfflineEnvironment(OfflineConfig())
    source = Skill(
        source=SkillContent(
            content="# Workflow\nSave the output.\n",
            name="test-skill",
            description="Task instructions",
            standard_format=False,
        )
    )
    samples = [DataSample(task=Task(id="train", instruction="Save it"), reward_data=RewardData())]
    original = module.write_json
    stopped = False

    def interrupt(path, value):
        nonlocal stopped
        original(path, value)
        target = {"collection": "evidence.json", "model": "analysis.json", "final": "final_skill.json"}[point]
        if path.name == target and not stopped:
            stopped = True
            raise RuntimeError("interrupted")

    from mindmemos_skill.algos.train.treeskill import artifacts

    monkeypatch.setattr(module, "write_json", interrupt)
    monkeypatch.setattr(artifacts, "write_json", interrupt)

    async def run():
        return await trainer.train(
            skill=source, env=env, train_samples=samples, validation_samples=[], work_dir=tmp_path
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        await run()
    calls = {path: path.read_bytes() for path in tmp_path.glob("model_calls/**/attempt-*.json")}
    result = await run()
    assert isinstance(result, TreeSkill)
    assert len(env.calls) == 1
    assert all(path.read_bytes() == value for path, value in calls.items())
    assert isinstance(await run(), TreeSkill)
    assert (await (await run()).assemble(Task(id="use", instruction="Save it"))).content
    assert read_json(tmp_path / "summary.json")["status"] == "completed"
    with pytest.raises(ValueError, match="mismatch"):
        await trainer.train(
            skill=Skill(
                source=SkillContent(
                    content="# Changed\nDifferent",
                    name="test-skill",
                    description="Task instructions",
                    standard_format=False,
                )
            ),
            env=env,
            train_samples=samples,
            validation_samples=[],
            work_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_parent_changed_during_training_preserves_result(setup_service, tmp_path, monkeypatch):
    service, initial, source, _ = setup_service
    request = request_for(tmp_path, initial, "treeskill")
    pipeline = service.backends["training_pipelines.treeskill"]
    original = pipeline.trainer.train

    async def concurrent_edit(**kwargs):
        selected = await original(**kwargs)
        await service.commit_skill(
            CommitRequest(
                project_id="project",
                skill_id=initial.skill_id,
                base_version_id=initial.skill_version_id,
                skill=StructuredSkillInfo(
                    **source.snapshot().model_dump(
                        include={"content", "resources", "name", "description", "standard_format"}
                    )
                ),
            )
        )
        return selected

    monkeypatch.setattr(pipeline.trainer, "train", concurrent_edit)
    with pytest.raises(SkillConflictError):
        await service.run_training(request)
    assert (tmp_path / "treeskill/selected_skill.json").exists()
    with pipeline.persistence.session(Scope(fields={"project_id": "project"})) as store:
        assert len((await store.skills.list_versions(initial.skill_id))) == 2


def test_service_example_config_is_valid():
    from pathlib import Path

    from mindmemos_skill.config.loader import read_skill_config

    path = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/dev.example.yaml"
    config = read_skill_config(path).service
    assert config.repository_pipeline.class_name == "repository"
    assert set(config.training_pipelines) == {"treeskill", "skill_forest", "treeskill_mindmemos_eval"}
    assert "training_environments" not in config.model_fields
    assert set(config.evolution_pipelines) == {"simple"}


@pytest.mark.asyncio
async def test_cancelled_async_workers_finish_cleanup_before_return():
    from mindmemos_skill.common.asyncio import map_bounded

    started = asyncio.Event()
    finished = []

    async def worker(item):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.append(item)

    task = asyncio.create_task(map_bounded(worker, [1], concurrency=1))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [1]
