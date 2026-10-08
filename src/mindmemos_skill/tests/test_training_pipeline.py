"""Offline service training, automatic recovery, and durable publication checks."""

import asyncio

import pytest
import pytest_asyncio
from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillTrainer
from mindmemos_skill.api.config import ServiceConfig
from mindmemos_skill.common.json import read_json
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.errors import SkillConflictError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.db_services.skills import SkillDataService
from mindmemos_skill.pipelines.typing import CommitRequest, RegisterRequest, ShowRequest, TrainingRequest
from mindmemos_skill.skill import Skill, SkillBundle, SkillResources
from pydantic import BaseModel
from test_skill_forest import FakeOptimizer
from test_skill_forest import config as forest_config
from test_treeskill_training import make_trainer, smoke

from mindmemos_skill import create_service


class OfflineConfig(BaseModel):
    """No external dependencies."""


class OfflineEnvironment(Environment[OfflineConfig]):
    config_type = OfflineConfig
    supports_resume = False

    def __init__(self, config):
        super().__init__(config)
        self.calls = []

    def _run(self, *, task, reward_data, skills, workspace):
        self.calls.append(task.id)
        text = skills[0].bundle.files["SKILL.md"]
        success = "improved" in text or "Verify" in text
        return RolloutResult(
            output="Saved",
            trajectory=[{"role": "assistant", "content": "Saved output"}],
            reward=Reward(score=float(success), success=success, detail="offline"),
        )


@pytest_asyncio.fixture
async def setup_service(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment, "_registry", dict(Environment._registry))
    Environment.register("training_offline")(OfflineEnvironment)
    model = make_trainer().config.model_dump(mode="json")
    config = ServiceConfig.model_validate(
        {
            "persistence": {
                "config": {"include_remote": False},
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'existing.db'}"}},
            },
            "repository_pipeline": {"class_name": "repository"},
            "training_pipelines": {
                name: {
                    "class_name": "training",
                    "config": {
                        "trainer": {"class_name": name, "config": options},
                        "environment": {"class_name": "training_offline"},
                    },
                }
                for name, options in (("treeskill", model), ("skill_forest", forest_config().model_dump(mode="json")))
            },
        }
    )
    service = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    source = Skill(
        bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"}),
        resources=SkillResources(files={"helper.txt": "keep"}),
    )
    with service.request_principal("test-user"):
        initial = await service.register_skill(
            RegisterRequest(project_id="project", name="Original", source=source.snapshot())
        )
    for name in config.training_pipelines:
        pipeline = service.backends[f"training_pipelines.{name}"]
        pipeline.trainer.agent_factory = smoke.SmokeAgent if name == "treeskill" else FakeOptimizer()
    with service.request_principal("test-user"):
        yield service, initial, source, config
    await service.close()


def request_for(tmp_path, initial, algorithm):
    ids = [f"train-{index}" for index in range(6)]
    return TrainingRequest(
        pipeline=algorithm,
        project_id="project",
        base_version_id=initial.version.version_id,
        work_dir=str(tmp_path / algorithm),
        tasks=[
            DataSample(task=Task(id=identifier, instruction="Save and verify"), reward_data=RewardData())
            for identifier in [*ids, "val", "test"]
        ],
        split={"train": ids, "val": ["val"], "test": ["test"]},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["treeskill", "skill_forest"])
async def test_service_selects_trainer_and_publishes_once(setup_service, tmp_path, algorithm):
    service, initial, source, config = setup_service
    request = request_for(tmp_path, initial, algorithm)
    result = await service.auto_train_skill(request)
    pipeline = service.backends[f"training_pipelines.{algorithm}"]
    assert isinstance(pipeline.trainer, TreeSkillTrainer if algorithm == "treeskill" else SkillForestTrainer)
    version = (await service.show_skill(ShowRequest(project_id="project", skill_ref=result.skill_id))).version
    assert version.parent_version_ids == [initial.version.version_id]
    assert version.version_label.endswith(algorithm.replace("_", "-"))
    if algorithm == "skill_forest":
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
    assert restored.bundle.files["SKILL.md"] == "# Workflow\nSave the output.\n"
    assert version.resources == initial.version.resources
    assert pipeline.environment.calls[-2:] == ["val", "test"]
    assert not {"val", "test"} & set(pipeline.environment.calls[:-2])
    count = len(pipeline.environment.calls)
    assert await service.auto_train_skill(request) == result
    assert len(pipeline.environment.calls) == count
    # A new service over an existing populated database recognizes the same publication.
    reopened = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    try:
        with reopened.request_principal("test-user"):
            assert await reopened.auto_train_skill(request) == result
    finally:
        await reopened.close()
    with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
        assert len(store.skills.list_versions(result.skill_id)) == 2
    assert "must-not-be-recorded" not in "\n".join(path.read_text() for path in (tmp_path / algorithm).rglob("*.json"))


@pytest.mark.asyncio
async def test_commit_failure_preserves_candidate_and_reuses_work(setup_service, tmp_path, monkeypatch):
    service, initial, _, _ = setup_service
    request = request_for(tmp_path, initial, "treeskill")
    pipeline = service.backends["training_pipelines.treeskill"]
    original = SkillDataService.commit_training

    def unavailable(*args, **kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(SkillDataService, "commit_training", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        await service.auto_train_skill(request)
    assert (tmp_path / "treeskill/selected_skill.json").exists()
    count = len(pipeline.environment.calls)
    monkeypatch.setattr(SkillDataService, "commit_training", original)
    result = await service.auto_train_skill(request)
    assert result.version_id != initial.version.version_id
    assert len(pipeline.environment.calls) == count


@pytest.mark.asyncio
async def test_commit_acknowledgement_loss_and_concurrent_parent(setup_service, tmp_path, monkeypatch):
    service, initial, source, _ = setup_service
    request = request_for(tmp_path, initial, "treeskill")
    original = SkillDataService.commit_training

    def lose_ack(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(SkillDataService, "commit_training", lose_ack)
    with pytest.raises(OSError, match="acknowledgement"):
        await service.auto_train_skill(request)
    monkeypatch.setattr(SkillDataService, "commit_training", original)
    published = await service.auto_train_skill(request)
    changed = await service.commit_skill(
        CommitRequest(
            project_id="project",
            skill_ref=initial.skill.skill_id,
            base_version_id=published.version_id,
            source=source.snapshot(),
        )
    )
    assert await service.auto_train_skill(request) == published
    assert (
        await service.show_skill(ShowRequest(project_id="project", skill_ref=initial.skill.skill_id))
    ).version.version_id == changed.version.version_id
    with pytest.raises(SkillConflictError):
        await service.auto_train_skill(request.model_copy(update={"work_dir": str(tmp_path / "other")}))
    with pytest.raises(SkillConflictError, match="different inputs"):
        await service.auto_train_skill(request.model_copy(update={"base_version_id": changed.version.version_id}))


@pytest.mark.asyncio
@pytest.mark.parametrize("point", ["collection", "model", "final"])
async def test_treeskill_automatically_recovers_without_repeating_calls(tmp_path, monkeypatch, point):
    from mindmemos_skill.algos.train.treeskill import trainer as module

    trainer = make_trainer()
    env = OfflineEnvironment(OfflineConfig())
    source = Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"}))
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

    monkeypatch.setattr(module, "write_json", interrupt)

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
    assert (await asyncio.to_thread((await run()).assemble, Task(id="use", instruction="Save it"))).bundle.files[
        "SKILL.md"
    ]
    assert read_json(tmp_path / "summary.json")["status"] == "completed"
    with pytest.raises(ValueError, match="mismatch"):
        await trainer.train(
            skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Changed\nDifferent"})),
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
                skill_ref=initial.skill.skill_id,
                base_version_id=initial.version.version_id,
                source=source.snapshot(),
            )
        )
        return selected

    monkeypatch.setattr(pipeline.trainer, "train", concurrent_edit)
    with pytest.raises(SkillConflictError):
        await service.auto_train_skill(request)
    assert (tmp_path / "treeskill/selected_skill.json").exists()
    with pipeline.persistence.session(Scope(fields={"project_id": "project"})) as store:
        assert len(store.skills.list_versions(initial.skill.skill_id)) == 2


def test_documented_service_config_is_valid():
    from pathlib import Path

    from mindmemos_skill.api.factory import load_service_config

    path = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/training/service.yaml"
    config = load_service_config(path.read_text())
    assert set(config.training_pipelines) == {"treeskill", "skill_forest"}
    assert {item.class_name for item in config.training_pipelines.values()} == {"training"}


@pytest.mark.asyncio
async def test_cancelled_worker_finishes_before_lock_owner_exits():
    from threading import Event

    from mindmemos_skill.common.asyncio import run_sync

    started, finish = Event(), Event()
    finished = []

    def worker():
        started.set()
        assert finish.wait(3)
        finished.append(True)

    task = asyncio.create_task(run_sync(worker))
    assert await asyncio.to_thread(started.wait, 3)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True]
