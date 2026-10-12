"""Facade dispatch, explicit composition, and scoped persistence after refactoring."""

import pytest
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.typing import ListRequest, RegisterRequest, ShowRequest
from mindmemos_skill.config.config import ServiceConfig
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.db_services.skills import SkillDataService
from mindmemos_skill.pipelines import AlgoSkillEvolveResult, Pipeline, TrainingPipeline, TrainingRequest
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import BaseModel, ValidationError

from mindmemos_skill import (
    create_service,
    demos,  # noqa: F401
)

from .service_config import deployment_config, repository_config
from .skill_sources import register_detail, structured


class Options(BaseModel):
    """No external resources are required by test implementations."""


def sample_request():
    """Build a training request requiring local project binding."""
    return TrainingRequest(
        project_id="project",
        algo="first",
        env={"class_name": "demo_offline"},
        init_skill_version_id="v1",
        work_dir="output/training-test",
        tasks=[{"task": {"id": "t1", "instruction": "Solve it"}, "reward_data": {}}],
        split={"train": ["t1"]},
    )


def test_training_rejects_obsolete_skill_reference():
    values = sample_request().model_dump()
    with pytest.raises(ValidationError, match="skill_ref"):
        TrainingRequest.model_validate(values | {"skill_ref": "alias"})
    assert "skill_ref" not in TrainingRequest.model_json_schema()["properties"]


@pytest.mark.asyncio
async def test_service_delegates_to_selected_training_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(Pipeline, "_registry", {})

    @Pipeline.register("recording_train")
    class RecordingTraining(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request, *, env, adapter=None):
            calls.append((self, request, env))
            return AlgoSkillEvolveResult(
                skill_id="skill",
                base_version_id=request.init_skill_version_id,
                version_id="v2",
                run_id="run",
                content="output",
                resources={},
                content_hash="hash",
                skill_hash="hash",
            )

    config = deployment_config(
        {
            "home": ".",
            "training_pipelines": {
                "first": {"class_name": "recording_train"},
                "second": {"class_name": "recording_train"},
            },
        },
        "project",
    )
    service = await create_service(config)
    request = sample_request().model_copy(update={"project_id": "project", "algo": "second"})
    result = await service.run_training(request)
    assert calls[0][0] is service.backends["training_pipelines.second"]
    assert calls[0][1] is not request
    assert calls[0][1].init_skill_version_id == "v1"
    assert calls[0][2] is not None
    assert result.version_id == "v2"
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["training_pipelines", "evolution_pipelines"])
@pytest.mark.parametrize("failure", ["missing", "list", "wrong_family"])
@pytest.mark.asyncio
async def test_invalid_pipeline_selection_fails_before_construction(monkeypatch, failure, field):
    """Reject unknown implementations, obsolete lists and wrong capabilities."""
    calls = []
    monkeypatch.setattr(Pipeline, "_registry", {})

    @Pipeline.register("other")
    class Other(Pipeline[Options]):
        config_type = Options

        def __init__(self, config, *, persistence=None):
            calls.append("constructed")
            super().__init__(config, persistence=persistence)

    values = {
        field: {
            "missing": {"first": {"class_name": "absent"}},
            "list": ["first", "second"],
            "wrong_family": {"first": {"class_name": "other"}},
        }[failure]
    }
    with pytest.raises(ValueError):
        await create_service(ServiceConfig.model_validate(values))
    assert calls == []


@pytest.mark.asyncio
async def test_repository_uses_shared_store_and_scoped_version_lookup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    with_service = await create_service(repository_config(home, "project"))
    try:
        service = with_service
        assert service.repository is service.backends["repository_pipeline"]
        assert service.repository.persistence is service.backends["persistence"]
        source = Skill(
            SkillContent(
                content="Persisted content", name="test-skill", description="Test instructions", standard_format=False
            )
        ).snapshot()
        item = await register_detail(
            service,
            RegisterRequest(
                skill=structured(source, name="Example"), alias="example", project_id=service.default_project_id
            ),
        )
        with service.backends["persistence"].session(
            Scope(fields={"project_id": service.default_project_id})
        ) as scoped:
            store = scoped.skills
            assert (await store.resolve_family(service.default_project_id, "example")).skill_id == item.skill.skill_id
            assert (await store.version_by_id(service.default_project_id, item.version.version_id)) == item.version
            with pytest.raises(SkillNotFoundError):
                (await store.version_by_id("another-project", item.version.version_id))
        request = ListRequest(project_id=service.default_project_id)
        calls = []
        original = service.repository.list_skills

        async def record(request):
            calls.append(request)
            return await original(request)

        monkeypatch.setattr(service.repository, "list_skills", record)
        assert (await service.list_skills(request))[0].skill_id == item.skill.skill_id
        assert request.project_id == service.default_project_id
        assert calls[0].project_id == service.default_project_id
    finally:
        await with_service.close()
    # Open the same populated database again without re-registering anything.
    reopened = await create_service(repository_config(home, "project"))
    try:
        assert (
            await reopened.show_skill(ShowRequest(project_id=reopened.default_project_id, skill_id="example"))
        ).version == item.version
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_factory_closes_constructed_database_after_dependency_failure(tmp_path, monkeypatch):
    from mindmemos_skill.infra.database import SQLiteDatabase

    monkeypatch.setattr(Pipeline, "_registry", {})
    closed = []
    original = SQLiteDatabase.close

    async def close(database):
        closed.append(database)
        await original(database)

    monkeypatch.setattr(SQLiteDatabase, "close", close)

    @Pipeline.register("failing")
    class Failing(TrainingPipeline[Options]):
        config_type = Options

        def __init__(self, config, *, persistence):
            raise RuntimeError("Construction failed")

        async def train(self, request, *, env, adapter=None):
            raise NotImplementedError

    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {"database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}}},
            "training_pipelines": {"failing": {"class_name": "failing"}},
        }
    )
    with pytest.raises(RuntimeError, match="Construction failed"):
        await create_service(config)
    assert len(closed) == 1


@pytest.mark.parametrize("operation", ["training", "evolution"])
@pytest.mark.asyncio
async def test_multiple_pipeline_request_selection(monkeypatch, operation):
    """Select configured pipelines through explicit supported request fields."""
    from fastapi.testclient import TestClient
    from mindmemos_skill.pipelines import EvolutionPipeline, SkillPipelineRequest

    monkeypatch.setattr(Pipeline, "_registry", {})
    calls = []

    async def record(instance, request):
        calls.append((instance, request))
        return AlgoSkillEvolveResult(
            skill_id="skill",
            base_version_id="v1",
            version_id="v2",
            run_id="run",
            content="output",
            resources={},
            content_hash="hash",
            skill_hash="hash",
        )

    @Pipeline.register("training")
    class RecordingTraining(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request, *, env, adapter=None):
            return await record(self, request)

    @Pipeline.register("evolution")
    class RecordingEvolution(EvolutionPipeline[Options]):
        config_type = Options

        async def evolve(self, request):
            return await record(self, request)

    config = deployment_config(
        {
            "home": ".",
            "training_pipelines": {name: {"class_name": "training"} for name in ("train_a", "train_b")},
            "evolution_pipelines": {name: {"class_name": "evolution"} for name in ("evolve_a", "evolve_b")},
        },
        "project",
    )
    service = await create_service(config)
    if operation == "training":
        invoke = service.run_training
        request = sample_request().model_copy(update={"project_id": "project"})
        names, wrong = list(config.service.training_pipelines), "evolve_a"
    else:
        invoke = service.run_evolution
        request = SkillPipelineRequest(project_id="project", skill_ref="skill")
        names, wrong = list(config.service.evolution_pipelines), "train_a"

    selector = "algo" if operation == "training" else "pipeline"
    for name in names:
        selected = request.model_copy(update={selector: name})
        await invoke(selected)
        assert calls[-1][0] is service.backends[f"{operation}_pipelines.{name}"]
        assert calls[-1][1] is not selected
    if operation == "evolution":
        assert request.pipeline is None
    count = len(calls)
    if operation == "evolution":
        with TestClient(create_app(service)) as client:
            for name in (None, "missing", "unused", wrong):
                invalid = request.model_copy(update={selector: name})
                with pytest.raises(ValueError):
                    await invoke(invalid)
                assert client.post("/v1/skills/algos/evolve", json=invalid.model_dump()).status_code == 422
            assert len(calls) == count
            response = client.post("/v1/skills/algos/evolve", json={"skill_version_id": "v1"})
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "pipeline_unavailable"
            assert len(calls) == count
    else:
        for name in (None, "missing", "unused", wrong):
            invalid = request.model_copy(update={selector: name})
            with pytest.raises(ValueError):
                await invoke(invalid)
        assert len(calls) == count

    # A single definition is inferred; an empty mapping disables the operation.
    for available in ({names[0]: {"class_name": operation}}, {}):
        changed = config.model_copy(deep=True)
        changed.service = ServiceConfig.model_validate(
            config.service.model_dump() | {f"{operation}_pipelines": available}
        )
        service = await create_service(changed)
        invoke = service.run_training if operation == "training" else service.run_evolution
        if available:
            await invoke(request.model_copy(update={selector: names[0]}))
            assert calls[-1][0] is service.backends[f"{operation}_pipelines.{names[0]}"]
        else:
            with pytest.raises(NotImplementedError):
                await invoke(request.model_copy(update={selector: names[0]}))


@pytest.mark.parametrize("field", ["training_pipeline", "evolution_pipeline"])
def test_singular_pipeline_config_is_rejected(field):
    """Removed selectors must not silently act as defaults."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        ServiceConfig.model_validate({"home": ".", field: "first"})


@pytest.mark.asyncio
async def test_repository_write_leaves_event_loop_available(tmp_path, monkeypatch):
    """An awaited storage write allows concurrent model work."""
    import asyncio
    from asyncio import Event

    entered = Event()
    release = Event()
    async with await create_service(repository_config(tmp_path / "home", "project")) as service:
        original = SkillDataService.register

        async def slow_register(self, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=2)
            return await original(self, **kwargs)

        monkeypatch.setattr(SkillDataService, "register", slow_register)

        async def model_work():
            await asyncio.wait_for(entered.wait(), timeout=1)
            await asyncio.sleep(0)
            release.set()

        request = RegisterRequest(
            skill=structured(
                Skill(
                    SkillContent(
                        content="Data", name="test-skill", description="Test instructions", standard_format=False
                    )
                ).snapshot(),
                name="Async storage",
            ),
            project_id=service.default_project_id,
        )
        try:
            item, _ = await asyncio.gather(register_detail(service, request), model_work())
        finally:
            release.set()
        assert (
            await service.show_skill(ShowRequest(project_id=service.default_project_id, skill_id=item.skill.skill_id))
        ).version == item.version


@pytest.mark.asyncio
async def test_pipeline_cleanup_awaits_owned_resources_before_database(tmp_path, monkeypatch):
    """Asynchronous cleanup stays on the caller's loop and precedes database close."""
    import asyncio

    events = []
    loop = asyncio.get_running_loop()

    @Pipeline.register("async_cleanup_test")
    class OwnedResources(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request, *, env, adapter=None):
            raise NotImplementedError

        async def close(self):
            assert asyncio.get_running_loop() is loop
            await asyncio.sleep(0)
            events.append("pipeline")

    try:
        service = await create_service(
            ServiceConfig.model_validate(
                {
                    "home": tmp_path / "home",
                    "persistence": {
                        "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}}
                    },
                    "training_pipelines": {"owned": {"class_name": "async_cleanup_test"}},
                }
            )
        )
        database = service.backends["database"]
        original = database.close

        async def close_database():
            events.append("database")
            await original()

        monkeypatch.setattr(database, "close", close_database)
        await service.close()
        assert events == ["pipeline", "database"]
    finally:
        Pipeline._registry.pop("async_cleanup_test", None)
