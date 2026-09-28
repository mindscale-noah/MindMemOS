"""Facade dispatch, explicit composition, and scoped persistence after refactoring."""

import pytest
from mindmemos_skill.api.config import ServiceConfig
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.db_services.skills import SkillDataService
from mindmemos_skill.pipelines import Pipeline, SkillPipelineResult, TrainingPipeline, TrainingRequest
from mindmemos_skill.pipelines.typing import ListRequest, RegisterRequest, ShowRequest
from mindmemos_skill.skill import Skill, SkillBundle
from pydantic import BaseModel, ValidationError

from mindmemos_skill import create_service


class Options(BaseModel):
    """No external resources are required by test implementations."""


def sample_request():
    """Build a training request requiring local project binding."""
    return TrainingRequest(
        base_version_id="v1",
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

        async def train(self, request):
            calls.append((self, request))
            return SkillPipelineResult(
                project_id=request.project_id,
                skill_id="skill",
                base_version_id=request.base_version_id,
                version_id="v2",
                run_id="run",
            )

    config = ServiceConfig.model_validate(
        {
            "local_project_id": "project",
            "training_pipelines": {
                "first": {"class_name": "recording_train"},
                "second": {"class_name": "recording_train"},
            },
        }
    )
    service = await create_service(config)
    request = sample_request().model_copy(update={"project_id": "project", "pipeline": "second"})
    result = await service.auto_train_skill(request)
    assert calls[0][0] is service.backends["training_pipelines.second"]
    assert calls[0][1] is not request
    assert calls[0][1].base_version_id == "v1"
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
    with_service = await create_service(home=home)
    try:
        service = with_service
        assert service.repository is service.backends["repository_pipeline"]
        assert service.repository.persistence is service.backends["persistence"]
        source = Skill(bundle=SkillBundle(files={"SKILL.md": "Persisted content"})).snapshot()
        item = await service.register_skill(RegisterRequest(name="Example", alias="example", source=source))
        with service.backends["persistence"].session(Scope(fields={"project_id": service.local_project_id})) as scoped:
            store = scoped.skills
            assert store.resolve(service.local_project_id, "example").skill_id == item.skill.skill_id
            assert store.version_by_id(service.local_project_id, item.version.version_id) == item.version
            with pytest.raises(SkillNotFoundError):
                store.version_by_id("another-project", item.version.version_id)
        request = ListRequest()
        calls = []
        original = service.repository.list_skills

        async def record(request):
            calls.append(request)
            return await original(request)

        monkeypatch.setattr(service.repository, "list_skills", record)
        assert (await service.list_skills(request))[0].skill_id == item.skill.skill_id
        assert request.project_id is None
        assert calls[0].project_id == service.local_project_id
    finally:
        await with_service.close()
    # Open the same populated database again without re-registering anything.
    reopened = await create_service(home=home)
    try:
        assert (await reopened.show_skill(ShowRequest(skill_ref="example"))).version == item.version
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_factory_closes_constructed_database_after_dependency_failure(tmp_path, monkeypatch):
    from mindmemos_skill.infra.database import SQLiteDatabase

    monkeypatch.setattr(Pipeline, "_registry", {})
    closed = []
    original = SQLiteDatabase.close

    def close(database):
        closed.append(database)
        original(database)

    monkeypatch.setattr(SQLiteDatabase, "close", close)

    @Pipeline.register("failing")
    class Failing(TrainingPipeline[Options]):
        config_type = Options

        def __init__(self, config, *, persistence):
            raise RuntimeError("Construction failed")

        async def train(self, request):
            raise NotImplementedError

    config = ServiceConfig.model_validate(
        {
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
    """Select configured pipelines and expose only evolution through HTTP."""
    from fastapi.testclient import TestClient
    from mindmemos_skill.pipelines import EvolutionPipeline, SkillPipelineRequest

    monkeypatch.setattr(Pipeline, "_registry", {})
    calls = []

    def record(instance, request):
        calls.append((instance, request))
        return SkillPipelineResult(
            project_id=request.project_id,
            skill_id="skill",
            base_version_id="v1",
            version_id="v2",
            run_id="run",
        )

    @Pipeline.register("training")
    class RecordingTraining(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request):
            return record(self, request)

    @Pipeline.register("evolution")
    class RecordingEvolution(EvolutionPipeline[Options]):
        config_type = Options

        async def evolve(self, request):
            return record(self, request)

    config = ServiceConfig.model_validate(
        {
            "local_project_id": "project",
            "training_pipelines": {name: {"class_name": "training"} for name in ("train_a", "train_b")},
            "evolution_pipelines": {name: {"class_name": "evolution"} for name in ("evolve_a", "evolve_b")},
        }
    )
    service = await create_service(config)
    if operation == "training":
        invoke = service.auto_train_skill
        request = sample_request().model_copy(update={"project_id": "project"})
        names, wrong = list(config.training_pipelines), "evolve_a"
    else:
        invoke = service.auto_evolve_skill
        request = SkillPipelineRequest(project_id="project", skill_ref="skill")
        names, wrong = list(config.evolution_pipelines), "train_a"

    for name in names:
        selected = request.model_copy(update={"pipeline": name})
        await invoke(selected)
        assert calls[-1][0] is service.backends[f"{operation}_pipelines.{name}"]
        assert calls[-1][1] is not selected
    assert request.pipeline is None
    count = len(calls)
    from mindmemos_skill.api.entries.config import FastAPIConfig

    if operation == "evolution":
        policy = FastAPIConfig(evolution_pipelines=set(names))
        with TestClient(create_app(service, api_config=policy)) as client:
            for name in (None, "missing", "unused", wrong):
                invalid = request.model_copy(update={"pipeline": name})
                with pytest.raises(ValueError):
                    await invoke(invalid)
                assert client.post("/skills/auto-evolve", json=invalid.model_dump()).status_code == 422
            assert len(calls) == count
            for name in names:
                response = client.post("/skills/auto-evolve", json=request.model_dump() | {"pipeline": name})
                assert response.status_code == 200
                assert calls[-1][0] is service.backends[f"{operation}_pipelines.{name}"]
    else:
        for name in (None, "missing", "unused", wrong):
            invalid = request.model_copy(update={"pipeline": name})
            with pytest.raises(ValueError):
                await invoke(invalid)
        assert len(calls) == count

    # A single definition is inferred; an empty mapping disables the operation.
    for available in ({names[0]: {"class_name": operation}}, {}):
        changed = ServiceConfig.model_validate(config.model_dump() | {f"{operation}_pipelines": available})
        service = await create_service(changed)
        invoke = service.auto_train_skill if operation == "training" else service.auto_evolve_skill
        if available:
            await invoke(request)
            assert calls[-1][0] is service.backends[f"{operation}_pipelines.{names[0]}"]
        else:
            with pytest.raises(NotImplementedError):
                await invoke(request)


@pytest.mark.parametrize("field", ["training_pipeline", "evolution_pipeline"])
def test_singular_pipeline_config_is_rejected(field):
    """Removed selectors must not silently act as defaults."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        ServiceConfig.model_validate({field: "first"})


@pytest.mark.asyncio
async def test_repository_write_leaves_event_loop_available(tmp_path, monkeypatch):
    """A synchronous storage write must allow concurrent async model work."""
    import asyncio
    from threading import Event

    entered = Event()
    release = Event()
    async with await create_service(home=tmp_path / "home") as service:
        original = SkillDataService.register

        def slow_register(self, **kwargs):
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("The event loop was blocked by storage")
            return original(self, **kwargs)

        monkeypatch.setattr(SkillDataService, "register", slow_register)

        async def model_work():
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.sleep(0)
            release.set()

        request = RegisterRequest(
            name="Async storage", source=Skill(bundle=SkillBundle(files={"SKILL.md": "Data"})).snapshot()
        )
        try:
            item, _ = await asyncio.gather(service.register_skill(request), model_work())
        finally:
            release.set()
        assert (await service.show_skill(ShowRequest(skill_ref=item.skill.skill_id))).version == item.version


@pytest.mark.asyncio
async def test_pipeline_cleanup_awaits_owned_resources_before_database(tmp_path, monkeypatch):
    """Asynchronous cleanup stays on the caller's loop and precedes database close."""
    import asyncio

    events = []
    loop = asyncio.get_running_loop()

    @Pipeline.register("async_cleanup_test")
    class OwnedResources(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request):
            raise NotImplementedError

        async def close(self):
            assert asyncio.get_running_loop() is loop
            await asyncio.sleep(0)
            events.append("pipeline")

    try:
        service = await create_service(
            ServiceConfig.model_validate(
                {
                    "persistence": {
                        "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}}
                    },
                    "training_pipelines": {"owned": {"class_name": "async_cleanup_test"}},
                }
            )
        )
        database = service.backends["database"]
        original = database.close

        def close_database():
            events.append("database")
            original()

        monkeypatch.setattr(database, "close", close_database)
        await service.close()
        assert events == ["pipeline", "database"]
    finally:
        Pipeline._registry.pop("async_cleanup_test", None)
