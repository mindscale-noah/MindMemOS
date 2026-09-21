"""Service assembly with one store and independently configured pipelines."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.infra.database import SQLiteDatabase
from mindmemos_skill.pipelines import Pipeline, TrainingPipeline
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindmemos_skill import MindMemOSSkill, ServiceConfig, create_service


class AgentOptions(BaseModel):
    """Typed options owned by the example pipeline's nested agent."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=1, gt=0)


class Options(BaseModel):
    """Each pipeline carries all of its agent settings locally."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentOptions = Field(default_factory=AgentOptions)


@pytest.fixture
def registered_pipeline(monkeypatch):
    """Observe independent construction and reverse resource cleanup."""
    events = []
    monkeypatch.setattr(Pipeline, "_registry", {})

    @Pipeline.register("training")
    class Training(TrainingPipeline[Options]):
        config_type = Options

        def __init__(self, config, *, persistence):
            super().__init__(config, persistence=persistence)
            self.agent = config.agent.model_copy(deep=True)
            events.append(("created", self))

        async def train(self, request):
            raise NotImplementedError

        async def close(self):
            events.append(("closed", self))

    return events


def write_config(tmp_path, value):
    """Write a test YAML file and return its path."""
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def assembly_config(tmp_path):
    """Repeat local configuration instead of referencing a shared resource."""
    return {
        "persistence": {"database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}}},
        "training_pipelines": {
            name: {"class_name": "training", "config": {"agent": {"count": 7}}} for name in ("first", "second")
        },
    }


@pytest.mark.parametrize("source", ["yaml", "config"])
@pytest.mark.asyncio
async def test_assembly_shares_only_global_persistence(tmp_path, registered_pipeline, source):
    """Create independent pipeline components and exactly one database/store."""
    config = assembly_config(tmp_path)
    value = ServiceConfig.model_validate(config) if source == "config" else write_config(tmp_path, config)
    service = await create_service(value)
    assert isinstance(service, MindMemOSSkill)
    first, second = (event[1] for event in registered_pipeline)
    assert first.persistence is second.persistence is service.backends["persistence"]
    assert first.persistence.database is service.backends["database"]
    assert first.agent is not second.agent
    first.agent.count = 9
    assert second.agent.count == 7
    assert create_app(service).state.service is service
    view = service.backends
    view.clear()
    assert len(service.backends) == 4
    await service.close()
    assert registered_pipeline[-2:] == [("closed", second), ("closed", first)]


@pytest.mark.parametrize("failure", ["class", "options", "reference", "module", "dependencies", "legacy"])
@pytest.mark.asyncio
async def test_invalid_assembly_does_not_construct_backends(tmp_path, registered_pipeline, monkeypatch, failure):
    """Validate all pipeline definitions before opening the global database."""
    opened = []
    original = SQLiteDatabase.__init__

    def observe(self, config):
        opened.append(True)
        original(self, config)

    monkeypatch.setattr(SQLiteDatabase, "__init__", observe)
    config = assembly_config(tmp_path)
    spec = config["training_pipelines"]["second"]
    if failure == "class":
        spec["class_name"] = "missing"
    elif failure == "options":
        spec["config"]["agent"]["count"] = 0
    elif failure == "reference":
        spec["config"]["agent"] = "shared_agent"
    elif failure == "module":
        spec["module"] = "pipelines"
    elif failure == "dependencies":
        spec["dependencies"] = {"persistence": "store"}
    else:
        config["backends"] = {}
    with pytest.raises((ValueError, TypeError)):
        await create_service(write_config(tmp_path, config))
    assert not opened
    assert registered_pipeline == []


@pytest.mark.parametrize(
    "field", ["persistence_backend", "backends", "resources", "training_pipeline", "evolution_pipeline"]
)
def test_removed_global_selectors_are_rejected(field):
    """Never silently accept the old graph or default selection format."""
    with pytest.raises(ValidationError):
        ServiceConfig.model_validate({field: {}})


@pytest.mark.parametrize("storage", [[{}, {}], {"first": {}, "second": {}}])
def test_multiple_persistence_definitions_are_rejected(storage):
    with pytest.raises(ValidationError):
        ServiceConfig.model_validate({"persistence": storage})


@pytest.mark.parametrize(
    "content", ["", "[]", "null", "mode: local\nunknown: true", "persistence: [", "!!python/object:object {}"]
)
@pytest.mark.asyncio
async def test_invalid_yaml(tmp_path, content):
    """Reject malformed, unsafe, non-mapping, and unknown-field input."""
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises((ValidationError, yaml.YAMLError)):
        await create_service(path)


@pytest.mark.asyncio
async def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        await create_service(tmp_path / "missing.yaml")


@pytest.mark.asyncio
async def test_mutated_config_revalidated_before_construction(tmp_path, registered_pipeline):
    config = ServiceConfig.model_validate(assembly_config(tmp_path))
    config.training_pipelines["first"].config["agent"]["count"] = 0
    with pytest.raises(ValidationError):
        await create_service(config)
    assert registered_pipeline == []


@pytest.mark.asyncio
async def test_builtin_loading_in_fresh_process(tmp_path):
    """An explicit empty config creates no resources or implicit repository."""
    path = write_config(tmp_path, {})
    source = Path(__file__).resolve().parents[2] / "src/mindmemos_skill"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from mindmemos_skill import create_service; import sys, asyncio; assert asyncio.run(create_service(sys.argv[1])).backends == {}",
            str(path),
        ],
        cwd=source,
        check=True,
    )
    assert (await create_service(path)).backends == {}


@pytest.mark.asyncio
async def test_persistence_does_not_implicitly_enable_repository(tmp_path):
    service = await create_service(
        ServiceConfig.model_validate(
            {"persistence": {"database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}}}}
        )
    )
    try:
        with pytest.raises(NotImplementedError):
            await service.initialize_storage()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cleanup_reaches_database_when_pipeline_close_fails(tmp_path, registered_pipeline, monkeypatch):
    """A component cleanup error must not leak the shared database or other pipelines."""
    service = await create_service(ServiceConfig.model_validate(assembly_config(tmp_path)))
    first, second = (event[1] for event in registered_pipeline)
    closed = []
    database = service.backends["database"]
    original = database.close

    def close_database():
        closed.append("database")
        original()

    async def fail_close():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(database, "close", close_database)
    monkeypatch.setattr(second, "close", fail_close)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await service.close()
    assert ("closed", first) in registered_pipeline
    assert closed == ["database"]


@pytest.mark.asyncio
async def test_factory_closes_completed_pipelines_on_later_failure(tmp_path, registered_pipeline, monkeypatch):
    """Roll back successful pipeline construction before closing the shared database."""
    database_closed = []
    original = SQLiteDatabase.close

    def close_database(self):
        database_closed.append(True)
        original(self)

    monkeypatch.setattr(SQLiteDatabase, "close", close_database)

    @Pipeline.register("failing")
    class Failing(TrainingPipeline[Options]):
        config_type = Options

        def __init__(self, config, *, persistence):
            raise RuntimeError("construction failed")

        async def train(self, request):
            raise NotImplementedError

    config = assembly_config(tmp_path)
    config["training_pipelines"]["second"] = {"class_name": "failing"}
    with pytest.raises(RuntimeError, match="construction failed"):
        await create_service(ServiceConfig.model_validate(config))
    assert len(registered_pipeline) == 2
    assert registered_pipeline[0][0] == "created"
    assert registered_pipeline[1] == ("closed", registered_pipeline[0][1])
    assert database_closed == [True]
