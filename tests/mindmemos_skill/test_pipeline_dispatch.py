"""Facade dispatch, explicit composition, and scoped persistence after refactoring."""

import pytest
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.api.config import ServiceConfig
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.errors import SkillNotFoundError
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
        tasks=[{"task": {"id": "t1", "instruction": "Solve it"}, "reward_data": {}}],
        split={"train": ["t1"]},
    )


def test_training_rejects_obsolete_skill_reference():
    values = sample_request().model_dump()
    with pytest.raises(ValidationError, match="skill_ref"):
        TrainingRequest.model_validate(values | {"skill_ref": "alias"})
    assert "skill_ref" not in TrainingRequest.model_json_schema()["properties"]


def test_service_delegates_to_selected_training_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(Pipeline, "_registry", {})

    @Pipeline.register("recording_train")
    class RecordingTraining(TrainingPipeline[Options]):
        config_type = Options

        def train(self, request):
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
            "training_pipeline": "second",
            "backends": {
                "first": {"module": "pipelines", "class_name": "recording_train"},
                "second": {"module": "pipelines", "class_name": "recording_train"},
            },
        }
    )
    service = create_service(config)
    request = sample_request().model_copy(update={"project_id": "project"})
    result = service.auto_train_skill(request)
    assert calls[0][0] is service.backends["second"]
    assert calls[0][1] is not request
    assert calls[0][1].base_version_id == "v1"
    assert result.version_id == "v2"
    # The same selected pipeline serves HTTP, with a structured request body.
    from fastapi.testclient import TestClient

    with TestClient(create_app(service)) as client:
        response = client.post("/skills/auto-train", json=request.model_dump())
    assert response.status_code == 200
    assert response.json()["version_id"] == "v2"
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["ambiguous", "wrong_family"])
def test_invalid_pipeline_selection_fails_before_construction(monkeypatch, failure):
    calls = []
    monkeypatch.setattr(Pipeline, "_registry", {})
    monkeypatch.setattr(Algorithm, "_registry", {})

    @Pipeline.register("train")
    class Training(TrainingPipeline[Options]):
        config_type = Options

        def __init__(self, config):
            calls.append("constructed")
            super().__init__(config)

        def train(self, request):
            raise NotImplementedError

    @Algorithm.register("algorithm")
    class Other(Algorithm[Options]):
        config_type = Options

    values = {
        "backends": {
            "first": {"module": "pipelines", "class_name": "train"},
            "second": {"module": "pipelines", "class_name": "train"},
            "other": {"module": "algos", "class_name": "algorithm"},
        }
    }
    if failure == "wrong_family":
        values["training_pipeline"] = "other"
    with pytest.raises(ValueError):
        create_service(ServiceConfig.model_validate(values))
    assert calls == []


def test_repository_uses_shared_store_and_scoped_version_lookup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    with_service = create_service(home=home)
    try:
        service = with_service
        assert service.repository is service.backends["repository_ops"]
        assert service.repository.persistence is service.backends["repository"]
        source = Skill(bundle=SkillBundle(files={"SKILL.md": "Persisted content"})).snapshot()
        item = service.register_skill(RegisterRequest(name="Example", alias="example", source=source))
        store = service.backends["repository"].skills
        assert store.resolve(service.local_project_id, "example").skill_id == item.skill.skill_id
        assert store.version_by_id(service.local_project_id, item.version.version_id) == item.version
        with pytest.raises(SkillNotFoundError):
            store.version_by_id("another-project", item.version.version_id)
        request = ListRequest()
        calls = []
        original = service.repository.list_skills

        def record(request):
            calls.append(request)
            return original(request)

        monkeypatch.setattr(service.repository, "list_skills", record)
        assert service.list_skills(request)[0].skill_id == item.skill.skill_id
        assert request.project_id is None
        assert calls[0].project_id == service.local_project_id
    finally:
        with_service.close()
    # Open the same populated database again without re-registering anything.
    reopened = create_service(home=home)
    try:
        assert reopened.show_skill(ShowRequest(skill_ref="example")).version == item.version
    finally:
        reopened.close()


def test_factory_closes_constructed_database_after_dependency_failure(tmp_path, monkeypatch):
    from mindmemos_skill.infra.database import SQLiteDatabase

    monkeypatch.setattr(Pipeline, "_registry", {})
    closed = []
    original = SQLiteDatabase.close

    def close(database):
        closed.append(database)
        original(database)

    monkeypatch.setattr(SQLiteDatabase, "close", close)

    @Pipeline.register("failing")
    class Failing(Pipeline[Options]):
        config_type = Options

        def __init__(self, config, *, database):
            raise RuntimeError("Construction failed")

    config = ServiceConfig.model_validate(
        {
            "backends": {
                "db": {"module": "infra", "class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'db'}"}},
                "pipeline": {"module": "pipelines", "class_name": "failing", "dependencies": {"database": "db"}},
            }
        }
    )
    with pytest.raises(RuntimeError, match="Construction failed"):
        create_service(config)
    assert len(closed) == 1
