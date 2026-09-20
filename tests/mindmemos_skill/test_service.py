"""YAML-driven service assembly without external resources."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.agents import Agent
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.infra import Infrastructure
from mindmemos_skill.logging import Logger
from mindmemos_skill.persistence import Persistence
from mindmemos_skill.skill import Skill
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindmemos_skill import MindMemOSSkill, ServiceConfig, create_service


class Options(BaseModel):
    """Backend configuration used to observe assembly."""

    model_config = ConfigDict(extra="forbid")
    """Reject unexpected implementation options."""

    count: int = Field(default=1, gt=0)
    """Positive number preserved on constructed test backends."""


@pytest.fixture
def registered_backends():
    """Register test implementations in isolated module registries."""
    events = []
    families = {
        "infra": Infrastructure,
        "algos": Algorithm,
        "agents": Agent,
        "logging": Logger,
        "persistence": Persistence,
        "envs": Environment,
    }
    original_registries = {family: family._registry for family in families.values()}
    for family in families.values():
        family._registry = {}

    @Infrastructure.register("Connection")
    class Connection(Infrastructure[Options]):
        config_type = Options

        def __init__(self, config):
            super().__init__(config)
            events.append("connection")

    @Persistence.register("Store")
    class Store(Persistence[Options]):
        config_type = Options

        def __init__(self, config, *, connection):
            super().__init__(config)
            self.connection = connection
            events.append("store")

    for family in families.values():
        if family is Environment:
            continue

        @family.register("Simple")
        class Simple(family):
            config_type = Options

    @Environment.register("Simple")
    class SimpleEnvironment(Environment[Options]):
        config_type = Options

        def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
            """Supply a concrete contract for service assembly tests."""
            return RolloutResult(
                output=task.instruction,
                trajectory=[],
                reward=Reward(score=0.0, success=False, detail="Answer did not pass"),
            )

    yield events
    for family, registry in original_registries.items():
        family._registry = registry


def write_config(tmp_path, value):
    """Write a test YAML file and return its path."""
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def assembly_config():
    """Place dependents first to verify dependency-driven ordering."""
    return {
        "backends": {
            "store": {"module": "persistence", "class_name": "Store", "dependencies": {"connection": "db"}},
            "second_store": {"module": "persistence", "class_name": "Store", "dependencies": {"connection": "db"}},
            "db": {"module": "infra", "class_name": "Connection", "config": {"count": 7}},
        },
    }


@pytest.mark.parametrize("source", ["yaml", "config"])
def test_assembly_reuses_dependencies(tmp_path, registered_backends, source):
    """Construct once per configured name and inject the same dependency object."""
    config = assembly_config()
    value = ServiceConfig.model_validate(config) if source == "config" else write_config(tmp_path, config)
    service = create_service(value)
    assert isinstance(service, MindMemOSSkill)
    assert registered_backends == ["connection", "store", "store"]
    assert service.backends["store"].connection is service.backends["db"]
    assert service.backends["second_store"].connection is service.backends["db"]
    assert service.backends["db"].config.count == 7
    app = create_app(service)
    assert app.state.service is service
    assert app.state.service.backends["db"] is service.backends["db"]
    assert registered_backends == ["connection", "store", "store"]
    view = service.backends
    view.clear()
    assert len(service.backends) == 3
    with pytest.raises(NotImplementedError):
        service.auto_train_skill()


@pytest.mark.parametrize("module", ["infra", "algos", "agents", "logging", "persistence", "envs"])
def test_all_backend_families(tmp_path, registered_backends, module):
    """Resolve every supported family through the shared registry mechanism."""
    config = {"backends": {"backend": {"module": module, "class_name": "Simple"}}}
    assert create_service(write_config(tmp_path, config)).backends["backend"].config.count == 1


@pytest.mark.parametrize("failure", ["class", "options", "missing", "cycle", "keyword", "required", "mode"])
def test_invalid_assembly_does_not_construct_backends(tmp_path, registered_backends, failure):
    """Reject invalid input before any backend constructor is called."""
    config = assembly_config()
    if failure == "class":
        config["backends"]["store"]["class_name"] = "Missing"
    elif failure == "options":
        config["backends"]["store"]["config"] = {"count": 0}
    elif failure == "missing":
        config["backends"]["store"]["dependencies"]["connection"] = "missing"
    elif failure == "cycle":
        config["backends"]["db"]["dependencies"] = {"store": "store"}
    elif failure == "keyword":
        config["backends"]["store"]["dependencies"] = {"wrong": "db"}
    elif failure == "required":
        config["backends"]["store"]["dependencies"] = {}
    else:
        config["mode"] = "missing"
    with pytest.raises((ValueError, TypeError)):
        create_service(write_config(tmp_path, config))
    assert registered_backends == []


@pytest.mark.parametrize(
    "content", ["", "[]", "null", "mode: local\nunknown: true", "backends: [", "!!python/object:object {}"]
)
def test_invalid_yaml(tmp_path, content):
    """Reject malformed, unsafe, non-mapping, and unknown-field input."""
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises((ValidationError, yaml.YAMLError)):
        create_service(path)


def test_missing_file(tmp_path):
    """Preserve a useful filesystem error for absent configuration."""
    with pytest.raises(FileNotFoundError):
        create_service(tmp_path / "missing.yaml")


def test_mutated_config_revalidated_before_construction(registered_backends):
    config = ServiceConfig.model_validate(assembly_config())
    config.backends["store"].dependencies["connection"] = "missing"
    with pytest.raises(ValueError, match="missing backend"):
        create_service(config)
    assert registered_backends == []


def test_builtin_loading_in_fresh_process(tmp_path):
    """Importing the package alone makes service assembly available."""
    path = write_config(tmp_path, {"backends": {}})
    source = Path(__file__).resolve().parents[2] / "src/mindmemos_skill"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from mindmemos_skill import create_service; import sys; assert create_service(sys.argv[1]).backends == {}",
            str(path),
        ],
        cwd=source,
        check=True,
    )
    assert create_service(path).backends == {}
