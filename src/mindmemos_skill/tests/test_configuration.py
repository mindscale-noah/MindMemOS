"""Deployment configuration owns data locations and project identity."""

import asyncio
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.api.auth import SkillHttpIdentity
from mindmemos_skill.api.typing import FileSkillInfo, ListRequest, RegisterFromLocalRequest, ShowRequest
from mindmemos_skill.config.config import HttpAuthConfig
from mindmemos_skill.config.loader import read_skill_config
from pydantic import ValidationError

from mindmemos_skill import ServiceConfig, SkillConfig, create_service

from .service_config import repository_config


def write_config(path: Path, home: str = "state", project: str = "personal") -> Path:
    """Write an explicit none-mode configuration next to its data root."""
    config = repository_config(Path(home), project).model_dump(mode="json")
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.asyncio
async def test_yaml_anchors_home_and_reopens_data_across_working_directories(tmp_path, monkeypatch):
    config = write_config(tmp_path / "service.yaml")
    monkeypatch.setenv("MINDMEMOS_HOME", str(tmp_path / "sdk"))
    monkeypatch.setenv("MINDMEMOS_CONFIG_DIR", str(tmp_path / "legacy"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: Example\ndescription: Instructions\n---\n# Example")
    async with await create_service(config) as service:
        item = await service.register_from_local(
            RegisterFromLocalRequest(project_id="personal", skill=FileSkillInfo(path=str(source)))
        )
        assert service.default_project_id == "personal"
        assert service.backends["database"].engine.url.database == str(tmp_path / "state/data/central.db")
    monkeypatch.chdir(tmp_path.parent)
    async with await create_service(config) as reopened:
        restored = await reopened.show_skill(ShowRequest(project_id="personal", skill_id=item.skill_id))
        assert restored.skill.skill_id == item.skill_id
    assert not (tmp_path / "sdk").exists()
    assert not (tmp_path / "state/workspace.json").exists()


@pytest.mark.asyncio
async def test_home_does_not_determine_project_and_explicit_project_isolation(tmp_path):
    config = repository_config(tmp_path / "state", "one")
    async with await create_service(config) as service:
        assert service.default_project_id == "one"
        with pytest.raises(ValueError, match="default project"):
            await service.list_skills(ListRequest(project_id="two"))
    other = config.model_copy(deep=True)
    other.fastapi.auth.default_project_id = "two"
    async with await create_service(other) as service:
        assert service.default_project_id == "two"
        assert await service.list_skills(ListRequest(project_id=service.default_project_id)) == []
    config.service.home = tmp_path / "another"
    async with await create_service(config) as service:
        assert service.default_project_id == "one"


@pytest.mark.asyncio
async def test_default_database_urls_and_explicit_connections(tmp_path):
    config = write_config(tmp_path / "service.yaml")
    value = yaml.safe_load(config.read_text())
    value["service"]["observability"] = {"backend": {"class_name": "sqlite"}}
    config.write_text(yaml.safe_dump(value))
    async with await create_service(config) as service:
        assert service.backends["observability_database"].engine.url.database == str(
            tmp_path / "state/data/observability.db"
        )
    explicit = tmp_path / "elsewhere.db"
    value["service"]["persistence"]["database"]["config"]["url"] = f"sqlite:///{explicit}"
    config.write_text(yaml.safe_dump(value))
    async with await create_service(config) as service:
        assert service.backends["database"].engine.url.database == str(explicit)


@pytest.mark.asyncio
async def test_concurrent_initialization_uses_configured_project(tmp_path):
    config = repository_config(tmp_path / "state", "personal")

    async def open_one():
        async with await create_service(config) as service:
            assert await service.list_skills(ListRequest(project_id=service.default_project_id)) == []
            return service.default_project_id

    assert set(await asyncio.gather(*(open_one() for _ in range(8)))) == {"personal"}


@pytest.mark.parametrize(
    "auth",
    [
        {"mode": "none"},
        {"mode": "none", "default_project_id": " "},
        {"mode": "local", "api_key_file": "key.yaml", "default_project_id": "personal"},
        {"mode": "gateway_jwt", "gateway_jwt_secret": "secret", "default_project_id": "personal"},
    ],
)
def test_invalid_authentication_combinations_are_rejected(auth):
    with pytest.raises(ValidationError):
        HttpAuthConfig.model_validate(auth)


def test_home_required_and_resource_only_yaml_rejected(tmp_path):
    with pytest.raises(ValidationError):
        ServiceConfig()
    path = tmp_path / "service.yaml"
    path.write_text("home: state\n")
    with pytest.raises(ValidationError):
        read_skill_config(path)


def test_local_key_file_and_tilde_paths_resolve_from_yaml(tmp_path):
    value = repository_config(Path("state"), "personal").model_dump(mode="json")
    value["fastapi"]["auth"] = {"mode": "local", "api_key_file": "secrets/key.yaml"}
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(value))
    config = read_skill_config(path)
    assert config.service.home == tmp_path / "state"
    assert config.fastapi.auth.api_key_file == str(tmp_path / "secrets/key.yaml")
    value["service"]["home"] = "~/skill-state"
    path.write_text(yaml.safe_dump(value))
    assert read_skill_config(path).service.home == Path.home() / "skill-state"


@pytest.mark.asyncio
async def test_authenticated_direct_calls_require_verified_principal(tmp_path):
    config = repository_config(tmp_path / "state").model_dump(mode="json")
    config["fastapi"]["auth"] = {"mode": "gateway_jwt", "gateway_jwt_secret": "secret"}
    async with await create_service(SkillConfig.model_validate(config)) as service:
        assert service.default_project_id is None
        with pytest.raises(PermissionError, match="authorization is required"):
            await service.list_skills(ListRequest(project_id="one"))
        with service.request_principal(SkillHttpIdentity(project_id="one")):
            assert await service.list_skills(ListRequest(project_id="one")) == []
            with pytest.raises(PermissionError):
                await service.list_skills(ListRequest(project_id="two"))
