"""Complete deployment YAML drives service assembly and HTTP identity."""

import pytest
from fastapi.testclient import TestClient
from mindmemos_skill.api.entries.fastapi import app
from mindmemos_skill.config.config import ImplementationConfig
from mindmemos_skill.config.loader import load_skill_config

from mindmemos_skill import ServiceConfig, create_service


def test_service_resources_do_not_select_http_authentication():
    config = load_skill_config(
        "service:\n  home: .\n  disabled_services: [training]\n"
        "fastapi:\n  auth:\n    mode: gateway_jwt\n    gateway_jwt_secret: test-secret\n"
    )
    assert config.service.disabled_services == {"training"}
    assert "fastapi" not in type(config.service).model_fields


@pytest.mark.asyncio
async def test_disabled_capabilities_are_not_assembled(tmp_path):
    config = ServiceConfig(
        home=tmp_path,
        disabled_services={"training"},
        training_pipelines={"unavailable": ImplementationConfig(class_name="unknown")},
    )
    async with await create_service(config) as service:
        assert service.disabled_services == {"training"}
        assert service.backends == {}
        with pytest.raises(NotImplementedError):
            _ = service.repository


def test_fastapi_entry_loads_joint_configuration(tmp_path, monkeypatch):
    config_path = tmp_path / "service.yaml"
    config_path.write_text(
        "service:\n  home: state\n  persistence:\n    database:\n      class_name: sqlite\n"
        "  repository_pipeline:\n    class_name: repository\n"
        "fastapi:\n  auth:\n    mode: none\n    default_project_id: personal\n"
    )
    monkeypatch.setenv("MINDMEMOS_SKILL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MINDMEMOS_SKILL_API_HOST", "127.0.0.1")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/v1/skills/list", json={}).status_code == 200
        assert app.state.service.default_project_id == "personal"


@pytest.mark.asyncio
async def test_resource_only_python_service_has_no_implicit_project(tmp_path):
    async with await create_service(ServiceConfig(home=tmp_path)) as service:
        assert service.default_project_id is None
        assert service.api_config is None
