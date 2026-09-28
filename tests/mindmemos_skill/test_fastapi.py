"""Verify HTTP access delegates to the existing Skill service contract."""

import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from fastapi.testclient import TestClient
from mindmemos_skill.api.entries.config import FastAPIConfig
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.persistence.observability import AlgorithmAuditPersistence

from mindmemos_skill import MindMemOSSkill, create_service


@pytest_asyncio.fixture
async def service(tmp_path):
    """Build a transport-independent service through the public YAML factory."""
    path = tmp_path / "http.yaml"
    path.write_text("{}\n")
    return await create_service(path)


def test_factory_and_http_discovery(service):
    """Expose health, docs and all operations from the same service object."""
    assert isinstance(service, MindMemOSSkill)
    assert service.backends == {}
    app = create_app(service)
    assert app.state.service is service
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
    assert "501" in schema["paths"]["/skills/auto-evolve"]["post"]["responses"]
    assert "/skills/auto-train" not in schema["paths"]


@pytest.mark.asyncio
async def test_pending_operations_return_501(service):
    """Keep placeholder methods visibly unimplemented over HTTP."""
    with TestClient(create_app(service)) as client:
        response = client.post("/skills/auto-evolve")
        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"]
        assert client.get("/skills/auto-evolve").status_code == 405
    with pytest.raises(NotImplementedError):
        await service.auto_train_skill()


def test_http_calls_service_implementation():
    """Bind routes to the concrete service instead of separate business stubs."""
    calls = []

    class RecordingService(SkillService):
        async def register_skill(self) -> None:
            calls.append(self.backends)

        async def delete_skill(self) -> None:
            raise RuntimeError("Unexpected backend failure")

    service = RecordingService(backends={}, observability=AlgorithmAuditPersistence(None))
    with TestClient(create_app(service), raise_server_exceptions=False) as client:
        assert client.post("/skills/register").status_code == 200
        assert calls == [{}]
        assert client.post("/skills/delete").status_code == 500


def test_pipeline_allowlist_rejects_before_service_call():
    """Deny unknown pipeline names without entering the execution service."""
    calls = []

    class RecordingService(SkillService):
        async def auto_evolve_skill(self, request=None):
            calls.append(request)
            return {"status": "ok"}

    service = RecordingService(backends={}, observability=AlgorithmAuditPersistence(None))
    request = {"project_id": "project", "skill_ref": "skill", "pipeline": "local_only"}
    with TestClient(create_app(service)) as client:
        denied = client.post("/skills/auto-evolve", json=request)
        assert denied.status_code == 422
        assert denied.json()["error"] == "unsupport"
    with TestClient(create_app(service, api_config=FastAPIConfig(evolution_pipelines={"cloud"}))) as client:
        denied = client.post("/skills/auto-evolve", json=request)
        assert denied.status_code == 422
        assert denied.json()["error"] == "unsupport"
        allowed = client.post("/skills/auto-evolve", json=request | {"pipeline": "cloud"})
        assert allowed.status_code == 200
    assert len(calls) == 1
    assert calls[0].pipeline == "cloud"


def test_documented_http_allowlist_defaults_to_deny():
    """Validate the shipped YAML policy as an empty execution allowlist."""
    path = Path(__file__).resolve().parents[2] / "config/mindmemos_skill/fastapi.yaml"
    policy = FastAPIConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert policy.evolution_pipelines == set()


def test_local_import_does_not_load_http_dependencies():
    """Service assembly and entry imports do not load the optional HTTP stack."""
    source = Path(__file__).resolve().parents[2] / "src/mindmemos_skill"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, asyncio; from mindmemos_skill import ServiceConfig, create_service; "
            "from mindmemos_skill.api.entries.fastapi import create_app; "
            "service = asyncio.run(create_service(ServiceConfig())); "
            "assert service.backends == {}; "
            "assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules",
        ],
        cwd=source,
        check=True,
    )
