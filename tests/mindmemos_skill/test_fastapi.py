"""Verify HTTP access delegates to the existing Skill service contract."""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.service import SkillService

from mindmemos_skill import MindMemOSSkill, create_service


@pytest.fixture
def service(tmp_path):
    """Build a transport-independent service through the public YAML factory."""
    path = tmp_path / "http.yaml"
    path.write_text("backends: {}\n")
    return create_service(path)


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
    for path in ("/skills/auto-evolve", "/skills/auto-train"):
        assert "501" in schema["paths"][path]["post"]["responses"]


@pytest.mark.parametrize("path", ["auto-evolve", "auto-train"])
def test_pending_operations_return_501(service, path):
    """Keep placeholder methods visibly unimplemented over HTTP."""
    with TestClient(create_app(service)) as client:
        response = client.post(f"/skills/{path}")
        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"]
        assert client.get(f"/skills/{path}").status_code == 405
    with pytest.raises(NotImplementedError):
        service.auto_train_skill()


def test_http_calls_service_implementation():
    """Bind routes to the concrete service instead of separate business stubs."""
    calls = []

    class RecordingService(SkillService):
        def register_skill(self) -> None:
            calls.append(self.backends)

        def delete_skill(self) -> None:
            raise RuntimeError("Unexpected backend failure")

    service = RecordingService(backends={})
    with TestClient(create_app(service), raise_server_exceptions=False) as client:
        assert client.post("/skills/register").status_code == 200
        assert calls == [{}]
        assert client.post("/skills/delete").status_code == 500


def test_local_import_does_not_load_http_dependencies():
    """Service assembly and entry imports do not load the optional HTTP stack."""
    source = Path(__file__).resolve().parents[2] / "src/mindmemos_skill"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mindmemos_skill import ServiceConfig, create_service; "
            "from mindmemos_skill.api.entries.fastapi import create_app; "
            "from mindmemos_skill.api.entries.local import main; "
            "service = create_service(ServiceConfig()); "
            "assert service.backends == {}; "
            "assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules",
        ],
        cwd=source,
        check=True,
    )
