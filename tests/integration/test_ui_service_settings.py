"""Service configuration editing preserves workspace state and validates before writes."""

from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_skill.api import create_service
from mindmemos_skill.api.typing import ListRequest
from mindmemos_ui.app import create_app


@pytest_asyncio.fixture
async def editor(tmp_path):
    manager = ConfigManager(tmp_path / "home")
    manager.config_dir.mkdir(parents=True)
    with TestClient(await create_app(manager)) as client:
        client.headers["x-workbench-token"] = client.get("/api/session").json()["token"]
        yield client, manager.config_dir


@pytest.mark.asyncio
async def test_create_edit_reopen_and_explicit_path(editor, tmp_path, monkeypatch):
    http, home = editor
    document = http.get("/api/service-config").json()
    assert not document["exists"]
    content = "# Preserve this comment\n" + document["content"]
    response = http.put("/api/service-config", json={"content": content, "revision": None})
    assert response.status_code == 200
    saved = response.json()
    assert (home / "service.yaml").read_text() == content
    async with await create_service(home / "service.yaml") as service:
        assert service.default_project_id == "personal"
        assert await service.list_skills(ListRequest(project_id="personal")) == []
    config = yaml.safe_load(content)
    config["service"]["persistence"]["database"]["config"]["timeout"] = 17
    response = http.put("/api/service-config", json={"content": yaml.safe_dump(config), "revision": saved["revision"]})
    assert response.status_code == 200
    monkeypatch.setenv("MINDMEMOS_HOME", str(tmp_path / "sdk-only"))
    async with await create_service(home / "service.yaml") as service:
        assert service.default_project_id == "personal"
        assert service.backends["database"].config.timeout == 17
        assert service.backends["database"].engine.url.database == str(home / "skill/data/central.db")
    assert not (home / "workspace.json").exists()


def test_validation_conflicts_and_no_side_effects(editor):
    http, home = editor
    document = http.get("/api/service-config").json()
    for content in ["broken: [", "[]", "unknown: secret", "{}", document["content"].replace("sqlite", "missing")]:
        response = http.put("/api/service-config", json={"content": content})
        assert response.status_code == 422
        assert "secret" not in response.text
        assert not (home / "service.yaml").exists()
    body = {"content": document["content"], "revision": None}
    assert http.post("/api/service-config/validate", json=body).status_code == 200
    assert not (home / "service.yaml").exists()
    assert http.put("/api/service-config", json=body).status_code == 200
    assert http.put("/api/service-config", json=body).status_code == 409
    saved = (home / "service.yaml").read_text()
    assert http.put("/api/service-config", json={**body, "content": "invalid: true"}).status_code == 422
    assert (home / "service.yaml").read_text() == saved
    assert http.put("/api/service-config", json=body, headers={"x-workbench-token": "bad"}).status_code == 403
    assert http.get("/api/service-config", headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert http.get("/api/service-config").headers["cache-control"] == "no-store"


def test_no_connection_during_validation(editor, tmp_path):
    http, home = editor
    document = http.get("/api/service-config").json()
    config = yaml.safe_load(document["content"])
    database = tmp_path / "not-created.db"
    config["service"]["persistence"]["database"]["config"]["url"] = f"sqlite:///{database}"
    assert http.post("/api/service-config/validate", json={"content": yaml.safe_dump(config)}).status_code == 200
    assert not database.exists()
    assert not (home / "service.yaml").exists()


def test_sdk_v2_settings_route_is_available_in_workbench(editor):
    http, home = editor
    document = http.get("/api/client-configurations/sdk")
    assert document.status_code == 200
    assert document.json()["path"] == str(home / "settings.json")
    assert document.json()["values"]["version"] == 2


def test_symlink_is_not_followed(editor, tmp_path):
    http, home = editor
    target = tmp_path / "external.yaml"
    target.write_text("private content")
    (home / "service.yaml").symlink_to(target)
    response = http.get("/api/service-config")
    assert response.status_code == 400
    assert "private content" not in response.text


def test_memory_sdk_roundtrip_and_exports(editor):
    import base64
    import io
    import zipfile

    http, home = editor
    manager = ConfigManager(home)
    config = manager.load_or_default()
    config.memory_service.auth.api_key = "saved-key"
    manager.save(config)
    document = http.get("/api/configurations/memory/default").json()
    settings = yaml.safe_load(document["content"])
    settings["base_url"] = "https://new.example.org"
    settings["api_key"] = ""
    body = {
        "content": yaml.safe_dump(settings),
        "revision": document["revision"],
        "env": [
            {"name": "TEST_LITERAL", "value": "quotes' slash\\ and ${NO_INTERPOLATION}\nnext"},
        ],
    }
    response = http.put("/api/configurations/memory/default", json=body)
    assert response.status_code == 200
    assert response.json()["env"] == body["env"]
    assert manager.load().memory_service.auth.api_key == "saved-key"
    assert manager.load().memory_service.base_url == "https://new.example.org"
    download = http.get("/api/configurations/memory/default/export")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert set(archive.namelist()) == {"config.yaml", "config.env"}
        assert "skills" not in yaml.safe_load(archive.read("config.yaml"))
    imported = http.post(
        "/api/configurations/import",
        json={
            "archive": base64.b64encode(download.content).decode(),
        },
    ).json()
    assert imported["env"] == body["env"]
    assert yaml.safe_load(imported["content"])["base_url"] == settings["base_url"]
    imported["revision"] = None
    assert http.put("/api/configurations/memory/alternate", json=imported).status_code == 200
    assert (home / "config/memory/alternate.yaml").is_file()
    assert len(http.get("/api/configurations").json()["memory"]) == 2


def test_home_selection_and_environment_validation(editor, tmp_path):
    http, home = editor
    alternate = tmp_path / "alternate"
    params = {"home": str(alternate)}
    listing = http.get("/api/configurations", params=params).json()
    assert listing["home"] == str(alternate)
    assert not alternate.exists()
    document = http.get("/api/configurations/skill/test-profile", params=params).json()
    body = {"content": document["content"], "env": [{"name": "EXAMPLE", "value": "literal"}]}
    assert http.put("/api/configurations/skill/test-profile", params=params, json=body).status_code == 200
    assert (alternate / "config/skill/test-profile.yaml").exists()
    assert not (home / "config/skill/test-profile.yaml").exists()
    assert http.get("/api/session").json()["home"] == str(home)
    body["env"].append(body["env"][0])
    assert http.post("/api/configurations/skill/test-profile/validate", json=body).status_code == 422
    body["env"] = [{"name": "INVALID=NAME", "value": "secret"}]
    response = http.post("/api/configurations/skill/test-profile/validate", json=body)
    assert response.status_code == 422
    assert "secret" not in response.text
    assert http.get("/api/configurations/skill/..evil").status_code == 422


@pytest.mark.asyncio
async def test_companion_environment_loads_only_during_service_creation(editor, monkeypatch):
    import os

    http, home = editor
    env_name = "MINDMEMOS_UI_CONFIG_TEST_VALUE"
    monkeypatch.delenv(env_name, raising=False)
    document = http.get("/api/configurations/skill/default").json()
    body = {"content": document["content"], "env": [{"name": env_name, "value": "${literal}"}]}
    assert http.put("/api/configurations/skill/default", json=body).status_code == 200
    assert env_name not in os.environ
    try:
        async with await create_service(home / "service.yaml"):
            assert os.environ[env_name] == "${literal}"
        monkeypatch.setenv(env_name, "existing")
        monkeypatch.setenv("MINDMEMOS_HOME", str(home))
        async with await create_service(home / "service.yaml"):
            assert os.environ[env_name] == "existing"
    finally:
        os.environ.pop(env_name, None)


def test_failed_yaml_publish_restores_environment(editor, monkeypatch):
    import mindmemos_ui.service_settings as settings

    http, home = editor
    document = http.get("/api/service-config").json()
    body = {"content": document["content"], "env": [{"name": "EXAMPLE", "value": "before"}]}
    first = http.put("/api/service-config", json=body).json()
    atomic_write = settings._atomic_write

    def fail_yaml(path, content):
        if path.suffix == ".yaml":
            raise OSError("simulated failure")
        atomic_write(path, content)

    monkeypatch.setattr(settings, "_atomic_write", fail_yaml)
    body["revision"] = first["revision"]
    body["env"][0]["value"] = "after"
    assert http.put("/api/service-config", json=body).status_code == 400
    assert http.get("/api/service-config").json() == first


def test_default_home_uses_environment(editor, tmp_path, monkeypatch):
    http, runtime_home = editor
    monkeypatch.delenv("MINDMEMOS_HOME", raising=False)
    monkeypatch.delenv("MINDMEMOS_CONFIG_DIR", raising=False)
    assert http.get("/api/configurations/default-home").json() == {"home": str(Path.home() / ".mindmemos")}
    selected = tmp_path / "selected"
    monkeypatch.setenv("MINDMEMOS_HOME", str(selected))
    assert http.get("/api/configurations/default-home").json() == {"home": str(selected)}
    assert selected != runtime_home
    assert not selected.exists()

    selected.mkdir()
    (selected / "service.yaml").write_text("persistence: {}")
    listing = http.get("/api/configurations", params={"home": str(selected)}).json()
    assert listing["skill"][0]["exists"]
    assert not listing["memory"][0]["exists"]
