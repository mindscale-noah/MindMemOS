"""Connection form/YAML parity, saved credentials and singleton configuration files."""

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_ui.connection_settings import connection_settings_router


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(connection_settings_router(tmp_path))
    with TestClient(app) as http:
        yield http, tmp_path


@pytest.mark.parametrize("kind", ["memory", "skill"])
def test_template_form_save_roundtrip(client, kind):
    http, home = client
    template = http.get(f"/api/client-configurations/{kind}/template")
    assert template.status_code == 200
    values = yaml.safe_load(template.text)
    assert values["timeout_seconds"] == 21600
    assert values["api_key"] is None
    assert "project_id" not in values and "max_retries" not in values
    parsed = http.post(f"/api/client-configurations/{kind}/normalize", json={"content": template.text})
    assert parsed.status_code == 200
    assert parsed.json()["values"] == values
    endpoint = f"/api/client-configurations/{kind}/default"
    response = http.put(endpoint, json={"content": template.text})
    assert response.status_code == 200
    assert response.json()["values"] == values
    assert http.put(endpoint, json={"content": template.text}).status_code == 409
    exported = http.get(endpoint + "/export")
    assert yaml.safe_load(exported.text) == values
    assert not (home / "service.yaml").exists()
    assert len(http.get("/api/client-configurations").json()[kind]) == 1


def test_memory_preserves_registry_and_clears_key(client):
    http, home = client
    manager = ConfigManager(home)
    config = manager.load_or_default()
    config.skills = [{"id": "preserved"}]
    config.auth.api_key = "old-test-key"
    manager.save(config)
    endpoint = "/api/client-configurations/memory/default"
    document = http.get(endpoint).json()
    assert "skill_cache_dir" not in document["values"]
    values = document["values"] | {"api_key": "", "timeout_seconds": 21600}
    response = http.put(endpoint, json={"content": yaml.safe_dump(values), "revision": document["revision"]})
    assert response.status_code == 200
    saved = manager.load()
    assert saved.skills == config.skills
    assert saved.auth.api_key is None
    assert saved.network.timeout_seconds == 21600
    assert saved.network.max_retries == 0


@pytest.mark.parametrize(
    "content",
    [
        "project_id: anything",
        "max_retries: 3",
        "timeout_seconds: 0",
        "timeout_seconds: nope",
        "api_key: one\napi_key: two",
        "base_url: ftp://host",
        "mode: local\nbase_url: https://remote.example",
        "experiment_dir: ''",
        "persistence: {}",
        "[one, two]",
    ],
)
def test_invalid_yaml_is_rejected_without_writing(client, content):
    http, home = client
    response = http.put("/api/client-configurations/skill/default", json={"content": content})
    assert response.status_code == 422
    assert not (home / "skill-client.yaml").exists()


def test_remote_key_may_be_empty_and_directory_stays_literal(client):
    http, home = client
    content = "mode: remote\nbase_url: https://skills.example/api\napi_key: ''\nexperiment_dir: '实验 输出'\n"
    response = http.put("/api/client-configurations/skill/default", json={"content": content})
    assert response.status_code == 200
    assert response.json()["values"]["api_key"] is None
    assert response.json()["values"]["experiment_dir"] == "实验 输出"
    assert not (home / "实验 输出").exists()


def test_server_config_is_not_a_client_config(client):
    http, home = client
    (home / "service.yaml").write_text("persistence: {}")
    listing = http.get("/api/client-configurations").json()
    assert not listing["skill"][0]["exists"]
    assert (home / "service.yaml").read_text() == "persistence: {}"
