"""The UI edits one complete SDK v2 settings.v2.json document."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_ui.sdk_settings import sdk_settings_router


def _client(home):
    app = FastAPI()
    app.include_router(sdk_settings_router(home))
    return TestClient(app)


def test_official_and_custom_connections_share_one_settings_file(tmp_path):
    home = tmp_path / "home"
    with _client(home) as http:
        initial = http.get("/api/client-configurations/sdk").json()
        assert initial["exists"] is False
        assert initial["values"]["version"] == 2
        assert initial["values"]["service_type"] == "official"
        values = initial["values"]
        values["memory_service"]["auth"]["api_key"] = "memory-key"
        first = http.put(
            "/api/client-configurations/sdk",
            json={"content": json.dumps(values), "revision": None},
        )
        assert first.status_code == 200, first.text
        assert ConfigManager(home).load().resolve_skill_connection() == (
            "https://mindmemos.cn/api/developer/skill",
            "memory-key",
        )

        values = first.json()["values"]
        values["service_type"] = "custom"
        values["memory_service"]["base_url"] = "https://memory.example"
        values["skill_service"] = {"base_url": "https://skill.example", "auth": {"api_key": "skill-key"}}
        updated = http.put(
            "/api/client-configurations/sdk",
            json={"content": json.dumps(values), "revision": first.json()["revision"]},
        )
        assert updated.status_code == 200, updated.text
        saved = ConfigManager(home).load()
        assert saved.resolve_memory_connection() == ("https://memory.example", "memory-key")
        assert saved.resolve_skill_connection() == ("https://skill.example", "skill-key")
        assert set(json.loads((home / "settings.v2.json").read_text())) == set(updated.json()["values"])
        assert "storage" not in updated.json()["values"]
        assert not (home / "skill-client.yaml").exists()


def test_v1_settings_are_read_without_replacing_the_legacy_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "base_url": "https://memory.example",
                "auth": {"api_key": "old-key"},
                "skills": [{"skill_id": "existing", "skill_name": "Existing", "path": str(home / "skill")}],
            }
        )
    )
    with _client(home) as http:
        opened = http.get("/api/client-configurations/sdk")
        assert opened.status_code == 200
        assert opened.json()["values"]["version"] == 2
        assert opened.json()["values"]["memory_service"]["auth"]["api_key"] == "old-key"
        assert json.loads((home / "settings.json").read_text())["version"] == 1
        assert not (home / "settings.v2.json").exists()
        assert not (home / "skill_registry.json").exists()


def test_invalid_json_and_stale_revision_do_not_write(tmp_path):
    home = tmp_path / "home"
    with _client(home) as http:
        initial = http.get("/api/client-configurations/sdk").json()
        valid = json.dumps(initial["values"])
        for source in [
            valid.replace('"version": 2', '"version": 2, "version": 2'),
            valid.replace('"version": 2', '"version": 1'),
            valid.replace('"official"', '"custom"'),
        ]:
            response = http.put("/api/client-configurations/sdk", json={"content": source})
            assert response.status_code == 422
            assert not (home / "settings.v2.json").exists()
        assert http.put("/api/client-configurations/sdk", json={"content": valid}).status_code == 200
        assert http.put("/api/client-configurations/sdk", json={"content": valid}).status_code == 409
