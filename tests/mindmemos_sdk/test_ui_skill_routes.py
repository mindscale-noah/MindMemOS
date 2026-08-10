"""HTTP integration tests for the centralized local Skill UI workflow."""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.errors import SkillCapabilityUnavailableError, SkillRemoteError
from mindmemos_sdk.ui import server


@contextmanager
def _running_ui(config_dir: Path) -> Iterator[tuple[httpx.Client, str]]:
    token = "test-launch-token"
    handler = functools.partial(
        server._LocalUIHandler,
        directory=str(server._static_directory()),
        config_manager=ConfigManager(config_dir=config_dir),
        launch_token=token,
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{httpd.server_address[1]}",
        headers={"X-MindMemOS-UI-Token": token},
    )
    try:
        yield client, token
    finally:
        client.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        'name: route-demo\ndescription: Route demo description\nversion: "1.0.0"\n\nInitial body\n',
        encoding="utf-8",
    )
    (source / "references").mkdir()
    (source / "references" / "private.md").write_text(
        "private reference\n",
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (SkillCapabilityUnavailableError("Skill remote sync is not configured"), 409, "skill_capability_unavailable"),
        (
            SkillRemoteError(
                "Skill remote service is unavailable",
                error_code="remote_unavailable",
                retryable=True,
                operation_id="operation-1",
            ),
            503,
            "remote_unavailable",
        ),
        (
            SkillRemoteError(
                "Skill remote rejected authentication",
                error_code="remote_unauthorized",
                retryable=False,
                request_id="request-1",
            ),
            401,
            "remote_unauthorized",
        ),
        (
            SkillRemoteError(
                "Skill remote rejected a conflicting request",
                error_code="remote_conflict",
                retryable=False,
            ),
            409,
            "remote_conflict",
        ),
    ],
)
def test_ui_skill_errors_use_structured_status_and_stable_code(error, expected_status, expected_code) -> None:
    captured = {}
    handler = object.__new__(server._LocalUIHandler)
    handler._send_json = lambda payload, *, status=200: captured.update(payload=payload, status=status)

    handler._send_skill_error(error)

    assert captured["status"] == expected_status
    assert captured["payload"]["error"] == expected_code


def test_skill_ui_keeps_registration_feedback_local_and_duplicate_policy_explicit():
    static_dir = server._static_directory()
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    javascript = (static_dir / "app.js").read_text(encoding="utf-8")
    css = (static_dir / "app.css").read_text(encoding="utf-8")

    assert "Ask before deciding" not in html
    assert '<option value="">Do not register it (default)</option>' in html
    library_start = html.index('id="skills-library-view"')
    register_start = html.index('id="skills-register-view"')
    assert library_start < html.index('id="register-skill"') < register_start
    assert register_start < html.index('id="register-skill-status"')
    assert html.index('id="register-skill-status"') < html.index('id="register-skill-path"')
    assert register_start < html.index('id="submit-register-skill"')
    assert 'setRegisterSkillStatus(duplicateHint, "error")' in javascript
    assert "skill.alias || skill.name" in javascript
    assert 'skill.description || "No description"' in javascript
    assert "latestVersionLabel(skill)" in javascript
    assert 'id="skill-editor-title"' in html
    assert 'id="skill-editor-subtitle" class="skill-document-name">SKILL.md' in html
    assert 'id="skill-file-tree"' in html
    assert 'id="skill-content-preview"' in html
    assert 'aria-label="Double-click to edit SKILL.md"' in html
    assert 'id="publish-skill-dialog"' in html
    assert 'id="publish-version-label"' in html
    assert 'id="skill-version-select"' in html
    assert 'id="skill-version-message"' in html
    assert 'id="open-export-skill"' in html
    assert html.index('id="skill-version-select"') < html.index('id="sync-skill"') < html.index('id="open-export-skill"')
    workspace_start = html.index('id="skills-library-view"')
    document_start = html.index('class="skill-folder-workspace"', workspace_start)
    info_start = html.index('class="panel-card skill-info-card"', workspace_start)
    for action_id in (
        "sync-skill",
        "open-export-skill",
        "evolve-skill",
        "open-delete-skill",
        "cancel-skill-edit",
        "publish-skill",
    ):
        assert workspace_start < html.index(f'id="{action_id}"') < document_start
    assert document_start < info_start
    assert "skill-editor-footer-actions" not in html
    assert javascript.count('headers: { "Idempotency-Key": crypto.randomUUID() }') == 1
    assert "operation_id: crypto.randomUUID()" not in javascript
    assert 'id="export-skill-dialog"' in html
    assert 'id="export-skill-path"' in html
    assert 'id="export-cli-command"' in html
    assert 'id="copy-export-command"' in html
    assert 'id="open-delete-skill"' in html
    assert 'id="delete-skill-dialog"' in html
    assert 'id="delete-skill-confirmation"' in html
    assert 'id="confirm-delete-skill"' in html
    assert 'id="skill-export-path"' not in html
    assert 'id="publish-commit-message"' in html
    assert 'id="skill-version-label"' not in html
    assert 'id="skill-commit-message"' not in html
    assert 'id="activate-skill-version"' not in html
    assert 'nextSkillVersionLabel()' in javascript
    assert 'addEventListener("dblclick", startSkillEditing)' in javascript
    assert 'files: versionedFiles' in javascript
    assert 'renderSkillFileTree' in javascript
    assert 'compactCommitMessage' in javascript
    assert 'commit_message: commitMessage || null' in javascript
    assert '"mindmemos skill export"' in javascript
    assert '"--version"' in javascript
    assert 'navigator.clipboard.writeText(command)' in javascript
    assert 'method: "DELETE"' in javascript
    assert 'captureCompareSelections()' in javascript
    assert 'Source files and cloud data were kept.' in javascript
    assert 'change "If an identical snapshot already exists"' in javascript
    assert '"Register a separate Skill"' in javascript
    assert ".register-skill-status[data-tone=\"error\"]" in css
    assert ".register-layout" in css
    assert ".checkbox-row" in css
    assert ".skill-editor-toolbar" in css
    assert "height: max(820px, calc(100vh - 160px))" in css
    assert "grid-template-columns: minmax(190px, 0.68fr) minmax(440px, 2fr) minmax(220px, 0.82fr)" in css
    assert ".delete-impact-grid" in css
    assert "white-space: nowrap" in css


def test_ui_http_register_publish_and_export_complete_snapshot(tmp_path):
    source = _source(tmp_path)
    export_dir = tmp_path / "exported"

    with _running_ui(tmp_path / "config") as (client, _token):
        registered_response = client.post(
            "/api/v1/skills/register",
            json={
                "source_path": str(source),
                "alias": "route-main",
                "version_label": "1.0.0",
                "commit_message": "Initial UI registration",
            },
        )
        assert registered_response.status_code == 201
        registered = registered_response.json()

        detail = client.get(f"/api/v1/skills/{registered['skill_id']}").json()
        assert detail["skill"]["latest_version_id"] == registered["version_id"]
        assert detail["skill"]["latest_version_label"] == "1.0.0"
        assert detail["versions"][0]["commit_message"] == "Initial UI registration"
        assert detail["versions"][0]["has_linked_files"] is True

        content_payload = client.get(f"/api/v1/skills/{registered['skill_id']}/content").json()
        assert set(content_payload["files"]) == {"SKILL.md", "references/private.md"}

        published_response = client.post(
            f"/api/v1/skills/{registered['skill_id']}/publish",
            json={
                "base_version_id": registered["version_id"],
                "files": {
                    "SKILL.md": 'name: route-demo\nversion: "1.1.0"\n\nPublished in UI\n',
                    "references/private.md": "edited private reference\n",
                },
                "version_label": "1.1.0",
                "commit_message": "Editor child version",
            },
        )
        assert published_response.status_code == 201
        published = published_response.json()
        version_id = published["result"]["version_id"]
        assert published["detail"]["skill"]["latest_version_id"] == version_id
        assert published["detail"]["skill"]["latest_version_label"] == "1.1.0"
        assert published["detail"]["versions"][-1]["commit_message"] == "Editor child version"

        exported_response = client.post(
            f"/api/v1/skills/{registered['skill_id']}/export",
            json={
                "target_path": str(export_dir),
                "version_id": version_id,
                "replace": True,
            },
        )
        assert exported_response.status_code == 200
        assert exported_response.json()["exported_files"] == [
            "SKILL.md",
            "references/private.md",
        ]

    assert (export_dir / "SKILL.md").read_text(encoding="utf-8").endswith(
        "Published in UI\n"
    )
    assert (export_dir / "references" / "private.md").read_text(
        encoding="utf-8"
    ) == "edited private reference\n"


def test_ui_http_delete_unregisters_local_family_without_deleting_source_or_cloud(tmp_path):
    source = _source(tmp_path)

    with _running_ui(tmp_path / "config") as (client, _token):
        registered = client.post(
            "/api/v1/skills/register",
            json={"source_path": str(source), "alias": "route-main"},
        ).json()
        published = client.post(
            f"/api/v1/skills/{registered['skill_id']}/publish",
            json={
                "base_version_id": registered["version_id"],
                "content": 'name: route-demo\nversion: "1.1.0"\n\nSecond version\n',
            },
        )
        assert published.status_code == 201

        blocked = client.delete(
            f"/api/v1/skills/{registered['skill_id']}",
            headers={"X-MindMemOS-UI-Token": "wrong"},
        )
        assert blocked.status_code == 403
        assert client.get("/api/v1/skills").json()["skills_count"] == 1

        response = client.delete(f"/api/v1/skills/{registered['skill_id']}")

        assert response.status_code == 200
        result = response.json()
        assert result == {
            "skill_id": registered["skill_id"],
            "name": "route-demo",
            "alias": "route-main",
            "deleted_version_count": 2,
            "deleted_pending_count": 2,
            "source_files_deleted": False,
            "cloud_skill_deleted": False,
        }
        assert client.get("/api/v1/skills").json()["skills"] == []

    assert source.is_dir()
    assert (source / "SKILL.md").is_file()


def test_ui_http_duplicate_registration_requires_explicit_choice(tmp_path):
    source = _source(tmp_path)

    with _running_ui(tmp_path / "config") as (client, _token):
        first = client.post(
            "/api/v1/skills/register",
            json={"source_path": str(source)},
        )
        assert first.status_code == 201

        undecided = client.post(
            "/api/v1/skills/register",
            json={"source_path": str(source)},
        )
        assert undecided.status_code == 400
        assert "duplicate_action" in undecided.json()["message"]

        reused = client.post(
            "/api/v1/skills/register",
            json={
                "source_path": str(source),
                "duplicate_action": "reuse",
            },
        )
        assert reused.status_code == 200
        assert reused.json()["action"] == "reused"
        assert reused.json()["skill_id"] == first.json()["skill_id"]


def test_ui_http_mutation_requires_launch_token(tmp_path):
    source = _source(tmp_path)

    with _running_ui(tmp_path / "config") as (client, _token):
        response = client.post(
            "/api/v1/skills/register",
            headers={"X-MindMemOS-UI-Token": "wrong"},
            json={"source_path": str(source)},
        )

    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"
