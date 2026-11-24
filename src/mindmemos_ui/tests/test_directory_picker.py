"""Native chooser routing, safe path transport and protected API behavior."""

import base64
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_ui.app import create_app
from mindmemos_ui.constants import DIRECTORY_PICKER_INITIAL_ENV

from mindmemos_ui import directory_picker


@pytest.mark.parametrize(
    "platform,desktop,program",
    [
        ("darwin", None, "/usr/bin/osascript"),
        ("win32", None, "powershell.exe"),
        ("linux", "zenity", "zenity"),
        ("linux", "kdialog", "kdialog"),
        ("linux", None, None),
    ],
)
def test_native_platform_commands(monkeypatch, tmp_path, platform, desktop, program):
    selected = tmp_path / "中文 folder ' $(literal)"
    selected.mkdir()
    calls = []
    monkeypatch.setattr(directory_picker.sys, "platform", platform)
    monkeypatch.setattr(directory_picker.shutil, "which", lambda name: name if name == desktop else None)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=str(selected) + "\n", stderr="")

    monkeypatch.setattr(directory_picker.subprocess, "run", run)
    assert directory_picker.choose_directory(selected / "missing") == selected
    command, options = calls[0]
    assert command[0] == (program or directory_picker.sys.executable)
    assert not options.get("shell")
    assert options["encoding"] == "utf-8"
    if platform == "win32":
        assert options["env"][DIRECTORY_PICKER_INITIAL_ENV] == str(selected)
        script = base64.b64decode(command[-1]).decode("utf-16-le")
        assert str(selected) not in script
        assert "FolderBrowserDialog" in script
        assert "-STA" in command
    elif platform == "darwin":
        assert command[-1] == str(selected)
        assert str(selected) not in command[2]
    elif program is None:
        assert command[1:3] == ["-m", "mindmemos_ui.directory_picker"]


@pytest.mark.parametrize("platform,returncode,stdout", [("darwin", 0, ""), ("win32", 0, ""), ("linux", 1, "")])
def test_cancel_preserves_selection(monkeypatch, tmp_path, platform, returncode, stdout):
    monkeypatch.setattr(directory_picker.sys, "platform", platform)
    monkeypatch.setattr(directory_picker.shutil, "which", lambda name: "zenity" if name == "zenity" else None)
    monkeypatch.setattr(
        directory_picker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr="",
        ),
    )
    assert directory_picker.choose_directory(tmp_path) is None


def test_failures_release_chooser_lock(monkeypatch, tmp_path):
    for failure in (OSError("no desktop"), subprocess.TimeoutExpired("picker", 300)):

        def fail(*args, **kwargs):
            raise failure

        monkeypatch.setattr(directory_picker.subprocess, "run", fail)
        with pytest.raises(RuntimeError):
            directory_picker.choose_directory(tmp_path)
        assert not directory_picker._picker_lock.locked()
    with directory_picker._picker_lock:
        with pytest.raises(RuntimeError, match="already open"):
            directory_picker.choose_directory(tmp_path)


@pytest.mark.asyncio
async def test_chooser_endpoint_selection_cancel_and_fallback(monkeypatch, tmp_path):
    from mindmemos_ui import service_settings

    home = tmp_path / "home"
    selected = tmp_path / "picked"
    selected.mkdir()
    calls = []

    def choose(initial):
        calls.append(initial)
        return selected

    monkeypatch.setattr(service_settings, "choose_directory", choose)
    with TestClient(await create_app(ConfigManager(home))) as http:
        endpoint = "/api/configurations/choose-directory"
        body = {"home": str(home)}
        assert http.post(endpoint, json=body).status_code == 403
        assert calls == []
        http.headers["x-workbench-token"] = http.get("/api/session").json()["token"]
        assert http.post(endpoint, json=body, headers={"Origin": "https://other.example"}).status_code == 403
        assert calls == []
        result = http.post(endpoint, json=body)
        assert result.json() == {"path": str(selected), "unavailable": False}
        assert calls == [home]
        assert http.get("/api/session").json()["home"] == str(home)
        assert not (selected / "workspace.json").exists()
        monkeypatch.setattr(service_settings, "choose_directory", lambda initial: None)
        assert http.post(endpoint, json=body).json() == {"path": None, "unavailable": False}

        def unavailable(initial):
            raise RuntimeError("No desktop")

        monkeypatch.setattr(service_settings, "choose_directory", unavailable)
        assert http.post(endpoint, json=body).json() == {"path": None, "unavailable": True}


def test_linux_display_failure_is_not_treated_as_cancel(monkeypatch, tmp_path):
    monkeypatch.setattr(directory_picker.sys, "platform", "linux")
    monkeypatch.setattr(directory_picker.shutil, "which", lambda name: "zenity")
    monkeypatch.setattr(
        directory_picker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Cannot open display",
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        directory_picker.choose_directory(tmp_path)
