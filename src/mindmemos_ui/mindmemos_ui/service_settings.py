"""Local Memory and Skill configuration files, environment rows and downloads."""

import base64
import binascii
import hashlib
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from filelock import FileLock
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.config.paths import get_mindmemos_home
from mindmemos_sdk.skills.cloud import SkillCloudClient
from mindmemos_sdk.transport import HttpTransport
from mindmemos_skill.api.factory import SERVICE_CONFIG_FILENAME, default_service_config, load_service_config
from pydantic import ValidationError

from .config import SettingsUpdate, apply_sdk_settings
from .directory_picker import choose_directory
from .typing import ConfigImportRequest, DirectorySelectionRequest, ServiceConfigUpdate

ConfigKind = Literal["memory", "skill"]


def _validate(content: str, kind: ConfigKind) -> None:
    """Validate without constructing resources or applying environment variables."""
    try:
        if kind == "skill":
            config = load_service_config(content)
            if config.persistence is None or config.repository_pipeline is None:
                raise HTTPException(422, "The local workbench requires persistence and repository_pipeline")
        else:
            SettingsUpdate.model_validate(yaml.safe_load(content))
    except yaml.YAMLError as exc:
        raise HTTPException(422, "Invalid YAML syntax; check indentation and punctuation") from exc
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) or "document" for error in exc.errors())
        raise HTTPException(422, f"Invalid configuration fields: {fields}") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, "Invalid backend definition; check class_name and pipeline options") from exc


def _atomic_write(path: Path, content: str) -> None:
    """Publish one UTF-8 file with private permissions and no partial content."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".config-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path, kind: ConfigKind) -> dict:
    """Read source and sibling environment without interpolating or applying values."""
    env_path = path.with_suffix(".env")
    if path.is_symlink() or env_path.is_symlink():
        raise HTTPException(400, "Symbolic link configuration files are not supported")
    content = path.read_text(encoding="utf-8") if path.exists() else None
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if kind == "memory":
        manager = ConfigManager(path.parent if path.suffix == ".json" else path.parents[2])
        config = manager.load_or_default()
        memory_content = yaml.safe_dump(
            {
                "base_url": config.base_url,
                "api_key": config.auth.api_key,
                "user_id": config.defaults.user_id,
                "skill_cache_dir": config.storage.skill_cache_dir,
                "skill_backup_dir": config.storage.skill_backup_dir,
                "timeout_seconds": config.network.timeout_seconds,
                "max_retries": config.network.max_retries,
            },
            sort_keys=False,
        )
    template = (
        yaml.safe_dump(default_service_config().model_dump(mode="json", exclude_none=True), sort_keys=False)
        if kind == "skill"
        else memory_content
    )
    return {
        "path": str(path),
        "env_path": str(env_path),
        "exists": content is not None,
        "content": memory_content
        if kind == "memory" and path.suffix == ".json"
        else (content if content is not None else template),
        "env": [
            {"name": key, "value": value or ""}
            for key, value in dotenv_values(stream=io.StringIO(env_text), interpolate=False).items()
        ],
        "revision": hashlib.sha256(((content or "") + "\0" + env_text).encode()).hexdigest()
        if content is not None or env_text
        else None,
    }


def service_settings_router(home: Path) -> APIRouter:
    """Expose configuration libraries under a user-selected home.

    Args:
        home: Initial workspace, used when no alternative home is selected.

    Returns:
        Same-origin routes protected by the enclosing workbench middleware.
    """
    router = APIRouter()

    @router.get("/api/configurations/default-home")
    def default_home():
        return {"home": str(get_mindmemos_home().resolve())}

    def root_for(selected: str | None) -> Path:
        return (Path(selected).expanduser() if selected else home).resolve()

    @router.post("/api/configurations/choose-directory")
    def select_directory(body: DirectorySelectionRequest):
        try:
            selected = choose_directory(root_for(body.home))
        except RuntimeError:
            return {"path": None, "unavailable": True}
        return {"path": str(selected) if selected is not None else None, "unavailable": False}

    def path_for(kind: ConfigKind, name: str, selected: str | None) -> Path:
        if not re.fullmatch(r"[\w-]{1,80}", name):
            raise HTTPException(422, "Use letters, digits, underscores or hyphens for the configuration name")
        root = root_for(selected)
        if name == "default":
            return root / (SERVICE_CONFIG_FILENAME if kind == "skill" else "settings.json")
        directory = root / "config" / kind
        if not directory.resolve().is_relative_to(root):
            raise HTTPException(400, "Configuration directory must stay inside HOME")
        return directory / f"{name}.yaml"

    def read(kind: ConfigKind, name: str, selected: str | None):
        return {**_read(path_for(kind, name, selected), kind), "name": name, "kind": kind}

    def save(kind: ConfigKind, name: str, body: ServiceConfigUpdate, selected: str | None):
        _validate(body.content, kind)
        path = path_for(kind, name, selected)
        path.parent.mkdir(parents=True, exist_ok=True)
        env_path = path.with_suffix(".env")
        with FileLock(str(path.parent / ".config.lock")):
            current = _read(path, kind)
            if current["revision"] != body.revision:
                raise HTTPException(409, "Configuration changed on disk; reload before saving")
            previous_env = env_path.read_text(encoding="utf-8") if env_path.exists() else None
            env_text = "".join(
                item.name + "='" + item.value.replace("\\", "\\\\").replace("'", "\\'") + "'\n" for item in body.env
            )
            try:
                _atomic_write(env_path, env_text)
                if kind == "memory" and name == "default":
                    manager = ConfigManager(path.parent)
                    config = manager.load_or_default()
                    apply_sdk_settings(config, SettingsUpdate.model_validate(yaml.safe_load(body.content)))
                    manager.save(config)
                else:
                    _atomic_write(path, body.content)
            except OSError:
                if previous_env is None:
                    env_path.unlink(missing_ok=True)
                else:
                    _atomic_write(env_path, previous_env)
                raise
            return read(kind, name, selected)

    @router.get("/api/configurations")
    def list_configurations(home: str | None = None):
        root = root_for(home)
        groups = {}
        for kind in ("memory", "skill"):
            default = path_for(kind, "default", home)
            paths = [default]
            directory = path_for(kind, "placeholder", home).parent
            if directory.exists():
                paths.extend(sorted(directory.glob("*.yaml")))
            groups[kind] = [
                {"name": "default" if path == default else path.stem, "exists": path.is_file(), "path": str(path)}
                for path in paths
                if not path.is_symlink() and re.fullmatch(r"[\w-]{1,80}", path.stem)
            ]
        return {"home": str(root), "runtime_home": str(root_for(None)), **groups}

    @router.post("/api/configurations/import")
    def import_configuration(body: ConfigImportRequest):
        try:
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(body.archive, validate=True))) as archive:
                if set(archive.namelist()) != {"config.yaml", "config.env"}:
                    raise HTTPException(422, "Expected config.yaml and config.env in the ZIP")
                if sum(item.file_size for item in archive.infolist()) > 1000000:
                    raise HTTPException(422, "Configuration archive is too large")
                content = archive.read("config.yaml").decode("utf-8")
                env = archive.read("config.env").decode("utf-8")
            return {
                "content": content,
                "env": [
                    {"name": key, "value": value or ""}
                    for key, value in dotenv_values(stream=io.StringIO(env), interpolate=False).items()
                ],
            }
        except (ValueError, binascii.Error, zipfile.BadZipFile, RuntimeError) as exc:
            raise HTTPException(422, "Invalid configuration archive") from exc

    @router.get("/api/configurations/{kind}/{name}")
    def get_configuration(kind: ConfigKind, name: str, home: str | None = None):
        return read(kind, name, home)

    @router.put("/api/configurations/{kind}/{name}")
    def save_configuration(kind: ConfigKind, name: str, body: ServiceConfigUpdate, home: str | None = None):
        return save(kind, name, body, home)

    @router.post("/api/configurations/{kind}/{name}/validate")
    def validate_configuration(kind: ConfigKind, name: str, body: ServiceConfigUpdate):
        _validate(body.content, kind)
        return {"ok": True}

    @router.post("/api/configurations/memory/{name}/test")
    def test_memory_connection(name: str, home: str | None = None):
        document = read("memory", name, home)
        if not document["exists"]:
            raise HTTPException(400, "Save the configuration before testing")
        config = SettingsUpdate.model_validate(yaml.safe_load(document["content"]))
        with HttpTransport(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        ) as transport:
            SkillCloudClient(transport).list_skills()
        return {"ok": True}

    @router.get("/api/configurations/{kind}/{name}/export")
    def export_configuration(kind: ConfigKind, name: str, home: str | None = None):
        path = path_for(kind, name, home)
        if not path.is_file():
            raise HTTPException(404, "Save the configuration before exporting")
        document = read(kind, name, home)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("config.yaml", document["content"])
            env_path = path.with_suffix(".env")
            archive.writestr("config.env", env_path.read_text(encoding="utf-8") if env_path.exists() else "")
        return Response(
            buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{kind}-config.zip"'},
        )

    @router.get("/api/service-config")
    def get_config():
        return read("skill", "default", None)

    @router.post("/api/service-config/validate")
    def validate_config(body: ServiceConfigUpdate):
        _validate(body.content, "skill")
        return {"ok": True}

    @router.put("/api/service-config")
    def save_config(body: ServiceConfigUpdate):
        return save("skill", "default", body, None)

    return router
