"""Form and YAML editing for client connections, without server configuration changes."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from filelock import FileLock
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.config.connections import MemoryConnectionConfig, SkillConnectionConfig
from pydantic import ValidationError

from .service_settings import _atomic_write
from .typing import ServiceConfigUpdate

Kind = Literal["memory", "skill"]
MODELS = {"memory": MemoryConnectionConfig, "skill": SkillConnectionConfig}


def parse_connection(kind: Kind, content: str):
    """Validate typed values and reject unsupported or duplicated YAML keys."""
    try:
        node = yaml.compose(content)
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("Expected a YAML mapping")
        keys = [key.value for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate YAML keys")
        return MODELS[kind].model_validate(yaml.safe_load(content))
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, item["loc"])) or "document" for item in exc.errors())
        raise HTTPException(422, f"Invalid configuration fields: {fields}") from exc
    except (yaml.YAMLError, ValueError, TypeError) as exc:
        raise HTTPException(422, "Expected valid YAML with unique connection fields") from exc


def connection_settings_router(runtime_home: Path) -> APIRouter:
    """Manage exactly one connection of each kind per selected client HOME."""
    router = APIRouter(prefix="/api/client-configurations")

    def root(home: str | None) -> Path:
        return (Path(home).expanduser() if home else runtime_home).resolve()

    def path_for(kind: Kind, home: str | None) -> Path:
        path = root(home) / ("settings.json" if kind == "memory" else "skill-client.yaml")
        if path.is_symlink():
            raise HTTPException(400, "Symbolic link configuration files are not supported")
        return path

    def read(kind: Kind, home: str | None):
        path = path_for(kind, home)
        raw = path.read_bytes() if path.exists() else None
        if kind == "memory" and raw is not None:
            config = ConfigManager(path.parent).load()
            values = MemoryConnectionConfig(
                base_url=config.base_url,
                api_key=config.auth.api_key,
                user_id=config.defaults.user_id,
                timeout_seconds=config.network.timeout_seconds,
            )
        elif raw is not None:
            values = parse_connection(kind, raw.decode("utf-8"))
        else:
            values = MODELS[kind]()
        return {
            "kind": kind,
            "name": "default",
            "path": str(path),
            "exists": raw is not None,
            "revision": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            "content": yaml.safe_dump(values.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            "values": values.model_dump(mode="json"),
        }

    @router.get("")
    def listing(home: str | None = None):
        return {
            "home": str(root(home)),
            **{
                kind: [{"name": "default", "path": str(path_for(kind, home)), "exists": path_for(kind, home).is_file()}]
                for kind in ("memory", "skill")
            },
        }

    @router.get("/{kind}/template")
    def template(kind: Kind):
        comments = (
            "# Memory FastAPI connection. No Skill or server algorithm fields.\n"
            "# api_key: null leaves credentials empty; user_id is optional.\n"
            if kind == "memory"
            else "# Skill FastAPI connection. mode: local | remote.\n"
            "# Local mode: loopback URL, API key may be null.\n"
            "# Project identity is assigned by the server; do not add project_id.\n"
            "# experiment_dir: client output/log directory, relative to client HOME.\n"
        )
        content = comments + "# timeout_seconds: 21600 = 6 hours. Algorithm requests are never retried.\n"
        content += yaml.safe_dump(MODELS[kind]().model_dump(mode="json"), sort_keys=False)
        return Response(
            content,
            media_type="application/yaml",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}-connection-template.yaml"',
            },
        )

    @router.post("/{kind}/normalize")
    def normalize(kind: Kind, body: ServiceConfigUpdate):
        if body.env:
            raise HTTPException(422, "Connection files do not accept ENV rows; use explicit connection fields")
        values = parse_connection(kind, body.content).model_dump(mode="json")
        return {"values": values, "content": yaml.safe_dump(values, sort_keys=False, allow_unicode=True)}

    @router.get("/{kind}/default")
    def get(kind: Kind, home: str | None = None):
        return read(kind, home)

    @router.put("/{kind}/default")
    def save(kind: Kind, body: ServiceConfigUpdate, home: str | None = None):
        values = normalize(kind, body)["values"]
        path = path_for(kind, home)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path.parent / ".config.lock")):
            current = read(kind, home)
            if current["revision"] != body.revision:
                raise HTTPException(409, "Configuration changed on disk; reload before saving")
            if kind == "memory":
                connection = MemoryConnectionConfig.model_validate(values)
                manager = ConfigManager(path.parent)
                config = manager.load_or_default()
                config.base_url = connection.base_url
                config.auth.api_key = connection.api_key
                config.defaults.user_id = connection.user_id
                config.network.timeout_seconds = connection.timeout_seconds
                config.network.max_retries = 0
                manager.save(config)
            else:
                _atomic_write(path, yaml.safe_dump(values, sort_keys=False, allow_unicode=True))
            return read(kind, home)

    @router.get("/{kind}/default/export")
    def export(kind: Kind, home: str | None = None):
        document = read(kind, home)
        if not document["exists"]:
            raise HTTPException(404, "Save the configuration before exporting")
        return Response(
            document["content"],
            media_type="application/yaml",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}-connection.yaml"',
            },
        )

    return router
