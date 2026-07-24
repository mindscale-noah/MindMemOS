"""Dependency-free local server for the SDK console and its JSON API."""

from __future__ import annotations

import functools
import http.server
import json
import threading
import webbrowser
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from ..config import ConfigManager, SDKConfig, mask_secret
from ..errors import ConfigError, MindMemOSSDKError
from ..memory import MemoryClient
from ..memory.core import MemoryDefaults
from ..skills import SkillCloudClient, SkillManager
from ..skills.bundle import bundle_files_from_content, compute_content_hash, read_local_bundle
from ..transport import HttpTransport


class _LocalUIHandler(http.server.SimpleHTTPRequestHandler):
    """Serve packaged assets and a small local-only JSON API."""

    server_version = "MindMemOSLocalUI/0.1"

    def __init__(self, *args: object, config_manager: ConfigManager, **kwargs: object) -> None:
        self._config_manager = config_manager
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/api/v1/"):
            self._handle_api_get(path)
            return
        super().do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/v1/config":
            self._handle_config_update()
            return
        if path.startswith("/api/v1/skills/") and path.endswith("/content"):
            self._handle_skill_write(path.removesuffix("/content"), publish=False)
            return
        self._send_json({"error": "not_found", "message": "Unknown API route."}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/api/v1/skills/") and path.endswith("/publish"):
            self._handle_skill_write(path.removesuffix("/publish"), publish=True)
            return
        self._send_json({"error": "not_found", "message": "Unknown API route."}, status=404)

    def _handle_api_get(self, path: str) -> None:
        try:
            if path == "/api/v1/health":
                self._send_json({"ok": True, "service": "mindmemos-sdk-ui"})
                return
            if path == "/api/v1/config":
                self._send_json(_config_payload(self._config_manager))
                return
            if path == "/api/v1/skills":
                self._send_json(_skills_payload(self._config_manager))
                return
            if path in {"/api/v1/memories", "/api/v1/memories/search"}:
                self._handle_memory_get(path)
                return
            if path.startswith("/api/v1/skills/"):
                self._handle_skill_get(path)
                return
            self._send_json({"error": "not_found", "message": "Unknown API route."}, status=404)
        except (ConfigError, MindMemOSSDKError, OSError, ValueError) as exc:
            self._send_json({"error": "sdk_error", "message": str(exc)}, status=400)

    def _handle_memory_get(self, path: str) -> None:
        query = parse_qs(urlsplit(self.path).query)
        client, transport, config = _memory_client(self._config_manager)
        try:
            user_id = config.defaults.user_id
            if not user_id:
                raise ValueError("Configure a User ID in Settings before loading Memory.")
            top_k = _query_top_k(query)
            if path.endswith("/search"):
                search_query = (query.get("q") or query.get("query") or [""])[0].strip()
                if not search_query:
                    raise ValueError("A search query is required.")
                kwargs: dict[str, object] = {"user_id": user_id}
                if top_k is not None:
                    kwargs["top_k"] = top_k
                result = client.search(search_query, **kwargs)
                mode = "search"
            else:
                kwargs = {"filters": _owned_memory_filters(config.memory.get_filters, user_id)}
                if top_k is not None:
                    kwargs["top_k"] = top_k
                result = client.get(**kwargs)
                mode = "list"
            self._send_json(
                {
                    "memories": [item.model_dump(mode="json") for item in result.memories],
                    "count": len(result.memories),
                    "mode": mode,
                    "user_id": user_id,
                    "request_id": result.request_id,
                }
            )
        finally:
            transport.close()

    def _handle_skill_get(self, path: str) -> None:
        suffix = path.removeprefix("/api/v1/skills/")
        parts = [unquote(part) for part in suffix.split("/") if part]
        if not parts:
            self._send_json({"error": "not_found", "message": "Skill reference is required."}, status=404)
            return
        skill_ref = parts[0]
        manager, transport = _skill_manager(self._config_manager)
        try:
            record = manager.show(skill_ref)
            if len(parts) == 1:
                self._send_json(_skill_detail_payload(manager, record))
                return
            if parts[1] == "content":
                query = parse_qs(urlsplit(self.path).query)
                version_id = query.get("version_id", [None])[0]
                content = _skill_content(manager, record, version_id)
                self._send_json({"skill_id": record.skill_id, "version_id": version_id, "content": content})
                return
            self._send_json({"error": "not_found", "message": "Unknown Skill route."}, status=404)
        finally:
            transport.close()

    def _handle_skill_write(self, path: str, *, publish: bool) -> None:
        suffix = path.removeprefix("/api/v1/skills/")
        parts = [unquote(part) for part in suffix.split("/") if part]
        if len(parts) != 1:
            self._send_json({"error": "not_found", "message": "Skill reference is required."}, status=404)
            return

        manager, transport = _skill_manager(self._config_manager)
        try:
            payload = self._read_json()
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Skill content must be a non-empty string.")
            record = manager.show(parts[0])
            manager.save_content(record.skill_id, content=content)
            if publish:
                version_label = payload.get("version_label")
                if version_label is not None and not isinstance(version_label, str):
                    raise ValueError("version_label must be a string or null.")
                label = version_label.strip() if isinstance(version_label, str) else None
                record = manager.push(record.skill_id, version_label=label or None)
                message = f"Published version {record.version_label or record.base_version_id}."
            else:
                record = manager.show(record.skill_id)
                message = "Saved the local Skill content."
            self._send_json({**_skill_detail_payload(manager, record), "message": message})
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "skill_write_failed", "message": str(exc)}, status=400)
        finally:
            transport.close()

    def _handle_config_update(self) -> None:
        try:
            payload = self._read_json()
            config = self._config_manager.load_or_default()
            _apply_config_update(config, payload)
            validated = SDKConfig.model_validate(config.model_dump())
            self._config_manager.save(validated)
            self._send_json(_config_payload(self._config_manager))
        except (ConfigError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": "invalid_config", "message": str(exc)}, status=400)

    def _read_json(self) -> dict[str, object]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ValueError("Content-Length is required.")
        length = int(length_header)
        if length > 2_000_000:
            raise ValueError("Request body is too large.")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object.")
        return value

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _static_directory() -> Path:
    """Resolve the packaged static directory in a source tree or wheel."""
    return Path(files("mindmemos_sdk.ui").joinpath("static"))


def _config_payload(config_manager: ConfigManager) -> dict[str, object]:
    config = config_manager.load_or_default()
    return {
        "config_path": str(config_manager.config_path),
        "base_url": config.base_url,
        "api_key_configured": bool(config.auth.api_key),
        "api_key_masked": mask_secret(config.auth.api_key),
        "defaults": config.defaults.model_dump(mode="json"),
        "memory": config.memory.model_dump(mode="json"),
        "storage": config.storage.model_dump(mode="json"),
        "network": config.network.model_dump(mode="json"),
        "skills_count": len(config.skills),
        "metadata": config.metadata.model_dump(mode="json"),
    }


def _apply_config_update(config: SDKConfig, payload: dict[str, object]) -> None:
    """Apply only UI-owned fields; an empty API key intentionally preserves it."""
    if isinstance(payload.get("base_url"), str) and payload["base_url"].strip():
        config.base_url = payload["base_url"].strip()

    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key:
        config.auth.api_key = api_key

    for field in ("user_id", "app_id", "agent_id", "session_id"):
        value = payload.get(field)
        if value is not None:
            setattr(config.defaults, field, str(value).strip() or None)

    for field in ("skill_cache_dir", "skill_backup_dir"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            setattr(config.storage, field, value.strip())

    for field in ("timeout_seconds", "max_retries"):
        value = payload.get(field)
        if value is not None:
            setattr(config.network, field, int(value))

    memory = payload.get("memory")
    if isinstance(memory, dict):
        for field in (
            "search_top_k",
            "search_strategy",
            "search_rerank",
            "search_score_threshold",
            "search_filters",
            "add_mode",
            "add_default_role",
            "add_auto_skill_context",
            "get_top_k",
            "get_filters",
            "feedback_mode",
            "dreaming_mode",
        ):
            if field in memory:
                setattr(config.memory, field, memory[field])


def _skill_manager(config_manager: ConfigManager) -> tuple[SkillManager, HttpTransport]:
    config = config_manager.load_or_default()
    transport = HttpTransport(
        base_url=config.base_url,
        api_key=config.auth.api_key,
        timeout_seconds=config.network.timeout_seconds,
        max_retries=config.network.max_retries,
    )
    return SkillManager.from_config_manager(config_manager, SkillCloudClient(transport)), transport


def _memory_client(config_manager: ConfigManager) -> tuple[MemoryClient, HttpTransport, SDKConfig]:
    config = config_manager.load_or_default()
    transport = HttpTransport(
        base_url=config.base_url,
        api_key=config.auth.api_key,
        timeout_seconds=config.network.timeout_seconds,
        max_retries=config.network.max_retries,
    )
    defaults = MemoryDefaults(
        user_id=config.defaults.user_id,
        app_id=config.defaults.app_id,
        agent_id=config.defaults.agent_id,
        session_id=config.defaults.session_id,
        add_mode=config.memory.add_mode,
        add_default_role=config.memory.add_default_role,
        add_auto_skill_context=config.memory.add_auto_skill_context,
        search_top_k=config.memory.search_top_k,
        search_strategy=config.memory.search_strategy,
        search_rerank=config.memory.search_rerank,
        search_score_threshold=config.memory.search_score_threshold,
        search_filters=config.memory.search_filters,
        get_top_k=config.memory.get_top_k,
        get_filters=config.memory.get_filters,
        feedback_mode=config.memory.feedback_mode,
        dreaming_mode=config.memory.dreaming_mode,
    )
    return MemoryClient(transport, memory_defaults=defaults), transport, config


def _query_top_k(query: dict[str, list[str]]) -> int | None:
    raw = (query.get("top_k") or [""])[0].strip()
    if not raw:
        return None
    top_k = int(raw)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    return top_k


def _owned_memory_filters(filters: dict[str, object] | None, user_id: str) -> dict[str, object]:
    """Keep the local Memory page scoped to the configured user."""
    owner = {"user_id": user_id}
    if not filters:
        return owner
    return {"AND": [filters, owner]}


def _skills_payload(config_manager: ConfigManager) -> dict[str, object]:
    manager, transport = _skill_manager(config_manager)
    try:
        records = manager.list()
        pending = manager.pending_uploads()
        return {
            "skills": [record.model_dump(mode="json") for record in records],
            "pending_uploads": [item.model_dump(mode="json") for item in pending],
            "skills_count": len(records),
            "pending_count": len(pending),
        }
    finally:
        transport.close()


def _skill_detail_payload(manager: SkillManager, record: object) -> dict[str, object]:
    """Build the local Skill detail DTO used by the editable library view."""
    from ..skills.models import SkillRecord

    if not isinstance(record, SkillRecord):
        raise ValueError("Invalid Skill record.")
    versions = manager.history(record.skill_id)
    pending = [item for item in manager.pending_uploads() if item.skill_id == record.skill_id]
    try:
        local_content_hash = compute_content_hash(read_local_bundle(record.path))
    except (OSError, ValueError):
        local_content_hash = None
    return {
        "record": record.model_dump(mode="json"),
        "versions": [item.model_dump(mode="json") for item in versions],
        "pending_uploads": [item.model_dump(mode="json") for item in pending],
        "local_content_hash": local_content_hash,
        "has_local_changes": bool(local_content_hash and local_content_hash != record.content_hash),
    }


def _skill_content(
    manager: SkillManager,
    record: object,
    version_id: str | None,
) -> str:
    """Read the human-editable ``SKILL.md`` text for a local or cached version."""
    from ..skills.models import SkillRecord

    if not isinstance(record, SkillRecord):
        raise ValueError("Invalid Skill record.")
    content = manager.get_content(record.skill_id, version_id=version_id)
    return bundle_files_from_content(content)["SKILL.md"]


def run_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    config_dir: str | Path | None = None,
) -> None:
    """Serve the unified local UI and SDK-backed API until interrupted."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The local MindMemOS UI only supports loopback hosts.")
    static_dir = _static_directory()
    config_manager = ConfigManager(config_dir)
    handler = functools.partial(
        _LocalUIHandler,
        directory=str(static_dir),
        config_manager=config_manager,
    )
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"MindMemOS local UI: {url}")
    print("Press Ctrl-C to stop.")

    if open_browser:
        threading.Timer(0.15, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
