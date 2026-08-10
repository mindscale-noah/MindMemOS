"""Dependency-free local server for the SDK console and its JSON API."""

from __future__ import annotations

import functools
import http.server
import json
import secrets
import threading
import webbrowser
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from ..client import MindMemOSClient
from ..config import CompiledSDKPortalConfigV2, ConfigManager, SDKPortalConfigV2, mask_secret
from ..errors import (
    ConfigError,
    MindMemOSSDKError,
    SkillCapabilityUnavailableError,
    SkillRemoteError,
)
from ..memory import MemoryClient
from ..skills import (
    ExportSkillRequest,
    PublishLocalRequest,
    RegisterLocalRequest,
    SkillManager,
)
from .skill_service import LocalSkillUIService


class _LocalUIHandler(http.server.SimpleHTTPRequestHandler):
    """Serve packaged assets and a small local-only JSON API."""

    server_version = "MindMemOSLocalUI/0.1"

    def __init__(
        self,
        *args: object,
        config_manager: ConfigManager,
        launch_token: str,
        **kwargs: object,
    ) -> None:
        self._config_manager = config_manager
        self._launch_token = launch_token
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
        if not self._validate_mutation_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/v1/config":
            self._handle_config_update()
            return
        if path.startswith("/api/v1/skills/") and path.endswith("/content"):
            self._send_json(
                {
                    "error": "immutable_version",
                    "message": "Existing Skill versions are immutable. Publish an editor draft instead.",
                },
                status=409,
            )
            return
        self._send_json({"error": "not_found", "message": "Unknown API route."}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._validate_mutation_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/v1/skills/register":
            self._handle_skill_register()
            return
        if path.startswith("/api/v1/skills/") and path.endswith("/publish"):
            self._handle_skill_publish(path.removesuffix("/publish"))
            return
        if path.startswith("/api/v1/skills/") and path.endswith("/export"):
            self._handle_skill_export(path.removesuffix("/export"))
            return
        if path.startswith("/api/v1/skills/") and path.endswith("/sync"):
            self._handle_skill_sync(path.removesuffix("/sync"))
            return
        if path.startswith("/api/v1/skills/") and path.endswith("/evolve"):
            self._handle_skill_evolve(path.removesuffix("/evolve"))
            return
        self._send_json({"error": "not_found", "message": "Unknown API route."}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._validate_mutation_request():
            return
        path = urlsplit(self.path).path
        if path.startswith("/api/v1/skills/"):
            self._handle_skill_unregister(path)
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
        client, owner, config = _memory_client(self._config_manager)
        try:
            user_id = config.profile.identity.user_id
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
                kwargs = {"filters": _owned_memory_filters(config.profile.memory_defaults.get_filters, user_id)}
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
            owner.close()

    def _handle_skill_get(self, path: str) -> None:
        suffix = path.removeprefix("/api/v1/skills/")
        parts = [unquote(part) for part in suffix.split("/") if part]
        if not parts:
            self._send_json({"error": "not_found", "message": "Skill reference is required."}, status=404)
            return
        skill_ref = parts[0]
        manager = _skill_manager(self._config_manager)
        service = LocalSkillUIService(manager)
        try:
            if len(parts) == 1:
                self._send_json(service.detail(skill_ref).model_dump(mode="json"))
                return
            if parts[1] == "content":
                query = parse_qs(urlsplit(self.path).query)
                version_id = query.get("version_id", [None])[0]
                self._send_json(service.content(skill_ref, version_id).model_dump(mode="json"))
                return
            if parts[1] == "compare":
                query = parse_qs(urlsplit(self.path).query)
                from_version_id = (query.get("from") or [None])[0]
                to_version_id = (query.get("to") or [None])[0]
                if not from_version_id or not to_version_id:
                    raise ValueError("compare requires from and to version IDs")
                self._send_json(service.compare(skill_ref, from_version_id, to_version_id).model_dump(mode="json"))
                return
            self._send_json({"error": "not_found", "message": "Unknown Skill route."}, status=404)
        finally:
            manager.close()

    def _handle_skill_register(self) -> None:
        manager = _skill_manager(self._config_manager)
        try:
            payload = self._read_json()
            result = LocalSkillUIService(manager).register(RegisterLocalRequest.model_validate(payload))
            self._send_json(result.model_dump(mode="json"), status=201 if result.action == "created" else 200)
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "skill_register_failed", "message": str(exc)}, status=400)
        finally:
            manager.close()

    def _handle_skill_unregister(self, path: str) -> None:
        manager = _skill_manager(self._config_manager)
        try:
            skill_ref = _single_skill_ref(path)
            result = LocalSkillUIService(manager).unregister(skill_ref)
            self._send_json(result.model_dump(mode="json"))
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError) as exc:
            self._send_json({"error": "skill_unregister_failed", "message": str(exc)}, status=400)
        finally:
            manager.close()

    def _handle_skill_publish(self, path: str) -> None:
        suffix = path.removeprefix("/api/v1/skills/")
        parts = [unquote(part) for part in suffix.split("/") if part]
        if len(parts) != 1:
            self._send_json({"error": "not_found", "message": "Skill reference is required."}, status=404)
            return

        manager = _skill_manager(self._config_manager)
        try:
            payload = self._read_json()
            content = payload.get("content")
            files = payload.get("files")
            if files is not None:
                if not isinstance(files, dict) or any(
                    not isinstance(file_path, str) or not isinstance(file_content, str)
                    for file_path, file_content in files.items()
                ):
                    raise ValueError("Skill files must map paths to text.")
                if not isinstance(files.get("SKILL.md"), str) or not files["SKILL.md"].strip():
                    raise ValueError("Skill files must contain a non-empty SKILL.md.")
                content = None
            elif not isinstance(content, str) or not content.strip():
                raise ValueError("Skill content must be a non-empty string.")
            request = PublishLocalRequest(
                skill_id=parts[0],
                base_version_id=_optional_string(payload, "base_version_id"),
                content=content,
                files=files,
                version_label=_optional_string(payload, "version_label"),
                commit_message=_optional_string(payload, "commit_message"),
            )
            result, detail = LocalSkillUIService(manager).publish(request)
            self._send_json(
                {
                    "result": result.model_dump(mode="json"),
                    "detail": detail.model_dump(mode="json"),
                    "message": f"Published immutable local version {result.version_id}.",
                },
                status=201,
            )
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "skill_publish_failed", "message": str(exc)}, status=400)
        finally:
            manager.close()

    def _handle_skill_export(self, path: str) -> None:
        skill_ref = _single_skill_ref(path)
        manager = _skill_manager(self._config_manager)
        try:
            payload = self._read_json()
            target_path = payload.get("target_path")
            if not isinstance(target_path, str) or not target_path.strip():
                raise ValueError("target_path must be a non-empty string")
            result = LocalSkillUIService(manager).export(
                ExportSkillRequest(
                    skill_id=skill_ref,
                    target_path=target_path,
                    version_id=_optional_string(payload, "version_id"),
                    replace=bool(payload.get("replace", True)),
                )
            )
            self._send_json(result.model_dump(mode="json"))
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "skill_export_failed", "message": str(exc)}, status=400)
        finally:
            manager.close()

    def _handle_skill_sync(self, path: str) -> None:
        skill_ref = _single_skill_ref(path)
        manager = _skill_manager(self._config_manager)
        try:
            payload = self._read_json()
            direction = payload.get("direction", "both")
            if not isinstance(direction, str):
                raise ValueError("direction must be a string")
            detail = LocalSkillUIService(manager).sync(
                skill_ref,
                direction=direction,
            )
            self._send_json(detail.model_dump(mode="json"))
        except (SkillCapabilityUnavailableError, SkillRemoteError) as exc:
            self._send_skill_error(exc)
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "skill_sync_failed", "message": str(exc)}, status=400)
        finally:
            manager.close()

    def _handle_skill_evolve(self, path: str) -> None:
        skill_ref = _single_skill_ref(path)
        manager = _skill_manager(self._config_manager)
        try:
            payload = self._read_json()
            mode = payload.get("mode", "sync")
            if mode not in {"sync", "async"}:
                raise ValueError("mode must be 'sync' or 'async'")
            result = LocalSkillUIService(manager).evolve(
                skill_ref,
                base_version_id=_optional_string(payload, "base_version_id"),
                algorithm=_optional_string(payload, "algorithm"),
                mode=mode,
                operation_id=self.headers.get("Idempotency-Key"),
            )
            self._send_json(result.model_dump(mode="json"))
        except (ConfigError, MindMemOSSDKError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": "skill_evolve_failed", "message": str(exc)}, status=400)
        finally:
            manager.close()

    def _handle_config_update(self) -> None:
        try:
            payload = self._read_json()
            config = self._config_manager.load_or_default_portal()
            _apply_config_update(config, payload)
            self._config_manager.save_portal(config)
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

    def _send_skill_error(self, exc: SkillCapabilityUnavailableError | SkillRemoteError) -> None:
        if isinstance(exc, SkillCapabilityUnavailableError):
            self._send_json(
                {
                    "error": "skill_capability_unavailable",
                    "message": str(exc),
                    "retryable": False,
                },
                status=409,
            )
            return
        status = {
            "remote_auth_required": 401,
            "remote_unauthorized": 401,
            "remote_forbidden": 403,
            "remote_not_found": 404,
            "remote_conflict": 409,
            "remote_rate_limited": 429,
            "remote_unavailable": 503,
            "remote_server_error": 503,
            "remote_invalid_response": 502,
        }.get(exc.error_code, 502)
        payload = {
            "error": exc.error_code,
            "message": str(exc),
            "retryable": exc.retryable,
        }
        if exc.request_id is not None:
            payload["request_id"] = exc.request_id
        if exc.operation_id is not None:
            payload["operation_id"] = exc.operation_id
        self._send_json(payload, status=status)

    def _validate_mutation_request(self) -> bool:
        return self._validate_local_request()

    def _validate_local_request(self) -> bool:
        supplied_token = self.headers.get("X-MindMemOS-UI-Token")
        if supplied_token is None or not secrets.compare_digest(supplied_token, self._launch_token):
            self._send_json({"error": "forbidden", "message": "Invalid local UI launch token."}, status=403)
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            server_port = self.server.server_address[1]
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.port != server_port
            ):
                self._send_json({"error": "forbidden", "message": "Invalid local UI origin."}, status=403)
                return False
        return True


def _static_directory() -> Path:
    """Resolve the packaged static directory in a source tree or wheel."""
    return Path(files("mindmemos_sdk.ui").joinpath("static"))


def _config_payload(config_manager: ConfigManager) -> dict[str, object]:
    config = config_manager.load_or_default_portal()
    compiled = config_manager.compile_portal() if config_manager.portal_exists() else None
    if compiled is None:
        from ..config import SDKConfigCompilerV2

        compiled = SDKConfigCompilerV2().compile(config)
    profile = config.profiles[config.active_profile]
    compiled_profile = compiled.profile
    connection = profile.connections[compiled_profile.memory_connection]
    manager = SkillManager.from_config_manager(config_manager)
    try:
        skills_count = len(manager.management_overview().skills)
    finally:
        manager.close()
    return {
        "config_path": str(config_manager.portal_path),
        "profile": config.active_profile,
        "connection": compiled_profile.memory_connection,
        "base_url": connection.base_url,
        "api_key_configured": bool(connection.api_key),
        "api_key_masked": mask_secret(connection.api_key),
        "defaults": profile.identity.model_dump(mode="json"),
        "memory": profile.memory.defaults.model_dump(mode="json"),
        "storage": {
            "skill_cache_dir": str(profile.skill.application.local.artifacts_dir),
            "skill_backup_dir": "",
        },
        "network": {
            "timeout_seconds": connection.timeout_seconds,
            "max_retries": connection.max_retries,
        },
        "skills_count": skills_count,
        "metadata": {"schema_version": config.version, "active_profile": config.active_profile},
    }


def _apply_config_update(config: SDKPortalConfigV2, payload: dict[str, object]) -> None:
    """Apply only UI-owned fields; an empty API key intentionally preserves it."""
    profile = config.profiles[config.active_profile]
    connection_name = profile.memory.connection or profile.default_connection
    connection = profile.connections[connection_name]
    connection_payload = connection.model_dump(mode="python")
    if isinstance(payload.get("base_url"), str) and payload["base_url"].strip():
        connection_payload["base_url"] = payload["base_url"].strip()

    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key:
        connection_payload["api_key"] = api_key

    for field in ("user_id", "app_id", "agent_id", "session_id"):
        value = payload.get(field)
        if value is not None:
            setattr(profile.identity, field, str(value).strip() or None)

    for field in ("timeout_seconds", "max_retries"):
        value = payload.get(field)
        if value is not None:
            connection_payload[field] = int(value)

    profile.connections[connection_name] = type(connection).model_validate(connection_payload)

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
                setattr(profile.memory.defaults, field, memory[field])

    profile.identity = type(profile.identity).model_validate(profile.identity.model_dump(mode="python"))
    profile.memory.defaults = type(profile.memory.defaults).model_validate(
        profile.memory.defaults.model_dump(mode="python")
    )


def _skill_manager(config_manager: ConfigManager) -> SkillManager:
    return SkillManager.from_config_manager(config_manager)


def _memory_client(
    config_manager: ConfigManager,
) -> tuple[MemoryClient, MindMemOSClient, CompiledSDKPortalConfigV2]:
    owner = MindMemOSClient(config_manager=config_manager)
    return owner.memory, owner, config_manager.compile_portal()


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
    manager = _skill_manager(config_manager)
    try:
        service = LocalSkillUIService(manager)
        skills, pending = service.overview()
        return {
            "skills": [item.model_dump(mode="json") for item in skills],
            "outbox_operations": [item.model_dump(mode="json") for item in pending],
            "skills_count": len(skills),
            "pending_count": len(pending),
        }
    finally:
        manager.close()


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    normalized = value.strip()
    return normalized or None


def _single_skill_ref(path: str) -> str:
    suffix = path.removeprefix("/api/v1/skills/")
    parts = [unquote(part) for part in suffix.split("/") if part]
    if len(parts) != 1:
        raise ValueError("Skill reference is required")
    return parts[0]


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
    launch_token = secrets.token_urlsafe(32)
    handler = functools.partial(
        _LocalUIHandler,
        directory=str(static_dir),
        config_manager=config_manager,
        launch_token=launch_token,
    )
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}/?token={launch_token}"
    print(f"MindMemOS local UI: {url}")
    print("Press Ctrl-C to stop.")

    if open_browser:
        threading.Timer(0.15, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
