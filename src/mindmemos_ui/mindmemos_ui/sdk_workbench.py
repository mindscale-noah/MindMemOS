"""Workbench routes backed by the configured Skill service through its SDK."""

import asyncio
import hashlib
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from difflib import unified_diff
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import (
    ActivateRequest,
    DeleteRequest,
    GraphSnapshotRequest,
    GraphSnapshotsRequest,
    HistoryRequest,
    ListRequest,
    ShowRequest,
    StructuredSkillInfo,
)
from mindmemos_sdk.skills import (
    CommitRequest as SkillCommitRequest,
)
from mindmemos_sdk.skills import (
    DiffRequest as SkillDiffRequest,
)
from mindmemos_sdk.skills import (
    RegisterRequest as SkillRegisterRequest,
)
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.envs.typing import Task
from mindmemos_skill.skill import Skill, SkillSnapshot, decode_resources, encode_resources

from .directory_picker import choose_directory
from .sdk_gateway import skill_client
from .typing import (
    ActiveRequest,
    AssembleRequest,
    CommitRequest,
    ConfirmRequest,
    DiffRequest,
    ExportRequest,
    RegisterRequest,
    VersionRef,
)


def _family(row: dict) -> dict:
    """Use the workbench's existing field names for a public Skill family."""
    return {**row, "active_version_id": row["active_skill_version_id"]}


def _version(row: dict) -> dict:
    """Use the workbench's existing field names for a public Skill version."""
    return {
        **row,
        "version_id": row["skill_version_id"],
        "status": row.get("status", "ready"),
        "origin": row.get("origin", "service"),
    }


def _snapshot(version: dict) -> SkillSnapshot:
    """Validate a portable source snapshot received from the Skill service."""
    return SkillSnapshot(
        implementation=version["implementation"],
        config_schema_version=version["config_schema_version"],
        config=version["config"],
        name=version["name"],
        description=version["description"],
        standard_format=version["standard_format"],
        content=version["content"],
        resources=version["resources"],
        content_hash=version["content_hash"],
        skill_hash=version["skill_hash"],
    )


def _content(version: dict) -> dict:
    """Display UTF-8 files and retain the names of binary resources."""
    files = {"SKILL.md": version["content"]}
    binary_files = []
    resources = decode_resources(version["resources"])
    for path, raw in resources.items():
        if b"\0" in raw:
            binary_files.append(path)
            continue
        try:
            files[path] = raw.decode("utf-8")
        except UnicodeDecodeError:
            binary_files.append(path)
    return {"files": files, "resources": list(resources), "binary_files": binary_files}


def _directory_hash(directory: Path) -> str:
    """Fingerprint a destination tree while rejecting symbolic links."""
    digest = hashlib.sha256()
    if directory.is_symlink():
        raise ValueError("Symbolic link destinations are not supported")
    if not directory.exists():
        return "absent"
    if not directory.is_dir():
        raise ValueError("Expected a directory")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic links are not supported")
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(b"D" if path.is_dir() else b"F")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _export_directory(value: str, home: Path) -> Path:
    """Resolve a dedicated destination outside the SDK workspace."""
    raw = Path(value).expanduser().absolute()
    if any(path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("Symbolic link destinations are not supported")
    target = raw.resolve()
    workspace = home.resolve()
    if target == workspace or workspace.is_relative_to(target) or target.is_relative_to(workspace):
        raise ValueError("Export outside the workspace home")
    return target


def _compare_contents(before: Skill, after: Skill, before_version: dict, after_version: dict) -> dict:
    """Compare complete assembled snapshots, including binary resources."""
    left = {"SKILL.md": before.content.encode("utf-8")}
    right = {"SKILL.md": after.content.encode("utf-8")}
    left.update(before.resources)
    right.update(after.resources)
    changes = []
    for path in sorted(left.keys() | right.keys()):
        if left.get(path) == right.get(path):
            continue
        change = "added" if path not in left else "deleted" if path not in right else "modified"
        binary = b"\0" in left.get(path, b"") or b"\0" in right.get(path, b"")
        if not binary:
            try:
                before_text = left.get(path, b"").decode("utf-8")
                after_text = right.get(path, b"").decode("utf-8")
            except UnicodeDecodeError:
                binary = True
        if binary:
            diff = None
        else:
            diff = "".join(
                unified_diff(
                    before_text.splitlines(keepends=True),
                    after_text.splitlines(keepends=True),
                    fromfile=f"before/{path}",
                    tofile=f"after/{path}",
                )
            )
        changes.append({"path": path, "change": change, "binary": binary, "diff": diff})
    return {
        "files": changes,
        "implementation_changed": (
            before_version["implementation"],
            before_version["config_schema_version"],
            before_version["config"],
        )
        != (
            after_version["implementation"],
            after_version["config_schema_version"],
            after_version["config"],
        ),
    }


def sdk_workbench_router(
    manager: ConfigManager,
    *,
    skill_loader: Callable[[SkillSnapshot], Skill] = load_runtime_skill,
) -> APIRouter:
    """Expose the workbench contract through the configured SDK connection.

    Args:
        manager: SDK settings owner for the UI process.
        skill_loader: Runtime loader used only for task assembly previews.

    Returns:
        Routes matching the existing browser workbench contract.
    """
    router = APIRouter(prefix="/api")
    plans: OrderedDict[str, tuple[float, str, object]] = OrderedDict()

    def remember(kind: str, value: object) -> str:
        """Save one bounded, short-lived reviewed operation."""
        plan_id = secrets.token_urlsafe(24)
        plans[plan_id] = (time.monotonic(), kind, value)
        while len(plans) > 128:
            plans.popitem(last=False)
        return plan_id

    def consume(plan_id: str, kind: str) -> object:
        """Use one matching reviewed operation exactly once."""
        entry = plans.pop(plan_id, None)
        if entry is None or entry[1] != kind or time.monotonic() - entry[0] > 600:
            raise HTTPException(409, "Preview expired; preview again")
        return entry[2]

    async def show_version(client, ref: VersionRef) -> dict:
        """Read an exact version through the Skill SDK."""
        detail = await client.show(ShowRequest(skill_id=ref.skill_id, skill_version_id=ref.version_id))
        return detail["version"]

    async def history(client, skill_id: str) -> list[dict]:
        """Read all version pages without assuming a page is the whole history."""
        rows = []
        while True:
            page = await client.history(HistoryRequest(skill_id=skill_id, limit=100, offset=len(rows)))
            rows.extend(_version(item) for item in page["items"])
            if not page["has_more"]:
                return rows

    @router.post("/skills/choose-directory")
    def select_skill_directory():
        try:
            selected = choose_directory(Path.home())
        except RuntimeError:
            return {"path": None, "unavailable": True}
        return {"path": str(selected) if selected else None, "unavailable": False}

    @router.get("/skills")
    async def skills():
        rows = []
        async with skill_client(manager) as client:
            cursor = None
            while True:
                page = await client.list(ListRequest(limit=100, cursor=cursor))
                rows.extend(_family(item) for item in page["items"])
                cursor = page.get("next_cursor")
                if not cursor:
                    break
            for row in rows:
                versions = await history(client, row["skill_id"])
                current = next((item for item in versions if item["version_id"] == row["active_version_id"]), None)
                latest = max(versions, key=lambda item: (item["created_at"], item["version_id"]), default=None)
                row["active_version_label"] = current["version_label"] if current else None
                row["implementation"] = current["implementation"] if current else None
                row["latest_version_id"] = latest["version_id"] if latest else None
                row["latest_version_label"] = latest["version_label"] if latest else None
                row["latest_implementation"] = latest["implementation"] if latest else None
        return rows

    @router.get("/skills/{skill_id}")
    async def show(skill_id: str):
        async with skill_client(manager) as client:
            detail = await client.show(ShowRequest(skill_id=skill_id))
        return {"skill": _family(detail["skill"]), "version": _version(detail["version"])}

    @router.get("/versions")
    async def versions(skill_id: str, source: str = "local"):
        VersionRef(source=source, skill_id=skill_id, version_id="active")
        async with skill_client(manager) as client:
            return await history(client, skill_id)

    @router.get("/content")
    async def content(source: str, skill_id: str, version_id: str):
        ref = VersionRef(source=source, skill_id=skill_id, version_id=version_id)
        async with skill_client(manager) as client:
            return _content(await show_version(client, ref))

    @router.get("/implementation")
    async def implementation(source: str, skill_id: str, version_id: str):
        ref = VersionRef(source=source, skill_id=skill_id, version_id=version_id)
        async with skill_client(manager) as client:
            version = await show_version(client, ref)
        return {"implementation": version["implementation"], "dynamic": version["implementation"] != "static"}

    @router.post("/diff")
    async def diff(body: DiffRequest):
        async with skill_client(manager) as client:
            result = await client.diff(
                SkillDiffRequest(
                    skill_id=body.before.skill_id,
                    from_skill_version_id=body.before.version_id,
                    to_skill_id=body.after.skill_id,
                    to_skill_version_id=body.after.version_id,
                )
            )
        return {
            **result,
            "diff": "\n".join(item.get("diff") or f"{item['change']}: {item['path']}" for item in result["files"]),
        }

    @router.post("/register/preview")
    def register_preview(body: RegisterRequest):
        snapshot = Skill.from_path(body.path).snapshot()
        source_name = snapshot.name
        resources = snapshot.resources
        config = manager.load_or_default()
        service_url, _ = config.resolve_skill_connection()
        return {
            "plan_id": remember("register", (body, snapshot)),
            "name": source_name,
            "path": body.path,
            "hash": snapshot.skill_hash,
            "files": sorted(["SKILL.md", *resources]),
            "service_url": service_url,
        }

    @router.post("/register/confirm")
    async def register_confirm(body: ConfirmRequest):
        request, snapshot = consume(body.plan_id, "register")
        if Skill.from_path(request.path).snapshot() != snapshot:
            raise HTTPException(409, "Source changed; preview again")
        source_name = snapshot.name
        async with skill_client(manager) as client:
            result = await client.register(
                SkillRegisterRequest(
                    alias=request.alias or source_name,
                    skill=StructuredSkillInfo(
                        content=snapshot.content,
                        resources=snapshot.resources,
                        commit_version=request.version_label,
                        commit_message=request.message,
                    ),
                )
            )
        return {"skill": {"skill_id": result["skill_id"]}, "version": {"version_id": result["skill_version_id"]}}

    @router.post("/commit")
    async def commit(body: CommitRequest):
        snapshot = Skill.from_path(body.path).snapshot()
        async with skill_client(manager) as client:
            result = await client.commit(
                SkillCommitRequest(
                    skill_id=body.skill_id,
                    skill=StructuredSkillInfo(
                        content=snapshot.content,
                        resources=snapshot.resources,
                        commit_message=body.message,
                        commit_version=body.version_label,
                        version_bump=body.version_bump,
                    ),
                    base_version_id=body.base_version_id,
                    active=True,
                )
            )
        return {"version": {"version_id": result["skill_version_id"]}}

    @router.delete("/skills/{skill_id}")
    async def remove(skill_id: str):
        async with skill_client(manager) as client:
            await client.delete(DeleteRequest(skill_id=skill_id))
        return {"ok": True}

    @router.post("/active/preview")
    async def active_preview(body: ActiveRequest):
        async with skill_client(manager) as client:
            current = await client.show(ShowRequest(skill_id=body.skill_id))
            result = await client.diff(SkillDiffRequest(skill_id=body.skill_id, to_skill_version_id=body.version_id))
        return {
            "plan_id": remember("active", (body, current["skill"]["active_skill_version_id"])),
            "diff": "\n".join(item.get("diff") or f"{item['change']}: {item['path']}" for item in result["files"]),
        }

    @router.post("/active/confirm")
    async def active_confirm(body: ConfirmRequest):
        request, expected = consume(body.plan_id, "active")
        async with skill_client(manager) as client:
            current = await client.show(ShowRequest(skill_id=request.skill_id))
            if current["skill"]["active_skill_version_id"] != expected:
                raise HTTPException(409, "Active version changed; preview again")
            result = await client.activate(
                ActivateRequest(
                    skill_id=request.skill_id, skill_version_id=request.version_id, expected_active_version_id=expected
                )
            )
        return {"active_version_id": result["active_skill_version_id"]}

    @router.post("/export/preview")
    async def export_preview(body: ExportRequest):
        target = _export_directory(body.directory, manager.config_dir)
        async with skill_client(manager) as client:
            version = await show_version(client, body.version)
        snapshot = _snapshot(version)
        fingerprint = _directory_hash(target)
        files = sorted(["SKILL.md", *snapshot.resources])
        return {
            "plan_id": remember("export", (target, fingerprint, snapshot)),
            "directory": str(target),
            "files": files,
            "overwrite": target.exists() and any(target.iterdir()),
            "diff": "\n".join(files),
        }

    @router.post("/export/confirm")
    async def export_confirm(body: ConfirmRequest):
        target, expected, snapshot = consume(body.plan_id, "export")
        target = _export_directory(str(target), manager.config_dir)
        if _directory_hash(target) != expected:
            raise HTTPException(409, "Destination changed; preview again")
        path = await asyncio.to_thread(Skill.from_snapshot(snapshot).export, target, overwrite=True)
        return {"path": str(path)}

    @router.get("/graphs")
    async def graphs():
        async with skill_client(manager) as client:
            return (await client.list_graphs())["items"]

    @router.get("/graphs/{graph_id}/snapshots")
    async def graph_snapshots(graph_id: str):
        async with skill_client(manager) as client:
            return (await client.graph_snapshots(GraphSnapshotsRequest(graph_id=graph_id)))["items"]

    @router.get("/graphs/{graph_id}/snapshots/{snapshot_id}")
    async def graph_snapshot(graph_id: str, snapshot_id: str):
        async with skill_client(manager) as client:
            return await client.graph_snapshot(GraphSnapshotRequest(graph_id=graph_id, graph_version_id=snapshot_id))

    @router.post("/assemble")
    async def assemble(body: AssembleRequest):
        if len({ref.skill_id for ref in body.versions}) != 1:
            raise HTTPException(400, "Compare versions of the same Skill")
        async with skill_client(manager) as client:
            versions = [await show_version(client, ref) for ref in body.versions]
        task = Task(id=str(uuid4()), instruction=body.instruction)
        try:
            assembled_by_version = {}
            assembled = []
            for ref, version in zip(body.versions, versions, strict=True):
                key = (ref.skill_id, ref.version_id)
                if key not in assembled_by_version:
                    runtime = skill_loader(_snapshot(version))
                    assembled_by_version[key] = await asyncio.to_thread(runtime.assemble, task.model_copy(deep=True))
                assembled.append(assembled_by_version[key])
        except NotImplementedError:
            raise HTTPException(501, "This Skill implementation does not support runtime assembly") from None
        except ValueError:
            raise HTTPException(400, "Skill source or runtime configuration is invalid") from None
        except Exception:
            raise HTTPException(
                502, "Skill assembly failed; check the server model configuration and connection"
            ) from None
        contents = [
            _content(
                {
                    **version,
                    "content": skill.content,
                    "resources": encode_resources(skill.resources),
                }
            )
            for version, skill in zip(versions, assembled, strict=True)
        ]
        result = {"contents": contents, "task": task.model_dump()}
        if len(contents) == 2:
            result["comparison"] = _compare_contents(*assembled, *versions)
        return result

    return router
