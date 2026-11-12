"""Forward saved Add records, Skill versions, and family deletions asynchronously."""

import asyncio
import base64
import hmac
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi.encoders import jsonable_encoder

from ..components.skill.bundle import compute_content_hash, deserialize_bundle
from ..config import get_config
from ..infra.db import get_database_clients
from ..logging import get_logger
from ..typing.skill import LegacySkillDeliveryKind

logger = get_logger(__name__)
_GATEWAY_URL_ENV = "MINDMEMOS_LEGACY_SKILL_GATEWAY_URL"
_VERSION_PATH = "/internal/v1/legacy-skill/versions"
_VERSION_SCOPE = "skill:legacy-version-ingest"
_DELETION_PATH = "/internal/v1/legacy-skill/deletions"
_DELETION_SCOPE = "skill:legacy-delete"
_INGEST_PATH = "/internal/v1/legacy-skill/trajectories"
_TOKEN_SUBJECT = "mindmemos-memory"
_TOKEN_AUDIENCE = "platform-legacy-skill"
_INGEST_SCOPE = "skill:legacy-ingest"
_DELIVERY_TARGETS: dict[LegacySkillDeliveryKind, tuple[str, str, str]] = {
    "trajectory": (_INGEST_PATH, _INGEST_SCOPE, "add_record_id"),
    "version": (_VERSION_PATH, _VERSION_SCOPE, "version_id"),
    "deletion": (_DELETION_PATH, _DELETION_SCOPE, "cloud_skill_id"),
}
"""Private path, exact scope, and token-bound identity field for each operation."""
_FIELDS = (
    "messages",
    "request_submitted_at",
    "task_completed_at",
    "event_time",
    "status",
    "error",
    "user_id",
    "agent_id",
    "session_id",
    "task_id",
    "score",
    "skill_bindings",
)
_VERSION_FIELDS = (
    "version_id",
    "cloud_skill_id",
    "parent_version_id",
    "content_hash",
    "skill_name",
    "version_label",
    "status",
    "origin",
    "created_at",
)
_queue: asyncio.Queue[tuple[str, str, LegacySkillDeliveryKind]] | None = None


def legacy_skill_relay_enabled() -> bool:
    """Return whether this process has an active legacy Skill delivery worker."""
    return _queue is not None


def enqueue_legacy_trajectory(project_id: str, add_record_id: str) -> None:
    """Schedule one saved Add record without waiting for delivery.

    Args:
        project_id: Original authenticated project namespace.
        add_record_id: Persisted trajectory identity.
    """
    _enqueue(project_id, add_record_id, kind="trajectory")


def enqueue_legacy_version(project_id: str, version_id: str) -> None:
    """Schedule a registered or evolved version that has already been saved.

    Args:
        project_id: Original authenticated project namespace.
        version_id: Exact identity in the Memory Skill store.
    """
    _enqueue(project_id, version_id, kind="version")


def enqueue_legacy_skill_deletion(project_id: str, cloud_skill_id: str) -> None:
    """Schedule removal of a legacy family from the central repository.

    Args:
        project_id: Original authenticated project namespace.
        cloud_skill_id: Deleted family identity, independent of its removed versions.
    """
    _enqueue(project_id, cloud_skill_id, kind="deletion")


def _enqueue(project_id: str, record_id: str, *, kind: LegacySkillDeliveryKind) -> None:
    """Share the existing bounded delivery queue."""
    if _queue is None:
        return
    try:
        _queue.put_nowait((project_id, record_id, kind))
    except asyncio.QueueFull:
        logger.warning("legacy Skill delivery queue is full", project_id=project_id, record_id=record_id)


@asynccontextmanager
async def legacy_trajectory_worker():
    """Run bounded async delivery workers for the Memory API lifetime."""
    url = os.getenv(_GATEWAY_URL_ENV)
    if not url:
        yield
        return
    auth = get_config().auth
    secret = auth.gateway_jwt_secret
    if auth.mode != "gateway_jwt" or not secret or len(secret.encode("utf-8")) < 32:
        logger.warning("legacy Skill delivery requires cloud Memory gateway JWT authentication")
        yield
        return
    try:
        origin = httpx.URL(url)
    except httpx.InvalidURL:
        origin = None
    if (
        origin is None
        or origin.scheme not in {"http", "https"}
        or not origin.host
        or origin.userinfo
        or origin.path not in {"", "/"}
        or origin.query
        or origin.fragment
    ):
        logger.warning("legacy Skill delivery settings are invalid")
        yield
        return

    global _queue
    queue: asyncio.Queue[tuple[str, str, LegacySkillDeliveryKind]] = asyncio.Queue(maxsize=1024)
    async with httpx.AsyncClient(
        base_url=origin,
        timeout=5,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        workers = [asyncio.create_task(_consume(queue, client, secret)) for _ in range(4)]
        _queue = queue
        try:
            yield
        finally:
            _queue = None
            try:
                await asyncio.wait_for(queue.join(), timeout=2)
            except TimeoutError:
                logger.warning("legacy Skill delivery stopped with pending records")
            finally:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)


async def _consume(
    queue: asyncio.Queue[tuple[str, str, LegacySkillDeliveryKind]], client: httpx.AsyncClient, secret: str
) -> None:
    """Read persisted records and isolate delivery failures from the old API."""
    while True:
        project_id, record_id, kind = await queue.get()
        try:
            if not await _deliver(client, project_id, record_id, secret, kind=kind):
                logger.warning("legacy Skill delivery remains pending", project_id=project_id, record_id=record_id)
        except Exception as exc:  # noqa: BLE001 - keep delivery isolated from other records
            logger.warning(
                "legacy Skill delivery failed",
                project_id=project_id,
                record_id=record_id,
                error_type=type(exc).__name__,
            )
        finally:
            queue.task_done()


async def _deliver(
    client: httpx.AsyncClient,
    project_id: str,
    record_id: str,
    secret: str,
    *,
    kind: LegacySkillDeliveryKind = "trajectory",
) -> bool:
    """Retry transient failures and missing version mappings without duplicating evidence."""
    for attempt in range(4):
        if attempt:
            await asyncio.sleep(2 ** (attempt - 1))
        try:
            if kind == "version":
                status = await _deliver_version(client, project_id, record_id, secret)
            elif kind == "deletion":
                path, _, id_field = _DELIVERY_TARGETS[kind]
                token = _mint_delivery_token(project_id, record_id, secret, kind=kind)
                response = await client.post(
                    path, headers={"Authorization": f"Bearer {token}"}, json={id_field: record_id}
                )
                status = response.status_code
            else:
                status = await _deliver_trajectory(client, project_id, record_id, secret)
        except (httpx.TransportError, OSError, TimeoutError):
            continue
        if status in {202, 429} or status >= 500:
            continue
        return 200 <= status < 300
    return False


async def _deliver_trajectory(client: httpx.AsyncClient, project_id: str, add_record_id: str, secret: str) -> int:
    """Relay the latest saved Add payload."""
    rows = await get_database_clients().qdrant.get_add_records_by_ids(project_id, [add_record_id])
    if not rows:
        return 202
    payload = rows[0].payload
    if not payload.get("skill_bindings"):
        return 200
    body = {key: payload[key] for key in _FIELDS if key in payload}
    for repair in (True, False):
        response = await client.post(
            _INGEST_PATH,
            headers={"Authorization": f"Bearer {_mint_delivery_token(project_id, add_record_id, secret)}"},
            json=jsonable_encoder(body | {"add_record_id": add_record_id}),
        )
        if response.status_code != 202 or not repair:
            return response.status_code
        version_ids = {binding["version_id"] for binding in payload["skill_bindings"] if binding.get("version_id")}
        if not version_ids:
            return 202
        for version_id in sorted(version_ids):
            await _deliver_version(client, project_id, version_id, secret)
    return 202


async def _deliver_version(client: httpx.AsyncClient, project_id: str, version_id: str, secret: str) -> int:
    """Deliver a version and fetch ancestors only when the destination needs them."""
    store = get_database_clients().skill
    pending = [version_id]
    requested = {version_id}
    while pending:
        current_id = pending[-1]
        record = await store.get_version(project_id, current_id)
        if record is None:
            return 202
        payload = record.payload
        blob = await store.get_blob(project_id, payload["content_hash"])
        if blob is None:
            return 202
        files = deserialize_bundle(blob.payload["content"])
        if files.keys() != {"SKILL.md"} or compute_content_hash(files) != payload["content_hash"]:
            raise ValueError("Invalid persisted Skill bundle")
        token = _mint_delivery_token(project_id, current_id, secret, kind="version")
        response = await client.post(
            _VERSION_PATH,
            headers={"Authorization": f"Bearer {token}"},
            json=jsonable_encoder(
                {key: payload[key] for key in _VERSION_FIELDS if key in payload} | {"content": files["SKILL.md"]}
            ),
        )
        if response.status_code == 200:
            pending.pop()
            continue
        parent_id = payload.get("parent_version_id")
        if response.status_code != 202 or not parent_id or parent_id in requested:
            return response.status_code
        requested.add(parent_id)
        pending.append(parent_id)
    return 200


def _base64url(value: bytes) -> bytes:
    """Encode one unpadded JWT segment."""
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _mint_delivery_token(
    project_id: str, record_id: str, secret: str, *, kind: LegacySkillDeliveryKind = "trajectory"
) -> str:
    """Issue a short-lived Memory service token for one record and audience."""
    issued_at = int(time.time())
    _, scope, id_field = _DELIVERY_TARGETS[kind]
    claims = {
        "iss": _TOKEN_SUBJECT,
        "sub": _TOKEN_SUBJECT,
        "aud": _TOKEN_AUDIENCE,
        "scopes": [scope],
        "project_id": project_id,
        id_field: record_id,
        "iat": issued_at,
        "exp": issued_at + 60,
    }
    header = _base64url(b'{"alg":"HS256","typ":"JWT"}')
    body = _base64url(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signed = header + b"." + body
    signature = hmac.digest(secret.encode("utf-8"), signed, "sha256")
    return (signed + b"." + _base64url(signature)).decode("ascii")
