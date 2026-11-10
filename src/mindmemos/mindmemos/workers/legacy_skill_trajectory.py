"""Forward persisted Skill-related Add records outside the public request."""

import asyncio
import os
from contextlib import asynccontextmanager

import httpx
from fastapi.encoders import jsonable_encoder

from ..infra.db import get_database_clients
from ..logging import get_logger

logger = get_logger(__name__)
_INGEST_PATH = "/internal/v1/legacy-skill/trajectories"
_FIELDS = (
    "messages",
    "request_submitted_at",
    "task_completed_at",
    "event_time",
    "status",
    "user_id",
    "agent_id",
    "session_id",
    "task_id",
    "score",
    "skill_bindings",
)
_queue: asyncio.Queue[tuple[str, str]] | None = None


def enqueue_legacy_trajectory(project_id: str, add_record_id: str) -> None:
    """Schedule one saved Add record without waiting for delivery.

    Args:
        project_id: Original authenticated project namespace.
        add_record_id: Persisted trajectory identity.
    """
    if _queue is None:
        return
    try:
        _queue.put_nowait((project_id, add_record_id))
    except asyncio.QueueFull:
        logger.warning("legacy Skill delivery queue is full", project_id=project_id, add_record_id=add_record_id)


@asynccontextmanager
async def legacy_trajectory_worker():
    """Run bounded async delivery workers for the Memory API lifetime."""
    url = os.getenv("MINDMEMOS_LEGACY_SKILL_INGEST_URL")
    secret = os.getenv("MINDMEMOS_LEGACY_SKILL_INGEST_SECRET")
    if not url or not secret:
        if url or secret:
            logger.warning("legacy Skill delivery requires both an internal URL and an ingest secret")
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
        or not secret.isascii()
        or not secret.isprintable()
    ):
        logger.warning("legacy Skill delivery settings are invalid")
        yield
        return

    global _queue
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=1024)
    async with httpx.AsyncClient(
        base_url=origin,
        headers={"Authorization": f"Bearer {secret}"},
        timeout=5,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        workers = [asyncio.create_task(_consume(queue, client)) for _ in range(4)]
        _queue = queue
        try:
            yield
        finally:
            _queue = None
            try:
                await asyncio.wait_for(queue.join(), timeout=2)
            except TimeoutError:
                logger.warning("legacy Skill delivery stopped with pending Add records")
            finally:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)


async def _consume(queue: asyncio.Queue[tuple[str, str]], client: httpx.AsyncClient) -> None:
    """Process references using the currently persisted version of the Add record."""
    while True:
        project_id, add_record_id = await queue.get()
        try:
            if not await _deliver(client, project_id, add_record_id):
                logger.warning(
                    "legacy Skill trajectory remains pending", project_id=project_id, add_record_id=add_record_id
                )
        except Exception as exc:  # noqa: BLE001 - keep delivery isolated from other records
            logger.warning(
                "legacy Skill trajectory delivery failed",
                project_id=project_id,
                add_record_id=add_record_id,
                error_type=type(exc).__name__,
            )
        finally:
            queue.task_done()


async def _deliver(client: httpx.AsyncClient, project_id: str, add_record_id: str) -> bool:
    """Retry transient failures and missing version mappings without duplicating evidence."""
    for attempt in range(4):
        if attempt:
            await asyncio.sleep(2 ** (attempt - 1))
        try:
            rows = await get_database_clients().qdrant.get_add_records_by_ids(project_id, [add_record_id])
            if not rows:
                continue
            payload = rows[0].payload
            if not payload.get("skill_bindings"):
                return True
            body = {key: payload[key] for key in _FIELDS if key in payload}
            response = await client.post(
                _INGEST_PATH, json=jsonable_encoder(body | {"project_id": project_id, "add_record_id": add_record_id})
            )
        except (httpx.TransportError, OSError, TimeoutError):
            continue
        if response.status_code == 202 or response.status_code == 429 or response.status_code >= 500:
            continue
        return response.is_success
    return False
