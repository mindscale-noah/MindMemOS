"""Kafka worker for durable Skill trajectory ingest operations."""

from __future__ import annotations

from ..api.services.skill_service import SKILL_TRAJECTORY_INGEST_TOPIC
from ..infra.db import get_database_clients
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger

TOPIC = SKILL_TRAJECTORY_INGEST_TOPIC
GROUP_ID = "skill-trajectory-ingest-worker"

logger = get_logger(__name__)


async def handle_skill_trajectory_ingest(msg: ConsumedMessage) -> None:
    body = msg.json()
    project_id = str(body["project_id"])
    operation_id = str(body["operation_id"])
    result = await get_database_clients().skill.resume_trajectory_ingest(
        project_id=project_id,
        operation_id=operation_id,
    )
    logger.info(
        "Skill trajectory ingest operation completed",
        project_id=project_id,
        operation_id=operation_id,
        item_count=len(result.items),
    )
