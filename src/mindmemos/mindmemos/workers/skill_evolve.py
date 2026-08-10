"""Kafka worker for durable Skill v2 evolution operations."""

from __future__ import annotations

from ..api.services.skill_service import SKILL_EVOLUTION_TOPIC, get_skill_service
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger

TOPIC = SKILL_EVOLUTION_TOPIC
GROUP_ID = "skill-evolve-worker"

logger = get_logger(__name__)


async def handle_skill_evolve(msg: ConsumedMessage) -> None:
    """Consume a queued skill evolve task and execute the configured pipeline."""

    body = msg.json()
    project_id = body["project_id"]
    operation_id = body["operation_id"]

    logger.info(
        "processing async skill evolve",
        request_id=body.get("request_id"),
        account_id=body.get("account_id"),
        project_id=project_id,
        operation_id=operation_id,
        topic=msg.topic,
        offset=msg.offset,
    )
    result = await get_skill_service().resume_evolution(
        project_id=project_id,
        operation_id=operation_id,
    )
    logger.info(
        "async skill evolve completed",
        request_id=body.get("request_id"),
        project_id=project_id,
        operation_id=operation_id,
        status=result.status,
        selected_version_id=result.selected_version_id,
    )
