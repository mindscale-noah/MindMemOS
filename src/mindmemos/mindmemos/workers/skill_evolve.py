"""Kafka worker for asynchronous skill evolution tasks."""

from __future__ import annotations

from ..config import get_config
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..pipelines.skill import SKILL_EVOLVE_TOPIC, SkillEvolvePipeline
from ..provider_bindings import provider_config_context
from ..typing import MemoryRequestContext

TOPIC = SKILL_EVOLVE_TOPIC
GROUP_ID = "skill-evolve-worker"

logger = get_logger(__name__)


async def handle_skill_evolve(msg: ConsumedMessage) -> None:
    """Consume a queued skill evolve task and execute the configured pipeline."""

    body = msg.json()
    project_id = body["project_id"]
    cloud_skill_id = body["cloud_skill_id"]
    raw_context = body.get("context")
    context = (
        MemoryRequestContext.model_validate(raw_context)
        if raw_context is not None
        else MemoryRequestContext(
            request_id=str(body.get("request_id") or f"legacy-skill:{msg.partition}:{msg.offset}"),
            account_id=str(body.get("account_id") or project_id),
            project_id=project_id,
            api_key_uuid=str(body.get("api_key_uuid") or "legacy-skill-worker"),
            memory_algorithm=body.get("memory_algorithm"),
            user_id=body.get("user_id"),
        )
    )
    config_context = await provider_config_context(context)
    with config_context:
        pipeline: SkillEvolvePipeline = create_pipeline(
            type="skill_evolve",
            name=get_config().pipelines["skill_evolve"],
        )

        logger.info(
            "processing async skill evolve",
            request_id=context.request_id,
            account_id=context.account_id,
            project_id=project_id,
            cloud_skill_id=cloud_skill_id,
            topic=msg.topic,
            offset=msg.offset,
        )
        result = await pipeline.evolve(project_id=project_id, cloud_skill_id=cloud_skill_id)
        logger.info(
            "async skill evolve completed",
            request_id=context.request_id,
            project_id=project_id,
            cloud_skill_id=cloud_skill_id,
            evolved=result.evolved,
            new_version_id=result.new_version_id,
        )
