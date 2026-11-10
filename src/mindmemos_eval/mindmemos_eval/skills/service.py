"""Open isolated Skill services for evaluations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mindmemos_skill.api import create_service
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.config.config import (
    FastAPIConfig,
    HttpAuthConfig,
    ImplementationConfig,
    PersistenceConfig,
    ServiceConfig,
    SkillConfig,
)


@asynccontextmanager
async def open_skill_service(resources: ServiceConfig, project_id: str) -> AsyncIterator[SkillService]:
    """Open and close the configured pipelines with run-local SQLite storage."""
    local = resources.model_copy(
        update={
            "persistence": PersistenceConfig(
                database=ImplementationConfig(
                    class_name="sqlite", config={"url": f"sqlite+aiosqlite:///{resources.home / 'skills.db'}"}
                )
            ),
            "repository_pipeline": ImplementationConfig(class_name="repository"),
        }
    )
    service = await create_service(
        SkillConfig(
            service=local, fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id=project_id))
        )
    )
    async with service:
        yield service
