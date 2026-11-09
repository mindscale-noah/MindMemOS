"""Service connections used by isolated Skill evolution evaluations."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.skills import typing as sdk_types
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.api import create_service
from mindmemos_skill.api import typing as api_types
from mindmemos_skill.common.skill_versions import snapshot_from_version
from mindmemos_skill.config.config import (
    FastAPIConfig,
    HttpAuthConfig,
    ImplementationConfig,
    PersistenceConfig,
    ServiceConfig,
    SkillConfig,
)
from mindmemos_skill.config.loader import read_skill_config
from mindmemos_skill.skill import Skill, SkillContent, decode_resources

from .config import SkillEvolveConfig


def _merge(left, right):
    """Merge nested profile settings without mutating the source configuration."""
    result = dict(left)
    for key, value in right.items():
        result[key] = (
            _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        )
    return result


class EvolutionService:
    """Normalize in-process and SDK operations at the evaluation boundary."""

    def __init__(self, config, project_id, *, service=None, client=None):
        """Retain one explicitly selected service connection."""
        self.config = config
        self.project_id = project_id
        self.service = service
        self.client = client

    async def register(self, skill: Skill, alias: str) -> dict:
        """Register exact source bytes and return family/version identities.

        Args:
            skill: Frozen initial source and resources.
            alias: Run-scoped unique family alias.

        Returns:
            Family ID and immutable version ID.
        """
        source = skill.snapshot()
        fields = dict(
            content=source.content,
            resources=source.resources,
            name=source.name,
            description=source.description,
            standard_format=source.standard_format,
        )
        if self.service is not None:
            result = await self.service.register_skill(
                api_types.RegisterRequest(
                    project_id=self.project_id, alias=alias, skill=api_types.StructuredSkillInfo(**fields)
                )
            )
            return result.model_dump(mode="json")
        return await self.client.register(
            sdk_types.RegisterRequest(alias=alias, skill=sdk_types.StructuredSkillInfo(**fields))
        )

    async def report(self, version_id: str, messages: list[dict], session_id: str) -> None:
        """Report public messages without verifier scores.

        Args:
            version_id: Exact version used during collection.
            messages: Sanitized standard environment messages.
            session_id: Stable local slot identity for audit.
        """
        fields = dict(
            user_id="mindmemos-eval",
            agent_id="skill-evolve",
            skill_version_id=version_id,
            messages=messages,
            session_id=session_id,
        )
        if self.service is not None:
            await self.service.report_trajectory(
                api_types.ReportTrajectoryRequest(project_id=self.project_id, **fields)
            )
        else:
            await self.client.report_trajectory(sdk_types.TrajectoryReportRequest(**fields))

    async def _sdk_evolve(self, **fields):
        """Wait for one owned asynchronous evolution job."""
        accepted = await self.client.evolve(
            sdk_types.EvolveRequest(auto_accept=False, pipeline=self.config.pipeline, **fields)
        )
        while True:
            task = await self.client.show_task(sdk_types.TaskShowRequest(task_id=accepted["task_id"]))
            if task["status"] == "succeeded":
                return task["result"]
            if task["status"] in {"failed", "cancelled"}:
                raise RuntimeError(f"Evolution task {accepted['task_id']}: {task}")
            await asyncio.sleep(self.config.poll_interval)

    async def evolve(self, registrations: list[dict]) -> list[dict]:
        """Evolve a collection or each single-Skill member without activation.

        Args:
            registrations: Registered initial family and version identities.

        Returns:
            Complete published member records including unchanged members.
        """
        if self.config.pipeline == "offline":
            versions = [item["skill_version_id"] for item in registrations]
            if self.service is not None:
                result = await self.service.run_evolution(
                    api_types.CollectionEvolutionRequest(
                        project_id=self.project_id,
                        pipeline=self.config.pipeline,
                        skill_version_ids=versions,
                        auto_accept=False,
                    )
                )
                return result.model_dump(mode="json")["members"]
            return (await self._sdk_evolve(skill_version_ids=versions))["members"]
        members = []
        for item in registrations:
            if self.service is not None:
                result = await self.service.run_evolution(
                    api_types.SkillPipelineRequest(
                        project_id=self.project_id,
                        pipeline=self.config.pipeline,
                        skill_ref=item["skill_id"],
                        base_version_id=item["skill_version_id"],
                        auto_accept=False,
                    )
                )
                version = await self.service.get_skill_version(result.version_id, project_id=self.project_id)
                snapshot = snapshot_from_version(version)
                version_id = result.version_id
            else:
                result = await self._sdk_evolve(skill_version_id=item["skill_version_id"])
                version_id = result["version_id"]
                version = (await self.client.show(sdk_types.ShowRequest(skill_version_id=version_id)))["version"]
                snapshot = Skill(
                    SkillContent(
                        content=version["content"],
                        resources=decode_resources(version["resources"]),
                        name=version["name"],
                        description=version["description"],
                        standard_format=version["standard_format"],
                    )
                ).snapshot()
            members.append(
                dict(
                    skill_id=item["skill_id"],
                    version_id=version_id,
                    base_version_id=item["skill_version_id"],
                    action="updated",
                    snapshot=snapshot.model_dump(mode="json"),
                )
            )
        return members


@asynccontextmanager
async def evolution_connection(config: SkillEvolveConfig, directory: Path, project_id: str):
    """Open isolated storage or the configured SDK connection.

    Args:
        config: Validated evaluation connection and profile settings.
        directory: Run output directory containing private SQLite state.
        project_id: Stable local project identity for restart support.

    Yields:
        Normalized evolution operations.
    """
    if config.connection == "sdk":
        settings = ConfigManager(config.sdk_home).load()
        url, key = settings.resolve_skill_connection()
        async with AsyncHttpTransport(
            base_url=url,
            api_key=key,
            timeout_seconds=settings.network.timeout_seconds,
            max_retries=0,
            require_api_key=False,
        ) as transport:
            yield EvolutionService(config, project_id, client=AsyncSkillClient(transport))
        return
    source = read_skill_config(config.service_config).service
    profile = source.evolution_pipelines[config.pipeline]
    payload = profile.model_dump(mode="python")
    payload["config"] = _merge(payload["config"], config.pipeline_overrides)
    payload["config"]["work_dir"] = str(directory / "evolution_work")
    state = directory / "state"
    resources = ServiceConfig(
        home=state,
        agents=source.agents,
        persistence=PersistenceConfig(
            database=ImplementationConfig(
                class_name="sqlite", config={"url": f"sqlite+aiosqlite:///{state / 'skills.db'}"}
            )
        ),
        repository_pipeline=ImplementationConfig(class_name="repository"),
        evolution_pipelines={config.pipeline: ImplementationConfig.model_validate(payload)},
    )
    async with await create_service(
        SkillConfig(
            service=resources, fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id=project_id))
        )
    ) as service:
        yield EvolutionService(config, project_id, service=service)
