"""Contracts for the public self-composing SkillApplication."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from mindmemos_skill.application import (
    AlgorithmBuildContext,
    AlgorithmResultStatus,
    SkillApplicationCapability,
)
from mindmemos_skill.application import components as component_module
from mindmemos_skill.config import SkillConfigCompiler
from mindmemos_skill.errors import (
    SkillCapabilityUnavailableError,
    SkillConflictError,
    SkillNotFoundError,
    SkillServiceClosedError,
)
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.management import PublishSkillRequest, RegisterSkillRequest
from mindmemos_skill.registry import ComponentRequirements, ComponentType, register
from mindmemos_skill.typing import (
    AgentExecutionRequest,
    AlgorithmIdentity,
    AlgorithmLog,
    AlgorithmStep,
    Rollout,
    Skill,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillOptimizationRequest,
    SkillOptimizationResult,
    SkillVersionOrigin,
    Task,
    compute_skill_content_hash,
)
from pydantic import BaseModel, ConfigDict

from mindmemos_skill import SkillApplication


class ApplicationAnalyzerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: str = "analyzed"


@register(
    type=ComponentType.ALGO,
    name="test_application_analyzer",
    config_model=ApplicationAnalyzerConfig,
    capabilities={"analyze"},
)
class ApplicationAnalyzer:
    lifecycle: ClassVar[list[str]] = []
    last_context: ClassVar[AlgorithmBuildContext | None] = None

    def __init__(self, *, config: ApplicationAnalyzerConfig, context: AlgorithmBuildContext) -> None:
        self.config = config
        type(self).last_context = context

    async def start(self) -> None:
        type(self).lifecycle.append("start")

    async def close(self) -> None:
        type(self).lifecycle.append("close")

    async def analyze(self, request: SkillAnalysisRequest) -> SkillAnalysisResult:
        return SkillAnalysisResult(summary=f"{self.config.prefix}:{request.skill.name}")


@register(
    type=ComponentType.ALGO,
    name="test_model_aware_application_analyzer",
    config_model=ApplicationAnalyzerConfig,
    capabilities={"analyze"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"analyst"})),
)
class ModelAwareApplicationAnalyzer(ApplicationAnalyzer):
    pass


@register(
    type=ComponentType.ALGO,
    name="test_application_optimizer",
    config_model=ApplicationAnalyzerConfig,
    capabilities={"optimize"},
)
class ApplicationOptimizer:
    def __init__(self, *, config: ApplicationAnalyzerConfig, context: AlgorithmBuildContext) -> None:
        self.config = config
        self.context = context

    async def optimize(self, request: SkillOptimizationRequest) -> SkillOptimizationResult:
        optimized = request.skill.model_copy(
            update={
                "blob": {
                    **request.skill.blob,
                    "SKILL.md": request.skill.blob["SKILL.md"] + "\nOptimized\n",
                }
            }
        )
        return SkillOptimizationResult(skill=optimized, changed=True)


def _config(tmp_path: Path) -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {"provider": "sqlite", "path": "state.db"},
        }
    }


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        'name: demo\ndescription: Demo description\nversion: "1.0.0"\n\nOriginal\n',
        encoding="utf-8",
    )
    return source


@pytest.mark.asyncio
async def test_from_config_builds_ready_application_and_owns_database_lifecycle(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))

    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
    published = await application.publish(
        PublishSkillRequest(
            skill_ref="demo",
            content='name: demo\nversion: "1.1.0"\n\nUpdated\n',
        )
    )

    assert registered.version_id != published.version_id
    assert application.config.local.database.options == {"path": str(tmp_path / "skill" / "state.db")}
    assert application.capabilities == frozenset(
        {"diff", "export", "list", "publish", "register", "show", "unregister"}
    )
    await application.close()

    with pytest.raises(SkillServiceClosedError, match="closed"):
        await application.list_skills()

    reopened = await SkillApplication.from_config(_config(tmp_path))
    assert (await reopened.get_skill("demo")).latest_version.version_id == published.version_id
    await reopened.close()


@pytest.mark.asyncio
async def test_unregister_atomically_removes_local_family(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
    await application.publish(
        PublishSkillRequest(
            skill_ref="demo",
            content='name: demo\nversion: "1.1.0"\n\nUpdated\n',
        )
    )

    removed = await application.unregister("demo")

    assert removed.skill_id == registered.skill_id
    assert await application.list_skills() == []
    await application.close()

    reopened = await SkillApplication.from_config(_config(tmp_path))
    with pytest.raises(SkillNotFoundError):
        await reopened.get_skill("demo")
    await reopened.close()


@pytest.mark.asyncio
async def test_from_config_accepts_precompiled_config_without_public_factory(tmp_path: Path) -> None:
    compiled = SkillConfigCompiler().compile(_config(tmp_path))

    application = await SkillApplication.from_config(compiled)

    assert application.config is compiled
    await application.close()


@pytest.mark.asyncio
async def test_management_aggregates_return_detail_and_outbox_in_one_application_result(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))

    overview = await application.get_management_overview()
    detail = await application.get_management_detail("demo")

    assert [item.skill.skill_id for item in overview.skills] == [registered.skill_id]
    assert overview.skills[0].skill.description == "Demo description"
    assert overview.skills[0].skill.latest_version_label == "1.0.0"
    assert overview.skills[0].sync_state.value == "local_only"
    assert len(overview.pending_operations) == 1
    assert detail.skill.skill_id == registered.skill_id
    assert [item.version_id for item in detail.versions] == [registered.version_id]
    assert detail.latest_version.version_id == registered.version_id
    assert detail.pending_operations == overview.pending_operations
    await application.close()


@pytest.mark.asyncio
async def test_management_only_application_reports_missing_algorithm_capability(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))

    with pytest.raises(SkillCapabilityUnavailableError, match="analysis"):
        await application.analyze({})  # type: ignore[arg-type]

    await application.close()


@pytest.mark.asyncio
async def test_application_self_composes_algorithm_runtime_and_lifecycle(tmp_path: Path) -> None:
    ApplicationAnalyzer.lifecycle = []
    config = _config(tmp_path)
    config["runtime"] = {
        "algorithms": {
            "analyzer": {
                "type": "test_application_analyzer",
                "config": {"prefix": "checked"},
            }
        }
    }
    application = await SkillApplication.from_config(config, id_generator=lambda: "analysis-log")
    request = SkillAnalysisRequest(
        skill=Skill(
            skill_id="skill-1",
            version_id="version-1",
            name="demo",
            blob={"SKILL.md": "Demo\n"},
            content_hash="hash",
            version_label="1.0.0",
            created_at=datetime(2026, 8, 5, tzinfo=UTC),
        )
    )

    result = await application.analyze(request)

    assert result.summary == "checked:demo"
    assert "analyze" in application.capabilities
    assert ApplicationAnalyzer.last_context is not None
    assert ApplicationAnalyzer.last_context.config_hash == application.config.config_hash
    assert ApplicationAnalyzer.lifecycle == ["start"]
    stored_log = await application.get_algorithm_log("analysis-log")
    assert stored_log.algorithm.name == "test_application_analyzer"
    assert stored_log.step.name == SkillApplicationCapability.ANALYZE.value
    assert stored_log.step.status == AlgorithmResultStatus.SUCCEEDED.value
    assert stored_log.step.payload["result"]["summary"] == "checked:demo"
    await application.close()
    assert ApplicationAnalyzer.lifecycle == ["start", "close"]


@pytest.mark.asyncio
async def test_algorithm_context_resolves_role_model_inside_classmethod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(component_module, "get_router", lambda config, alias: (object(), 0))
    ModelAwareApplicationAnalyzer.lifecycle = []
    ModelAwareApplicationAnalyzer.last_context = None
    config = _config(tmp_path)
    config["runtime"] = {
        "models": {"target": {"model": "openai/demo-model"}},
        "algorithms": {
            "analyzer": {
                "type": "test_model_aware_application_analyzer",
                "model_roles": {"analyst": "target"},
            }
        },
    }

    application = await SkillApplication.from_config(config)

    assert ModelAwareApplicationAnalyzer.last_context is not None
    assert set(ModelAwareApplicationAnalyzer.last_context.models) == {"analyst"}
    await application.close()


@pytest.mark.asyncio
async def test_optimization_persists_normalized_immutable_version_without_switching(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["runtime"] = {"algorithms": {"optimizer": {"type": "test_application_optimizer"}}}
    application = await SkillApplication.from_config(config)
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
    base = Skill.from_record(await application.get_version("demo", registered.version_id))

    result = await application.optimize(SkillOptimizationRequest(skill=base))

    detail = await application.get_skill("demo")
    assert result.skill.version_id != base.version_id
    assert result.skill.parent_version_ids == [base.version_id]
    assert result.skill.version_label == "1.0.1"
    assert result.skill.origin is SkillVersionOrigin.EVOLUTION
    assert result.skill.content_hash == compute_skill_content_hash(result.skill.blob)
    assert result.skill.metadata["skill_application"] == {"config_hash": application.config.config_hash}
    assert detail.latest_version.version_id == result.skill.version_id
    assert detail.skill.version_count == 2
    await application.close()


@pytest.mark.asyncio
async def test_configured_agent_execution_persists_trajectory_attempt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["runtime"] = {
        "agents": {
            "executor": {
                "type": "openclaw",
                "config": {"cli_path": "/definitely/missing/openclaw"},
            }
        }
    }
    application = await SkillApplication.from_config(config)
    request = AgentExecutionRequest(
        trajectory_id="trajectory-1",
        task=Task(task_id="task-1", instruction="Do it"),
        rollout=Rollout(rollout_id="rollout-1"),
    )

    trajectory = await application.execute("executor", request)

    assert trajectory == await application.get_trajectory("trajectory-1")
    assert trajectory.metadata["skill_application"] == {"config_hash": application.config.config_hash}
    with pytest.raises(SkillConflictError, match="immutable persistence record already exists"):
        await application.record_trajectory(trajectory)
    with pytest.raises(SkillConflictError, match="rollout attempt already exists"):
        await application.record_trajectory(trajectory.model_copy(update={"trajectory_id": "trajectory-2"}))
    await application.close()


@pytest.mark.asyncio
async def test_configured_react_trajectory_snapshots_effective_model_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubModelClient:
        async def chat(self, *args, **kwargs) -> ChatResponse:
            return ChatResponse(finish_reason="stop", content="done", model="test-model-2026")

    monkeypatch.setattr(component_module, "_build_model_client", lambda *args, **kwargs: StubModelClient())
    config = _config(tmp_path)
    config["runtime"] = {
        "models": {
            "target": {
                "model": "openai/test-model",
                "api_base": "https://models.example/v1",
                "api_key": "sk-trajectory-secret",
                "temperature": 0.2,
                "options": {
                    "num_retries": 3,
                    "timeout": 45,
                    "max_completion_tokens": 2048,
                },
            }
        },
        "agents": {
            "executor": {
                "type": "react",
                "model_ref": "target",
                "config": {"max_turns": 4},
            }
        },
    }
    application = await SkillApplication.from_config(config)
    request = AgentExecutionRequest(
        trajectory_id="trajectory-profile",
        task=Task(task_id="task-profile", instruction="Do it reproducibly"),
        rollout=Rollout(rollout_id="rollout-profile"),
        options={"temperature": 0.7},
    )

    trajectory = await application.execute("executor", request)

    expected_digest = f"sha256:{hashlib.sha256(b'sk-trajectory-secret').hexdigest()}"
    assert trajectory.agent.provider == "openai"
    assert trajectory.agent.model == "openai/test-model"
    assert trajectory.agent.base_url == "https://models.example/v1"
    assert trajectory.agent.temperature == 0.7
    assert trajectory.agent.max_retries == 3
    assert trajectory.agent.timeout_seconds == 45
    assert trajectory.agent.max_completion_tokens == 2048
    assert trajectory.agent.api_key is not None
    assert trajectory.agent.api_key.get_secret_value() == expected_digest
    assert "sk-trajectory-secret" not in repr(trajectory)
    assert trajectory == await application.get_trajectory("trajectory-profile")
    await application.close()


@pytest.mark.asyncio
async def test_algorithm_log_uses_same_application_persistence_boundary(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    log = AlgorithmLog(
        log_id="log-1",
        algorithm=AlgorithmIdentity(name="demo"),
        step=AlgorithmStep(
            component_name="analyzer",
            name="summarize",
            status="succeeded",
            payload={"count": 1},
            created_at=datetime(2026, 8, 5, tzinfo=UTC),
        ),
    )

    await application.record_algorithm_log(log)

    stored = await application.get_algorithm_log("log-1")
    assert stored.step.payload == {
        "count": 1,
        "skill_application": {"config_hash": application.config.config_hash},
    }
    await application.close()


@pytest.mark.asyncio
async def test_application_rejects_calls_from_another_event_loop(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))

    def call_from_another_loop() -> None:
        with pytest.raises(RuntimeError, match="across event loops"):
            asyncio.run(application.list_skills())

    await asyncio.to_thread(call_from_another_loop)
    await application.close()


@pytest.mark.asyncio
async def test_application_resolves_memory_skill_context_from_effective_state(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    registered = await application.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
    content = 'name: demo\nversion: "1.0.0"\n\nOriginal\n'

    contexts = await application.resolve_skill_context(
        [
            {
                "role": "assistant",
                "content": '[tool_call] read({"path":"/workspace/demo/SKILL.md"})',
            },
            {"role": "tool", "content": content},
        ],
        ensure_remote=False,
    )

    assert [context.model_dump(mode="json") for context in contexts] == [
        {
            "name": "demo",
            "content_hash": compute_skill_content_hash({"SKILL.md": content}),
            "base_version_id": registered.version_id,
            "version_label": "1.0.0",
            "usage": "injected",
        }
    ]
    await application.close()
