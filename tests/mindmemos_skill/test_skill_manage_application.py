"""AC-P1-005 (and P1 dispatch/lifecycle): typed clients, dispatch, readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mindmemos_skill.application import SkillApplication
from mindmemos_skill.application import components as component_module
from mindmemos_skill.application.components import RuntimeComponents
from mindmemos_skill.application.enums import SkillApplicationCapability
from mindmemos_skill.application.skill_manage import SkillManageService
from mindmemos_skill.config import SkillConfigCompiler
from mindmemos_skill.errors import SkillCapabilityUnavailableError
from mindmemos_skill.registry import ComponentRequirements, ComponentType, register
from mindmemos_skill.service import SkillAlgorithms, SkillManageAlgorithm
from mindmemos_skill.typing import (
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
    SkillManageEmbedOutput,
    SkillManageMutationPlan,
    SkillManageRemoveInput,
    SkillManageRetrieveInput,
    SkillManageRetrieveOutput,
    SkillManageUpsertInput,
)
from pydantic import BaseModel, ConfigDict


class DemoManageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@register(
    type=ComponentType.ALGO,
    name="test_dual_role_manage",
    config_model=DemoManageConfig,
    capabilities={"skill_manage"},
    requirements=ComponentRequirements(
        required_model_roles=frozenset({"embedding", "chat"}),
        required_model_kinds={"embedding": "embedding", "chat": "chat"},
    ),
)
class DualRoleManage:
    last_context: Any = None

    def __init__(self, *, config: DemoManageConfig, context: Any) -> None:
        type(self).last_context = context

    async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
        raise NotImplementedError

    async def plan_upsert(self, request: SkillManageUpsertInput) -> SkillManageMutationPlan:
        raise NotImplementedError

    async def plan_remove(self, request: SkillManageRemoveInput) -> SkillManageMutationPlan:
        raise NotImplementedError

    async def embed(self, request: SkillManageEmbedInput) -> SkillManageEmbedOutput:
        raise NotImplementedError

    async def retrieve(self, request: SkillManageRetrieveInput) -> SkillManageRetrieveOutput:
        raise NotImplementedError


class FakeSkillManager:
    def __init__(self) -> None:
        self.retrieved: list[SkillManageRetrieveInput] = []

    async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
        raise NotImplementedError

    async def plan_upsert(self, request: SkillManageUpsertInput) -> SkillManageMutationPlan:
        raise NotImplementedError

    async def plan_remove(self, request: SkillManageRemoveInput) -> SkillManageMutationPlan:
        raise NotImplementedError

    async def embed(self, request: SkillManageEmbedInput) -> SkillManageEmbedOutput:
        raise NotImplementedError

    async def retrieve(self, request: SkillManageRetrieveInput) -> SkillManageRetrieveOutput:
        self.retrieved.append(request)
        raise NotImplementedError


def _typed_context_config(tmp_path: Path) -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {"provider": "postgres", "options": {"dsn": "postgresql://u:p@h/db"}},
            "vector_store": {"provider": "pgvector"},
        },
        "runtime": {
            "models": {
                "chat": {"kind": "chat", "model": "openai/gpt-5-mini"},
                "embed": {"kind": "embedding", "model": "openai/text-embedding-3-large", "dimension": 3072},
            },
            "algorithms": {
                "graphs": {
                    "type": "test_dual_role_manage",
                    "model_roles": {"chat": "chat", "embedding": "embed"},
                    "config": {},
                }
            },
        },
    }


def test_compose_runtime_builds_typed_chat_and_embedding_clients(tmp_path: Path, monkeypatch) -> None:
    class FakeChat:
        def __init__(self) -> None:
            self.kind = "chat"

    class FakeEmbed:
        def __init__(self) -> None:
            self.kind = "embedding"

    built: dict[str, str] = {}

    def fake_build(name: str, model, *, call_sink=None):
        if model.kind == "embedding":
            built[name] = "embedding"
            return FakeEmbed()
        built[name] = "chat"
        return FakeChat()

    monkeypatch.setattr(component_module, "_build_model_client", fake_build)
    compiled = SkillConfigCompiler().compile(_typed_context_config(tmp_path))

    component_module.compose_runtime(compiled)

    assert built == {"chat": "chat", "embed": "embedding"}
    context = DualRoleManage.last_context
    assert set(context.chat_models) == {"chat"}
    assert set(context.embedding_models) == {"embedding"}
    assert set(context.models) == {"chat", "embedding"}
    assert isinstance(context.chat_models["chat"], FakeChat)
    assert isinstance(context.embedding_models["embedding"], FakeEmbed)


def test_dispatch_single_and_multiple_skill_managers() -> None:
    single = SkillAlgorithms(skill_manager=FakeSkillManager())
    assert "skill_manage" in single.capabilities
    assert single.algorithm_names("skill_manage") == ("default",)
    assert single.resolve_name("skill_manage") == "default"
    assert single.resolve_name("skill_manage", "default") == "default"

    multi = SkillAlgorithms(
        skill_managers={"alpha": FakeSkillManager(), "beta": FakeSkillManager()},
    )
    assert multi.resolve_name("skill_manage", "alpha") == "alpha"
    with pytest.raises(SkillCapabilityUnavailableError, match="algorithm name"):
        multi.resolve_name("skill_manage")
    with pytest.raises(SkillCapabilityUnavailableError, match="not configured"):
        multi.resolve_name("skill_manage", "missing")


def test_skill_manage_protocol_runtime_checkable() -> None:
    manager = FakeSkillManager()
    assert isinstance(manager, SkillManageAlgorithm)

    class MissingMethods:
        pass

    assert not isinstance(MissingMethods(), SkillManageAlgorithm)


class FakeStoragePort:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.started = 0
        self.closed = 0

    async def is_ready(self) -> bool:
        return self.ready

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1


def _management_config(tmp_path: Path) -> dict:
    return {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}


@pytest.mark.asyncio
async def test_skill_manage_service_lifecycle_is_idempotent_and_close_on_exit(tmp_path: Path) -> None:
    storage = FakeStoragePort()
    service = SkillManageService(algorithm=FakeSkillManager(), storage=storage)

    await service.start()
    await service.start()
    await service.close()
    await service.close()

    assert storage.started == 1
    assert storage.closed == 1


@pytest.mark.asyncio
async def test_production_from_config_does_not_expose_skill_manage_in_p1(tmp_path: Path) -> None:
    # A config declaring skill_manage must still not claim the capability before
    # the P3 storage adapter reports readiness. P1 only proves the skeleton
    # composes through explicit dependency injection.
    assert SkillApplicationCapability.SKILL_MANAGE.value == "skill_manage"

    # The management-only application exposes no algorithm capabilities.
    application = await SkillApplication.from_config(_management_config(tmp_path))
    assert "skill_manage" not in application.capabilities
    await application.close()


@pytest.mark.asyncio
async def test_runtime_components_close_resources_in_reverse_order_once() -> None:
    events: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def start(self) -> None:
            events.append(f"start:{self.name}")

        async def close(self) -> None:
            events.append(f"close:{self.name}")

    components = RuntimeComponents(resources=[Resource("a"), Resource("b"), Resource("c")])
    await components.start()
    await components.close()
    await components.close()

    assert events == ["start:a", "start:b", "start:c", "close:c", "close:b", "close:a"]
