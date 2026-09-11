"""Public SkillApplication method delegation + capability gating."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from mindmemos_skill.application.skill_manage.models import (
    SkillManageBuildRequest,
    SkillManageRemoveRequest,
    SkillManageRetrieveRequest,
    SkillManageUpsertRequest,
)
from mindmemos_skill.errors import SkillCapabilityUnavailableError, SkillServiceClosedError
from mindmemos_skill.typing import SkillManageRemoveSelector, SkillVersionRef

from mindmemos_skill import SkillApplication


def test_public_skill_manage_api_exports_and_signatures() -> None:
    from mindmemos_skill.application import skill_manage

    for name in ("build_skill_index", "upsert_skill_index", "remove_skill_from_index", "retrieve_skills"):
        method = getattr(SkillApplication, name)
        assert inspect.iscoroutinefunction(method)
        assert "request" in inspect.signature(method).parameters
    operation_parameters = inspect.signature(SkillApplication.get_skill_manage_operation).parameters
    assert {"operation_id", "project_id", "graph_id"} <= operation_parameters.keys()
    for dto in (
        "SkillManageBuildRequest",
        "SkillManageUpsertRequest",
        "SkillManageRemoveRequest",
        "SkillManageRetrieveRequest",
        "SkillManageRetrieveResult",
        "SkillManageOperation",
        "SkillManageOperationResult",
    ):
        assert dto in skill_manage.__all__
        assert getattr(skill_manage, dto) is getattr(skill_manage.models, dto)


class FakeManageService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def is_ready(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def build_skill_index(self, request: object) -> object:
        self.calls.append(("build", request))
        return {"status": "committed"}

    async def upsert_skill_index(self, request: object) -> object:
        self.calls.append(("upsert", request))
        return {"status": "committed"}

    async def remove_skill_from_index(self, request: object) -> object:
        self.calls.append(("remove", request))
        return {"status": "committed"}

    async def retrieve_skills(self, request: object) -> object:
        self.calls.append(("retrieve", request))
        return {"hits": ()}

    async def get_skill_manage_operation(
        self,
        operation_id: str,
        *,
        project_id: str,
        graph_id: str,
    ) -> object:
        self.calls.append(("operation", operation_id))
        return {"operation_id": operation_id}

    async def repair_skill_index(
        self,
        *,
        project_id: str,
        graph_id: str,
        stale_candidate_age_seconds: float | None = None,
    ) -> object:
        self.calls.append(("repair", (project_id, graph_id, stale_candidate_age_seconds)))
        return {"actions": ()}

    async def maintain_skill_index(self, *, project_id: str, graph_id: str) -> object:
        self.calls.append(("maintain", (project_id, graph_id)))
        return {"removed_generation_ids": ()}


def _config(tmp_path: Path) -> dict:
    return {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}


@pytest.mark.asyncio
async def test_public_methods_delegate_to_di_service(tmp_path: Path) -> None:
    service = FakeManageService()
    application = await SkillApplication.from_config(_config(tmp_path), skill_manage_service=service)
    try:
        assert "skill_manage" in application.capabilities

        await application.build_skill_index(SkillManageBuildRequest(project_id="p", graph_id="g"))
        await application.upsert_skill_index(
            SkillManageUpsertRequest(
                project_id="p", graph_id="g", version_ref=SkillVersionRef(skill_id="s", version_id="v")
            )
        )
        await application.remove_skill_from_index(
            SkillManageRemoveRequest(
                project_id="p", graph_id="g", selector=SkillManageRemoveSelector(kind="family", skill_id="s")
            )
        )
        await application.retrieve_skills(SkillManageRetrieveRequest(project_id="p", graph_id="g", query="q"))
        await application.get_skill_manage_operation("op-1", project_id="p", graph_id="g")
        await application.repair_skill_index(project_id="p", graph_id="g", stale_candidate_age_seconds=5.0)
        await application.maintain_skill_index(project_id="p", graph_id="g")

        names = [name for name, _ in service.calls]
        assert names == ["build", "upsert", "remove", "retrieve", "operation", "repair", "maintain"]
        assert service.calls[5] == ("repair", ("p", "g", 5.0))
        assert service.calls[6] == ("maintain", ("p", "g"))
    finally:
        await application.close()


@pytest.mark.asyncio
async def test_public_methods_unavailable_without_service(tmp_path: Path) -> None:
    application = await SkillApplication.from_config(_config(tmp_path))
    try:
        assert "skill_manage" not in application.capabilities
        with pytest.raises(SkillCapabilityUnavailableError, match="skill_manage"):
            await application.build_skill_index(SkillManageBuildRequest(project_id="p", graph_id="g"))
    finally:
        await application.close()


_ALGORITHM_CONFIG = {
    "writer_lease_ttl_seconds": 90.0,
    "writer_lease_renew_interval_seconds": 30.0,
    "retired_generations_to_keep": 5,
    "retired_generation_min_age_seconds": 3600.0,
}


def _skill_manage_config(tmp_path: Path) -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {"provider": "postgres", "options": {"dsn": "postgresql://u:p@h/db"}},
            "vector_store": {"provider": "pgvector", "options": {"dsn": "postgresql://u:p@h/db"}},
        },
        "runtime": {
            "models": {"emb": {"kind": "embedding", "model": "openai/text-embedding-3-large", "dimension": 3072}},
            "algorithms": {
                "graphs": {"type": "skill_manage", "model_roles": {"embedding": "emb"}, "config": _ALGORITHM_CONFIG}
            },
        },
    }


@pytest.mark.asyncio
async def test_production_from_config_wires_skill_manage_service(tmp_path: Path, monkeypatch) -> None:
    """AC-P1-001: config declaring skill_manage bootstraps vector store + service.

    ``from_config`` must construct the ``SkillManageService`` through
    ``bootstrap_vector_store`` and expose ``skill_manage`` only once the storage
    backend is ready. Backends are mocked so this runs without PostgreSQL.
    """

    class _FakeDB:
        async def ensure_schema(self, tables: object) -> None:
            return None

        async def close(self) -> None:
            return None

        async def get_records(self, *args, **kwargs):
            return []

        async def query_records(self, *args, **kwargs):
            return ([], None)

        async def upsert_records(self, *args, **kwargs):
            return None

        async def insert_record_if_absent(self, *args, **kwargs):
            return False

        async def patch_record(self, *args, **kwargs):
            return None

        def transaction(self):
            class _Tx:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

                async def get_records(self, *args, **kwargs):
                    return []

                async def upsert_records(self, *args, **kwargs):
                    return None

                async def patch_record(self, *args, **kwargs):
                    return None

            return _Tx()

    def _async(value: object):
        class _Awaitable:
            def __await__(self):
                async def _inner():
                    return value

                return _inner().__await__()

        return _Awaitable()

    from mindmemos_skill import application as application_pkg

    monkeypatch.setattr(application_pkg.skill_application, "bootstrap_database", lambda *a, **k: _async(_FakeDB()))

    from mindmemos_skill.application.skill_manage import production as production_mod

    class _FakeVectorStore:
        async def ensure_schema(self, tables: object) -> None:
            return None

        async def close(self) -> None:
            return None

    async def fake_bootstrap(config: object, tables: object) -> object:
        return _FakeVectorStore()

    monkeypatch.setattr(production_mod, "bootstrap_vector_store", fake_bootstrap)

    application = await SkillApplication.from_config(_skill_manage_config(tmp_path))
    try:
        assert "skill_manage" in application.capabilities
        # The write-path policy comes from the algorithm config, not from defaults.
        settings = application._skill_manage_service.settings
        assert settings.writer_lease_ttl_seconds == 90.0
        assert settings.writer_lease_renew_interval_seconds == 30.0
        assert settings.retired_generations_to_keep == 5
        assert settings.retired_generation_min_age_seconds == 3600.0
        assert settings.operation_wait_timeout_seconds == 30.0
    finally:
        await application.close()
    # Once closed, the capability is withdrawn and the delegations fail closed
    # the same way every other application method does.
    assert "skill_manage" not in application.capabilities
    with pytest.raises(SkillServiceClosedError):
        await application.maintain_skill_index(project_id="p", graph_id="g")


@pytest.mark.asyncio
async def test_production_closes_the_vector_store_it_opened_when_composition_fails(tmp_path: Path, monkeypatch) -> None:
    """A store opened by ``build_skill_manage_service`` is not leaked when a later step raises."""

    from mindmemos_skill.application.skill_manage import production as production_mod

    class _FakeVectorStore:
        closed = 0

        async def ensure_schema(self, tables: object) -> None:
            return None

        async def close(self) -> None:
            self.closed += 1

    store = _FakeVectorStore()

    async def fake_bootstrap(config: object, tables: object) -> object:
        return store

    monkeypatch.setattr(production_mod, "bootstrap_vector_store", fake_bootstrap)

    def broken_settings(config: object) -> object:
        raise ValueError("timing settings rejected")

    monkeypatch.setattr(production_mod.SkillManageServiceSettings, "from_algorithm_config", broken_settings)
    from mindmemos_skill.config import SkillConfigCompiler

    compiled = SkillConfigCompiler().compile(_skill_manage_config(tmp_path))

    class _Algorithms:
        capabilities = frozenset({"skill_manage"})

        def get_skill_manager(self, name: str | None = None) -> object:
            return object()

    with pytest.raises(ValueError, match="timing settings rejected"):
        await production_mod.build_skill_manage_service(
            compiled=compiled, skill_algorithms=_Algorithms(), database=object(), repository=object()
        )
    assert store.closed == 1
