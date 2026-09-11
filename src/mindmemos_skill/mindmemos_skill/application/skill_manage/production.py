"""Production ``SkillManageService`` composition for ``SkillApplication.from_config``.

This wires the real PostgreSQL/pgvector backends for the ``skill_manage``
capability. The application remains the single lifecycle root and owns the
structured ``ScopedDatabase``; this module only bootstraps the pgvector store
and composes the domain storage adapter plus orchestration service around it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ...config import CompiledSkillApplicationConfig
from ...infra.skill_manage import (
    PgSkillGraphStorageAdapter,
    build_skill_manage_vector_tables,
)
from ...infra.vector_store import (
    BackendConfig,
    BackendRequirements,
    ScopedVectorStore,
    bootstrap_vector_store,
)
from ...service import SkillAlgorithms
from ...typing.skill_manage import EmbeddingProfile
from .loader import LocalSkillVersionLoader
from .models import SkillManageServiceSettings
from .service import SkillManageService


async def build_skill_manage_service(
    *,
    compiled: CompiledSkillApplicationConfig,
    skill_algorithms: SkillAlgorithms,
    database: Any,
    repository: Any,
    clock: Callable[[], datetime] | None = None,
    id_generator: Callable[[], str] | None = None,
    vector_store: ScopedVectorStore | None = None,
) -> SkillManageService:
    """Build a skill_manage service driven by the configured algorithms.

    The embedding dimension is taken from the compiled embedding model bound to
    the ``embedding`` role of any configured skill_manage instance; the compiler
    already enforces one shared dimension across instances.
    """

    embedding_model, algorithm_config = _skill_manage_bindings(compiled)
    vector_tables = build_skill_manage_vector_tables(embedding_model.dimension)
    owns_vector_store = vector_store is None
    if vector_store is None:
        vector_store = await bootstrap_vector_store(
            BackendConfig(
                provider=compiled.local.vector_store.provider,
                options=dict(compiled.local.vector_store.options),
                required=BackendRequirements(
                    dense_vector=True,
                    metadata_filtering=True,
                    batch_record_io=True,
                ),
            ),
            vector_tables,
        )
    else:
        # An injected store must still be schema-initialized before use.
        await vector_store.ensure_schema(vector_tables)

    # Anything failing after the store was opened here must close it again:
    # the application only owns the store once the service is handed back.
    try:
        adapter = PgSkillGraphStorageAdapter(
            database=database,
            vector_store=vector_store,
            owns_database=False,
        )
        return SkillManageService(
            algorithm_resolver=skill_algorithms.get_skill_manager,
            storage=adapter,
            embedding_profile=EmbeddingProfile(model=embedding_model.model, dimension=embedding_model.dimension),
            settings=SkillManageServiceSettings.from_algorithm_config(algorithm_config),
            clock=clock,
            id_generator=id_generator,
            version_loader=LocalSkillVersionLoader(repository),
        )
    except BaseException:
        if owns_vector_store:
            await vector_store.close()
        raise


def _skill_manage_bindings(compiled: CompiledSkillApplicationConfig) -> tuple[Any, BaseModel]:
    """The embedding model and validated config of the first configured skill_manage instance."""

    for algorithm in compiled.runtime.algorithms.values():
        if "skill_manage" in algorithm.component.capabilities:
            embedding_ref = algorithm.model_roles.get("embedding")
            if embedding_ref is not None:
                return compiled.runtime.models[embedding_ref], algorithm.config
    raise ValueError("skill_manage requires an embedding model role")


__all__ = ["build_skill_manage_service"]
