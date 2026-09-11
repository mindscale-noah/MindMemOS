"""AC-P1-001..004: ``skill_manage`` config, registry, snapshot and model-kind contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mindmemos_skill.algos.skill_manage import DefaultSkillManageAlgorithm, SkillManageConfig
from mindmemos_skill.config import SkillApplicationConfig, SkillConfigCompiler, SkillModelConfig
from mindmemos_skill.errors import SkillConfigurationError
from mindmemos_skill.infra.database import DatabaseRegistry
from mindmemos_skill.infra.vector_store import BackendRegistry, register_builtin_vector_stores
from mindmemos_skill.registry import ComponentType, get_component
from pydantic import ValidationError


def _counting_registries():
    class CountingDatabaseRegistry(DatabaseRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.opened = 0

        def create(self, config, tables):
            self.opened += 1
            raise AssertionError("database resource must not open during compile")

    class CountingVectorRegistry(BackendRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.opened = 0

        def create(self, config, tables):
            self.opened += 1
            raise AssertionError("vector resource must not open during compile")

    database_registry = CountingDatabaseRegistry()
    database_registry.register("postgres", lambda options, tables: (_ for _ in ()).throw(AssertionError()))
    vector_registry = CountingVectorRegistry()
    vector_registry.register("pgvector", lambda options, tables: (_ for _ in ()).throw(AssertionError()))
    return database_registry, vector_registry


def test_invalid_skill_manage_config_opens_zero_resources(tmp_path: Path) -> None:
    database_registry, vector_registry = _counting_registries()
    config = _skill_manage_config(tmp_path)
    config["runtime"]["algorithms"]["graphs"].update(model_roles={})

    with pytest.raises(SkillConfigurationError, match="missing model roles"):
        SkillConfigCompiler(
            database_registry=database_registry,
            vector_store_registry=vector_registry,
        ).compile(config)

    assert database_registry.opened == 0
    assert vector_registry.opened == 0


def _skill_manage_config(tmp_path: Path, **algorithm_config: object) -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {
                "provider": "postgres",
                "options": {"dsn": "postgresql://user:top-secret-password@db.example:5432/app?sslmode=require"},
            },
            "vector_store": {
                "provider": "pgvector",
                "options": {"dsn": "postgresql://user:top-secret-password@db.example:5432/app"},
            },
        },
        "runtime": {
            "models": {
                "embedder": {
                    "kind": "embedding",
                    "model": "openai/text-embedding-3-large",
                    "dimension": 3072,
                }
            },
            "algorithms": {
                "graphs": {
                    "type": "skill_manage",
                    "model_roles": {"embedding": "embedder"},
                    "config": dict(algorithm_config),
                }
            },
        },
    }


def test_skill_manage_component_is_registered_as_algo_with_embedding_role() -> None:
    component = get_component(type=ComponentType.ALGO, name="skill_manage")

    assert component.factory is DefaultSkillManageAlgorithm
    assert component.config_model is SkillManageConfig
    assert component.capabilities == frozenset({"skill_manage"})
    assert component.requirements.required_model_roles == frozenset({"embedding"})
    assert component.requirements.required_model_kinds == {"embedding": "embedding"}


def test_skill_manage_config_frozen_defaults() -> None:
    config = SkillManageConfig()

    assert config.operation_wait_timeout_seconds == 30.0
    assert config.operation_poll_interval_seconds == 0.1
    assert config.shutdown_grace_period_seconds == 30.0
    assert config.writer_lease_ttl_seconds == 60.0
    assert config.writer_lease_renew_interval_seconds == 20.0
    assert config.llm_linking is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation_poll_interval_seconds": 30.0, "operation_wait_timeout_seconds": 1.0},
        {"writer_lease_renew_interval_seconds": 30.0, "writer_lease_ttl_seconds": 30.0},
        {"operation_wait_timeout_seconds": 0.0},
    ],
)
def test_skill_manage_config_rejects_invalid_timing(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        SkillManageConfig.model_validate(overrides)


def test_valid_skill_manage_config_compiles(tmp_path: Path) -> None:
    compiled = SkillConfigCompiler().compile(_skill_manage_config(tmp_path))

    assert compiled.local.database.provider == "postgres"
    assert compiled.local.vector_store.provider == "pgvector"
    algorithm = compiled.runtime.algorithms["graphs"]
    assert algorithm.type == "skill_manage"
    assert isinstance(algorithm.config, SkillManageConfig)
    assert compiled.runtime.models["embedder"].kind == "embedding"
    assert compiled.runtime.models["embedder"].dimension == 3072
    assert len(compiled.config_hash) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["local"]["database"].update(provider="sqlite"), "postgres"),
        (lambda c: c["local"]["vector_store"].update(provider="qdrant"), "pgvector"),
        (lambda c: c["runtime"]["models"]["embedder"].update(kind="chat"), "embedding"),
        (lambda c: c["runtime"]["algorithms"]["graphs"].update(model_roles={}), "missing model roles"),
        (
            lambda c: c["runtime"]["algorithms"]["graphs"]["config"].update(llm_linking=True),
            "missing model roles",
        ),
    ],
)
def test_skill_manage_config_rejects_before_resource_open(tmp_path: Path, mutate, message: str) -> None:
    config = _skill_manage_config(tmp_path)
    mutate(config)

    with pytest.raises(SkillConfigurationError, match=message):
        SkillConfigCompiler().compile(config)


def test_embedding_model_requires_compile_time_dimension() -> None:
    with pytest.raises(ValidationError):
        SkillModelConfig(kind="embedding", model="openai/text-embedding-3-large")

    config = SkillModelConfig(kind="embedding", model="openai/text-embedding-3-large", dimension=3072)
    assert config.dimension == 3072


def test_multiple_skill_manage_instances_require_identical_embedding_dimension(tmp_path: Path) -> None:
    config = _skill_manage_config(tmp_path)
    config["runtime"]["models"]["other_embedder"] = {
        "kind": "embedding",
        "model": "openai/text-embedding-3-small",
        "dimension": 1536,
    }
    config["runtime"]["algorithms"]["second"] = {
        "type": "skill_manage",
        "model_roles": {"embedding": "other_embedder"},
        "config": {},
    }

    with pytest.raises(SkillConfigurationError, match="same embedding model"):
        SkillConfigCompiler().compile(config)


def test_llm_linking_disabled_does_not_require_chat_role(tmp_path: Path) -> None:
    config = _skill_manage_config(tmp_path, llm_linking=False)

    compiled = SkillConfigCompiler().compile(config)

    assert compiled.runtime.algorithms["graphs"].model_roles == {"embedding": "embedder"}


def test_snapshot_redacts_whole_dsn_and_is_hash_sensitive_to_non_secret_options(
    tmp_path: Path,
) -> None:
    base = _skill_manage_config(tmp_path)
    first = SkillConfigCompiler().compile(base)
    serialized = json.dumps(first.config_snapshot, sort_keys=True)

    assert "top-secret-password" not in serialized
    assert "postgresql://user" not in serialized
    assert "top-secret-password" not in repr(first)
    assert first.config_snapshot["local"]["database"]["options"]["dsn"] == "<redacted>"
    assert first.config_snapshot["local"]["vector_store"]["options"]["dsn"] == "<redacted>"
    assert first.config_snapshot["local"]["vector_store"]["provider"] == "pgvector"

    changed = _skill_manage_config(tmp_path, algorithm_version="2")
    second = SkillConfigCompiler().compile(changed)
    assert first.config_hash != second.config_hash

    only_secret_changed = _skill_manage_config(tmp_path)
    only_secret_changed["local"]["database"]["options"]["dsn"] = "postgresql://user:another-secret@db.example:5432/app"
    third = SkillConfigCompiler().compile(only_secret_changed)
    assert first.config_hash == third.config_hash


def test_compiler_uses_injected_vector_store_registry(tmp_path: Path) -> None:
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    config = _skill_manage_config(tmp_path)

    compiled = SkillConfigCompiler(vector_store_registry=registry).compile(config)

    assert compiled.local.vector_store.provider == "pgvector"
    assert "pgvector" in registry.providers


def test_unknown_vector_store_provider_fails_with_available_list(tmp_path: Path) -> None:
    config = _skill_manage_config(tmp_path)
    config["local"]["vector_store"]["provider"] = "qdrant"

    with pytest.raises(SkillConfigurationError, match="unknown vector store provider"):
        SkillConfigCompiler().compile(config)


def test_config_accepts_frozen_typed_application_config(tmp_path: Path) -> None:
    typed = SkillApplicationConfig.model_validate(_skill_manage_config(tmp_path))
    assert typed.runtime.algorithms["graphs"].type == "skill_manage"
