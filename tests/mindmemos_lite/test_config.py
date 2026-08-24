from dataclasses import dataclass, field, fields
from pathlib import Path

import pytest
import yaml
from mindmemos_lite.config import (
    DatabaseBackendConfig,
    DatabaseBackendRequirementsConfig,
    DatabaseConfig,
    MemoryConfig,
    MemoryModePipelineConfig,
    MessageChunkerConfig,
    MindMemOSConfig,
    MixedAddPipelineConfig,
    ModelEndpointConfig,
    ModelRouterConfig,
    ObservabilityConfig,
    PgVectorConfig,
    PipelineRoutingConfig,
    TextProcessingConfig,
    VanillaAddConfig,
    VanillaAddRecallConfig,
    VanillaAddSafetyGateConfig,
    VanillaAlgorithmConfig,
    VanillaSearchConfig,
    bind_config_overrides,
    build_config,
    dump_config,
    get_config,
    init_config,
    reset_config,
    validate_tree,
)
from mindmemos_lite.errors import InvalidConfigError, MissingConfigValueError
from mindmemos_lite.persistence import build_backend_config
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ReadonlyConfigError

EXAMPLE_CONFIG_PATH = "config/mindmemos_lite/dev.example.yaml"


def _config_shape(value):
    if isinstance(value, dict):
        return {key: _config_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_config_shape(item) for item in value]
    return None


def test_all_lite_config_schemas_share_the_recursive_base() -> None:
    assert issubclass(MemoryConfig, MindMemOSConfig)
    assert issubclass(ObservabilityConfig, MindMemOSConfig)
    assert issubclass(ModelRouterConfig, MindMemOSConfig)
    assert issubclass(ModelEndpointConfig, MindMemOSConfig)
    assert issubclass(DatabaseConfig, MindMemOSConfig)
    assert issubclass(DatabaseBackendConfig, MindMemOSConfig)
    assert issubclass(DatabaseBackendRequirementsConfig, MindMemOSConfig)
    assert issubclass(PgVectorConfig, MindMemOSConfig)
    assert issubclass(PipelineRoutingConfig, MindMemOSConfig)
    assert issubclass(MemoryModePipelineConfig, MindMemOSConfig)
    assert issubclass(MixedAddPipelineConfig, MindMemOSConfig)
    assert issubclass(VanillaAlgorithmConfig, MindMemOSConfig)


def test_memory_config_types_are_owned_by_component_modules() -> None:
    assert MemoryConfig.__module__ == "mindmemos_lite.config.memory"
    assert ObservabilityConfig.__module__ == "mindmemos_lite.config.observability"
    assert ModelEndpointConfig.__module__ == "mindmemos_lite.config.model.endpoint"
    assert ModelRouterConfig.__module__ == "mindmemos_lite.config.model.router"
    assert DatabaseBackendRequirementsConfig.__module__ == "mindmemos_lite.config.database.backend"
    assert DatabaseBackendConfig.__module__ == "mindmemos_lite.config.database.backend"
    assert PgVectorConfig.__module__ == "mindmemos_lite.config.database.pgvector"
    assert DatabaseConfig.__module__ == "mindmemos_lite.config.database.database"
    assert PipelineRoutingConfig.__module__ == "mindmemos_lite.config.pipelines"
    assert MemoryModePipelineConfig.__module__ == "mindmemos_lite.config.pipelines"
    assert MixedAddPipelineConfig.__module__ == "mindmemos_lite.config.pipelines"
    assert MessageChunkerConfig.__module__ == "mindmemos_lite.config.components.message_chunker"
    assert TextProcessingConfig.__module__ == "mindmemos_lite.config.components.text_processing"
    assert VanillaAddConfig.__module__ == "mindmemos_lite.config.vanilla.add"
    assert VanillaAddRecallConfig.__module__ == "mindmemos_lite.config.vanilla.add"
    assert VanillaAddSafetyGateConfig.__module__ == "mindmemos_lite.config.vanilla.add"


def test_recursive_validation_runs_children_before_parent() -> None:
    calls: list[str] = []

    @dataclass
    class ChildConfig(MindMemOSConfig):
        @classmethod
        def validate_self(cls, value, path: str) -> None:
            calls.append(path)

    @dataclass
    class ParentConfig(MindMemOSConfig):
        child: ChildConfig = field(default_factory=ChildConfig)

        @classmethod
        def validate_self(cls, value, path: str) -> None:
            calls.append(path)

    validate_tree(OmegaConf.structured(ParentConfig))

    assert calls == ["child", ""]


def test_example_config_selects_pgvector_with_typed_options() -> None:
    cfg = build_config(config_path=EXAMPLE_CONFIG_PATH)

    assert [field.name for field in fields(MemoryConfig)] == [
        "observability",
        "chat_model_router",
        "embed_model_router",
        "rerank_model_router",
        "database",
        "pipelines",
        "algo_config",
    ]
    assert cfg.observability.enabled is True
    assert cfg.observability.exporter == "sqlite"
    assert cfg.observability.sqlite_path == ".mindmemos/observability/traces.db"
    assert cfg.chat_model_router.endpoints[0].model == "openai/gpt-4.1-mini"
    assert cfg.embed_model_router.endpoints[0].dimensions == 2560
    assert cfg.rerank_model_router.endpoints[0].model == "cohere/qwen3-reranker-4b"
    assert cfg.database.backend.provider == "pgvector"
    assert cfg.database.backend.graph_enabled is True
    assert cfg.database.backend.required.max_vector_dimensions == 2560
    assert cfg.database.pgvector.schema == "mindmemos"
    assert cfg.database.pgvector.dsn
    assert OmegaConf.get_type(cfg.pipelines) is PipelineRoutingConfig
    assert OmegaConf.get_type(cfg.pipelines.modes.vanilla) is MemoryModePipelineConfig
    assert OmegaConf.get_type(cfg.pipelines.mixed_add) is MixedAddPipelineConfig
    assert cfg.pipelines.default_add_pipeline == "trajectory_add"
    assert cfg.pipelines.default_search_pipeline == "task_experience_search"
    assert cfg.pipelines.default_search_mode == "experience"
    assert list(cfg.pipelines.modes) == ["vanilla", "experience"]
    assert cfg.pipelines.modes.vanilla.add_pipeline == "vanilla_add"
    assert cfg.pipelines.modes.vanilla.search_pipeline == "vanilla_search"
    assert cfg.pipelines.modes.experience.add_pipeline == "trajectory_add"
    assert cfg.pipelines.modes.experience.search_pipeline == "task_experience_search"
    assert list(cfg.pipelines.mixed_add.modes) == ["vanilla"]
    assert OmegaConf.get_type(cfg.algo_config) is VanillaAlgorithmConfig
    assert OmegaConf.get_type(cfg.algo_config.text_processing) is TextProcessingConfig
    assert OmegaConf.get_type(cfg.algo_config.add) is VanillaAddConfig
    assert [field.name for field in fields(VanillaAddConfig)] == [
        "enable_entities",
        "chunker",
        "recall",
        "safety_gate",
        "embedding_batch_size",
    ]
    assert OmegaConf.get_type(cfg.algo_config.add.chunker) is MessageChunkerConfig
    assert OmegaConf.get_type(cfg.algo_config.add.recall) is VanillaAddRecallConfig
    assert OmegaConf.get_type(cfg.algo_config.add.safety_gate) is VanillaAddSafetyGateConfig
    assert OmegaConf.get_type(cfg.algo_config.search) is VanillaSearchConfig
    assert cfg.algo_config.add.chunker.chunk_hard_token_budget == 32000
    assert cfg.algo_config.add.embedding_batch_size == 32
    assert cfg.algo_config.add.recall.top_k == 5
    assert cfg.algo_config.search.recall_size == 20


def test_example_config_supports_isolated_pgvector_schema(monkeypatch) -> None:
    monkeypatch.setenv("PGVECTOR_SCHEMA", "eval_locomo_mixed_vanilla")

    cfg = build_config(config_path=EXAMPLE_CONFIG_PATH)

    assert cfg.database.pgvector.schema == "eval_locomo_mixed_vanilla"


def test_example_config_explicitly_matches_the_complete_memory_config_shape() -> None:
    raw = yaml.safe_load(Path(EXAMPLE_CONFIG_PATH).read_text(encoding="utf-8"))
    cfg = build_config(config_path=EXAMPLE_CONFIG_PATH)
    resolved = OmegaConf.to_container(cfg, resolve=False)

    assert _config_shape(raw) == _config_shape(resolved)


def test_database_config_maps_to_backend_neutral_contract() -> None:
    cfg = build_config(config_path=EXAMPLE_CONFIG_PATH)

    backend = build_backend_config(cfg.database)

    assert backend.provider == "pgvector"
    assert backend.options["dsn"] == cfg.database.pgvector.dsn
    assert backend.options["schema"] == "mindmemos"
    assert backend.options["hybrid_prefetch_factor"] == 4
    assert backend.options["rrf_k"] == 2
    assert backend.required.hybrid_search is True
    assert backend.required.max_vector_dimensions == 2560


def test_request_override_restores_base_and_masks_api_keys() -> None:
    try:
        init_config(config_path=EXAMPLE_CONFIG_PATH)

        with bind_config_overrides(
            project_config={
                "chat_model_router": {
                    "endpoints": [
                        {
                            "model": "project/model",
                            "api_key": "project-secret",
                            "api_base": "https://project.test/v1",
                        }
                    ]
                }
            }
        ):
            assert get_config().chat_model_router.endpoints[0].model == "project/model"
            assert dump_config()["chat_model_router"]["endpoints"][0]["api_key"] == "*****"
            assert dump_config()["database"]["pgvector"]["dsn"] == "*****"

        assert get_config().chat_model_router.endpoints[0].model == "openai/gpt-4.1-mini"
    finally:
        reset_config()


def test_request_overrides_cannot_replace_process_database_config() -> None:
    try:
        init_config(config_path=EXAMPLE_CONFIG_PATH)

        with pytest.raises(ReadonlyConfigError):
            with bind_config_overrides(
                project_config={
                    "database": {
                        "pgvector": {
                            "dsn": "postgresql://other/tenant",
                        }
                    }
                }
            ):
                pass
    finally:
        reset_config()


def test_request_overrides_cannot_replace_process_observability_config() -> None:
    try:
        init_config(config_path=EXAMPLE_CONFIG_PATH)

        with pytest.raises(ReadonlyConfigError):
            with bind_config_overrides(
                project_config={
                    "observability": {
                        "enabled": False,
                    }
                }
            ):
                pass
    finally:
        reset_config()


def test_unknown_database_backend_sections_are_rejected(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
database:
  qdrant:
    vector_size: 2560
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigKeyError, match="qdrant"):
        build_config(config_path=config_path)


def test_database_provider_must_be_registered_in_the_config_schema(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
database:
  backend:
    provider: qdrant
  pgvector:
    dsn: postgresql://localhost/mindmemos
""",
        encoding="utf-8",
    )

    with pytest.raises(InvalidConfigError, match=r"database\.backend\.provider"):
        build_config(config_path=config_path)


def test_pgvector_pool_sizes_are_validated(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
database:
  pgvector:
    dsn: postgresql://localhost/mindmemos
    min_pool_size: 5
    max_pool_size: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(InvalidConfigError, match=r"database\.pgvector\.min_pool_size"):
        build_config(config_path=config_path)


def test_embedding_endpoints_must_share_one_dimension(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
embed_model_router:
  endpoints:
    - model: embedding-a
      api_key: secret-a
      api_base: https://a.test/v1
      dimensions: 1024
    - model: embedding-b
      api_key: secret-b
      api_base: https://b.test/v1
      dimensions: 2560
database:
  pgvector:
    dsn: postgresql://localhost/mindmemos
""",
        encoding="utf-8",
    )

    with pytest.raises(InvalidConfigError, match="dimensions"):
        build_config(config_path=config_path)


def test_endpoint_config_validates_its_own_fields(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
chat_model_router:
  endpoints:
    - model: chat-model
      api_key: ""
      api_base: https://chat.test/v1
database:
  pgvector:
    dsn: postgresql://localhost/mindmemos
""",
        encoding="utf-8",
    )

    with pytest.raises(MissingConfigValueError, match=r"chat_model_router\.endpoints\[0\]\.api_key"):
        build_config(config_path=config_path)


def test_router_config_validates_its_own_fields(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
chat_model_router:
  routing_strategy: ""
database:
  pgvector:
    dsn: postgresql://localhost/mindmemos
""",
        encoding="utf-8",
    )

    with pytest.raises(MissingConfigValueError, match=r"chat_model_router\.routing_strategy"):
        build_config(config_path=config_path)
