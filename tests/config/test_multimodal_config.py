import pytest
from mindmemos.config import MemoryConfig, build, build_config, validate_config
from mindmemos.config.algo.vanilla_mm import VANILLA_MM_FEATURES, VanillaMmFeatures
from mindmemos.errors import InvalidConfigError, MissingConfigValueError

MM = {"model": "siglip", "api_base": "http://mm/v1", "dimensions": 768}


def _validate(overrides: dict) -> None:
    validate_config(build(MemoryConfig, overrides))


def test_example_config_has_multimodal_defaults() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml")
    assert cfg.mm_embed_model_router.endpoints == []
    assert cfg.blob_store.allowed_path_roots == []
    assert cfg.database.qdrant.image_collection == "image_item_v1"
    assert not any(getattr(cfg.algo_config.vanilla_mm.features, name) for name in VANILLA_MM_FEATURES)


def test_feature_switch_list_matches_the_dataclass() -> None:
    assert set(VANILLA_MM_FEATURES) == set(VanillaMmFeatures.__dataclass_fields__)


def test_mm_endpoint_validation() -> None:
    _validate({"mm_embed_model_router": {"endpoints": [MM]}})
    with pytest.raises(MissingConfigValueError):
        _validate({"mm_embed_model_router": {"endpoints": [{**MM, "dimensions": None}]}})
    with pytest.raises(InvalidConfigError):
        _validate({"mm_embed_model_router": {"endpoints": [{**MM, "protocol": "grpc"}]}})
    with pytest.raises(InvalidConfigError):
        _validate({"mm_embed_model_router": {"endpoints": [MM, {**MM, "dimensions": 512}]}})


def test_unimplemented_features_and_dependencies_are_rejected(monkeypatch) -> None:
    with pytest.raises(InvalidConfigError, match="not implemented"):
        _validate({"algo_config": {"vanilla_mm": {"features": {"image_index": True}}}})

    monkeypatch.setattr("mindmemos.config.validation.VANILLA_MM_FEATURES_IMPLEMENTED", frozenset(VANILLA_MM_FEATURES))
    with pytest.raises(InvalidConfigError, match="image_index, multimodal_extraction"):
        _validate({"algo_config": {"vanilla_mm": {"features": {"visual_entity_linking": True}}}})
    _validate({"algo_config": {"vanilla_mm": {"features": {"visual_entity_linking": True, "image_index": True}}}})
