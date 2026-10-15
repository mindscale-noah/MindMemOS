import pytest
from mindmemos.config import MemoryConfig, build, build_config, validate_config
from mindmemos.errors import InvalidConfigError, MissingConfigValueError

MM = {"model": "siglip", "api_base": "http://mm/v1", "dimensions": 768}


def _validate(overrides: dict) -> None:
    validate_config(build(MemoryConfig, overrides))


def test_example_config_has_multimodal_defaults() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml")
    assert cfg.mm_embed_model_router.endpoints == []
    assert cfg.blob_store.allowed_path_roots == []
    assert cfg.database.qdrant.image_collection == "image_item_v1"
    assert cfg.algo_config.multimodal.mode == "off"


def test_mm_endpoint_validation() -> None:
    _validate({"mm_embed_model_router": {"endpoints": [MM]}})
    with pytest.raises(MissingConfigValueError):
        _validate({"mm_embed_model_router": {"endpoints": [{**MM, "dimensions": None}]}})
    with pytest.raises(InvalidConfigError):
        _validate({"mm_embed_model_router": {"endpoints": [{**MM, "protocol": "grpc"}]}})
    with pytest.raises(InvalidConfigError):
        _validate({"mm_embed_model_router": {"endpoints": [MM, {**MM, "dimensions": 512}]}})


def test_multimodal_mode_ranges_and_choices_are_validated() -> None:
    for mode in ("off", "basic", "enhanced"):
        _validate({"algo_config": {"multimodal": {"mode": mode}}})
    with pytest.raises(InvalidConfigError, match="quote it"):
        _validate({"algo_config": {"multimodal": {"mode": False}}})  # YAML 1.1 reads a bare `off` as false
    with pytest.raises(InvalidConfigError):
        _validate({"algo_config": {"multimodal": {"extraction_image_detail": "ultra"}}})
    with pytest.raises(InvalidConfigError):
        _validate({"algo_config": {"multimodal": {"agentic_max_steps": 0}}})
    with pytest.raises(InvalidConfigError):
        _validate({"algo_config": {"multimodal": {"image_text_weight": 1.5}}})
