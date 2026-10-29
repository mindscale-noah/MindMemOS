import pytest
from mindmemos.config import MemoryConfig, build, build_config, validate_config
from mindmemos.errors import InvalidConfigError, MissingConfigValueError

MM = {"model": "siglip", "api_base": "http://mm/v1", "dimensions": 768}
S3 = {
    "backend": "s3",
    "s3_endpoint": "http://localhost:9100",
    "s3_bucket": "mindmemos-blobs",
    "s3_access_key": "minioadmin",
    "s3_secret_key": "minioadmin",
}


def _validate(overrides: dict) -> None:
    validate_config(build(MemoryConfig, overrides))


def test_example_config_has_multimodal_defaults() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml")
    assert cfg.mm_embed_model_router.endpoints == []
    assert cfg.blob_store.allowed_path_roots == []
    assert cfg.blob_store.s3_presign_endpoint is None
    assert cfg.blob_store.s3_presign_expiry_seconds == 3600
    assert cfg.database.qdrant.image_collection == "image_item_v1"
    assert cfg.algo_config.multimodal.mode == "off"


def test_mm_endpoint_validation() -> None:
    _validate({"mm_embed_model_router": {"endpoints": [MM]}})
    with pytest.raises(MissingConfigValueError):
        _validate({"mm_embed_model_router": {"endpoints": [{**MM, "dimensions": None}]}})
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


def test_blob_store_s3_backend_validation() -> None:
    _validate({"blob_store": S3})  # local_root not required for the s3 backend
    for missing in ("s3_endpoint", "s3_bucket", "s3_access_key", "s3_secret_key"):
        partial = {key: value for key, value in S3.items() if key != missing}
        with pytest.raises(MissingConfigValueError, match=f"blob_store.{missing}"):
            _validate({"blob_store": partial})
    with pytest.raises(InvalidConfigError, match="http://"):
        _validate({"blob_store": {**S3, "s3_endpoint": "localhost:9100"}})
    with pytest.raises(MissingConfigValueError, match="blob_store.s3_region"):
        # an explicit empty region builds a region-less client whose presigning hits the network
        _validate({"blob_store": {**S3, "s3_region": ""}})
    with pytest.raises(InvalidConfigError):
        _validate({"blob_store": {"backend": "gcs"}})


def test_blob_store_presign_validation() -> None:
    _validate({"blob_store": {**S3, "s3_presign_endpoint": "https://blobs.example.com", "s3_presign_expiry_seconds": 300}})
    _validate({"blob_store": {**S3, "s3_presign_expiry_seconds": 604800}})  # exactly 7 days, the S3 maximum
    _validate({"blob_store": S3})  # both fields optional; defaults apply

    with pytest.raises(InvalidConfigError, match="s3_presign_endpoint"):
        _validate({"blob_store": {**S3, "s3_presign_endpoint": "blobs.example.com"}})  # no scheme
    for bad_expiry in (0, -1, 604801):
        with pytest.raises(InvalidConfigError, match="s3_presign_expiry_seconds"):
            _validate({"blob_store": {**S3, "s3_presign_expiry_seconds": bad_expiry}})


def test_blob_store_presign_env_overlay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_S3_PRESIGN_ENDPOINT", "https://cdn.example.com")
    monkeypatch.setenv("MINDMEMOS_S3_PRESIGN_EXPIRY_SECONDS", "7200")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
blob_store:
  backend: s3
  s3_endpoint: http://minio.example:9100
  s3_bucket: blobs
  s3_access_key: access
  s3_secret_key: secret
  s3_presign_endpoint: https://yaml.example.com
  s3_presign_expiry_seconds: 300
""",
        encoding="utf-8",
    )

    cfg = build_config(config_path=config_path)

    assert cfg.blob_store.s3_presign_endpoint == "https://cdn.example.com"
    assert cfg.blob_store.s3_presign_expiry_seconds == 7200


def test_blob_store_s3_env_overlay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_S3_ENDPOINT", "http://minio.example:9100")
    monkeypatch.setenv("MINDMEMOS_S3_SECRET_KEY", "topsecret")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
blob_store:
  backend: s3
  s3_endpoint: http://yaml.example:9100
  s3_bucket: yaml-blobs
  s3_access_key: yaml-access
  s3_secret_key: yaml-secret
""",
        encoding="utf-8",
    )

    cfg = build_config(config_path=config_path)

    assert cfg.blob_store.backend == "s3"
    assert cfg.blob_store.s3_endpoint == "http://minio.example:9100"
    assert cfg.blob_store.s3_bucket == "yaml-blobs"
    assert cfg.blob_store.s3_access_key == "yaml-access"
    assert cfg.blob_store.s3_secret_key == "topsecret"
