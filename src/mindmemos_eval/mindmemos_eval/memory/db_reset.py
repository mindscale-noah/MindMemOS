"""Project-scoped reset of memory stores for benchmark runs.

Before the add stage the runner clears every memory record belonging to the
current run's ``project_id`` directly from Qdrant and Neo4j, so each fresh add
starts from a clean slate. ``--no-add`` skips this reset to reuse memories from
a prior run of the same project.

This mirrors the direct-DB access already used by
:mod:`mindmemos_eval.memory.metrics` (Qdrant + ClickHouse) and reuses the same
environment-variable configuration as the server (``MINDMEMOS_QDRANT_URL``,
``MINDMEMOS_NEO4J_*``, ``MINDMEMOS_S3_*``), so eval and server share one config
source in local dev.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

logger = logging.getLogger("mindmemos_eval.memory.db_reset")

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = ""  # dev — set MINDMEMOS_NEO4J_PASSWORD in your environment
DEFAULT_NEO4J_DATABASE = "neo4j"

# Must match the server's QdrantConfig defaults (config/app.py) so eval clears the
# same collections the server writes.
DEFAULT_COLLECTIONS: tuple[str, ...] = (
    "memory_item_v1",
    "entity_item_v1",
    "source_item_v1",
    "add_record_v1",
    "schema_add_buffer_v1",
    "search_record_v1",
    "image_item_v1",
)


@dataclass(frozen=True)
class S3BlobSettings:
    """Where the server's S3-compatible blob store (MinIO) lives; images live only there."""

    endpoint: str = field(default_factory=lambda: os.getenv("MINDMEMOS_S3_ENDPOINT", "http://localhost:9100"))
    bucket: str = field(default_factory=lambda: os.getenv("MINDMEMOS_S3_BUCKET", "mindmemos-blobs"))
    access_key: str = field(default_factory=lambda: os.getenv("MINDMEMOS_S3_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("MINDMEMOS_S3_SECRET_KEY", "minioadmin"))
    region: str = field(default_factory=lambda: os.getenv("MINDMEMOS_S3_REGION", "us-east-1"))


@dataclass(frozen=True)
class ResetConfig:
    """Connection settings for direct Qdrant/Neo4j/S3 cleanup.

    Defaults read the same environment variables as the server config, so eval and
    server share one configuration source in local dev.
    """

    qdrant_url: str = field(default_factory=lambda: os.getenv("MINDMEMOS_QDRANT_URL", DEFAULT_QDRANT_URL))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("MINDMEMOS_QDRANT_API_KEY"))
    neo4j_uri: str = field(default_factory=lambda: os.getenv("MINDMEMOS_NEO4J_URI", DEFAULT_NEO4J_URI))
    neo4j_username: str = field(default_factory=lambda: os.getenv("MINDMEMOS_NEO4J_USERNAME", DEFAULT_NEO4J_USERNAME))
    neo4j_password: str = field(default_factory=lambda: os.getenv("MINDMEMOS_NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD))
    neo4j_database: str = field(default_factory=lambda: os.getenv("MINDMEMOS_NEO4J_DATABASE", DEFAULT_NEO4J_DATABASE))
    collections: tuple[str, ...] = DEFAULT_COLLECTIONS
    s3_blobs: S3BlobSettings | None = None
    """S3 settings matching the server's ``blob_store``; None skips blob cleanup.
    The runner sets it from ``--server-config`` (``resolve_s3_blob_settings``)."""


class ProjectResetError(RuntimeError):
    """Raised when a project store cannot be proven empty before evaluation."""

    def __init__(
        self,
        *,
        project_id: str,
        store: str,
        operation: str,
        resource: str,
        reason: str,
    ) -> None:
        self.project_id = project_id
        self.store = store
        self.operation = operation
        self.resource = resource
        self.reason = reason
        super().__init__(
            "project reset failed: "
            f"project_id={project_id} store={store} operation={operation} "
            f"resource={resource} reason={reason}"
        )


_QDRANT_COLLECTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("memory_collection", "memory_item_v1"),
    ("entity_collection", "entity_item_v1"),
    ("source_collection", "source_item_v1"),
    ("add_record_collection", "add_record_v1"),
    ("schema_add_buffer_collection", "schema_add_buffer_v1"),
    ("search_record_collection", "search_record_v1"),
    ("image_collection", "image_item_v1"),
)


def resolve_collections(server_config_path: str | Path | None = None) -> tuple[str, ...]:
    """Resolve the Qdrant collection names to clear for a run.

    When ``server_config_path`` points at the server's yaml config, the actual
    collection names are read from ``database.qdrant.*_collection`` so the add-stage
    cleanup clears exactly the collections the server writes - even when they are
    renamed away from the ``_v1`` defaults. Any field missing from the yaml falls
    back to its default, mirroring the server's dataclass defaults. On any
    read/parse error the function logs a warning and falls back to
    :data:`DEFAULT_COLLECTIONS` rather than blocking the benchmark run.
    """
    if server_config_path is None:
        logger.info(
            "using_default_qdrant_collections reason=no_server_config collections=%s",
            DEFAULT_COLLECTIONS,
        )
        return DEFAULT_COLLECTIONS

    path = Path(server_config_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError) as exc:
        logger.warning(
            "resolve_qdrant_collections_failed path=%s error=%s fallback=%s",
            path,
            f"{type(exc).__name__}: {exc}",
            DEFAULT_COLLECTIONS,
        )
        return DEFAULT_COLLECTIONS

    if not isinstance(raw, dict):
        logger.warning(
            "resolve_qdrant_collections_failed path=%s error=root_not_mapping fallback=%s",
            path,
            DEFAULT_COLLECTIONS,
        )
        return DEFAULT_COLLECTIONS

    database = raw.get("database")
    qdrant = database.get("qdrant") if isinstance(database, dict) else None
    if not isinstance(qdrant, dict):
        logger.warning(
            "resolve_qdrant_collections_failed path=%s error=database_qdrant_not_mapping fallback=%s",
            path,
            DEFAULT_COLLECTIONS,
        )
        return DEFAULT_COLLECTIONS

    collections = tuple(
        str(qdrant.get(field_name, default_name)) for field_name, default_name in _QDRANT_COLLECTION_FIELDS
    )
    logger.info("resolved_qdrant_collections path=%s collections=%s", path, collections)
    return collections


async def reset_project(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Physically delete every memory record scoped to ``project_id``.

    Qdrant: delete-by-filter on each configured collection. Neo4j: DETACH DELETE
    all nodes carrying this project_id. S3: every object of the project's blob
    namespace. Returns per-store deletion counts. Raises ``ValueError`` if
    project_id is empty and :class:`ProjectResetError` whenever a store cannot
    be proven empty.
    """
    if not project_id:
        raise ValueError("project_id is required to reset a project")

    counts: dict[str, int] = {}
    counts.update(await _reset_qdrant(cfg, project_id))
    counts.update(await _reset_neo4j(cfg, project_id))
    counts.update(await _reset_blobs(cfg, project_id))
    logger.info("project_reset_done project_id=%s counts=%s", project_id, counts)
    return counts


def resolve_s3_blob_settings(server_config_path: str | Path | None = None) -> S3BlobSettings:
    """The server's ``blob_store`` S3 settings; the environment defaults when they cannot be read.

    Reads the same ``blob_store.s3_*`` fields the server validates, so the reset deletes
    exactly the objects the server wrote — even when eval and server run with different
    env vars. The env-var defaults match the compose dev stack. Like the server's config
    loader, an explicit ``MINDMEMOS_S3_*`` env var wins over the yaml field; a read or
    parse failure logs a warning and falls back (this drives a destructive cleanup, so
    the fallback must never be silent).
    """

    defaults = S3BlobSettings()

    def _merged(yaml_key: str, env_name: str, default: str) -> str:
        # server precedence: env over yaml (config/app.py _merge_blob_env)
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
        yaml_value = blob_store.get(yaml_key) if isinstance(blob_store, dict) else None
        return str(yaml_value) if yaml_value else default

    blob_store: dict | None = None
    if server_config_path is not None:
        try:
            raw = yaml.safe_load(Path(server_config_path).read_text(encoding="utf-8")) or {}
        except (FileNotFoundError, yaml.YAMLError, OSError) as exc:
            logger.warning(
                "resolve_s3_blob_settings_failed path=%s error=%s fallback=env_defaults bucket=%s",
                server_config_path,
                f"{type(exc).__name__}: {exc}",
                defaults.bucket,
            )
            raw = {}
        if isinstance(raw, dict):
            maybe = raw.get("blob_store")
            if not isinstance(maybe, dict):
                logger.warning(
                    "resolve_s3_blob_settings_failed path=%s error=blob_store_not_mapping fallback=env_defaults bucket=%s",
                    server_config_path,
                    defaults.bucket,
                )
            blob_store = maybe if isinstance(maybe, dict) else None
    return S3BlobSettings(
        endpoint=_merged("s3_endpoint", "MINDMEMOS_S3_ENDPOINT", defaults.endpoint),
        bucket=_merged("s3_bucket", "MINDMEMOS_S3_BUCKET", defaults.bucket),
        access_key=_merged("s3_access_key", "MINDMEMOS_S3_ACCESS_KEY", defaults.access_key),
        secret_key=_merged("s3_secret_key", "MINDMEMOS_S3_SECRET_KEY", defaults.secret_key),
        region=_merged("s3_region", "MINDMEMOS_S3_REGION", defaults.region),
    )


async def _reset_blobs(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Delete the project's stored image objects from S3 and verify the namespace is empty."""

    if cfg.s3_blobs is None:
        logger.warning(
            "project_reset blob cleanup skipped: no S3 settings resolved, so the stored image"
            " objects of project_id=%s stay in the object store",
            project_id,
        )
        return {}
    from mindmemos.infra.blob.s3 import S3BlobStore

    try:
        store = S3BlobStore(
            endpoint=cfg.s3_blobs.endpoint,
            bucket=cfg.s3_blobs.bucket,
            access_key=cfg.s3_blobs.access_key,
            secret_key=cfg.s3_blobs.secret_key,
            region=cfg.s3_blobs.region,
        )
        count = await store.count_namespace(project_id)
        if count:
            await store.delete_namespace(project_id)
        remaining = await store.count_namespace(project_id)
    except ProjectResetError:
        raise
    except Exception as exc:
        raise ProjectResetError(
            project_id=project_id,
            store="s3",
            operation="delete",
            resource=cfg.s3_blobs.bucket,
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    if remaining:
        raise ProjectResetError(
            project_id=project_id,
            store="s3",
            operation="verify_empty",
            resource=cfg.s3_blobs.bucket,
            reason=f"{remaining} image objects remain after delete",
        )
    return {"s3:images": count}


async def _reset_qdrant(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Delete and verify all project points in every configured collection."""
    try:
        client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, trust_env=False)
    except Exception as exc:
        raise ProjectResetError(
            project_id=project_id,
            store="qdrant",
            operation="connect",
            resource="client",
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    filter_ = qmodels.Filter(
        must=[qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id))]
    )
    selector = qmodels.FilterSelector(filter=filter_)
    counts: dict[str, int] = {}
    try:
        for collection in cfg.collections:
            # Collections created on demand (e.g. the multimodal image collection) exist only once used.
            if not await client.collection_exists(collection):
                counts[f"qdrant:{collection}"] = 0
                continue
            try:
                counted = await client.count(collection_name=collection, count_filter=filter_, exact=True)
                counted_before = int(counted.count)
            except Exception as exc:
                raise ProjectResetError(
                    project_id=project_id,
                    store="qdrant",
                    operation="count_before",
                    resource=collection,
                    reason=f"{type(exc).__name__}: {exc}",
                ) from exc

            try:
                await client.delete(
                    collection_name=collection,
                    points_selector=selector,
                    wait=True,
                )
            except Exception as exc:
                raise ProjectResetError(
                    project_id=project_id,
                    store="qdrant",
                    operation="delete",
                    resource=collection,
                    reason=f"{type(exc).__name__}: {exc}",
                ) from exc

            try:
                counted = await client.count(collection_name=collection, count_filter=filter_, exact=True)
                remaining_after = int(counted.count)
            except Exception as exc:
                raise ProjectResetError(
                    project_id=project_id,
                    store="qdrant",
                    operation="count_after",
                    resource=collection,
                    reason=f"{type(exc).__name__}: {exc}",
                ) from exc

            if remaining_after != 0:
                raise ProjectResetError(
                    project_id=project_id,
                    store="qdrant",
                    operation="verify_empty",
                    resource=collection,
                    reason=f"{remaining_after} project-scoped points remain after delete",
                )
            counts[f"qdrant:{collection}"] = counted_before
    finally:
        try:
            await client.close()
        except Exception as exc:
            logger.warning(
                "qdrant_reset_close_failed project_id=%s error_type=%s",
                project_id,
                type(exc).__name__,
            )
    return counts


async def _reset_neo4j(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Delete and verify all graph nodes carrying project_id."""
    # Imported lazily so the eval package stays importable without the neo4j
    # driver installed; cleanup is the only path that needs it.
    try:
        from neo4j import AsyncGraphDatabase, RoutingControl
    except ImportError as exc:
        raise ProjectResetError(
            project_id=project_id,
            store="neo4j",
            operation="load_driver",
            resource=cfg.neo4j_database,
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        driver = AsyncGraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    except Exception as exc:
        raise ProjectResetError(
            project_id=project_id,
            store="neo4j",
            operation="connect",
            resource=cfg.neo4j_database,
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        try:
            result = await driver.execute_query(
                "MATCH (n) WHERE n.`project_id` = $project_id RETURN count(n) AS total",
                {"project_id": project_id},
                routing_=RoutingControl.READ,
                database_=cfg.neo4j_database,
            )
            total = int(result.records[0]["total"]) if result.records else 0
        except Exception as exc:
            raise ProjectResetError(
                project_id=project_id,
                store="neo4j",
                operation="count_before",
                resource=cfg.neo4j_database,
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            await driver.execute_query(
                "MATCH (n) WHERE n.`project_id` = $project_id DETACH DELETE n",
                {"project_id": project_id},
                routing_=RoutingControl.WRITE,
                database_=cfg.neo4j_database,
            )
        except Exception as exc:
            raise ProjectResetError(
                project_id=project_id,
                store="neo4j",
                operation="delete",
                resource=cfg.neo4j_database,
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            result = await driver.execute_query(
                "MATCH (n) WHERE n.`project_id` = $project_id RETURN count(n) AS total",
                {"project_id": project_id},
                routing_=RoutingControl.READ,
                database_=cfg.neo4j_database,
            )
            remaining_after = int(result.records[0]["total"]) if result.records else 0
        except Exception as exc:
            raise ProjectResetError(
                project_id=project_id,
                store="neo4j",
                operation="count_after",
                resource=cfg.neo4j_database,
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

        if remaining_after != 0:
            raise ProjectResetError(
                project_id=project_id,
                store="neo4j",
                operation="verify_empty",
                resource=cfg.neo4j_database,
                reason=f"{remaining_after} project-scoped nodes remain after delete",
            )
    finally:
        try:
            await driver.close()
        except Exception as exc:
            logger.warning(
                "neo4j_reset_close_failed project_id=%s database=%s error_type=%s",
                project_id,
                cfg.neo4j_database,
                type(exc).__name__,
            )
    return {"neo4j:nodes": total}
