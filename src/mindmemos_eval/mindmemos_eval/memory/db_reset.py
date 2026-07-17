"""Project-scoped reset of memory stores for benchmark runs.

Before the add stage the runner clears every memory record belonging to the
current run's ``project_id`` directly from Qdrant and Neo4j, so each fresh add
starts from a clean slate. ``--no-add`` skips this reset to reuse memories from
a prior run of the same project.

This mirrors the direct-DB access already used by
:mod:`mindmemos_eval.memory.metrics` (Qdrant + ClickHouse) and reuses the same
environment-variable configuration as the server (``MINDMEMOS_QDRANT_URL``,
``MINDMEMOS_NEO4J_*``), so eval and server share one config source in local dev.
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
DEFAULT_NEO4J_PASSWORD = "mindmemos_dev_password"
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
)


@dataclass(frozen=True)
class ResetConfig:
    """Connection settings for direct Qdrant/Neo4j cleanup.

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


_QDRANT_COLLECTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("memory_collection", "memory_item_v1"),
    ("entity_collection", "entity_item_v1"),
    ("source_collection", "source_item_v1"),
    ("add_record_collection", "add_record_v1"),
    ("schema_add_buffer_collection", "schema_add_buffer_v1"),
    ("search_record_collection", "search_record_v1"),
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
        str(qdrant.get(field_name, default_name))
        for field_name, default_name in _QDRANT_COLLECTION_FIELDS
    )
    logger.info("resolved_qdrant_collections path=%s collections=%s", path, collections)
    return collections


async def reset_project(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Physically delete every memory record scoped to ``project_id``.

    Qdrant: delete-by-filter on each configured collection. Neo4j: DETACH DELETE
    all nodes carrying this project_id. Returns per-store deletion counts. Raises
    ``ValueError`` if project_id is empty.
    """
    if not project_id:
        raise ValueError("project_id is required to reset a project")

    counts: dict[str, int] = {}
    counts.update(await _reset_qdrant(cfg, project_id))
    counts.update(await _reset_neo4j(cfg, project_id))
    logger.info("project_reset_done project_id=%s counts=%s", project_id, counts)
    return counts


async def _reset_qdrant(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Delete all points matching project_id in every configured collection."""
    client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, trust_env=False)
    filter_ = qmodels.Filter(
        must=[qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id))]
    )
    selector = qmodels.FilterSelector(filter=filter_)
    counts: dict[str, int] = {}
    try:
        for collection in cfg.collections:
            deleted: int
            try:
                counted = await client.count(collection_name=collection, count_filter=filter_, exact=True)
                deleted = counted.count
            except Exception:  # noqa: BLE001 - collection may not exist yet
                deleted = -1
            try:
                await client.delete(collection_name=collection, points_selector=selector)
            except Exception as exc:  # noqa: BLE001 - collection may not exist yet
                logger.warning("qdrant_reset_skipped collection=%s error=%s", collection, exc)
                deleted = 0
            counts[f"qdrant:{collection}"] = deleted
    finally:
        await client.close()
    return counts


async def _reset_neo4j(cfg: ResetConfig, project_id: str) -> dict[str, int]:
    """Delete all graph nodes (and their relationships) carrying project_id."""
    # Imported lazily so the eval package stays importable without the neo4j
    # driver installed; cleanup is the only path that needs it.
    from neo4j import AsyncGraphDatabase, RoutingControl

    driver = AsyncGraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    total = 0
    try:
        result = await driver.execute_query(
            "MATCH (n) WHERE n.`project_id` = $project_id RETURN count(n) AS total",
            {"project_id": project_id},
            routing_=RoutingControl.READ,
            database_=cfg.neo4j_database,
        )
        if result.records:
            total = int(result.records[0]["total"])
        await driver.execute_query(
            "MATCH (n) WHERE n.`project_id` = $project_id DETACH DELETE n",
            {"project_id": project_id},
            routing_=RoutingControl.WRITE,
            database_=cfg.neo4j_database,
        )
    finally:
        await driver.close()
    return {"neo4j:nodes": total}
