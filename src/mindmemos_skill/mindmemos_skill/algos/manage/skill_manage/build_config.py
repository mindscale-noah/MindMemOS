"""Restore snapshot construction settings using current transport credentials."""

from copy import deepcopy

from pydantic import JsonValue

from ....persistence.db_services.graphs.writer import graph_digest
from .config import SkillManageConfig
from .constant import (
    METADATA_BASE_URL,
    METADATA_BUILD_SCENE,
    METADATA_BUILD_SETTINGS,
    METADATA_CHAT,
    METADATA_CONFIG,
    METADATA_EMBEDDING,
    METADATA_ENDPOINT_HASH,
)


def restore_build_config(current: SkillManageConfig, metadata: dict[str, JsonValue]) -> SkillManageConfig:
    """Pin construction options while retaining runtime retrieval and transport options.

    Args:
        current: Service defaults, including runtime credentials and retrieval options.
        metadata: Snapshot metadata with a build_settings.config object as persisted by
            SkillManageAlgorithm; connection objects contain endpoint_hash, not secrets.

    Returns:
        Validated settings using the snapshot's construction and model parameters.

    Raises:
        ValueError: Construction settings or a matching model endpoint are unavailable.
    """
    saved = deepcopy(metadata.get(METADATA_BUILD_SETTINGS, {}).get(METADATA_CONFIG))
    if not isinstance(saved, dict) or METADATA_BUILD_SCENE not in saved or METADATA_EMBEDDING not in saved:
        raise ValueError("Graph snapshot has no complete construction settings")
    if saved[METADATA_BUILD_SCENE] and METADATA_CHAT not in saved:
        raise ValueError("Graph snapshot has no captured Scene extraction model settings")
    runtime = current.model_dump(mode="python")
    for name in (METADATA_EMBEDDING, METADATA_CHAT):
        if name not in saved or saved[name] is None:
            continue
        connection = saved[name]
        transport = runtime[name]
        if transport is None or connection.pop(METADATA_ENDPOINT_HASH, None) != graph_digest(
            transport[METADATA_BASE_URL]
        ):
            raise ValueError(f"Graph snapshot requires its original {name} endpoint")
        saved[name] = {**transport, **connection}
    return SkillManageConfig.model_validate({**runtime, **saved})
