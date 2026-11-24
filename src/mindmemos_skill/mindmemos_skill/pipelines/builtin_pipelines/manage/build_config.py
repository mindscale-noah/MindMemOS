"""Restore snapshot construction settings using current transport credentials."""

from copy import deepcopy

from pydantic import JsonValue

from ....algos.manage.skill_manage.config import SkillManageConfig
from ....algos.manage.skill_manage.constant import (
    METADATA_BASE_URL,
    METADATA_BUILD_SCENE,
    METADATA_BUILD_SETTINGS,
    METADATA_CHAT,
    METADATA_CONFIG,
    METADATA_EMBEDDING,
    METADATA_ENDPOINT_HASH,
)
from ....infra.embedding import OpenAIEmbeddingConfig
from ....infra.model_access.typing import EmbeddingSpace
from ....persistence.db_services.graphs.config import EmbeddingProfile
from ....persistence.db_services.graphs.writer import graph_digest


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
        endpoint_hash = connection.pop(METADATA_ENDPOINT_HASH, None)
        saved_mode = connection.get("transport", "direct")
        if transport is None or (
            saved_mode == "direct" and endpoint_hash != graph_digest(transport[METADATA_BASE_URL])
        ):
            raise ValueError(f"Graph snapshot requires its original {name} endpoint")
        saved[name] = {**transport, **connection, "transport": saved_mode}
        if name == METADATA_EMBEDDING and saved_mode == "direct":
            saved[name].pop("space_id", None)
    return SkillManageConfig.model_validate({**runtime, **saved})


def pin_embedding_space(
    config: SkillManageConfig, runtime: OpenAIEmbeddingConfig, space: EmbeddingSpace
) -> SkillManageConfig:
    """Combine an authorized vector space with the graph's rendering settings.

    Args:
        config: Graph build parameters and optional prior rendering settings.
        runtime: Deployment embedding transport and batching options.
        space: Model identity and dimensions resolved by the connection layer.

    Returns:
        Build configuration pinned to the resolved vector space.
    """
    profile = EmbeddingProfile(space_id=space.space_id, model=space.model, dimensions=space.dimensions)
    previous = config.embedding_profile
    if previous is not None:
        profile = profile.model_copy(
            update={
                "metric": previous.metric,
                "normalization": previous.normalization,
                "node_template": previous.node_template,
                "query_template": previous.query_template,
            }
        )
    embedding = runtime.model_copy(
        update={
            "space_id": profile.space_id,
            "model": profile.model,
            "dimensions": profile.dimensions,
        }
    )
    return config.model_copy(update={"embedding": embedding, "embedding_profile": profile})
