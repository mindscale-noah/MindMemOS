"""Embedding-text template and model port for the ``skill_manage`` algorithm.

``embedding_text`` is the native counterpart of GOS ``SkillNode.to_str()``: the
same ``[TAG] value`` lines in the same order, restricted to the fields the
native node carries. The template is versioned through
``EMBEDDING_TEMPLATE_VERSION`` and rejected when a caller's profile names a
different version, so a template change can never mix vectors inside one
generation.

The model port is structural: anything exposing ``EmbedClient.embed`` works,
and this module never imports a provider SDK.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol, Sequence

from ...errors import EmbeddingDimensionError
from ...typing.skill_manage import EmbeddingProfile, SkillGraphEmbeddingText, SkillGraphNode
from .errors import (
    SkillManageError,
    embedding_count_mismatch,
    embedding_dimension_mismatch,
    embedding_non_finite,
    embedding_profile_mismatch,
    embedding_provider_error,
)

EMBEDDING_TEMPLATE_VERSION = "1"
EMBEDDING_NORMALIZATION = "nfc"
EMBEDDING_BATCH_SIZE = 64


class EmbeddingResponseLike(Protocol):
    embeddings: list[list[float]]


class EmbeddingModel(Protocol):
    """Structural view of ``mindmemos_skill.llm.EmbedClient``."""

    async def embed(
        self,
        task: str,
        text: str | list[str],
        *,
        model: str | None = None,
        expected_dim: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponseLike: ...


def embedding_text(node: SkillGraphNode) -> str:
    parts = [f"[{'Skill' if node.node_type == 'skill' else 'Scene'}] {node.name}"]
    if node.description:
        parts.append(f"[DESCRIPTION] {node.description}")
    if node.one_line_capability:
        parts.append(f"[CAPABILITY] {node.one_line_capability}")
    if node.inputs:
        parts.append(f"[INPUTS] {_join(node.inputs)}")
    if node.outputs:
        parts.append(f"[OUTPUTS] {_join(node.outputs)}")
    if node.domain_tags:
        parts.append(f"[DOMAIN_TAGS] {_join(node.domain_tags)}")
    if node.tooling:
        parts.append(f"[TOOLING] {_join(node.tooling)}")
    if node.example_tasks:
        parts.append(f"[EXAMPLE_TASKS] {_join(node.example_tasks)}")
    if node.rendered_snippet:
        parts.append(f"[SNIPPET] {node.rendered_snippet}")
    return "\n".join(parts)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_content_hash(nodes: Sequence[SkillGraphNode]) -> tuple[SkillGraphNode, ...]:
    """Stamp each skill node with the hash of its embedding text."""

    return tuple(
        node.model_copy(update={"content_hash": content_hash(embedding_text(node))})
        if node.node_type == "skill"
        else node
        for node in nodes
    )


def skill_embedding_texts(nodes: Sequence[SkillGraphNode]) -> tuple[SkillGraphEmbeddingText, ...]:
    texts: list[SkillGraphEmbeddingText] = []
    for node in nodes:
        if node.node_type != "skill":
            continue
        text = embedding_text(node)
        texts.append(
            SkillGraphEmbeddingText(
                graph_node_id=node.graph_node_id,
                content_hash=node.content_hash or content_hash(text),
                text=text,
            )
        )
    return tuple(texts)


def require_template(profile: EmbeddingProfile) -> None:
    """Reject a profile whose template or normalization this algorithm does not produce."""

    expected = {"template_version": EMBEDDING_TEMPLATE_VERSION, "normalization": EMBEDDING_NORMALIZATION}
    actual = {"template_version": profile.template_version, "normalization": profile.normalization}
    if actual != expected:
        raise embedding_profile_mismatch(expected=expected, actual=actual)


async def embed_texts(
    model: EmbeddingModel,
    profile: EmbeddingProfile,
    texts: Sequence[str],
    *,
    task: str,
) -> tuple[tuple[float, ...], ...]:
    """Embed ``texts`` in order, validating count, dimension and finiteness.

    Provider exceptions are reported by type name only: their text can carry
    endpoint URLs, keys or response bodies, and the typed envelope is persisted
    and returned to callers.
    """

    require_template(profile)
    vectors: list[tuple[float, ...]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
        try:
            response = await model.embed(task, batch, model=profile.model, expected_dim=profile.dimension)
        except SkillManageError:
            raise
        except EmbeddingDimensionError as exc:
            raise embedding_dimension_mismatch(expected=exc.expected, actual=exc.actual, model=profile.model) from exc
        except Exception as exc:
            raise embedding_provider_error(model=profile.model, task=task, reason=type(exc).__name__) from exc
        embeddings = list(getattr(response, "embeddings", None) or [])
        if len(embeddings) != len(batch):
            raise embedding_count_mismatch(expected=len(batch), actual=len(embeddings), model=profile.model)
        for offset, raw in enumerate(embeddings):
            vector = tuple(float(value) for value in raw)
            if len(vector) != profile.dimension:
                raise embedding_dimension_mismatch(expected=profile.dimension, actual=len(vector), model=profile.model)
            if not all(math.isfinite(value) for value in vector):
                raise embedding_non_finite(index=start + offset, model=profile.model)
            vectors.append(vector)
    return tuple(vectors)


def _join(values: Sequence[str]) -> str:
    return "\n".join(value.strip() for value in values if value and value.strip())


__all__ = [
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_NORMALIZATION",
    "EMBEDDING_TEMPLATE_VERSION",
    "EmbeddingModel",
    "EmbeddingResponseLike",
    "content_hash",
    "embed_texts",
    "embedding_text",
    "require_template",
    "skill_embedding_texts",
    "with_content_hash",
]
