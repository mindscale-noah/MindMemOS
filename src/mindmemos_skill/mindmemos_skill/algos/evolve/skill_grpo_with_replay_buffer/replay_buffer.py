"""Concatenated find/replace embedding clusters with incremental LLM fusion."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from .config import ReplayBufferConfig
from .contracts import ReplayClusterState, ReplayEditRecord, SkillTextEdit
from .models import ChatModel, EmbeddingModel, chat_content, embedding_vectors
from .prompts import fusion_messages


@dataclass(slots=True)
class TouchedCluster:
    cluster: ReplayClusterState
    find_sources: list[tuple[str, str]] = field(default_factory=list)


class FusedReplayBuffer:
    def __init__(
        self,
        *,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel | None,
        config: ReplayBufferConfig,
        clusters: list[ReplayClusterState] | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self.config = config
        self.clusters = [cluster.model_copy(deep=True) for cluster in clusters or []]
        self.embedding_dimension = embedding_dimension
        self._validate_restored_dimensions()
        self._next_cluster_index = self._derive_next_cluster_index()

    async def ingest(
        self,
        batch_index: int,
        edits: list[tuple[SkillTextEdit, str, float]],
    ) -> list[TouchedCluster]:
        edits = [(edit, source, advantage) for edit, source, advantage in edits if edit.find or edit.replace.strip()]
        if not edits:
            return []
        vectors = await self._concat_vectors([edit for edit, _, _ in edits])
        pending: dict[str, TouchedCluster] = {}
        prior_counts: dict[str, int] = {}
        new_replaces: dict[str, list[str]] = {}
        new_keys: dict[str, list[str]] = {}

        for (edit, source_task_id, advantage), vector in zip(edits, vectors):
            key_text = self.key_text(edit)
            cluster = self._nearest(vector, key_text)
            if cluster is None:
                cluster = ReplayClusterState(
                    cluster_id=f"cluster-{self._next_cluster_index:06d}",
                    centroid=list(vector or []),
                    centroid_text=key_text,
                    last_seen_batch=batch_index,
                )
                self._next_cluster_index += 1
                self.clusters.append(cluster)
            else:
                self._fold_centroid(cluster, vector)
            if cluster.cluster_id not in pending:
                pending[cluster.cluster_id] = TouchedCluster(cluster=cluster)
                prior_counts[cluster.cluster_id] = len(cluster.records)
                new_replaces[cluster.cluster_id] = []
                new_keys[cluster.cluster_id] = []
            cluster.records.append(
                ReplayEditRecord(
                    edit=edit,
                    batch_index=batch_index,
                    source_task_id=source_task_id,
                    advantage=advantage,
                )
            )
            cluster.last_seen_batch = batch_index
            pending[cluster.cluster_id].find_sources.append((edit.find, source_task_id))
            new_replaces[cluster.cluster_id].append(edit.replace)
            new_keys[cluster.cluster_id].append(key_text)

        for cluster_id, touched in pending.items():
            await self._fuse(
                touched.cluster,
                prior_count=prior_counts[cluster_id],
                new_replaces=list(dict.fromkeys(new_replaces[cluster_id])),
                new_key_texts=list(dict.fromkeys(new_keys[cluster_id])),
            )
        self._evict(batch_index)
        return list(pending.values())

    async def _concat_vectors(self, edits: list[SkillTextEdit]) -> list[list[float] | None]:
        if self._embedding_model is None:
            return [None] * len(edits)
        texts: list[str] = []
        slots: list[tuple[int | None, int | None]] = []
        for edit in edits:
            find_index = len(texts) if edit.find.strip() else None
            if find_index is not None:
                texts.append(edit.find.strip())
            replace_index = len(texts) if edit.replace.strip() else None
            if replace_index is not None:
                texts.append(edit.replace.strip())
            slots.append((find_index, replace_index))
        if not texts:
            return [None] * len(edits)
        try:
            embedded = await embedding_vectors(self._embedding_model, task="skill_grpo.edit_cluster", texts=texts)
            if len(embedded) != len(texts) or not embedded or not embedded[0]:
                raise ValueError("embedding response shape mismatch")
            dimension = len(embedded[0])
            if any(len(vector) != dimension for vector in embedded):
                raise ValueError("embedding dimensions are inconsistent")
            if self.embedding_dimension is not None and dimension != self.embedding_dimension:
                raise ValueError(f"embedding dimension changed from {self.embedding_dimension} to {dimension}")
            self.embedding_dimension = dimension
        except Exception:
            return [None] * len(edits)
        zero = [0.0] * dimension
        return [
            (embedded[find_index] if find_index is not None else zero)
            + (embedded[replace_index] if replace_index is not None else zero)
            if find_index is not None or replace_index is not None
            else None
            for find_index, replace_index in slots
        ]

    def _nearest(self, vector: list[float] | None, key_text: str) -> ReplayClusterState | None:
        if vector is None:
            return next((cluster for cluster in self.clusters if cluster.centroid_text == key_text), None)
        best = None
        best_similarity = self.config.similarity_threshold
        for cluster in self.clusters:
            if len(cluster.centroid) != len(vector):
                continue
            similarity = cosine_similarity(vector, cluster.centroid)
            if similarity >= best_similarity:
                best, best_similarity = cluster, similarity
        return best

    @staticmethod
    def _fold_centroid(cluster: ReplayClusterState, vector: list[float] | None) -> None:
        if not vector:
            return
        if not cluster.centroid or len(cluster.centroid) != len(vector):
            cluster.centroid = list(vector)
            return
        prior_count = len(cluster.records)
        cluster.centroid = [
            (cluster.centroid[index] * prior_count + value) / (prior_count + 1) for index, value in enumerate(vector)
        ]

    async def _fuse(
        self,
        cluster: ReplayClusterState,
        *,
        prior_count: int,
        new_replaces: list[str],
        new_key_texts: list[str],
    ) -> None:
        if prior_count == 0 and len(new_replaces) <= 1:
            cluster.committed_replace = new_replaces[0] if new_replaces else ""
            if new_key_texts:
                cluster.centroid_text = new_key_texts[0]
            return
        history_replace = cluster.committed_replace if prior_count else None
        history_centroid_text = cluster.centroid_text if prior_count else ""
        fallback_replace = history_replace or (new_replaces[-1] if new_replaces else "")
        fallback_text = history_centroid_text or (
            new_key_texts[-1] if new_key_texts else fallback_replace
        )
        try:
            raw = await chat_content(
                self._chat_model,
                task="skill_grpo.cluster_fusion",
                messages=fusion_messages(
                    history_replace=history_replace,
                    history_count=prior_count,
                    new_replaces=new_replaces,
                    history_centroid_text=history_centroid_text,
                    new_key_texts=new_key_texts,
                ),
            )
            parsed = _parse_fused(raw)
            merged, centroid_text = parsed if parsed is not None else ("", "")
        except Exception:
            merged, centroid_text = "", ""
        cluster.committed_replace = merged or fallback_replace
        cluster.centroid_text = centroid_text or fallback_text

    def mark_committed(self, cluster_ids: set[str]) -> None:
        for cluster in self.clusters:
            if cluster.cluster_id in cluster_ids:
                cluster.uses += 1
        if self.config.max_uses > 0:
            self.clusters = [cluster for cluster in self.clusters if cluster.uses < self.config.max_uses]

    def snapshot(self) -> list[ReplayClusterState]:
        return [cluster.model_copy(deep=True) for cluster in self.clusters]

    def _evict(self, batch_index: int) -> None:
        del batch_index
        if self.config.capacity <= 0 or len(self.clusters) <= self.config.capacity:
            return
        self.clusters.sort(
            key=lambda cluster: len({record.source_task_id for record in cluster.records}),
            reverse=True,
        )
        self.clusters = self.clusters[: self.config.capacity]

    def _derive_next_cluster_index(self) -> int:
        indices: list[int] = []
        for cluster in self.clusters:
            try:
                indices.append(int(cluster.cluster_id.rsplit("-", 1)[-1]))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def _validate_restored_dimensions(self) -> None:
        dimensions = {len(cluster.centroid) for cluster in self.clusters if cluster.centroid}
        if len(dimensions) > 1:
            raise ValueError("restored replay clusters have inconsistent centroid dimensions")
        restored = next(iter(dimensions), None)
        if restored is not None and self.embedding_dimension is not None and restored != self.embedding_dimension * 2:
            raise ValueError("restored replay centroid dimension does not match embedding_dimension")
        if restored is not None and self.embedding_dimension is None:
            if restored % 2:
                raise ValueError("concatenated replay centroid dimension must be even")
            self.embedding_dimension = restored // 2

    @staticmethod
    def key_text(edit: SkillTextEdit) -> str:
        find_hint = edit.find.strip()[:120]
        replace = edit.replace.strip()
        return f"{find_hint}\n>>>\n{replace}" if find_hint else replace


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _parse_fused(raw: str) -> tuple[str, str] | None:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    merged = str(data.get("merged_replace", "")).strip()
    centroid = str(data.get("centroid_text", "")).strip()
    if not merged and not centroid:
        return None
    return merged, centroid


__all__ = ["FusedReplayBuffer", "TouchedCluster", "cosine_similarity"]
