"""Trajectory selection, compression, embedding, and retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ....typing import Task
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from ..skill_grpo_with_replay_buffer.models import ChatModel, EmbeddingModel, chat_content, embedding_vectors
from .contracts import (
    RetrievedTrajectoryMemory,
    TaskRetrievalRecord,
    TrajectoryMemoryItem,
    TrajectorySnapshot,
    TrajectorySummary,
)
from .prompts import summary_messages

_TASK_QUERY_PATTERN = re.compile(r"Your task is to:\s*([^\n]+)", re.IGNORECASE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")


def task_retrieval_key(task: Task) -> str:
    """Create the same semantic key for train and test tasks without opening the environment."""
    task_type = str(task.metadata.get("task_type") or task.metadata.get("instruction_type") or "unknown")
    gamefile = str(task.metadata.get("gamefile") or task.metadata.get("resolved_gamefile") or "")
    descriptor = ""
    if gamefile:
        parts = Path(gamefile).parts
        if len(parts) >= 3:
            descriptor = parts[-3]
    fields = descriptor.split("-") if descriptor else []
    if len(fields) >= 5:
        source_type, target, movable, destination = fields[0], fields[1], fields[2], fields[3]
        task_type = source_type or task_type
        return (
            f"ALFWorld task type: {task_type.replace('_', ' ')}; "
            f"target object: {_humanize(target)}; "
            f"movable receptacle: {_humanize(movable)}; "
            f"destination or appliance: {_humanize(destination)}"
        )
    return f"ALFWorld task type: {task_type.replace('_', ' ')}; task: {task.instruction}"


def query_from_events(events: list[dict[str, Any]], *, fallback: str) -> str:
    for event in events:
        if event.get("role") != "user":
            continue
        content = event.get("content")
        if not isinstance(content, str):
            continue
        match = _TASK_QUERY_PATTERN.search(content)
        if match:
            return match.group(1).strip().rstrip(".")
    return fallback


def snapshot_from_outcome(outcome: RolloutOutcome) -> TrajectorySnapshot | None:
    trajectory = outcome.trajectory
    if trajectory is None:
        return None
    events = [dict(event) for event in trajectory.events]
    fallback = task_retrieval_key(trajectory.task)
    return TrajectorySnapshot(
        task=trajectory.task,
        rollout_id=trajectory.rollout.rollout_id,
        query=query_from_events(events, fallback=fallback),
        events=events,
        reward_score=trajectory.reward.score,
        n_turn=trajectory.execution.n_turn,
        metadata={**trajectory.metadata, "sample_index": outcome.spec.sample_index},
    )


def select_trajectory_snapshots(
    snapshots: list[TrajectorySnapshot],
    *,
    success_reward: float,
    max_examples_per_task: int,
) -> list[TrajectorySnapshot]:
    """Prefer concise verified successes, falling back to concise failures."""
    grouped: dict[str, list[TrajectorySnapshot]] = defaultdict(list)
    order: list[str] = []
    for snapshot in snapshots:
        task_id = snapshot.task.task_id
        if task_id not in grouped:
            order.append(task_id)
        grouped[task_id].append(snapshot)
    selected: list[TrajectorySnapshot] = []
    for task_id in order:
        ranked = sorted(
            grouped[task_id],
            key=lambda item: (
                0 if item.reward_score is not None and item.reward_score >= success_reward else 1,
                item.n_turn,
                item.rollout_id,
            ),
        )
        selected.extend(ranked[:max_examples_per_task])
    return selected


def render_trajectory(snapshot: TrajectorySnapshot, *, max_chars: int) -> str:
    chunks: list[str] = []
    for event in snapshot.events:
        role = event.get("role")
        content = event.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            if role == "user":
                content = content.split("\n\nAdmissible actions:", 1)[0]
            chunks.append(f"{role.upper()}: {content.strip()}")
            continue
        action = event.get("action")
        feedback = event.get("env_feedback")
        if isinstance(action, str) or isinstance(feedback, str):
            chunks.append(f"ACTION: {action or '(none)'}\nOBSERVATION: {feedback or '(none)'}")
    rendered = "\n\n".join(chunk for chunk in chunks if chunk.strip())
    if len(rendered) <= max_chars:
        return rendered
    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars
    return f"{rendered[:head_chars]}\n\n...[middle omitted for length]...\n\n{rendered[-tail_chars:]}"


class TrajectoryMemoryBankBuilder:
    def __init__(
        self,
        *,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel,
        max_trajectory_chars: int,
        max_summary_chars: int,
        max_concurrent_summaries: int,
    ) -> None:
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._max_trajectory_chars = max_trajectory_chars
        self._max_summary_chars = max_summary_chars
        self._summary_slots = asyncio.Semaphore(max_concurrent_summaries)

    async def build(self, snapshots: list[TrajectorySnapshot]) -> list[TrajectoryMemoryItem]:
        async def summarize(snapshot: TrajectorySnapshot) -> TrajectoryMemoryItem:
            async with self._summary_slots:
                raw = await chat_content(
                    self._chat_model,
                    task=f"trajectory_memory.summarize.{snapshot.task.task_id}",
                    messages=summary_messages(
                        snapshot,
                        rendered_trajectory=render_trajectory(
                            snapshot,
                            max_chars=self._max_trajectory_chars,
                        ),
                    ),
                )
            summary = _parse_summary(raw, max_chars=self._max_summary_chars)
            retrieval_key = task_retrieval_key(snapshot.task)
            digest = hashlib.sha256(
                f"{snapshot.task.task_id}\x1f{snapshot.rollout_id}\x1f{retrieval_key}".encode()
            ).hexdigest()[:20]
            return TrajectoryMemoryItem(
                memory_id=f"trajectory-memory-{digest}",
                source_task_id=snapshot.task.task_id,
                source_rollout_id=snapshot.rollout_id,
                retrieval_key=retrieval_key,
                reward_score=snapshot.reward_score,
                n_turn=snapshot.n_turn,
                summary=summary,
            )

        items = list(await asyncio.gather(*(summarize(snapshot) for snapshot in snapshots)))
        if not items:
            return []
        vectors = await embedding_vectors(
            self._embedding_model,
            task="trajectory_memory.embed.index",
            texts=[item.retrieval_key for item in items],
        )
        _validate_vectors(vectors, expected_count=len(items))
        for item, vector in zip(items, vectors, strict=True):
            item.embedding = vector
        return items

    async def retrieve(
        self,
        tasks: list[Task],
        memory_bank: list[TrajectoryMemoryItem],
        *,
        top_k: int,
    ) -> list[TaskRetrievalRecord]:
        if not tasks:
            return []
        if not memory_bank:
            raise ValueError("cannot retrieve from an empty trajectory memory bank")
        keys = [task_retrieval_key(task) for task in tasks]
        vectors = await embedding_vectors(
            self._embedding_model,
            task="trajectory_memory.embed.query",
            texts=keys,
        )
        _validate_vectors(vectors, expected_count=len(tasks), expected_dim=len(memory_bank[0].embedding))
        records: list[TaskRetrievalRecord] = []
        for task, key, vector in zip(tasks, keys, vectors, strict=True):
            ranked = sorted(
                ((_cosine_similarity(vector, item.embedding), item) for item in memory_bank),
                key=lambda pair: (-pair[0], pair[1].memory_id),
            )[:top_k]
            records.append(
                TaskRetrievalRecord(
                    task_id=task.task_id,
                    retrieval_key=key,
                    memories=[
                        RetrievedTrajectoryMemory(rank=rank, similarity=similarity, item=item)
                        for rank, (similarity, item) in enumerate(ranked, start=1)
                    ],
                )
            )
        return records


def _parse_summary(raw: str, *, max_chars: int) -> TrajectorySummary:
    payload = _load_json_object(raw)
    if payload is None:
        compact = raw.strip()[:max_chars] or "No usable summary was returned."
        return TrajectorySummary(
            title="Unstructured trajectory memory",
            task_summary="See strategy text.",
            strategy=compact,
            cautions=["The summarizer returned unstructured text; apply this memory conservatively."],
        )
    return TrajectorySummary(
        title=_text(payload.get("title"), fallback="Trajectory strategy", max_chars=160),
        task_summary=_text(payload.get("task_summary"), fallback="Task pattern not stated.", max_chars=400),
        strategy=_text(payload.get("strategy"), fallback="Strategy not stated.", max_chars=max_chars),
        key_steps=_string_list(payload.get("key_steps"), limit=6, item_chars=500),
        transferable_lessons=_string_list(payload.get("transferable_lessons"), limit=4, item_chars=500),
        cautions=_string_list(payload.get("cautions"), limit=4, item_chars=500),
    )


def _load_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _text(value: Any, *, fallback: str, max_chars: int) -> str:
    text = str(value or "").strip()
    return (text or fallback)[:max_chars]


def _string_list(value: Any, *, limit: int, item_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:item_chars] for item in value[:limit] if str(item).strip()]


def _validate_vectors(
    vectors: list[list[float]],
    *,
    expected_count: int,
    expected_dim: int | None = None,
) -> None:
    if len(vectors) != expected_count or not vectors:
        raise ValueError(f"embedding response count mismatch: expected {expected_count}, got {len(vectors)}")
    dimension = len(vectors[0])
    if dimension < 1 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding vectors must have one consistent non-zero dimension")
    if expected_dim is not None and dimension != expected_dim:
        raise ValueError(f"embedding dimension changed from {expected_dim} to {dimension}")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cannot compare embeddings with different dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def _humanize(value: str) -> str:
    if not value or value.lower() == "none":
        return "none"
    return _CAMEL_BOUNDARY.sub(" ", value).replace("_", " ").lower()


__all__ = [
    "TrajectoryMemoryBankBuilder",
    "query_from_events",
    "render_trajectory",
    "select_trajectory_snapshots",
    "snapshot_from_outcome",
    "task_retrieval_key",
]
