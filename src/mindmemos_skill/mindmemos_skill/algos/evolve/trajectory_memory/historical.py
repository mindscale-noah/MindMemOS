"""Recover trajectory snapshots from replay-free runs that only recorded LLM calls."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from ....typing import Task
from .contracts import TrajectorySnapshot
from .memory import query_from_events, task_retrieval_key


def reconstruct_replay_free_trajectories(
    database_path: str | Path,
    *,
    tasks: list[Task],
    phase: str = "train",
) -> list[TrajectorySnapshot]:
    """Reconstruct sequential trajectories and align them to per-task completion logs.

    This importer is intentionally strict: each start-of-conversation boundary must
    have exactly one rollout completion record for the same task.
    """
    task_by_id = {task.task_id: task for task in tasks}
    calls_by_task: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
    logs_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    connection = sqlite3.connect(str(database_path))
    try:
        for task_id, started_at, request, response in connection.execute(
            "SELECT task, started_at, request, response FROM llm_calls ORDER BY task, started_at, _record_id"
        ):
            if task_id in task_by_id:
                calls_by_task[task_id].append((started_at, request, response))
        for payload in connection.execute(
            "SELECT payload FROM algorithm_logs "
            "WHERE component_name = 'rollout' AND step_name = 'rollout_completed' ORDER BY created_at, _record_id"
        ):
            value = json.loads(payload[0])
            task_id = value.get("task_id")
            if value.get("phase") == phase and task_id in task_by_id:
                logs_by_task[task_id].append(value)
    finally:
        connection.close()

    snapshots: list[TrajectorySnapshot] = []
    for task in tasks:
        groups = _conversation_groups(calls_by_task.get(task.task_id, []))
        logs = logs_by_task.get(task.task_id, [])
        if len(groups) != len(logs):
            raise ValueError(
                f"historical trajectory boundary mismatch for {task.task_id}: "
                f"{len(groups)} conversations versus {len(logs)} rollout logs"
            )
        for index, (group, log) in enumerate(zip(groups, logs, strict=True)):
            _, request_text, response_text = group[-1]
            request = json.loads(request_text)
            events = [dict(message) for message in request.get("messages") or []]
            assistant = _assistant_message(response_text)
            if assistant is not None:
                events.append(assistant)
            fallback = task_retrieval_key(task)
            snapshots.append(
                TrajectorySnapshot(
                    task=task,
                    rollout_id=str(log.get("rollout_id") or f"historical-{task.task_id}-{index}"),
                    query=query_from_events(events, fallback=fallback),
                    events=events,
                    reward_score=float(log["score"]) if log.get("score") is not None else None,
                    n_turn=sum(event.get("role") == "assistant" for event in events),
                    metadata={
                        "source": "reconstructed_llm_calls",
                        "historical_index": index,
                        "call_count": len(group),
                    },
                )
            )
    return snapshots


def _conversation_groups(rows: list[tuple[str, str, str | None]]) -> list[list[tuple[str, str, str | None]]]:
    groups: list[list[tuple[str, str, str | None]]] = []
    for row in rows:
        request = json.loads(row[1])
        messages = request.get("messages") or []
        if len(messages) == 2:
            groups.append([])
        if not groups:
            raise ValueError("LLM call history does not start with a two-message trajectory boundary")
        groups[-1].append(row)
    return groups


def _assistant_message(response_text: str | None) -> dict[str, Any] | None:
    if not response_text:
        return None
    response = json.loads(response_text)
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return {"role": "assistant", "content": content or ""}


__all__ = ["reconstruct_replay_free_trajectories"]
