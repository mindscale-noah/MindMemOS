"""Deterministic epoch and batch planning."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ....typing import Task


@dataclass(frozen=True, slots=True)
class TaskBatch:
    epoch: int
    batch_index: int
    tasks: tuple[Task, ...]


class TaskBatchPlanner:
    def build(self, tasks: list[Task], *, epochs: int, batch_size: int, seed: int) -> list[TaskBatch]:
        if not tasks:
            return []
        rng = random.Random(seed)
        batches: list[TaskBatch] = []
        batch_index = 0
        for epoch in range(epochs):
            shuffled = list(tasks)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), batch_size):
                batches.append(
                    TaskBatch(
                        epoch=epoch,
                        batch_index=batch_index,
                        tasks=tuple(shuffled[start : start + batch_size]),
                    )
                )
                batch_index += 1
        return batches


__all__ = ["TaskBatch", "TaskBatchPlanner"]
