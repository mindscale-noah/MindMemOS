"""Deterministic schedules for STATE-Bench self-evolution experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RoundPlan:
    """One round of ten (or configured-size) training trajectories."""

    round_index: int
    train_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackEvoSchedule:
    """Full training schedule and the read-only evaluation split."""

    rounds: tuple[RoundPlan, ...]
    train_task_ids: tuple[str, ...]
    eval_task_ids: tuple[str, ...]


def build_schedule(
    train_task_ids: list[str],
    test_task_ids: list[str],
    *,
    rounds: int = 10,
    train_per_round: int = 10,
    seed: int = 42,
    eval_count: int | None = None,
) -> FeedbackEvoSchedule:
    """Build a reproducible, non-overlapping train/eval schedule.

    ``eval_count`` exists for the small technical smoke run. Formal runs leave
    it unset and therefore use all 50 test tasks.
    """

    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if train_per_round < 1:
        raise ValueError("train_per_round must be >= 1")
    if eval_count is not None and eval_count < 1:
        raise ValueError("eval_count must be >= 1 when provided")

    required_train = rounds * train_per_round
    train_ids = list(train_task_ids)
    test_ids = list(test_task_ids)
    if len(train_ids) < required_train:
        raise ValueError(
            f"train split has {len(train_ids)} tasks but the schedule needs "
            f"{required_train} ({rounds} rounds x {train_per_round}/round)"
        )
    required_eval = 50 if eval_count is None else eval_count
    if len(test_ids) < required_eval:
        raise ValueError(f"test split has {len(test_ids)} tasks but eval needs {required_eval}")

    rng = random.Random(seed)
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    selected_eval = tuple(test_ids if eval_count is None else test_ids[:eval_count])
    train_chunks = [tuple(train_ids[i : i + train_per_round]) for i in range(0, required_train, train_per_round)]
    selected_train = tuple(train_ids[:required_train])
    if len(set(selected_train)) != len(selected_train):
        raise ValueError("train split contains duplicate task IDs")
    if len(set(selected_eval)) != len(selected_eval):
        raise ValueError("test split contains duplicate task IDs")
    if set(selected_train) & set(selected_eval):
        raise ValueError("train and test splits overlap")
    round_plans = tuple(RoundPlan(round_index=index + 1, train_task_ids=train_chunks[index]) for index in range(rounds))
    return FeedbackEvoSchedule(
        rounds=round_plans,
        train_task_ids=selected_train,
        eval_task_ids=selected_eval,
    )
