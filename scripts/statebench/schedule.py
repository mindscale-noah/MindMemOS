"""Pure scheduling helpers for the STATE-Bench × feedback_evo 10-round loop.

The loop contract (as confirmed with the user):

* train tasks only accumulate memory (add); they are chunked into
  ``rounds`` non-overlapping groups, one per round;
* a subset of the test split provides task-end feedback (one chunk per round);
  every round always triggers evolution, but the number of detected feedback
  signals is recorded per round;
* a reserved subset of the test split is never used as feedback and is only
  evaluated at the end for an unbiased final measurement.

All functions here are deterministic and side-effect free so the schedule can
be unit-tested and reproduced from the same seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RoundPlan:
    """One iteration: a train chunk (memory only) and a feedback-test chunk."""

    round_index: int
    train_task_ids: tuple[str, ...]
    feedback_test_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackEvoSchedule:
    """Full 10-round schedule plus the untouched reserved test tasks."""

    rounds: tuple[RoundPlan, ...]
    reserved_test_task_ids: tuple[str, ...]
    train_task_ids: tuple[str, ...]
    feedback_test_task_ids: tuple[str, ...]


def build_schedule(
    train_task_ids: list[str],
    test_task_ids: list[str],
    *,
    rounds: int = 10,
    train_per_round: int = 10,
    feedback_test_per_round: int = 4,
    reserved_test_count: int = 10,
    seed: int = 42,
) -> FeedbackEvoSchedule:
    """Build a deterministic chunked schedule.

    Raises:
        ValueError: When the split does not contain enough tasks for the
            requested chunking (train or test).
    """

    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if train_per_round < 1:
        raise ValueError("train_per_round must be >= 1")
    if feedback_test_per_round < 1:
        raise ValueError("feedback_test_per_round must be >= 1")
    if reserved_test_count < 0:
        raise ValueError("reserved_test_count must be >= 0")

    required_train = rounds * train_per_round
    required_feedback = rounds * feedback_test_per_round
    required_test = required_feedback + reserved_test_count

    train_ids = list(train_task_ids)
    test_ids = list(test_task_ids)
    if len(train_ids) < required_train:
        raise ValueError(
            f"train split has {len(train_ids)} tasks but the schedule needs "
            f"{required_train} ({rounds} rounds x {train_per_round}/round)"
        )
    if len(test_ids) < required_test:
        raise ValueError(
            f"test split has {len(test_ids)} tasks but the schedule needs "
            f"{required_test} ({required_feedback} feedback + {reserved_test_count} reserved)"
        )

    rng = random.Random(seed)
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)

    reserved = tuple(test_ids[:reserved_test_count])
    feedback_pool = test_ids[reserved_test_count : reserved_test_count + required_feedback]

    train_chunks = [
        tuple(train_ids[i : i + train_per_round])
        for i in range(0, required_train, train_per_round)
    ]
    feedback_chunks = [
        tuple(feedback_pool[i : i + feedback_test_per_round])
        for i in range(0, required_feedback, feedback_test_per_round)
    ]
    round_plans = tuple(
        RoundPlan(
            round_index=index + 1,
            train_task_ids=train_chunks[index],
            feedback_test_task_ids=feedback_chunks[index],
        )
        for index in range(rounds)
    )

    return FeedbackEvoSchedule(
        rounds=round_plans,
        reserved_test_task_ids=reserved,
        train_task_ids=tuple(train_ids[:required_train]),
        feedback_test_task_ids=tuple(feedback_pool),
    )
