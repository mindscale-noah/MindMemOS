"""Tests for the STATE-Bench × feedback_evo 10-round schedule helpers."""

from __future__ import annotations

import pytest

from scripts.statebench.schedule import build_schedule


def test_build_schedule_chunks_without_overlap():
    train = [f"train-{i}" for i in range(100)]
    test = [f"test-{i}" for i in range(50)]

    schedule = build_schedule(train, test, seed=42)

    assert len(schedule.rounds) == 10
    for plan in schedule.rounds:
        assert len(plan.train_task_ids) == 10
        assert len(plan.feedback_test_task_ids) == 4
    # Non-overlapping chunks: each task appears exactly once in its pool.
    train_ids = [task for plan in schedule.rounds for task in plan.train_task_ids]
    feedback_ids = [task for plan in schedule.rounds for task in plan.feedback_test_task_ids]
    assert len(set(train_ids)) == len(train_ids) == 100
    assert len(set(feedback_ids)) == len(feedback_ids) == 40
    # Reserved test tasks are disjoint from feedback tasks.
    assert len(schedule.reserved_test_task_ids) == 10
    assert set(schedule.reserved_test_task_ids).isdisjoint(feedback_ids)


def test_build_schedule_is_deterministic_for_seed():
    train = [f"train-{i}" for i in range(100)]
    test = [f"test-{i}" for i in range(50)]

    first = build_schedule(train, test, seed=7)
    second = build_schedule(train, test, seed=7)
    third = build_schedule(train, test, seed=8)

    assert first.rounds == second.rounds
    assert first.reserved_test_task_ids == second.reserved_test_task_ids
    assert first.rounds != third.rounds


def test_build_schedule_raises_when_split_too_small():
    with pytest.raises(ValueError, match="train split has 90 tasks"):
        build_schedule([f"t-{i}" for i in range(90)], [f"e-{i}" for i in range(50)])
    with pytest.raises(ValueError, match="test split has 45 tasks"):
        build_schedule([f"t-{i}" for i in range(100)], [f"e-{i}" for i in range(45)])
