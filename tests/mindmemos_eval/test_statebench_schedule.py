"""Tests for the STATE-Bench × feedback_evo 10-round schedule helpers."""

from __future__ import annotations

import pytest
from mindmemos_eval.memory.envs.statebench.schedule import build_schedule


def test_build_schedule_chunks_without_overlap():
    train = [f"train-{i}" for i in range(100)]
    test = [f"test-{i}" for i in range(50)]

    schedule = build_schedule(train, test, seed=42)

    assert len(schedule.rounds) == 10
    for plan in schedule.rounds:
        assert len(plan.train_task_ids) == 10
    # Non-overlapping chunks: each train task appears exactly once.
    train_ids = [task for plan in schedule.rounds for task in plan.train_task_ids]
    assert len(set(train_ids)) == len(train_ids) == 100
    assert len(schedule.eval_task_ids) == 50
    assert set(schedule.eval_task_ids).isdisjoint(train_ids)


def test_build_schedule_is_deterministic_for_seed():
    train = [f"train-{i}" for i in range(100)]
    test = [f"test-{i}" for i in range(50)]

    first = build_schedule(train, test, seed=7)
    second = build_schedule(train, test, seed=7)
    third = build_schedule(train, test, seed=8)

    assert first.rounds == second.rounds
    assert first.eval_task_ids == second.eval_task_ids
    assert first.rounds != third.rounds


def test_build_schedule_raises_when_split_too_small():
    with pytest.raises(ValueError, match="train split has 90 tasks"):
        build_schedule([f"t-{i}" for i in range(90)], [f"e-{i}" for i in range(50)])
    with pytest.raises(ValueError, match="test split has 45 tasks"):
        build_schedule([f"t-{i}" for i in range(100)], [f"e-{i}" for i in range(45)])


def test_build_schedule_supports_small_smoke_eval():
    schedule = build_schedule(
        [f"train-{i}" for i in range(20)],
        [f"test-{i}" for i in range(50)],
        rounds=2,
        train_per_round=10,
        eval_count=10,
        seed=42,
    )

    assert len(schedule.rounds) == 2
    assert len(schedule.train_task_ids) == 20
    assert len(schedule.eval_task_ids) == 10
