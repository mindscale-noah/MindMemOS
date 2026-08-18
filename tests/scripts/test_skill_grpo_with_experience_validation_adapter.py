from __future__ import annotations

from pathlib import Path

import pytest
from experiments import skill_grpo_with_experience_validation as adapter
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import EvolutionEvent
from mindmemos_skill.datasets import LiveMathIdSplitDataset, SpreadsheetBenchIdSplitDataset


def test_bounded_history_env_and_seed_reach_experience_validation_run_config(tmp_path: Path) -> None:
    args = adapter.parse_args(
        [
            "--benchmark",
            "alfworld",
            "--data-root",
            str(tmp_path / "data"),
            "--split-dir",
            str(tmp_path / "splits"),
            "--initial-skill",
            str(tmp_path / "SKILL.md"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            "bounded-history-gate",
            "--max-turns",
            "50",
            "--env-seed",
            "42",
            "--max-concurrent-rollouts",
            "64",
            "--test-rollouts",
            "3",
        ]
    )

    config = adapter.build_run_config(args)

    assert config.dataset.env_ref == "alfworld_bounded_history"
    assert config.dataset.env_options == {"max_turns": 50, "seed": 42}
    assert config.rollout.max_concurrent_rollouts == 64
    assert config.rollout.test.params == {"group_size": 3}
    assert config.rollout.experience_validation.params == {"group_size": 1}
    assert config.algorithm.previous_batch_invalid_experience_feedback.enabled is False


def test_previous_batch_invalid_experience_feedback_flag_reaches_run_config(tmp_path: Path) -> None:
    args = adapter.parse_args(
        [
            "--benchmark",
            "alfworld",
            "--initial-skill",
            str(tmp_path / "SKILL.md"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            "invalid-feedback",
            "--max-turns",
            "50",
            "--previous-batch-invalid-experience-feedback",
        ]
    )

    config = adapter.build_run_config(args)

    assert config.algorithm.previous_batch_invalid_experience_feedback.enabled is True


def test_livemath_dataset_env_and_protocol_reach_experience_validation(tmp_path: Path) -> None:
    args = adapter.parse_args(
        [
            "--benchmark",
            "livemath",
            "--data-root",
            str(tmp_path / "data"),
            "--split-dir",
            str(tmp_path / "splits"),
            "--initial-skill",
            str(tmp_path / "SKILL.md"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            "livemath-new-contrast-prompt",
            "--epochs",
            "4",
            "--batch-size",
            "40",
            "--mini-batch-size",
            "8",
            "--train-rollouts",
            "4",
            "--test-rollouts",
            "3",
            "--max-turns",
            "1",
            "--no-use-theorem",
            "--no-use-sketch",
            "--shuffle-choices",
        ]
    )

    dataset = adapter.build_dataset(args)
    config = adapter.build_run_config(args)

    assert isinstance(dataset, LiveMathIdSplitDataset)
    assert dataset.seed == 0
    assert dataset.shuffle_choices is True
    assert config.training.model_dump()["epochs"] == 4
    assert config.training.batch_size == 40
    assert config.training.mini_batch_size == 8
    assert config.algorithm.reflection.enabled is True
    assert config.dataset.env_ref == "livemath"
    assert config.dataset.env_options == {"max_turns": 1, "use_theorem": False, "use_sketch": False}
    assert config.rollout.train.params == {"group_size": 4}
    assert config.rollout.experience_validation.params == {"group_size": 1}
    assert config.rollout.test.params == {"group_size": 3}


def test_spreadsheetbench_dataset_env_and_protocol_reach_experience_validation(tmp_path: Path) -> None:
    verified_root = tmp_path / "data" / "spreadsheetbench_verified_400"
    verified_root.mkdir(parents=True)
    (verified_root / "dataset.json").write_text("[]", encoding="utf-8")
    args = adapter.parse_args(
        [
            "--benchmark",
            "spreadsheetbench",
            "--data-root",
            str(tmp_path / "data"),
            "--split-dir",
            str(tmp_path / "splits"),
            "--initial-skill",
            str(tmp_path / "SKILL.md"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            "spreadsheetbench-experience-validation",
            "--epochs",
            "4",
            "--train-rollouts",
            "4",
            "--test-rollouts",
            "3",
            "--max-turns",
            "30",
            "--shell-timeout",
            "120",
            "--max-tool-observation-chars",
            "24000",
        ]
    )

    dataset = adapter.build_dataset(args)
    config = adapter.build_run_config(args)

    assert isinstance(dataset, SpreadsheetBenchIdSplitDataset)
    assert config.training.model_dump()["epochs"] == 4
    assert config.dataset.env_ref == "spreadsheetbench"
    assert config.dataset.env_options == {"max_turns": 30, "shell_timeout_seconds": 120}
    assert config.rollout.train.params == {"group_size": 4}
    assert config.rollout.experience_validation.params == {"group_size": 1}
    assert config.rollout.test.params == {"group_size": 3}
    assert config.algorithm.experience.max_tool_observation_chars == 24_000


@pytest.mark.asyncio
async def test_batch_checkpoint_writer_persists_exact_skill_and_batch(tmp_path: Path) -> None:
    writer = adapter.BatchCheckpointWriter(output_dir=tmp_path)
    skill = {"blob": {"SKILL.md": "# Batch 1\n"}, "content_hash": "hash-1"}
    batch = {"batch_index": 0, "skill_hash_after": "hash-1"}

    await writer.handle(
        EvolutionEvent(
            run_id="livemath-run",
            name="checkpoint_ready",
            payload={"batch_index": 0, "current_skill": skill, "batch": batch},
        )
    )

    batch_dir = tmp_path / "batch_artifacts" / "batch_0001"
    assert (batch_dir / "skill.md").read_text(encoding="utf-8") == "# Batch 1\n"
    assert (batch_dir / "skill.json").is_file()
    assert (batch_dir / "batch.json").is_file()
    assert (batch_dir / "checkpoint.json").is_file()
    assert (tmp_path / "checkpoint.json").is_file()

    feedback_payload = {
        "source_batch_index": 0,
        "rejected_experience_count": 1,
        "baseline_success_count": 2,
        "injected_success_count": 1,
        "degraded_success_count": 1,
    }
    await writer.handle(
        EvolutionEvent(
            run_id="livemath-run",
            name="invalid_experience_feedback_generated",
            payload=feedback_payload,
        )
    )
    persisted_feedback = batch_dir / "invalid_experience_feedback.json"
    assert persisted_feedback.is_file()
    assert '"degraded_success_count": 1' in persisted_feedback.read_text(encoding="utf-8")
