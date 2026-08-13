from __future__ import annotations

from pathlib import Path

from experiments import skill_grpo_with_experience_validation as adapter


def test_bounded_history_env_and_seed_reach_experience_validation_run_config(tmp_path: Path) -> None:
    args = adapter.parse_args(
        [
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
