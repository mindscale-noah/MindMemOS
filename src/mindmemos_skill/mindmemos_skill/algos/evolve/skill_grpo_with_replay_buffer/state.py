"""Fingerprinting and resume validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ....typing import Skill, Task
from .config import SkillGrpoRunConfig
from .contracts import EvolutionState


def config_fingerprint(config: SkillGrpoRunConfig) -> str:
    return _fingerprint(config.model_dump(mode="json"))


def input_fingerprint(
    base_skill: Skill,
    train_tasks: list[Task],
    validation_tasks: list[Task],
    test_tasks: list[Task],
) -> str:
    return _fingerprint(
        {
            "base_skill_hash": base_skill.content_hash,
            "train": [task.model_dump(mode="json") for task in train_tasks],
            "validation": [task.model_dump(mode="json") for task in validation_tasks],
            "test": [task.model_dump(mode="json") for task in test_tasks],
        }
    )


def validate_resume(
    state: EvolutionState,
    *,
    run_id: str,
    algorithm_version: str,
    expected_input_fingerprint: str,
    expected_config_fingerprint: str,
    base_skill_hash: str,
) -> None:
    expected = {
        "schema_version": "2",
        "algorithm_name": "skill_grpo_with_replay_buffer",
        "run_id": run_id,
        "algorithm_version": algorithm_version,
        "input_fingerprint": expected_input_fingerprint,
        "config_fingerprint": expected_config_fingerprint,
        "base_skill_hash": base_skill_hash,
    }
    actual = {
        "schema_version": state.schema_version,
        "algorithm_name": state.algorithm_name,
        "run_id": state.run_id,
        "algorithm_version": state.algorithm_version,
        "input_fingerprint": state.input_fingerprint,
        "config_fingerprint": state.config_fingerprint,
        "base_skill_hash": state.base_skill_hash,
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise ValueError(f"resume state mismatch: {', '.join(mismatches)}")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["config_fingerprint", "input_fingerprint", "validate_resume"]
