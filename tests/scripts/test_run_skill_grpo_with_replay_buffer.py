from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments import skill_grpo_with_replay_buffer as SCRIPT
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import EvolutionState
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    BatchEvolutionRecord,
    EvolutionEvent,
)
from mindmemos_skill.typing import Skill, compute_skill_content_hash


def make_skill() -> Skill:
    blob = {"SKILL.md": "# Initial\n"}
    return Skill(
        skill_id="benchmark-skill",
        version_id="run:base",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="benchmark",
        blob=blob,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def make_state(skill: Skill) -> EvolutionState:
    return EvolutionState(
        algorithm_version="2.0.0",
        run_id="run",
        input_fingerprint="input",
        config_fingerprint="config",
        base_skill_hash=skill.content_hash,
        current_skill=skill,
    )


def test_load_resume_artifacts_round_trips_checkpoint_and_base_skill(tmp_path: Path) -> None:
    skill = make_skill()
    state = make_state(skill)
    SCRIPT.write_json_atomic(tmp_path / "checkpoint.json", state.model_dump(mode="json"))
    (tmp_path / "base_skill.json").write_text(skill.model_dump_json(), encoding="utf-8")

    loaded_state, loaded_skill = SCRIPT.load_resume_artifacts(tmp_path)

    assert loaded_state == state
    assert loaded_skill == skill


def test_load_resume_artifacts_rejects_pre_resume_runner_output(tmp_path: Path) -> None:
    state = make_state(make_skill())
    SCRIPT.write_json_atomic(tmp_path / "checkpoint.json", state.model_dump(mode="json"))

    with pytest.raises(FileNotFoundError, match="predates runner resume support"):
        SCRIPT.load_resume_artifacts(tmp_path)


def test_validate_resume_arguments_ignores_only_resume_flag(tmp_path: Path) -> None:
    SCRIPT.write_json_atomic(
        tmp_path / "arguments.json",
        {"benchmark": "livemath", "epochs": 4, "resume": False},
    )

    SCRIPT.validate_resume_arguments(
        tmp_path,
        SimpleNamespace(benchmark="livemath", epochs=4, resume=True),
    )
    with pytest.raises(ValueError, match="resume argument mismatch: epochs"):
        SCRIPT.validate_resume_arguments(
            tmp_path,
            SimpleNamespace(benchmark="livemath", epochs=5, resume=True),
        )


def test_clear_incomplete_rollout_workspaces_preserves_checkpointed_rollouts(tmp_path: Path) -> None:
    preserved = tmp_path / "workspace" / "train" / "task-1" / "preserved" / "0"
    partial = tmp_path / "workspace" / "train" / "task-1" / "partial" / "0"
    unrelated = tmp_path / "workspace" / "notes" / "keep"
    preserved.mkdir(parents=True)
    partial.mkdir(parents=True)
    unrelated.mkdir(parents=True)

    removed = SCRIPT.clear_incomplete_rollout_workspaces(tmp_path / "workspace", {"preserved"})

    assert removed == 1
    assert preserved.is_dir()
    assert not partial.exists()
    assert unrelated.is_dir()


@pytest.mark.asyncio
async def test_checkpoint_writer_saves_checkpoint_and_batch_skill(tmp_path: Path) -> None:
    before = make_skill()
    content = "# Initial\n\nUse the improved guidance.\n"
    blob = {"SKILL.md": content}
    after = before.model_copy(
        update={
            "blob": blob,
            "content_hash": compute_skill_content_hash(blob),
        },
        deep=True,
    )
    state = make_state(after)
    state.batches.append(
        BatchEvolutionRecord(
            epoch=0,
            batch_index=0,
            task_ids=["task-1"],
            skill_hash_before=before.content_hash,
            skill_hash_after=after.content_hash,
            experiences=[],
            applied_edits=[],
        )
    )
    writer = SCRIPT.CheckpointWriter(output_dir=tmp_path)

    await writer.handle(
        EvolutionEvent(
            run_id="run",
            name="checkpoint_ready",
            payload={"state": state.model_dump(mode="json")},
        )
    )

    skill_path = tmp_path / "batch_artifacts" / "batch_0001" / "skill.md"
    assert skill_path.read_text(encoding="utf-8") == content
    assert EvolutionState.model_validate_json((tmp_path / "checkpoint.json").read_text(encoding="utf-8")) == state
    assert not (tmp_path / "events.jsonl").exists()
