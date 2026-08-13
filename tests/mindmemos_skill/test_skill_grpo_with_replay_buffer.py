from __future__ import annotations

import asyncio
import hashlib
import json
import random
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.agents.skill_runtime import SkillInjection
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import (
    MappingAgentResolver,
    SkillGrpoEvolveInput,
    SkillGrpoRunConfig,
    SkillGrpoWithReplayBuffer,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.ablation import (
    AblationCandidate,
    AblationEvaluator,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.config import (
    AblationConfig,
    ReplayBufferConfig,
    RolloutConfig,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    EvolutionState,
    ReplayClusterState,
    ReplayEditRecord,
    RolloutPhase,
    SkillTextEdit,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.fileedit import apply_best_effort
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.models import chat_content
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.prompts import (
    EXPERIENCE_EXTRACTION_SYSTEM,
    EXPERIENCE_PATCH_SYSTEM,
    FUSION_SYSTEM,
    PATCH_REPAIR_USER,
    experience_extraction_messages,
    fusion_messages,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.replay_buffer import (
    FusedReplayBuffer,
    TouchedCluster,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.rollout import (
    AblationTarget,
    FixedGroupPlan,
    PairedAblationPlan,
    RegistryEnvFactory,
    RolloutScheduler,
    RolloutStrategyRegistry,
)
from mindmemos_skill.datasets import SpreadsheetBenchIdSplitDataset
from mindmemos_skill.envs import BaseEnv, EnvRolloutContext, PreparedRollout, SpreadsheetBenchEnv
from mindmemos_skill.envs.registered_envs.livemath import SYSTEM_PROMPT as LIVEMATH_SYSTEM
from mindmemos_skill.envs.registered_envs.spreadsheetbench import SYSTEM_PROMPT as SPREADSHEET_SYSTEM
from mindmemos_skill.llm import ChatResponse, current_llm_run_id
from mindmemos_skill.persistence.enums import SkillInjectionMode, TrajectoryStatus
from mindmemos_skill.typing import (
    AgentProfile,
    EnvConfig,
    ExecutionInfo,
    Reward,
    Rollout,
    Skill,
    Task,
    Trajectory,
    compute_skill_content_hash,
)


def make_skill(content: str = "# Demo\n\nold guidance\n") -> Skill:
    now = datetime.now(UTC)
    blob = {"SKILL.md": content}
    return Skill(
        skill_id="demo",
        version_id="v0",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="demo",
        blob=blob,
        created_at=now,
    )


class FakeChatModel:
    def __init__(self) -> None:
        self.run_ids: list[str | None] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del messages, kwargs
        self.run_ids.append(current_llm_run_id())
        if ".experience." in task:
            return SimpleNamespace(content='{"experiences": [{"lesson": "use new guidance"}]}')
        if task.startswith("skill_grpo.patch"):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "edits": [
                            {
                                "find": "old guidance",
                                "replace": "new guidance",
                                "supporting_experience_sets": [1],
                                "support_count": 1,
                            }
                        ]
                    }
                )
            )
        if task == "skill_grpo.cluster_fusion":
            return SimpleNamespace(
                content='{"merged_replace": "new guidance", "centroid_text": "replace old guidance"}'
            )
        raise AssertionError(f"unexpected chat task {task}")


class FakeEmbeddingModel:
    async def embed(self, task: str, text: str | list[str], **kwargs: Any) -> Any:
        del task, kwargs
        texts = [text] if isinstance(text, str) else text
        return SimpleNamespace(embeddings=[[1.0, float(index + 1)] for index, _ in enumerate(texts)])


class ConcurrencyTracker:
    def __init__(
        self,
        delay: float = 0.0,
        trajectory_status: TrajectoryStatus = TrajectoryStatus.SUCCEEDED,
    ) -> None:
        self.active = 0
        self.maximum = 0
        self.delay = delay
        self.trajectory_status = trajectory_status
        self.workspace_scopes: list[str] = []
        self.active_workspace_scopes: list[str] = []
        self.workspace_scope_overlaps: list[set[str]] = []


class FakeEnv(BaseEnv[EnvConfig]):
    def __init__(self, config, tracker: ConcurrencyTracker) -> None:
        super().__init__(config)
        self.tracker = tracker

    async def rollout(self, agent, task, skills, *, context):
        del agent
        self.tracker.active += 1
        self.tracker.maximum = max(self.tracker.maximum, self.tracker.active)
        self.tracker.workspace_scopes.append(context.workspace_scope)
        self.tracker.active_workspace_scopes.append(context.workspace_scope)
        self.tracker.workspace_scope_overlaps.append(set(self.tracker.active_workspace_scopes))
        try:
            if self.tracker.delay:
                await asyncio.sleep(self.tracker.delay)
            now = datetime.now(UTC)
            score = 1.0 if "new guidance" in skills[0].content else 0.0
            return Trajectory(
                trajectory_id=f"{context.rollout.rollout_id}:{context.rollout.attempt_no}",
                task=task,
                rollout=context.rollout,
                injected_skills=list(skills),
                agent=AgentProfile(),
                events=[{"role": "assistant", "content": "done"}],
                reward=Reward(score=score),
                execution=ExecutionInfo(
                    status=self.tracker.trajectory_status,
                    started_at=now,
                    finished_at=now,
                    n_turn=1,
                ),
            )
        finally:
            self.tracker.active_workspace_scopes.remove(context.workspace_scope)
            self.tracker.active -= 1

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        del trajectory, prepared
        return Reward(score=0.0)


class FakeEnvFactory:
    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self.tracker = tracker

    def create(self, ref: str, config) -> FakeEnv:
        assert ref == "fake"
        return FakeEnv(config, self.tracker)


@pytest.mark.asyncio
async def test_scheduler_has_one_global_rollout_concurrency_budget() -> None:
    tracker = ConcurrencyTracker(delay=0.02)
    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"fake": object()}),  # type: ignore[arg-type]
        env_factory=FakeEnvFactory(tracker),
        config=RolloutConfig(max_concurrent_rollouts=3),
    )
    strategy = RolloutStrategyRegistry.with_builtins().get("fixed_group")
    specs = strategy.plan(
        FixedGroupPlan(
            run_id="concurrency",
            scope="batch_0",
            phase=RolloutPhase.TRAIN.value,
            tasks=[Task(task_id="t1", instruction="solve")],
            skills=[make_skill()],
            sequence_start=0,
            group_size=8,
            agent_ref="fake",
            env_ref="fake",
            seed=1,
        )
    )

    outcomes = await scheduler.run(specs)

    assert len(outcomes) == 8
    assert tracker.maximum == 3
    assert [outcome.spec.sequence_no for outcome in outcomes] == list(range(8))
    assert {outcome.trajectory.environment.env_ref for outcome in outcomes if outcome.trajectory} == {"fake"}


@pytest.mark.asyncio
async def test_scheduler_shares_rollout_budget_across_concurrent_phase_runs() -> None:
    tracker = ConcurrencyTracker(delay=0.02)
    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"fake": object()}),  # type: ignore[arg-type]
        env_factory=FakeEnvFactory(tracker),
        config=RolloutConfig(max_concurrent_rollouts=2),
    )
    strategy = RolloutStrategyRegistry.with_builtins().get("fixed_group")
    before_specs = strategy.plan(
        FixedGroupPlan(
            run_id="concurrent-phases",
            scope="ablation_0",
            phase=RolloutPhase.ABLATION_BEFORE.value,
            tasks=[Task(task_id="before", instruction="solve")],
            skills=[make_skill()],
            sequence_start=0,
            group_size=3,
            agent_ref="fake",
            env_ref="fake",
            seed=1,
        )
    )
    after_specs = strategy.plan(
        FixedGroupPlan(
            run_id="concurrent-phases",
            scope="ablation_0",
            phase=RolloutPhase.ABLATION_AFTER.value,
            tasks=[Task(task_id="after", instruction="solve")],
            skills=[make_skill("# Demo\n\nnew guidance\n")],
            sequence_start=3,
            group_size=3,
            agent_ref="fake",
            env_ref="fake",
            seed=1,
        )
    )

    before, after = await asyncio.gather(scheduler.run(before_specs), scheduler.run(after_specs))

    assert len(before) == 3
    assert len(after) == 3
    assert tracker.maximum == 2
    assert any(
        overlap == {RolloutPhase.ABLATION_BEFORE.value, RolloutPhase.ABLATION_AFTER.value}
        for overlap in tracker.workspace_scope_overlaps
    )


@pytest.mark.asyncio
async def test_scheduler_preserves_returned_failed_trajectory_as_training_evidence() -> None:
    tracker = ConcurrencyTracker(trajectory_status=TrajectoryStatus.FAILED)
    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"fake": object()}),  # type: ignore[arg-type]
        env_factory=FakeEnvFactory(tracker),
        config=RolloutConfig(max_concurrent_rollouts=1),
    )
    spec = (
        RolloutStrategyRegistry.with_builtins()
        .get("fixed_group")
        .plan(
            FixedGroupPlan(
                run_id="failed-evidence",
                scope="batch_0",
                phase=RolloutPhase.TRAIN.value,
                tasks=[Task(task_id="t1", instruction="solve")],
                skills=[make_skill()],
                sequence_start=0,
                group_size=1,
                agent_ref="fake",
                env_ref="fake",
                seed=1,
            )
        )[0]
    )

    outcome = (await scheduler.run([spec]))[0]

    assert outcome.succeeded is True
    assert outcome.trajectory is not None
    assert outcome.trajectory.execution.status is TrajectoryStatus.FAILED


def test_paired_ablation_reuses_before_and_preserves_source_phase_order() -> None:
    task_1 = Task(task_id="t1", instruction="one")
    task_2 = Task(task_id="t2", instruction="two")
    before = make_skill()
    after = make_skill("# Demo\n\nnew guidance\n")
    specs = (
        RolloutStrategyRegistry.with_builtins()
        .get("paired_ablation")
        .plan(
            PairedAblationPlan(
                run_id="paired",
                scope="batch_0",
                tasks=[task_1, task_2],
                before_skill=before,
                targets=[AblationTarget(candidate_id="c1", skill=after, task_ids=["t1"])],
                sequence_start=0,
                samples_per_case=2,
                agent_ref="fake",
                env_ref="fake",
                seed=3,
            )
        )
    )

    before_specs = [spec for spec in specs if spec.phase is RolloutPhase.ABLATION_BEFORE]
    after_specs = [spec for spec in specs if spec.phase is RolloutPhase.ABLATION_AFTER]
    assert len(before_specs) == 4
    assert len(after_specs) == 2
    assert specs == [*before_specs, *after_specs]
    assert [spec.sample_index for spec in specs] == list(range(1_000_001, 1_000_007))
    for after_spec in after_specs:
        paired_before = next(spec for spec in before_specs if spec.pair_id == after_spec.pair_id)
        assert paired_before.seed is None
        assert after_spec.seed is None


@pytest.mark.asyncio
async def test_complete_algorithm_applies_positive_ablation_candidate_and_returns_state() -> None:
    tracker = ConcurrencyTracker(delay=0.01)
    chat_model = FakeChatModel()
    events = []

    async def capture_event(event) -> None:
        events.append(event)

    algorithm = SkillGrpoWithReplayBuffer(
        chat_model=chat_model,
        embedding_model=FakeEmbeddingModel(),
        agent_resolver=MappingAgentResolver({"fake": object()}),  # type: ignore[arg-type]
        env_factory=FakeEnvFactory(tracker),
        on_event=capture_event,
    )
    config = SkillGrpoRunConfig.model_validate(
        {
            "algorithm": {
                "replay": {"min_cluster_edits": 1},
                "ablation": {"max_source_cases_per_candidate": 1, "commit_topk": 1},
            },
            "training": {"epochs": 1, "batch_size": 1, "success_reward": 1.0, "seed": 7},
            "rollout": {
                "max_concurrent_rollouts": 2,
                "train": {"name": "fixed_group", "params": {"group_size": 2}},
                "ablation": {"name": "paired_ablation", "params": {"samples_per_case": 1}},
                "test": {"name": "fixed_group", "params": {"group_size": 1}},
            },
            "dataset": {"env_ref": "fake", "agent_ref": "fake"},
        }
    )

    result = await algorithm.evolve(
        SkillGrpoEvolveInput(
            run_id="run-1",
            base_skill=make_skill(),
            train_tasks=[Task(task_id="train-1", instruction="train")],
            test_tasks=[Task(task_id="test-1", instruction="test")],
            config=config,
        )
    )

    assert result.changed is True
    assert len(result.trajectories) == sum(
        attempt.trajectory is not None for outcome in result.rollouts for attempt in outcome.attempts
    )
    assert "new guidance" in result.final_skill.content
    assert result.final_skill.metadata["evolution"]["unpersisted_candidate"] is True
    assert result.batches[0].candidates[0].net_effect == 1.0
    assert result.batches[0].candidates[0].chosen is True
    assert result.metrics.edits_applied == 1
    assert result.metrics.test_score_mean == 1.0
    assert result.state.replay_clusters[0].uses == 1
    assert result.state.replay_clusters[0].records[0].committed is False
    assert result.state.ablation_rng_state is not None
    assert any(
        overlap == {RolloutPhase.ABLATION_BEFORE.value, RolloutPhase.ABLATION_AFTER.value}
        for overlap in tracker.workspace_scope_overlaps
    )
    assert result.artifacts[0].name == "evolution_state"
    assert set(chat_model.run_ids) == {"run-1"}
    assert current_llm_run_id() is None
    event_names = {event.name for event in events}
    assert {
        "run_started",
        "batch_started",
        "phase_started",
        "phase_completed",
        "stage_started",
        "stage_completed",
        "batch_completed",
        "checkpoint_ready",
        "run_finished",
    } <= event_names
    train_phase = next(event for event in events if event.name == "phase_started" and event.payload["phase"] == "train")
    assert train_phase.payload["rollout_count"] == 2
    batch_completed = next(event for event in events if event.name == "batch_completed")
    assert batch_completed.payload["train_score"] == result.batches[0].train_score
    assert batch_completed.payload["gate_kept"] == 1
    assert batch_completed.payload["gate_rejected"] == 0
    assert batch_completed.payload["skill_chars_delta"] == len("new guidance") - len("old guidance")
    assert batch_completed.payload["applied_edit_details"] == [{"find": "old guidance", "replace": "new guidance"}]
    assert batch_completed.payload["duration_seconds"] >= 0
    gate_event = next(
        event for event in events if event.name == "stage_completed" and event.payload["stage"] == "ablation_scoring"
    )
    assert gate_event.payload["kept_count"] == 1
    assert gate_event.payload["rejected_count"] == 0
    assert gate_event.payload["decisions"][0]["decision"] == "kept"
    assert gate_event.payload["decisions"][0]["edit"] == {
        "find": "old guidance",
        "replace": "new guidance",
    }
    replay_gate_event = next(
        event for event in events if event.name == "stage_completed" and event.payload["stage"] == "candidate_selection"
    )
    assert replay_gate_event.payload["decisions"][0]["decision"] == "kept"
    assert replay_gate_event.payload["decisions"][0]["reason"] == "passed_record_count_gate"
    assert replay_gate_event.payload["decisions"][0]["record_count"] == 1
    skill_update = next(
        event for event in events if event.name == "stage_completed" and event.payload["stage"] == "skill_update"
    )
    assert skill_update.payload["chars_delta"] == len("new guidance") - len("old guidance")
    assert skill_update.payload["edits"] == [{"find": "old guidance", "replace": "new guidance"}]
    checkpoints = [event for event in events if event.name == "checkpoint_ready"]
    assert checkpoints[-1].payload["state"]["final_test_completed"] is True

    resumed = await algorithm.evolve(
        SkillGrpoEvolveInput(
            run_id="run-1",
            base_skill=make_skill(),
            train_tasks=[Task(task_id="train-1", instruction="train")],
            test_tasks=[Task(task_id="test-1", instruction="test")],
            config=config,
            resume_state=EvolutionState.model_validate_json(result.state.model_dump_json()),
        )
    )
    assert len(resumed.batches) == 1
    assert len(resumed.candidates) == 1
    assert len(resumed.rollouts) == len(result.rollouts)
    assert resumed.final_skill.content_hash == result.final_skill.content_hash


@pytest.mark.asyncio
async def test_checkpoint_callback_failure_is_not_silenced() -> None:
    async def fail_checkpoint(event) -> None:
        del event
        raise OSError("checkpoint disk failure")

    algorithm = SkillGrpoWithReplayBuffer(
        chat_model=FakeChatModel(),
        embedding_model=FakeEmbeddingModel(),
        agent_resolver=MappingAgentResolver({"fake": object()}),  # type: ignore[arg-type]
        env_factory=FakeEnvFactory(ConcurrencyTracker()),
        on_event=fail_checkpoint,
    )

    await algorithm._emit("run", "ordinary_event", {}, critical=False)
    with pytest.raises(OSError, match="checkpoint disk failure"):
        await algorithm._emit("run", "checkpoint_ready", {}, critical=True)


class AlwaysAChatClient:
    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        del task, messages, kwargs
        return ChatResponse(finish_reason="stop", content="<answer>A</answer>")


@pytest.mark.asyncio
async def test_complete_algorithm_runs_on_livemath_env_without_policy() -> None:
    agent = ReactAgent(
        {"model": "fake", "skill_injection_mode": "system_prompt", "max_turns": 1},
        llm=AlwaysAChatClient(),
    )
    algorithm = SkillGrpoWithReplayBuffer(
        chat_model=FakeChatModel(),
        embedding_model=FakeEmbeddingModel(),
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
    )
    task = Task(
        task_id="live-1",
        instruction="Choose A.",
        metadata={
            "id": "live-1",
            "question": "Which answer?",
            "choices": [{"label": "A", "text": "correct"}, {"label": "B", "text": "wrong"}],
            "correct_choice": {"label": "A", "text": "correct"},
        },
    )
    config = SkillGrpoRunConfig.model_validate(
        {
            "algorithm": {
                "replay": {"min_cluster_edits": 1},
                "ablation": {"max_source_cases_per_candidate": 1},
            },
            "training": {"epochs": 1, "batch_size": 1},
            "rollout": {
                "max_concurrent_rollouts": 2,
                "train": {"name": "fixed_group", "params": {"group_size": 2}},
                "ablation": {"name": "paired_ablation", "params": {"samples_per_case": 1}},
            },
            "dataset": {"env_ref": "livemath", "agent_ref": "react"},
        }
    )

    result = await algorithm.evolve(
        SkillGrpoEvolveInput(run_id="livemath-run", base_skill=make_skill(), train_tasks=[task], config=config)
    )

    assert result.metrics.rollouts_completed == 4
    assert result.batches[0].train_score == 1.0
    assert result.batches[0].candidates[0].net_effect == 0.0
    assert result.batches[0].candidates[0].chosen is False
    assert result.changed is False

    non_strict_config = config.model_copy(
        update={
            "algorithm": config.algorithm.model_copy(
                update={"ablation": config.algorithm.ablation.model_copy(update={"improvement_threshold": 0.0})}
            )
        }
    )
    non_strict = await algorithm.evolve(
        SkillGrpoEvolveInput(
            run_id="livemath-non-strict",
            base_skill=make_skill(),
            train_tasks=[task],
            config=non_strict_config,
        )
    )
    assert non_strict.batches[0].candidates[0].net_effect == 0.0
    assert non_strict.batches[0].candidates[0].chosen is True
    assert non_strict.changed is True


class SpreadsheetScriptAgent:
    def __init__(self) -> None:
        self.turn = 0

    async def respond(self, request, messages, *, tools=()):
        del request, messages, tools
        self.turn += 1
        if self.turn == 1:
            return ChatResponse(
                finish_reason="tool_calls",
                content="",
                tool_calls=[
                    {
                        "id": "copy",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"commands": ["cp input.xlsx output.xlsx"]}),
                        },
                    }
                ],
            )
        return ChatResponse(finish_reason="stop", content="done")

    def on_skill_runtime_task(self, request, *, context=None):
        del request, context

        @asynccontextmanager
        async def scope():
            yield SkillInjection(mode=SkillInjectionMode.TOOL)

        return scope()

    @staticmethod
    def apply_skill_injection(messages, injection):
        del injection
        return messages

    def build_trajectory(
        self,
        *,
        request,
        messages,
        started_at,
        ended_at,
        n_turn,
        is_success,
        error_info,
        metadata=None,
    ):
        return Trajectory(
            trajectory_id=request.trajectory_id,
            task=request.task,
            rollout=request.rollout,
            environment=request.environment,
            injected_skills=request.skills,
            agent=AgentProfile(),
            events=messages,
            execution=ExecutionInfo(
                status=TrajectoryStatus.SUCCEEDED if is_success else TrajectoryStatus.FAILED,
                started_at=datetime.fromtimestamp(started_at, tz=UTC),
                finished_at=datetime.fromtimestamp(ended_at, tz=UTC),
                n_turn=n_turn,
                error_info=error_info,
            ),
            metadata=metadata or {},
        )


@pytest.mark.asyncio
async def test_spreadsheetbench_dataset_and_attempt_isolated_env(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    data_root = tmp_path / "data"
    source_dir = data_root / "spreadsheetbench_verified_400" / "spreadsheet" / "case-1"
    source_dir.mkdir(parents=True)
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 42
    workbook.save(source_dir / "case_init.xlsx")
    workbook.save(source_dir / "case_golden.xlsx")
    workbook.close()
    verified = data_root / "spreadsheetbench_verified_400"
    (verified / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "instruction": "Preserve A1.",
                    "spreadsheet_path": "spreadsheet/case-1",
                    "answer_position": "A1",
                    "answer_sheet": "Sheet",
                    "instruction_type": "cell",
                }
            ]
        ),
        encoding="utf-8",
    )
    split_root = tmp_path / "resources" / "spreadsheetbench" / "splits"
    split_dir = split_root / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "items.json").write_text(json.dumps([{"id": "case-1"}]), encoding="utf-8")
    task = SpreadsheetBenchIdSplitDataset(data_root=data_root, split_dir=split_root).train_tasks()[0]

    trajectory = await SpreadsheetBenchEnv({}).rollout(
        SpreadsheetScriptAgent(),  # type: ignore[arg-type]
        task,
        [make_skill()],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="sheet-rollout"),
            workspace_root=tmp_path / "runs",
            workspace_scope="train/batch_0",
        ),
    )

    assert trajectory.reward.score == 1.0
    assert trajectory.environment.running_dir is not None
    workspace = Path(trajectory.environment.running_dir)
    assert (workspace / "input.xlsx").exists()
    assert (workspace / "output.xlsx").exists()
    assert workspace == tmp_path / "runs" / "train" / "batch_0" / "case-1" / "sheet-rollout" / "0"


def test_source_prompt_assets_and_dynamic_builders_are_exact() -> None:
    expected_hashes = {
        EXPERIENCE_EXTRACTION_SYSTEM: "672b087af6c3e5d2891a6370f081157d42263653778289a0394452323efbc8d1",
        EXPERIENCE_PATCH_SYSTEM: "02c457eb6af6835baaaa49ea8c5f163bb2d58b3bf7584efb068eab618c368901",
        FUSION_SYSTEM: "8c8bac16ef539fe346bdf42ff4ebf3fd80cdfe6caa69da27f91d895d9ddaa23d",
        SPREADSHEET_SYSTEM: "d51300508e68fc8257be6349a25523f208ad4d3df2a57dd38cc89aade68c4e6e",
        LIVEMATH_SYSTEM: "243ce14b9c9432e6012c3b4b5f9489b5483f67a0df016faa4b80e937c3205e9e",
    }
    for content, expected in expected_hashes.items():
        assert hashlib.sha256(content.encode()).hexdigest() == expected

    skill = make_skill("# Current Guidance\n\n- Use tools carefully.")
    task = Task(task_id="task-a", instruction="instruction-task-a")
    now = datetime.now(UTC)
    trajectory = Trajectory(
        trajectory_id="prompt-trajectory",
        task=task,
        rollout=Rollout(rollout_id="prompt-rollout"),
        injected_skills=[skill],
        agent=AgentProfile(),
        events=[
            {
                "role": "assistant",
                "content": "alpha",
                "tool_calls": [
                    {
                        "function": {
                            "name": "example_tool",
                            "arguments": {"unabridged": "alphaalpha"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "observed-alpha"},
            {
                "role": "user",
                "content": f"Loaded skill 'demo'.\n----- demo/SKILL.md -----\n{skill.content}",
            },
        ],
        reward=Reward(score=1.0),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )
    extraction_user = experience_extraction_messages(
        task=task,
        skill=skill,
        trajectories=[trajectory],
        max_experiences=3,
    )[1]["content"]
    assert hashlib.sha256(extraction_user.encode()).hexdigest() == (
        "753cdadba528c4f53d142810bb7c344c2f09ae1603fac4b2c49c795f40570123"
    )
    assert "#### [1] ASSISTANT" in extraction_user
    assert "Observation:\n```text\nobserved-alpha\n```" in extraction_user
    assert "[Skill content omitted: identical to the Current Skill shown above.]" in extraction_user

    fusion_user = fusion_messages(
        history_replace=None,
        history_count=0,
        new_replaces=["r1", ""],
        history_centroid_text="",
        new_key_texts=["k1", "k2"],
    )[1]["content"]
    assert hashlib.sha256(fusion_user.encode()).hexdigest() == (
        "c59775a3f0eae4d14c0ed1b471b6a7254479b4d417315424b7de6a939da05d64"
    )
    assert PATCH_REPAIR_USER.format(error="ValueError: bad") == (
        "The previous response violated the patch output contract: ValueError: bad\n"
        "Return one complete corrected JSON object. Every item in `edits` must contain both string fields "
        "`find` and `replace`."
    )


@pytest.mark.asyncio
async def test_replay_capacity_uses_distinct_source_support_and_keeps_touched_results() -> None:
    buffer = FusedReplayBuffer(
        chat_model=FakeChatModel(),
        embedding_model=None,
        config=ReplayBufferConfig(capacity=1, min_cluster_edits=1),
    )
    touched = await buffer.ingest(
        0,
        [
            (SkillTextEdit(find="A", replace="RA"), "task-a", 0.0),
            (SkillTextEdit(find="B", replace="RB"), "task-b", 0.0),
            (SkillTextEdit(find="B", replace="RB"), "task-c", 0.0),
        ],
    )

    assert len(touched) == 2
    assert len(buffer.clusters) == 1
    assert {record.source_task_id for record in buffer.clusters[0].records} == {"task-b", "task-c"}


def test_ablation_ties_and_find_score_ties_preserve_source_arrival_order() -> None:
    evaluator = AblationEvaluator(AblationConfig(positive_only=False, commit_topk=1), success_reward=1.0)
    current = make_skill("# Demo\n\nA\nB")
    touched = TouchedCluster(
        cluster=ReplayClusterState(
            cluster_id="cluster",
            centroid_text="x",
            committed_replace="replacement",
            records=[],
            last_seen_batch=0,
        ),
        find_sources=[("B", "task-z"), ("A", "task-a")],
    )
    assert evaluator._pick_find(touched, "replacement", current.content, {}) == ("B", "task-z")

    first = AblationCandidate("first", "one", SkillTextEdit(find="A", replace="X"), "t", [], current)
    second = AblationCandidate("second", "two", SkillTextEdit(find="B", replace="Y"), "t", [], current)
    records = evaluator.score([first, second], [])
    assert [record.candidate_id for record in records] == ["first", "second"]
    assert records[0].chosen is True
    assert records[1].chosen is False


def test_ablation_sampling_matches_one_stateful_python_rng_and_round_trips() -> None:
    config = AblationConfig(max_source_cases_per_candidate=2, seed=11)
    evaluator = AblationEvaluator(config, success_reward=1.0)
    skill = make_skill("# Demo\n\nA\nB")
    tasks = {task_id: Task(task_id=task_id, instruction=task_id) for task_id in ("a", "b", "c")}

    def cluster(cluster_id: str, find: str, replace: str) -> TouchedCluster:
        edit = SkillTextEdit(find=find, replace=replace)
        return TouchedCluster(
            cluster=ReplayClusterState(
                cluster_id=cluster_id,
                centroid_text=cluster_id,
                committed_replace=replace,
                records=[ReplayEditRecord(edit=edit, batch_index=0, source_task_id=task_id) for task_id in tasks],
                last_seen_batch=0,
            ),
            find_sources=[(find, "a")],
        )

    candidates, counter = evaluator.build_candidates(
        run_id="rng",
        batch_index=0,
        current_skill=skill,
        touched=[cluster("one", "A", "X"), cluster("two", "B", "Y")],
        task_registry=tasks,
        case_total_scores={},
        skill_factory=lambda base, content, tag: base.model_copy(update={"blob": {"SKILL.md": content}}),
        sample_counter=0,
        min_cluster_edits=1,
    )
    expected_rng = random.Random(11)
    expected = [expected_rng.sample(sorted(tasks), 2), expected_rng.sample(sorted(tasks), 2)]
    assert [[task.task_id for task in candidate.sampled_tasks] for candidate in candidates] == expected
    assert counter == 2

    restored = AblationEvaluator(config, success_reward=1.0)
    restored.load_random_state(evaluator.random_state())
    assert restored._rng.random() == evaluator._rng.random()


def test_multiple_append_edits_keep_source_fileedit_application_order() -> None:
    output, applied = apply_best_effort(
        [
            SkillTextEdit(find="", replace="first"),
            SkillTextEdit(find="", replace="second"),
        ],
        "base",
    )
    assert applied == [
        SkillTextEdit(find="", replace="first"),
        SkillTextEdit(find="", replace="second"),
    ]
    assert output == "base\n\nfirst\n\nsecond"


@pytest.mark.asyncio
async def test_optimizer_chat_content_strips_like_source_call_adapter() -> None:
    class WhitespaceChat:
        async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            del task, messages, kwargs
            return SimpleNamespace(content="  result\n")

    assert await chat_content(WhitespaceChat(), task="test", messages=[]) == "result"


def test_best_effort_reports_applied_edits_in_source_input_order() -> None:
    output, applied = apply_best_effort(
        [
            SkillTextEdit(find="B", replace="second"),
            SkillTextEdit(find="A", replace="first"),
        ],
        "A B",
    )
    assert applied == [
        SkillTextEdit(find="B", replace="second"),
        SkillTextEdit(find="A", replace="first"),
    ]
    assert output == "first second"
