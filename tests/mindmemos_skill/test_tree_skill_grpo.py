from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from mindmemos_skill.algos.evolve.tree_skill_grpo import (
    NewSkillTreeNode,
    SkillTree,
    SkillTreeEdit,
    TreeSkillGrpoAlgorithm,
    TreeSkillGrpoConfig,
)
from mindmemos_skill.algos.evolve.tree_skill_grpo.prompts import (
    REFLECTION_SYSTEM_PROMPT,
    _render_events,
    reflection_messages,
    task_with_reflection,
)
from mindmemos_skill.envs import BaseEnv, EnvRolloutContext, PreparedRollout
from mindmemos_skill.envs.registered_envs.alfworld_bounded_history.env import ALFWorldBoundedHistoryEnv
from mindmemos_skill.envs.registered_envs.livemath.env import LiveMathEnv
from mindmemos_skill.envs.registered_envs.spreadsheetbench.env import SpreadsheetBenchEnv
from mindmemos_skill.infra.database import RecordQuery
from mindmemos_skill.llm import DatabaseLLMCallSink, LLMClient
from mindmemos_skill.persistence import LLM_CALL_TABLE, LLMCallRecord, bootstrap_skill_database, from_database_record
from mindmemos_skill.registry import ComponentType, register
from mindmemos_skill.typing import (
    AgentProfile,
    EnvConfig,
    Environment,
    EvolveInput,
    ExecutionInfo,
    Reward,
    Rollout,
    Skill,
    Task,
    Trajectory,
    TrajectoryStatus,
    compute_skill_content_hash,
)


class ScriptedEnvConfig(EnvConfig):
    pass


@register(type=ComponentType.ENV, name="tree_skill_grpo_test")
class ScriptedEnv(BaseEnv[ScriptedEnvConfig]):
    config_type = ScriptedEnvConfig
    scores: ClassVar[dict[str, list[float]]] = {}
    calls: ClassVar[list[dict[str, Any]]] = []

    async def rollout(self, agent, task, skills, *, context: EnvRolloutContext) -> Trajectory:
        del agent
        attempt = context.rollout.attempt_no
        score = self.scores[task.task_id][attempt]
        self.calls.append(
            {
                "task_id": task.task_id,
                "attempt": attempt,
                "system_prompt": task.system_prompt,
                "skill": skills[0].content,
            }
        )
        now = datetime.now(UTC)
        return Trajectory(
            trajectory_id=f"{context.rollout.rollout_id}:{attempt}",
            task=task,
            rollout=context.rollout,
            environment=Environment(env_ref=context.env_ref),
            agent=AgentProfile(),
            injected_skills=list(skills),
            events=[
                {"role": "system", "content": task.system_prompt or ""},
                {"role": "assistant", "content": f"answer-{task.task_id}-{attempt}"},
            ],
            reward=Reward(score=score, detail=None if score else "incorrect"),
            execution=ExecutionInfo(
                status=TrajectoryStatus.SUCCEEDED,
                started_at=now,
                finished_at=now,
                n_turn=1,
            ),
        )

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        raise AssertionError((trajectory, prepared))


class ScriptedChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        self.calls.append((task, messages))
        if ".reflection." in task:
            return SimpleNamespace(content="Do not repeat the failed approach; verify the decisive step.")
        node_id_match = re.search(r'"node_id": "(node-[0-9a-f]+)"', messages[1]["content"])
        assert node_id_match is not None
        node_id = node_id_match.group(1)
        if ".fusion." in task:
            content = "Fused batch one guidance." if "batch-1." in task else "Fused batch two guidance."
            return SimpleNamespace(content=f'{{"content":"{content}","children":[]}}')
        find = "Initial guidance." if "batch-1." in task else "Fused batch one guidance."
        return SimpleNamespace(
            content=(
                f'{{"operation":"edit_node","target_node_id":"{node_id}",'
                f'"find":"{find}","replace":"Task-local proposal.",'
                '"reason":"task-local causal evidence"}'
            )
        )


class InvalidStageChatModel:
    def __init__(self, invalid_stage: str) -> None:
        self.invalid_stage = invalid_stage

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        if ".reflection." in task:
            return SimpleNamespace(content="Try a corrected approach.")
        if self.invalid_stage in task:
            return SimpleNamespace(content="not valid JSON")
        node_id_match = re.search(r'"node_id": "(node-[0-9a-f]+)"', messages[1]["content"])
        assert node_id_match is not None
        return SimpleNamespace(
            content=(
                f'{{"operation":"edit_node","target_node_id":"{node_id_match.group(1)}",'
                '"find":"Initial guidance.","replace":"valid proposal"}'
            )
        )


def make_skill() -> Skill:
    content = "# Root\n\nInitial guidance.\n\n## Child\n\nKeep child.\n"
    blob = {"SKILL.md": content}
    return Skill(
        skill_id="tree-skill",
        version_id="tree-skill-v1",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash(blob),
        name="tree-skill",
        blob=blob,
        created_at=datetime.now(UTC),
    )


def test_render_events_truncates_each_long_message_in_the_middle() -> None:
    now = datetime.now(UTC)
    trajectory = Trajectory(
        trajectory_id="render-events",
        task=Task(task_id="render-events", instruction="render"),
        rollout=Rollout(rollout_id="render-events"),
        events=[
            {"role": "system", "content": "excluded system"},
            {"role": "user", "content": "A" * 1_100 + "M" * 3_000 + "Z" * 1_100},
            {"role": "assistant", "content": "short response"},
        ],
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )

    rendered = _render_events(trajectory)

    assert rendered.count("## Message ") == 2
    assert "excluded system" not in rendered
    assert "A" * 100 in rendered
    assert "Z" * 100 in rendered
    assert "M" * 100 not in rendered
    assert "...[message middle truncated]..." in rendered
    assert "short response" in rendered


def test_reflection_prompt_requires_grounded_diagnosis_and_actionable_verification() -> None:
    now = datetime.now(UTC)
    trajectory = Trajectory(
        trajectory_id="reflection-prompt",
        task=Task(task_id="reflection-prompt", instruction="Complete the original task"),
        rollout=Rollout(rollout_id="reflection-prompt"),
        events=[
            {"role": "assistant", "content": "I assumed the target was ready."},
            {"role": "tool", "content": "Target is not ready."},
        ],
        reward=Reward(score=0.0, detail="incorrect final state"),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )

    messages = reflection_messages(trajectory)

    assert messages[0]["content"] == REFLECTION_SYSTEM_PROMPT
    assert "Do not merely summarize" in messages[0]["content"]
    assert "tool, environment, evaluator, and infrastructure failures" in messages[0]["content"]
    assert "Diagnosis:" in messages[0]["content"]
    assert "Verification:" in messages[0]["content"]
    assert "# Environment" not in messages[1]["content"]
    assert "# Outcome\nfail" in messages[1]["content"]
    assert "incorrect final state" not in messages[1]["content"]
    assert "Scalar reward" not in messages[1]["content"]
    assert "Failure detail" not in messages[1]["content"]
    assert "Target is not ready" in messages[1]["content"]


def test_retry_context_preserves_original_system_prompt_and_treats_reflection_as_fallible() -> None:
    task = Task(
        task_id="retry-prompt",
        instruction="Complete the original task",
        system_prompt="Keep the required environment protocol.",
    )

    reflected = task_with_reflection(
        task,
        previous_answer="previous answer",
        reflection="check target readiness first",
    )

    assert reflected.system_prompt.startswith("Keep the required environment protocol.")
    assert "previous answer" in reflected.system_prompt
    assert "check target readiness first" in reflected.system_prompt
    assert "fallible hypothesis" in reflected.system_prompt
    assert task.system_prompt == "Keep the required environment protocol."


def test_tree_edits_exact_replace_append_and_create_recursive_children() -> None:
    tree = SkillTree.from_markdown("# Root\n\nold\n\n## Existing\n\nkeep\n")
    root_id = tree.roots[0].node_id
    existing_id = tree.roots[0].children[0].node_id

    assert tree.apply(
        SkillTreeEdit(operation="edit_node", target_node_id=root_id, find="old", replace="new direct content")
    )
    assert tree.node(root_id).content == "new direct content"
    assert tree.node(existing_id).content == "keep"
    assert tree.apply(SkillTreeEdit(operation="append_node", target_node_id=root_id, content="appended guidance"))
    assert tree.node(root_id).content == "new direct content\n\nappended guidance"

    assert tree.apply(
        SkillTreeEdit(
            operation="create_child",
            target_node_id=root_id,
            child=NewSkillTreeNode(
                heading="Added",
                content="added guidance",
                children=[NewSkillTreeNode(heading="Nested", content="nested guidance")],
            ),
        )
    )
    added = next(child for child in tree.node(root_id).children if child.heading == "Added")
    assert added.level == 2
    assert added.children[0].level == 3
    assert SkillTree.from_markdown(tree.to_markdown()) == tree
    assert not tree.apply(SkillTreeEdit(operation="reject", reason="no evidence"))


def test_tree_edit_requires_find_to_match_exactly_once() -> None:
    tree = SkillTree.from_markdown("# Root\n\nrepeat and repeat\n")

    with pytest.raises(ValueError, match="match exactly once"):
        tree.apply(
            SkillTreeEdit(
                operation="edit_node",
                target_node_id=tree.roots[0].node_id,
                find="repeat",
                replace="replacement",
            )
        )


def test_tree_ids_stay_stable_when_a_child_is_inserted_before_later_nodes() -> None:
    tree = SkillTree.from_markdown("# First\n\nA\n\n## Existing\n\nB\n\n# Later\n\nC\n")
    original_ids = {node.heading: node.node_id for node in [tree.roots[0], *tree.roots[0].children, tree.roots[1]]}

    tree.apply(
        SkillTreeEdit(
            operation="create_child",
            target_node_id=tree.roots[0].node_id,
            child=NewSkillTreeNode(heading="Added", content="D"),
        )
    )
    reparsed = SkillTree.from_markdown(tree.to_markdown())

    assert reparsed == tree
    assert reparsed.roots[0].node_id == original_ids["First"]
    assert reparsed.roots[0].children[0].node_id == original_ids["Existing"]
    assert reparsed.roots[1].node_id == original_ids["Later"]


def test_application_config_constructs_tree_skill_grpo() -> None:
    config = TreeSkillGrpoConfig(env_ref="tree_skill_grpo_test", workspace_root="workspace")
    context = SimpleNamespace(models={"chat": ScriptedChatModel()}, agents={"react": object()})

    algorithm = TreeSkillGrpoAlgorithm(config=config, context=context)

    assert algorithm.algorithm_name == "tree_skill_grpo"


@pytest.mark.asyncio
async def test_evolve_saves_complete_skill_tree_after_each_epoch(tmp_path) -> None:
    ScriptedEnv.scores = {"epoch-task": [1.0]}
    ScriptedEnv.calls = []
    checkpoint_dir = tmp_path / "epochs"
    algorithm = TreeSkillGrpoAlgorithm(
        env_ref="tree_skill_grpo_test",
        chat_model=ScriptedChatModel(),
        agent=object(),
        epochs=2,
        checkpoint_dir=checkpoint_dir,
        max_concurrency=1,
    )

    result = await algorithm.evolve(
        EvolveInput(
            run_id="epoch-checkpoints",
            base_skill=make_skill(),
            train_tasks=[Task(task_id="epoch-task", instruction="succeed")],
        )
    )

    assert len(result.trajectories) == 2
    for epoch in (1, 2):
        epoch_dir = checkpoint_dir / f"epoch_{epoch:04d}"
        assert SkillTree.from_markdown((epoch_dir / "SKILL.md").read_text(encoding="utf-8"))
        payload = json.loads((epoch_dir / "skill_tree.json").read_text(encoding="utf-8"))
        assert payload["roots"][0]["heading"] == "Root"


@pytest.mark.asyncio
async def test_retry_context_is_loaded_by_each_environment_system_prompt(tmp_path) -> None:
    reflection = "# Previous unsuccessful attempt\nUse the corrected strategy."
    skill = make_skill()
    context = EnvRolloutContext(
        rollout=Rollout(rollout_id="reflection-rollout"),
        workspace_root=tmp_path / "workspace",
    )

    alfworld = await ALFWorldBoundedHistoryEnv({})._prepare(
        task=Task(
            task_id="alfworld-reflection",
            instruction="complete the task",
            system_prompt=reflection,
            metadata={"resolved_gamefile": "/tmp/game.tw-pddl"},
        ),
        skills=[skill],
        context=context,
    )
    assert alfworld.runtime_state["system"].endswith(reflection)

    livemath = await LiveMathEnv({})._prepare(
        task=Task(
            task_id="livemath-reflection",
            instruction="choose",
            system_prompt=reflection,
            metadata={
                "question": "Which?",
                "choices": [{"label": "A", "text": "A"}],
                "correct_choice": {"label": "A", "text": "A"},
            },
        ),
        skills=[skill],
        context=context,
    )
    assert livemath.runtime_state["system"].endswith(reflection)

    source_dir = tmp_path / "spreadsheet-source"
    source_dir.mkdir()
    (source_dir / "case_init.xlsx").write_bytes(b"init")
    (source_dir / "case_golden.xlsx").write_bytes(b"golden")
    spreadsheet = await SpreadsheetBenchEnv({})._prepare(
        task=Task(
            task_id="spreadsheet-reflection",
            instruction="edit workbook",
            system_prompt=reflection,
            metadata={"src_dir": str(source_dir)},
        ),
        skills=[skill],
        context=context,
    )
    spreadsheet_system = spreadsheet.runtime_state["messages"][0]["content"]
    assert "A `skill` tool is available with expert skills: tree-skill." in spreadsheet_system
    assert 'Call it first (e.g. skill(name="tree-skill"))' in spreadsheet_system
    assert spreadsheet_system.endswith(reflection)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_stage", [".attribution.", ".fusion."])
async def test_invalid_optimizer_json_is_dropped_without_failing_the_batch(invalid_stage: str) -> None:
    ScriptedEnv.scores = {"invalid-output": [0.0, 1.0]}
    ScriptedEnv.calls = []
    algorithm = TreeSkillGrpoAlgorithm(
        env_ref="tree_skill_grpo_test",
        chat_model=InvalidStageChatModel(invalid_stage),
        agent=object(),
    )

    result = await algorithm.evolve(
        EvolveInput(
            run_id=f"invalid-{invalid_stage}",
            base_skill=make_skill(),
            train_tasks=[Task(task_id="invalid-output", instruction="recover")],
        )
    )

    assert len(result.trajectories) == 2
    assert result.changed is False
    assert result.final_skill.content == SkillTree.from_markdown(make_skill().content).to_markdown()


@pytest.mark.asyncio
async def test_evolve_retries_with_system_reflection_and_updates_before_next_batch() -> None:
    ScriptedEnv.scores = {
        "success-first": [1.0],
        "failure-a": [0.0, 1.0],
        "failure-b": [0.0, 0.0, 1.0],
        "always-fails": [0.0, 0.0, 0.0, 0.0],
    }
    ScriptedEnv.calls = []
    chat_model = ScriptedChatModel()
    algorithm = TreeSkillGrpoAlgorithm(
        env_ref="tree_skill_grpo_test",
        chat_model=chat_model,
        agent=object(),
        batch_size=3,
        max_task_attempts=4,
    )
    request = EvolveInput(
        run_id="tree-run",
        base_skill=make_skill(),
        train_tasks=[
            Task(task_id="success-first", instruction="first"),
            Task(task_id="failure-a", instruction="second"),
            Task(task_id="failure-b", instruction="third"),
            Task(task_id="always-fails", instruction="fourth"),
        ],
    )

    result = await algorithm.evolve(request)

    attempts = {
        task_id: [call for call in ScriptedEnv.calls if call["task_id"] == task_id] for task_id in ScriptedEnv.scores
    }
    assert len(attempts["success-first"]) == 1
    assert len(attempts["failure-a"]) == 2
    assert len(attempts["failure-b"]) == 3
    assert len(attempts["always-fails"]) == 4
    for task_id in ("failure-a", "failure-b"):
        assert attempts[task_id][0]["system_prompt"] is None
        for retry in attempts[task_id][1:]:
            assert retry["system_prompt"].count("# Previous unsuccessful attempt") == 1
            assert "Do not repeat the failed approach" in retry["system_prompt"]
    assert "Fused batch one guidance." in attempts["always-fails"][0]["skill"]
    assert len([task for task, _messages in chat_model.calls if ".reflection." in task]) == 6
    attribution_calls = [(task, messages) for task, messages in chat_model.calls if ".attribution." in task]
    assert len(attribution_calls) == 3
    first_attribution = attribution_calls[0][1][1]["content"]
    assert "task_id=failure-a" in first_attribution
    assert "task_id=success-first" not in first_attribution
    assert "task_id=failure-b" not in first_attribution
    assert "task_id=always-fails" not in first_attribution
    assert "success" in first_attribution
    assert "failure" in first_attribution
    second_attribution = attribution_calls[1][1][1]["content"]
    assert "task_id=failure-b" in second_attribution
    assert "Fused batch one guidance." not in first_attribution
    assert "Fused batch one guidance." not in second_attribution
    fusion_calls = [(task, messages) for task, messages in chat_model.calls if ".fusion." in task]
    assert len(fusion_calls) == 2
    first_fusion = fusion_calls[0][1][1]["content"]
    assert first_fusion.count('"content": "Initial guidance."') == 1
    assert '"find": "Initial guidance."' in first_fusion
    assert '"replace": "Task-local proposal."' in first_fusion
    assert "task_id" not in first_fusion
    assert "base_tree_hash" not in first_fusion
    assert "outcome_pattern" not in first_fusion
    assert "Keep child." not in first_fusion
    assert len(result.trajectories) == 10
    assert SkillTree.from_markdown(result.final_skill.content).roots[0].content == "Fused batch two guidance."


@pytest.mark.asyncio
async def test_evolve_persists_all_optimizer_llm_calls_under_run_id(tmp_path) -> None:
    root_node_id = SkillTree.from_markdown(make_skill().content).roots[0].node_id

    class ScriptedRouter:
        def __init__(self) -> None:
            self.responses = [
                "Diagnose the failure and verify the corrected step.",
                (
                    f'{{"operation":"edit_node","target_node_id":"{root_node_id}",'
                    '"find":"Initial guidance.","replace":"Task-local proposal.",'
                    '"reason":"task-local causal evidence"}'
                ),
                '{"content":"Fused guidance.","children":[]}',
            ]

        async def acompletion(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.responses.pop(0)), finish_reason="stop")],
                model="chat-model",
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )

    database = await bootstrap_skill_database(tmp_path / "state.db")
    try:
        ScriptedEnv.scores = {"sqlite-run": [0.0, 1.0]}
        ScriptedEnv.calls = []
        chat_model = LLMClient(ScriptedRouter(), call_sink=DatabaseLLMCallSink(database))
        algorithm = TreeSkillGrpoAlgorithm(
            env_ref="tree_skill_grpo_test",
            chat_model=chat_model,
            agent=object(),
            batch_size=1,
            max_task_attempts=2,
        )

        await algorithm.evolve(
            EvolveInput(
                run_id="tree-sqlite-run",
                base_skill=make_skill(),
                train_tasks=[Task(task_id="sqlite-run", instruction="retry once")],
            )
        )

        rows, cursor = await database.query_records(LLM_CALL_TABLE, RecordQuery())
        records = [from_database_record(row, LLMCallRecord) for row in rows]
        assert cursor is None
        assert len(records) == 3
        assert {record.run_id for record in records} == {"tree-sqlite-run"}
        assert {record.task for record in records} == {
            "tree_skill_grpo.reflection.batch-1.sqlite-run.attempt-1",
            "tree_skill_grpo.attribution.tree-sqlite-run.batch-1.task-1",
            f"tree_skill_grpo.fusion.tree-sqlite-run.batch-1.{root_node_id}",
        }
    finally:
        await database.close()
