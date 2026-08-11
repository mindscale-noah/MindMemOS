from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any

import pytest
from mindmemos_skill.agents import Agent, AgentConfig, AgentExecutionRequest, SkillInjection
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.prompts import experience_extraction_messages
from mindmemos_skill.datasets import LiveMathIdSplitDataset
from mindmemos_skill.envs import ALFWorldBoundedHistoryEnv, ALFWorldEnv, EnvRolloutContext, LiveMathEnv
from mindmemos_skill.envs.registered_envs.alfworld import (
    SYSTEM_PROMPT as ALFWORLD_SYSTEM_PROMPT,
)
from mindmemos_skill.envs.registered_envs.alfworld import format_observation
from mindmemos_skill.envs.registered_envs.alfworld.env import ALFWorldEnvConfig
from mindmemos_skill.envs.registered_envs.alfworld_bounded_history import (
    ALFWORLD_SYSTEM_PROMPT as BOUNDED_HISTORY_ALFWORLD_SYSTEM_PROMPT,
)
from mindmemos_skill.envs.registered_envs.alfworld_bounded_history import (
    format_bounded_history_observation,
)
from mindmemos_skill.envs.registered_envs.alfworld_bounded_history.env import ALFWorldBoundedHistoryEnvConfig
from mindmemos_skill.envs.registered_envs.livemath import build_system, build_user, evaluate, refinement
from mindmemos_skill.envs.registered_envs.livemath.env import LiveMathEnvConfig
from mindmemos_skill.envs.registered_envs.spreadsheetbench.env import SpreadsheetBenchEnvConfig
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.typing import (
    Rollout,
    Skill,
    SkillBinding,
    SkillInjectionMode,
    SkillUsageType,
    Task,
    Trajectory,
)


class ScriptedMessageAgent(Agent[AgentConfig]):
    def __init__(self, responses: list[str]) -> None:
        super().__init__({})
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def inject_skills(self, skills, *, mode: SkillInjectionMode | None = None):
        del skills
        return nullcontext(SkillInjection(mode=mode or SkillInjectionMode.SYSTEM_PROMPT))

    def bind_skills(self, trajectory: Trajectory) -> list[SkillBinding]:
        del trajectory
        return []

    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        raise AssertionError(f"benchmark env must use respond(), not execute(): {request.task.task_id}")

    async def respond(
        self,
        request: AgentExecutionRequest,
        messages: list[dict[str, Any]],
        *,
        tools=(),
    ) -> ChatResponse:
        del request
        self.calls.append((list(messages), None if tools is None else list(tools)))
        return ChatResponse(finish_reason="stop", content=self.responses.pop(0), model="fake")


class RecordingChatClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], *, model=None, **kwargs):
        self.calls.append({"task": task, "messages": messages, "model": model, **kwargs})
        return ChatResponse(finish_reason="stop", content=self.content, model="fake")


def test_builtin_env_configs_only_accept_max_turns() -> None:
    config_types = (ALFWorldEnvConfig, ALFWorldBoundedHistoryEnvConfig, LiveMathEnvConfig, SpreadsheetBenchEnvConfig)

    for config_type in config_types:
        assert "max_turns" in config_type.model_fields
        assert "max_steps" not in config_type.model_fields

    with pytest.raises(ValueError, match="max_steps"):
        ALFWorldEnvConfig.model_validate({"max_steps": 3})
    with pytest.raises(ValueError, match="max_steps"):
        ALFWorldBoundedHistoryEnvConfig.model_validate({"max_steps": 3})


def test_alfworld_bounded_history_prompt_keeps_only_two_recent_steps() -> None:
    prompt = format_bounded_history_observation(
        current_observation="current observation",
        admissible_actions=["help", "look"],
        task_description="put the mug in the cabinet",
        history=[("old observation", "old action"), ("recent one", "action one"), ("recent two", "action two")],
    )

    assert "already taken 3 step(s)" in prompt
    assert "most recent 2 observations" in prompt
    assert "old observation" not in prompt
    assert "old action" not in prompt
    assert "recent one" in prompt
    assert "recent two" in prompt
    assert "'help'" not in prompt


def make_skill(name: str = "main", content: str = "Compare every option.") -> Skill:
    return Skill(
        skill_id=f"skill-{name}",
        version_id=f"version-{name}",
        version_label="1.0.0",
        content_hash=f"sha256:{name}",
        name=name,
        blob={"SKILL.md": content},
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_livemath_messages_and_reward_match_skill_grpo(tmp_path) -> None:
    choices = [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}]
    item = {
        "id": "202601:1",
        "question": "Which option is exact?",
        "choices": choices,
        "correct_choice": {"label": "B", "text": "second"},
        "theorem": "",
        "sketch": "",
    }
    task = Task(task_id=item["id"], instruction="dataset text", metadata=item)
    skill = make_skill()
    agent = ScriptedMessageAgent(["I am unsure.", "<answer>B</answer>"])
    env = LiveMathEnv({"max_turns": 2})

    trajectory = await env.rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="rollout-live"),
            workspace_root=tmp_path,
        ),
    )

    system = build_system("Compare every option.")
    user = build_user(item, False, False)
    assert agent.calls == [
        ([{"role": "system", "content": system}, {"role": "user", "content": user}], []),
        (
            [
                {"role": "system", "content": system},
                {"role": "user", "content": refinement("I am unsure.")},
            ],
            [],
        ),
    ]
    assert trajectory.events == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "I am unsure."},
        {"role": "user", "content": refinement("I am unsure.")},
        {"role": "assistant", "content": "<answer>B</answer>"},
    ]
    assert trajectory.reward.score == 1.0
    assert trajectory.reward.metadata == evaluate("<answer>B</answer>", item["correct_choice"], choices)
    assert trajectory.metadata["conversation"] == [
        {"type": "message", "turn": 1, "content": "I am unsure."},
        {"type": "message", "turn": 2, "content": "<answer>B</answer>"},
    ]

    workspace = tmp_path / "rollout" / "202601_1" / "rollout-live" / "0"
    assert (workspace / "target_system_prompt.txt").read_text(encoding="utf-8") == system
    assert (workspace / "target_user_prompt.txt").read_text(encoding="utf-8") == user
    prediction = json.loads((workspace / "prediction" / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["hard"] == 1
    assert prediction["soft"] == 1.0


def test_livemath_dataset_normalization_and_choice_shuffle_match_source(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "data" / "202601"
    raw_dir.mkdir(parents=True)
    (raw_dir / "qa_202601_final.json").write_text(
        json.dumps(
            [
                {
                    "month": "202601",
                    "no": 1,
                    "mcq": {
                        "question": "Which option is correct?",
                        "choices": [
                            {"label": "A", "text": "wrong"},
                            {"label": "B", "text": "right"},
                        ],
                        "correct_choice": {"label": "B", "text": "right"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    split_dir = tmp_path / "split" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "items.json").write_text(
        json.dumps([{"id": "202601:1", "source_file": "data/202601/qa_202601_final.json"}]),
        encoding="utf-8",
    )

    task = LiveMathIdSplitDataset(
        data_path=tmp_path / "raw",
        split_dir=tmp_path / "split",
        seed=42,
    ).train_tasks()[0]

    assert task.task_id == "202601:1"
    assert task.metadata["correct_choice"]["text"] == "right"
    assert task.metadata["correct_choice"]["label"] in {"A", "B"}
    assert "## Question\nWhich option is correct?" in task.instruction


@pytest.mark.asyncio
async def test_livemath_uses_react_message_interface_without_reformatting_prompt() -> None:
    item = {
        "id": "live:react",
        "question": "Choose.",
        "choices": [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}],
        "correct_choice": {"label": "A", "text": "yes"},
    }
    task = Task(task_id=item["id"], instruction="ignored", metadata=item)
    skill = make_skill()
    llm = RecordingChatClient("<answer>A</answer>")
    agent = ReactAgent({"model": "test-model", "temperature": 0.2}, llm=llm)

    trajectory = await LiveMathEnv({}).rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(rollout=Rollout(rollout_id="react-rollout")),
    )

    assert llm.calls == [
        {
            "task": "live:react",
            "messages": [
                {"role": "system", "content": build_system(skill.content)},
                {"role": "user", "content": build_user(item, False, False)},
            ],
            "model": "test-model",
            "temperature": 0.2,
            "tools": [],
        }
    ]
    assert trajectory.agent.skill_injection_mode is SkillInjectionMode.SYSTEM_PROMPT
    assert trajectory.skill_bindings[0].usage is SkillUsageType.INJECTED
    assert trajectory.reward.score == 1.0


class FakeALFWorldSimulator:
    def __init__(self) -> None:
        self.admissible_actions = ["help", "look", "go to cabinet 1"]
        self.responses: list[str] = []
        self.closed = False

    def reset(self):
        return "Welcome. Your task is to: put the mug in the cabinet.", {}

    def step(self, model_response: str):
        self.responses.append(model_response)
        if len(self.responses) == 1:
            self.admissible_actions = ["look", "open cabinet 1"]
            return "You see a closed cabinet.", 0.0, False, {"won": False, "is_action_valid": 1}
        return "You won the game!", 10.0, True, {"won": True, "is_action_valid": 1}

    def close(self) -> None:
        self.closed = True


class FakeALFWorldEnv(ALFWorldEnv):
    def __init__(self, config, simulator: FakeALFWorldSimulator) -> None:
        super().__init__(config)
        self.simulator = simulator
        self.build_args: tuple[Task, int] | None = None

    def _build_simulator(self, task: Task, sample_index: int):
        self.build_args = (task, sample_index)
        return self.simulator


class FakeALFWorldBoundedHistoryEnv(ALFWorldBoundedHistoryEnv):
    def __init__(self, config, simulator: FakeALFWorldSimulator) -> None:
        super().__init__(config)
        self.simulator = simulator
        self.build_args: tuple[Task, int] | None = None

    def _build_simulator(self, task: Task, sample_index: int):
        self.build_args = (task, sample_index)
        return self.simulator


@pytest.mark.asyncio
async def test_alfworld_is_lean_history_and_preserves_step_and_final_rewards(tmp_path) -> None:
    simulator = FakeALFWorldSimulator()
    env = FakeALFWorldEnv({"max_turns": 3, "seed": 42}, simulator)
    agent = ScriptedMessageAgent(
        [
            "I forgot the tags.",
            "<think>open it now</think><action>open cabinet 1</action>",
        ]
    )
    task = Task(
        task_id="valid_seen:1",
        instruction="Complete the ALFWorld task.",
        system_prompt=ALFWORLD_SYSTEM_PROMPT,
        tags=["validation"],
        metadata={
            "gamefile": "/json_2.1.1/valid_seen/task/game.tw-pddl",
            "resolved_gamefile": "/data/task/game.tw-pddl",
            "task_type": "pick_and_place",
        },
    )
    skill = make_skill("route", "Open closed receptacles before placing objects.")

    trajectory = await env.rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="rollout-alf"),
            workspace_root=tmp_path,
            metadata={"sample_index": 3},
        ),
    )

    system = trajectory.events[0]["content"]
    assert system.startswith(ALFWORLD_SYSTEM_PROMPT)
    assert "## Skill Knowledge" in system
    assert "### Skill: 000_route" in system
    first_user = format_observation(
        "Welcome. Your task is to: put the mug in the cabinet.",
        ["help", "look", "go to cabinet 1"],
    )
    second_user = format_observation(
        "You see a closed cabinet.",
        ["look", "open cabinet 1"],
    )
    fallback = "<think>missing action tag</think><action>look</action>"
    assert agent.calls[0] == ([{"role": "system", "content": system}, {"role": "user", "content": first_user}], [])
    assert agent.calls[1][0] == [
        {"role": "system", "content": system},
        {"role": "user", "content": first_user},
        {"role": "assistant", "content": fallback},
        {"role": "user", "content": second_user},
    ]
    assert trajectory.events[-1] == {
        "role": "assistant",
        "content": "<think>open it now</think><action>open cabinet 1</action>",
    }
    assert simulator.responses == [fallback, "<think>open it now</think><action>open cabinet 1</action>"]
    assert simulator.closed is True
    assert env.build_args == (task, 3)
    assert trajectory.metadata["conversation"][0]["reward"] == 0.0
    assert trajectory.metadata["conversation"][1]["reward"] == 10.0
    assert trajectory.metadata["invalid_actions"] == 1
    assert trajectory.reward.score == 1.0
    assert trajectory.reward.metadata["won"] is True


@pytest.mark.asyncio
async def test_alfworld_bounded_history_matches_agent_inputs_and_extraction_trajectory(tmp_path) -> None:
    simulator = FakeALFWorldSimulator()
    env = FakeALFWorldBoundedHistoryEnv({"max_turns": 3, "seed": 42}, simulator)
    agent = ScriptedMessageAgent(
        [
            "<think>inspect first</think><action>look</action>",
            "<think>open it now</think><action>open cabinet 1</action>",
        ]
    )
    task = Task(
        task_id="valid_seen:bounded-history",
        instruction="Complete the ALFWorld task.",
        tags=["validation"],
        metadata={
            "gamefile": "/json_2.1.1/valid_seen/task/game.tw-pddl",
            "resolved_gamefile": "/data/task/game.tw-pddl",
            "task_type": "pick_and_place",
        },
    )
    skill = make_skill("route", "Open closed receptacles before placing objects.")

    trajectory = await env.rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="rollout-bounded-history"),
            workspace_root=tmp_path,
            metadata={"sample_index": 3},
        ),
    )

    first_user = """

## Skill Knowledge
Below is a skill document with learned strategies. Use these guidelines to inform your decisions:

Open closed receptacles before placing objects.



You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: Welcome. Your task is to: put the mug in the cabinet.
Your admissible actions of the current situation are: ['look'
 'go to cabinet 1'].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
    second_user = """

## Skill Knowledge
Below is a skill document with learned strategies. Use these guidelines to inform your decisions:

Open closed receptacles before placing objects.



You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: put the mug in the cabinet.
Prior to this step, you have already taken 1 step(s). Below are the most recent 1 observations and the corresponding actions you took: [Observation 1: 'Welcome. Your task is to: put the mug in the cabinet.', Action 1: 'look']
You are now at step 2 and your current observation is: You see a closed cabinet.
Your admissible actions of the current situation are: ['look'
 'open cabinet 1'].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
    assert agent.calls == [
        (
            [
                {"role": "system", "content": BOUNDED_HISTORY_ALFWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": first_user},
            ],
            None,
        ),
        (
            [
                {"role": "system", "content": BOUNDED_HISTORY_ALFWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": second_user},
            ],
            None,
        ),
    ]
    assert trajectory.events == [
        {
            "step": 0,
            "action": "look",
            "reasoning": "inspect first",
            "model_response": "<think>inspect first</think><action>look</action>",
            "env_feedback": "You see a closed cabinet.",
            "reward": 0.0,
            "done": False,
        },
        {
            "step": 1,
            "action": "open cabinet 1",
            "reasoning": "open it now",
            "model_response": "<think>open it now</think><action>open cabinet 1</action>",
            "env_feedback": "You won the game!",
            "reward": 10.0,
            "done": True,
        },
    ]
    assert all("role" not in event for event in trajectory.events)
    assert trajectory.reward.score == 1.0
    assert env.build_args == (task, 3)

    extraction_user = experience_extraction_messages(
        task=task,
        skill=skill,
        trajectories=[trajectory],
        max_experiences=3,
    )[1]["content"]
    assert BOUNDED_HISTORY_ALFWORLD_SYSTEM_PROMPT not in extraction_user
    assert "[step 0 think] inspect first" in extraction_user
    assert "[step 0 action] look" in extraction_user
    assert "[step 0 obs]    You see a closed cabinet." in extraction_user
    assert ("[step 0 obs]    You see a closed cabinet.\n[step 1 think] open it now") in extraction_user
    assert "#### [1] SYSTEM" not in extraction_user

    workspace = tmp_path / "rollout" / "valid_seen_bounded-history" / "rollout-bounded-history" / "0"
    saved = json.loads((workspace / "prediction" / "conversation.json").read_text(encoding="utf-8"))
    assert saved == trajectory.events
