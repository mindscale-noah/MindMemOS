"""ALFWorld interaction protocol matching SkillOpt's target-agent inputs."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from ....agents.base import Agent
from ....registry import ComponentType, register
from ....typing import EnvConfig, Reward, Skill, Task, Trajectory
from ...base import BaseEnv, EnvRolloutContext, PreparedRollout
from ..alfworld.runtime import ALFWorldSimulator, project_action

ALFWORLD_SYSTEM_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."

# These strings intentionally preserve the leading and trailing newlines from
# SkillOpt's skillopt/envs/alfworld/prompts/rollout_*.md files.
_NO_HISTORY_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

_WITH_HISTORY_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

_HISTORY_LENGTH = 2


class ALFWorldSkillOptEnvConfig(EnvConfig):
    """The fixed SkillOpt protocol plus benchmark execution limits."""

    max_steps: int = Field(default=50, ge=1)
    seed: int = 42


@register(type=ComponentType.ENV, name="alfworld_skillopt")
class ALFWorldSkillOptEnv(BaseEnv[ALFWorldSkillOptEnvConfig]):
    """Send the same stateless two-message request used by SkillOpt each step."""

    config_type = ALFWorldSkillOptEnvConfig

    async def _prepare(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: EnvRolloutContext,
    ) -> PreparedRollout:
        if len(skills) > 1:
            raise ValueError("alfworld_skillopt accepts at most one Skill, matching SkillOpt's single skill document")
        prepared = await super()._prepare(task=task, skills=skills, context=context)
        prepared.agent_request.options["skill_injection_mode"] = "system_prompt"
        gamefile = task.metadata.get("resolved_gamefile")
        if not isinstance(gamefile, str) or not gamefile:
            raise ValueError(f"Task {task.task_id} missing resolved_gamefile in metadata")
        sample_index = context.metadata.get("sample_index", context.rollout.attempt_no)
        if not isinstance(sample_index, int):
            raise TypeError("ALFWorld context metadata sample_index must be an integer")
        prepared.runtime_state = {
            "sample_index": sample_index,
            "simulator": None,
            "skill_content": skills[0].content if skills else "",
            "won": False,
            "turns": 0,
            "invalid_actions": 0,
            "error": None,
            "conversation": [],
        }
        return prepared

    async def _execute(self, *, agent: Agent[Any], prepared: PreparedRollout) -> Trajectory:
        state = prepared.runtime_state
        task = prepared.agent_request.task
        conversation: list[dict[str, Any]] = []
        history: list[tuple[str, str]] = []
        error: str | None = None
        won = False
        turns = 0
        invalid_actions = 0
        started = time.time()

        try:
            simulator = await asyncio.to_thread(self._build_simulator, task, state["sample_index"])
            state["simulator"] = simulator
            current_observation, _info = await asyncio.to_thread(simulator.reset)
            task_description = extract_task_description(current_observation)

            for step_index in range(self.config.max_steps):
                observation_prompt = format_skillopt_observation(
                    current_observation=current_observation,
                    admissible_actions=simulator.admissible_actions,
                    task_description=task_description,
                    history=history,
                )
                user_prompt = build_skillopt_user_prompt(state["skill_content"], observation_prompt)
                messages = [
                    {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    response = await agent.respond(prepared.agent_request, messages, tools=None)
                    assistant_text = response.content.strip()
                    if not assistant_text:
                        assistant_text = "<think>empty model response</think><action>look</action>"
                    if extract_action(assistant_text) is None:
                        assistant_text = "<think>missing action tag</think><action>look</action>"
                except Exception:
                    assistant_text = "<think>error</think><action>look</action>"

                projected_action, valid = project_action(assistant_text)
                invalid_actions += int(not valid)
                raw_feedback, step_reward, done, info = await asyncio.to_thread(simulator.step, assistant_text)
                conversation.append(
                    {
                        "step": step_index,
                        "action": extract_action(assistant_text),
                        "reasoning": extract_think(assistant_text),
                        "model_response": assistant_text,
                        "env_feedback": raw_feedback,
                        "reward": float(step_reward),
                        "done": bool(done),
                    }
                )
                history.append((current_observation, projected_action))
                current_observation = raw_feedback
                turns = step_index + 1
                if done:
                    won = bool(info.get("won", False))
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            simulator = state.get("simulator")
            if simulator is not None:
                await asyncio.to_thread(simulator.close)
                state["simulator"] = None

        if not won and error is None and turns >= self.config.max_steps:
            error = f"Timeout after {self.config.max_steps} steps"
        elif not won and error is None:
            error = "Episode ended without completing the task"
        ended = time.time()
        state.update(
            {
                "won": won,
                "turns": turns,
                "invalid_actions": invalid_actions,
                "error": error,
                "conversation": conversation,
            }
        )
        self._write_artifacts(prepared, conversation)
        return agent.build_trajectory(
            request=prepared.agent_request,
            messages=conversation,
            started_at=started,
            ended_at=ended,
            n_turn=turns,
            is_success=error is None,
            error_info=error,
            metadata={
                "finished": won,
                "turns": turns,
                "invalid_actions": invalid_actions,
                "error": error,
                "started_at": started,
                "ended_at": ended,
                "sample_index": state["sample_index"],
                "conversation": conversation,
                "workspace_scope": prepared.environment.metadata["workspace_scope"],
            },
        )

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        del trajectory
        state = prepared.runtime_state
        return Reward(
            score=1.0 if state["won"] else 0.0,
            detail=state["error"],
            metadata={
                "won": state["won"],
                "turns": state["turns"],
                "invalid_actions": state["invalid_actions"],
            },
        )

    def _build_simulator(self, task: Task, sample_index: int) -> ALFWorldSimulator:
        return ALFWorldSimulator(
            seed=self.config.seed + sample_index,
            is_train="train" in task.tags,
            eval_dataset=eval_dataset(task),
            gamefile=str(task.metadata["resolved_gamefile"]),
        )

    @staticmethod
    def _write_artifacts(prepared: PreparedRollout, conversation: list[dict[str, Any]]) -> None:
        running_dir = prepared.environment.running_dir
        if running_dir is None:
            return
        prediction_dir = Path(running_dir) / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(conversation, ensure_ascii=False, indent=2) + "\n"
        (prediction_dir / "conversation.json").write_text(payload, encoding="utf-8")


def build_skillopt_user_prompt(skill_content: str, observation_prompt: str) -> str:
    """Apply SkillOpt's exact Skill prefix construction."""

    if not skill_content or not skill_content.strip():
        return observation_prompt
    skill_prompt = (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )
    return skill_prompt + "\n" + observation_prompt


def format_skillopt_observation(
    *,
    current_observation: str,
    admissible_actions: list[str],
    task_description: str,
    history: list[tuple[str, str]],
) -> str:
    """Render SkillOpt's no-history or two-step-history user prompt."""

    formatted_actions = "\n ".join(f"'{action}'" for action in admissible_actions if action != "help")
    if not history:
        return _NO_HISTORY_TEMPLATE.format(
            current_observation=current_observation,
            admissible_actions=formatted_actions,
        )
    recent = history[-_HISTORY_LENGTH:]
    start_index = len(history) - len(recent)
    action_history = "\n".join(
        f"[Observation {start_index + offset + 1}: '{observation}', "
        f"Action {start_index + offset + 1}: '{action}']"
        for offset, (observation, action) in enumerate(recent)
    )
    return _WITH_HISTORY_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_length=len(recent),
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=current_observation,
        admissible_actions=formatted_actions,
    )


def extract_task_description(observation: str) -> str:
    marker = "Your task is to: "
    start = observation.find(marker)
    if start == -1:
        raise ValueError("Task description not found in text observation.")
    return observation[start + len(marker) :].strip()


def extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_think(model_response: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def eval_dataset(task: Task) -> str:
    gamefile = str(task.metadata.get("gamefile", ""))
    if "/valid_seen/" in gamefile:
        return "eval_in_distribution"
    if "/valid_unseen/" in gamefile:
        return "eval_out_of_distribution"
    return "train"


__all__ = [
    "ALFWORLD_SYSTEM_PROMPT",
    "ALFWorldSkillOptEnv",
    "ALFWorldSkillOptEnvConfig",
    "build_skillopt_user_prompt",
    "format_skillopt_observation",
]
