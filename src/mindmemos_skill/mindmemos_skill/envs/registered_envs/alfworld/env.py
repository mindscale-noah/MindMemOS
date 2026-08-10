"""Lean-history ALFWorld environment migrated from Skill-GRPO."""

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

SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED Embodied Environment.\n\n"
    "## Output Format\n"
    "You MUST respond in the following format:\n"
    "<think>your chain-of-thought reasoning about what to do next</think>\n"
    "<action>the exact action command to execute</action>\n\n"
    "Available actions include: look, go to <location>, open <object>, close <object>, "
    "pick up <object>, put <object> in/on <receptacle>, "
    "toggle <object> on/off, heat <object> with <appliance>, "
    "cool <object> with <appliance>, clean <object> with <appliance>, "
    "slice <object> with <tool>, inventory.\n"
    "For navigation: use 'go to <receptacle> <index>' (e.g. 'go to cabinet 1').\n"
    "You can only interact with objects that are visible in the current observation."
)


class ALFWorldEnvConfig(EnvConfig):
    """ALFWorld rollout settings matching the lean-history source env."""

    max_steps: int = Field(default=50, ge=1)
    seed: int = 42


@register(type=ComponentType.ENV, name="alfworld")
class ALFWorldEnv(BaseEnv[ALFWorldEnvConfig]):
    """Run clean O(N) message history over one ALFWorld simulator."""

    config_type = ALFWorldEnvConfig

    async def _prepare(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: EnvRolloutContext,
    ) -> PreparedRollout:
        prepared = await super()._prepare(task=task, skills=skills, context=context)
        prepared.agent_request.options["skill_injection_mode"] = "system_prompt"
        gamefile = task.metadata.get("resolved_gamefile")
        if not isinstance(gamefile, str) or not gamefile:
            raise ValueError(f"Task {task.task_id} missing resolved_gamefile in metadata")
        skill_prompt = self._build_skill_prompt(skills, prepared.environment.running_dir)
        system = f"{SYSTEM_PROMPT}\n\n{skill_prompt}" if skill_prompt else SYSTEM_PROMPT
        sample_index = context.metadata.get("sample_index", context.rollout.attempt_no)
        if not isinstance(sample_index, int):
            raise TypeError("ALFWorld context metadata sample_index must be an integer")
        prepared.runtime_state = {
            "gamefile": gamefile,
            "sample_index": sample_index,
            "system": system,
            "simulator": None,
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
        transcript: list[dict[str, Any]] = [{"role": "system", "content": state["system"]}]
        conversation: list[dict[str, Any]] = []
        error: str | None = None
        won = False
        turns = 0
        invalid_actions = 0
        started = time.time()

        try:
            simulator = await asyncio.to_thread(
                self._build_simulator,
                task,
                state["sample_index"],
            )
            state["simulator"] = simulator
            raw_observation, _info = await asyncio.to_thread(simulator.reset)
            transcript.append(
                {
                    "role": "user",
                    "content": format_observation(raw_observation, simulator.admissible_actions),
                }
            )

            for step_index in range(self.config.max_steps):
                response = await agent.respond(prepared.agent_request, transcript, tools=[])
                assistant_text = response.content.strip()
                if not assistant_text:
                    assistant_text = "<think>empty model response</think><action>look</action>"
                if extract_action(assistant_text) is None:
                    invalid_actions += 1
                    assistant_text = "<think>missing action tag</think><action>look</action>"
                transcript.append({"role": "assistant", "content": assistant_text})

                raw_feedback, step_reward, done, info = await asyncio.to_thread(
                    simulator.step,
                    assistant_text,
                )
                conversation.append(
                    {
                        "step": step_index,
                        "action": extract_action(assistant_text),
                        "reasoning": extract_think(assistant_text),
                        "model_response": assistant_text,
                        "env_feedback": raw_feedback,
                        "reward": float(step_reward),
                        "done": bool(done),
                        "is_action_valid": jsonable(info.get("is_action_valid")),
                    }
                )
                turns = step_index + 1
                if done:
                    won = bool(info.get("won", False))
                    break
                transcript.append(
                    {
                        "role": "user",
                        "content": format_observation(raw_feedback, simulator.admissible_actions),
                    }
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            simulator = state.get("simulator")
            if simulator is not None:
                await asyncio.to_thread(simulator.close)
                state["simulator"] = None

        if not won and error is None and turns >= self.config.max_steps:
            error = f"Timeout after {self.config.max_steps} steps"
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
        self._write_artifacts(prepared, transcript, conversation)
        return agent.build_trajectory(
            request=prepared.agent_request,
            messages=transcript,
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

    def _build_simulator(self, task: Task, sample_index: int):
        from .runtime import ALFWorldSimulator

        return ALFWorldSimulator(
            seed=self.config.seed + sample_index,
            is_train="train" in task.tags,
            eval_dataset=eval_dataset(task),
            gamefile=str(task.metadata["resolved_gamefile"]),
        )

    @classmethod
    def _build_skill_prompt(cls, skills: Sequence[Skill], running_dir: str | None) -> str:
        parts: list[str] = []
        skills_dir = Path(running_dir) / "skills" if running_dir is not None else None
        if skills_dir is not None:
            skills_dir.mkdir(parents=True, exist_ok=True)
        for index, skill in enumerate(skills):
            stem = f"{index:03d}_{safe_filename(skill.name)}"
            text = format_skill(skill).strip()
            if skills_dir is not None:
                (skills_dir / f"{stem}.md").write_text(format_skill(skill), encoding="utf-8")
            if text:
                parts.append(f"### Skill: {stem}\n{text}")
        if not parts:
            return ""
        return (
            "## Skill Knowledge\n"
            "Below are learned strategies. Use them to choose admissible actions.\n\n"
            + "\n\n".join(parts)
            + "\n"
        )

    @staticmethod
    def _write_artifacts(
        prepared: PreparedRollout,
        transcript: list[dict[str, Any]],
        conversation: list[dict[str, Any]],
    ) -> None:
        running_dir = prepared.environment.running_dir
        if running_dir is None:
            return
        task = prepared.agent_request.task
        prediction_dir = Path(running_dir) / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        (prediction_dir / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (prediction_dir / "result.json").write_text(
            json.dumps(
                {
                    "id": task.task_id,
                    "task_type": task.metadata.get("task_type"),
                    "gamefile": task.metadata.get("gamefile"),
                    "resolved_gamefile": task.metadata.get("resolved_gamefile"),
                    "conversation": conversation,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (prediction_dir / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def format_admissible(actions: list[str]) -> str:
    return ", ".join(f"'{action}'" for action in actions if action != "help")


def format_observation(observation: str, actions: list[str]) -> str:
    return f"{observation}\n\nAdmissible actions: [{format_admissible(actions)}]"


def extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_think(model_response: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def eval_dataset(task: Task) -> str:
    gamefile = str(task.metadata.get("gamefile", ""))
    if "/valid_seen/" in gamefile:
        return "eval_in_distribution"
    if "/valid_unseen/" in gamefile:
        return "eval_out_of_distribution"
    return "train"


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "skill"


def format_skill(skill: Skill) -> str:
    sections = [f"# {skill.name}", ""]
    if skill.description:
        sections.extend([skill.description, ""])
    sections.append(skill.content)
    if not skill.content.endswith("\n"):
        sections.append("")
    return "\n".join(sections)


__all__ = [
    "ALFWorldEnv",
    "ALFWorldEnvConfig",
    "SYSTEM_PROMPT",
    "extract_action",
    "extract_think",
    "format_admissible",
    "format_observation",
]
