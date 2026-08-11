"""LiveMathematicianBench multiple-choice environment."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from ....agents.base import Agent
from ....algos.evolve.skill_grpo_with_replay_buffer.prompts import (
    LIVEMATH_SYSTEM as SYSTEM_PROMPT,
)
from ....algos.evolve.skill_grpo_with_replay_buffer.prompts import (
    livemath_refinement_prompt,
    livemath_system_prompt,
    livemath_user_prompt,
)
from ....registry import ComponentType, register
from ....typing import EnvConfig, Reward, Skill, Task, Trajectory
from ...base import BaseEnv, EnvRolloutContext, PreparedRollout

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


class LiveMathEnvConfig(EnvConfig):
    """LiveMath prompt and retry settings matching Skill-GRPO."""

    max_turns: int = Field(default=1, ge=1)
    use_theorem: bool = False
    use_sketch: bool = False


@register(type=ComponentType.ENV, name="livemath")
class LiveMathEnv(BaseEnv[LiveMathEnvConfig]):
    """Run one LiveMath question with the original chat protocol."""

    config_type = LiveMathEnvConfig

    async def _prepare(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: EnvRolloutContext,
    ) -> PreparedRollout:
        prepared = await super()._prepare(task=task, skills=skills, context=context)
        prepared.agent_request.options["skill_injection_mode"] = "system_prompt"
        item = task.metadata
        sample_index = context.metadata.get("sample_index", context.rollout.attempt_no)
        if not isinstance(sample_index, int):
            raise TypeError("LiveMath context metadata sample_index must be an integer")
        prepared.runtime_state = {
            "item": item,
            "sample_index": sample_index,
            "system": build_system(skill_text(skills)),
            "user": build_user(item, self.config.use_theorem, self.config.use_sketch),
            "response": "",
            "error": None,
            "conversation": [],
            "evaluation": {},
        }
        return prepared

    async def _execute(self, *, agent: Agent[Any], prepared: PreparedRollout) -> Trajectory:
        state = prepared.runtime_state
        system = state["system"]
        user = state["user"]
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        conversation: list[dict[str, Any]] = []
        response_text = ""
        error: str | None = None
        started = time.time()

        try:
            for turn in range(self.config.max_turns):
                messages = (
                    [{"role": "system", "content": system}, {"role": "user", "content": user}]
                    if turn == 0
                    else [
                        {"role": "system", "content": system},
                        {"role": "user", "content": refinement(response_text)},
                    ]
                )
                response = await agent.respond(prepared.agent_request, messages, tools=[])
                response_text = str(response.content or "")
                if turn == 0:
                    transcript = [*messages, {"role": "assistant", "content": response_text}]
                else:
                    transcript.extend([messages[1], {"role": "assistant", "content": response_text}])
                conversation.append({"type": "message", "turn": turn + 1, "content": response_text})
                if "<answer>" in response_text.lower():
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        evaluation = evaluate(response_text, state["item"]["correct_choice"], state["item"]["choices"])
        ended = time.time()
        state.update(
            {
                "response": response_text,
                "error": error,
                "conversation": conversation,
                "evaluation": evaluation,
            }
        )
        self._write_artifacts(prepared, state)
        return agent.build_trajectory(
            request=prepared.agent_request,
            messages=transcript,
            started_at=started,
            ended_at=ended,
            n_turn=len(conversation),
            is_success=error is None,
            error_info=error,
            metadata={
                "finished": error is None,
                "turns": len(conversation),
                "error": error,
                "started_at": started,
                "ended_at": ended,
                "sample_index": state["sample_index"],
                **evaluation,
                "conversation": conversation,
                "workspace_scope": prepared.environment.metadata["workspace_scope"],
            },
        )

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        del trajectory
        state = prepared.runtime_state
        evaluation = state["evaluation"]
        score = 0.0 if state["error"] else float(evaluation["em"])
        predicted = evaluation["predicted_label"] or evaluation["predicted_answer"]
        detail = None
        if state["error"]:
            detail = f"error: {state['error']}"
        elif not evaluation["em"]:
            detail = f"MCQ=0: predicted '{predicted}' but expected '{evaluation['correct_label']}'"
        return Reward(score=score, detail=detail, metadata=evaluation)

    @staticmethod
    def _write_artifacts(prepared: PreparedRollout, state: dict[str, Any]) -> None:
        running_dir = prepared.environment.running_dir
        if running_dir is None:
            return
        workspace = Path(running_dir)
        evaluation = state["evaluation"]
        error = state["error"]
        fail_reason = f"error: {error}" if error else ""
        if not error and not evaluation["em"]:
            predicted = evaluation["predicted_label"] or evaluation["predicted_answer"]
            fail_reason = f"MCQ=0: predicted '{predicted}' but expected '{evaluation['correct_label']}'"
        item = state["item"]
        prediction = {
            "id": item.get("id"),
            "question": item.get("question"),
            "hard": int(evaluation["em"]),
            "soft": evaluation["f1"],
            "response": state["response"],
            "fail_reason": fail_reason,
            "agent_ok": error is None,
            "n_turns": len(state["conversation"]),
            **evaluation,
        }
        prediction_dir = workspace / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        (prediction_dir / "prediction.json").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (prediction_dir / "conversation.json").write_text(
            json.dumps(state["conversation"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (workspace / "target_system_prompt.txt").write_text(state["system"], encoding="utf-8")
        (workspace / "target_user_prompt.txt").write_text(state["user"], encoding="utf-8")


def build_system(skill_content: str) -> str:
    return livemath_system_prompt(skill_content)


def build_user(item: Mapping[str, Any], use_theorem: bool, use_sketch: bool) -> str:
    return livemath_user_prompt(dict(item), use_theorem, use_sketch)


def refinement(previous_response: str) -> str:
    return livemath_refinement_prompt(previous_response)


def skill_text(skills: Sequence[Skill]) -> str:
    return "\n\n".join(skill.content.strip() for skill in skills if skill.content.strip())


def extract_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else (text or "").strip()


def normalize_label(value: str) -> str:
    return str(value).strip().upper().rstrip(".):")


def evaluate(
    response: str,
    correct_choice: Mapping[str, Any],
    choices: list[dict[str, Any]],
) -> dict[str, Any]:
    answer = extract_answer(response)
    valid_labels = {normalize_label(choice.get("label", "")) for choice in choices}
    predicted_label = normalize_label(answer)
    if predicted_label not in valid_labels:
        for choice in choices:
            if str(choice.get("text", "")).strip().lower() == answer.lower():
                predicted_label = normalize_label(choice.get("label", ""))
                break
        else:
            first = normalize_label(answer.split()[0]) if answer.split() else ""
            predicted_label = first if first in valid_labels else predicted_label
    correct_label = normalize_label(correct_choice.get("label", ""))
    predicted_text = next(
        (
            str(choice.get("text", "")).strip()
            for choice in choices
            if normalize_label(choice.get("label", "")) == predicted_label
        ),
        "",
    )
    correct_text = str(correct_choice.get("text", "")).strip()
    em = float(predicted_label == correct_label)
    return {
        "em": em,
        "f1": em,
        "sub_em": em,
        "predicted_answer": predicted_label or answer,
        "predicted_label": predicted_label,
        "predicted_text": predicted_text,
        "correct_label": correct_label,
        "correct_text": correct_text,
    }


__all__ = [
    "LiveMathEnv",
    "LiveMathEnvConfig",
    "SYSTEM_PROMPT",
    "build_system",
    "build_user",
    "evaluate",
]
