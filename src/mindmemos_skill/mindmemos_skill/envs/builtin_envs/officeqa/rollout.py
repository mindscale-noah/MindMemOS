"""Bounded document research followed by private OfficeQA scoring."""

import json
import re
from pathlib import Path

from ....agents import OpenAIChatAgent
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import RolloutResult, Task
from ..common import skill_text, total_usage
from .config import OfficeQAConfig
from .prompts import SYSTEM_PROMPT
from .reward import evaluate
from .tools import TOOL_SCHEMAS, run_tool
from .typing import OfficeQARewardData, OfficeQATask


@Environment.register("officeqa")
class OfficeQAEnvironment(Environment[OfficeQAConfig]):
    """Own one document-tool conversation and its evaluated result."""

    task_type = OfficeQATask
    """Public OfficeQA task schema."""

    reward_data_type = OfficeQARewardData
    """Private OfficeQA scoring schema."""

    config_type = OfficeQAConfig
    """Model, tool-turn budget, and scoring identity."""

    def __init__(self, config: OfficeQAConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Create or inject the model backend.

        Args:
            config: Audited execution settings.
            agent: Optional matching backend.
        """
        super().__init__(config)
        if agent is not None and agent.config != config.agent:
            raise ValueError("Injected agent config must match the environment's agent config")
        self.agent = agent if agent is not None else OpenAIChatAgent(config.agent)

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Expose only allowlisted public documents through explicit tools."""
        if not isinstance(task, OfficeQATask) or not isinstance(reward_data, OfficeQARewardData):
            raise TypeError("OfficeQA requires OfficeQATask and OfficeQARewardData")
        text = skill_text(skills)
        system = SYSTEM_PROMPT.format(skill_section=f"## Skill\n{text.strip()}\n\n" if text.strip() else "")
        candidates = [
            name
            for name in task.documents
            if any(name == hint or Path(name).stem == Path(hint).stem for hint in task.source_files)
        ]
        files = "\n".join(f"- {name}" for name in candidates[:20]) or "- none resolved"
        user = f"## Question\n{task.instruction}\n\n## Candidate Files\n{files}"
        if task.oracle_context:
            user += f"\n\n## Oracle Parsed Pages\n{task.oracle_context}"
        if task.source_docs:
            user += "\n\n## Source Hints\n" + "\n".join(f"- {hint}" for hint in task.source_docs)
        (workspace / "system.md").write_text(system, encoding="utf-8")
        (workspace / "task.md").write_text(user, encoding="utf-8")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        trajectory, responses = list(messages), []
        output, answer, failure = "", "", ""
        for turn in range(1, self.config.max_tool_turns + 1):
            response = self.agent.run_messages(messages=messages, tools=TOOL_SCHEMAS)
            if not isinstance(response.output, str):
                raise TypeError("OfficeQA requires a text agent output")
            output = response.output
            responses.append(response)
            event = response.trajectory[-1]
            assistant = {"role": "assistant", "content": output}
            calls = event.get("tool_calls", [])
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            trajectory.append(event)
            for call in calls:
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"] or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                except (ValueError, TypeError) as exc:
                    observation = f"Error: {exc}"
                else:
                    observation = run_tool(function["name"], arguments, task.documents, self.config.max_tool_chars)
                feedback = {"role": "tool", "tool_call_id": call["id"], "content": observation}
                messages.append(feedback)
                trajectory.append(feedback)
            if calls:
                continue
            if "<answer>" in output.lower():
                match = re.search(r"<answer>(.*?)</answer>", output, re.IGNORECASE | re.DOTALL)
                lines = [line.strip() for line in output.splitlines() if line.strip()]
                answer = match.group(1).strip() if match else (lines[-1] if lines else output.strip())
                break
            failure = (
                f"Exceeded tool-turn budget ({self.config.max_tool_turns})"
                if turn == self.config.max_tool_turns
                else "Model neither produced a tool request nor a final answer"
            )
            break
        (workspace / "response.txt").write_text(output, encoding="utf-8")
        return RolloutResult(
            output=output,
            trajectory=trajectory,
            reward=evaluate(answer, reward_data, fail_reason=failure),
            usage=total_usage(responses),
        )
