"""Shared workbook execution with overridable prompts and tool observations."""

from pathlib import Path

from ....agents import OpenAIChatAgent
from ....agents.buitin_agents.openai_chat.callbacks import (
    native_observation,
    parse_native_tools,
    parse_text_action,
    text_format_error,
    text_observation,
)
from ....agents.typing import AgentResult
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import RolloutResult, Task, Usage
from ..common import export_skills, total_usage
from . import bash_only, workspace_tools
from .config import SpreadsheetBenchConfig
from .reward import evaluate
from .typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    SpreadsheetContext,
    SpreadsheetInteractionResult,
    SpreadsheetObservation,
)
from .workbooks import prepare_cases


@Environment.register("spreadsheetbench")
class SpreadsheetBenchEnvironment(Environment[SpreadsheetBenchConfig]):
    """Own workbook lifecycle and scoring while subclasses customize interaction."""

    config_type = SpreadsheetBenchConfig
    """Model, protocol, loop limits, and scoring settings."""

    def __init__(self, config: SpreadsheetBenchConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Construct the default agent protocol.

        Args:
            config: Recorded execution and protocol settings.
            agent: Optional model transport wrapped by the default tool parser.
        """
        super().__init__(config)
        textual = config.tool_agent == "text_action"
        self.agent = OpenAIChatAgent(
            config.agent,
            model=agent,
            parse_response=parse_text_action if textual else parse_native_tools,
            format_error=text_format_error if textual else str,
            observation=text_observation if textual else native_observation,
            native_tools=not textual,
        )

    def _system_prompt(self, context: SpreadsheetContext) -> str:
        """Render the system message from public rollout-local inputs."""
        if self.config.interaction_mode == "bash_only":
            return bash_only.system_prompt(context, self.config)
        return workspace_tools.system_prompt(context, self.config)

    def _user_prompt(self, context: SpreadsheetContext) -> str:
        """Render initial task instructions without private scoring inputs."""
        if self.config.interaction_mode == "bash_only":
            return bash_only.user_prompt(context, self.config)
        return workspace_tools.user_prompt(context, self.config)

    def _get_observation(self, response: AgentResult, context: SpreadsheetContext) -> SpreadsheetObservation:
        """Decode a response, execute its tools, and encode public feedback."""
        if self.config.interaction_mode == "bash_only":
            return bash_only.get_observation(response, context, self.config, self.agent)
        return workspace_tools.get_observation(response, context, self.config, self.agent)

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Prepare public inputs, run the shared loop, and score private references."""
        if not isinstance(task, SpreadsheetBenchTask) or not isinstance(reward_data, SpreadsheetBenchRewardData):
            raise TypeError("SpreadsheetBench requires SpreadsheetBenchTask and SpreadsheetBenchRewardData")
        textual = self.config.interaction_mode == "bash_only"
        if textual and len(task.cases) != 1:
            raise ValueError("bash_only requires exactly one workbook case")
        directories = prepare_cases(task, reward_data, workspace)
        interactions, trajectory = [], []
        for case, directory in zip(task.cases, directories, strict=True):
            skill_directory = directory if textual else directory / "skills/xlsx"
            text = export_skills(skills, skill_directory)
            context = SpreadsheetContext(
                task=task,
                directory=directory,
                skill_directory=skill_directory,
                skill_text=text,
                tools=self.config.tools
                if self.config.tools is not None
                else ([] if textual else workspace_tools.available_tools(self.config, text)),
            )
            interaction = self._interact(context)
            interactions.append(interaction)
            if trajectory:
                trajectory.append({"role": "system", "content": f"[CASE {case.number}]"})
            trajectory.extend(interaction.trajectory)
        diagnostics = dict(interactions[-1].diagnostics)
        diagnostics["n_turns"] = sum(item.diagnostics["n_turns"] for item in interactions)
        if not textual:
            diagnostics["skill_loaded"] = all(item.diagnostics["skill_loaded"] for item in interactions)
        reward = evaluate(reward_data, directories).model_copy(update=diagnostics)
        trajectory.append({"role": "system", "content": reward.detail})
        return RolloutResult(
            output=interactions[-1].output,
            trajectory=trajectory,
            reward=reward,
            usage=Usage(
                input_tokens=sum(item.usage.input_tokens for item in interactions),
                output_tokens=sum(item.usage.output_tokens for item in interactions),
                total_tokens=sum(item.usage.total_tokens for item in interactions),
                reported=all(item.usage.reported for item in interactions),
                calls=[call for item in interactions for call in item.usage.calls],
            ),
        )

    def _interact(self, context: SpreadsheetContext) -> SpreadsheetInteractionResult:
        """Drive one case using subclass hooks and rollout-local conversation state."""
        textual = self.config.interaction_mode == "bash_only"
        directory, text = context.directory, context.skill_text
        responses, trajectory = [], []
        output = ""
        system, user = self._system_prompt(context), self._user_prompt(context)
        (directory / "system.md").write_text(system, encoding="utf-8")
        (directory / "task.md").write_text(user, encoding="utf-8")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        trajectory.extend(messages)
        loaded = self.config.skill_loading_mode == "system_prompt" and bool(text.strip())
        previous, repeats, reminded = "", 0, False
        finished, failure = False, ""
        for _ in range(self.config.max_turns):
            response = (
                self.agent.run(messages=messages, tools=context.tools)
                if context.tools
                else self.agent.run(messages=messages)
            )
            responses.append(response)
            observation = self._get_observation(response, context)
            messages.append(observation.assistant)
            trajectory.append(response.trajectory[-1] if response.trajectory else observation.assistant)
            output = observation.output
            loaded = loaded or observation.skill_loaded
            if observation.complete:
                if not textual or (directory / "output.xlsx").is_file():
                    finished = True
                    break
                if reminded or len(responses) >= self.config.max_turns:
                    failure = f"Output file was not created: {directory / 'output.xlsx'}"
                    break
                reminded = True
                feedback = [
                    {
                        "role": "user",
                        "content": f"[System Check] The output file was NOT created at: {directory / 'output.xlsx'}\n"
                        "Please return to work again until you create the output file at the exact path "
                        "specified above, then signal ACTION: TASK_COMPLETE again.",
                    }
                ]
            else:
                feedback = observation.messages
            messages.extend(feedback)
            trajectory.extend(feedback)
            if observation.signature:
                repeats = repeats + 1 if observation.signature == previous else 1
                previous = observation.signature
                if self.config.stagnation_limit is not None and repeats >= self.config.stagnation_limit:
                    failure = (
                        f"ReAct agent repeated an identical {observation.repeated_kind} "
                        f"{self.config.stagnation_limit} consecutive times"
                    )
                    break
        if not finished and not failure:
            failure = "Max turns exceeded"
            if textual:
                output = "Max turns exceeded. Could not complete the task."
        diagnostics = {"n_turns": len(responses)}
        if textual:
            diagnostics.update(finished=finished, stop_reason=failure)
        else:
            diagnostics.update(
                skill_loaded=loaded,
                skill_loading_mode=self.config.skill_loading_mode,
                skill_tool_exposed=any(tool["function"]["name"] == "skill" for tool in context.tools),
            )
        return SpreadsheetInteractionResult(
            output=output, trajectory=trajectory, usage=total_usage(responses), diagnostics=diagnostics
        )
