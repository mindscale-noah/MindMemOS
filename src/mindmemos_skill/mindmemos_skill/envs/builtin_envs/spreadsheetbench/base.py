"""Shared workbook lifecycle and interaction scheduling for spreadsheet environments."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import JsonValue

from ....agents import OpenAIChatAgent
from ....agents.typing import AgentResult
from ....infra.execution import ExecutionConfig
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import RolloutResult, Task, Usage
from ..common import total_usage
from .config import SpreadsheetBenchExecutionConfig
from .execution import check_execution, prepare_execution
from .reward import evaluate
from .typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    SpreadsheetContext,
    SpreadsheetInteractionResult,
    SpreadsheetObservation,
)
from .workbooks import prepare_cases

ConfigT = TypeVar("ConfigT", bound=SpreadsheetBenchExecutionConfig)


class SpreadsheetExecutionEnvironment(Environment[ConfigT]):
    """Share workbook preparation, audit scheduling, and scoring without choosing a protocol."""

    task_type = SpreadsheetBenchTask
    """Public task schema shared by both protocols."""

    reward_data_type = SpreadsheetBenchRewardData
    """Private workbook scoring inputs."""

    agent: OpenAIChatAgent
    """Protocol-specific model transport owned by the concrete environment."""

    execution: ExecutionConfig
    """Internal execution settings bound to the managed Python installation."""

    def __init__(self, config: ConfigT, *, protocol: str) -> None:
        """Prepare the shared managed runtime before any task or model call.

        Args:
            config: Protocol-specific model and interaction settings.
            protocol: Fixed dependency profile selected by the concrete environment.
        """
        super().__init__(config)
        self.execution = prepare_execution(protocol)
        check_execution(self.execution, Path(self.execution.python_command[0]).parent)

    @abstractmethod
    def _prepare_context(self, task: SpreadsheetBenchTask, directory: Path, skills: list[Skill]) -> SpreadsheetContext:
        """Export instructions and expose tools for one workbook case."""
        raise NotImplementedError

    @abstractmethod
    def _system_prompt(self, context: SpreadsheetContext) -> str:
        """Build the protocol-specific system prompt."""
        raise NotImplementedError

    @abstractmethod
    def _user_prompt(self, context: SpreadsheetContext) -> str:
        """Build the public task prompt."""
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self, response: AgentResult, context: SpreadsheetContext) -> SpreadsheetObservation:
        """Execute the protocol's actions and encode their feedback."""
        raise NotImplementedError

    @abstractmethod
    def _initial_skill_loaded(self, context: SpreadsheetContext) -> bool:
        """Return whether instructions were preloaded into the system prompt."""
        raise NotImplementedError

    @abstractmethod
    def _diagnostics(
        self, context: SpreadsheetContext, loaded: bool, finished: bool, failure: str
    ) -> dict[str, JsonValue]:
        """Record protocol-specific completion information."""
        raise NotImplementedError

    def _validate_task(self, task: SpreadsheetBenchTask) -> None:
        """Allow multiple workbook cases unless the concrete protocol restricts them."""

    def _completion_error(self, context: SpreadsheetContext) -> str | None:
        """Accept completion unless the concrete protocol requires another action."""
        return None

    def _completion_reminder(self, context: SpreadsheetContext, error: str) -> str:
        """Ask the agent to resolve an unmet completion requirement."""
        return error

    def _turn_limit_output(self, output: str) -> str:
        """Retain the final model output when the turn budget expires."""
        return output

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Prepare public inputs, run the shared loop, and score private references."""
        if not isinstance(task, SpreadsheetBenchTask) or not isinstance(reward_data, SpreadsheetBenchRewardData):
            raise TypeError("SpreadsheetBench requires SpreadsheetBenchTask and SpreadsheetBenchRewardData")
        self._validate_task(task)
        directories = prepare_cases(task, reward_data, workspace)
        (workspace.parent / "audit" / "execution.json").write_text(
            json.dumps({"python": self.execution.python_command[0]}, indent=2) + "\n", encoding="utf-8"
        )
        interactions: list[SpreadsheetInteractionResult] = []
        trajectory: list[dict[str, JsonValue]] = []
        for case, directory in zip(task.cases, directories, strict=True):
            context = self._prepare_context(task, directory, skills)
            interaction = self._interact(context)
            interactions.append(interaction)
            if trajectory:
                trajectory.append({"role": "system", "content": f"[CASE {case.number}]"})
            trajectory.extend(interaction.trajectory)
        diagnostics = dict(interactions[-1].diagnostics)
        diagnostics["n_turns"] = sum(item.diagnostics["n_turns"] for item in interactions)
        if "skill_loaded" in diagnostics:
            diagnostics["skill_loaded"] = all(item.diagnostics["skill_loaded"] for item in interactions)
        reward = evaluate(reward_data, directories).model_copy(update=diagnostics)
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
        directory = context.directory
        responses: list[AgentResult] = []
        trajectory: list[dict[str, JsonValue]] = []
        output = ""
        system, user = self._system_prompt(context), self._user_prompt(context)
        (directory / "system.md").write_text(system, encoding="utf-8")
        (directory / "task.md").write_text(user, encoding="utf-8")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        trajectory.extend(messages)
        loaded = self._initial_skill_loaded(context)
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
                completion_error = self._completion_error(context)
                if completion_error is None:
                    finished = True
                    break
                if reminded or len(responses) >= self.config.max_turns:
                    failure = completion_error
                    break
                reminded = True
                feedback = [{"role": "user", "content": self._completion_reminder(context, completion_error)}]
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
            output = self._turn_limit_output(output)
        diagnostics = {"n_turns": len(responses), **self._diagnostics(context, loaded, finished, failure)}
        return SpreadsheetInteractionResult(
            output=output, trajectory=trajectory, usage=total_usage(responses), diagnostics=diagnostics
        )
