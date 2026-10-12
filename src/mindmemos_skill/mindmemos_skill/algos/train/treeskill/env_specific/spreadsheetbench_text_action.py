"""Released-style SpreadsheetBench success and failure trajectory analysis."""

from __future__ import annotations

import json
import re
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

from .....agents import OpenAIChatAgent
from .....envs.base import Environment
from .....envs.builtin_envs.spreadsheetbench.execution import prepare_execution
from .....envs.builtin_envs.spreadsheetbench.ranges import split_targets
from .....envs.builtin_envs.spreadsheetbench.recalculation_cli import (
    TASK_LOCAL_HELPER_FILENAME,
    append_recalculation_instructions,
)

# The task-execution environment never receives training-analysis gold files.
from .....envs.builtin_envs.spreadsheetbench.text_action_mode.environment import (
    SpreadsheetBenchTextActionEnvironment,
    spreadsheet_preview,
)
from .....envs.builtin_envs.spreadsheetbench.text_action_mode.policy import (
    TASK_COMPLETE_SIGNAL,
    PolicyResponseType,
    format_reference_observation,
    parse_policy_response,
    run_reference_bash,
)
from .....envs.builtin_envs.spreadsheetbench.text_action_mode.reference_reward import (
    compare_workbooks,
    workbook_used_ranges,
)
from .....envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
)
from .....envs.typing import Task
from .....skill import Skill
from ..env_adapter import TreeSkillEnvAdapter
from ..typing import (
    AnalysisInput,
    AnalysisItem,
    AnalysisOutcome,
    TraceEvidence,
    TrajectoryAnalysisRecord,
    TrajectoryAnalyzer,
    TreeSkillAnalysisRuntime,
    TreeSkillRoutingContext,
)
from ..typing import TrainingRollout as Trajectory
from .config import TreeSkillSpreadsheetBenchReferenceEnvAdapterConfig
from .spreadsheetbench_prompts import ROUTING_SYSTEM_PROMPT

_ERROR_ITEM_PATTERN = re.compile(
    r"^#\s+(Failure Cause Item|Failure Memory Item)\s+(\d+)\s*\n"
    r"(.*?)(?=\n#\s+(?:Failure Cause Item|Failure Memory Item)\s+\d+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SUCCESS_ITEM_PATTERN = re.compile(
    r"^#+\s+Success Memory Item\s+(\d+)\s*$",
    re.MULTILINE,
)
_SECTION_PATTERN = re.compile(
    r"^##\s+{name}\s*\n(.*?)(?=\n##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SUCCESS_SECTION_PATTERN = re.compile(
    r"^(#+)\s+(Title|Description|Content)\s*$",
    re.MULTILINE,
)


class SpreadsheetBenchReferenceAnalyzer:
    """Use the released one-call success and pass-gated agentic failure workflows."""

    def __init__(
        self,
        *,
        agent: OpenAIChatAgent,
        failure_agent: OpenAIChatAgent,
        success_score_threshold: float,
        max_turns: int = 20,
        shell_timeout_seconds: int = 120,
    ) -> None:
        """Bind direct agents and the reference execution limits.

        Args:
            agent: Direct agent for one-call success analysis.
            failure_agent: Direct agent for tool-using failure analysis.
            success_score_threshold: Minimum reward classified as successful.
            max_turns: Maximum failure-analysis tool turns.
            shell_timeout_seconds: Timeout per shell command.
        """
        self._agent = agent
        self._failure_agent = failure_agent
        self._success_score_threshold = success_score_threshold
        self._max_turns = max_turns
        self._shell_timeout_seconds = shell_timeout_seconds

    async def __call__(self, inputs: AnalysisInput) -> AnalysisOutcome:
        """Analyze one item and return reports for the trainer to save.

        Args:
            inputs: Labeled training evidence and a trainer-prepared tool workspace.

        Returns:
            Analysis, reports and completed model responses, including failures.
        """
        evidence, trajectory, workspace = inputs.evidence, inputs.trajectory, inputs.workspace
        outcome = AnalysisOutcome()
        try:
            if evidence.score is not None and evidence.score >= self._success_score_threshold:
                outcome.record = await self._analyze_success(evidence, trajectory, outcome)
            elif evidence.score is not None:
                outcome.record = await self._analyze_failure(evidence, trajectory, workspace, outcome)
            else:
                raise ValueError("SpreadsheetBench reference analysis requires outcome labels")
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    async def _analyze_success(
        self,
        evidence: TraceEvidence,
        trajectory: Trajectory,
        outcome: AnalysisOutcome,
    ) -> TrajectoryAnalysisRecord:
        system = _prompt("success_analysis_system_llm.txt")
        user = _prompt("success_analysis_user_llm.txt").replace(
            "{agent_log}",
            _render_agent_log(trajectory),
        )
        outcome.artifacts["prompt.json"] = json.dumps(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            ensure_ascii=False,
            indent=2,
        )
        try:
            response = await self._agent.run(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception:
            outcome.failed_calls += 1
            raise
        outcome.responses.append(response)
        report = response.output or ""
        outcome.artifacts["success_analysis.md"] = report
        items = _parse_success_items(report)
        if not items:
            raise ValueError("success analysis returned no parseable Success Memory Items")
        return TrajectoryAnalysisRecord(
            instance_id=evidence.trajectory_id,
            task_id=evidence.task_id,
            record_source="success",
            source_file="success_analysis.md",
            items=tuple(items),
        )

    async def _analyze_failure(
        self,
        evidence: TraceEvidence,
        trajectory: Trajectory,
        output_dir: Path,
        outcome: AnalysisOutcome,
    ) -> TrajectoryAnalysisRecord:
        execution = await prepare_execution("text_action")
        work_dir = output_dir / "agent_work"
        agent_log = _render_agent_log(trajectory)
        outcome.artifacts["agent_log.md"] = agent_log

        answer_position = _answer_position(trajectory)
        system = _prompt("error_analysis_system.txt").format(working_directory=str(output_dir.resolve()))
        user = _prompt("error_analysis_user.txt")
        user = user.replace("{agent_log}", agent_log)
        user = user.replace("{working_dir}", str(output_dir.resolve()))
        user = user.replace("{evaluate_usage}", _evaluate_usage(output_dir, answer_position))
        if (work_dir / TASK_LOCAL_HELPER_FILENAME).is_file():
            user = append_recalculation_instructions(
                user,
                working_dir=output_dir,
                output_file=work_dir / "output.xlsx",
                python_executable=Path(execution.python_command[0]),
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        outcome.artifacts["prompt.json"] = json.dumps(messages, ensure_ascii=False, indent=2)

        passed = False
        report = ""
        completion_reminded = False
        try:
            for _turn in range(1, self._max_turns + 1):
                try:
                    response = await self._failure_agent.run(
                        messages=list(messages),
                    )
                except Exception:
                    outcome.failed_calls += 1
                    raise
                outcome.responses.append(response)
                content = response.output or ""
                messages.append({"role": "assistant", "content": content})
                parsed = parse_policy_response(content)
                if parsed.response_type is PolicyResponseType.TASK_COMPLETE:
                    candidate = content.split(TASK_COMPLETE_SIGNAL, 1)[0].strip()
                    if passed and _parse_error_items(candidate):
                        report = candidate
                        break
                    if completion_reminded:
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System Check] You have not yet produced both a PASS evaluation and a parseable final "
                                "analysis report. Continue the required workflow, then return Failure Cause Items and "
                                "Failure Memory Items before signaling ACTION: TASK_COMPLETE."
                            ),
                        }
                    )
                    completion_reminded = True
                    continue
                if parsed.response_type is PolicyResponseType.FORMAT_ERROR:
                    messages.append(
                        {
                            "role": "user",
                            "content": format_reference_observation(parsed.error_message or "Invalid action format."),
                        }
                    )
                    continue

                assert parsed.action is not None
                if parsed.action.name == "bash":
                    command = parsed.action.arguments.get("command")
                    if not isinstance(command, str) or not command.strip():
                        observation = "Error: bash arguments.command must be a non-empty string"
                    else:
                        observation = await run_reference_bash(
                            command,
                            working_dir=output_dir,
                            timeout_seconds=self._shell_timeout_seconds,
                            execution=execution,
                        )
                elif parsed.action.name == "evaluate_output":
                    observation, passed_now = await self._evaluate_output(
                        output_dir, parsed.action.arguments, answer_position
                    )
                    passed = passed or passed_now
                else:
                    observation = f"Error: unknown tool {parsed.action.name!r}"
                messages.append({"role": "user", "content": format_reference_observation(observation)})

        finally:
            outcome.artifacts["error_analysis_chat.json"] = json.dumps(messages, ensure_ascii=False, indent=2)
        if not report:
            raise ValueError("failure analysis did not complete a pass-gated, parseable report")
        outcome.artifacts["analysis_report.md"] = report
        outcome.artifacts["evaluate_passed.flag"] = "PASS\n"
        return TrajectoryAnalysisRecord(
            instance_id=evidence.trajectory_id,
            task_id=evidence.task_id,
            record_source="error",
            source_file="analysis_report.md",
            items=tuple(_parse_error_items(report)),
        )

    @staticmethod
    def _stage_failure_workspace(trajectory: Trajectory, destination: Path) -> None:
        source = trajectory.work_dir / "workspace"
        task = trajectory.sample.task
        if not isinstance(task, SpreadsheetBenchTask) or len(task.cases) != 1:
            raise ValueError("Failure analysis requires exactly one SpreadsheetBench case")
        case_directory = source / task.cases[0].number
        if case_directory.is_dir():
            source = case_directory
        if not source.is_dir():
            raise FileNotFoundError(f"failed trajectory workspace does not exist: {source}")
        shutil.copytree(source, destination)
        if not (destination / "gold.xlsx").is_file():
            golden = _find_golden_workbook(trajectory)
            if golden is None:
                raise FileNotFoundError("failed trajectory has no gold.xlsx analysis artifact")
            shutil.copyfile(golden, destination / "gold.xlsx")

    @staticmethod
    async def _evaluate_output(
        analysis_dir: Path,
        arguments: dict[str, Any],
        default_answer_position: str,
    ) -> tuple[str, bool]:
        output = _safe_analysis_path(analysis_dir, str(arguments.get("output_file") or "agent_work/output.xlsx"))
        golden = _safe_analysis_path(analysis_dir, str(arguments.get("ground_truth") or "agent_work/gold.xlsx"))
        position = str(arguments.get("answer_position") or default_answer_position)
        if not position:
            position = workbook_used_ranges(golden)
        correct, detail = compare_workbooks(golden, output, position)
        status = "PASS" if correct else "FAIL"
        report = f"Result: {status}"
        if detail:
            report += f"\n{detail}"
        return report, correct


def _prompt(name: str) -> str:
    return files(f"{__package__}.spreadsheetbench_prompts").joinpath(name).read_text(encoding="utf-8")


def _render_agent_log(trajectory: Trajectory) -> str:
    lines = ["# Chat History: react", ""]
    for index, event in enumerate(trajectory.result.trajectory, start=1):
        role = str(event.get("role") or "event").upper()
        content = event.get("content")
        rendered = content if isinstance(content, str) else json.dumps(event, ensure_ascii=False, sort_keys=True)
        lines.extend([f"## [{index}] {role}", "", rendered, "", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def _answer_position(trajectory: Trajectory) -> str:
    reward = trajectory.sample.reward_data
    if not isinstance(reward, SpreadsheetBenchRewardData):
        raise TypeError("Reference analysis requires SpreadsheetBenchRewardData")
    position = reward.answer_position
    if position and reward.answer_sheet and "!" not in position:
        sheet = split_targets(reward.answer_sheet)[0].strip("'\"")
        return f"{sheet}!{position}"
    return position


def _evaluate_usage(output_dir: Path, answer_position: str) -> str:
    output = output_dir.resolve() / "agent_work" / "output.xlsx"
    golden = output_dir.resolve() / "agent_work" / "gold.xlsx"
    position_line = f',\n        "answer_position": "{answer_position}"' if answer_position else ""
    return (
        "To run the evaluation, use the `evaluate_output` tool:\n\n"
        "Action:\n"
        "{\n"
        '    "name": "evaluate_output",\n'
        '    "arguments": {\n'
        f'        "output_file": "{output}",\n'
        f'        "ground_truth": "{golden}"{position_line}\n'
        "    }\n"
        "}\n"
    )


def _parse_error_items(text: str) -> list[AnalysisItem]:
    result: list[AnalysisItem] = []
    for match in _ERROR_ITEM_PATTERN.finditer(_strip_response_wrappers(text)):
        heading, number, body = match.groups()
        kind = "failure_cause" if heading == "Failure Cause Item" else "failure_memory"
        fields = {
            name: _extract_section(body, name)
            for name in (
                "Title",
                "Description",
                "Content",
                "Relation to Skill",
                "Skill Reflection",
            )
        }
        result.append(
            AnalysisItem(
                item_id=f"{kind}_{number}",
                kind=kind,
                number=int(number),
                title=fields["Title"],
                description=fields["Description"],
                content=fields["Content"],
                relation_to_skill=fields["Relation to Skill"] if kind == "failure_cause" else "",
                skill_reflection=fields["Skill Reflection"] if kind == "failure_memory" else "",
            )
        )
    return result


def _parse_success_items(text: str) -> list[AnalysisItem]:
    text = _strip_response_wrappers(text)
    matches = list(_SUCCESS_ITEM_PATTERN.finditer(text))
    result: list[AnalysisItem] = []
    for index, match in enumerate(matches):
        body = text[match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        sections: dict[str, str] = {}
        section_matches = list(_SUCCESS_SECTION_PATTERN.finditer(body))
        for section_index, section in enumerate(section_matches):
            end = section_matches[section_index + 1].start() if section_index + 1 < len(section_matches) else len(body)
            sections[section.group(2)] = _clean_success_section(body[section.end() : end])
        result.append(
            AnalysisItem(
                item_id=f"success_memory_{match.group(1)}",
                kind="success_memory",
                number=int(match.group(1)),
                title=sections.get("Title", ""),
                description=sections.get("Description", ""),
                content=sections.get("Content", ""),
            )
        )
    return result


def _clean_success_section(text: str) -> str:
    cleaned = re.sub(r"\n(?:---+|\*\*\*+|___+)\s*\Z", "", text.strip())
    return cleaned.strip()


def _extract_section(body: str, name: str) -> str:
    match = re.search(_SECTION_PATTERN.pattern.format(name=re.escape(name)), body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _strip_response_wrappers(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped


def _find_golden_workbook(trajectory: Trajectory) -> Path | None:
    reward = trajectory.sample.reward_data
    if not isinstance(reward, SpreadsheetBenchRewardData) or len(reward.cases) != 1:
        raise ValueError("Reference analysis requires exactly one private workbook case")
    return reward.cases[0].path


def _safe_analysis_path(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    resolved = (root / candidate if not candidate.is_absolute() else candidate).resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"analysis tool path escapes its workspace: {raw!r}")
    return resolved


__all__ = ["SpreadsheetBenchReferenceAnalyzer"]


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


REFERENCE_SKILL_SHA256 = {
    "SKILL.md": "35d45452ebb3963706ff26fc5892b3216b8d5e4c0c06145a7c563ed747bf4468",
    "recalc.py": "ab1ef0c94536bb23b6c6a3d32769b0401ec3cc85e73c247d574dd84ec73af15d",
    "LICENSE.txt": "79f6d8f5b427252fa3b1c11ecdbdb6bf610b944f7530b4de78f770f38741cfaa",
}
"""Authorized Human-Written xlsx package required by the upstream reference experiment."""


def validate_reference_skill(skill: Skill) -> None:
    """Check the exact initial package without downloading restricted resources.

    Args:
        skill: Caller-supplied authorized xlsx package.

    Raises:
        ValueError: Files or content differ from the pinned reference package.
    """
    from hashlib import sha256

    package = {"SKILL.md": skill.content.encode("utf-8"), **skill.resources}
    if set(package) != set(REFERENCE_SKILL_SHA256) or any(
        sha256(package[name]).hexdigest() != expected for name, expected in REFERENCE_SKILL_SHA256.items()
    ):
        raise ValueError("Reference mode requires the unmodified authorized Human-Written xlsx package")


@TreeSkillEnvAdapter.register("spreadsheetbench_text_action")
class TreeSkillSpreadsheetBenchReferenceEnvAdapter(TreeSkillEnvAdapter):
    """Bind released spreadsheet analysis and routing without changing rollout execution."""

    config_type = TreeSkillSpreadsheetBenchReferenceEnvAdapterConfig
    """Reject generic analysis overrides that the released analyzer cannot apply."""

    def validate_environment(self, env: Environment) -> None:
        """Require the released execution protocol for reference training.

        Args:
            env: Caller-owned task environment.

        Raises:
            TypeError: The environment does not implement the reference spreadsheet protocol.
        """
        if not isinstance(env, SpreadsheetBenchTextActionEnvironment):
            raise TypeError("SpreadsheetBench reference adaptation requires SpreadsheetBenchTextActionEnvironment")

    def render_task(self, task: Task) -> str:
        """Read only the public workbook preview used by the released router.

        Args:
            task: Public task; generic tasks can supply an explicit public preview instead.

        Returns:
            First-sheet preview, or an empty string for tasks without a workbook.
        """
        return spreadsheet_preview(task.cases[0].path) if isinstance(task, SpreadsheetBenchTask) else ""

    def build_routing_messages(self, context: TreeSkillRoutingContext) -> list[dict[str, str]]:
        """Construct the exact released spreadsheet routing request.

        Args:
            context: Public task, tree payload and optional public field overrides.

        Returns:
            Complete messages matching the reference protocol.
        """
        task = context.task
        public = dict(context.public_context)
        if isinstance(task, SpreadsheetBenchTask):
            public.setdefault("instruction_type", task.instruction_type)
            public.setdefault("answer_position", task.answer_position)
        if "spreadsheet_content" not in public:
            public["spreadsheet_content"] = self.render_task(task)
        payload = {
            "task": {
                "instance_id": public.get("instance_id", task.id),
                "instruction_type": public.get("instruction_type", ""),
                "answer_position": public.get("answer_position", ""),
                "instruction": public.get("instruction", task.instruction),
                "spreadsheet_content": _truncate(str(public.get("spreadsheet_content", "") or ""), 5000),
            },
            "skill_tree": context.tree,
        }
        return [
            {
                "role": "system",
                "content": self.config.routing_system
                if self.config.routing_system is not None
                else ROUTING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Route skill subtrees for this spreadsheet task.\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
            },
        ]

    def prepare_analysis_workspace(self, trajectory: Trajectory, workspace: Path) -> None:
        """Stage private training workbooks only for a trainer-owned analysis.

        Args:
            trajectory: Failed training execution and its permitted gold workbook.
            workspace: Fresh directory allocated by the trainer.
        """
        SpreadsheetBenchReferenceAnalyzer._stage_failure_workspace(trajectory, workspace / "agent_work")

    def analysis_phases(self, evidence: list[TraceEvidence], threshold: float) -> list[list[TraceEvidence]]:
        """Finish failure analysis before independent success analysis.

        Args:
            evidence: Labeled training evidence.
            threshold: Minimum score classified as successful.

        Returns:
            Error, success and unlabeled groups in reference order.
        """
        return [
            [item for item in evidence if item.score is not None and item.score < threshold],
            [item for item in evidence if item.score is not None and item.score >= threshold],
            [item for item in evidence if item.score is None],
        ]

    def create_analyzer(self, runtime: TreeSkillAnalysisRuntime) -> TrajectoryAnalyzer:
        """Create the released success and tool-using failure analyzer.

        Args:
            runtime: Direct agents and fixed stage settings.

        Returns:
            Reference analyzer; no model or tool call occurs until analyze().
        """
        config = runtime.config
        return SpreadsheetBenchReferenceAnalyzer(
            agent=runtime.agent,
            failure_agent=runtime.failure_agent,
            success_score_threshold=config.success_score_threshold,
            max_turns=config.analysis_max_turns,
            shell_timeout_seconds=config.shell_timeout_seconds,
        )
