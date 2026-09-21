"""Released-style SpreadsheetBench success and failure trajectory analysis."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

# The task-execution environment never receives training-analysis gold files.
from .....agents import OpenAIChatAgent
from .....agents.typing import AgentResult
from .....envs.builtin_envs.spreadsheetbench.reference_reward import compare_workbooks, workbook_used_ranges
from .....envs.builtin_envs.spreadsheetbench.rollout import SpreadsheetBenchEnvironment
from .....envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    SpreadsheetContext,
    SpreadsheetObservation,
)
from ..errors import TreeSkillModelRequestError
from ..typing import AnalysisItem, ChatModel, TraceEvidence, TrajectoryAnalysisRecord
from ..typing import TrainingRollout as Trajectory
from .config import SpreadsheetBenchReferenceConfig
from .spreadsheetbench_policy import (
    TASK_COMPLETE_SIGNAL,
    PolicyResponseType,
    build_reference_messages,
    format_reference_observation,
    parse_policy_response,
    run_reference_bash,
)

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
        chat_model: ChatModel,
        failure_chat_model: ChatModel | None = None,
        task: str,
        output_root: Path,
        concurrency: int,
        success_score_threshold: float,
        temperature: float,
        max_tokens: int,
        max_turns: int = 20,
        shell_timeout_seconds: int = 120,
    ) -> None:
        """init  .

        Args:
            chat_model: Model client implementing the recorded chat request protocol.
            failure_chat_model: Optional separate transport for reference failure analysis.
            task: Stage audit label or public task supplied by the caller.
            output_root: Fresh root for reference analysis workspaces and reports.
            concurrency: Maximum simultaneous requests in this stage.
            success_score_threshold: Minimum recorded reward classified as successful.
            temperature: Sampling temperature passed unchanged to the model.
            max_tokens: Completion-token budget passed to the model.
            max_turns: Maximum tool-use turns allowed for one failure analysis.
            shell_timeout_seconds: Timeout in seconds for each analysis shell command.
        """
        self._success_chat_model = chat_model
        self._failure_chat_model = failure_chat_model or chat_model
        self._task = task
        self._output_root = output_root
        self._concurrency = concurrency
        self._success_score_threshold = success_score_threshold
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._shell_timeout_seconds = shell_timeout_seconds

    async def analyze(
        self,
        evidence: list[TraceEvidence],
        *,
        trajectories_by_id: Mapping[str, Trajectory],
    ) -> tuple[list[TrajectoryAnalysisRecord], list[str]]:
        """Analyze.

        Args:
            evidence: Ordered trajectory-derived evidence available to this stage.
            trajectories_by_id: Completed training executions indexed by physical trajectory ID.
        """
        self._output_root.mkdir(parents=True, exist_ok=False)
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(item: TraceEvidence) -> tuple[TrajectoryAnalysisRecord | None, str | None]:
            """Run.

            Args:
                item: Explicit item for this operation.
            """
            trajectory = trajectories_by_id.get(item.trajectory_id)
            if trajectory is None:
                return None, item.trajectory_id
            try:
                async with semaphore:
                    if item.score is not None and item.score >= self._success_score_threshold:
                        record = await self._analyze_success(item, trajectory)
                    elif item.score is not None:
                        record = await self._analyze_failure(item, trajectory)
                    else:
                        raise ValueError("SpreadsheetBench reference analysis requires outcome labels")
                return record, None
            except Exception as exc:
                failure_dir = self._output_root / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / f"{_safe_name(item.trajectory_id)}.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
                return None, item.trajectory_id

        async def run_phase(items: list[TraceEvidence]) -> list[tuple[TrajectoryAnalysisRecord | None, str | None]]:
            """Run phase.

            Args:
                items: Explicit items for this operation.
            """
            tasks = [asyncio.create_task(run(item)) for item in items]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        error_evidence = [
            item for item in evidence if item.score is not None and item.score < self._success_score_threshold
        ]
        success_evidence = [
            item for item in evidence if item.score is not None and item.score >= self._success_score_threshold
        ]
        unlabeled_evidence = [item for item in evidence if item.score is None]
        results: list[tuple[TrajectoryAnalysisRecord | None, str | None]] = []
        # Match the Spreadsheet pipeline: finish error analysis before starting
        # the independent one-call success-analysis stage.
        for phase in (error_evidence, success_evidence, unlabeled_evidence):
            results.extend(await run_phase(phase))
        records = [record for record, _ in results if record is not None]
        failures = [trajectory_id for _, trajectory_id in results if trajectory_id is not None]
        return records, failures

    async def _analyze_success(
        self,
        evidence: TraceEvidence,
        trajectory: Trajectory,
    ) -> TrajectoryAnalysisRecord:
        output_dir = self._output_root / "success" / _safe_name(evidence.trajectory_id)
        output_dir.mkdir(parents=True)
        system = _prompt("success_analysis_system_llm.txt")
        user = _prompt("success_analysis_user_llm.txt").replace(
            "{agent_log}",
            _render_agent_log(trajectory),
        )
        (output_dir / "prompt.json").write_text(
            json.dumps(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            response = await self._success_chat_model.chat(
                task=f"{self._task}:success",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise TreeSkillModelRequestError(
                stage="success analysis",
                item_id=evidence.trajectory_id,
                cause=exc,
            ) from exc
        report = response.content or ""
        (output_dir / "success_analysis.md").write_text(report, encoding="utf-8")
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
    ) -> TrajectoryAnalysisRecord:
        output_dir = self._output_root / "error" / _safe_name(evidence.trajectory_id)
        work_dir = output_dir / "agent_work"
        self._stage_failure_workspace(trajectory, work_dir)
        agent_log = _render_agent_log(trajectory)
        (output_dir / "agent_log.md").write_text(agent_log, encoding="utf-8")

        answer_position = _answer_position(trajectory)
        system = _prompt("error_analysis_system.txt").format(working_directory=str(output_dir.resolve()))
        user = _prompt("error_analysis_user.txt")
        user = user.replace("{agent_log}", agent_log)
        user = user.replace("{working_dir}", str(output_dir.resolve()))
        user = user.replace("{evaluate_usage}", _evaluate_usage(output_dir, answer_position))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        (output_dir / "prompt.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        passed = False
        report = ""
        completion_reminded = False
        for _turn in range(1, self._max_turns + 1):
            try:
                response = await self._failure_chat_model.chat(
                    task=f"{self._task}:error",
                    messages=list(messages),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception as exc:
                raise TreeSkillModelRequestError(
                    stage="error analysis",
                    item_id=evidence.trajectory_id,
                    cause=exc,
                ) from exc
            content = response.content or ""
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
                    observation = await asyncio.to_thread(
                        run_reference_bash,
                        command,
                        working_dir=output_dir,
                        timeout_seconds=self._shell_timeout_seconds,
                    )
            elif parsed.action.name == "evaluate_output":
                observation, passed_now = await asyncio.to_thread(
                    self._evaluate_output,
                    output_dir,
                    parsed.action.arguments,
                    answer_position,
                )
                passed = passed or passed_now
            else:
                observation = f"Error: unknown tool {parsed.action.name!r}"
            messages.append({"role": "user", "content": format_reference_observation(observation)})

        (output_dir / "error_analysis_chat.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not report:
            raise ValueError("failure analysis did not complete a pass-gated, parseable report")
        (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
        (output_dir / "evaluate_passed.flag").write_text("PASS\n", encoding="utf-8")
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
    def _evaluate_output(
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
        return f"{reward.answer_sheet}!{position}"
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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160] or "trajectory"


__all__ = ["SpreadsheetBenchReferenceAnalyzer"]


def routing_user_prompt(
    tree,
    task,
    routing_context: Mapping[str, Any] | None = None,
) -> str:
    """Routing user prompt.

    Args:
        tree: Parsed Markdown tree with stable heading identities.
        task: Stage audit label or public task supplied by the caller.
        routing_context: Optional public spreadsheet preview and task-field overrides.
    """
    from ..fusion_tree import tree_prompt_payload

    context = dict(routing_context or {})
    if isinstance(task, SpreadsheetBenchTask):
        context.setdefault("instruction_type", task.instruction_type)
        context.setdefault("answer_position", task.answer_position)
        context.setdefault("spreadsheet_content", spreadsheet_preview(task.cases[0].path))
    payload = {
        "task": {
            "instance_id": context.get("instance_id", task.id),
            "instruction_type": context.get("instruction_type", ""),
            "answer_position": context.get("answer_position", ""),
            "instruction": context.get("instruction", task.instruction),
            "spreadsheet_content": _truncate(
                str(context.get("spreadsheet_content", "") or ""),
                5000,
            ),
        },
        "skill_tree": tree_prompt_payload(tree),
    }
    return f"Route skill subtrees for this spreadsheet task.\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def spreadsheet_preview(file_path: Path, max_rows: int = 5) -> str:
    """Return the workbook preview consumed by the TreeSkill routing prompt.

    Args:
        file_path: Public input workbook used to build the first-sheet preview.
        max_rows: Maximum number of preview rows before a truncation notice.
    """

    try:
        import openpyxl

        workbook = openpyxl.load_workbook(file_path, data_only=True)
        worksheet = workbook.active
        lines: list[str] = []
        for index, row in enumerate(worksheet.iter_rows(values_only=True), 1):
            if index > max_rows:
                lines.append(f"... ({worksheet.max_row - max_rows} more rows)")
                break
            values = [str(cell) if cell is not None else "" for cell in row]
            lines.append(str(tuple(values)))
        workbook.close()
        return "\n".join(lines)
    except Exception as exc:
        return f"[Could not read spreadsheet: {exc}]"


class SpreadsheetBenchReferenceEnvironment(SpreadsheetBenchEnvironment):
    """Customize TreeSkill interaction while inheriting workbook execution and scoring."""

    config_type = SpreadsheetBenchReferenceConfig
    """Recorded TreeSkill interaction settings and shared evaluation version."""

    def __init__(self, config: SpreadsheetBenchReferenceConfig, *, agent=None):
        """Construct a raw-text model for the TreeSkill response hook.

        Args:
            config: TreeSkill prompt, execution, and interaction settings.
            agent: Optional raw-text agent for injection and offline testing.
        """
        super().__init__(config, agent=agent)
        self.agent = agent if agent is not None else OpenAIChatAgent(config.agent)

    def _system_prompt(self, context: SpreadsheetContext) -> str:
        """Render the pinned system message from prepared Skill content."""
        return build_reference_messages(
            task=context.task,
            working_dir=context.directory,
            input_file=context.directory / "input.xlsx",
            output_file=context.directory / "output.xlsx",
            spreadsheet_content="",
            skill_content=context.skill_text,
            skill_dir=context.skill_directory,
            transactional_recalculation=False,
        )[0]["content"]

    def _user_prompt(self, context: SpreadsheetContext) -> str:
        """Render the pinned task prompt and optional recalculation instructions."""
        return build_reference_messages(
            task=context.task,
            working_dir=context.directory,
            input_file=context.directory / "input.xlsx",
            output_file=context.directory / "output.xlsx",
            spreadsheet_content=spreadsheet_preview(context.directory / "input.xlsx"),
            skill_content=context.skill_text,
            skill_dir=context.skill_directory,
            transactional_recalculation=self.config.transactional_recalculation,
        )[1]["content"]

    def _get_observation(self, response: AgentResult, context: SpreadsheetContext) -> SpreadsheetObservation:
        """Parse TreeSkill actions, execute bash, and encode its text feedback."""
        if not isinstance(response.output, str):
            raise TypeError("Reference policy requires text output")
        content = response.output
        assistant = {"role": "assistant", "content": content}
        parsed = parse_policy_response(content)
        if parsed.response_type is PolicyResponseType.TASK_COMPLETE:
            return SpreadsheetObservation(assistant=assistant, complete=True, output=content)
        if parsed.response_type is PolicyResponseType.FORMAT_ERROR:
            feedback = format_reference_observation(parsed.error_message or "Invalid action format.")
            signature = json.dumps(
                {"response_type": "format_error", "response": content, "feedback": feedback},
                sort_keys=True,
                ensure_ascii=False,
            )
            kind = "format error"
        else:
            action = parsed.action
            if action.name != "bash":
                observation = f"Error: unknown tool {action.name!r}; only 'bash' is available"
            else:
                command = action.arguments.get("command")
                observation = (
                    run_reference_bash(
                        command, working_dir=context.directory, timeout_seconds=self.config.shell_timeout_seconds
                    )
                    if isinstance(command, str) and command.strip()
                    else "Error: bash arguments.command must be a non-empty string"
                )
            feedback = format_reference_observation(observation)
            signature = json.dumps(
                {
                    "response_type": "action",
                    "name": action.name,
                    "arguments": action.arguments,
                    "observation": observation,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            kind = "action/result pair"
        return SpreadsheetObservation(
            assistant=assistant,
            messages=[{"role": "user", "content": feedback}],
            signature=signature,
            repeated_kind=kind,
            output=content,
        )


REFERENCE_SKILL_SHA256 = {
    "SKILL.md": "35d45452ebb3963706ff26fc5892b3216b8d5e4c0c06145a7c563ed747bf4468",
    "recalc.py": "ab1ef0c94536bb23b6c6a3d32769b0401ec3cc85e73c247d574dd84ec73af15d",
    "LICENSE.txt": "79f6d8f5b427252fa3b1c11ecdbdb6bf610b944f7530b4de78f770f38741cfaa",
}
"""Authorized Human-Written xlsx package required by the upstream reference experiment."""


def validate_reference_skill(skill) -> None:
    """Check the exact initial package without downloading restricted resources.

    Args:
        skill: Caller-supplied authorized xlsx package.

    Raises:
        ValueError: Files or content differ from the pinned reference package.
    """
    from hashlib import sha256

    package = {"SKILL.md": skill.bundle.files["SKILL.md"].encode("utf-8"), **skill.resources.files}
    if set(package) != set(REFERENCE_SKILL_SHA256) or any(
        sha256(package[name]).hexdigest() != expected for name, expected in REFERENCE_SKILL_SHA256.items()
    ):
        raise ValueError("Reference mode requires the unmodified authorized Human-Written xlsx package")
