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

from ....algos.trace2skill.contracts import TraceEvidence
from ....algos.trace2skill.treeskill.analysis import ChatModel
from ....algos.trace2skill.treeskill.errors import TreeSkillModelRequestError
from ....algos.trace2skill.treeskill.models import AnalysisItem, TrajectoryAnalysisRecord
from ....typing import Trajectory
from .evaluator import compare_workbooks, workbook_used_ranges
from .trace2skill_compat import (
    TASK_COMPLETE_SIGNAL,
    PolicyResponseType,
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
        self._output_root.mkdir(parents=True, exist_ok=False)
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(item: TraceEvidence) -> tuple[TrajectoryAnalysisRecord | None, str | None]:
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
            except TreeSkillModelRequestError:
                raise
            except Exception as exc:
                failure_dir = self._output_root / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / f"{_safe_name(item.trajectory_id)}.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
                return None, item.trajectory_id

        async def run_phase(items: list[TraceEvidence]) -> list[tuple[TrajectoryAnalysisRecord | None, str | None]]:
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
                        "content": format_reference_observation(
                            parsed.error_message or "Invalid action format."
                        ),
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
        source_value = trajectory.environment.running_dir
        if not source_value:
            raise ValueError("failed trajectory has no persisted SpreadsheetBench workspace")
        source = Path(source_value)
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
    return files(f"{__package__}.prompt_templates").joinpath(name).read_text(encoding="utf-8")


def _render_agent_log(trajectory: Trajectory) -> str:
    lines = [f"# Chat History: {trajectory.agent.agent_type.value}", ""]
    for index, event in enumerate(trajectory.events, start=1):
        role = str(event.get("role") or "event").upper()
        content = event.get("content")
        rendered = content if isinstance(content, str) else json.dumps(event, ensure_ascii=False, sort_keys=True)
        lines.extend([f"## [{index}] {role}", "", rendered, "", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def _answer_position(trajectory: Trajectory) -> str:
    position = str(trajectory.task.metadata.get("answer_position") or "")
    sheet = trajectory.task.metadata.get("answer_sheet")
    if position and sheet and "!" not in position:
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
    source = Path(str(trajectory.task.metadata.get("src_dir") or ""))
    if not source.is_dir():
        return None
    for pattern in ("*golden*.xlsx", "*answer*.xlsx", "gold.xlsx"):
        hits = sorted(source.glob(pattern))
        if hits:
            return hits[0]
    return None


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
