"""Static LLM-based analysis of one Skill source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar

from ....agents import AgentResult, OpenAIChatAgent
from ....skill import Skill
from ...base import Algorithm
from .._json_parse import extract_json_object
from ..base import SkillAnalyzer
from ..typing import SkillAnalysisResult, SkillFinding, SkillIdentity, SkillJudgeRequest
from .config import SkillJudgeConfig
from .models import SkillJudgeVerdict, WindowScan
from .prompts import build_merge_prompt, build_prompt, build_window_prompt

_SENSITIVE_NAMES = {
    ".env",
    ".git",
    ".hg",
    ".svn",
    "credentials.json",
    "secrets.json",
}
_SENSITIVE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx"}
_DIMENSIONS = ("safety", "completeness", "executability", "maintainability")


@dataclass(slots=True)
class _Window:
    """One sliding-window content block plus what it covers."""

    index: int
    coverage: str
    text: str


@Algorithm.register("skill_judge")
class SkillJudge(SkillAnalyzer[SkillJudgeConfig, SkillJudgeRequest]):
    """Audit Skill content and resources without filesystem side effects."""

    config_type = SkillJudgeConfig

    algorithm_name: ClassVar[str] = "skill_judge"

    def __init__(self, config: SkillJudgeConfig) -> None:
        """Configure one structured-output Agent for each prompt shape."""

        super().__init__(config)
        self._verdict_agent = OpenAIChatAgent(config.judge, output_parser=self._parse_verdict)
        self._window_agent = OpenAIChatAgent(config.judge, output_parser=self._parse_window_scan)

    async def analyze(self, request: SkillJudgeRequest) -> SkillAnalysisResult:
        """Audit the complete supplied Skill source."""

        identity = request.identity or SkillIdentity.from_skill(request.skill)
        skill_name = identity.name or request.skill.snapshot().name or "unnamed-skill"
        windows, manifest, metadata = self._build_windows(request.skill)
        if len(windows) == 1:
            prompt = build_prompt(skill_name=skill_name, content=windows[0].text, manifest=manifest)
            response = await self._verdict_agent.run(messages=[{"role": "user", "content": prompt}])
            verdict = response.output
            if not isinstance(verdict, SkillJudgeVerdict):
                raise TypeError("skill_judge returned an invalid parsed verdict")
            return self._to_result(request, identity, verdict, metadata, response)

        scans: list[WindowScan] = []
        for window in windows:
            scans.append(await self._scan_window(skill_name, window, manifest, len(windows)))
        verdict, merge_response = await self._merge_scans(skill_name, scans, manifest)
        metadata = {**metadata, "windows": len(windows), "window_findings": sum(len(scan.findings) for scan in scans)}
        return self._to_result(request, identity, verdict, metadata, merge_response)

    @staticmethod
    def _parse_verdict(content: str) -> SkillJudgeVerdict:
        """Parse one final or merged judge verdict."""

        return SkillJudgeVerdict.model_validate(extract_json_object(content))

    @staticmethod
    def _parse_window_scan(content: str) -> WindowScan:
        """Parse one per-window evidence scan."""

        return WindowScan.model_validate(extract_json_object(content))

    async def _scan_window(
        self,
        skill_name: str,
        window: _Window,
        manifest: list[str],
        window_total: int,
    ) -> WindowScan:
        """Collect findings from one complete Sliding-window block."""

        prompt = build_window_prompt(
            skill_name=skill_name,
            window_index=window.index,
            window_total=window_total,
            window_coverage=window.coverage,
            manifest=manifest,
            content=window.text,
        )
        response = await self._window_agent.run(messages=[{"role": "user", "content": prompt}])
        scan = response.output
        if not isinstance(scan, WindowScan):
            raise TypeError("skill_judge returned an invalid window scan")
        return scan

    async def _merge_scans(
        self,
        skill_name: str,
        scans: list[WindowScan],
        manifest: list[str],
    ) -> tuple[SkillJudgeVerdict, AgentResult]:
        """Merge complete per-window coverage into one scored verdict."""

        reports: list[str] = []
        for index, scan in enumerate(scans, 1):
            lines = [f"--- window {index}/{len(scans)} ---"]
            if not scan.findings:
                lines.append("(no findings; this window was unremarkable)")
            for finding in scan.findings:
                lines.append(f"[{finding.severity}] {finding.description} (location: {finding.location})")
            reports.append("\n".join(lines))
        prompt = build_merge_prompt(skill_name=skill_name, window_reports="\n\n".join(reports), manifest=manifest)
        response = await self._verdict_agent.run(messages=[{"role": "user", "content": prompt}])
        verdict = response.output
        if not isinstance(verdict, SkillJudgeVerdict):
            raise TypeError("skill_judge returned an invalid merged verdict")
        return verdict, response

    def _build_windows(self, skill: Skill) -> tuple[list[_Window], list[str], dict[str, Any]]:
        """Slice the Skill into windows that cover every textual source file."""

        files: list[tuple[str, str, int]] = [
            ("SKILL.md", skill.content, len(skill.content.encode("utf-8"))),
        ]
        skipped: list[str] = []
        for path, payload in sorted(skill.resources.items()):
            reason = self._skip_reason(path)
            if reason is not None:
                skipped.append(f"{path}: {reason}")
                continue
            text, reason = self._decode_text_resource(payload)
            if text is not None:
                files.append((path, text, len(payload)))
            else:
                skipped.append(f"{path}: {reason}")

        cap = self.config.per_file_cap
        overlap = self.config.window_overlap
        windows: list[_Window] = []
        truncated = bool(skipped)

        def add_window(coverage: str, text: str) -> None:
            if len(windows) >= self.config.max_windows:
                raise ValueError(
                    f"skill exceeds max_windows={self.config.max_windows} at {len(text)} chars; "
                    "refusing to audit with coverage gaps"
                )
            windows.append(_Window(index=len(windows) + 1, coverage=coverage, text=text))

        primary_path, primary, _primary_size = files[0]
        primary_lines = primary.splitlines()
        if len(primary_lines) > self.config.max_md_lines:
            primary = "\n".join(primary_lines[: self.config.max_md_lines])
            truncated = True
        if len(primary) <= cap:
            add_window(f"{primary_path} (entire file)", primary)
        else:
            step = cap - overlap
            for start in range(0, len(primary), step):
                chunk = primary[start : start + cap]
                add_window(f"{primary_path}@{start}-{start + len(chunk)}", chunk)
                if start + cap >= len(primary):
                    break

        budget = 0
        parts: list[str] = []
        current_coverage: list[str] = []

        def flush() -> None:
            nonlocal budget, parts, current_coverage
            if parts:
                add_window("; ".join(current_coverage), "\n\n".join(parts))
                budget, parts, current_coverage = 0, [], []

        for path, text, original_size in files[1:]:
            if len(text) > cap:
                text = text[:cap]
                coverage = f"{path}@0-{len(text)} (of {original_size} bytes)"
                truncated = True
            else:
                coverage = f"{path} ({original_size} bytes)"
            if budget + len(text) > self.config.total_cap:
                flush()
            budget += len(text)
            parts.append(f"--- {path} ---\n{text}")
            current_coverage.append(coverage)
        flush()

        manifest = [f"{path} | {size} bytes" for path, _text, size in files]
        manifest.extend(f"{item.split(':', 1)[0]} | skipped | {item.split(':', 1)[1].strip()}" for item in skipped)
        metadata = {
            "included_files": len(files),
            "skipped_files": skipped,
            "content_chars": sum(len(window.text) for window in windows),
            "truncated": truncated,
            "windows": len(windows),
        }
        return windows, manifest, metadata

    @staticmethod
    def _decode_text_resource(payload: bytes) -> tuple[str | None, str | None]:
        """Return UTF-8 text and an optional stable skip reason."""

        if b"\x00" in payload:
            return None, "binary_file"
        try:
            return payload.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, "non_utf8"

    @staticmethod
    def _skip_reason(path: str) -> str | None:
        """Return why one resource path must not be sent to the judge."""

        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            return "unsafe_path"
        if any(part in _SENSITIVE_NAMES for part in pure.parts):
            return "sensitive_path"
        if pure.name.lower() in _SENSITIVE_NAMES or pure.suffix.lower() in _SENSITIVE_SUFFIXES:
            return "sensitive_file"
        return None

    def _to_result(
        self,
        request: SkillJudgeRequest,
        identity: SkillIdentity,
        verdict: SkillJudgeVerdict,
        content_metadata: dict[str, Any],
        response: AgentResult,
    ) -> SkillAnalysisResult:
        """Normalize one judge verdict into the shared analysis contract."""

        concerns = verdict.concerns[: self.config.max_findings]
        evidence = verdict.evidence[: len(concerns)]
        findings = [
            SkillFinding(
                category="static.quality",
                message=concern,
                severity="warning",
                evidence=[evidence[index]] if index < len(evidence) else [],
            )
            for index, concern in enumerate(concerns)
        ]
        metrics: dict[str, Any] = {
            "scores": {name: verdict.scores[name].score for name in _DIMENSIONS},
            "score_reasons": {name: verdict.scores[name].reason for name in _DIMENSIONS},
            "score_mean": sum(verdict.scores[name].score for name in _DIMENSIONS) / len(_DIMENSIONS),
        }
        judge_tokens = self._token_metrics(response.usage)
        if judge_tokens:
            metrics["judge_tokens"] = judge_tokens
        metadata = {
            "judge_type": "static_skill",
            "algorithm": self.algorithm_name,
            "algorithm_version": self.config.algorithm_version,
            "prompt_version": self.config.prompt_version,
            "request_id": request.request_id,
            "project_id": identity.project_id,
            "skill_id": identity.skill_id,
            "version_id": identity.version_id,
            "content_hash": identity.content_hash,
            "skill_hash": identity.skill_hash,
            **content_metadata,
        }
        return SkillAnalysisResult(
            summary=verdict.overall_summary,
            findings=findings,
            metrics=metrics,
            metadata=metadata,
        )

    @staticmethod
    def _token_metrics(usage: dict[str, Any]) -> dict[str, int]:
        """Copy the Agent's normalized token counters into result metrics."""

        return {
            name: value
            for name in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance((value := usage.get(name)), int) and not isinstance(value, bool)
        }


__all__ = ["SkillJudge"]
