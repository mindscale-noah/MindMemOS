"""Per-trajectory extraction: one LLM call per trajectory, minimal-key contract.

Contract (docs/refactor-plan-minimal-contract.md §2):
  trajectory level: ``tasks`` (task-only paraphrase, no approach) and
  ``skills`` ([{name, evidence}], name checked against the scope's skill
  library, evidence checked as an exact substring of the raw trajectory).
  behavior level: ``summary`` (free narrative, consumed only by the evolve
  prompt) and ``keywords`` (verbatim fragments, each an exact substring of
  the raw trajectory; the code's only anchors).

Validation is format-agnostic only: exact substring checks and a directory
listing. Semantic constraints (keywords from the agent's own output, tasks
without approach) live in the prompt, not in code.

Entry points: :func:`build_extraction_records`, :func:`summarize_extraction_results`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_int
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import invoke as llm_invoke
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import stage_tools
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import (
    SCORE_ARTIFACT,
    read_score_text,
    scope_artifact_dir,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prompt_budget import cut_text
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.parallel import run_parallel_batch

# === Schema ===


def extraction_output_schema() -> dict[str, Any]:
    """Return the provider-level schema for one trajectory extraction.

    Harness-owned metadata (``trajectory_id``) is absent by design; the
    runner injects it after validating model output.
    """
    skill_entry = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["name", "evidence"],
        "additionalProperties": False,
    }
    behavior = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "keywords"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "tasks": {"type": "string"},
            "skills": {"type": "array", "items": skill_entry},
            "behaviors": {"type": "array", "items": behavior},
        },
        "required": ["tasks", "skills", "behaviors"],
        "additionalProperties": False,
    }


# === Validation ===


class ExtractionValidationError(Exception):
    """The extraction result violates a semantic contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _clip(text: str, cap: int = 60) -> str:
    return text if len(text) <= cap else text[: cap - 1] + "…"


def anchor_variants(fragment: str) -> tuple[str, ...]:
    """The fragment plus its JSON-escaped form, when they differ.

    Trajectory files are often JSON containers whose content strings escape
    quotes/backslashes/newlines; the model reads the rendered text and
    faithfully copies the SEMANTIC form. json.dumps is a deterministic
    character transform -- no rewording tolerance is introduced.
    """
    escaped = json.dumps(fragment, ensure_ascii=False)[1:-1]
    return (fragment,) if escaped == fragment else (fragment, escaped)


def anchor_in(fragment: str, raw_text: str) -> bool:
    """Exact-substring check across the fragment's anchor variants."""
    return any(variant in raw_text for variant in anchor_variants(fragment))


def scope_skill_names(workspace_root: str | Path, scope_id: str) -> frozenset[str]:
    """Directory listing of the scope's skill library (format-agnostic)."""
    skills_dir = Path(workspace_root) / scope_id / "skills"
    if not skills_dir.is_dir():
        return frozenset()
    return frozenset(path.name for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def validate_extraction_record(
    payload: Any,
    *,
    trajectory_id: str,
    raw_text: str | None = None,
    skill_names: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Validate model-derived fields and inject harness-owned metadata.

    ``raw_text`` enables the anchor checks (keywords and skill evidence must
    be exact substrings of the raw trajectory). ``skill_names`` enables the
    library check. Both are optional so persisted artifacts can be
    structurally re-validated on resume without re-reading the trajectory.
    """
    if not isinstance(payload, dict):
        raise ExtractionValidationError(["payload must be a JSON object"])

    errors: list[str] = []

    tasks = payload.get("tasks")
    if not isinstance(tasks, str) or not tasks.strip():
        errors.append("tasks must be a non-empty string")
        tasks = ""
    else:
        tasks = tasks.strip()

    skills: list[dict[str, str]] = []
    raw_skills = payload.get("skills")
    if not isinstance(raw_skills, list):
        errors.append("skills must be a list (possibly empty)")
    else:
        seen_names: set[str] = set()
        for index, raw_entry in enumerate(raw_skills):
            if not isinstance(raw_entry, dict):
                errors.append(f"skills[{index}] must be an object")
                continue
            name = raw_entry.get("name")
            evidence = raw_entry.get("evidence")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"skills[{index}].name must be a non-empty string")
                continue
            name = name.strip()
            if name in seen_names:
                errors.append(f"skills[{index}].name {name!r} is duplicated")
                continue
            if skill_names is not None and name not in skill_names:
                errors.append(f"skills[{index}].name {name!r} is not a skill directory in this scope's library")
                continue
            if not isinstance(evidence, str) or not evidence.strip():
                errors.append(f"skills[{index}].evidence must be a non-empty string")
                continue
            if raw_text is not None and not anchor_in(evidence, raw_text):
                errors.append(
                    f"skills[{index}].evidence is not an exact substring of the raw trajectory: {_clip(evidence)!r}"
                )
                continue
            seen_names.add(name)
            skills.append({"name": name, "evidence": evidence})

    behaviors: list[dict[str, Any]] = []
    raw_behaviors = payload.get("behaviors")
    if not isinstance(raw_behaviors, list) or not raw_behaviors:
        errors.append("behaviors must be a non-empty list")
    else:
        for index, raw_behavior in enumerate(raw_behaviors):
            if not isinstance(raw_behavior, dict):
                errors.append(f"behaviors[{index}] must be an object")
                continue
            summary = raw_behavior.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                errors.append(f"behaviors[{index}].summary must be a non-empty string")
                continue
            raw_keywords = raw_behavior.get("keywords")
            if not isinstance(raw_keywords, list) or not raw_keywords:
                errors.append(f"behaviors[{index}].keywords must be a non-empty list")
                continue
            keywords: list[str] = []
            for k_index, keyword in enumerate(raw_keywords):
                if not isinstance(keyword, str) or not keyword.strip():
                    errors.append(f"behaviors[{index}].keywords[{k_index}] must be a non-empty string")
                    continue
                if raw_text is not None and not anchor_in(keyword, raw_text):
                    errors.append(
                        f"behaviors[{index}].keywords[{k_index}] is not an exact substring "
                        f"of the raw trajectory: {_clip(keyword)!r}"
                    )
                    continue
                keywords.append(keyword)
            if len(keywords) != len(raw_keywords):
                continue
            behaviors.append({"summary": summary, "keywords": keywords})

    if errors:
        raise ExtractionValidationError(errors)

    return {
        "trajectory_id": trajectory_id,
        "tasks": tasks,
        "skills": skills,
        "behaviors": behaviors,
    }


def load_and_validate_record(
    path: str | Path,
    *,
    trajectory_id: str,
) -> dict[str, Any]:
    """Load an existing artifact, verify its identity, and validate its shape.

    Structural checks only: anchor checks ran at creation time; resume does
    not re-read the raw trajectory.
    """
    target = Path(path)
    if not target.is_file():
        raise ExtractionValidationError([f"extraction file does not exist: {target}"])
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExtractionValidationError([f"failed to read extraction file: {exc}"]) from exc
    except json.JSONDecodeError as exc:
        raise ExtractionValidationError([f"extraction file is not valid JSON: {exc}"]) from exc
    if not isinstance(payload, dict):
        raise ExtractionValidationError(["payload must be a JSON object"])
    reported = payload.get("trajectory_id")
    if reported != trajectory_id:
        raise ExtractionValidationError([f"trajectory_id mismatch: got {reported!r}, expected {trajectory_id!r}"])
    model_payload = {key: value for key, value in payload.items() if key in ("tasks", "skills", "behaviors")}
    return validate_extraction_record(model_payload, trajectory_id=trajectory_id)


# === Prompt ===


def build_extraction_prompt(*, trajectory_text: str, skill_names: list[str]) -> tuple[str, str]:
    """Return ``(system, user)`` for one extraction turn.

    The system half is constant across retries -- role, field semantics,
    rules -- so it survives compression and react session resumption. The
    user half is the inline trajectory text plus its truncation marker and
    the scope's skill-library listing.
    """
    system = """You are indexing one agent execution trajectory into a structured record.

If the trajectory contains a truncation marker naming a readable file, use Read to inspect the omitted content before finalizing. Otherwise do not call tools unnecessarily.

## Structured result fields

- `tasks`: one sentence describing ONLY the task the agent was given -- what was requested, its inputs, and the expected deliverable. Never mention the approach, tools, or libraries the agent chose; two trajectories solving the same task differently must produce near-identical `tasks` lines.
- `skills`: which skill documents from the library listing (given in the input) the agent consulted, invoked, or visibly followed. For each, `evidence` is a VERBATIM fragment copied exactly from the trajectory that shows the consultation (e.g. the line that reads or invokes the skill). An empty list means no skill was consulted. Never list a skill that is absent from the library listing.
- `behaviors`: the agent's activity split into semantic moves, in execution order. A move is one assistant message, or consecutive assistant messages carrying out one intention. Environment/tool/user/evaluator messages are feedback, never moves -- but DO tell their story:
  - `summary`: a free narrative of this move -- what the agent did, why, and how it ended (errors hit, fixes applied, verifications run, approaches abandoned and redone). Write it so a later reader understands this move without opening the trajectory. Outcomes and causality belong here.
  - `keywords`: 1-5 short fragments copied EXACTLY, character for character, from the agent's own output within this move (code snippets, API calls, distinctive phrases). They are grep anchors: each must appear verbatim in the trajectory file. Choose fragments specific to this move, not generic words. Never take keywords from tool results, error dumps, or other environment text.

## Rules

- Behaviors partition the agent's activity: everything the agent did belongs to exactly one behavior.
- `keywords` and `skills[].evidence` are exact copies; everything else is your own wording.
- Write `tasks` and every `summary` in the same language as the trajectory's dominant content. Do not translate: two trajectories doing the same task must share the task's own vocabulary, and translation splits shared terms apart.
- Do not evaluate whether the agent did well; describe what happened.

Return the complete structured result when done.
"""
    listed = "\n".join(f"- {name}" for name in skill_names) if skill_names else "(empty library)"
    user = f"""## Input

Skill library of this scope (the only legal values for `skills[].name`):
{listed}

Trajectory (full text, or a marked head+tail view):
<<<TRAJECTORY_BEGIN>>>
{trajectory_text}
<<<TRAJECTORY_END>>>
"""
    return system, user


# === Config ===


@dataclass(frozen=True)
class HybridConfig:
    """Extraction knobs read from the settings file (all keys required)."""

    # Per-trajectory char cap for the prompt body; truncation stamps the record.
    trajectory_char_cap: int

    @classmethod
    def from_settings(cls) -> "HybridConfig":
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_int

        return cls(
            trajectory_char_cap=setting_int("extract.trajectory_char_cap"),
        )


# === Runner ===


@dataclass
class ExtractionTaskResult:
    """Outcome of one trajectory's extraction attempt."""

    input_path: str
    output_path: str
    stem: str
    scope_id: str
    record: dict[str, Any] | None
    score_outcome: str | None
    error: Exception | None
    attempts: int = 0


@dataclass
class ExtractionBatchSummary:
    successes: list[ExtractionTaskResult] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return len(self.successes)

    @property
    def n_skipped(self) -> int:
        return len(self.failures)


def _output_max_attempts() -> int:
    """Maximum attempts for final-output correction."""
    return max(1, setting_int("agent.invokes_max_attempts"))


def join_supervised_score(
    record: dict[str, Any],
    *,
    workspace_root: str | Path,
    scope_id: str,
    stem: str,
) -> dict[str, Any]:
    """Join ``scores/<stem>.txt`` into ``record['score_outcome']`` when present.

    Mechanical transcription: the file's stripped text becomes the score
    value verbatim. The extraction model cannot emit this field.
    Unsupervised runs have no scores/ files, so the record passes through.
    """
    raw = read_score_text(scope_artifact_dir(workspace_root, scope_id, SCORE_ARTIFACT), stem)
    if not raw:
        return record
    out = dict(record)
    out["score_outcome"] = raw
    out["outcome_source"] = "supervised_score"
    return out


def _read_trajectory_text(path: Path) -> str:
    """Read the trajectory file as text. No parsing: the LLM reads it raw."""
    return path.read_text(encoding="utf-8", errors="replace")


def _output_retry_prompt(errors: list[str]) -> str:
    feedback = "; ".join(errors[:5])
    return (
        "The previous structured result failed final-output validation: "
        f"{feedback}. Correct those issues only, then return the complete "
        "structured result again."
    )


def _run_one(
    *,
    input_path: str,
    final_path: str,
    workspace_root: str,
    config: HybridConfig,
) -> ExtractionTaskResult:
    """Extract, anchor-validate, score-join, and persist one record."""
    source = Path(input_path)
    stem = source.stem
    scope_id = Path(final_path).parent.parent.name
    trajectory_dir = source.resolve().parent
    cwd_path = str(trajectory_dir)
    raw_text = _read_trajectory_text(source)
    trajectory_text, was_truncated = cut_text(
        raw_text,
        char_cap=config.trajectory_char_cap,
        rel_path=source.name,
        tail_fraction=0.5,
        what="chars of this trajectory's middle",
    )
    skill_names = scope_skill_names(workspace_root, scope_id)

    system_prompt, user_prompt = build_extraction_prompt(
        trajectory_text=trajectory_text,
        skill_names=sorted(skill_names),
    )
    schema = extraction_output_schema()
    attempts = _output_max_attempts()
    record: dict[str, Any] | None = None
    last_errors: list[str] = []
    session_id: str | None = None
    used_attempts = 0

    for attempt in range(attempts):
        used_attempts = attempt + 1
        if attempt:
            time.sleep(min(10.0, 2.0 * attempt))
        prompt = user_prompt if attempt == 0 or session_id is None else _output_retry_prompt(last_errors)
        if attempt > 0 and session_id is None:
            prompt += "\n\n" + _output_retry_prompt(last_errors)
        try:
            result = llm_invoke(
                system_prompt,
                prompt,
                tools=stage_tools(["Read", "Grep"], with_bash=False),
                output_schema=schema,
                permission_mode="plan",
                cwd=cwd_path,
                read_roots=[cwd_path],
                write_roots=[],
                session_id=session_id,
                trace_name=f"extraction/{stem}/attempt-{attempt + 1}",
                logs_root=Path(final_path).resolve().parent.parent / "logs",
            )
        except InvokeError as exc:
            return ExtractionTaskResult(
                input_path=input_path,
                output_path=final_path,
                stem=stem,
                scope_id=scope_id,
                record=None,
                score_outcome=None,
                error=exc,
                attempts=used_attempts,
            )
        except Exception as exc:  # noqa: BLE001 -- isolate one trajectory
            return ExtractionTaskResult(
                input_path=input_path,
                output_path=final_path,
                stem=stem,
                scope_id=scope_id,
                record=None,
                score_outcome=None,
                error=exc,
                attempts=used_attempts,
            )

        session_id = result.session_id
        try:
            payload = json.loads(result.text)
        except json.JSONDecodeError as exc:
            last_errors = [f"invalid JSON: {exc}"]
            continue
        try:
            record = validate_extraction_record(
                payload,
                trajectory_id=stem,
                raw_text=raw_text,
                skill_names=skill_names,
            )
        except ExtractionValidationError as exc:
            record = None
            last_errors = list(exc.errors)
            continue
        break

    if record is None:
        return ExtractionTaskResult(
            input_path=input_path,
            output_path=final_path,
            stem=stem,
            scope_id=scope_id,
            record=None,
            score_outcome=None,
            error=RuntimeError(
                f"extraction failed final-output validation for {scope_id}/{stem} "
                f"after {attempts} attempts; last errors: {last_errors[:3]}"
            ),
            attempts=used_attempts,
        )

    if was_truncated:
        record["trajectory_truncated"] = True
    try:
        record = join_supervised_score(
            record,
            workspace_root=workspace_root,
            scope_id=scope_id,
            stem=stem,
        )
    except Exception as exc:
        return ExtractionTaskResult(
            input_path=input_path,
            output_path=final_path,
            stem=stem,
            scope_id=scope_id,
            record=None,
            score_outcome=None,
            error=exc,
            attempts=used_attempts,
        )

    target = Path(final_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ExtractionTaskResult(
        input_path=input_path,
        output_path=final_path,
        stem=stem,
        scope_id=scope_id,
        record=record,
        score_outcome=record.get("score_outcome"),
        error=None,
        attempts=used_attempts,
    )


def _bounded_parallelism(parallelism: int, n_inputs: int) -> int:
    if n_inputs <= 0:
        return 0
    return max(1, min(int(parallelism), n_inputs))


def summarize_extraction_results(
    results: list[ExtractionTaskResult],
) -> ExtractionBatchSummary:
    summary = ExtractionBatchSummary()
    for item in results:
        if item.error is not None:
            summary.failures.append(
                {
                    "input_path": item.input_path,
                    "stem": item.stem,
                    "scope_id": item.scope_id,
                    "error": f"{type(item.error).__name__}: {item.error}",
                    "attempts": str(item.attempts),
                }
            )
            continue
        if item.record is None:
            summary.failures.append(
                {
                    "input_path": item.input_path,
                    "stem": item.stem,
                    "scope_id": item.scope_id,
                    "error": "RuntimeError: extraction returned no record",
                    "attempts": str(item.attempts),
                }
            )
            continue
        summary.successes.append(item)
        summary.output_paths.append(item.output_path)
    return summary


def build_extraction_records(
    inputs: list[str],
    outputs: list[str],
    *,
    workspace_root: str,
    parallelism: int,
    config: HybridConfig,
    progress_callback=None,
) -> list[ExtractionTaskResult]:
    """Run one LLM call per trajectory, in parallel, with per-call validation."""
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs length mismatch")
    pairs = list(zip(inputs, outputs))
    workers = _bounded_parallelism(parallelism, len(pairs))

    def invoke_one(index: int, pair: tuple[str, str]) -> ExtractionTaskResult:
        del index
        input_path, output_path = pair
        return _run_one(
            input_path=input_path,
            final_path=output_path,
            workspace_root=workspace_root,
            config=config,
        )

    return run_parallel_batch(
        pairs,
        invoke_one,
        workers=workers,
        progress_callback=progress_callback,
    )


__all__ = [
    "HybridConfig",
    "ExtractionTaskResult",
    "ExtractionBatchSummary",
    "ExtractionValidationError",
    "anchor_in",
    "anchor_variants",
    "extraction_output_schema",
    "validate_extraction_record",
    "load_and_validate_record",
    "build_extraction_prompt",
    "join_supervised_score",
    "build_extraction_records",
    "scope_skill_names",
    "summarize_extraction_results",
]
