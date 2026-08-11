#!/usr/bin/env python3
"""Script-side live smoke evaluation for the built-in ``mindmemos_skill`` agents.

The evaluation uses a per-run random token that exists only inside the
injected Skill.  An agent therefore passes the adherence check only when its
final answer proves that it received and followed the Skill.  The script also
checks family-specific Skill-loading evidence and validates the canonical
Trajectory -> TrajectoryRecord -> Trajectory round trip.

Examples:

    UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache uv run python \
      scripts/run_mindmemos_skill_experiment.sh --config \
        config/mindmemos_skill/agent_evaluation/local/default.yaml

    UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache uv run python \
      scripts/run_mindmemos_skill_experiment.sh --config \
        config/mindmemos_skill/agent_evaluation/local/default.yaml \
        --set evaluation.agents='[react]' --set evaluation.strict=true
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mindmemos_skill.agents import get_agent, list_agents
from mindmemos_skill.llm import LLMClient, build_router
from mindmemos_skill.typing import (
    AgentExecutionRequest,
    AgentType,
    Environment,
    Rollout,
    RolloutType,
    Skill,
    SkillInjectionMode,
    SkillUsageType,
    Task,
    Trajectory,
    TrajectoryStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ENV_FILE = REPO_ROOT / ".skill.env"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "agent_skill_eval"
DEFAULT_MODEL = "claude-sonnet-4-6"
SKILL_NAME = "mindmemos-runtime-eval"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_env_file(path: Path) -> dict[str, str]:
    """Load the small KEY=VALUE subset needed by this standalone script."""

    if not path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {path}")

    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"invalid environment assignment at {path}:{line_no}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid environment name at {path}:{line_no}: {name!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def _required_env(values: dict[str, str], name: str, env_file: Path) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is missing or empty in {env_file}")
    return value


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute http(s) URL")


def _configure_claude_gateway(*, api_key: str, base_url: str) -> None:
    """Map the requested OpenAI-compatible credentials to Claude Code names.

    The configured gateway advertises Anthropic and OpenAI protocol support.
    ``ANTHROPIC_AUTH_TOKEN`` selects bearer-token authentication, matching the
    ``Authorization: Bearer`` convention used by ``OPENAI_API_KEY``.
    """

    os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
    os.environ["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")


def _build_skill(*, run_id: str, expected_token: str) -> Skill:
    content = f"""---
name: {SKILL_NAME}
description: Runtime evaluation Skill with a private per-run completion token.
---

# MindMemOS Runtime Evaluation

When the user asks to use this Skill, complete the task by replying with exactly
the following token and no other text, punctuation, or Markdown:

{expected_token}
"""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return Skill(
        skill_id=f"eval-skill-{run_id}",
        version_id=f"eval-version-{run_id}",
        version_label="1.0.0",
        content_hash=f"sha256:{digest}",
        name=SKILL_NAME,
        description="Return a private token to prove Skill discovery and adherence.",
        blob={"SKILL.md": content},
        created_at=datetime.now(UTC),
        metadata={"evaluation": "agent_skill_runtime"},
    )


def _build_request(*, case_id: str, run_id: str, skill: Skill) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        trajectory_id=f"trajectory-{run_id}-{case_id}",
        task=Task(
            task_id=f"task-{run_id}-{case_id}",
            instruction=(
                f"Use the injected Skill named {SKILL_NAME}. Follow its instructions exactly. "
                "Return only the final answer required by that Skill."
            ),
            system_prompt="This is a deterministic runtime evaluation. Use the requested injected Skill.",
            tags=["agent-smoke", "skill-injection", "trajectory"],
        ),
        rollout=Rollout(
            rollout_id=f"rollout-{run_id}-{case_id}",
            attempt_no=0,
            rollout_type=RolloutType.EVALUATE,
        ),
        environment=Environment(running_dir=str(REPO_ROOT)),
        skills=[skill],
        metadata={"evaluation_run_id": run_id, "case_id": case_id},
    )


def _last_assistant_text(events: Sequence[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("role") != "assistant":
            continue
        content = event.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _skill_tool_calls(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in events:
        for call in event.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and function.get("name") == "skill":
                calls.append(call)
    return calls


def _trajectory_round_trip_ok(trajectory: Trajectory) -> tuple[bool, str | None]:
    try:
        validated = Trajectory.model_validate_json(trajectory.model_dump_json())
        restored = Trajectory.from_record(validated.to_record())
        expected = validated.model_dump(mode="json")
        actual = restored.model_dump(mode="json")
        if actual != expected:
            return False, "TrajectoryRecord round trip changed the trajectory payload"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _assess_trajectory(
    trajectory: Trajectory,
    *,
    expected_token: str,
    mode: SkillInjectionMode,
) -> dict[str, Any]:
    binding = next((item for item in trajectory.skill_bindings if item.name == SKILL_NAME), None)
    binding_loaded = binding is not None and binding.usage is SkillUsageType.INJECTED
    tool_calls = _skill_tool_calls(trajectory.events)
    final_answer = _last_assistant_text(trajectory.events)
    trajectory_valid, trajectory_error = _trajectory_round_trip_ok(trajectory)

    prompt_exposure = mode is SkillInjectionMode.SYSTEM_PROMPT and any(
        event.get("role") == "system" and isinstance(event.get("content"), str) and expected_token in event["content"]
        for event in trajectory.events
    )
    native_discovery_observable = mode is not SkillInjectionMode.SYSTEM_PROMPT
    skill_discovered = bool(tool_calls and binding_loaded) if native_discovery_observable else None
    skill_injected = binding_loaded and (bool(tool_calls) or prompt_exposure)
    skill_applied = final_answer == expected_token
    agent_ran = trajectory.execution.status is TrajectoryStatus.SUCCEEDED and bool(final_answer)
    passed = bool(
        agent_ran and skill_injected and skill_applied and trajectory_valid and (skill_discovered is not False)
    )

    return {
        "passed": passed,
        "checks": {
            "agent_ran": agent_ran,
            "skill_injected": skill_injected,
            "native_skill_discovery": skill_discovered,
            "skill_applied_to_task": skill_applied,
            "trajectory_produced_and_round_trips": trajectory_valid,
        },
        "evidence": {
            "trajectory_status": trajectory.execution.status.value,
            "turns": trajectory.execution.n_turn,
            "event_count": len(trajectory.events),
            "skill_tool_call_count": len(tool_calls),
            "binding_usage": binding.usage.value if binding and binding.usage else None,
            "injection_mode": binding.injection_mode.value if binding and binding.injection_mode else None,
            "final_answer_matches_private_token": skill_applied,
            "native_discovery_observable": native_discovery_observable,
        },
        "error": trajectory.execution.error_info or trajectory_error,
    }


def _redact(value: Any, *, secrets_to_hide: Sequence[str]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in secrets_to_hide:
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        return redacted
    if isinstance(value, list):
        return [_redact(item, secrets_to_hide=secrets_to_hide) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets_to_hide=secrets_to_hide) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Any, *, secrets_to_hide: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _redact(payload, secrets_to_hide=secrets_to_hide)
    path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_react_llm(*, api_key: str, base_url: str, model: str, timeout: float) -> LLMClient:
    router, _ = build_router(
        {
            "endpoints": [
                {
                    "model": f"openai/{model}",
                    "api_key": api_key,
                    "api_base": base_url.rstrip("/"),
                    "timeout": timeout,
                    "num_retries": 0,
                }
            ]
        },
        LLMClient.ALIAS,
        num_retries=0,
    )
    return LLMClient(router, default_model=LLMClient.ALIAS, max_attempts=1)


async def _run_case(
    *,
    case_id: str,
    agent_type: AgentType,
    mode: SkillInjectionMode,
    model: str,
    timeout: float,
    request: AgentExecutionRequest,
    llm: LLMClient | None,
    expected_token: str,
    output_dir: Path,
    secrets_to_hide: Sequence[str],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": LLMClient.ALIAS if agent_type is AgentType.REACT else model,
        "max_turns": 4,
        "skill_injection_mode": mode,
    }
    kwargs: dict[str, Any] = {}
    if agent_type is AgentType.REACT:
        kwargs["llm"] = llm
        config.update({"temperature": 0, "max_tokens": 128})
    elif agent_type is AgentType.CLAUDE:
        config.update({"timeout_seconds": timeout, "dangerously_skip_permissions": True})
    elif agent_type is AgentType.CLAUDE_SDK:
        config.update({"permission_mode": "bypassPermissions"})

    started_at = datetime.now(UTC)
    trajectory: Trajectory | None = None
    harness_error: str | None = None
    try:
        agent = get_agent(agent_type=agent_type, config=config, **kwargs)
        trajectory = await asyncio.wait_for(agent.execute(request), timeout=timeout + 5)
    except Exception as exc:
        harness_error = f"{type(exc).__name__}: {exc}"

    result: dict[str, Any] = {
        "case_id": case_id,
        "agent_type": agent_type.value,
        "skill_injection_mode": mode.value,
        "model": model,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "trajectory_file": None,
    }
    if trajectory is None:
        result.update(
            {
                "passed": False,
                "checks": {
                    "agent_ran": False,
                    "skill_injected": False,
                    "native_skill_discovery": False if mode is not SkillInjectionMode.SYSTEM_PROMPT else None,
                    "skill_applied_to_task": False,
                    "trajectory_produced_and_round_trips": False,
                },
                "evidence": {},
                "error": harness_error or "agent returned no trajectory",
            }
        )
        return _redact(result, secrets_to_hide=secrets_to_hide)

    trajectory_file = output_dir / "trajectories" / f"{case_id}.json"
    _write_json(
        trajectory_file,
        trajectory.model_dump(mode="json"),
        secrets_to_hide=secrets_to_hide,
    )
    result["trajectory_file"] = str(trajectory_file)
    result.update(_assess_trajectory(trajectory, expected_token=expected_token, mode=mode))
    return _redact(result, secrets_to_hide=secrets_to_hide)


def _selected_cases(
    agent_names: Sequence[str], react_modes: Sequence[str]
) -> list[tuple[str, AgentType, SkillInjectionMode]]:
    requested = set(agent_names)
    if "all" in requested:
        requested = {"react", "claude", "claude-sdk"}

    cases: list[tuple[str, AgentType, SkillInjectionMode]] = []
    if "react" in requested:
        for raw_mode in react_modes:
            mode = SkillInjectionMode(raw_mode)
            cases.append((f"react_{mode.value}", AgentType.REACT, mode))
    if "claude" in requested:
        cases.append(("claude_cli_filesystem", AgentType.CLAUDE, SkillInjectionMode.FILESYSTEM))
    if "claude-sdk" in requested:
        cases.append(("claude_sdk_filesystem", AgentType.CLAUDE_SDK, SkillInjectionMode.FILESYSTEM))
    return cases


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", help=f"Gateway model id (default: env or {DEFAULT_MODEL})")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=("all", "react", "claude", "claude-sdk"),
        default=["all"],
    )
    parser.add_argument(
        "--react-modes",
        nargs="+",
        choices=(SkillInjectionMode.TOOL.value, SkillInjectionMode.SYSTEM_PROMPT.value),
        default=[SkillInjectionMode.TOOL.value, SkillInjectionMode.SYSTEM_PROMPT.value],
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-case timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="Run output directory")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any selected case fails")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


async def _async_main(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    env_file = args.env_file.expanduser().resolve()
    env_values = _load_env_file(env_file)
    api_key = _required_env(env_values, "OPENAI_API_KEY", env_file)
    base_url = _required_env(env_values, "OPENAI_BASE_URL", env_file)
    _validate_base_url(base_url)
    model = args.model or env_values.get("OPENAI_MODEL") or env_values.get("AGENT_EVAL_MODEL") or DEFAULT_MODEL

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    output_dir = (args.output_dir or DEFAULT_RESULTS_ROOT / run_id).expanduser().resolve()
    expected_token = f"MINDMEMOS_SKILL_EVAL_PASS_{secrets.token_hex(5).upper()}"
    skill = _build_skill(run_id=run_id, expected_token=expected_token)
    cases = _selected_cases(args.agents, args.react_modes)
    if not cases:
        raise ValueError("no evaluation cases selected")

    _configure_claude_gateway(api_key=api_key, base_url=base_url)
    react_llm = _build_react_llm(api_key=api_key, base_url=base_url, model=model, timeout=args.timeout)
    credentials_to_hide = (api_key, base_url)

    results: list[dict[str, Any]] = []
    for case_id, agent_type, mode in cases:
        request = _build_request(case_id=case_id, run_id=run_id, skill=skill)
        print(f"[{case_id}] running {agent_type.value} with {mode.value} injection...", flush=True)
        result = await _run_case(
            case_id=case_id,
            agent_type=agent_type,
            mode=mode,
            model=model,
            timeout=args.timeout,
            request=request,
            llm=react_llm if agent_type is AgentType.REACT else None,
            expected_token=expected_token,
            output_dir=output_dir,
            secrets_to_hide=credentials_to_hide,
        )
        results.append(result)
        print(f"[{case_id}] {'PASS' if result['passed'] else 'FAIL'}", flush=True)

    passed_count = sum(bool(result["passed"]) for result in results)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "environment_file": str(env_file),
        "credentials": {
            "openai_api_key_configured": True,
            "openai_base_url_configured": True,
            "plaintext_credentials_persisted": False,
        },
        "model": model,
        "registered_agents": list_agents(),
        "private_token_redacted_from_report": True,
        "trajectory_artifacts_preserve_challenge_evidence": True,
        "summary": {
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "total": len(results),
            "all_passed": passed_count == len(results),
        },
        "cases": results,
        "notes": [
            "For tool/filesystem injection, native Skill discovery requires family-specific Skill tool evidence.",
            "For system_prompt injection, discovery is not separately observable; the check records prompt exposure and adherence.",
        ],
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report, secrets_to_hide=credentials_to_hide)
    return report, report_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report, report_path = asyncio.run(_async_main(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Report: {report_path}")
    return 1 if args.strict and not report["summary"]["all_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
