"""Run OpenClaw locally with per-call configuration and session state."""

import json
import math
import tempfile
from pathlib import Path
from uuid import uuid4

from ...base import Agent
from ...typing import AgentResult
from ..cli import environment, parse_json_lines, run_cli, token_usage
from .config import OpenClawConfig

OPENCLAW_API_KEY_ENV = "MINDMEMOS_OPENCLAW_API_KEY"
"""Child-only credential referenced by the generated provider configuration."""

OPENCLAW_CONFIG_PATH_ENV = "OPENCLAW_CONFIG_PATH"
"""Native override for the configuration file."""

OPENCLAW_STATE_DIR_ENV = "OPENCLAW_STATE_DIR"
"""Native override for session and authentication storage."""

OPENCLAW_HOME_ENV = "OPENCLAW_HOME"
"""Native override for home-relative OpenClaw paths."""

OPENCLAW_AGENT_DIR_ENV = "OPENCLAW_AGENT_DIR"
"""Native override preventing inherited authentication storage from being reused."""

PI_CODING_AGENT_DIR_ENV = "PI_CODING_AGENT_DIR"
"""Underlying runtime's authentication directory override."""


@Agent.register("openclaw")
class OpenClawAgent(Agent[OpenClawConfig]):
    """Invoke the embedded harness using the shared asynchronous Agent contract."""

    config_type = OpenClawConfig
    """Settings validated by the shared registry."""

    async def run(self, *, system: str, user: str, workspace: Path) -> AgentResult:
        """Execute one isolated local agent turn.

        Args:
            system: Public instructions prepended to the task message.
            user: Public task prompt.
            workspace: Existing task directory; Skills may be exported to .agents/skills or skills.

        Returns:
            Final text, available native session events, and reported token usage.

        Raises:
            ValueError: Explicit connection settings are missing.
            RuntimeError: The CLI fails or returns an invalid or unsuccessful result.
        """
        self.config.require_connection()
        workspace = workspace.resolve()
        session_id = str(uuid4())
        prompt = f"System instructions:\n{system}\n\nUser:\n{user}" if system else user
        with tempfile.TemporaryDirectory(prefix="mindmemos-openclaw-") as directory:
            state = Path(directory)
            config_path = state / "openclaw.json"
            native_config = {
                "agents": {
                    "defaults": {
                        "workspace": str(workspace),
                        "skipBootstrap": True,
                        "model": {"primary": f"mindmemos/{self.config.model}"},
                    }
                },
                "models": {
                    "mode": "replace",
                    "providers": {
                        "mindmemos": {
                            "baseUrl": self.config.base_url,
                            "apiKey": "${" + OPENCLAW_API_KEY_ENV + "}",
                            "api": self.config.api,
                            "models": [
                                {
                                    "id": self.config.model,
                                    "name": self.config.model,
                                    "reasoning": self.config.reasoning_effort not in (None, "off"),
                                    "contextWindow": self.config.context_window,
                                    "maxTokens": self.config.max_completion_tokens,
                                }
                            ],
                        }
                    },
                },
                "skills": {"load": {"extraDirs": [str(workspace / ".agents/skills")], "watch": False}},
            }
            config_path.write_text(json.dumps(native_config), encoding="utf-8")
            overrides = environment(self.config, OPENCLAW_API_KEY_ENV)
            overrides.update(
                {
                    OPENCLAW_CONFIG_PATH_ENV: str(config_path),
                    OPENCLAW_STATE_DIR_ENV: str(state),
                    OPENCLAW_HOME_ENV: str(state / "home"),
                    OPENCLAW_AGENT_DIR_ENV: str(state / "agents/main/agent"),
                    PI_CODING_AGENT_DIR_ENV: str(state / "agents/main/agent"),
                }
            )
            command = [
                self.config.cli_path,
                "agent",
                "--local",
                "--json",
                "--session-id",
                session_id,
                "--timeout",
                str(math.ceil(self.config.timeout_s)),
            ]
            if self.config.reasoning_effort:
                command.extend(["--thinking", self.config.reasoning_effort])
            command.extend([*self.config.extra_args, "--message", prompt])
            output = await run_cli(command, workspace, self.config, overrides)
            result = _parse_result(output)
            transcript = state / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
            events = (
                parse_json_lines(transcript.read_text(encoding="utf-8"), cli_name="OpenClaw")
                if transcript.is_file()
                else []
            )
            result.trajectory = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                *events,
                *result.trajectory,
            ]
            return result


def _parse_result(stdout: str) -> AgentResult:
    """Validate the native local JSON envelope and normalize reported token counts."""
    try:
        result = json.loads(stdout)
    except ValueError as exc:
        raise RuntimeError("OpenClaw returned malformed JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("OpenClaw returned a non-object result")
    meta = result.get("meta")
    payloads = result.get("payloads")
    if not isinstance(meta, dict) or not isinstance(payloads, list):
        raise RuntimeError("OpenClaw returned an invalid result envelope")
    if (
        meta.get("aborted")
        or meta.get("error")
        or meta.get("yielded")
        or meta.get("stopReason") in ("error", "aborted", "tool_calls")
    ):
        raise RuntimeError("OpenClaw did not complete")
    if any(not isinstance(payload, dict) or payload.get("isError") for payload in payloads):
        raise RuntimeError("OpenClaw returned an error payload")
    output = "\n".join(payload["text"] for payload in payloads if isinstance(payload.get("text"), str))
    if not output.strip():
        raise RuntimeError("OpenClaw returned an empty result")
    agent_meta = meta.get("agentMeta", {})
    raw_usage = agent_meta.get("usage") if isinstance(agent_meta, dict) else None
    usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    for native, normalized in (("input", "input_tokens"), ("output", "output_tokens"), ("total", "total_tokens")):
        if type(usage.get(native)) is int and usage[native] >= 0:
            usage[normalized] = usage[native]
    # Native totals may include cached tokens that input + output omit.
    if "total_tokens" not in usage:
        usage = token_usage(usage)
    return AgentResult(output=output, trajectory=[result], usage=usage)
