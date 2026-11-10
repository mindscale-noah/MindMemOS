"""Codex exec integration using the shared synchronous agent contract."""

import json
from pathlib import Path

from ...base import Agent
from ...typing import AgentResult
from ..cli import environment, parse_json_lines, run_cli, token_usage
from ..constants import CODEX_API_KEY_ENV
from .config import CodexConfig


@Agent.register("codex")
class CodexAgent(Agent[CodexConfig]):
    """Run Codex with native tools in a caller-prepared workspace."""

    config_type = CodexConfig
    """Settings validated by the shared registry."""

    async def run(self, *, system: str, user: str, workspace: Path) -> AgentResult:
        """Execute a complete native agent run.

        Args:
            system: Public developer instructions.
            user: Public task prompt.
            workspace: Existing task directory; export Skills to .agents/skills first.

        Returns:
            Final answer, native event trajectory, and reported token usage.
        """
        self.config.require_connection()
        command = [
            self.config.cli_path,
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--config",
            'approval_policy="never"',
        ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        if self.config.reasoning_effort:
            command.extend(["--config", f"model_reasoning_effort={json.dumps(self.config.reasoning_effort)}"])
        if self.config.base_url is not None:
            url = self.config.base_url
            for setting in [
                'model_provider="mindmemos"',
                'model_providers.mindmemos.name="MindMemOS"',
                f"model_providers.mindmemos.base_url={json.dumps(url.rstrip('/'))}",
                f"model_providers.mindmemos.env_key={json.dumps(CODEX_API_KEY_ENV)}",
                'model_providers.mindmemos.wire_api="responses"',
                "model_providers.mindmemos.supports_websockets=false",
            ]:
                command.extend(["--config", setting])
        if system:
            command.extend(["--config", f"developer_instructions={json.dumps(system)}"])
        command.extend([*self.config.extra_args, "-"])
        stream = parse_json_lines(
            (await self._run_cli(command, workspace, environment(self.config, CODEX_API_KEY_ENV), user)),
            cli_name="Codex",
        )
        terminal = next(
            (event for event in reversed(stream) if event.get("type") in {"turn.completed", "turn.failed"}), None
        )
        if terminal is None or terminal["type"] == "turn.failed":
            raise RuntimeError(f"Codex did not complete: {terminal}")
        output = ""
        for event in reversed(stream):
            item = event.get("item")
            if event.get("type") == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
                output = item.get("text", "")
                break
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("Codex returned an empty result")
        return AgentResult(
            output=output,
            trajectory=[{"role": "developer", "content": system}, {"role": "user", "content": user}, *stream],
            usage=token_usage(terminal.get("usage")),
        )

    async def _run_cli(self, command: list[str], workspace: Path, overrides: dict[str, str], prompt: str) -> str:
        """Execute the prepared command; subclasses may specialize invocation options."""
        return await run_cli(command, workspace, self.config, overrides, prompt)
