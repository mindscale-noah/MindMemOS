"""Run Claude Code through the shared native agent contract."""

from os import getenv
from pathlib import Path

from ....infra.model_access.platform import PLATFORM_CHAT_MODEL
from ...base import Agent
from ...typing import AgentResult
from ..cli import environment, parse_json_lines, run_cli, token_usage
from ..constants import ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_BASE_URL_ENV, CLAUDE_MAX_TOKENS_ENV
from .config import ClaudeConfig

_NO_PROXY_ENV = "NO_PROXY"
_NO_PROXY_LOWER_ENV = "no_proxy"
_DISABLE_NONESSENTIAL_TRAFFIC_ENV = "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"


@Agent.register("claude")
class ClaudeAgent(Agent[ClaudeConfig]):
    """Run Claude Code with its native tools and filesystem Skill discovery."""

    config_type = ClaudeConfig
    """Settings validated by the shared registry."""

    async def run(self, *, system: str, user: str, workspace: Path) -> AgentResult:
        """Execute a complete native agent run.

        Args:
            system: Public system instructions.
            user: Public task prompt.
            workspace: Existing task directory; export Skills to .claude/skills first.

        Returns:
            Final answer, native event trajectory, and reported token usage.
        """
        use_platform = self.config.transport == "platform_gateway"
        if not use_platform:
            self.config.require_connection()
        command = [
            self.config.cli_path,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
        ]
        if use_platform or self.config.api_key is not None or self.config.base_url is not None:
            # Settings-file env values otherwise override the explicit process environment.
            command.extend(["--setting-sources", ""])
        model = PLATFORM_CHAT_MODEL if use_platform else self.config.model
        if model:
            command.extend(["--model", model])
        if self.config.reasoning_effort:
            command.extend(["--effort", self.config.reasoning_effort])
        if self.config.max_turns is not None:
            command.extend(["--max-turns", str(self.config.max_turns)])
        if system:
            command.extend(["--system-prompt", system])
        command.extend(self.config.extra_args)
        overrides = {}
        if self.config.max_completion_tokens is not None:
            overrides[CLAUDE_MAX_TOKENS_ENV] = str(self.config.max_completion_tokens)
        if use_platform:
            from ....infra.model_access.anthropic import open_messages_relay

            async with open_messages_relay() as (base_url, capability):
                overrides.update(
                    {
                        ANTHROPIC_BASE_URL_ENV: base_url,
                        ANTHROPIC_AUTH_TOKEN_ENV: capability,
                        _NO_PROXY_ENV: f"{getenv(_NO_PROXY_ENV, '')},127.0.0.1".lstrip(","),
                        _NO_PROXY_LOWER_ENV: f"{getenv(_NO_PROXY_LOWER_ENV, '')},127.0.0.1".lstrip(","),
                        _DISABLE_NONESSENTIAL_TRAFFIC_ENV: "1",
                    }
                )
                stdout = await self._run_cli(command, workspace, overrides, user)
        else:
            overrides.update(environment(self.config, ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_BASE_URL_ENV))
            stdout = await self._run_cli(command, workspace, overrides, user)
        stream = parse_json_lines(stdout, cli_name="Claude")
        terminal = next((event for event in reversed(stream) if event.get("type") == "result"), None)
        if terminal is None or terminal.get("is_error") or terminal.get("subtype") not in {None, "success"}:
            raise RuntimeError(f"Claude did not complete: {terminal}")
        output = terminal.get("result")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("Claude returned an empty result")
        return AgentResult(
            output=output,
            trajectory=[{"role": "system", "content": system}, {"role": "user", "content": user}, *stream],
            usage=token_usage(terminal.get("usage")),
        )

    async def _run_cli(self, command: list[str], workspace: Path, overrides: dict[str, str], prompt: str) -> str:
        """Execute the prepared command; subclasses may specialize invocation options."""
        return await run_cli(command, workspace, self.config, overrides, prompt)
