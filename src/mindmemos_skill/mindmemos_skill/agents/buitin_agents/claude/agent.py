"""Claude Code integration using the shared synchronous agent contract."""

from pathlib import Path

from ...base import Agent
from ...typing import AgentResult
from ..cli import environment, parse_json_lines, run_cli, token_usage
from ..constants import ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_BASE_URL_ENV, CLAUDE_MAX_TOKENS_ENV
from .config import ClaudeConfig


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
        if self.config.api_key is not None or self.config.base_url is not None:
            # Settings-file env values otherwise override the explicit process environment.
            command.extend(["--setting-sources", ""])
        if self.config.model:
            command.extend(["--model", self.config.model])
        if self.config.reasoning_effort:
            command.extend(["--effort", self.config.reasoning_effort])
        if self.config.max_turns is not None:
            command.extend(["--max-turns", str(self.config.max_turns)])
        if system:
            command.extend(["--system-prompt", system])
        command.extend(self.config.extra_args)
        overrides = environment(
            self.config,
            ANTHROPIC_AUTH_TOKEN_ENV if self.config.base_url else ANTHROPIC_API_KEY_ENV,
            ANTHROPIC_BASE_URL_ENV,
        )
        if self.config.max_completion_tokens is not None:
            overrides[CLAUDE_MAX_TOKENS_ENV] = str(self.config.max_completion_tokens)
        stream = parse_json_lines((await self._run_cli(command, workspace, overrides, user)), cli_name="Claude")
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
