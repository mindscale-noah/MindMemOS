"""DeepSeek Harness integration with retained per-run session events."""

import tempfile
from pathlib import Path

import yaml

from ...base import Agent
from ...typing import AgentResult
from ..cli import environment, run_cli
from ..constants import DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL_ENV, DSH_HOME_ENV
from .config import DeepSeekHarnessConfig
from .session import read_session_events


@Agent.register("deepseek_harness")
class DeepSeekHarnessAgent(Agent[DeepSeekHarnessConfig]):
    """Run the native headless harness in a caller-prepared workspace."""

    config_type = DeepSeekHarnessConfig
    """Settings validated by the shared registry."""

    def run(self, *, system: str, user: str, workspace: Path) -> AgentResult:
        """Execute a complete native agent run and validate its terminal event.

        Args:
            system: Public instructions prepended to the task prompt.
            user: Public task prompt.
            workspace: Existing task directory; export Skills to .dsh/skills first.

        Returns:
            Final answer and native events; usage stays empty when unavailable.
        """
        self.config.require_connection()
        workspace = workspace.resolve()
        audit = Path(tempfile.mkdtemp(prefix=".mindmemos-dsh-", dir=workspace))
        rows = [
            {"id": "session-persistence-jsonl", "config": {"root": str(audit / "sessions"), "compression": "none"}},
            {
                "id": "skill-filesystem",
                "config": {
                    "includeDefaultRoots": False,
                    "watch": False,
                    "customSkillDirs": [str(workspace / ".dsh/skills")],
                },
            },
        ]
        if self.config.model:
            rows.append(
                {"id": "agent-default-model", "config": {"provider": self.config.provider, "model": self.config.model}}
            )
        patch = audit / "patch.yml"
        patch.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")
        prompt = f"System instructions:\n{system}\n\nUser:\n{user}" if system else user
        command = [
            self.config.cli_path,
            "--profile",
            self.config.profile,
            "--patch",
            str(patch),
            *self.config.extra_args,
            prompt,
        ]
        overrides = environment(self.config, DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL_ENV)
        overrides[DSH_HOME_ENV] = str(audit / "home")
        output = run_cli(command, workspace, self.config, overrides).rstrip("\n")
        _, events = read_session_events(audit / "sessions")
        if not output.strip():
            raise RuntimeError("DeepSeek Harness returned an empty result")
        return AgentResult(
            output=output,
            trajectory=[{"role": "system", "content": system}, {"role": "user", "content": user}, *events],
        )
