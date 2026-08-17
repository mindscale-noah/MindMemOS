"""DeepSeek Harness CLI agent implementation.

Wraps ``dsh --profile headless "<task>"`` as a MindMemOS agent: one fresh
Harness Agent answers the task, prints its final assistant message to stdout,
and exits ``0`` when the final turn completed (``1`` otherwise, with a
``dsh: <code>: <message>`` line on stderr).

Two execution surfaces are supported:

- :meth:`execute` — full rollouts.  Skills are materialized into a temporary
  ``.dsh/skills/`` tree, while the ``dsh`` process keeps the environment's
  working directory so task files remain accessible.
- :meth:`respond` — interactive benchmark environments (for example LiveMath)
  that own their conversation and drive the agent turn by turn; each turn is
  one ``dsh`` run whose task text serializes the environment's messages.

A temporary ``--patch`` overlay forwards ``model``/``provider`` to the booted
profile's ``agent-default-model`` row and registers a materialized Skills
root through an isolated ``skill-filesystem`` provider.  Every invocation also
uses an Agent-owned private ``$DSH_HOME`` so user settings cannot override the
requested model or provider.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...llm import ChatResponse
from ...registry import ComponentType, register
from ...typing import AgentExecutionRequest, AgentType, SkillInjectionMode, Trajectory
from ..base import Agent
from .config import _DEFAULT_PROFILE, DeepSeekHarnessAgentConfig
from .skill_runtime import DeepSeekHarnessSkillRuntime

_DEFAULT_PROVIDER = "deepseek-official"
_NPX_PACKAGE = "@deepseek-ai/dsh"


@dataclass(slots=True)
class _HeadlessResult:
    """Normalized outcome of one ``dsh --profile headless`` run."""

    final_text: str
    stderr_text: str
    returncode: int
    attempts: int = 1


@register(type=ComponentType.AGENT, name=AgentType.DEEPSEEK_HARNESS.value, capabilities={"execute", "respond"})
class DeepSeekHarnessAgent(Agent[DeepSeekHarnessAgentConfig]):
    """Run tasks through the DeepSeek Harness ``headless`` one-shot app."""

    agent_type = AgentType.DEEPSEEK_HARNESS
    config_type = DeepSeekHarnessAgentConfig
    skill_runtime_types = {
        SkillInjectionMode.FILESYSTEM: DeepSeekHarnessSkillRuntime,
        SkillInjectionMode.SYSTEM_PROMPT: DeepSeekHarnessSkillRuntime,
    }

    def __init__(self, config: DeepSeekHarnessAgentConfig | Mapping[str, Any]) -> None:
        super().__init__(config)
        self._cli_command: tuple[str, ...] | None = None
        self._dsh_home = tempfile.TemporaryDirectory(prefix="mindmemos_dsh_home_")
        self._profile_init_lock = asyncio.Lock()
        self._initialized_profiles: set[str] = set()

    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        started_at = time.time()
        config = self.config.with_overrides(request.options)

        trajectory_messages: list[dict[str, Any]] = []
        if request.task.system_prompt:
            trajectory_messages.append({"role": "system", "content": request.task.system_prompt})
        trajectory_messages.append({"role": "user", "content": request.task.instruction})

        # Resolve CLI path early so we don't create temp files on failure.
        try:
            cli = self._resolve_cli()
        except RuntimeError as exc:
            return self._error_result(request, started_at, trajectory_messages, str(exc))

        timeout = config.timeout_seconds
        runtime_metadata: dict[str, Any] = {}
        try:
            async with self.on_skill_runtime_task(request, mode=config.skill_injection_mode) as injection:
                runtime_metadata = dict(injection.metadata)
                trajectory_messages = self.apply_skill_injection(trajectory_messages, injection)
                result = await self._run_headless(
                    cli=cli,
                    config=config,
                    task_text=_compose_headless_task(
                        user_text=request.task.instruction,
                        system_text=_joined_system_instructions(trajectory_messages),
                    ),
                    skills_root=self._skills_root(injection.workspace),
                    cwd=request.environment.running_dir,
                )
        except asyncio.TimeoutError:
            return self._error_result(
                request,
                started_at,
                trajectory_messages,
                f"DeepSeek Harness CLI timed out after {timeout}s",
            )
        except Exception as exc:
            return self._error_result(
                request,
                started_at,
                trajectory_messages,
                f"DeepSeek Harness CLI execution failed: {exc}",
            )

        is_success = result.returncode == 0 and bool(result.final_text.strip())
        if result.final_text:
            trajectory_messages.append({"role": "assistant", "content": result.final_text})
        empty_result_error = "dsh exited successfully without a final assistant message"
        return self._build_trajectory(
            request=request,
            config=config,
            messages=trajectory_messages,
            started_at=started_at,
            ended_at=time.time(),
            n_turn=1 if is_success else 0,
            is_success=is_success,
            error_info=(
                result.stderr_text
                or (None if is_success else empty_result_error if result.returncode == 0 else f"dsh exited with code {result.returncode}")
            ),
            metadata={
                **runtime_metadata,
                "profile": config.profile,
                "headless_attempts": result.attempts,
                **({"model": config.model} if config.model else {}),
                **({"provider": config.provider} if config.provider else {}),
            },
        )

    async def respond(
        self,
        request: AgentExecutionRequest,
        messages: list[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = (),
    ) -> ChatResponse:
        """Generate one benchmark-controlled turn through one ``dsh`` run.

        The environment owns the conversation and already embedded any Skills
        into the system message it passes here; this method serializes those
        messages into the headless task text.  Failures raise so the
        environment records a failed trajectory with the harness diagnostics.
        """

        if tools:
            raise TypeError("DeepSeek Harness headless respond() does not support environment-provided tools")
        config = self.config.with_overrides(request.options)
        cli = self._resolve_cli()
        result = await self._run_headless(
            cli=cli,
            config=config,
            task_text=_serialize_messages(messages),
            cwd=request.environment.running_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr_text or f"dsh exited with code {result.returncode}")
        if not result.final_text.strip():
            raise RuntimeError("dsh exited successfully without a final assistant message")
        return ChatResponse(
            finish_reason="stop",
            content=result.final_text,
            model=config.model or "",
        )

    async def _run_headless(
        self,
        *,
        cli: Sequence[str],
        config: DeepSeekHarnessAgentConfig,
        task_text: str,
        skills_root: str | None = None,
        cwd: str | None = None,
    ) -> _HeadlessResult:
        """Run one isolated ``dsh`` invocation and normalize its output."""

        patch_path: str | None = None
        stdout_bytes = b""
        stderr_bytes = b""
        try:
            with _stage_skills_in_workspace(skills_root, cwd=cwd) as accessible_skills_root:
                patch_path = self._write_patch_overlay(config, skills_root=accessible_skills_root)
                command = self._build_command(cli=cli, config=config, task=task_text, patch_path=patch_path)
                env = self._build_env(self._dsh_home.name)
                init_error = await self._initialize_profile(
                    cli=cli,
                    config=config,
                    cwd=cwd,
                    env=env,
                )
                if init_error is not None:
                    return init_error
                for attempts in range(1, config.empty_output_retries + 2):
                    stdout_bytes, stderr_bytes, returncode = await _communicate(
                        command,
                        cwd=cwd,
                        env=env,
                        timeout=config.timeout_seconds,
                    )
                    if returncode != 0 or stdout_bytes.strip():
                        break
        finally:
            if patch_path is not None:
                try:
                    os.unlink(patch_path)
                except OSError:
                    pass

        stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr_text = (stderr_bytes.decode("utf-8", errors="replace") or "").strip() if stderr_bytes else ""
        return _HeadlessResult(
            final_text=stdout_text.rstrip("\n"),
            stderr_text=stderr_text,
            returncode=returncode,
            attempts=attempts,
        )

    async def _initialize_profile(
        self,
        *,
        cli: Sequence[str],
        config: DeepSeekHarnessAgentConfig,
        cwd: str | None,
        env: Mapping[str, str],
    ) -> _HeadlessResult | None:
        """Initialize each profile once in this Agent's private DSH_HOME."""

        if config.profile in self._initialized_profiles:
            return None
        async with self._profile_init_lock:
            if config.profile in self._initialized_profiles:
                return None
            _, stderr, returncode = await _communicate(
                [*cli, "--profile", config.profile, "--dump-config"],
                cwd=cwd,
                env=env,
                timeout=config.timeout_seconds,
            )
            if returncode != 0:
                return _HeadlessResult(
                    final_text="",
                    stderr_text=stderr.decode("utf-8", errors="replace").strip(),
                    returncode=returncode,
                    attempts=0,
                )
            self._initialized_profiles.add(config.profile)
        return None

    def _build_command(
        self,
        *,
        cli: Sequence[str],
        config: DeepSeekHarnessAgentConfig,
        task: str,
        patch_path: str | None,
    ) -> list[str]:
        command = [*cli, "--profile", config.profile]
        if patch_path is not None:
            command += ["--patch", patch_path]
        command.append(task)
        return command

    @staticmethod
    def _skills_root(workspace: str | None) -> str | None:
        """Return the materialized Skills root, or ``None`` when nothing was injected."""

        if workspace is None:
            return None
        return os.path.join(workspace, ".dsh", "skills")

    def _write_patch_overlay(self, config: DeepSeekHarnessAgentConfig, *, skills_root: str | None) -> str | None:
        """Write a ``--patch`` overlay overriding model selection and Skill roots.

        The ``agent-default-model`` row selects the provider/model when
        configured.  The ``skill-filesystem`` row registers the materialized
        Skills root through ``customSkillDirs``, so the Harness discovers the
        injected Skills regardless of the project root it infers from ``cwd``
        (the nearest ``.git`` ancestor would otherwise win over a temporary
        workspace inside a checkout).
        """

        rows: list[str] = []
        if config.model or config.provider:
            provider = config.provider or _DEFAULT_PROVIDER
            model = config.model or ""
            if model:
                rows.append(
                    "- id: agent-default-model\n"
                    "  config:\n"
                    f"    provider: {provider}\n"
                    f"    model: {model}\n"
                )
        if skills_root is not None:
            rows.append(
                "- id: skill-filesystem\n"
                "  name: '@deepseek-ai/dsh-skill-filesystem'\n"
                "  config:\n"
                "    includeDefaultRoots: false\n"
                "    watch: false\n"
                "    customSkillDirs:\n"
                f"      - {skills_root}\n"
            )
        if not rows:
            return None
        document = "\n".join(rows)
        file = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", prefix="mindmemos_dsh_", delete=False, encoding="utf-8")
        try:
            file.write(document)
        finally:
            file.close()
        return file.name

    @staticmethod
    def _build_env(dsh_home: str) -> dict[str, str]:
        """Forward the process environment, bridging MindMemOS OpenAI-style keys.

        The Harness resolves the DeepSeek adapter's ``DEEPSEEK_API_KEY`` and
        ``$DEEPSEEK_BASE_URL`` through the environment; when a MindMemOS
        deployment only exports ``OPENAI_API_KEY``/``OPENAI_BASE_URL`` (the
        launcher convention), bridge them so the same gateway serves the
        Harness without duplicating credentials.  ``DSH_HOME`` is always the
        private Agent-owned directory supplied by :meth:`_run_headless`, so
        user settings cannot alter it.
        """

        env = dict(os.environ)
        if not env.get("DEEPSEEK_API_KEY") and env.get("OPENAI_API_KEY"):
            env["DEEPSEEK_API_KEY"] = env["OPENAI_API_KEY"]
        if not env.get("DEEPSEEK_BASE_URL"):
            base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_ENDPOINT")
            if base_url:
                env["DEEPSEEK_BASE_URL"] = base_url
        env["DSH_HOME"] = dsh_home
        return env

    def _resolve_cli(self) -> tuple[str, ...]:
        if self._cli_command is not None:
            return self._cli_command
        if self.config.cli_path:
            path = shutil.which(self.config.cli_path)
            if not path:
                raise RuntimeError(f"DeepSeek Harness CLI executable {self.config.cli_path!r} was not found.")
            self._cli_command = (path,)
            return self._cli_command
        path = shutil.which("dsh")
        if path:
            self._cli_command = (path,)
            return self._cli_command
        cached_dsh = _find_cached_npx_dsh()
        if cached_dsh:
            self._cli_command = (cached_dsh,)
            return self._cli_command
        npx = shutil.which("npx")
        if npx:
            self._cli_command = (npx, _NPX_PACKAGE)
            return self._cli_command
        raise RuntimeError("DeepSeek Harness CLI was not found as either 'dsh' or the 'npx' launcher.")

    def _error_result(
        self,
        request: AgentExecutionRequest,
        started_at: float,
        messages: list[dict[str, Any]],
        error_info: str,
    ) -> Trajectory:
        return self._build_trajectory(
            request=request,
            config=self.config.with_overrides(request.options),
            messages=messages,
            started_at=started_at,
            ended_at=time.time(),
            n_turn=0,
            is_success=False,
            error_info=error_info,
            metadata={"profile": self.config.profile or _DEFAULT_PROFILE},
        )


def _serialize_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render one text-only user turn, with system instructions first.

    Headless DSH accepts one ordinary user message.  Reject histories whose
    role or content structure cannot be represented without semantic loss.
    """

    user_parts: list[str] = []
    system_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _text_content(message.get("content"))
        if not content.strip():
            continue
        if role in {"system", "developer"}:
            system_parts.append(content)
            continue
        if role != "user":
            raise TypeError(
                "DeepSeek Harness headless respond() only supports system/developer instructions plus one user turn; "
                f"received role {role!r}"
            )
        user_parts.append(content)
    if len(user_parts) != 1:
        raise TypeError(
            "DeepSeek Harness headless respond() requires exactly one non-empty user turn; "
            f"received {len(user_parts)}"
        )
    return _compose_headless_task(
        user_text=user_parts[0],
        system_text="\n\n".join(system_parts) or None,
    )


def _find_cached_npx_dsh() -> str | None:
    """Locate an already-installed npx cache binary before invoking the launcher."""

    cache_root = Path(os.environ.get("npm_config_cache", Path.home() / ".npm")).expanduser()
    candidates = [
        path
        for path in cache_root.glob("_npx/*/node_modules/.bin/dsh")
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


@contextmanager
def _stage_skills_in_workspace(skills_root: str | None, *, cwd: str | None) -> Iterator[str | None]:
    """Make the isolated custom Skill root readable through DSH's workspace sandbox."""

    if skills_root is None:
        yield None
        return
    workspace = cwd or os.getcwd()
    with tempfile.TemporaryDirectory(prefix=".mindmemos_dsh_skills_", dir=workspace) as staging_dir:
        staged_root = Path(staging_dir) / "skills"
        shutil.copytree(skills_root, staged_root)
        yield str(staged_root)


async def _communicate(
    command: Sequence[str],
    *,
    cwd: str | None,
    env: Mapping[str, str],
    timeout: float,
) -> tuple[bytes, bytes, int]:
    """Run one DSH process and reap it on timeout, cancellation, or interruption."""

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except BaseException:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
        raise
    return stdout, stderr, process.returncode


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"DeepSeek Harness headless only supports text message content, got {type(content).__name__}")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise TypeError("DeepSeek Harness headless only supports text content blocks")
        parts.append(block["text"])
    return "\n".join(parts)


def _joined_system_instructions(messages: Sequence[Mapping[str, Any]]) -> str | None:
    parts = [
        _text_content(message.get("content")).strip()
        for message in messages
        if message.get("role") in {"system", "developer"}
    ]
    joined = "\n\n".join(part for part in parts if part)
    return joined or None


def _compose_headless_task(*, user_text: str, system_text: str | None) -> str:
    """Reuse DSH's system prompt and place the MindMemOS instructions before the user task."""

    parts: list[str] = []
    if system_text and system_text.strip():
        parts.append(f"System instructions:\n{system_text.strip()}")
    parts.append(f"User:\n{user_text.strip()}")
    return "\n\n".join(parts)


__all__ = ["DeepSeekHarnessAgent"]
