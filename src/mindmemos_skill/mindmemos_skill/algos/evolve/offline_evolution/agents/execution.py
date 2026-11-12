"""Keep offline environment filtering and streamed logs around the shared process executor."""

import json
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from .....agents.buitin_agents.config import CLIConfig
from .....agents.buitin_agents.constants import NATIVE_CONNECTION_ENV
from .....infra.execution.config import ExecutionConfig
from .....infra.execution.executor import execute


async def run_cli(
    command: list[str],
    workspace: Path,
    config: CLIConfig,
    overrides: dict[str, str],
    prompt: str | None = None,
    *,
    excluded_environment: Sequence[str] = (),
    log_path: Path | None = None,
) -> str:
    """Execute an offline invocation with its own environment and persistent stdout.

    Args:
        command: Shared Agent command after offline-specific adjustments.
        workspace: Algorithm execution directory.
        config: Validated native timeout settings.
        overrides: Explicit connection and per-run child environment.
        prompt: UTF-8 task or feedback supplied on stdin.
        excluded_environment: Additional ambient variables excluded only from this child.
        log_path: Raw events retained during execution and on failure.

    Returns:
        Native JSONL stdout for the existing shared result parser.

    Raises:
        RuntimeError: The native executable exits unsuccessfully.
    """
    with ExitStack() as resources:
        output = None
        if log_path is not None:
            log_path = log_path.expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            output = resources.enter_context(log_path.open("wb"))
        result = await execute(
            command,
            workspace,
            ExecutionConfig(
                timeout_s=config.timeout_s,
                environment=overrides,
                excluded_environment=[*NATIVE_CONNECTION_ENV, *excluded_environment],
            ),
            input_text=prompt,
            output=output,
        )
    stdout = log_path.read_bytes().decode("utf-8", errors="replace") if log_path is not None else result.stdout
    if result.returncode:
        detail = "\n".join(part for part in (_error_detail(stdout), result.stderr.strip()[-2000:]) if part)
        raise RuntimeError(f"Harness exited with status {result.returncode}: {detail}")
    return stdout


def _error_detail(stdout: str) -> str:
    """Extract the last native failure, falling back to a bounded tail for non-JSON output."""
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" and event.get("is_error"):
            detail = event.get("result") or event.get("errors")
        elif event.get("type") in {"error", "turn.failed"}:
            detail = event.get("error") or event.get("message")
        else:
            continue
        if isinstance(detail, dict):
            detail = detail.get("message") or detail
        if detail:
            return (detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False))[:2000]
    return stdout.strip()[-2000:]
