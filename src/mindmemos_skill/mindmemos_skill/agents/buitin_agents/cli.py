"""Shared process execution and JSON event decoding for native harnesses."""

import json
from pathlib import Path

from ...infra.execution.config import ExecutionConfig
from ...infra.execution.executor import execute
from .config import CLIConfig
from .constants import NATIVE_CONNECTION_ENV


def environment(config: CLIConfig, key_name: str, url_name: str | None = None) -> dict[str, str]:
    """Map resolved connection values to the native CLI environment.

    Args:
        config: Explicit model credentials and endpoint.
        key_name: Native credential variable to populate.
        url_name: Native endpoint variable, when supported.

    Returns:
        Child-only overrides; secrets never appear in command arguments.
    """
    config.require_connection()
    values = {}
    if config.api_key is not None:
        values[key_name] = config.api_key
    if config.base_url is not None and url_name is not None:
        values[url_name] = config.base_url
    return values


def run_cli(
    command: list[str], workspace: Path, config: CLIConfig, overrides: dict[str, str], prompt: str | None = None
) -> str:
    """Execute one native harness and require a successful exit.

    Args:
        command: Literal executable and arguments.
        workspace: Existing public task directory.
        config: Timeout settings.
        overrides: Child-only environment variables.
        prompt: Optional UTF-8 stdin content.

    Returns:
        Captured stdout for backend-specific decoding.

    Raises:
        RuntimeError: The executable exits unsuccessfully.
    """
    result = execute(
        command,
        workspace,
        ExecutionConfig(
            timeout_s=config.timeout_s, environment=overrides, excluded_environment=list(NATIVE_CONNECTION_ENV)
        ),
        input_text=prompt,
    )
    if result.returncode:
        raise RuntimeError(f"Harness exited with status {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def parse_json_lines(text: str, *, cli_name: str) -> list[dict]:
    """Decode a nonempty stream of JSON objects.

    Args:
        text: Captured JSONL output.
        cli_name: Executable label for diagnostic messages.

    Returns:
        Ordered native event objects.
    """
    events = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            raise RuntimeError(f"{cli_name} returned malformed JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"{cli_name} returned a non-object event")
        events.append(event)
    if not events:
        raise RuntimeError(f"{cli_name} returned no events")
    return events


def token_usage(raw: object) -> dict:
    """Preserve native usage and add a token total when both counts exist.

    Args:
        raw: Native terminal usage payload, possibly absent.

    Returns:
        Available usage only; absent measurements remain unknown.
    """
    if not isinstance(raw, dict):
        return {}
    usage = dict(raw)
    counts = [usage.get("input_tokens"), usage.get("output_tokens")]
    if all(type(value) is int and value >= 0 for value in counts):
        usage["total_tokens"] = sum(counts)
    return usage
