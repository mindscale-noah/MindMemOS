"""Retain native provider errors across streamed and captured CLI execution."""

import json
import sys

import pytest
from mindmemos_skill.agents.buitin_agents.config import CLIConfig
from mindmemos_skill.algos.evolve.offline_evolution.agents.execution import run_cli


@pytest.mark.parametrize("logged", [False, True])
@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"type": "result", "is_error": True, "result": "API Error: 400 tool_calls above 128"}, "tool_calls above 128"),
        ({"type": "result", "is_error": True, "errors": ["provider unavailable"]}, "provider unavailable"),
        ({"type": "turn.failed", "error": {"message": "invalid request"}}, "invalid request"),
        ({"type": "error", "message": "rate limit exceeded"}, "rate limit exceeded"),
        (None, "plain native failure"),
    ],
)
async def test_nonzero_exit_preserves_stdout_error_and_stderr(tmp_path, logged, event, expected):
    """Both backends report provider failures on stdout, including when stderr only has a warning."""
    stdout = json.dumps(event) + "\n" if event is not None else "plain native failure\n"
    stderr = "native warning\n"
    script = f"import sys\nsys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\nsys.exit(1)\n"
    log = tmp_path / "events.jsonl" if logged else None
    with pytest.raises(RuntimeError) as error:
        await run_cli([sys.executable, "-c", script], tmp_path, CLIConfig(timeout_s=5), {}, log_path=log)
    assert "status 1" in str(error.value)
    assert expected in str(error.value) and "native warning" in str(error.value)
    if log is not None:
        assert log.read_text() == stdout


async def test_stderr_is_preserved_when_stdout_has_no_error_event(tmp_path):
    script = "import sys\nsys.stderr.write('executable failed')\nsys.exit(2)\n"
    with pytest.raises(RuntimeError, match="status 2: executable failed"):
        await run_cli([sys.executable, "-c", script], tmp_path, CLIConfig(timeout_s=5), {})
