"""Load and validate the preinstalled SpreadsheetBench execution environment."""

from pathlib import Path

from ....infra.execution import ExecutionConfig, execute
from ....infra.execution.python_env import prepare_python_environment


async def prepare_execution(protocol: str) -> ExecutionConfig:
    """Prepare the protocol's fixed interpreter beneath MindMemOS home.

    Args:
        protocol: SpreadsheetBench protocol, text_action or tool_calling.

    Returns:
        Internal execution settings bound to the managed Python environment.
    """
    if protocol not in {"text_action", "tool_calling"}:
        raise ValueError(f"Unknown SpreadsheetBench protocol: {protocol}")
    mode = {"text_action": "text_action_mode", "tool_calling": "tool_call_mode"}[protocol]
    requirements = Path(__file__).parent / mode / "requirements" / "requirements.txt"
    python = await prepare_python_environment(f"spreadsheetbench_{protocol}", requirements)
    return ExecutionConfig(python_command=[str(python)])


async def check_execution(config: ExecutionConfig, directory: Path) -> None:
    """Check that required packages import in the selected interpreter.

    Args:
        config: Preinstalled Python interpreter or uv launcher settings.
        directory: Existing directory used for the dependency probe.

    Raises:
        RuntimeError: Required packages are unavailable in the selected interpreter.
        OSError: The configured launcher cannot be started.
    """
    result = await execute(["python", "-c", "import openpyxl, pandas, numpy"], directory, config)
    if result.returncode:
        raise RuntimeError(
            "SpreadsheetBench Python dependency check failed before model execution. "
            f"Inspect the managed environment at {config.python_command[0]}.\n{result.stderr}"
        )
