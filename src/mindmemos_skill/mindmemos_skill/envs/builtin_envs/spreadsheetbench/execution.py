"""Load and validate the preinstalled SpreadsheetBench execution environment."""

from pathlib import Path

from ....infra.execution import ExecutionConfig, execute
from ....infra.execution.python_env import prepare_python_environment


def prepare_execution(protocol: str) -> ExecutionConfig:
    """Prepare the protocol's fixed interpreter beneath MindMemOS home.

    Args:
        protocol: SpreadsheetBench protocol, text_action or tool_calling.

    Returns:
        Internal execution settings bound to the managed Python environment.
    """
    if protocol not in {"text_action", "tool_calling"}:
        raise ValueError(f"Unknown SpreadsheetBench protocol: {protocol}")
    requirements = Path(__file__).parent / "runtime" / "requirements" / protocol / "requirements.txt"
    python = prepare_python_environment(f"spreadsheetbench_{protocol}", requirements)
    return ExecutionConfig(python_command=[str(python)])


def check_execution(config: ExecutionConfig, directory: Path) -> None:
    """Check the selected interpreter, dependencies, and workbook round trip.

    Args:
        config: Preinstalled Python interpreter or uv launcher settings.
        directory: Existing directory used for the dependency probe.

    Raises:
        RuntimeError: Required packages or workbook read/write support are unavailable.
        OSError: The configured launcher cannot be started.
    """
    probe = """
import io
import openpyxl, pandas, numpy
stream = io.BytesIO()
workbook = openpyxl.Workbook()
workbook.active['A1'] = 42
workbook.save(stream)
stream.seek(0)
assert openpyxl.load_workbook(stream).active['A1'].value == 42
"""
    result = execute(["python", "-c", probe], directory, config)
    if result.returncode:
        raise RuntimeError(
            "SpreadsheetBench Python dependency check failed before model execution. "
            f"Inspect the managed environment at {config.python_command[0]}.\n{result.stderr}"
        )
