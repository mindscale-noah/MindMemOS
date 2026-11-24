"""Environment variable names used by the local command runner."""

PATH_ENV = "PATH"
"""Executable search path, extended with per-command Python launchers."""

PYTHON_UNBUFFERED_ENV = "PYTHONUNBUFFERED"
"""Disable buffering so command output is available promptly."""

PYTHON_IO_ENCODING_ENV = "PYTHONIOENCODING"
"""Use UTF-8 when decoding Python subprocess output on every platform."""

PIP_NO_INDEX_ENV = "PIP_NO_INDEX"
"""Disable package index access in workspace tools."""

UV_OFFLINE_ENV = "UV_OFFLINE"
"""Disable uv network package resolution in workspace tools."""

INPUT_FILE_ENV = "INPUT_FILE"
"""Public input workbook available to bash commands."""

OUTPUT_FILE_ENV = "OUTPUT_FILE"
"""Required output workbook available to bash commands."""
