"""Explicit catalog of built-in Skill-Forest environment adapters."""

from importlib import import_module

BUILTIN_ADAPTER_NAMES: tuple[str, ...] = (
    "livemath",
    "searchqa",
    "officeqa",
    "spreadsheetbench_tool_calling",
    "spreadsheetbench_text_action",
    "alfworld_bounded_history",
)
"""Names shared by built-in adapter registration and automatic environment matching."""


def ensure_registered() -> None:
    """Load the explicit catalog once through Python's module cache."""
    for name in BUILTIN_ADAPTER_NAMES:
        import_module(f".env_specific.{name}", package=__package__)
