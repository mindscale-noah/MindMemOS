"""Explicit catalog of built-in TreeSkill environment adapters."""

from importlib import import_module

BUILTIN_ADAPTER_MODULES: dict[str, str] = {
    "spreadsheetbench_text_action": ".env_specific.spreadsheetbench_text_action",
    "legacy": ".env_specific.legacy",
}
"""Modules that register adapters without creating runtime resources."""


def ensure_registered() -> None:
    """Import the explicit catalog through Python's module cache."""
    for module in BUILTIN_ADAPTER_MODULES.values():
        import_module(module, package=__package__)
