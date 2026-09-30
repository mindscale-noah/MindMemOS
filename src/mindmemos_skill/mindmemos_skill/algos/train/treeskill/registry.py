"""Explicit catalog of built-in TreeSkill environment adapters."""

from importlib import import_module

_BUILTIN_MODULES = (".env_specific.spreadsheetbench", ".env_specific.legacy")
"""Modules that register adapters without creating runtime resources."""


def ensure_registered() -> None:
    """Import the explicit catalog through Python's module cache."""
    for module in _BUILTIN_MODULES:
        import_module(module, package=__package__)
