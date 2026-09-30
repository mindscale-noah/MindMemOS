"""Explicit catalog of built-in Skill-Forest environment adapters."""

from importlib import import_module

_BUILTIN_MODULES = (
    ".env_specific.livemath",
    ".env_specific.searchqa",
    ".env_specific.officeqa",
    ".env_specific.spreadsheetbench",
    ".env_specific.alfworld_bounded_history",
)
"""Imports execute registrations without constructing runtime resources."""


def ensure_registered() -> None:
    """Load the explicit catalog once through Python's module cache."""
    for module in _BUILTIN_MODULES:
        import_module(module, package=__package__)
