"""Load the explicit business persistence implementation catalog."""

from importlib import import_module

_BUILTIN_MODULES = (".store",)
"""Modules whose decorators register persistence implementations."""


def ensure_registered() -> None:
    """Load implementations without connecting to a database."""
    for module in _BUILTIN_MODULES:
        import_module(module, package=__package__)
