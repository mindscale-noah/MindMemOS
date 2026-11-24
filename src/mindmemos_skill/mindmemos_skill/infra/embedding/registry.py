"""Explicit built-in embedding backend catalog."""

from importlib import import_module

_BUILTIN_MODULES = (".client",)
"""Modules whose imports register embedding implementations."""


def ensure_registered() -> None:
    """Load built-in registrations without constructing model clients."""
    for module in _BUILTIN_MODULES:
        import_module(module, package=__package__)
