"""Load explicitly registered infrastructure implementations."""

from importlib import import_module

_BUILTIN_MODULES = (".database.builtin_backends",)
"""Built-in infrastructure implementations available to service configuration."""


def ensure_registered() -> None:
    """Import backend definitions without opening external resources."""
    for module in _BUILTIN_MODULES:
        import_module(module, package=__package__)
