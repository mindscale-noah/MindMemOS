"""Explicit built-in algorithm catalog."""

from importlib import import_module

_BUILTIN_MODULES = (
    ".train.treeskill.trainer",
    ".train.skill_forest.trainer",
    ".manage.skill_manage.algorithm",
    ".evolve.simple",
)
"""Modules whose decorators register supported algorithm implementations."""


def ensure_registered() -> None:
    """Load built-in implementations without constructing resources."""
    for name in _BUILTIN_MODULES:
        import_module(name, package=__package__)
