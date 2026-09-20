"""Explicit loading of built-in environment modules."""

from importlib import import_module

BUILTIN_MODULES = (
    "mindmemos_skill.envs.builtin_envs.livemath",
    "mindmemos_skill.envs.builtin_envs.searchqa",
    "mindmemos_skill.envs.builtin_envs.officeqa",
    "mindmemos_skill.envs.builtin_envs.spreadsheetbench",
    "mindmemos_skill.envs.builtin_envs.alfworld_bounded_history",
)
"""Modules that register concrete built-in environments and their data loaders."""


def ensure_registered() -> None:
    """Import built-ins without constructing models or accessing external resources."""
    for module in BUILTIN_MODULES:
        import_module(module)
