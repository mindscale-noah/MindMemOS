"""Explicit built-in agent catalog."""

from importlib import import_module

_BUILTIN_MODULES = (
    ".buitin_agents.openai_chat",
    ".buitin_agents.codex",
    ".buitin_agents.claude",
    ".buitin_agents.openclaw",
    ".buitin_agents.deepseek_harness",
)
"""Modules whose imports register built-in agents."""


def ensure_registered() -> None:
    """Import built-in agents without creating model clients."""
    for module in _BUILTIN_MODULES:
        import_module(module, package=__package__)
