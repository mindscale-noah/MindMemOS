"""Test-scoped replacement of the shared Agent construction entry point."""

from mindmemos_skill.agents import Agent


def agent_config(monkeypatch, constructor, config):
    """Patch Agent creation for this test and return its unchanged configuration."""
    monkeypatch.setattr(Agent, "from_config", constructor)
    return config
