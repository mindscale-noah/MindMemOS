"""Agent invocation implementations."""

from .base import Agent
from .buitin_agents.claude import ClaudeAgent, ClaudeConfig
from .buitin_agents.codex import CodexAgent, CodexConfig
from .buitin_agents.deepseek_harness import DeepSeekHarnessAgent, DeepSeekHarnessConfig
from .buitin_agents.openai_chat import OpenAIChatAgent, OpenAIChatConfig
from .buitin_agents.openclaw import OpenClawAgent, OpenClawConfig
from .config import AgentCatalog, AgentConfig, AgentDefinition
from .registry import ensure_registered
from .typing import AgentResult

ensure_registered()

__all__ = [
    "Agent",
    "AgentCatalog",
    "AgentConfig",
    "AgentDefinition",
    "AgentResult",
    "CodexAgent",
    "CodexConfig",
    "ClaudeAgent",
    "ClaudeConfig",
    "DeepSeekHarnessAgent",
    "DeepSeekHarnessConfig",
    "OpenAIChatAgent",
    "OpenAIChatConfig",
    "OpenClawAgent",
    "OpenClawConfig",
]
