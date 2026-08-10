"""Agent abstraction layer for MindMemOS."""

from ..registry import get_agent, list_agents
from ..typing import AgentExecutionRequest
from .base import Agent
from .claude import ClaudeAgentConfig, ClaudeSDKAgentConfig
from .config import AgentConfig
from .openclaw import OpenClawAgentConfig
from .react import ReactAgentConfig, Tool, tool
from .skill_runtime import SkillInjection, SkillRuntime

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentExecutionRequest",
    "ClaudeAgentConfig",
    "ClaudeSDKAgentConfig",
    "OpenClawAgentConfig",
    "ReactAgentConfig",
    "SkillInjection",
    "SkillRuntime",
    "Tool",
    "get_agent",
    "list_agents",
    "tool",
]
