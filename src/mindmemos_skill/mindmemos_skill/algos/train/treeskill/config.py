"""Fixed model settings for TreeSkill's task-time subtree router."""

from pydantic import BaseModel, ConfigDict

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig


class TreeSkillConfig(BaseModel):
    """Configure routing without storing optimizable instructions outside bundle."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown router settings."""

    router: OpenAIChatConfig
    """Router model settings, including generation limits and provider retries."""
