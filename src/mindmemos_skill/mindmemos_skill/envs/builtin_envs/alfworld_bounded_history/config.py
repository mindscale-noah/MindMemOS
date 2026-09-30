"""Single-game ALFWorld settings with bounded prompt history."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig


class ALFWorldDataConfig(BaseModel):
    """Load fixed game IDs without importing the simulator."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown source settings."""

    data_path: Path
    """JSON or JSONL items with id and gamefile fields."""

    game_root: Path | None = None
    """Base directory for relative game paths; defaults to the items directory."""


class ALFWorldConfig(BaseModel):
    """Configure one independent simulator and its answering model."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecorded runtime settings."""

    agent: OpenAIChatConfig
    """Recorded model and transport settings."""

    simulator: dict[str, JsonValue]
    """Complete upstream ALFWorld YAML contents, recorded in resume identity."""

    seed: int = 42
    """Simulator random seed."""

    max_steps: int = Field(default=50, ge=1)
    """Maximum actions in this episode."""

    history_length: int = Field(default=5, ge=0)
    """Number of preceding observation/action pairs included in each prompt."""

    evaluator_version: str = "alfworld-bounded-history-v1"
    """Identity of task success and failure-text semantics."""
