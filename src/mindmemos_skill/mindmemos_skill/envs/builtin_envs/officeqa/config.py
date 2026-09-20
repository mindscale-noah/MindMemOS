"""OfficeQA data and local document research settings."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig


class OfficeQADataConfig(BaseModel):
    """Read questions separately from the public document corpus."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported source options."""

    data_path: Path
    """Official CSV or normalized JSON/JSONL questions."""

    docs_root: Path
    """Root containing the released parsed text documents."""


class OfficeQAConfig(BaseModel):
    """Run bounded local document research with the official binary scorer."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecorded execution options."""

    agent: OpenAIChatConfig
    """Recorded model and transport settings."""

    max_tool_turns: int = Field(default=12, ge=1)
    """Maximum model requests in the research loop."""

    max_tool_chars: int = Field(default=16000, ge=1)
    """Character limit for each public document-tool response."""

    evaluator_version: str = "309dc01ad21113d240700eecb74d2f3e59f163b6"
    """Pinned databricks/officeqa reward protocol revision, with zero tolerance."""
