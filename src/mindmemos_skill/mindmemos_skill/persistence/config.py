"""Configuration for registering business tables on a shared database."""

from pydantic import BaseModel, ConfigDict


class SkillPersistenceConfig(BaseModel):
    """Configure the Skill repository database."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject misspelled configuration fields."""

    prefixed_ids: bool = True
    """Create prefixed Skill IDs."""
