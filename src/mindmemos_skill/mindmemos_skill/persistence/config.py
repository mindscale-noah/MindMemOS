"""Configuration for registering business tables on a shared database."""

from pydantic import BaseModel, ConfigDict


class SkillPersistenceConfig(BaseModel):
    """Select whether this database participates in remote synchronization."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled configuration fields."""

    include_remote: bool = True
    """Register remote bindings and operations alongside core and graph tables."""
