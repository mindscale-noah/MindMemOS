"""Typed configuration for the built-in ``skill_manage`` algorithm."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillManageConfig(BaseModel):
    """Strict, frozen settings for one ``skill_manage`` algorithm instance.

    The five timing fields only steer synchronous join/wait, the close barrier
    and the writer lease. They never turn a call into a background task. The
    two retention fields bound the deferred GC that runs after a publish.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm_version: str = Field(default="1", min_length=1)
    llm_linking: bool = False
    build_scene: bool = True
    query_rewrite: bool = False
    max_input_chars: int = Field(default=1_000_000, gt=0)

    operation_wait_timeout_seconds: float = Field(default=30.0, gt=0)
    operation_poll_interval_seconds: float = Field(default=0.1, gt=0)
    shutdown_grace_period_seconds: float = Field(default=30.0, gt=0)
    writer_lease_ttl_seconds: float = Field(default=60.0, gt=0)
    writer_lease_renew_interval_seconds: float = Field(default=20.0, gt=0)
    retired_generations_to_keep: int = Field(default=2, ge=0)
    retired_generation_min_age_seconds: float = Field(default=86400.0, ge=0)

    @model_validator(mode="after")
    def _validate_timing_relationships(self) -> SkillManageConfig:
        if self.operation_poll_interval_seconds >= self.operation_wait_timeout_seconds:
            raise ValueError("operation_poll_interval_seconds must be less than operation_wait_timeout_seconds")
        if self.writer_lease_renew_interval_seconds >= self.writer_lease_ttl_seconds / 2:
            raise ValueError("writer_lease_renew_interval_seconds must be less than writer_lease_ttl_seconds / 2")
        return self


__all__ = ["SkillManageConfig"]
