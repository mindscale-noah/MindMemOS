"""Schema add drain configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrainConfig:
    """Configuration for draining schema add buffered records."""

    episode_generation_max_retries: int = field(default=3)
    """Max retry attempts for a single episode generation task."""

    episode_retry_backoff_base: float = field(default=1.0)
    """Base seconds for exponential backoff between episode retries (doubles each attempt)."""

    episode_retry_backoff_max: float = field(default=30.0)
    """Maximum seconds between episode retries (backoff is capped at this value)."""

    cleanup_processed_buffer: bool = field(default=True)
    """Whether to delete buffer records after successful episode generation."""
