"""Connection settings for the built-in databases."""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.engine import make_url


class SQLiteConfig(BaseModel):
    """Configure a local SQLite database."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown settings."""

    url: str = "sqlite+pysqlite:///./data/mindmemos.db"
    """SQLite file URL; memory databases are intended for single-threaded tests."""

    timeout: float = Field(default=5, gt=0)
    """Seconds to wait for a database write lock."""

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require the supported synchronous SQLite driver."""
        if make_url(value).drivername not in {"sqlite", "sqlite+pysqlite"}:
            raise ValueError("Expected a sqlite+pysqlite URL")
        return value


class PostgresConfig(BaseModel):
    """Configure a PostgreSQL database with pgvector installed."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown settings."""

    url: str = Field(repr=False)
    """PostgreSQL connection URL, including credentials when required."""

    pool_size: int = Field(default=5, gt=0)
    """Persistent connections per service process."""

    max_overflow: int = Field(default=10, ge=0)
    """Additional temporary connections per process."""

    pool_timeout: float = Field(default=30, gt=0)
    """Seconds to wait for a pooled connection."""

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require the supported synchronous PostgreSQL driver."""
        if make_url(value).drivername != "postgresql+psycopg":
            raise ValueError("Expected a postgresql+psycopg URL")
        return value
