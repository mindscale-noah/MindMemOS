"""Load local experiment credentials without writing them to output artifacts."""

import os
from pathlib import Path

from dotenv import dotenv_values

from .constants import (
    AGENT_API_KEY_ENV,
    AGENT_BASE_URL_ENV,
    CHILD_API_KEY_ENV,
    CHILD_BASE_URL_ENV,
    PARENT_API_KEY_ENV,
    PARENT_BASE_URL_ENV,
)


def load_credentials(path: Path, *, require_models: bool = False) -> None:
    """Load a dotenv file while keeping existing process variables authoritative."""
    if not path.is_file():
        raise FileNotFoundError(f"Credential file not found: {path}")
    for name, value in dotenv_values(path).items():
        if value and not os.environ.get(name):
            os.environ[name] = value
    for destination, fallback in (
        (PARENT_API_KEY_ENV, AGENT_API_KEY_ENV),
        (CHILD_API_KEY_ENV, AGENT_API_KEY_ENV),
        (PARENT_BASE_URL_ENV, AGENT_BASE_URL_ENV),
        (CHILD_BASE_URL_ENV, AGENT_BASE_URL_ENV),
    ):
        value = os.environ.get(destination) or os.environ.get(fallback)
        if value:
            os.environ[destination] = value
        elif require_models:
            raise ValueError(f"Missing credential setting: {destination}")
