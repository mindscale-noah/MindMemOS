"""Resolve the SDK home when configuration is created."""

import os
from pathlib import Path

from .constants import CONFIG_DIR_ENV, MINDMEMOS_HOME_ENV


def get_mindmemos_home() -> Path:
    """Return the selected SDK home, expanding user paths.

    Returns:
        MINDMEMOS_HOME, the legacy config directory, or ~/.mindmemos.
    """
    home = os.environ.get(MINDMEMOS_HOME_ENV) or os.environ.get(CONFIG_DIR_ENV)
    return Path(home).expanduser() if home else Path.home() / ".mindmemos"
