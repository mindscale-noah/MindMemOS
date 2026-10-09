"""Resolve and persist the default SDK home."""

import json
import os
import tempfile
from pathlib import Path

from .constants import CONFIG_DIR_ENV, MINDMEMOS_HOME_ENV


def _home_preference_path() -> Path:
    """Return the current user's persistent home preference file."""
    return Path.home() / ".config" / "mindmemos" / "home.json"


def set_default_mindmemos_home(directory: str | Path) -> Path:
    """Save an absolute default SDK home for future processes.

    Args:
        directory: Absolute configuration directory, with optional user expansion.

    Returns:
        The expanded directory saved as the user's default.

    Raises:
        ValueError: If the directory is empty or relative.
    """
    value = str(directory).strip()
    if not value:
        raise ValueError("Configuration directory must not be empty")
    home = Path(value).expanduser()
    if not home.is_absolute():
        raise ValueError("Configuration directory must be an absolute path")

    preference = _home_preference_path()
    preference.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".home-", dir=preference.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"home": str(home)}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, preference)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return home


def get_mindmemos_home() -> Path:
    """Return the selected SDK home, expanding user paths.

    Returns:
        MINDMEMOS_HOME, the legacy config directory, the user's saved default,
        or ~/.mindmemos.
    """
    home = os.environ.get(MINDMEMOS_HOME_ENV) or os.environ.get(CONFIG_DIR_ENV)
    if home:
        return Path(home).expanduser()
    preference = _home_preference_path()
    if preference.is_file():
        saved = json.loads(preference.read_text(encoding="utf-8"))["home"]
        return Path(saved)
    return Path.home() / ".mindmemos"
