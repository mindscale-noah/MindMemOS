"""Environment variable names and fixed file names of the evolver."""

from typing import Final

HOST_DIR: Final[str] = ".offline-evolution"
"""Run subdirectory owned by the Host (state, logs, Claude configuration); agents may not read or write it.
Its name must stay distinctive: Bash deny rules match it as a substring of commands."""

COLLECTION_FILE: Final[str] = "collection.json"
"""Manifest naming the evidence collector."""

CHANGES_FILE: Final[str] = "changes.json"
"""Change claims written by the change agent."""
