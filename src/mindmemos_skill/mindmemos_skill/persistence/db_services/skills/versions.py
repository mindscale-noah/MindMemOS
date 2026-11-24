"""Semantic display labels, independent of immutable version identities."""

import re
from collections.abc import Iterable
from typing import Literal

USER_VERSION_PATTERN = r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
"""Three numeric components with no leading zeros."""

ALGORITHM_PATTERN = r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"
"""Stable lowercase algorithm identifier used as a system-generated suffix."""

VERSION_PATTERN = USER_VERSION_PATTERN[:-1] + r"(?:-([a-z][a-z0-9]*(?:-[a-z0-9]+)*))?$"
"""User labels and generated algorithm labels accepted by storage."""


def parse_label(label: str | None) -> tuple[int, int, int, str | None] | None:
    """Parse a canonical label, returning None for invalid labels.

    Args:
        label: Optional stored display label.
    """
    match = re.fullmatch(VERSION_PATTERN, label or "")
    if match is None:
        return None
    try:
        return int(match[1]), int(match[2]), int(match[3]), match[4]
    except ValueError:
        return None


def next_label(
    current: str | None,
    used: Iterable[str | None],
    *,
    bump: Literal["patch", "minor", "major"] = "patch",
    algorithm: str | None = None,
) -> str:
    """Allocate above the used numbers in the selected line and namespace.

    Args:
        current: Input version; None indicates an initial registration.
        used: Existing labels of this project and Skill, including existing branches.
        bump: Component to advance; lower components reset to zero.
        algorithm: Reserved suffix; None selects the human namespace.
    """
    if algorithm is not None and re.fullmatch(ALGORITHM_PATTERN, algorithm) is None:
        raise ValueError("Invalid algorithm identifier")
    if bump not in {"patch", "minor", "major"}:
        raise ValueError("Unknown version increment")
    suffix = f"-{algorithm}" if algorithm else ""
    if current is None:
        return f"v1.0.0{suffix}"
    parsed = parse_label(current)
    if parsed is None:
        raise ValueError("Invalid active version label")
    major, minor, patch, _ = parsed
    prior = [value for label in used if (value := parse_label(label)) is not None and value[3] == algorithm]
    if bump == "major":
        major, minor, patch = max([major, *(v[0] for v in prior)]) + 1, 0, 0
    elif bump == "minor":
        minor, patch = max([minor, *(v[1] for v in prior if v[0] == major)]) + 1, 0
    else:
        patch = max([patch, *(v[2] for v in prior if v[:2] == (major, minor))]) + 1
    return f"v{major}.{minor}.{patch}{suffix}"
