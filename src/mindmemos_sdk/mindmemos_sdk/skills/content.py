"""Portable Skill content hashing and resource validation for the standalone SDK."""

import base64
import hashlib
import json
from pathlib import PurePosixPath

import yaml

_HASH_SCHEMA = "mindmemos-skill-v1"


def validate_resource_paths(resources: dict[str, bytes]) -> None:
    """Reject unsafe, noncanonical, or overlapping relative resource paths.

    Args:
        resources: Resource names and their original bytes.

    Raises:
        ValueError: A path is unsafe or two paths overlap as files and directories.
    """
    for name in resources:
        path = PurePosixPath(name)
        if (
            not name
            or not path.parts
            or path.is_absolute()
            or "\\" in name
            or "\x00" in name
            or path.as_posix() != name
            or name.endswith("/")
            or ".." in path.parts
            or path.parts[0] == "SKILL.md"
        ):
            raise ValueError(f"Invalid Skill resource path: {name!r}")
        if any(parent.as_posix() in resources for parent in path.parents):
            raise ValueError(f"Resource path overlaps a file: {name!r}")


def encode_resources(resources: dict[str, bytes]) -> dict[str, str]:
    """Encode original resource bytes as canonical Base64 for JSON transport.

    Args:
        resources: Validated auxiliary files by relative path.

    Returns:
        Path-to-Base64 mapping in path order.
    """
    validate_resource_paths(resources)
    return {name: base64.b64encode(resources[name]).decode("ascii") for name in sorted(resources)}


def decode_resources(resources: dict[str, str]) -> dict[str, bytes]:
    """Decode and validate a JSON transport resource mapping.

    Args:
        resources: Relative paths to standard Base64 strings.

    Returns:
        Original bytes by relative path.

    Raises:
        ValueError: A value is not canonical Base64 or a path is invalid.
    """
    decoded = {}
    for name, value in resources.items():
        raw = base64.b64decode(value, validate=True)
        if base64.b64encode(raw).decode("ascii") != value:
            raise ValueError(f"Resource is not canonical Base64: {name!r}")
        decoded[name] = raw
    validate_resource_paths(decoded)
    return decoded


def content_hash(content: str) -> str:
    """Hash the exact UTF-8 bytes of the stored Skill document.

    Args:
        content: Full SKILL.md text without newline normalization.

    Returns:
        Lowercase SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def skill_hash(content: str, resources: dict[str, bytes]) -> str:
    """Hash the Skill document and all auxiliary resource names and bytes.

    Args:
        content: Full SKILL.md text without newline normalization.
        resources: Auxiliary file paths and original bytes.

    Returns:
        Lowercase SHA-256 digest of the versioned canonical JSON encoding.
    """
    encoded = encode_resources(resources)
    payload = {
        "schema": _HASH_SCHEMA,
        "content": content,
        "resources": [{"path": name, "base64": encoded[name]} for name in encoded],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_frontmatter(content: str) -> dict:
    """Parse required standard Skill fields without changing source text.

    Args:
        content: Complete UTF-8 Skill document.

    Returns:
        YAML frontmatter containing nonempty name and description.
    """
    lines = content.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("Standard Skill content requires YAML frontmatter")
    end = next((i for i in range(1, len(lines)) if lines[i] == "---"), None)
    if end is None:
        raise ValueError("Skill frontmatter is not closed")
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid Skill YAML frontmatter") from exc
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValueError("Skill frontmatter must be an object")
    for field in ("name", "description"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Skill frontmatter requires a nonempty {field}")
    data["name"] = data["name"].strip()
    data["description"] = data["description"].strip()
    return data
