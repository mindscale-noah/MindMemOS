"""Separate metadata from model-visible synthesis bodies."""

import yaml


def split_merge_content(content: str) -> tuple[str, str]:
    """Separate an optional YAML mapping header from a Markdown body.

    Args:
        content: Candidate content or a model-generated synthesis response.

    Returns:
        Original header and body, or an empty header and unchanged Markdown.

    Raises:
        ValueError: A YAML metadata header is malformed or unclosed.
    """
    text = content.lstrip()
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return "", content
    closing = next((index for index, line in enumerate(lines[1:], 1) if line.rstrip("\r\n") == "---"), None)
    payload = "".join(lines[1:closing])
    try:
        metadata = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise ValueError("Skill synthesis frontmatter contains invalid YAML") from error
    if not isinstance(metadata, dict):
        return "", content
    if closing is None:
        raise ValueError("Skill synthesis frontmatter has no closing delimiter")
    return "".join(lines[: closing + 1]), "".join(lines[closing + 1 :])
