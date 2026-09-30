"""Read-only document tools restricted to a per-task public allowlist."""

from fnmatch import fnmatch
from pathlib import Path

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find public document files by pattern.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read public text by line window.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "limit": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a public text file for a literal pattern.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}},
                "required": ["path", "pattern"],
            },
        },
    },
]
"""Only these read-only tools are exposed to the answering model."""


def run_tool(name: str, arguments: dict, documents: dict[str, Path], limit: int) -> str:
    """Dispatch a read-only tool without granting directory access.

    Args:
        name: Declared tool name.
        arguments: Model-provided tool arguments.
        documents: Allowed public names and resolved file paths.
        limit: Maximum output characters.

    Returns:
        Public excerpt or a tool argument error.
    """
    if name == "glob":
        pattern = str(arguments.get("pattern", "*"))
        return "\n".join(key for key in documents if fnmatch(key, pattern))[:limit]
    if name not in {"read", "grep"}:
        return f"Error: unknown tool {name!r}"
    path = str(arguments.get("path", ""))
    if path not in documents:
        return f"Error: document is not in the public allowlist: {path}"
    lines = documents[path].read_text(encoding="utf-8", errors="replace").splitlines()
    if name == "grep":
        pattern = str(arguments.get("pattern", ""))
        selected = [(number, line) for number, line in enumerate(lines, 1) if pattern.lower() in line.lower()]
    else:
        try:
            start = max(1, int(arguments.get("start", 1)))
            count = max(1, min(500, int(arguments.get("limit", 100))))
        except (ValueError, TypeError):
            return "Error: start and limit must be integers"
        selected = list(enumerate(lines[start - 1 : start - 1 + count], start))
    return "\n".join(f"{number}: {line}" for number, line in selected)[:limit]
