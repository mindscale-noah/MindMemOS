"""Read-only document tools restricted to a per-task public allowlist."""

import json
from fnmatch import fnmatch
from pathlib import Path

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find candidate local document files by filename or relative-path glob pattern.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a local text document excerpt by path and line window.",
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
            "description": "Search a local text document for a literal pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern", "path"],
            },
        },
    },
]
"""Only these read-only tools are exposed to the answering model."""


def format_tool_command(name: str, arguments: str, observation: str = "") -> str:
    """Render the original OfficeQA dispatcher command without rerunning tools.

    Args:
        name: Recorded tool name.
        arguments: Recorded JSON argument object.
        observation: Recorded response used to identify rejected reads.

    Returns:
        Command string matching the original local document protocol.
    """
    try:
        values = json.loads(arguments or "{}")
        if not isinstance(values, dict):
            return f"{name}({arguments})"
        path = str(values.get("path") or "")
        if name == "glob":
            return f"glob(pattern={str(values.get('pattern') or '*')!r})"
        if name == "grep":
            pattern = str(values.get("pattern") or "").lower()
            return f"grep(pattern={pattern!r}, path={path!r})"
        if name == "read":
            if not path or observation.startswith("[read error:"):
                return f"read(path={path!r})"
            start = max(int(values.get("start") or 1), 1)
            count = max(int(values.get("limit") or 80), 1)
            return f"read(path={path!r}, start={start}, limit={count})"
    except (ValueError, TypeError):
        return f"{name}({arguments})"
    return name


async def run_tool(name: str, arguments: dict, documents: dict[str, Path], limit: int) -> str:
    """Dispatch original-format document tools within the public allowlist.

    Args:
        name: Declared tool name.
        arguments: Model-provided tool arguments.
        documents: Allowed public names and resolved file paths.
        limit: Additional response character cap configured by the caller.

    Returns:
        Original-format public excerpt or tool error.
    """
    if name == "glob":
        pattern = str(arguments.get("pattern") or "*")
        matches = [key for key in documents if fnmatch(key, pattern) or fnmatch(Path(key).name, pattern)][:50]
        return ("\n".join(matches) if matches else "[no matches]")[:limit]
    if name not in {"read", "grep"}:
        return f"[tool error: unknown tool {name}]"
    path = str(arguments.get("path") or "")
    pattern = str(arguments.get("pattern") or "").lower()
    if name == "read" and not path:
        return "[read error: missing path]"
    if name == "grep" and (not path or not pattern):
        return "[grep error: missing pattern or path]"
    if path not in documents:
        return f"[{name} error: path not allowed]"
    with documents[path].open(encoding="utf-8") as stream:
        lines = stream.readlines()
    if name == "grep":
        matches = [f"{number}: {line.rstrip()}" for number, line in enumerate(lines, 1) if pattern in line.lower()][:20]
        return ("\n".join(matches) if matches else "[no matches]")[:limit]
    start = max(int(arguments.get("start") or 1), 1)
    count = max(int(arguments.get("limit") or 80), 1)
    excerpt = "".join(lines[start - 1 : start - 1 + count])
    return excerpt[: min(limit, 4000)] or "[empty file]"
