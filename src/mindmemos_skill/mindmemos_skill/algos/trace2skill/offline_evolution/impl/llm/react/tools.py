"""Sandboxed local tools used by API-backed agent loops."""

from __future__ import annotations

import glob
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import LLMToolDefinition

READ_TOOL_NAME = "Read"
GLOB_TOOL_NAME = "Glob"
GREP_TOOL_NAME = "Grep"
EDIT_TOOL_NAME = "Edit"
WRITE_TOOL_NAME = "Write"

# The alias an out-of-cwd result store answers to: the model addresses
# ``tool-results/<name>`` and ``_resolve_file`` maps it onto the real store.
_RESULT_STORE_ALIAS = "tool-results"


def read_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name=READ_TOOL_NAME,
        description=(
            "Read a bounded range from a UTF-8 text file using a path relative "
            "to the current working directory. file_path, offset, and limit "
            "are required. offset is the 1-based starting line; limit is the "
            "maximum number of lines to return. The result contains 1-based "
            "line numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["file_path", "offset", "limit"],
            "additionalProperties": False,
        },
    )


def glob_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name=GLOB_TOOL_NAME,
        description=(
            "Find files under the current working directory. Pass a relative "
            "glob pattern; returned paths are relative and sorted."
        ),
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
            "additionalProperties": False,
        },
    )


def grep_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name=GREP_TOOL_NAME,
        description="Search UTF-8 files under allowed read roots with a regular expression.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 1},
                "path": {"type": "string", "default": "**/*"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    )


def edit_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name=EDIT_TOOL_NAME,
        description="Replace one exact unique string in an allowed UTF-8 file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "old_string": {"type": "string", "minLength": 1},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    )


def write_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name=WRITE_TOOL_NAME,
        description="Create or overwrite one UTF-8 file under an allowed write root.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
            "additionalProperties": False,
        },
    )


class SandboxedToolRuntime:
    """Execute shell-free local ReAct tools with bounded result size."""

    def __init__(
        self,
        *,
        cwd: str,
        read_roots: Iterable[str] = (),
        write_roots: Iterable[str] = (),
        glob_result_limit: int = 1_000,
        tool_result_max_chars: int = 30_000,
        result_store: str | None = None,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        roots = [Path(root).resolve() for root in read_roots]
        self.read_roots = tuple(roots or [self.cwd])
        self.write_roots = tuple(Path(root).resolve() for root in write_roots)
        self.glob_result_limit = glob_result_limit
        if tool_result_max_chars < 512:
            raise ValueError("tool_result_max_chars must be >= 512")
        self.tool_result_max_chars = tool_result_max_chars
        self.result_store = Path(result_store).resolve() if result_store is not None else None
        self._result_counter = 0
        for root in self.read_roots:
            if not root.is_dir():
                raise ValueError(f"read root is not a directory: {root}")
            if not _is_relative_to(root, self.cwd):
                raise ValueError(f"read root is outside cwd: {root}")
        for root in self.write_roots:
            if not root.is_dir():
                raise ValueError(f"write root is not a directory: {root}")
            if not _is_relative_to(root, self.cwd):
                raise ValueError(f"write root is outside cwd: {root}")

    def definitions(self, names: Iterable[str]) -> tuple[LLMToolDefinition, ...]:
        requested = list(dict.fromkeys(str(name) for name in names))
        supported = {
            READ_TOOL_NAME,
            GLOB_TOOL_NAME,
            GREP_TOOL_NAME,
            EDIT_TOOL_NAME,
            WRITE_TOOL_NAME,
        }
        unsupported = sorted(set(requested) - supported)
        if unsupported:
            raise ValueError(f"unsupported API tools: {unsupported}")
        out: list[LLMToolDefinition] = []
        if READ_TOOL_NAME in requested:
            out.append(read_tool_definition())
        if GLOB_TOOL_NAME in requested:
            out.append(glob_tool_definition())
        if GREP_TOOL_NAME in requested:
            out.append(grep_tool_definition())
        if EDIT_TOOL_NAME in requested:
            if not self.write_roots:
                raise ValueError("Edit requires write_roots")
            out.append(edit_tool_definition())
        if WRITE_TOOL_NAME in requested:
            if not self.write_roots:
                raise ValueError("Write requires write_roots")
            out.append(write_tool_definition())
        return tuple(out)

    def execute(self, name: str, arguments: str) -> str:
        try:
            payload = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON arguments: {exc}"
        if not isinstance(payload, dict):
            return "Error: tool arguments must be an object"
        try:
            if name == READ_TOOL_NAME:
                result = self._read(payload)
            elif name == GLOB_TOOL_NAME:
                result = self._glob(payload)
            elif name == GREP_TOOL_NAME:
                result = self._grep(payload)
            elif name == EDIT_TOOL_NAME:
                result = self._edit(payload)
            elif name == WRITE_TOOL_NAME:
                result = self._write(payload)
            else:
                result = f"Error: unsupported tool: {name}"
            return self.limit_result(result, label=name)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    def archive_result(self, content: str, *, label: str) -> str:
        """Persist full tool content and return the reference to read it by.

        The store is never inside cwd; without a configured ``result_store``
        it falls back to a system temporary directory.
        """
        if self.result_store is None:
            self.result_store = Path(tempfile.mkdtemp(prefix="llm-api-tool-results-")).resolve()
        self.result_store.mkdir(parents=True, exist_ok=True)
        self._result_counter += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
        target = self.result_store / (f"{self._result_counter:04d}-{safe_label or 'tool'}.txt")
        target.write_text(content, encoding="utf-8")
        return f"{_RESULT_STORE_ALIAS}/{target.name}"

    def limit_result(self, content: str, *, label: str) -> str:
        """Bound a tool observation while retaining its full archived value."""
        if len(content) <= self.tool_result_max_chars:
            return content
        reference = self.archive_result(content, label=label)
        prefix = f"[Tool result truncated: original_chars={len(content)}]\n"
        if reference:
            prefix += f"Full result: {reference}\nUse Read with this path to inspect the full result if needed.\n\n"
        else:
            prefix += "\n"
        remaining = max(0, self.tool_result_max_chars - len(prefix) - 32)
        head_size = remaining * 2 // 3
        tail_size = remaining - head_size
        body = content[:head_size]
        if tail_size:
            body += "\n... truncated ...\n" + content[-tail_size:]
        return prefix + body

    def _read(self, payload: dict[str, Any]) -> str:
        offset = _required_positive_int(payload, "offset")
        limit = _required_positive_int(payload, "limit")
        target, relative = self._resolve_file(payload.get("file_path"))
        if not target.is_file():
            raise FileNotFoundError(relative)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(len(lines), offset - 1 + limit)
        return "\n".join(f"{line_number:6d}\t{lines[line_number - 1]}" for line_number in range(offset, end + 1))

    def _glob(self, payload: dict[str, Any]) -> str:
        pattern = payload.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("pattern must be a non-empty string")
        normalized = pattern.replace("\\", "/").strip()
        path_pattern = Path(normalized)
        if path_pattern.is_absolute() or ".." in path_pattern.parts:
            raise ValueError("absolute patterns and '..' are not allowed")
        matches: list[str] = []
        for raw_match in glob.iglob(
            str(self.cwd / normalized),
            recursive=True,
        ):
            candidate = Path(raw_match).resolve()
            if not candidate.is_file():
                continue
            if not _is_relative_to(candidate, self.cwd):
                continue
            if not any(_is_relative_to(candidate, root) for root in self.read_roots):
                continue
            matches.append(candidate.relative_to(self.cwd).as_posix())
        unique = sorted(set(matches))
        if len(unique) > self.glob_result_limit:
            shown = unique[: self.glob_result_limit]
            return "\n".join(
                [
                    *shown,
                    f"... truncated {len(unique) - len(shown)} additional file(s)",
                ]
            )
        return "\n".join(unique)

    def _resolve_write_file(self, raw_path: Any) -> tuple[Path, str]:
        target, relative = self._resolve_file(raw_path, roots=self.write_roots)
        return target, relative

    def _resolve_file(
        self,
        raw_path: Any,
        *,
        roots: tuple[Path, ...] | None = None,
    ) -> tuple[Path, str]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("file_path must be a non-empty string")
        relative = Path(raw_path.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("absolute paths and '..' are not allowed")
        archived = self._resolve_archived_result(relative, roots)
        if archived is not None:
            return archived
        target = (self.cwd / relative).resolve()
        allowed = self.read_roots if roots is None else roots
        if not any(_is_relative_to(target, root) for root in allowed):
            label = "read_roots" if roots is None else "write_roots"
            raise ValueError(f"path is outside configured {label}")
        return target, target.relative_to(self.cwd).as_posix()

    def _resolve_archived_result(
        self,
        relative: Path,
        roots: tuple[Path, ...] | None,
    ) -> tuple[Path, str] | None:
        """Resolve ``tool-results/<name>`` against the out-of-cwd store.

        Read-only: the alias names the run's record of tool output and must
        never be writable. ``roots`` is non-None exactly for the write path.
        """
        if roots is not None or self.result_store is None:
            return None
        parts = relative.parts
        if len(parts) != 2 or parts[0] != _RESULT_STORE_ALIAS:
            return None
        target = (self.result_store / parts[1]).resolve()
        if not _is_relative_to(target, self.result_store):
            raise ValueError("path is outside configured read_roots")
        return target, relative.as_posix()

    def _grep(self, payload: dict[str, Any]) -> str:
        pattern = payload.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        compiled = re.compile(pattern)
        path_pattern = str(payload.get("path") or "**/*")
        matches: list[str] = []
        for raw_match in glob.iglob(str(self.cwd / path_pattern), recursive=True):
            candidate = Path(raw_match).resolve()
            if not candidate.is_file() or not any(_is_relative_to(candidate, root) for root in self.read_roots):
                continue
            for line_number, line in enumerate(candidate.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if compiled.search(line):
                    relative = candidate.relative_to(self.cwd).as_posix()
                    matches.append(f"{relative}:{line_number}:{line}")
                    if len(matches) >= self.glob_result_limit:
                        return "\n".join(matches + ["... results truncated"])
        return "\n".join(matches)

    def _edit(self, payload: dict[str, Any]) -> str:
        target, relative = self._resolve_write_file(payload.get("file_path"))
        if not target.is_file():
            raise FileNotFoundError(relative)
        old = payload.get("old_string")
        new = payload.get("new_string")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("old_string must be non-empty and new_string must be a string")
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old_string must occur exactly once; found {count}")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {relative}"

    def _write(self, payload: dict[str, Any]) -> str:
        target, relative = self._resolve_write_file(payload.get("file_path"))
        content = payload.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {relative}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _required_positive_int(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be provided as a positive integer")
    return value


__all__ = [
    "GLOB_TOOL_NAME",
    "GREP_TOOL_NAME",
    "EDIT_TOOL_NAME",
    "READ_TOOL_NAME",
    "WRITE_TOOL_NAME",
    "SandboxedToolRuntime",
    "glob_tool_definition",
    "read_tool_definition",
]
