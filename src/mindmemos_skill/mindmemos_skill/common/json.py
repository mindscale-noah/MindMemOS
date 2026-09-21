"""JSON files, append-only records, and guarded decoding of model responses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeVar

from filelock import FileLock
from json_repair import repair_json
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def _serialize(value: object, *, indent: int | None = None) -> str:
    """Encode models recursively without accepting non-finite numbers."""

    def encode(item: object) -> object:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"Cannot serialize {type(item).__name__} as JSON")

    return json.dumps(value, default=encode, ensure_ascii=False, allow_nan=False, indent=indent)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file without repairing stored data.

    Args:
        path: Existing file to decode.

    Returns:
        The decoded JSON value.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file contains invalid JSON or non-finite numbers.
    """
    return _loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: object) -> None:
    """Replace a JSON file atomically, creating its parent directories.

    Each writer uses a unique temporary file; concurrent replacements are
    last-writer-wins. This is not a transactional read/modify/write operation.

    Args:
        path: Destination file.
        value: JSON-compatible data, including nested Pydantic models.

    Raises:
        OSError: Writing or replacing the file fails.
        TypeError: A value is not serializable.
        ValueError: A number is non-finite.
    """
    content = _serialize(value, indent=2) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, value: object) -> None:
    """Append one JSONL record under a cooperative interprocess file lock.

    Args:
        path: Destination file; parents are created as needed.
        value: One record, including a Pydantic model. A list is one record.

    Raises:
        OSError: The file cannot be written.
        TypeError: A value is not serializable.
        ValueError: A number is non-finite.
    """
    record = (_serialize(value) + "\n").encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        with destination.open("a+b") as stream:
            if stream.tell():
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write(record)


def read_jsonl(path: str | Path) -> Iterator[Any]:
    """Stream JSONL records, skipping blank lines and reporting corrupt lines.

    Read after writers finish; this does not snapshot an actively growing file.

    Args:
        path: Existing UTF-8 JSONL file.

    Yields:
        Each decoded record in file order.

    Raises:
        OSError: The file cannot be read.
        ValueError: A record is invalid; the error includes its line number.
    """
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield _loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {number}: {exc}") from exc


def _reject_constant(value: str) -> None:
    """Reject JavaScript non-finite constants accepted by the standard decoder."""
    raise ValueError(f"Invalid JSON number: {value}")


def _loads(text: str) -> Any:
    """Decode strict JSON syntax without allowing NaN or Infinity."""
    return json.loads(text, parse_constant=_reject_constant)


def _candidates(text: str, *, allow_incomplete: bool) -> list[str]:
    """Scan non-overlapping containers without extracting their nested values."""
    candidates = []
    start = None
    closing = []
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char == '"' or (start is not None and char == "'"):
            quote = char
        elif char in "{[":
            if start is None:
                start = index
            closing.append("}" if char == "{" else "]")
        elif start is not None and char in "}]":
            if char != closing.pop():
                raise ValueError("Mismatched JSON brackets")
            if not closing:
                candidates.append(text[start : index + 1])
                start = None
    if start is not None:
        if not allow_incomplete or quote is not None or not text[start + 1 :].strip():
            raise ValueError("Unterminated JSON container or string")
        candidates.append(text[start:].rstrip() + "".join(reversed(closing)))
    return candidates


def extract_json(text: str, *, repair: bool = True, allow_incomplete: bool = False) -> dict | list:
    """Extract a unique JSON object or array from an LLM reply.

    A JSON code fence takes precedence over surrounding prose. Multiple
    candidates in the selected region are rejected. Repair is limited to
    JSON-looking containers and never applied to ordinary prose or stored files.

    Args:
        text: Model reply, optionally containing fences and explanatory text.
        repair: Repair simple syntax errors, such as trailing commas or quotes.
        allow_incomplete: Close missing final brackets when no string is open.
            Disabled by default; intended for legacy textual tool protocols.

    Returns:
        The decoded object or array.

    Raises:
        ValueError: JSON is absent, ambiguous, truncated, or cannot be decoded.
        TypeError: The response is not text.
    """
    if not isinstance(text, str):
        raise TypeError("JSON extraction requires text")
    stripped = text.strip().lstrip("\ufeff")
    try:
        value = _loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(value, (dict, list)):
            raise ValueError("Expected a JSON object or array")
        return value

    fences = re.findall(r"```(?:json\b)?[ \t]*(?:\r?\n)?(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    regions = [block for block in fences if block.lstrip().startswith(("{", "["))]
    candidates = []
    for region in regions or [stripped]:
        candidates.extend(_candidates(region, allow_incomplete=allow_incomplete))
    if len(candidates) != 1:
        raise ValueError(f"Expected one JSON container, found {len(candidates)}")
    candidate = candidates[0]
    try:
        return _loads(candidate)
    except json.JSONDecodeError:
        if not repair:
            raise
    # Bare prose such as {operation: delete} or [todo] is not repairable input.
    interior = candidate[1:].lstrip()
    allowed = "\"'}" if candidate.startswith("{") else "\"'[{]-0123456789"
    if not interior or (
        interior[0] not in allowed
        and not (candidate.startswith("[") and interior.startswith(("true", "false", "null")))
    ):
        raise ValueError("Response does not contain a JSON-like container")
    value = repair_json(candidate, return_objects=True, skip_json_loads=True, strict=True)
    if not isinstance(value, (dict, list)):
        raise ValueError("JSON repair did not produce a container")
    _serialize(value)
    return value


def extract_json_object(text: str, *, repair: bool = True, allow_incomplete: bool = False) -> dict[str, Any]:
    """Extract a JSON object, rejecting arrays rather than selecting their items.

    Args:
        text: Raw model reply.
        repair: Allow guarded syntax repair.
        allow_incomplete: Allow missing closing brackets in legacy tool replies.

    Returns:
        A decoded mapping.

    Raises:
        ValueError: Extraction fails or the payload is not an object.
    """
    value = extract_json(text, repair=repair, allow_incomplete=allow_incomplete)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def extract_json_array(text: str, *, repair: bool = True) -> list[Any]:
    """Extract a JSON array without selecting nested arrays from objects.

    Args:
        text: Raw model reply.
        repair: Allow guarded syntax repair.

    Returns:
        A decoded array.

    Raises:
        ValueError: Extraction fails or the payload is not an array.
    """
    value = extract_json(text, repair=repair)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return value


def parse_model(text: str, model_type: type[ModelT], *, repair: bool = True) -> ModelT:
    """Extract JSON and validate it with the requested Pydantic model.

    Args:
        text: Raw model reply.
        model_type: Model class, including RootModel for array payloads.
        repair: Allow guarded syntax repair before schema validation.

    Returns:
        A validated instance of model_type.

    Raises:
        ValueError: Extraction fails or Pydantic validation rejects the payload.
    """
    return model_type.model_validate(extract_json(text, repair=repair))


def strict_json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible strict JSON schema request constraint.

    Args:
        name: Stable schema name sent to the provider.
        schema: JSON schema accepted by the provider.

    Returns:
        A response_format mapping; the input schema is not modified.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
