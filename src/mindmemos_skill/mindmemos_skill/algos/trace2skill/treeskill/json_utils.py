"""Small JSON response helpers shared by TreeSkill model calls."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one balanced JSON object, respecting braces inside strings."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    start = stripped.find("{")
    if start < 0:
        raise ValueError("response did not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(stripped[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("response JSON must be an object")
                return parsed
    raise ValueError("response contained an unterminated JSON object")


def parse_model(text: str, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(extract_json_object(text))


__all__ = ["extract_json_object", "parse_model"]
