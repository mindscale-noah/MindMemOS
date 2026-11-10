"""Portable structured payload redaction and serialization."""

import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from ..common.secrets import is_sensitive_key

_MAX_CHARS = 32768
_SECRET = re.compile(
    r"""(?i)(\b(?:[\w-]*api[_-]?key|[\w-]*token|[\w-]*password|[\w-]*secret|authorization)\b["']?\s*[:=]\s*)(?:["'][^"'\n]*["']|[^\s,;}]+)"""
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def safe_text(text: str) -> str:
    """Redact credential assignments and bound a rendered text payload.

    Args:
        text: Rendered message or traceback without exception locals.

    Returns:
        Redacted text limited to 32768 characters, including its truncation marker.
    """
    text = _SECRET.sub(r"\1[REDACTED]", _BEARER.sub("Bearer [REDACTED]", text))
    marker = "...[truncated]"
    return text if len(text) <= _MAX_CHARS else text[: _MAX_CHARS - len(marker)] + marker


def structured_text(value: Any) -> str:
    """Serialize a structured log payload with credential redaction and a size limit.

    Args:
        value: Nested mappings, sequences, Pydantic models or JSON-compatible values.

    Returns:
        JSON text with redacted credential keys, limited to 32768 characters.
        Truncated output ends with a marker and may no longer be valid JSON.
    """

    def sanitize(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return sanitize(item.model_dump(mode="python"))
        if isinstance(item, Mapping):
            return {
                str(key): "[REDACTED]" if is_sensitive_key(str(key)) else sanitize(child) for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [sanitize(child) for child in item]
        if isinstance(item, str):
            return safe_text(item)
        if item is None or isinstance(item, (int, float, bool)):
            return item
        try:
            return safe_text(str(item))
        except Exception:
            return f"<{type(item).__name__}>"

    content = json.dumps(sanitize(value), ensure_ascii=False, default=str)
    marker = "...[truncated]"
    return content if len(content) <= _MAX_CHARS else content[: _MAX_CHARS - len(marker)] + marker
