"""Credential field recognition and redaction for JSON configuration snapshots."""

from pydantic import JsonValue

_SENSITIVE_KEYS = {
    "api_key",
    "password",
    "token",
    "secret",
    "authorization",
    "base_url",
    "model",
    "model_name",
    "agent",
}


def is_sensitive_key(key: str) -> bool:
    """Recognize credential field names without inspecting their values.

    Args:
        key: Configuration or structured log field name.

    Returns:
        Whether the field contains a credential, including prefixed names.
    """
    name = key.lower().replace("-", "_")
    return name in _SENSITIVE_KEYS or name.endswith(("_api_key", "_token", "_password", "_secret"))


def redact_credentials(value: JsonValue) -> JsonValue:
    """Copy a JSON snapshot and mask credential fields at every nesting level.

    Args:
        value: Serialized configuration containing JSON objects, arrays and scalars.
            Credential fields are identified by name; ordinary text is unchanged.

    Returns:
        A new snapshot with nonempty credentials replaced by ``[REDACTED]``.
        Missing or empty credentials remain distinguishable from configured ones.
    """
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if is_sensitive_key(key) and child not in (None, "") else redact_credentials(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_credentials(child) for child in value]
    return value
