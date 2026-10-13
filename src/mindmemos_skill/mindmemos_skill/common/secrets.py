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


def redact_credentials(value: JsonValue, *, preserve_execution_settings: bool = False) -> JsonValue:
    """Copy a JSON snapshot and mask credential fields at every nesting level.

    Args:
        value: Serialized configuration containing JSON objects, arrays and scalars.
            Credential fields are identified by name; ordinary text is unchanged.
        preserve_execution_settings: Keep model names, provider URLs and nested
            agent settings when comparing execution configurations. Mask empty
            credentials too so credential availability does not affect comparison.

    Returns:
        A new snapshot with nonempty credentials replaced by ``[REDACTED]``.
        By default, empty credentials remain distinguishable from configured ones.
    """
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            sensitive = is_sensitive_key(key)
            if preserve_execution_settings and key.lower().replace("-", "_") in {
                "base_url",
                "model",
                "model_name",
                "agent",
            }:
                sensitive = False
            result[key] = (
                "[REDACTED]"
                if sensitive and (preserve_execution_settings or child not in (None, ""))
                else redact_credentials(child, preserve_execution_settings=preserve_execution_settings)
            )
        return result
    if isinstance(value, list):
        return [redact_credentials(child, preserve_execution_settings=preserve_execution_settings) for child in value]
    return value
