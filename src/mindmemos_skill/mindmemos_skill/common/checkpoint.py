"""Atomic input-bound checkpoints shared by local execution owners."""

import hashlib
import json
from pathlib import Path

from .json import read_json, write_json
from .typing import RecoveryRecord


def fingerprint(value) -> str:
    """Hash a JSON-compatible execution identity.

    Args:
        value: Canonical request inputs, excluding credentials.

    Returns:
        SHA-256 digest independent of mapping insertion order.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def boundary(path: Path, identity, *, payload=None):
    """Read a matching checkpoint or atomically commit its complete payload.

    Args:
        path: Recovery record location.
        identity: Inputs that must remain unchanged across recovery.
        payload: Completed data to save; None requests a read only.

    Returns:
        Saved payload, or None when no completed record exists.

    Raises:
        ValueError: A checkpoint has changed identity or corrupted contents.
    """
    key = fingerprint(identity)
    if path.exists():
        saved = read_boundary(path)
        if saved.identity != key:
            raise ValueError(f"Recovery checkpoint mismatch: {path}")
        if payload is not None and saved.payload != payload:
            raise ValueError(f"Recovery checkpoint contents changed: {path}")
        return saved.payload
    if payload is not None:
        write_json(path, RecoveryRecord(identity=key, payload=payload, digest=fingerprint(payload)))
    return payload


def read_boundary(path: Path) -> RecoveryRecord:
    """Validate a stored checkpoint before replay enters error-tolerant stages.

    Args:
        path: Existing checkpoint with an input identity and payload checksum.

    Returns:
        Validated record; the owner still checks its expected input identity.

    Raises:
        ValueError: The record is invalid or its payload checksum differs.
    """
    saved = RecoveryRecord.model_validate(read_json(path))
    if saved.digest != fingerprint(saved.payload):
        raise ValueError(f"Recovery checkpoint mismatch: {path}")
    return saved


def redact(value):
    """Remove credentials from JSON-compatible run identity data.

    Args:
        value: Serialized configuration containing nested mappings and sequences.

    Returns:
        A new value without credential fields; the input is unchanged.
    """
    if isinstance(value, dict):
        return {
            key: redact(item) for key, item in value.items() if key not in {"api_key", "password", "token", "secret"}
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
