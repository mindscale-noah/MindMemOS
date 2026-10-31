"""Safe durable metadata for deferred dense-vector repair."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def embedding_model_fingerprint(*, expected_dimension: int | None, models: list[dict[str, Any]] | None = None) -> str:
    """Hash only non-secret embedding identity fields for operational comparison."""

    endpoints = [
        {
            "model": str(endpoint.get("model") or ""),
            "transport": str(endpoint.get("transport") or "litellm"),
            "dimensions": endpoint.get("dimensions"),
        }
        for endpoint in (models or [])
    ]
    encoded = json.dumps(
        {"expected_dimension": expected_dimension, "endpoints": endpoints},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def initial_vector_repair_metadata(expected_dimension: int | None) -> dict[str, Any]:
    """Return restart-safe initial state without leaking provider configuration."""

    return {
        "vector_pending": True,
        "vector_expected_dimension": expected_dimension,
        "vector_retry_count": 0,
        "vector_next_retry_at_ms": 0,
        "vector_last_error_code": "embedding.provider_unavailable",
        "vector_model_fingerprint": embedding_model_fingerprint(expected_dimension=expected_dimension),
    }
