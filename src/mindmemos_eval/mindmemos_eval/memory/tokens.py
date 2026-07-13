"""Shared token accounting helpers for benchmark environments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def stage_metrics(
    prefix: str,
    *,
    llm_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
) -> dict[str, int]:
    """Return normalized metric keys for one benchmark stage."""
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    total = int(total_tokens) if total_tokens is not None else prompt + completion
    return {
        f"{prefix}_llm_calls": int(llm_calls or 0),
        f"{prefix}_prompt_tokens": prompt,
        f"{prefix}_completion_tokens": completion,
        f"{prefix}_total_tokens": total,
    }


def completion_stage_metrics(prefix: str, completion: Any | None) -> dict[str, int]:
    """Build stage metrics from an LLM completion object."""
    if completion is None:
        return stage_metrics(prefix)
    return stage_metrics(
        prefix,
        llm_calls=1,
        prompt_tokens=int(getattr(completion, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(completion, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(completion, "total_tokens", 0) or 0),
    )


def aggregate_stage_metrics(items: Iterable[Any], *prefixes: str) -> dict[str, int]:
    """Aggregate prefixed token metrics across result rows."""
    values = list(items)
    merged: dict[str, int] = {}
    for prefix in prefixes:
        merged.update(
            stage_metrics(
                prefix,
                llm_calls=sum(int(getattr(item, f"{prefix}_llm_calls", 0) or 0) for item in values),
                prompt_tokens=sum(int(getattr(item, f"{prefix}_prompt_tokens", 0) or 0) for item in values),
                completion_tokens=sum(
                    int(getattr(item, f"{prefix}_completion_tokens", 0) or 0) for item in values
                ),
                total_tokens=sum(int(getattr(item, f"{prefix}_total_tokens", 0) or 0) for item in values),
            )
        )
    return merged
