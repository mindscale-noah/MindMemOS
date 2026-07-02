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


def search_stage_metrics(search: Any | None) -> dict[str, int]:
    """Build search-stage token metrics from a search result object."""
    if search is None:
        return stage_metrics("search")
    prompt_tokens = int(getattr(search, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(search, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(search, "total_tokens", 0) or 0)
    llm_calls = int(getattr(search, "llm_call_count", 0) or 0)
    if llm_calls == 0 and total_tokens > 0:
        llm_calls = 1
    return stage_metrics(
        "search",
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
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
