"""Shared bounded-concurrency execution and progress helpers."""

from __future__ import annotations

import contextvars
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel_batch(
    items: Sequence[T],
    fn: Callable[[int, T], R],
    *,
    workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Run ``fn(index, item)`` concurrently and return results in input order.

    Exceptions from ``fn`` propagate. ``fn`` runs in a copy of the caller's
    context so contextvars (e.g. the bound LLM backend) stay visible.
    ``progress_callback(completed, total)`` fires once per finished item.
    """
    n = len(items)
    if n == 0:
        return []
    workers = max(1, min(int(workers), n)) if workers > 0 else 1
    results: list[R | None] = [None] * n
    completed = 0
    parent_context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(parent_context.copy().run, fn, i, items[i]): i for i in range(n)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, n)
    return [r for r in results if r is not None]  # type: ignore[misc]


@dataclass
class TqdmProgress:
    """Adapter matching the batch callback signature ``(completed, total)``."""

    label: str
    total: int
    enabled: bool = True

    def __post_init__(self) -> None:
        self._bar = None
        if not self.enabled:
            return
        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            self.enabled = False
            return

        self._bar = tqdm(
            total=max(0, self.total),
            desc=self.label,
            unit="item",
            file=sys.stderr,
            leave=False,
        )

    def __call__(self, completed: int, total: int) -> None:
        if not self.enabled or self._bar is None:
            return
        if total != self.total:
            self._bar.total = total
        delta = completed - self._bar.n
        if delta > 0:
            self._bar.update(delta)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


__all__ = ["TqdmProgress", "run_parallel_batch"]
