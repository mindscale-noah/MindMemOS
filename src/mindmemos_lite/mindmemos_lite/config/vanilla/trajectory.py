"""Trajectory task+experience add configuration.

Drives the trajectory pipeline that imports one trajectory as a single task
entity plus a set of reusable experience memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryAddConfig:
    """Compose the trajectory add pipeline configuration."""

    experience_recall_top_k: int = field(default=5)
    """Maximum existing experiences recalled for one candidate before LLM dedup."""

    embedding_batch_size: int = field(default=32)
    """Maximum number of experience texts sent in one native embedding request."""

    min_content_chars: int = field(default=1)
    """Minimum normalized experience content length; shorter candidates are skipped."""

    enable_task_entity_embedding: bool = field(default=True)
    """Whether to embed the task entity (dense semantic + sparse bm25). Task
    experience search matches tasks semantically, so tasks need embeddings.
    NER-based entity extraction stays off regardless."""

    task_search_score_threshold: float | None = field(default=0.45)
    """Minimum dense similarity (1 - cosine distance, ~[-1, 1]) for a task
    candidate to be returned by task experience search. Requests can override
    per-call via the search ``score_threshold``. Short Chinese task names share
    "帮我…" prefixes that inflate similarity; 0.45 keeps genuine paraphrases
    while dropping clearly unrelated queries. Set lower for fuzzier matches."""

    tool_message_head_tokens: int = field(default=4000)
    """Head budget kept verbatim from an oversized ``tool`` message.

    Counted with the shared whitespace/CJK token heuristic, which under-counts
    real provider tokens (roughly 2x on tool output), so treat this as a
    heuristic budget rather than an exact one. The trajectory pipeline sends
    every message to the extractor in one call (no chunk planning), so one huge
    tool output is the main thing that can dominate the prompt. Its head and
    tail survive verbatim; the middle is replaced by a truncation marker."""

    tool_message_tail_tokens: int = field(default=4000)
    """Tail budget kept verbatim from an oversized ``tool`` message (heuristic tokens)."""
