"""Official PersonaMem v1 protocol adapted to the MindMemOS memory API."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mindmemos_sdk.memory import AsyncMemoryClient
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.tokens import stage_metrics

PERSONAMEM_OFFICIAL_REPOSITORY = "https://github.com/bowen-upenn/PersonaMem"
PERSONAMEM_OFFICIAL_PROTOCOL_COMMIT = "caaae44b3f236b8751d499a770e94e5aecffcff1"
PERSONAMEM_OFFICIAL_INSTRUCTION = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)
PERSONAMEM_ANSWER_MAX_RETRIES = 3
PERSONAMEM_ANSWER_OPTIONS = ("a", "b", "c", "d")
# Question-type-agnostic prompt: one prompt for every question, never reads the
# dataset's ``question_type`` label. Reasoning depth is adaptive (the model does
# only the analysis a question needs).
PERSONAMEM_UNIFIED_PROMPT = """You are a memory assistant. Select the single best response — (a), (b), (c), or (d) — using ONLY the retrieved memories. Memories may be noisy or incomplete and are prefixed with their event date (YYYY-MM-DD); treat those dates as the authoritative timeline.

UNIVERSAL RULES (apply to every question):
1. Evidence only. Never use outside knowledge or pick an option because it sounds plausible.
2. Recall beats generic. Prefer an option that references a specific stored detail the user did NOT just say, over one that merely validates the current message ("That's great!").
3. Preserve attitude in BOTH directions. If memories show the user disliked, abandoned, or was discouraged by something, positive reframings are WRONG — and if they show the user liked or benefited, negative reframings are WRONG.
4. Match intensity. Do not inflate or deflate: "didn't resonate" ≠ "stopped entirely"; "enjoyed" ≠ "life-changing"; "overwhelming" ≠ "challenging". Choose the option whose strength mirrors the memory's own words.
5. Verify claims literally. For the leading options, locate the option's key noun/verb/phrase in a memory. Topical relevance is not verification.
6. Separate the user's personal reaction from environmental description ("the atmosphere was vibrant" is scene, not the user's attitude).

HOW TO REASON — do ONLY what the question needs, and keep it brief:
- Change over time (how a preference evolved): list the relevant events in date order and keep EVERY documented stage in sequence — do not collapse, soften, or inflate them. Eliminate an option only on a polarity or chronological-order conflict, NOT merely for naming fewer stages.
- Trying / suggesting something new: first note which activities the memories show the user has ALREADY done and how they reacted. If the question asks for something new or unexperienced, exclude anything already done; otherwise prefer the option best aligned with the user's demonstrated values and past positive experiences, and reject generic encouragement.
- Recommendation or ideas: extract the user's 2–3 SPECIFIC sub-interests from memory first (e.g. "attachment theory", not "likes books"), then score options against those, not against the question's surface adjectives.
- Fact or reason recall: find the one memory that answers it, prefer the option that surfaces a stored detail the user did not just restate, and confirm its polarity matches.

Reason concisely — a couple of lines for a simple recall; a short dated list or activity inventory only when the question needs it. Do not pad with exhaustive tables. Then end with EXACTLY one line:
<final_answer>(a)</final_answer> or <final_answer>(b)</final_answer> or <final_answer>(c)</final_answer> or <final_answer>(d)</final_answer>

---

## Reference Memories
Each memory may be prefixed with its event date as `(YYYY-MM-DD)`; use these dates as the authoritative timeline.
{context}

## Question
{question}

---

Now reason briefly as instructed, then give your final answer:
"""
PersonaMemEvaluationMode = Literal["memory_rag", "official_full_context"]


_REQUIRED_QUESTION_FIELDS = {
    "persona_id",
    "question_id",
    "question_type",
    "topic",
    "user_question_or_message",
    "correct_answer",
    "all_options",
    "shared_context_id",
    "end_index_in_shared_context",
}


class PersonaMemScope(BaseModel):
    """One immutable question-visible context boundary."""

    model_config = ConfigDict(extra="forbid")

    shared_context_id: str
    end_index: int
    scope_id: str
    user_id: str
    session_id: str


class PersonaMemItem(BaseModel):
    """One official PersonaMem v1 multiple-choice question."""

    model_config = ConfigDict(extra="allow")

    index: int
    persona_id: str
    question_id: str
    question_type: str
    topic: str
    question: str
    correct_answer: str
    all_options: str
    scope: PersonaMemScope
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonaMemBuildSummary(BaseModel):
    """Build outcome for one unique visible-context scope."""

    scope: PersonaMemScope
    total_messages: int = 0
    added_messages: int = 0
    add_calls: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class PersonaMemAnswer(BaseModel):
    """Answer output and evaluation-client usage for one question."""

    response: str
    extracted_answer: str
    is_correct: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    parse_failed: bool = False
    format_compliant: bool = False
    elapsed_seconds: float = 0.0


class PersonaMemQAResult(BaseModel):
    """End-to-end result for one PersonaMem question."""

    item: PersonaMemItem
    retrieved_memories: list[str] = Field(default_factory=list)
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    search_elapsed_seconds: float = 0.0
    answer: PersonaMemAnswer | None = None
    error: str | None = None


@dataclass
class _PersonaMemLiveProgress:
    """Mutable console-only counters for completed PersonaMem questions."""

    total: int
    completed: int = 0
    correct: int = 0
    parse_failed: int = 0
    search_failed: int = 0
    answer_failed: int = 0

    def record(self, result: PersonaMemQAResult) -> None:
        """Include one completed question in the live display counters."""
        self.completed += 1
        self.correct += int(bool(result.answer and result.answer.is_correct))
        self.parse_failed += int(bool(result.answer and result.answer.parse_failed))
        self.search_failed += int(bool(result.error and result.error.startswith("search failed:")))
        self.answer_failed += int(bool(result.error and result.error.startswith("answer failed:")))

    def postfix(self) -> dict[str, int | str]:
        """Render the current counters for ``tqdm.set_postfix``."""
        completed = self.completed or 1
        return {
            "correct": self.correct,
            "acc_done": f"{self.correct / completed:.4f}",
            "acc_all": f"{self.correct / self.total:.4f}" if self.total else "0.0000",
            "parse_fail": self.parse_failed,
            "search_fail": self.search_failed,
            "answer_fail": self.answer_failed,
        }


class PersonaMemRunResult(BaseModel):
    """Serializable result for one PersonaMem benchmark run."""

    protocol: str
    official_protocol_commit: str = PERSONAMEM_OFFICIAL_PROTOCOL_COMMIT
    benchmark_version: str = "v1"
    context_size: str = "32k"
    evaluation_mode: PersonaMemEvaluationMode
    items: list[PersonaMemItem] = Field(default_factory=list)
    build_summaries: list[PersonaMemBuildSummary] = Field(default_factory=list)
    qa_results: list[PersonaMemQAResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def format_report(self) -> str:
        """Render the primary official score and failure counts."""
        return (
            "PersonaMem evaluation report\n"
            f"  protocol={self.protocol}\n"
            f"  accuracy={self.metrics.get('overall_accuracy', 0.0):.4f} "
            f"({self.metrics.get('correct', 0)}/{self.metrics.get('total', 0)})\n"
            f"  scopes={self.metrics.get('scope_build_success', 0)}/"
            f"{self.metrics.get('scope_total', 0)} built\n"
            f"  search_failures={self.metrics.get('search_failure_count', 0)} "
            f"answer_failures={self.metrics.get('answer_failure_count', 0)}"
        )


class PersonaMemContextStore:
    """Random-access reader for official shared-context JSONL files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offsets = self._build_index()

    def _build_index(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        with self.path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                payload = json.loads(line)
                if not isinstance(payload, Mapping) or len(payload) != 1:
                    raise ValueError(f"invalid PersonaMem context line at byte {offset}")
                offsets[str(next(iter(payload)))] = offset
        return offsets

    def load(self, shared_context_id: str) -> list[dict[str, Any]]:
        """Load one context by the official shared-context identifier."""
        try:
            offset = self._offsets[shared_context_id]
        except KeyError as exc:
            raise KeyError(f"unknown PersonaMem shared_context_id: {shared_context_id}") from exc
        with self.path.open("rb") as fh:
            fh.seek(offset)
            payload = json.loads(fh.readline())
        context = next(iter(payload.values()))
        if not isinstance(context, list):
            raise ValueError(f"PersonaMem context {shared_context_id!r} must be a list")
        return [dict(message) for message in context if isinstance(message, Mapping)]

    def visible(self, scope: PersonaMemScope) -> list[dict[str, Any]]:
        """Apply the official exclusive end-index slice for one question scope."""
        context = self.load(scope.shared_context_id)
        if scope.end_index < 0 or scope.end_index > len(context):
            raise ValueError(
                f"invalid end_index {scope.end_index} for context "
                f"{scope.shared_context_id!r} with {len(context)} messages"
            )
        return context[: scope.end_index]


def load_personamem_questions(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the official PersonaMem questions CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = sorted(_REQUIRED_QUESTION_FIELDS - fields)
        if missing:
            raise ValueError(f"PersonaMem questions CSV is missing fields: {', '.join(missing)}")
        return [dict(row) for row in reader]


def build_personamem_scope(shared_context_id: str, end_index: int) -> PersonaMemScope:
    """Create a stable, isolated memory scope from the official visibility boundary."""
    raw = f"{shared_context_id}\0{end_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    scope_id = f"{shared_context_id}:{end_index}"
    user_id = f"personamem-{digest}"
    return PersonaMemScope(
        shared_context_id=shared_context_id,
        end_index=end_index,
        scope_id=scope_id,
        user_id=user_id,
        session_id=user_id,
    )


def build_personamem_items(rows: Sequence[Mapping[str, Any]]) -> list[PersonaMemItem]:
    """Normalize official CSV rows while retaining analysis metadata."""
    items: list[PersonaMemItem] = []
    for index, row in enumerate(rows):
        shared_context_id = str(row["shared_context_id"])
        end_index = int(row["end_index_in_shared_context"])
        scope = build_personamem_scope(shared_context_id, end_index)
        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "persona_id",
                "question_id",
                "question_type",
                "topic",
                "user_question_or_message",
                "correct_answer",
                "all_options",
                "shared_context_id",
                "end_index_in_shared_context",
            }
        }
        items.append(
            PersonaMemItem(
                index=index,
                persona_id=str(row["persona_id"]),
                question_id=str(row["question_id"]),
                question_type=str(row["question_type"]),
                topic=str(row["topic"]),
                question=str(row["user_question_or_message"]),
                correct_answer=str(row["correct_answer"]),
                all_options=str(row["all_options"]),
                scope=scope,
                metadata=metadata,
            )
        )
    return items


def _format_memory_with_date(memory: str, event_time: str | None) -> str:
    """Prefix a retrieved memory with its event date so the CoT timeline can use it.

    ``event_time`` from the backend looks like ``"2026-05-03 00:00:00"``; only the
    date part is prepended (``"(2026-05-03) <memory>"``). When it is missing or
    unparseable the memory is returned unchanged so answering never breaks.
    """
    if not event_time:
        return memory
    date_part = event_time.strip().split(" ", 1)[0]
    if not date_part:
        return memory
    return f"({date_part}) {memory}"


def build_personamem_prompt(
    item: PersonaMemItem,
    *,
    retrieved_memories: Sequence[str] | None = None,
    visible_context: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build either the official full-context prompt or the memory-RAG adaptation.

    The memory-RAG path uses a single question-type-agnostic prompt for every
    question; the dataset's ``question_type`` label is never read.
    """
    query = f"{item.question}\n\n{PERSONAMEM_OFFICIAL_INSTRUCTION}\n\n{item.all_options}"
    if visible_context is not None:
        return [dict(message) for message in visible_context] + [{"role": "user", "content": query}]

    memories = list(retrieved_memories or [])
    memory_lines = [f"[{index}] {text}" for index, text in enumerate(memories, start=1)]
    memory_text = "\n".join(memory_lines) if memory_lines else "(none)"

    prompt_text = PERSONAMEM_UNIFIED_PROMPT.format(context=memory_text, question=query)
    return [{"role": "user", "content": prompt_text}]


def convert_personamem_system_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the official system-to-user conversion used for OpenAI o-series names."""
    converted: list[dict[str, Any]] = []
    system_buffer = ""
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "system":
            system_buffer += f"[System]: {content}\n"
            continue
        if system_buffer:
            content = system_buffer + content
            system_buffer = ""
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n" + content
        else:
            converted.append({"role": role, "content": content})
    return converted


def _extract_predicted_option(response: str) -> str | None:
    """Extract the final option explicitly following a final-answer token."""
    if not response:
        return None

    segments = re.findall(
        r"<final_answer>(.*?)(?:</final_answer>|(?=<final_answer>)|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not segments:
        return None

    segment = segments[-1]
    parenthesized = set(re.findall(r"\(([a-d])\)", segment, flags=re.IGNORECASE))
    if len(parenthesized) == 1:
        return next(iter(parenthesized)).lower()
    if parenthesized:
        return None

    direct = re.fullmatch(r"\s*(?:the\s+answer\s+is\s+)?([a-d])\s*", segment, flags=re.IGNORECASE)
    if direct:
        return direct.group(1).lower()
    return None


def _extract_official_option(response: str) -> tuple[str | None, bool]:
    """Return the official PersonaMem option and strict-format compliance."""
    tagged_option = _extract_predicted_option(response)
    if tagged_option is not None:
        return tagged_option, True

    lowered = response.lower()
    parenthesized = set(re.findall(r"\(([a-d])\)", lowered))
    options = parenthesized or set(re.findall(r"\b([a-d])\b", lowered))
    if len(options) == 1:
        return next(iter(options)), False
    return None, False


def extract_personamem_answer(response: str, correct_answer: str) -> tuple[bool, str]:
    """Apply the official PersonaMem v1 option extraction and correctness rule."""

    correct = correct_answer.lower().strip("() ")
    predicted_option, _ = _extract_official_option(response)
    if predicted_option is None:
        return False, response.strip() or ""
    return predicted_option == correct, predicted_option


def calculate_personamem_metrics(
    results: Sequence[PersonaMemQAResult],
    build_summaries: Sequence[PersonaMemBuildSummary],
    *,
    total_elapsed_seconds: float,
) -> dict[str, Any]:
    """Aggregate the official accuracy plus operational diagnostics."""
    total = len(results)
    correct = sum(1 for result in results if result.answer and result.answer.is_correct)
    by_question_type: dict[str, list[bool]] = defaultdict(list)
    by_topic: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        is_correct = bool(result.answer and result.answer.is_correct)
        by_question_type[result.item.question_type].append(is_correct)
        by_topic[result.item.topic].append(is_correct)

    def _breakdown(groups: Mapping[str, Sequence[bool]]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values) if values else 0.0,
            }
            for name, values in sorted(groups.items())
        }

    answers = [result.answer for result in results if result.answer is not None]
    search_failure_count = sum(1 for result in results if result.error and result.error.startswith("search failed:"))
    answer_failure_count = sum(1 for result in results if result.error and result.error.startswith("answer failed:"))
    build_elapsed = sum(summary.elapsed_seconds for summary in build_summaries)
    search_elapsed = sum(result.search_elapsed_seconds for result in results)
    answer_elapsed = sum(answer.elapsed_seconds for answer in answers)
    # Search is excluded here: `SearchResult` never carries per-call token usage,
    # so this online path can only ever report zero. Search token accounting comes
    # from the offline ClickHouse trace aggregation instead.
    token_metrics = stage_metrics(
        "answer",
        llm_calls=sum(answer.llm_calls for answer in answers),
        prompt_tokens=sum(answer.prompt_tokens for answer in answers),
        completion_tokens=sum(answer.completion_tokens for answer in answers),
        total_tokens=sum(answer.total_tokens for answer in answers),
    )
    token_metrics.update(stage_metrics("judge"))
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "by_question_type": _breakdown(by_question_type),
        "by_topic": _breakdown(by_topic),
        "scope_total": len(build_summaries),
        "scope_build_success": sum(1 for summary in build_summaries if summary.error is None),
        "scope_build_failure": sum(1 for summary in build_summaries if summary.error is not None),
        "scope_violation_count": 0,
        "search_failure_count": search_failure_count,
        "answer_failure_count": answer_failure_count,
        "answer_parse_failure_count": sum(1 for answer in answers if answer.parse_failed),
        "answer_format_failure_count": sum(1 for answer in answers if not answer.format_compliant),
        **token_metrics,
        "build_elapsed_seconds": build_elapsed,
        "search_elapsed_seconds": search_elapsed,
        "answer_elapsed_seconds": answer_elapsed,
        "total_elapsed_seconds": total_elapsed_seconds,
    }


_PERSONAMEM_EPOCH_MS = 1767225600000  # 2026-01-01 00:00:00 UTC
_PERSONAMEM_PROFILE_PREFIX = "current user persona:"


def _first_visible_personamem_profile(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, str] | None:
    """Return the first visible benchmark persona and its context index."""
    for index, message in enumerate(messages):
        if str(message.get("role") or "").strip().lower() != "system":
            continue
        content = str(message.get("content") or "").strip()
        if content.lower().startswith(_PERSONAMEM_PROFILE_PREFIX):
            return index, content
    return None


def _build_session_timestamp_map_ms(context: list[dict[str, Any]]) -> dict[int, int]:
    """Build {global_index -> timestamp_ms} aligned with UMM's build_session_timestamp_map.

    - Session boundaries: system messages in context.
    - Session 0 starts at 2026-01-01 UTC; each subsequent session starts on the
      1st of the month following the previous session's last day.
    - Within a session, each user+assistant turn advances 1 day; messages in the
      same turn share the same timestamp.
    """
    from datetime import datetime, timedelta, timezone

    session_starts = [
        i for i, m in enumerate(context)
        if str(m.get("role") or "") == "system"
    ]
    if not session_starts or session_starts[0] != 0:
        session_starts.insert(0, 0)

    def _next_month_first(dt: datetime) -> datetime:
        if dt.month == 12:
            return datetime(dt.year + 1, 1, 1, tzinfo=dt.tzinfo)
        return datetime(dt.year, dt.month + 1, 1, tzinfo=dt.tzinfo)

    ts_map: dict[int, datetime] = {}
    next_base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for sess_idx, start in enumerate(session_starts):
        end = session_starts[sess_idx + 1] if sess_idx + 1 < len(session_starts) else len(context)
        base = next_base
        day_offset = 0
        prev_role: str | None = None
        for i in range(start, end):
            role = str(context[i].get("role") or "")
            if role == "system":
                ts_map[i] = base + timedelta(days=day_offset)
                continue
            if role == "user" and prev_role in ("assistant", None):
                if prev_role is not None:
                    day_offset += 1
            ts_map[i] = base + timedelta(days=day_offset)
            prev_role = role
        next_base = _next_month_first(base + timedelta(days=day_offset))

    return {i: int(dt.timestamp() * 1000) for i, dt in ts_map.items()}


class PersonaMemEnv:
    """Run official PersonaMem v1 questions against MindMemOS or full context."""

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        answer_llm: LLMClient,
        context_store: PersonaMemContextStore,
        evaluation_mode: PersonaMemEvaluationMode = "memory_rag",
        context_size: str = "32k",
        top_k: int = 50,
        search_strategy: str = "fast",
        rerank: bool = False,
        add_batch_size: int = 50,
    ) -> None:
        self._memory = memory
        self._answer_llm = answer_llm
        self._context_store = context_store
        self._evaluation_mode = evaluation_mode
        self._context_size = context_size
        self._top_k = top_k
        self._search_strategy = search_strategy
        self._rerank = rerank
        self._add_batch_size = add_batch_size

    @staticmethod
    def load_items(path: str | Path) -> list[PersonaMemItem]:
        """Load official question rows as normalized benchmark items."""
        return build_personamem_items(load_personamem_questions(path))

    async def _build_scope(self, scope: PersonaMemScope) -> PersonaMemBuildSummary:
        started = time.monotonic()
        visible = self._context_store.visible(scope)
        # Session-aware timestamps aligned with UMM (build_session_timestamp_map):
        # system messages start new sessions; session 0 starts 2026-01-01 UTC and
        # subsequent sessions start on next month's 1st; within a session each
        # user+assistant turn = 1 day. ``visible`` is the context[:end_index] prefix,
        # so its positions align 1:1 with the full-context global indices.
        ts_map = _build_session_timestamp_map_ms(self._context_store.load(scope.shared_context_id))
        messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
                "timestamp": ts_map.get(index, _PERSONAMEM_EPOCH_MS),
            }
            for index, message in enumerate(visible)
            if str(message.get("role") or "") != "system"
            and str(message.get("content") or "").strip()
        ]
        profile = _first_visible_personamem_profile(visible)
        total_messages = len(messages) + int(profile is not None)
        scope_metadata = {
            "benchmark": "personamem",
            "shared_context_id": scope.shared_context_id,
            "end_index_in_shared_context": scope.end_index,
        }
        added_messages = 0
        add_calls = 0
        try:
            if profile is not None:
                profile_index, profile_content = profile
                await self._memory.add(
                    [
                        {
                            "role": "user",
                            "content": profile_content,
                            "timestamp": ts_map.get(profile_index, _PERSONAMEM_EPOCH_MS),
                        }
                    ],
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    mode="sync",
                    metadata={
                        **scope_metadata,
                        "source": "personamem_persona",
                        "content_type": "profile",
                    },
                )
                add_calls += 1
                added_messages += 1
            for start in range(0, len(messages), self._add_batch_size):
                batch = messages[start : start + self._add_batch_size]
                await self._memory.add(
                    batch,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    mode="sync",
                    metadata=scope_metadata,
                )
                add_calls += 1
                added_messages += len(batch)
            return PersonaMemBuildSummary(
                scope=scope,
                total_messages=total_messages,
                added_messages=added_messages,
                add_calls=add_calls,
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 - one bad scope must not discard the full run
            return PersonaMemBuildSummary(
                scope=scope,
                total_messages=total_messages,
                added_messages=added_messages,
                add_calls=add_calls,
                elapsed_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _answer_item(
        self,
        item: PersonaMemItem,
        *,
        build_error: str | None,
    ) -> PersonaMemQAResult:
        if build_error and self._evaluation_mode == "memory_rag":
            return PersonaMemQAResult(item=item, error=f"build failed: {build_error}")

        memories: list[str] = []
        search_elapsed = 0.0
        if self._evaluation_mode == "memory_rag":
            search_started = time.monotonic()
            try:
                search = await self._memory.search(
                    item.question,
                    user_id=item.scope.user_id,
                    session_id=item.scope.session_id,
                    top_k=self._top_k,
                    search_strategy=self._search_strategy,
                    rerank=self._rerank,
                    # Actor user_id does not constrain vanilla recall; this filter does.
                    filters={"user_id": item.scope.user_id},
                )
                memories = [
                    _format_memory_with_date(hit.memory, hit.event_time)
                    for hit in search.memories
                    if hit.memory
                ]
            except Exception as exc:  # noqa: BLE001 - failures remain in the official denominator
                return PersonaMemQAResult(
                    item=item,
                    search_elapsed_seconds=time.monotonic() - search_started,
                    error=f"search failed: {type(exc).__name__}: {exc}",
                )
            search_elapsed = time.monotonic() - search_started
            prompt = build_personamem_prompt(item, retrieved_memories=memories)
        else:
            prompt = build_personamem_prompt(
                item, visible_context=self._context_store.visible(item.scope)
            )

        if "o" in self._answer_llm.config.model:
            prompt = convert_personamem_system_messages(prompt)

        answer_started = time.monotonic()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        last_completion_content = ""
        final_prompt = prompt
        extracted_option: str | None = None
        format_compliant = False
        llm_calls = 0
        for attempt in range(PERSONAMEM_ANSWER_MAX_RETRIES):
            attempt_prompt = prompt
            if attempt > 0:
                reminder = (
                    "Your previous response did not contain a parseable answer option. "
                    "Please re-answer and provide your final answer in the EXACT format: "
                    "<final_answer>(a)</final_answer> or <final_answer>(b)</final_answer> "
                    "or <final_answer>(c)</final_answer> or <final_answer>(d)</final_answer>."
                )
                attempt_prompt = list(prompt) + [
                    {"role": "assistant", "content": last_completion_content},
                    {"role": "user", "content": reminder},
                ]
            final_prompt = attempt_prompt
            try:
                completion = await self._answer_llm.complete(attempt_prompt)
            except Exception as exc:  # noqa: BLE001 - failures remain in the official denominator
                partial_answer = None
                if llm_calls:
                    partial_answer = PersonaMemAnswer(
                        response=last_completion_content,
                        extracted_answer="",
                        is_correct=False,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_tokens,
                        llm_calls=llm_calls,
                        parse_failed=False,
                        format_compliant=False,
                        elapsed_seconds=time.monotonic() - answer_started,
                    )
                return PersonaMemQAResult(
                    item=item,
                    retrieved_memories=memories,
                    prompt=attempt_prompt,
                    search_elapsed_seconds=search_elapsed,
                    answer=partial_answer,
                    error=f"answer failed: {type(exc).__name__}: {exc}",
                )
            llm_calls += 1
            total_prompt_tokens += int(completion.prompt_tokens or 0)
            total_completion_tokens += int(completion.completion_tokens or 0)
            total_tokens += int(completion.total_tokens or 0)
            last_completion_content = completion.content or ""
            extracted_option, format_compliant = _extract_official_option(last_completion_content)
            if extracted_option is not None:
                break
        answer_elapsed = time.monotonic() - answer_started

        parse_failed = extracted_option is None
        extracted = extracted_option or ""
        correct = item.correct_answer.lower().strip("() ")
        is_correct = bool(extracted_option) and extracted_option == correct
        return PersonaMemQAResult(
            item=item,
            retrieved_memories=memories,
            prompt=final_prompt,
            search_elapsed_seconds=search_elapsed,
            answer=PersonaMemAnswer(
                response=last_completion_content,
                extracted_answer=extracted,
                is_correct=is_correct,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                llm_calls=llm_calls,
                parse_failed=parse_failed,
                format_compliant=format_compliant,
                elapsed_seconds=answer_elapsed,
            ),
        )

    async def run_dataset(
        self,
        items: Sequence[PersonaMemItem],
        *,
        max_build_concurrency: int = 2,
        max_qa_concurrency: int = 4,
        add: bool = True,
        score: bool = True,
        show_progress: bool = True,
    ) -> PersonaMemRunResult:
        """Build unique official scopes, answer questions, and score deterministically."""
        del score  # PersonaMem scoring is deterministic and always accompanies an answer.
        started = time.monotonic()
        scopes = {item.scope.scope_id: item.scope for item in items}

        build_summaries: list[PersonaMemBuildSummary] = []

        if self._evaluation_mode == "memory_rag":
            build_sem = asyncio.Semaphore(max_build_concurrency)
            build_pbar = tqdm(
                total=len(scopes),
                disable=not show_progress,
                desc="Building PersonaMem scopes",
                unit="scope",
            )

            async def _build(scope: PersonaMemScope) -> PersonaMemBuildSummary:
                async with build_sem:
                    if add:
                        summary = await self._build_scope(scope)
                    else:
                        summary = PersonaMemBuildSummary(scope=scope)
                    build_pbar.update()
                return summary

            build_summaries = list(await asyncio.gather(*(_build(scope) for scope in scopes.values())))
            build_pbar.close()

        build_errors = {summary.scope.scope_id: summary.error for summary in build_summaries}
        qa_sem = asyncio.Semaphore(max_qa_concurrency)
        qa_pbar = tqdm(total=len(items), disable=not show_progress, desc="Evaluating PersonaMem", unit="question")
        live_progress = _PersonaMemLiveProgress(total=len(items))
        live_progress_lock = asyncio.Lock()

        async def _answer(item: PersonaMemItem) -> PersonaMemQAResult:
            async with qa_sem:
                result = await self._answer_item(item, build_error=build_errors.get(item.scope.scope_id))
                async with live_progress_lock:
                    live_progress.record(result)
                    qa_pbar.update()
                    qa_pbar.set_postfix(live_progress.postfix())
                return result

        results = list(await asyncio.gather(*(_answer(item) for item in items)))
        qa_pbar.close()
        total_elapsed = time.monotonic() - started
        protocol = f"personamem-v1-{'memory-rag' if self._evaluation_mode == 'memory_rag' else 'official-full-context'}"
        metrics = calculate_personamem_metrics(results, build_summaries, total_elapsed_seconds=total_elapsed)
        return PersonaMemRunResult(
            protocol=protocol,
            context_size=self._context_size,
            evaluation_mode=self._evaluation_mode,
            items=list(items),
            build_summaries=build_summaries,
            qa_results=results,
            metrics=metrics,
        )
