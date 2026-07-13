"""Official PersonaMem v1 protocol adapted to the MindMemOS memory API."""

from __future__ import annotations

import asyncio
import csv
import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from mindmemos_sdk.memory import AsyncMemoryClient
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from mindmemos_eval.llm import LLMClient

PERSONAMEM_OFFICIAL_REPOSITORY = "https://github.com/bowen-upenn/PersonaMem"
PERSONAMEM_OFFICIAL_PROTOCOL_COMMIT = "caaae44b3f236b8751d499a770e94e5aecffcff1"
PERSONAMEM_OFFICIAL_INSTRUCTION = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)
PERSONAMEM_ANSWER_MAX_RETRIES = 3
PERSONAMEM_ANSWER_OPTIONS = ("a", "b", "c", "d")
PERSONAMEM_COT_PROMPT = """You are an intelligent memory assistant. Your task is to select the most appropriate response to the user based on memories.

# CONTEXT:
You have access to structured temporal memories from conversations that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to select the correct answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection.

# CRITICAL REQUIREMENTS:
1. Answer based ONLY on evidence in the memories. Never guess or use general knowledge.
2. **Memory recall trumps generic validation.** Prefer options that demonstrate recall of stored memories ("I remember you mentioned…", referencing specific details from past conversations) over options that merely acknowledge what the user just said ("That's great!", "Sounds like you're exploring!"). Apply this bright-line test: Does the option reference ANY specific detail not already present in the user's current message? If NO → it is generic. When a memory-recall option exists alongside a generic one, strongly prefer the memory-recall option.
3. **Faithfully preserve attitude polarity in BOTH directions.** If memories record that the user disliked, abandoned, or was discouraged by something, options reframing this as positive are WRONG. Equally: if memories record that the user liked or found something beneficial, options reframing this as negative are WRONG. When truly uncertain, mark as "unclear" rather than picking a direction.
4. **Distinguish the user's personal reaction from environmental descriptions.** "The atmosphere was vibrant" describes the setting. "Felt overwhelmed and discouraged" describes the user's reaction. Only the user's personal reaction determines attitude polarity.
5. **Intensity fidelity mapping**: Memory language maps to specific intensity levels. Do NOT inflate or deflate:
   - "didn't resonate" / "wasn't drawn to" = mild disinterest ≠ "stopped engaging entirely" (complete cessation)
   - "disliked" / "found it a chore" = active negative ≠ "hesitated" / "was unsure" (mere uncertainty)
   - "found it beneficial" / "enjoyed" = positive ≠ "was life-changing" / "transformed"
   - "overwhelming" = negative-strong ≠ "challenging" (neutral/mild)
   - "renewed interest" / "rekindled" = return to positive ≠ "became obsessed" (extreme)
   Match the option whose intensity CLOSEST mirrors the memory's own words.
6. **Verbatim anchoring for claim verification.** When checking whether an option is supported by memory, locate the EXACT noun, verb, or phrase from the option in the memory text. "Sounds like something the user would like" is NEVER sufficient. If the option says "competitive rankings" you must find "competitive rankings" or a near-exact synonym in memory. Topical relevance alone is not verification.

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: QUESTION ANALYSIS
- Question scenario: [select exactly one:
  `preference evolution` | `trying a new activity` | `novelty suggestion` | `fact-or-reason query` | `recommendation` | `generalization`

  Classification rules — apply in order, first match wins:
  1. If options describe multi-stage sequences ("initially X → then Y → now Z") OR the question asks how a preference changed over time → `preference evolution`
  2. If the question asks "should I try X?" — HOLD classification until after reading memories in Step 2. If memories contain structurally analogous past experiences (similar behavioral tensions: structure-vs-freedom, solo-vs-social, competitive-vs-relaxed, large-scale-vs-intimate) → `generalization`. Otherwise → `trying a new activity`
  3. If the question **explicitly** asks for something the user has NOT experienced ("haven't tried", "brand new", "recommend something different") → `novelty suggestion`
  4. If the question asks about a specific fact, reason for a change, or how the user felt → `fact-or-reason query`
  5. If the question asks for a recommendation, suggestion, or ideas → `recommendation`
  ]
- What the question is really asking: [one-sentence paraphrase — preserve full semantic units, not keyword fragments]

## STEP 2: RELEVANT MEMORIES EXTRACTION WITH VERBATIM ANCHORING

**2A. Activities Inventory** (ONLY for "trying a new activity", "novelty suggestion", or "generalization" scenarios — skip otherwise):
- Activity: [name] | Reaction: [exact words from memory] | Date: [date] | Memory index: [N]

**2B. Memory Extraction:**
For EACH relevant memory, extract the user's personal reaction by **copying the memory's exact words verbatim**. Do NOT paraphrase.
- Memory [N] (Date: [date]): [content summary]
  - Verbatim quote: "[exact words from memory]"
  - Attitude: [positive / negative / mixed / unclear]
  - Polarity proof: [quote the specific word(s) — e.g., "beneficial" = positive, "tedious" = negative]
  - Environmental vs. personal: [note if relevant]

**2C. Generalization Classification Check** (ONLY if classification was held in Step 1):
Check for structural analogies across memories:
- Structure vs. freedom, solo vs. social, competitive vs. relaxed, large-scale vs. intimate, routine vs. novelty, teamwork/collaboration patterns
If found → classify as `generalization`. Otherwise → `trying a new activity`.

## STEP 3: CROSS-MEMORY LINKING & TEMPORAL TRACKING
[For simple fact-or-reason queries with a single relevant memory, skip to STEP 4.]

**Date-sorted timeline table:**
| # | Date | Event/Activity | Verbatim quote of user's attitude | Polarity |
|---|------|---------------|-----------------------------------|----------|
| 1 | [EARLIEST] | ... | "..." | pos/neg/mixed |
| 2 | [next] | ... | "..." | pos/neg/mixed |

**INITIAL ATTITUDE ANCHOR** (for preference evolution): Read ONLY Row 1. Write: "Initial attitude toward [topic] was: '[verbatim quote from Row 1]' → [polarity]."
Do NOT back-project later experiences onto Row 1.

**Phase count**: Total documented polarity phases = [N].

**Trajectory shape**: List ALL phases from the table in order. Rules:
- Do NOT collapse phases (e.g., "disliked → tried again → liked" must NOT become "disliked → liked")
- Do NOT soften language ("disliked" must NOT become "hesitated")
- Do NOT inflate language ("didn't resonate as much" must NOT become "stopped engaging entirely")
- If a phase is unclear, write "unclear" — do NOT guess a direction.

**For generalization scenarios — ABSTRACT PATTERN EXTRACTION** (mandatory):
Synthesize across memories: "The user tends to [abstract pattern] — evidence: [cite 2+ memories]."
Then: "The new scenario involves [tension type]. Past evidence shows the user [reacted how] to the same tension in [different domain]."

## STEP 4: SCENARIO-SPECIFIC OPTION ANALYSIS

**CLAIM VERIFICATION TABLE (mandatory for ALL options):**
For each option, extract its specific factual claim(s) and verify against memory:

| Option | Specific claim (exact words) | Found in memory? (quote verbatim or "NOT FOUND") | Match quality |
|--------|------------------------------|--------------------------------------------------|---------------|
| (a) | "[claim]" | "[memory quote]" or "NOT FOUND" | Exact / Synonym / Different concept / NOT FOUND |
| (b) | ... | ... | ... |
| ... | ... | ... | ... |

Match quality: Exact > Synonym > Different concept = NOT FOUND (both = no support).

**Then apply scenario-specific logic:**

**Preference evolution:**
1. **Initial attitude gate**: Compare each option's stated initial attitude against your INITIAL ATTITUDE ANCHOR. Any polarity mismatch → ELIMINATE.
2. **Phase completeness**: If an option has fewer phases than your timeline → ELIMINATE (it omits documented phases).
3. **Intensity fidelity**: Compare option language against memory language using the intensity mapping in Critical Requirement 5.
4. **Temporal order**: Phases must appear in correct chronological order.

**Trying a new activity:**
- Verify cited past experiences exist in STEP 2A and are genuinely relevant.
- Priority: (1) aligned with stated preference domain and values; (2) concrete reasoning from real past experience; (3) does not force-fit unrelated activities.

**Novelty suggestion:**
- If explicitly requiring something unexperienced: exclude options describing activities in STEP 2A.
- Among genuinely new options, pick the best match to demonstrated interests.

**Fact-or-reason query:**
- Select the option with the strongest verbatim match from the claim verification table.
- Distinguish "facts the user stated in this question" from "stored facts the user did NOT state this time." The correct option surfaces the latter.
- Verify attitude polarity matches the memory.

**Recommendation:**
- Extract the user's top 2–3 core sub-interests from memories BEFORE evaluating options. Be specific:
  - NOT "likes books" → YES "interested in attachment theory and psychological aspects of love"
  - NOT "likes cooking" → YES "values cultural exchange through food and family bonding"
- **Surface keyword warning**: The question's framing words ("retreat," "cultural," "creative") are scene-setting. Score options against your extracted sub-interests, not the question's adjectives.
- Verify specific claims in options via the claim verification table.

**Generalization:**
- Score each option against the ABSTRACT BEHAVIORAL PATTERN and STRUCTURAL ANALOGY from Step 3.
- The correct option applies the user's documented behavioral pattern to the new scenario via structural analogy.
- Options offering generic encouragement or balanced hedging without referencing the user's documented patterns are usually wrong.

## STEP 5: FINAL SELECTION

For each option, assign: **ELIMINATE** (hard evidence against), **WEAK**, **MODERATE**, or **STRONG**.

**Final verification:**
1. For each specific claim in my chosen option: can I point to EXACT words in a memory? [cite them]
2. Does my chosen option's attitude polarity match the memory's documented attitude?
3. In preference evolution: does my chosen option's initial polarity match my INITIAL ATTITUDE ANCHOR? [compare side by side]
4. In generalization: does my chosen option reference the abstract behavioral pattern, or does it give generic advice?
5. Memory-recall accuracy check: If the chosen option claims to recall something ("I remember you mentioned X"), verify X matches memory with correct polarity and detail.

Final choice: [option] — because [one sentence citing specific memory evidence with a verbatim quote]

Provide your final answer:
<final_answer>(a)</final_answer> or <final_answer>(b)</final_answer> or <final_answer>(c)</final_answer> or <final_answer>(d)</final_answer>

---

## Reference Memories
{context}

## Question
{question}

---

Now, follow the Chain-of-Thought process above to answer the question:
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
    """Build outcome for one incremental context segment [start_index, end_index)."""

    scope: PersonaMemScope
    start_index: int = 0
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
    elapsed_seconds: float = 0.0


class PersonaMemQAResult(BaseModel):
    """End-to-end result for one PersonaMem question."""

    item: PersonaMemItem
    retrieved_memories: list[str] = Field(default_factory=list)
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    search_elapsed_seconds: float = 0.0
    answer: PersonaMemAnswer | None = None
    error: str | None = None


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


def build_personamem_scope(shared_context_id: str, end_index: int, persona_id: str) -> PersonaMemScope:
    """Create an isolated memory scope keyed by shared_context.

    Both ``user_id`` and ``session_id`` are derived from
    ``shared_context_id`` so that each of the 37 shared contexts is a
    fully independent memory scope.  Previously ``user_id`` was derived
    from ``persona_id`` (only 20 unique values), which caused the two
    contexts of the same persona to share a single user-level memory
    store — the shared-prefix messages were ingested twice and entity
    modelling merged facts from both contexts.
    """
    scope_id = f"{shared_context_id}:{end_index}"
    return PersonaMemScope(
        shared_context_id=shared_context_id,
        end_index=end_index,
        scope_id=scope_id,
        user_id=f"personamem-{shared_context_id}",
        session_id=f"personamem-{shared_context_id}",
    )


def build_personamem_items(rows: Sequence[Mapping[str, Any]]) -> list[PersonaMemItem]:
    """Normalize official CSV rows while retaining analysis metadata."""
    items: list[PersonaMemItem] = []
    for index, row in enumerate(rows):
        shared_context_id = str(row["shared_context_id"])
        end_index = int(row["end_index_in_shared_context"])
        scope = build_personamem_scope(shared_context_id, end_index, str(row["persona_id"]))
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


def build_personamem_prompt(
    item: PersonaMemItem,
    *,
    retrieved_memories: Sequence[str] | None = None,
    visible_context: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build either the official full-context prompt or the memory-RAG adaptation."""
    query = f"{item.question}\n\n{PERSONAMEM_OFFICIAL_INSTRUCTION}\n\n{item.all_options}"
    if visible_context is not None:
        return [dict(message) for message in visible_context] + [{"role": "user", "content": query}]

    memories = list(retrieved_memories or [])
    memory_lines = [f"[{index}] {text}" for index, text in enumerate(memories, start=1)]
    memory_text = "\n".join(memory_lines) if memory_lines else "(none)"
    prompt_text = PERSONAMEM_COT_PROMPT.format(context=memory_text, question=query)
    return [
        {"role": "user", "content": prompt_text},
    ]


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
    """Extract a single predicted option (a/b/c/d) from response, or None if not found."""
    if not response:
        return None
    lowered = response.lower()

    # Pattern 1: <final_answer>(X)</final_answer> (strict, requires closing tag)
    fa_match = re.search(r"<final_answer>\s*\(?([a-d])\)?\s*</final_answer>", lowered)
    if fa_match:
        return fa_match.group(1)

    # Pattern 2: content after <final_answer> token
    if "<final_answer>" in lowered:
        after = lowered.split("<final_answer>")[-1]
        opts = re.findall(r"\(([a-d])\)", after)
        if not opts:
            opts = re.findall(r"\b([a-d])\b", after)
        if opts:
            return opts[-1]

    # Pattern 3: content before <final_answer> token (LLM put option before token)
    if "<final_answer>" in lowered:
        before = lowered.split("<final_answer>")[0]
        opts = re.findall(r"\(([a-d])\)", before)
        if opts:
            return opts[-1]
        opts = re.findall(r"\b([a-d])\b", before)
        if opts:
            return opts[-1]

    # Pattern 4: no <final_answer> token, scan whole response, take last option mention
    opts = re.findall(r"\(([a-d])\)", lowered)
    if opts:
        return opts[-1]
    opts = re.findall(r"\b([a-d])\b", lowered)
    if opts:
        return opts[-1]

    return None


def extract_personamem_answer(response: str, correct_answer: str) -> tuple[bool, str]:
    """Apply the official PersonaMem v1 option extraction and correctness rule."""

    correct = correct_answer.lower().strip("() ")
    predicted_option = _extract_predicted_option(response)
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
        "answer_llm_calls": len(answers),
        "answer_prompt_tokens": sum(answer.prompt_tokens for answer in answers),
        "answer_completion_tokens": sum(answer.completion_tokens for answer in answers),
        "answer_total_tokens": sum(answer.total_tokens for answer in answers),
        "build_elapsed_seconds": build_elapsed,
        "search_elapsed_seconds": search_elapsed,
        "answer_elapsed_seconds": answer_elapsed,
        "total_elapsed_seconds": total_elapsed_seconds,
    }


_PERSONAMEM_EPOCH_MS = 1767225600000  # 2026-01-01 00:00:00 UTC


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
        add_batch_size: int = 20,
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

    async def _build_scope_segment(
        self,
        scope: PersonaMemScope,
        *,
        context: list[dict[str, Any]],
        start_index: int,
    ) -> PersonaMemBuildSummary:
        """Add context[start_index:scope.end_index) to this scope's memory store."""
        started = time.monotonic()
        segment = context[start_index:scope.end_index]
        # Session-aware timestamps aligned with UMM (inference_mem.build_session_timestamp_map):
        # - system messages start new sessions
        # - session 0 starts 2026-01-01 UTC, subsequent sessions start on next month's 1st
        # - within a session, each user+assistant turn = 1 day (same turn shares a timestamp)
        ts_map = _build_session_timestamp_map_ms(context)

        messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
                "timestamp": ts_map.get(start_index + index, _PERSONAMEM_EPOCH_MS),
            }
            for index, message in enumerate(segment)
            if str(message.get("content") or "").strip()
        ]
        added_messages = 0
        add_calls = 0
        try:
            for start in range(0, len(messages), self._add_batch_size):
                batch = messages[start : start + self._add_batch_size]
                await self._memory.add(
                    batch,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    mode="sync",
                    metadata={
                        "benchmark": "personamem",
                        "shared_context_id": scope.shared_context_id,
                        "end_index_in_shared_context": scope.end_index,
                        "start_index_in_shared_context": start_index,
                    },
                )
                add_calls += 1
                added_messages += len(batch)
            return PersonaMemBuildSummary(
                scope=scope,
                start_index=start_index,
                total_messages=len(messages),
                added_messages=added_messages,
                add_calls=add_calls,
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 - one bad scope must not discard the full run
            return PersonaMemBuildSummary(
                scope=scope,
                start_index=start_index,
                total_messages=len(messages),
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
                    filters={"user_id": item.scope.user_id},
                )
                memories = [hit.memory for hit in search.memories if hit.memory]
            except Exception as exc:  # noqa: BLE001 - failures remain in the official denominator
                return PersonaMemQAResult(
                    item=item,
                    search_elapsed_seconds=time.monotonic() - search_started,
                    error=f"search failed: {type(exc).__name__}: {exc}",
                )
            search_elapsed = time.monotonic() - search_started
            prompt = build_personamem_prompt(item, retrieved_memories=memories)
        else:
            prompt = build_personamem_prompt(item, visible_context=self._context_store.visible(item.scope))

        if "o" in self._answer_llm.config.model:
            prompt = convert_personamem_system_messages(prompt)

        answer_started = time.monotonic()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        last_completion_content = ""
        extracted_option: str | None = None
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
            try:
                completion = await self._answer_llm.complete(attempt_prompt)
            except Exception as exc:  # noqa: BLE001 - failures remain in the official denominator
                return PersonaMemQAResult(
                    item=item,
                    retrieved_memories=memories,
                    prompt=attempt_prompt,
                    search_elapsed_seconds=search_elapsed,
                    error=f"answer failed: {type(exc).__name__}: {exc}",
                )
            total_prompt_tokens += int(completion.prompt_tokens or 0)
            total_completion_tokens += int(completion.completion_tokens or 0)
            total_tokens += int(completion.total_tokens or 0)
            last_completion_content = completion.content or ""
            extracted_option = _extract_predicted_option(last_completion_content)
            if extracted_option is not None:
                break
        answer_elapsed = time.monotonic() - answer_started

        # All retries exhausted without a parseable answer: random guess
        if extracted_option is None:
            extracted_option = random.choice(PERSONAMEM_ANSWER_OPTIONS)

        correct = item.correct_answer.lower().strip("() ")
        is_correct = extracted_option == correct
        return PersonaMemQAResult(
            item=item,
            retrieved_memories=memories,
            prompt=prompt,
            search_elapsed_seconds=search_elapsed,
            answer=PersonaMemAnswer(
                response=last_completion_content,
                extracted_answer=extracted_option,
                is_correct=is_correct,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
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
        """Interleave incremental context builds with boundary-scoped answering.

        Questions are grouped by ``shared_context_id``. Each shared context is
        an independent memory scope (isolated by ``session_id``): a question in
        one context cannot retrieve memories from another context, even for
        the same persona. Within each context, boundaries are processed in
        ascending ``end_index`` order: the segment up to each boundary is
        added first, then the boundary's questions are answered, and only
        then are later messages added. Each question therefore sees exactly
        the officially visible prefix ``[0, end_index)`` of its own context.
        """
        del score  # PersonaMem scoring is deterministic and always accompanies an answer.
        started = time.monotonic()

        # Group by shared_context_id -> end_index.
        by_context: dict[str, dict[int, list[PersonaMemItem]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for item in items:
            by_context[item.scope.shared_context_id][item.scope.end_index].append(item)

        build_summaries: list[PersonaMemBuildSummary] = []
        results: list[PersonaMemQAResult] = []

        if self._evaluation_mode == "memory_rag":
            segment_total = sum(len(boundaries) for boundaries in by_context.values())
            context_sem = asyncio.Semaphore(max_build_concurrency)
            qa_sem = asyncio.Semaphore(max_qa_concurrency)
            build_pbar = tqdm(
                total=segment_total, disable=not show_progress, desc="Building PersonaMem segments", unit="segment"
            )
            qa_pbar = tqdm(total=len(items), disable=not show_progress, desc="Evaluating PersonaMem", unit="question")

            async def _answer(item: PersonaMemItem, build_error: str | None) -> PersonaMemQAResult:
                async with qa_sem:
                    result = await self._answer_item(item, build_error=build_error)
                    qa_pbar.update()
                    return result

            async def _run_context(
                ctx_id: str,
                boundaries: dict[int, list[PersonaMemItem]],
            ) -> tuple[list[PersonaMemBuildSummary], list[PersonaMemQAResult]]:
                async with context_sem:
                    context = self._context_store.load(ctx_id)
                    summaries: list[PersonaMemBuildSummary] = []
                    ctx_results: list[PersonaMemQAResult] = []
                    build_error: str | None = None
                    prev_end = 0
                    for end_index in sorted(boundaries.keys()):
                        boundary_items = boundaries[end_index]
                        scope = boundary_items[0].scope
                        if add and build_error is None:
                            summary = await self._build_scope_segment(
                                scope, context=context, start_index=prev_end
                            )
                            if summary.error is not None:
                                build_error = summary.error
                        else:
                            summary = PersonaMemBuildSummary(
                                scope=scope, start_index=prev_end, error=build_error
                            )
                        summaries.append(summary)
                        build_pbar.update()
                        prev_end = end_index
                        ctx_results.extend(
                            await asyncio.gather(*(_answer(item, build_error) for item in boundary_items))
                        )
                    return summaries, ctx_results

            context_outputs = await asyncio.gather(
                *(_run_context(ctx_id, boundaries) for ctx_id, boundaries in by_context.items())
            )
            build_pbar.close()
            qa_pbar.close()
            for summaries, ctx_results in context_outputs:
                build_summaries.extend(summaries)
                results.extend(ctx_results)
        else:
            qa_sem = asyncio.Semaphore(max_qa_concurrency)
            qa_pbar = tqdm(total=len(items), disable=not show_progress, desc="Evaluating PersonaMem", unit="question")

            async def _answer_full(item: PersonaMemItem) -> PersonaMemQAResult:
                async with qa_sem:
                    result = await self._answer_item(item, build_error=None)
                    qa_pbar.update()
                    return result

            results = list(await asyncio.gather(*(_answer_full(item) for item in items)))
            qa_pbar.close()

        results.sort(key=lambda result: result.item.index)
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
