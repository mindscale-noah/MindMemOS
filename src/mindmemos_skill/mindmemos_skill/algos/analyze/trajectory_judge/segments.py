"""Per-skill trajectory segmentation for LLM failure attribution.

Adapted from the reference implementation
``skill_and_trajectory_evaluation/trajectory_judge.py`` (``_skill_segments`` /
``_render_segments``): split one trajectory into labeled skill segments so the
attribution LLM can assign per-segment suspicion and pick a primary cause.
The reference split ATIF-v1.7 steps on ``Skill`` tool calls; here the event
dicts come from the framework's own agent runtimes, so invocation detection
covers the structured evidence each runtime records (explicit Skill tool calls
for Claude/ReAct/DeepSeek, OpenClaw SKILL.md content access) instead of
porting the reference's prefix-string injection sniffing.

Pure functions only: no I/O, no LLM calls, unit-testable without fakes.
Deliberately does not import from ``agents/*`` to keep the algo layer decoupled;
the few local regexes mirror the runtime-side detectors and cite their source.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from ..typing import TrajectoryAnalysisInput


@dataclass(frozen=True, slots=True)
class SkillInvocation:
    """One detected skill-invocation boundary in ``trajectory.events``."""

    index: int
    """0-based index into ``trajectory.events``."""

    name: str
    """Canonical skill display name (resolved against ``injected_skills``)."""


@dataclass(frozen=True, slots=True)
class SkillSegment:
    """One labeled span of events handed to the attribution LLM."""

    label: str
    """LLM-facing unique label: a skill name, ``<no-skill>/<pre>`` or ``name#2``."""

    name: str | None
    """Canonical skill name; ``None`` for no-skill segments."""

    start: int
    """0-based inclusive first event index."""

    end: int
    """0-based exclusive last event index."""

    tokens: dict[str, int]
    """``{"input", "output", "cached", "total"}`` summed over ``[start, end)``."""


NO_SKILL_LABEL = "<no-skill>"
NO_SKILL_PRE_LABEL = "<no-skill>/<pre>"

# Mirror of agents/openclaw/skill_runtime.py _TOOL_CALL / _SKILL_MD.
_OPENCLAW_TOOL_CALL_RE = re.compile(r"^\s*\[tool_call\]\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$", re.DOTALL)
_OPENCLAW_SKILL_MD_RE = re.compile(r"/skills/([^/\"'\s\\]+)/SKILL\.md")


def _canonical_name(name: str, injected: dict[str, str]) -> str:
    """Resolve a detected name against injected skill names (case-insensitive)."""

    return injected.get(name.strip().lower(), name)


def _injected_name_map(trajectory: TrajectoryAnalysisInput) -> dict[str, str]:
    """Lowercased skill name -> display name from ``trajectory.injected_skills``."""

    return {skill.name.strip().lower(): skill.name for skill in trajectory.injected_skills}


def _skill_tool_call_names(event: dict[str, Any]) -> list[str]:
    """All skill names invoked by explicit Skill tool calls in one event
    (Claude ``Skill``/``arguments.skill`` or ReAct/DeepSeek ``skill``/
    ``arguments.name``); empty when the event invokes no skill.  Parallel
    tool calls in one assistant message each yield a boundary."""

    if event.get("role") != "assistant":
        return []
    names: list[str] = []
    for tool_call in event.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict) or str(function.get("name", "")).lower() != "skill":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        name = arguments.get("skill") or arguments.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _openclaw_marker_name(event: dict[str, Any], injected: dict[str, str]) -> str | None:
    """Skill name from an OpenClaw ``[tool_call] read(...skills/<dir>/SKILL.md)``
    content tool call; ``None`` otherwise."""

    if event.get("role") != "assistant":
        return None
    match = _OPENCLAW_TOOL_CALL_RE.match(str(event.get("content") or ""))
    if match is None or match.group(1).lower() not in {"read", "write", "edit"}:
        return None
    path_match = _OPENCLAW_SKILL_MD_RE.search(match.group(2))
    if path_match is None:
        return None
    return _canonical_name(path_match.group(1), injected)


def detect_skill_invocations(trajectory: TrajectoryAnalysisInput) -> list[SkillInvocation]:
    """Detect skill-invocation boundaries across runtime event formats.

    Covers explicit Skill tool calls (Claude ``Skill``/ReAct/DeepSeek ``skill``,
    including parallel calls within one assistant message) and OpenClaw
    ``[tool_call] read(...skills/<dir>/SKILL.md)`` content access.  ReAct's
    ``Loaded skill '<name>'.`` confirmation user-message is deliberately not a
    boundary — the assistant tool call already claims that index.
    """

    injected = _injected_name_map(trajectory)
    claimed: dict[int, list[str]] = {}
    for index, event in enumerate(trajectory.events):
        names = _skill_tool_call_names(event)
        if not names:
            openclaw_name = _openclaw_marker_name(event, injected)
            if openclaw_name:
                names = [openclaw_name]
        for name in names:
            claimed.setdefault(index, []).append(_canonical_name(name, injected))
    return [SkillInvocation(index=index, name=name) for index, names in sorted(claimed.items()) for name in names]


def _merge_consecutive_same_name(invocations: list[SkillInvocation]) -> list[SkillInvocation]:
    """Collapse runs of consecutive same-name invocations into one boundary."""

    merged: list[SkillInvocation] = []
    for invocation in invocations:
        if merged and merged[-1].name == invocation.name and merged[-1].index != invocation.index:
            continue  # consecutive same-skill re-invocation: keep the earlier boundary
        merged.append(invocation)
    return merged


def _unique_label(name: str, taken: set[str]) -> str:
    """Return ``name`` or ``name#2``/``name#3``... so labels stay unique."""

    if name not in taken:
        return name
    occurrence = 2
    while f"{name}#{occurrence}" in taken:
        occurrence += 1
    return f"{name}#{occurrence}"


def event_tokens(event: dict[str, Any]) -> dict[str, int]:
    """Per-event token bucket extracted from recorded usage metrics."""

    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        return {"input": 0, "output": 0, "cached": 0}
    input_tokens = int(metrics.get("prompt_tokens", metrics.get("prompt", 0)) or 0)
    output_tokens = int(metrics.get("completion_tokens", metrics.get("completion", 0)) or 0)
    cached_tokens = int(metrics.get("cached_tokens", metrics.get("cached", 0)) or 0)
    return {"input": input_tokens, "output": output_tokens, "cached": cached_tokens}


def _empty_tokens() -> dict[str, int]:
    return {"input": 0, "output": 0, "cached": 0, "total": 0}


def _span_tokens(events: list[dict[str, Any]], start: int, end: int) -> dict[str, int]:
    tokens = _empty_tokens()
    for event in events[start:end]:
        bucket = event_tokens(event)
        for key in ("input", "output", "cached"):
            tokens[key] += bucket[key]
        tokens["total"] += bucket["input"] + bucket["output"]
    return tokens


def _segment(events: list[dict[str, Any]], label: str, name: str | None, start: int, end: int) -> SkillSegment | None:
    if end <= start:
        return None
    return SkillSegment(label=label, name=name, start=start, end=end, tokens=_span_tokens(events, start, end))


def build_skill_segments(trajectory: TrajectoryAnalysisInput, invocations: list[SkillInvocation]) -> list[SkillSegment]:
    """Assemble labeled segments from invocation boundaries.

    Ported from the reference ``_skill_segments`` with three deliberate
    differences: consecutive same-name invocations merge into one span; same-index
    boundaries (parallel tool calls, SYSTEM_PROMPT skills) each get the shared
    span ``[index, next-distinct-index-or-end)`` so every skill stays attributable;
    non-consecutive duplicate names get ``name#2``-style suffixes so labels stay
    unique.  With no invocations the whole trajectory is one ``<no-skill>``
    segment; the span before the first boundary is ``<no-skill>/<pre>``; the last
    skill segment extends to the end (no ``<no-skill>/<post>``).
    """

    events = trajectory.events
    bounds = _merge_consecutive_same_name(invocations)
    if not bounds:
        segment = _segment(events, NO_SKILL_LABEL, None, 0, len(events))
        return [segment] if segment else []

    segments: list[SkillSegment] = []
    taken: set[str] = set()
    if bounds[0].index > 0:
        pre = _segment(events, NO_SKILL_PRE_LABEL, None, 0, bounds[0].index)
        if pre is not None:
            segments.append(pre)
            taken.add(pre.label)
    for position, invocation in enumerate(bounds):
        end = next((b.index for b in bounds[position + 1 :] if b.index != invocation.index), len(events))
        segment = _segment(events, _unique_label(invocation.name, taken), invocation.name, invocation.index, end)
        if segment is not None:
            segments.append(segment)
            taken.add(segment.label)
    return segments


def truncate_text(text: str, cap: int) -> str:
    """Head/tail truncation around a marker (from ``TrajectoryJudge._truncate``)."""

    text = text.strip()
    if len(text) <= cap:
        return text
    half = max(1, (cap - 40) // 2)
    return f"{text[:half]}\n[… event truncated …]\n{text[-half:]}"


def render_event_text(event: dict[str, Any], max_chars: int) -> str:
    """Render one event dict into compact LLM transcript text.

    Same branch order as the original ``TrajectoryJudge._event_text``, plus a
    ``tool_calls`` branch so Claude/ReAct skill invocations no longer fall
    through to raw JSON dumps.
    """

    if "tool_calls" in event and event.get("tool_calls"):
        parts = []
        role = event.get("role", "assistant")
        content = str(event.get("content") or "").strip()
        if content:
            parts.append(f"[{role}] {content}")
        for tool_call in event["tool_calls"]:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = function.get("name", "?")
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            parts.append(f"tool: {name}({arguments})")
        text = "\n".join(parts)
    elif "role" in event and "content" in event:
        text = f"[{event.get('role')}] {event.get('content')}"
    elif "message" in event:
        text = f"[message] {event.get('message')}"
    elif "tool" in event or "tool_name" in event:
        text = f"[tool] {event.get('tool') or event.get('tool_name')}: {event}"
    else:
        text = json.dumps(event, ensure_ascii=False, sort_keys=True)
    return truncate_text(str(text), max_chars)


def render_attribution_trace(
    trajectory: TrajectoryAnalysisInput,
    segments: list[SkillSegment],
    *,
    max_chars: int,
    render_event: Callable[[dict[str, Any]], str],
) -> str:
    """Render the task plus per-segment labeled blocks for the attribution LLM.

    Same-span segments (parallel tool calls, SYSTEM_PROMPT skills) share their
    body: only the first span occurrence renders the events, later ones get a
    one-line note.  The assembled text is hard-truncated at ``max_chars`` like
    the reference ``_render_segments``.
    """

    blocks: list[str] = []
    rendered_spans: dict[tuple[int, int], str] = {}
    for segment in segments:
        span = (segment.start, segment.end)
        first_label = rendered_spans.get(span)
        if first_label is not None:
            body = f"(same events as segment '{first_label}' above)"
        else:
            body = (
                "\n".join(render_event(event) for event in trajectory.events[segment.start : segment.end])
                or "[no events recorded]"
            )
            rendered_spans[span] = segment.label
        blocks.append(f"--- SKILL SEGMENT: {segment.label} (events {segment.start + 1}-{segment.end}) ---\n{body}")
    text = f"=== TASK ===\n{trajectory.task_instruction}\n=== END TASK ===\n\n" + "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... trace truncated to char limit ...]"
    return text


def _strip_label_decorations(candidate: str) -> list[str]:
    """Candidate forms of a model-returned label, most-to-least literal.

    Models sometimes copy the rendered segment header along with the label
    (``name (events 3-7)``, ``name [traj-id]``) or add prose.  Generate
    progressively stripped variants: raw, drop a trailing bracket/paren
    suffix, then drop everything from the first bracket/paren onward.
    """

    forms = [candidate]
    trimmed = re.sub(r"\s*[\[(][^\])]*[\])]$", "", candidate).strip()
    if trimmed:
        forms.append(trimmed)
    split = re.split(r"[\[(]", candidate, maxsplit=1)[0].strip()
    if split:
        forms.append(split)
    return forms


def normalize_segment_label(raw: str, labels: Collection[str]) -> str:
    """Map a model-returned label onto an allowed segment label.

    Resolution ladder: exact match, case-insensitive match, then both retried
    over progressively stripped variants (trailing ``[...]``/``(...)`` suffixes
    such as the legacy ``name [trajectory_id]`` pollution or a copied
    ``(events 3-7)`` header fragment).  Raises ``ValueError`` for unknown
    labels so ``chat``'s ``format_parser`` feedback retry can correct the
    model.
    """

    allowed = list(labels)
    for text in _strip_label_decorations(raw.strip()):
        if text in allowed:
            return text
        lowered = text.lower()
        for label in allowed:
            if label.lower() == lowered:
                return label
    raise ValueError(f"unknown segment label {raw!r}; use exactly one of {sorted(allowed)}")


__all__ = [
    "NO_SKILL_LABEL",
    "NO_SKILL_PRE_LABEL",
    "SkillInvocation",
    "SkillSegment",
    "build_skill_segments",
    "detect_skill_invocations",
    "event_tokens",
    "normalize_segment_label",
    "render_attribution_trace",
    "render_event_text",
    "truncate_text",
]
