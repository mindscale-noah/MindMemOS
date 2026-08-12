"""Feedback-driven self-evolution components (``feedback_evo`` mode)."""

from __future__ import annotations

import json
from typing import Any

from ...config import get_config
from ...infra.db.evolution_state import EvolutionStateStore
from ...llm import LLMClient, get_llm_client
from ...logging import get_logger
from ...prompts.EN.feedback_evo import (
    FEEDBACK_EVO_ENTITY_TAGGING_PROMPT,
    FEEDBACK_EVO_EXTRACTION_SYSTEM_PROMPT,
    PARAM_PLANNING_PROMPT,
)
from ...typing import (
    EvolutionResult,
    EvolutionState,
    EvolutionTrigger,
    FeedbackEvoEvent,
    ParameterChange,
)

logger = get_logger(__name__)

# The only evolvable parameter paths (must match the planning prompt list).
# Sub-paths under ``weights`` are allowed (e.g. ``weights.fact``); everything
# else the planner proposes is dropped so ghost parameters never enter state.
EVOLVABLE_ADD_PATHS = ("extraction_prompt", "entity_tagging_prompt", "entity_types")
EVOLVABLE_SEARCH_PATHS = ("top_k", "rerank", "score_threshold", "weights")

# Change-threshold bounds for non-prompt evolvable items.
PROMPT_PATHS = {"extraction_prompt", "entity_tagging_prompt"}
MAX_TOP_K = 200


def is_evolvable_path(path: str) -> bool:
    """Return True when a planner change targets a consumed evolvable path."""

    if path.startswith("add_config."):
        return path.removeprefix("add_config.") in EVOLVABLE_ADD_PATHS
    if path.startswith("search_config."):
        rest = path.removeprefix("search_config.")
        return rest in EVOLVABLE_SEARCH_PATHS or rest.startswith("weights.")
    return False


def build_initial_evolution_state(project_id: str) -> EvolutionState:
    """Build a v1 evolution state seeded with only the evolvable fields.

    Only fields the feedback_evo pipelines actually consume are stored, so the
    state (and the planner's view of it) never exposes unused vanilla fields
    (recall_size, hybrid_prefetch_*, fusion weights, ...) as ghost paths.
    """

    cfg = get_config()
    fe_cfg = cfg.algo_config.add.feedback_evo
    add_config = {
        "extraction_prompt": fe_cfg.extraction_prompt or FEEDBACK_EVO_EXTRACTION_SYSTEM_PROMPT,
        "entity_tagging_prompt": fe_cfg.entity_tagging_prompt or FEEDBACK_EVO_ENTITY_TAGGING_PROMPT,
        "entity_types": list(fe_cfg.entity_types),
    }
    # Search evolvable items (top_k / rerank / score_threshold / weights) are
    # request- and evolution-driven with no static baseline; start empty so
    # requests keep their defaults until evolution sets a value.
    search_config: dict[str, Any] = {}
    return EvolutionState(
        project_id=project_id,
        version=1,
        is_current=True,
        add_config=add_config,
        search_config=search_config,
    )


async def ensure_evolution_state(
    state_store: EvolutionStateStore,
    project_id: str,
) -> EvolutionState:
    """Return the current evolution state, seeding v1 from vanilla when absent."""

    current = await state_store.get_current(project_id)
    if current is not None:
        return current
    initial = build_initial_evolution_state(project_id)
    await state_store.apply(
        project_id,
        add_config=initial.add_config,
        search_config=initial.search_config,
        changes=[],
    )
    return initial


def _valid_signals(events: list[FeedbackEvoEvent]) -> list[dict[str, Any]]:
    """Flatten event signals, keeping only whitelisted evolvable-path signals.

    Raw signals (with ``evolvable_path`` / ``confidence`` / ``reason``) are
    handed straight to the planner; only deterministic gates stay in code.
    """

    valid: list[dict[str, Any]] = []
    for event in events:
        for signal in event.signals:
            path = signal.get("evolvable_path")
            if isinstance(path, str) and is_evolvable_path(path):
                valid.append(signal)
    return valid


def _signal_confidence(signals: list[dict[str, Any]]) -> float:
    """Mean confidence over valid signals (missing/invalid -> 1.0)."""

    if not signals:
        return 0.0
    total = 0.0
    for signal in signals:
        confidence = signal.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            confidence = 1.0
        total += float(confidence)
    return total / len(signals)


class EvolutionPlanner:
    """LLM planner: propose concrete parameter changes for the root cause."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_changes_per_evolution: int | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._max_changes_per_evolution = max_changes_per_evolution

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def plan(
        self,
        signals: list[dict[str, Any]],
        current: EvolutionState | None,
        *,
        max_changes: int | None = None,
    ) -> list[ParameterChange]:
        """Plan parameter changes, optionally capped by an explicit budget."""

        limit = max_changes if max_changes is not None else self._max_changes_per_evolution
        payload = {
            "signals": signals,
            "current_config": _evolvable_config_view(current),
        }
        system_prompt = PARAM_PLANNING_PROMPT
        response = await self._client.chat(
            task="feedback_evo.plan_parameters",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            format_parser=_parse_plan,
            temperature=0,
        )
        changes = response.parsed
        if not isinstance(changes, list):
            raise TypeError("evolution planner expected a parsed list of changes")
        filtered = []
        for change in changes:
            if is_evolvable_path(change.path):
                filtered.append(change)
            else:
                logger.warning("evolution_change_dropped_not_evolvable", path=change.path)
        return filtered if limit is None else filtered[:limit]


class EvolutionExecutor:
    """Orchestrate signals -> gates -> plan -> threshold -> apply -> record."""

    def __init__(
        self,
        *,
        planner: EvolutionPlanner | None = None,
        state_store: EvolutionStateStore | None = None,
        min_signals_to_evolve: int = 5,
        require_signal_confidence: float = 0.7,
        max_numeric_change_ratio: float = 0.5,
        max_entity_type_delta: int = 2,
    ) -> None:
        self._planner = planner or EvolutionPlanner()
        self._state_store = state_store or EvolutionStateStore()
        self._min_signals_to_evolve = min_signals_to_evolve
        self._require_signal_confidence = require_signal_confidence
        self._max_numeric_change_ratio = max_numeric_change_ratio
        self._max_entity_type_delta = max_entity_type_delta

    async def run(
        self,
        project_id: str,
        events: list[FeedbackEvoEvent],
    ) -> EvolutionResult:
        """Run one evolution round over the accumulated feedback events."""

        signals = _valid_signals(events)
        if len(signals) < self._min_signals_to_evolve:
            return EvolutionResult(project_id=project_id, version=0, changes=[])

        current = await self._state_store.get_current(project_id)
        confidence = _signal_confidence(signals)
        if confidence < self._require_signal_confidence:
            logger.info(
                "evolution_skipped_low_signal_confidence",
                confidence=confidence,
                threshold=self._require_signal_confidence,
            )
            return EvolutionResult(
                project_id=project_id,
                version=current.version if current is not None else 0,
                changes=[],
            )
        planned = await self._planner.plan(signals, current)
        changes = _filter_changes_by_threshold(
            current,
            planned,
            max_numeric_change_ratio=self._max_numeric_change_ratio,
            max_entity_type_delta=self._max_entity_type_delta,
        )
        if not changes:
            return EvolutionResult(
                project_id=project_id,
                version=current.version if current is not None else 0,
                changes=[],
            )

        add_config, search_config = _apply_changes(current, changes)
        signal_ids = [
            f"{event.event_id}#{index}"
            for event in events
            for index in range(len(event.signals))
        ]
        trigger = EvolutionTrigger(
            signal_ids=signal_ids,
        )
        return await self._state_store.apply(
            project_id,
            add_config=add_config,
            search_config=search_config,
            changes=changes,
            trigger=trigger,
        )


def _apply_changes(
    current: EvolutionState | None,
    changes: list[ParameterChange],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (add_config, search_config) with the planned changes applied."""

    add_config = dict(current.add_config) if current is not None else {}
    search_config = dict(current.search_config) if current is not None else {}
    for change in changes:
        if change.path.startswith("add_config."):
            _set_path(add_config, change.path.removeprefix("add_config."), change.after)
        elif change.path.startswith("search_config."):
            _set_path(search_config, change.path.removeprefix("search_config."), change.after)
        else:
            logger.warning("evolution_change_skipped", path=change.path)
    return add_config, search_config


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Set a value at a dotted path inside a dict, creating intermediate dicts."""

    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            node = {}
    node[parts[-1]] = value


def _get_path(target: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path, or None when absent."""

    node: Any = target
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _evolvable_config_view(current: EvolutionState | None) -> dict[str, Any] | None:
    """Trim an evolution state to the evolvable fields only (no ghost paths).

    The planner must never see unused vanilla fields (recall_size,
    hybrid_prefetch_*, fusion weights, ...), otherwise it invents changes for
    paths that are not consumed. ``weights`` is included whole so sub-paths
    like ``weights.fact`` are visible.
    """

    if current is None:
        return None
    add_config = {key: current.add_config[key] for key in EVOLVABLE_ADD_PATHS if key in current.add_config}
    search_config = {key: current.search_config[key] for key in EVOLVABLE_SEARCH_PATHS if key in current.search_config}
    return {
        "version": current.version,
        "add_config": add_config,
        "search_config": search_config,
    }


def _filter_changes_by_threshold(
    current: EvolutionState | None,
    changes: list[ParameterChange],
    *,
    max_numeric_change_ratio: float,
    max_entity_type_delta: int,
) -> list[ParameterChange]:
    """Drop changes outside evolvable whitelist or value-change thresholds."""

    add_config = dict(current.add_config) if current is not None else {}
    search_config = dict(current.search_config) if current is not None else {}
    kept: list[ParameterChange] = []
    for change in changes:
        if not is_evolvable_path(change.path):
            logger.warning("evolution_change_dropped_not_evolvable", path=change.path)
            continue
        if _change_within_threshold(
            add_config,
            search_config,
            change,
            max_numeric_change_ratio=max_numeric_change_ratio,
            max_entity_type_delta=max_entity_type_delta,
        ):
            kept.append(change)
        else:
            logger.warning(
                "evolution_change_dropped_out_of_bounds",
                path=change.path,
                after=change.after,
            )
    return kept


def _change_within_threshold(
    add_config: dict[str, Any],
    search_config: dict[str, Any],
    change: ParameterChange,
    *,
    max_numeric_change_ratio: float,
    max_entity_type_delta: int,
) -> bool:
    """Validate one planned change against prompts/thresholds/domains."""

    path = change.path
    if path.startswith("add_config."):
        rest = path.removeprefix("add_config.")
        target = add_config
    else:
        rest = path.removeprefix("search_config.")
        target = search_config
    after = change.after

    if rest in PROMPT_PATHS:
        return isinstance(after, str) and bool(after.strip())
    if rest == "entity_types":
        return _entity_types_delta_within(target.get("entity_types"), after, max_entity_type_delta)
    if rest == "top_k":
        return _numeric_within(target.get("top_k"), after, max_numeric_change_ratio, integer=True, lo=1, hi=MAX_TOP_K)
    if rest == "score_threshold":
        return _numeric_within(target.get("score_threshold"), after, max_numeric_change_ratio, integer=False, lo=0.0, hi=1.0)
    if rest == "rerank":
        return isinstance(after, bool)
    if rest == "weights" or rest.startswith("weights."):
        return _numeric_within(_get_path(target, rest), after, max_numeric_change_ratio, integer=False, lo=0.0, hi=1.0)
    return False


def _numeric_within(
    old: Any,
    new: Any,
    ratio: float,
    *,
    integer: bool,
    lo: float,
    hi: float,
) -> bool:
    """Return True when ``new`` is a valid numeric value near ``old``."""

    if isinstance(new, bool) or not isinstance(new, (int, float)):
        return False
    new = float(new)
    if new < lo or new > hi:
        return False
    if integer and new != int(new):
        return False
    if old is None:
        return True
    if isinstance(old, bool) or not isinstance(old, (int, float)):
        return False
    old = float(old)
    floor = 1.0 if integer else 0.05
    limit = max(abs(old) * ratio, floor)
    return abs(new - old) <= limit


def _entity_types_delta_within(old: Any, new: Any, max_delta: int) -> bool:
    """Return True when the entity-type vocabulary change is small enough."""

    if not isinstance(new, list) or not all(isinstance(item, str) for item in new):
        return False
    old_set = set(old) if isinstance(old, list) else set()
    new_set = set(new)
    delta = len(new_set - old_set) + len(old_set - new_set)
    return delta <= max_delta


def _parse_plan(content: str) -> list[ParameterChange]:
    data = json.loads(_json_object_text(content))
    changes = data.get("changes", []) if isinstance(data, dict) else data
    return [ParameterChange.model_validate(item) for item in changes]


def _json_object_text(content: str) -> str:
    text = content.strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return text[start : end + 1]
