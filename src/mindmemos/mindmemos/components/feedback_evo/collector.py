"""Task-end feedback collector for the ``feedback_evo`` mode (方案 A)."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from ...infra.db import FeedbackEventStore
from ...llm import LLMClient, get_llm_client
from ...logging import get_logger
from ...prompts.EN.feedback_evo import EVO_SIGNAL_DETECTION_PROMPT
from ...typing import FeedbackEvoEvent, MemoryRequestContext

logger = get_logger(__name__)


def _extract_recalled_memories(messages: list[dict[str, Any]]) -> list[str]:
    """Extract memories the agent actually recalled via ``retrieve_learnings``.

    The STATE-Bench trajectory keeps tool-call results inline in the
    conversation (``tool_calls[].result.learnings``), so the recalled memories
    are read from the dialogue itself — no post-hoc search needed.
    """

    recalled: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict) or tool_call.get("name") != "retrieve_learnings":
                continue
            result = tool_call.get("result") or {}
            for learning in result.get("learnings") or []:
                if isinstance(learning, str) and learning not in seen:
                    seen.add(learning)
                    recalled.append(learning)
    return recalled


class FeedbackEvoCollector:
    """Collect user feedback from a finished task's context and persist it.

    Called synchronously at task end (方案 A). Detects signals with its own
    LLM call (dedicated evo signal-detection prompt) and writes the detected
    signals to ``feedback_event_v1`` for the evolution loop — no dependency on
    the existing implicit-feedback pipeline components or prompt.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        event_store: FeedbackEventStore | None = None,
        max_round_messages: int = 20,
    ) -> None:
        self._llm_client = llm_client
        self._event_store = event_store or FeedbackEventStore()
        self._max_round_messages = max_round_messages

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def collect(
        self,
        context: MemoryRequestContext,
        *,
        task_messages: list[dict[str, Any]],
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> FeedbackEvoEvent:
        """Build rounds, detect signals, persist one feedback event."""

        resolved_session = session_id or context.session_id or task_id or "task"
        rounds = [
            task_messages[i : i + self._max_round_messages]
            for i in range(0, len(task_messages), self._max_round_messages)
        ]
        if not rounds:
            rounds = [[]]

        signals = await self._detect_signals(resolved_session, rounds)
        signals = await self._enrich_signals(signals, rounds, task_messages)
        event = FeedbackEvoEvent(
            event_id=str(uuid4()),
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            user_id=context.user_id,
            session_id=session_id or context.session_id,
            app_id=context.app_id,
            agent_id=context.agent_id,
            task_id=task_id,
            signals=signals,
        )
        await self._event_store.append(context, event)
        return event

    async def _enrich_signals(
        self,
        signals: list[dict[str, Any]],
        rounds: list[list[dict[str, Any]]],
        task_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach the feedback-round dialogue and related memories to signals.

        The planner needs the user/agent exchange around each feedback point and
        the memories the agent actually recalled for that scenario, otherwise it
        cannot judge whether the add stage failed to extract/store the right
        knowledge. Recalled memories come from the conversation's
        ``retrieve_learnings`` tool results, preferring the signal's own round.
        """

        task_recalled = _extract_recalled_memories(task_messages)
        for signal in signals:
            round_index = signal.get("round_index")
            round_messages: list[dict[str, Any]] = []
            if isinstance(round_index, int) and 0 <= round_index < len(rounds):
                round_messages = rounds[round_index]
            signal["round_messages"] = round_messages
            signal["related_memories"] = (
                _extract_recalled_memories(round_messages) or task_recalled
            )
        return signals

    async def _detect_signals(
        self,
        session_id: str,
        rounds: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Detect negative feedback signals from task rounds (own LLM call)."""

        payload = {
            "session_id": session_id,
            "rounds": [
                {"round_index": index, "messages": messages}
                for index, messages in enumerate(rounds)
            ],
        }
        response = await self._client.chat(
            task="feedback_evo.detect_signals",
            messages=[
                {"role": "system", "content": EVO_SIGNAL_DETECTION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format_parser=_parse_signals,
            temperature=0,
        )
        result = response.parsed
        if not isinstance(result, list):
            raise TypeError("feedback_evo signal detector expected a parsed list")
        return result


def _parse_signals(content: str) -> list[dict[str, Any]]:
    """Parse the detector JSON, tolerating a surrounding object envelope."""

    text = content.strip()
    try:
        data = json.loads(text)
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        data = json.loads(text[start : end + 1])
    if isinstance(data, dict):
        return list(data.get("signals", []))
    if isinstance(data, list):
        return data
    return []
