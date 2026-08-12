"""STATE-Bench Agent Learning Track adapter backed by MindMemOS feedback_evo.

This file is installed by ``run_feedback_evo_loop.py`` into the STATE-Bench
repo-root ``agents/`` directory (the official extension point for
``--agent-class``). It must stay self-contained: it imports only the standard
library plus ``httpx``, because the STATE-Bench loader executes it inside the
STATE-Bench venv, which does not have the ``mindmemos`` package installed.

Behavior is controlled by environment variables set by the runner:

* ``MINDMEMOS_API_BASE``   MindMemOS server base URL (default
  ``http://localhost:8000``).
* ``MINDMEMOS_API_KEY``    Bearer API key with ``memory_algorithm:
  feedback_evo`` and read/write scopes (required).
* ``MINDMEMOS_ROLE``       ``train`` (add + retrieve), ``feedback`` or ``eval``
  (retrieve only; task-end feedback is collected by the runner). Default
  ``eval`` so accidental runs are read-only.
* ``MINDMEMOS_TIMEOUT_SECONDS`` HTTP timeout for add/search calls (default 120).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from state_bench.agents.state_bench import StateBenchAgent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _user_id(agent: StateBenchAgent, trajectory: Any | None = None) -> str:
    runtime = getattr(agent, "runtime_context", None)
    if runtime is not None and getattr(runtime, "user_id", None):
        return runtime.user_id
    if trajectory is not None and getattr(trajectory, "user_id", None):
        return trajectory.user_id
    return "statebench"


class MindMemOSAgent(StateBenchAgent):
    """StateBenchAgent whose retrieval/memory comes from MindMemOS feedback_evo."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._api_base = _env("MINDMEMOS_API_BASE", "http://localhost:8000").rstrip("/")
        self._api_key = _env("MINDMEMOS_API_KEY")
        if not self._api_key:
            raise RuntimeError("MINDMEMOS_API_KEY is required for MindMemOSAgent")
        self._role = (_env("MINDMEMOS_ROLE", "eval") or "eval").strip().lower()
        if self._role not in {"train", "feedback", "eval"}:
            raise ValueError(f"unknown MINDMEMOS_ROLE {self._role!r}")
        self._timeout = float(_env("MINDMEMOS_TIMEOUT_SECONDS", "120") or 120)
        # Number of user messages already covered by an automatic retrieval.
        self._retrieved_user_count = 0

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._api_base}{path}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        """Search MindMemOS feedback_evo memories and return content strings."""

        data = self._request(
            "/v1/memory/search",
            {
                "query": query,
                "top_k": top_k,
                "search_strategy": "fast",
                "user_id": _user_id(self),
            },
        )
        memories = (data.get("data") or {}).get("memories") or []
        return [str(item["memory"]) for item in memories if item.get("memory")]

    def prepare_conversation(self, conversation: list[Any]) -> list[Any]:
        """Auto-retrieve procedural learnings on every new user message.

        ``StateBenchAgent`` only instructs the model to call
        ``retrieve_learnings`` before the first substantive answer, so in
        practice retrieval happens once per task and later user answers get no
        fresh guidance. Here we force a retrieval whenever a new user message
        appears and inject the results as a system message placed right before
        that user turn. The canonical transcript is left untouched.
        """

        user_messages = [
            item
            for item in conversation
            if isinstance(item, dict)
            and item.get("role") == "user"
            and str(item.get("content") or "").strip()
        ]
        new_users = user_messages[self._retrieved_user_count :]
        self._retrieved_user_count = len(user_messages)
        prepared = super().prepare_conversation(conversation)
        if not new_users:
            return prepared

        query = str(new_users[-1]["content"]).strip()
        try:
            learnings = self.retrieve_learnings(query, top_k=self.retrieve_learnings_top_k)
        except Exception as exc:  # pragma: no cover - defensive, benchmark continues
            print(f"[MindMemOSAgent] retrieval failed for user turn: {exc}", flush=True)
            return prepared
        if not learnings:
            return prepared

        content = (
            "Relevant procedural learnings retrieved from past user interactions "
            "(auto-refreshed for the latest user message):\n"
            + "\n".join(f"- {item}" for item in learnings)
        )
        return self.inject_system_message(prepared, content, before_last_user=True)

    def ingest_trajectory(self, trajectory: Any) -> None:
        """Write a finished task's conversation into MindMemOS (train role only).

        Memory ingestion is best-effort: failures are logged and must not abort
        the benchmark run. Feedback/eval roles never write memories; their
        feedback is collected out-of-band by the runner.
        """

        if self._role != "train":
            return
        messages = _to_add_messages(trajectory.conversation)
        if not messages:
            return
        try:
            self._request(
                "/v1/memory/add",
                {
                    "messages": messages,
                    "mode": "sync",
                    "task_id": getattr(trajectory, "task_id", None),
                    "user_id": _user_id(self, trajectory),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive, benchmark continues
            print(f"[MindMemOSAgent] add failed for {getattr(trajectory, 'task_id', '?')}: {exc}", flush=True)


def _to_add_messages(conversation: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map STATE-Bench conversation items to MindMemOS dialogue messages.

    ``tool_calls`` are not accepted by the add schema, so they are embedded
    into the assistant content when the message has no other text.
    """

    messages: list[dict[str, str]] = []
    for message in conversation:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "assistant")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content is None or not str(content).strip():
            if not tool_calls:
                continue
            content = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
        messages.append({"role": role, "content": str(content)})
    return messages
