"""STATE-Bench Agent Learning Track adapter backed by MindMemOS feedback_evo.

This file is installed by the StateBench runner into the STATE-Bench
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

STATE-Bench runtime failures are soft-fail: the adapter records search/add
failures and mapping-size telemetry in the saved trajectory metadata so the
runner can report the affected task and continue the batch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import httpx
from state_bench.agents.state_bench import StateBenchAgent

try:
    from .trajectory_mapping import TRAJECTORY_MAPPING_VERSION, mapping_stats, to_add_messages
except ImportError:  # Installed into STATE-Bench's top-level agents/ folder.
    try:
        from mindmemos_eval.memory.envs.statebench.trajectory_mapping import (
            TRAJECTORY_MAPPING_VERSION,
            mapping_stats,
            to_add_messages,
        )
    except ImportError:
        _mapping_path = Path(__file__).with_name("trajectory_mapping.py")
        _mapping_spec = importlib.util.spec_from_file_location(
            "_statebench_trajectory_mapping",
            _mapping_path,
        )
        if _mapping_spec is None or _mapping_spec.loader is None:
            raise ImportError(f"cannot load STATE-Bench mapping module: {_mapping_path}")
        _mapping_module = importlib.util.module_from_spec(_mapping_spec)
        _mapping_spec.loader.exec_module(_mapping_module)
        TRAJECTORY_MAPPING_VERSION = _mapping_module.TRAJECTORY_MAPPING_VERSION
        mapping_stats = _mapping_module.mapping_stats
        to_add_messages = _mapping_module.to_add_messages

# Kept as a compatibility alias for existing callers and tests.
_to_add_messages = to_add_messages


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _user_id(agent: StateBenchAgent, trajectory: Any | None = None) -> str:
    runtime = getattr(agent, "runtime_context", None)
    if runtime is not None and getattr(runtime, "user_id", None):
        raw_user_id = runtime.user_id
    elif trajectory is not None and getattr(trajectory, "user_id", None):
        raw_user_id = trajectory.user_id
    else:
        raw_user_id = "statebench"
    prefix = _env("MINDMEMOS_USER_ID_PREFIX")
    if not prefix:
        return str(raw_user_id)
    prefix = prefix.rstrip(":")
    raw_user_id = str(raw_user_id)
    if raw_user_id == prefix or raw_user_id.startswith(prefix + "::"):
        return raw_user_id
    return prefix + "::" + raw_user_id


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
        self._failures: list[dict[str, Any]] = []
        self._searches: list[dict[str, Any]] = []
        self._add_result: dict[str, Any] | None = None

    def _record_failure(self, stage: str, exc: Exception, **details: Any) -> None:
        self._failures.append(
            {
                "stage": stage,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                **details,
            }
        )

    def _attach_metadata(self, trajectory: Any) -> None:
        metadata = getattr(trajectory, "metadata", None)
        if not isinstance(metadata, dict):
            return
        state = metadata.setdefault("mindmemos", {})
        state["mapping_version"] = TRAJECTORY_MAPPING_VERSION
        state["failures"] = list(self._failures)
        state["searches"] = list(self._searches)
        if self._add_result is not None:
            state["add"] = dict(self._add_result)

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._api_base}{path}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _retrieve_learnings(
        self,
        query: str,
        top_k: int,
        *,
        source: str,
    ) -> list[str]:
        """Search memories and retain the evidence injected into this turn.

        STATE-Bench's canonical conversation only contains results from explicit
        ``retrieve_learnings`` tool calls.  Automatic retrieval happens while
        preparing the model input, so without this record its memory IDs and
        contents disappear from the saved trajectory.  Keep the public method's
        signature unchanged for STATE-Bench tool-schema compatibility and use
        this private entry point to distinguish both retrieval paths.
        """

        user_id = _user_id(self)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        try:
            data = self._request(
                "/v1/memory/search",
                {
                    "query": query,
                    "top_k": top_k,
                    "search_strategy": "fast",
                    "user_id": user_id,
                    "filters": {"user_id": user_id},
                },
            )
            memories = (data.get("data") or {}).get("memories") or []
            result: list[str] = []
            provenance: list[dict[str, Any]] = []
            for item in memories:
                if not isinstance(item, dict) or not item.get("memory"):
                    continue
                content = str(item["memory"])
                result.append(content)
                provenance.append(
                    {
                        "rank": len(provenance) + 1,
                        "memory_id": str(item.get("id") or ""),
                        "memory": content,
                        "memory_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "score": item.get("score"),
                        "memory_type": item.get("memory_type"),
                        "last_update_at": item.get("last_update_at"),
                        "event_time": item.get("event_time"),
                        "source_timestamp": item.get("source_timestamp"),
                        "lineage": item.get("lineage"),
                    }
                )
            self._searches.append(
                {
                    "provenance_version": "statebench-search-provenance-v1",
                    "search_index": len(self._searches) + 1,
                    "status": "ok",
                    "source": source,
                    "query": query,
                    "query_chars": len(query),
                    "query_sha256": query_sha256,
                    "requested_top_k": top_k,
                    "api_result_count": len(memories),
                    "result_count": len(result),
                    "score_available": any(item.get("score") is not None for item in provenance),
                    "results": provenance,
                    "user_id": user_id,
                }
            )
            print(f"[MindMemOS] search_results={len(memories)}", flush=True)
            return result
        except Exception as exc:  # STATE-Bench soft-fail: retain the task trace.
            self._record_failure("search", exc, query_chars=len(query), user_id=user_id)
            self._searches.append(
                {
                    "provenance_version": "statebench-search-provenance-v1",
                    "search_index": len(self._searches) + 1,
                    "status": "failed",
                    "source": source,
                    "query": query,
                    "query_chars": len(query),
                    "query_sha256": query_sha256,
                    "requested_top_k": top_k,
                    "api_result_count": 0,
                    "result_count": 0,
                    "score_available": False,
                    "results": [],
                    "user_id": user_id,
                    "error": str(exc),
                }
            )
            print(f"[MindMemOSAgent] retrieval failed for user turn: {exc}", flush=True)
            return []

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        """STATE-Bench tool entry point for explicit memory retrieval."""

        return self._retrieve_learnings(query, top_k, source="agent_tool")

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
            if isinstance(item, dict) and item.get("role") == "user" and str(item.get("content") or "").strip()
        ]
        new_users = user_messages[self._retrieved_user_count :]
        self._retrieved_user_count = len(user_messages)
        prepared = super().prepare_conversation(conversation)
        if not new_users:
            return prepared

        query = str(new_users[-1]["content"]).strip()
        learnings = self._retrieve_learnings(
            query,
            self.retrieve_learnings_top_k,
            source="auto_prepare",
        )
        if not learnings:
            return prepared

        content = (
            "Relevant procedural learnings retrieved from past user interactions "
            "(auto-refreshed for the latest user message):\n" + "\n".join(f"- {item}" for item in learnings)
        )
        return self.inject_system_message(prepared, content, before_last_user=True)

    def ingest_trajectory(self, trajectory: Any) -> None:
        """Write a finished task's conversation into MindMemOS (train role only).

        Memory ingestion is best-effort: failures are logged and must not abort
        the benchmark run. Feedback/eval roles never write memories; their
        feedback is collected out-of-band by the runner.
        """

        if self._role != "train":
            self._attach_metadata(trajectory)
            return
        try:
            messages = to_add_messages(trajectory.conversation)
            stats = mapping_stats(trajectory.conversation, messages)
            payload = {
                "messages": messages,
                "mode": "sync",
                "task_id": getattr(trajectory, "task_id", None),
                "user_id": _user_id(self, trajectory),
            }
            payload_bytes = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
            self._add_result = {
                "provenance_version": "statebench-add-provenance-v1",
                "status": "skipped" if not messages else "pending",
                "mapping_version": TRAJECTORY_MAPPING_VERSION,
                **stats,
                "payload_bytes": len(payload_bytes),
                "payload_sha256": payload_sha256,
            }
        except Exception as exc:
            self._record_failure(
                "add_prepare",
                exc,
                task_id=getattr(trajectory, "task_id", None),
            )
            self._add_result = {
                "provenance_version": "statebench-add-provenance-v1",
                "status": "failed",
                "mapping_version": TRAJECTORY_MAPPING_VERSION,
                "payload_bytes": 0,
                "payload_sha256": None,
            }
            self._attach_metadata(trajectory)
            return
        if not messages:
            self._record_failure("add", ValueError("trajectory has no add messages"))
            self._add_result["status"] = "failed"
            self._attach_metadata(trajectory)
            return
        try:
            response = self._request("/v1/memory/add", payload)
            response_memories = (response.get("data") or {}).get("memories") or []
            memory_events: list[dict[str, Any]] = []
            for event in response_memories:
                if not isinstance(event, dict):
                    continue
                content = str(event.get("content") or "")
                memory_events.append(
                    {
                        "operation": event.get("operation"),
                        "memory_id": str(event.get("memory_id") or ""),
                        "memory": content,
                        "memory_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "memory_type": event.get("memory_type") or event.get("mem_type"),
                        "confidence": event.get("confidence"),
                        "related_memory_ids": event.get("related_memory_ids") or [],
                        "graph_edge_count": event.get("graph_edge_count"),
                    }
                )
            self._add_result["status"] = "ok"
            self._add_result["memory_event_count"] = len(memory_events)
            self._add_result["memory_events"] = memory_events
            print(f"[MindMemOS] add_ok task_id={getattr(trajectory, 'task_id', '?')}", flush=True)
        except Exception as exc:  # pragma: no cover - defensive, benchmark continues
            print(f"[MindMemOSAgent] add failed for {getattr(trajectory, 'task_id', '?')}: {exc}", flush=True)
            self._record_failure(
                "add",
                exc,
                task_id=getattr(trajectory, "task_id", None),
                payload_sha256=payload_sha256,
            )
            self._add_result["status"] = "failed"
            self._add_result["error"] = str(exc)
        finally:
            self._attach_metadata(trajectory)
