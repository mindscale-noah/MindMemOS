"""Runner for STATE-Bench feedback-evolution comparison experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from .paths import REPO_ROOT, STATE_BENCH_ROOT
from .schedule import FeedbackEvoSchedule, RoundPlan, build_schedule
from .trajectory_mapping import TRAJECTORY_MAPPING_VERSION, mapping_stats, to_add_messages

DEFAULT_STATE_BENCH_DIR = STATE_BENCH_ROOT
DEFAULT_CHECKPOINTS = "0,2,4,6,8,10"
AGENT_SOURCE = Path(__file__).resolve().parent / "mindmemos_agent.py"
MAPPING_SOURCE = Path(__file__).resolve().parent / "trajectory_mapping.py"


def _load_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _env(state_bench_dir: Path, extra: dict[str, str]) -> dict[str, str]:
    result = dict(os.environ)
    result.update(_load_dotenv_file(state_bench_dir / ".env"))
    result.update(extra)
    return result


def _experiment_id() -> str | None:
    value = os.environ.get("STATEBENCH_EXPERIMENT_ID", "").strip()
    return value or None


def _scoped_user_id(raw_user_id: Any, prefix: str | None = None) -> str | None:
    if raw_user_id is None:
        return None
    user_id = str(raw_user_id)
    experiment_id = prefix if prefix is not None else _experiment_id()
    if not experiment_id:
        return user_id
    experiment_id = str(experiment_id).rstrip(":")
    if user_id == experiment_id or user_id.startswith(f"{experiment_id}::"):
        return user_id
    return f"{experiment_id}::{user_id}"


def _batch_complete(output_dir: Path, task_ids: list[str], num_runs: int) -> bool:
    if not output_dir.exists():
        return False
    expected = set(task_ids)
    return all(
        {path.stem for path in _trajectory_files(output_dir, run_index)} == expected
        for run_index in range(1, num_runs + 1)
    )


def _run_batch(
    state_bench_dir: Path,
    *,
    domain: str,
    task_ids: list[str],
    output_dir: Path,
    log_path: Path,
    agent_class: str,
    agent_model_name: str,
    reasoning_level: str | None,
    num_workers: int,
    no_score: bool,
    role: str,
    api_base: str,
    api_key: str,
    num_runs: int = 1,
    user_id_prefix: str | None = None,
) -> dict[str, Any]:
    failure_path = log_path.with_suffix(".failures.json")

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        _write_json(
            failure_path,
            {
                "role": role,
                "output_dir": str(output_dir),
                "failures": result.get("failures", []),
            },
        )
        return result

    resume_batches = None
    if output_dir.exists():
        if _batch_complete(output_dir, task_ids, num_runs):
            result = _batch_observations(output_dir, task_ids, num_runs, role=role)
            result["reused"] = 1
            return finish(result)
        pending = []
        for run_index in range(1, num_runs + 1):
            seen = {path.stem for path in _trajectory_files(output_dir, run_index)}
            missing = [task_id for task_id in task_ids if task_id not in seen]
            if missing:
                pending.append((run_index, missing))
        if pending:
            resume_batches = pending
        elif any(output_dir.iterdir()):
            return finish(
                {
                    "status": "incomplete",
                    "add_ok": 0,
                    "search_results": 0,
                    "reused": 0,
                    "failures": [
                        {
                            "stage": "batch",
                            "status": "failed",
                            "error": "existing output has no recoverable missing task set",
                            "output_dir": str(output_dir),
                        }
                    ],
                }
            )
    python = state_bench_dir / ".venv" / "bin" / "python"
    if not python.exists():
        return finish(
            {
                "status": "failed",
                "add_ok": 0,
                "search_results": 0,
                "reused": 0,
                "failures": [
                    {
                        "stage": "batch",
                        "status": "failed",
                        "error": f"STATE-Bench venv not found: {python}",
                    }
                ],
            }
        )
    common = [
        str(python),
        "-u",
        "-m",
        "state_bench.scripts.run_batch",
        "--domain",
        domain,
        "--agent-class",
        agent_class,
        "--agent-model-name",
        agent_model_name,
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(num_workers),
    ]
    if reasoning_level:
        common += ["--agent-model-reasoning-level", reasoning_level]
    if no_score:
        common.append("--no-score")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    batches = (
        [(None, task_ids, num_runs)]
        if resume_batches is None
        else [(run_index, missing, 1) for run_index, missing in resume_batches]
    )
    returncodes = []
    for batch_index, (run_index, pending_ids, batch_runs) in enumerate(batches):
        command = [*common, "--tasks", ",".join(pending_ids), "--num-runs", str(batch_runs)]
        if run_index is not None:
            command += ["--num-runs-idx-start", str(run_index)]
        mode = "ab" if batch_index or resume_batches is not None else "wb"
        try:
            with log_path.open(mode) as log_file:
                if mode == "ab":
                    log_file.write(
                        f"\n[runner resume batch run={run_index or 'all'} tasks={len(pending_ids)}]\n".encode()
                    )
                result = subprocess.run(
                    command,
                    cwd=state_bench_dir,
                    env=_env(
                        state_bench_dir,
                        {
                            "MINDMEMOS_API_BASE": api_base,
                            "MINDMEMOS_API_KEY": api_key,
                            "MINDMEMOS_ROLE": role,
                            "MINDMEMOS_USER_ID_PREFIX": (
                                user_id_prefix if user_id_prefix is not None else (_experiment_id() or "")
                            ),
                        },
                    ),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            returncodes.append(result.returncode)
        except Exception as exc:
            returncodes.append(-1)
            with log_path.open("ab") as log_file:
                log_file.write(f"\n[runner exception] {exc}\n".encode())
    observations = _batch_observations(output_dir, task_ids, num_runs, role=role)
    if any(code != 0 for code in returncodes):
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        observations.setdefault("failures", []).append(
            {
                "stage": "batch",
                "status": "failed",
                "error": f"run_batch exited with status {returncodes}",
                "log_tail": tail,
            }
        )
    observations["reused"] = 0
    observations["status"] = "completed_with_failures" if observations.get("failures") else "ok"
    return finish(observations)


def _post(api_base: str, api_key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=600) as client:
        response = client.post(
            f"{api_base.rstrip('/')}{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def _trajectory_user_id(
    trajectory: dict[str, Any],
    user_id_prefix: str | None = None,
) -> str | None:
    direct = trajectory.get("user_id")
    if direct:
        return _scoped_user_id(direct, user_id_prefix)
    for message in trajectory.get("conversation") or []:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        match = re.search(
            r"\b(?:customer|traveler|shopper|user)[ _-]?(?:id)?\s*[:=]?\s*([A-Za-z][A-Za-z0-9_-]*)",
            str(message.get("content") or ""),
            re.I,
        )
        if match:
            return _scoped_user_id(match.group(1), user_id_prefix)
    return None


def _collect(
    api_base: str,
    api_key: str,
    trajectory: dict[str, Any],
    user_id_prefix: str | None = None,
) -> dict[str, Any]:
    task_id = str(trajectory.get("task_id") or "")
    payload = {
        "task_messages": trajectory.get("conversation") or [],
        "task_id": task_id,
        "user_id": _trajectory_user_id(trajectory, user_id_prefix),
    }
    payload_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "task_id": task_id,
        "status": "pending",
        "collect_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "collect_payload_bytes": len(payload_bytes),
    }
    try:
        body = _post(api_base, api_key, "/v1/memory/feedback-evo/collect", payload)
        data = body.get("data") or {}
        result.update(
            {
                "status": "ok",
                "event_id": data.get("event_id"),
                "signal_count": int(data.get("signal_count") or 0),
                "signals": data.get("signals") or [],
                "message": body.get("message") or "",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "signal_count": 0,
                "signals": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def _evolve(
    api_base: str,
    api_key: str,
    *,
    event_selection: str = "all",
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "force": True,
        "event_selection": event_selection,
    }
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    try:
        body = _post(api_base, api_key, "/v1/memory/self-evolve", payload)
        data = body.get("data") or {}
        return {
            "status": "ok",
            "evolved": bool(data.get("evolved")),
            "version": int(data.get("version") or 0),
            "changes": data.get("changes") or [],
            "signal_count": int(data.get("signal_count") or 0),
            "selected_event_count": int(data.get("selected_event_count") or 0),
            "consumed_event_count": int(data.get("consumed_event_count") or 0),
            "idempotent_replay": bool(data.get("idempotent_replay")),
            "cost_usd": data.get("cost_usd", data.get("evolution_cost_usd")),
            "message": body.get("message") or "",
            "event_selection": event_selection,
            "idempotency_key": idempotency_key,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "evolved": False,
            "version": None,
            "changes": [],
            "signal_count": 0,
            "selected_event_count": 0,
            "consumed_event_count": 0,
            "cost_usd": 0.0,
            "message": str(exc),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "event_selection": event_selection,
            "idempotency_key": idempotency_key,
        }


def _add_payload(trajectory: dict[str, Any]) -> dict[str, Any]:
    messages = to_add_messages(trajectory.get("conversation") or [])
    if not messages:
        raise ValueError(f"trajectory {trajectory.get('task_id')} has no add messages")
    return {
        "messages": messages,
        "mode": "sync",
        "task_id": trajectory.get("task_id"),
        "user_id": _trajectory_user_id(trajectory),
    }


def _replay_add(api_base: str, api_key: str, trajectory: dict[str, Any]) -> dict[str, Any]:
    task_id = trajectory.get("task_id")
    try:
        payload = _add_payload(trajectory)
        stats = mapping_stats(trajectory.get("conversation") or [], payload["messages"])
        payload_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result = {
            "task_id": task_id,
            "status": "pending",
            "message_count": len(payload["messages"]),
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "payload_bytes": len(payload_bytes),
            "mapping_version": TRAJECTORY_MAPPING_VERSION,
            **stats,
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "status": "failed",
            "message_count": 0,
            "payload_sha256": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        result["response"] = _post(api_base, api_key, "/v1/memory/add", payload).get("data") or {}
        result["status"] = "ok"
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def _trajectory_files(directory: Path, run: int = 1) -> list[Path]:
    return sorted((directory / f"run{run}").glob("*.json"))


def _batch_observations(
    output_dir: Path,
    task_ids: list[str],
    num_runs: int,
    *,
    role: str,
) -> dict[str, Any]:
    """Inspect saved STATE-Bench output without turning failures into raises."""

    expected = set(task_ids)
    failures: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    add_ok = 0
    add_failed = 0
    added_memory_events = 0
    added_memory_events_with_id = 0
    search_calls = 0
    search_failures = 0
    search_results = 0
    search_provenance_calls = 0
    search_source_counts: dict[str, int] = {}
    retrieved_memory_refs = 0
    retrieved_memory_refs_with_id = 0
    retrieved_memory_refs_with_score = 0
    raw_content_chars = 0
    mapped_content_chars = 0
    payload_bytes = 0
    expansion_ratios: list[float] = []
    completed_tasks = 0

    for run_index in range(1, num_runs + 1):
        files = _trajectory_files(output_dir, run_index)
        seen = {path.stem for path in files}
        for task_id in sorted(expected - seen):
            failures.append(
                {
                    "stage": "trajectory",
                    "status": "failed",
                    "task_id": task_id,
                    "run_index": run_index,
                    "error": "trajectory file missing",
                }
            )
        for path in files:
            if path.stem not in expected:
                failures.append(
                    {
                        "stage": "trajectory",
                        "status": "failed",
                        "task_id": path.stem,
                        "run_index": run_index,
                        "error": "unexpected trajectory file",
                    }
                )
                continue
            try:
                trajectory = _load(path)
                if not isinstance(trajectory, dict):
                    raise TypeError("trajectory JSON must be an object")
            except Exception as exc:
                failures.append(
                    {
                        "stage": "trajectory",
                        "status": "failed",
                        "task_id": path.stem,
                        "run_index": run_index,
                        "error": str(exc),
                    }
                )
                continue
            completed_tasks += 1
            task_id = str(trajectory.get("task_id") or path.stem)
            mindmemos = trajectory.get("mindmemos") or {}
            if not isinstance(mindmemos, dict):
                failures.append(
                    {
                        "stage": "telemetry",
                        "status": "failed",
                        "task_id": task_id,
                        "run_index": run_index,
                        "error": "mindmemos metadata is not an object",
                    }
                )
                mindmemos = {}
            task_failure_records = []
            failure_items = mindmemos.get("failures") or []
            if not isinstance(failure_items, list):
                failures.append(
                    {
                        "stage": "telemetry",
                        "status": "failed",
                        "task_id": task_id,
                        "run_index": run_index,
                        "error": "failures metadata is not an array",
                    }
                )
                failure_items = []
            for failure in failure_items:
                if not isinstance(failure, dict):
                    failures.append(
                        {
                            "stage": "telemetry",
                            "status": "failed",
                            "task_id": task_id,
                            "run_index": run_index,
                            "error": "invalid failure record in mindmemos metadata",
                        }
                    )
                    continue
                record = {"task_id": task_id, "run_index": run_index, **failure}
                failures.append(record)
                task_failure_records.append(record)
            add = mindmemos.get("add") or {}
            if not isinstance(add, dict):
                failures.append(
                    {
                        "stage": "telemetry",
                        "status": "failed",
                        "task_id": task_id,
                        "run_index": run_index,
                        "error": "add metadata is not an object",
                    }
                )
                add = {}
            if role == "train" and not add:
                failure = {
                    "stage": "add",
                    "status": "failed",
                    "task_id": task_id,
                    "run_index": run_index,
                    "error": "missing mindmemos.add metadata",
                }
                failures.append(failure)
                task_failure_records.append(failure)
            if add.get("status") == "ok":
                add_ok += 1
                memory_events = add.get("memory_events") or []
                if isinstance(memory_events, list):
                    added_memory_events += len(memory_events)
                    added_memory_events_with_id += sum(
                        1 for event in memory_events if isinstance(event, dict) and event.get("memory_id")
                    )
            elif role == "train":
                add_failed += 1
                if not any(item.get("stage") == "add" for item in task_failure_records):
                    failures.append(
                        {
                            "stage": "add",
                            "status": "failed",
                            "task_id": task_id,
                            "run_index": run_index,
                            "error": add.get("error") or "add failed",
                        }
                    )
            searches = mindmemos.get("searches") or []
            if not isinstance(searches, list):
                failures.append(
                    {
                        "stage": "telemetry",
                        "status": "failed",
                        "task_id": task_id,
                        "run_index": run_index,
                        "error": "searches metadata is not an array",
                    }
                )
                searches = []
            for search in searches:
                if not isinstance(search, dict):
                    failures.append(
                        {
                            "stage": "telemetry",
                            "status": "failed",
                            "task_id": task_id,
                            "run_index": run_index,
                            "error": "invalid search record in mindmemos metadata",
                        }
                    )
                    continue
                search_calls += 1
                search_results += _safe_int(search.get("result_count"))
                if search.get("status") == "failed":
                    search_failures += 1
                if search.get("provenance_version") == "statebench-search-provenance-v1":
                    search_provenance_calls += 1
                    source = str(search.get("source") or "unknown")
                    search_source_counts[source] = search_source_counts.get(source, 0) + 1
                    results = search.get("results") or []
                    if not isinstance(results, list):
                        failures.append(
                            {
                                "stage": "telemetry",
                                "status": "failed",
                                "task_id": task_id,
                                "run_index": run_index,
                                "error": "search provenance results is not an array",
                            }
                        )
                        results = []
                    retrieved_memory_refs += len(results)
                    for result in results:
                        if not isinstance(result, dict):
                            failures.append(
                                {
                                    "stage": "telemetry",
                                    "status": "failed",
                                    "task_id": task_id,
                                    "run_index": run_index,
                                    "error": "invalid search provenance result",
                                }
                            )
                            continue
                        if result.get("memory_id"):
                            retrieved_memory_refs_with_id += 1
                        if result.get("score") is not None:
                            retrieved_memory_refs_with_score += 1
            raw_content_chars += _safe_int(add.get("raw_content_chars"))
            mapped_content_chars += _safe_int(add.get("mapped_content_chars"))
            payload_bytes += _safe_int(add.get("payload_bytes"))
            if add.get("content_expansion_ratio") is not None:
                try:
                    expansion_ratios.append(float(add["content_expansion_ratio"]))
                except (TypeError, ValueError):
                    failures.append(
                        {
                            "stage": "telemetry",
                            "status": "failed",
                            "task_id": task_id,
                            "run_index": run_index,
                            "error": "invalid content_expansion_ratio",
                        }
                    )

            if role == "eval" and trajectory.get("task_completion_pass") is None:
                failure = {
                    "stage": "score",
                    "status": "failed",
                    "task_id": task_id,
                    "run_index": run_index,
                    "error": "task_completion_pass is missing",
                }
                failures.append(failure)
                task_failure_records.append(failure)
            if trajectory.get("error"):
                failure = {
                    "stage": "task",
                    "status": "failed",
                    "task_id": task_id,
                    "run_index": run_index,
                    "error": trajectory["error"],
                }
                failures.append(failure)
                task_failure_records.append(failure)
            task_records.append(
                {
                    "task_id": task_id,
                    "run_index": run_index,
                    "failures": task_failure_records,
                    "add": add,
                    "searches": searches,
                }
            )

    expected_count = len(expected) * num_runs
    return {
        "status": "completed_with_failures" if failures else "ok",
        "expected_tasks": expected_count,
        "completed_tasks": completed_tasks,
        "add_ok": add_ok,
        "add_failed": add_failed,
        "added_memory_events": added_memory_events,
        "added_memory_events_with_id": added_memory_events_with_id,
        "search_calls": search_calls,
        "search_failures": search_failures,
        "search_results": search_results,
        "search_provenance_calls": search_provenance_calls,
        "search_provenance_coverage": (search_provenance_calls / search_calls if search_calls else 1.0),
        "search_source_counts": dict(sorted(search_source_counts.items())),
        "retrieved_memory_refs": retrieved_memory_refs,
        "retrieved_memory_refs_with_id": retrieved_memory_refs_with_id,
        "retrieved_memory_refs_with_score": retrieved_memory_refs_with_score,
        "raw_content_chars": raw_content_chars,
        "mapped_content_chars": mapped_content_chars,
        "content_expansion_chars": mapped_content_chars - raw_content_chars,
        "content_expansion_ratio_mean": (sum(expansion_ratios) / len(expansion_ratios) if expansion_ratios else None),
        "content_expansion_ratio_max": max(expansion_ratios) if expansion_ratios else None,
        "payload_bytes": payload_bytes,
        "task_records": task_records,
        "failures": failures,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_digest(trajectory: dict[str, Any]) -> str:
    return _digest(trajectory)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _path_counts(signals: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        path = signal.get("evolvable_path")
        if isinstance(path, str) and path:
            counts[path] = counts.get(path, 0) + 1
    return counts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _install_agent(state_bench_dir: Path) -> None:
    target_dir = state_bench_dir / "agents"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(AGENT_SOURCE, target_dir / "mindmemos_agent.py")
    shutil.copyfile(MAPPING_SOURCE, target_dir / "trajectory_mapping.py")


def _collect_round(
    directory: Path,
    task_ids: tuple[str, ...],
    api_base: str,
    api_key: str,
    artifact_path: Path | None = None,
    user_id_prefix: str | None = None,
    *,
    retry_failed: bool = False,
) -> dict[str, Any]:
    cached_events: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    if artifact_path is not None and artifact_path.exists():
        try:
            cached = _load(artifact_path)
            cached_events = {
                str(item.get("task_id")): item
                for item in cached.get("collected_events", [])
                if item.get("task_id") and (not retry_failed or item.get("status") == "ok")
            }
        except Exception as exc:
            failures.append(
                {
                    "stage": "feedback_artifact",
                    "status": "failed",
                    "error": str(exc),
                    "path": str(artifact_path),
                }
            )

    files: dict[str, Path] = {}
    expected = set(task_ids)
    for path in _trajectory_files(directory):
        try:
            trajectory = _load(path)
            task_id = str(trajectory.get("task_id") or path.stem)
        except Exception as exc:
            failures.append(
                {
                    "stage": "trajectory",
                    "status": "failed",
                    "task_id": path.stem,
                    "error": str(exc),
                    "path": str(path),
                }
            )
            continue
        if task_id not in expected:
            failures.append(
                {
                    "stage": "trajectory",
                    "status": "failed",
                    "task_id": task_id,
                    "error": "unexpected trajectory file",
                    "path": str(path),
                }
            )
            continue
        if task_id in files:
            failures.append(
                {
                    "stage": "trajectory",
                    "status": "failed",
                    "task_id": task_id,
                    "error": "duplicate trajectory task_id",
                    "path": str(path),
                }
            )
            continue
        files[task_id] = path

    events: list[dict[str, Any]] = []
    total = 0
    paths: dict[str, int] = {}
    for task_id in task_ids:
        path = files.get(task_id)
        if path is None:
            event = {
                "task_id": task_id,
                "status": "failed",
                "signal_count": 0,
                "signals": [],
                "trajectory_sha256": None,
                "error": "trajectory missing",
            }
            failures.append(
                {
                    "stage": "feedback",
                    "status": "failed",
                    "task_id": task_id,
                    "error": "trajectory missing",
                }
            )
        else:
            try:
                trajectory = _load(path)
                digest = _trajectory_digest(trajectory)
                if task_id in cached_events:
                    event = cached_events[task_id]
                    if event.get("trajectory_sha256") != digest:
                        event = {
                            "task_id": task_id,
                            "status": "failed",
                            "signal_count": 0,
                            "signals": [],
                            "trajectory_sha256": digest,
                            "error": "feedback artifact hash mismatch",
                        }
                        failures.append(
                            {
                                "stage": "feedback_artifact",
                                "status": "failed",
                                "task_id": task_id,
                                "error": "trajectory hash differs from cached feedback artifact",
                            }
                        )
                else:
                    event = {
                        "trajectory_sha256": digest,
                        **_collect(api_base, api_key, trajectory, user_id_prefix),
                    }
                    cached_events[task_id] = event
            except Exception as exc:
                event = {
                    "task_id": task_id,
                    "status": "failed",
                    "signal_count": 0,
                    "signals": [],
                    "trajectory_sha256": None,
                    "error": str(exc),
                }
                failures.append(
                    {
                        "stage": "feedback",
                        "status": "failed",
                        "task_id": task_id,
                        "error": str(exc),
                    }
                )
        event.setdefault("task_id", task_id)
        if event.get("status") == "failed" and not any(
            item.get("task_id") == task_id and item.get("stage") == "feedback" for item in failures
        ):
            failures.append(
                {
                    "stage": "feedback",
                    "status": "failed",
                    "task_id": task_id,
                    "error": event.get("error") or "feedback collection failed",
                }
            )
        total += int(event.get("signal_count") or 0)
        signals = event.get("signals") or []
        if isinstance(signals, list):
            for path_name, count in _path_counts([item for item in signals if isinstance(item, dict)]).items():
                paths[path_name] = paths.get(path_name, 0) + count
        events.append(event)
        if artifact_path is not None:
            _write_json(
                artifact_path,
                {
                    "collected_events": events,
                    "signals_this_round": total,
                    "signal_evolvable_path_counts": paths,
                    "failures": failures,
                },
            )
    result = {
        "collected_events": events,
        "collected_event_count": sum(item.get("status") == "ok" for item in events),
        "expected_event_count": len(task_ids),
        "signals_this_round": total,
        "signal_evolvable_path_counts": paths,
        "failures": failures,
    }
    if artifact_path is not None:
        _write_json(artifact_path, result)
    return result


def _eval_metrics(trajectory: dict[str, Any], run_index: int) -> dict[str, Any]:
    return {
        "task_id": trajectory.get("task_id"),
        "run_index": run_index,
        "state_requirements_met": trajectory.get("state_requirements_met"),
        "task_requirements_met": trajectory.get("task_requirements_met"),
        "task_completion_pass": trajectory.get("task_completion_pass"),
        "ux_score": trajectory.get("ux_score"),
        "turns": trajectory.get("turns"),
        "tool_calls": trajectory.get("tool_calls"),
        "tool_errors": trajectory.get("tool_errors"),
        "cost_usd": trajectory.get("cost_usd"),
    }


def _summarize_eval(directory: Path, task_ids: tuple[str, ...], num_runs: int) -> dict[str, Any]:
    expected = set(task_ids)
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_index in range(1, num_runs + 1):
        files = _trajectory_files(directory, run_index)
        seen = {path.stem for path in files}
        for task_id in sorted(expected - seen):
            failures.append(
                {
                    "stage": "score",
                    "status": "failed",
                    "task_id": task_id,
                    "run_index": run_index,
                    "error": "evaluation trajectory missing",
                }
            )
        for path in files:
            if path.stem not in expected:
                failures.append(
                    {
                        "stage": "score",
                        "status": "failed",
                        "task_id": path.stem,
                        "run_index": run_index,
                        "error": "unexpected evaluation trajectory",
                    }
                )
                continue
            try:
                trajectory = _load(path)
                if not isinstance(trajectory, dict):
                    raise TypeError("evaluation trajectory JSON must be an object")
            except Exception as exc:
                failures.append(
                    {
                        "stage": "score",
                        "status": "failed",
                        "task_id": path.stem,
                        "run_index": run_index,
                        "error": str(exc),
                    }
                )
                continue
            item = _eval_metrics(trajectory, run_index)
            metrics.append(item)
            if item["task_completion_pass"] is None:
                failures.append(
                    {
                        "stage": "score",
                        "status": "failed",
                        "task_id": str(item.get("task_id") or path.stem),
                        "run_index": run_index,
                        "error": "task_completion_pass is missing",
                    }
                )
    scored = [item for item in metrics if item["task_completion_pass"] is not None]
    passed = [bool(item["task_completion_pass"]) for item in scored]
    by_task: dict[str, list[bool]] = {}
    for item in scored:
        by_task.setdefault(str(item["task_id"]), []).append(bool(item["task_completion_pass"]))

    def numeric(field: str) -> list[float]:
        values: list[float] = []
        for item in metrics:
            value = item.get(field)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                failures.append(
                    {
                        "stage": "score",
                        "status": "failed",
                        "task_id": str(item.get("task_id") or ""),
                        "run_index": item.get("run_index"),
                        "error": f"invalid numeric score field: {field}",
                        "value": value,
                    }
                )
        return values

    ux = numeric("ux_score")

    def rate(field: str) -> float | None:
        values = [bool(item[field]) for item in metrics if item.get(field) is not None]
        return sum(values) / len(values) if values else None

    def mean(field: str) -> float | None:
        values = numeric(field)
        return sum(values) / len(values) if values else None

    costs = numeric("cost_usd")
    pass_at_5 = sum(all(values) for values in by_task.values()) / len(by_task) if num_runs == 5 and by_task else None
    return {
        "task_ids": sorted(expected),
        "num_runs": num_runs,
        "per_task": metrics,
        "task_completion_pass_at_1": sum(passed) / len(passed) if passed else None,
        "task_completion_pass_at_5": pass_at_5,
        "task_completion_pass^5": pass_at_5,
        "ux_score_mean": sum(ux) / len(ux) if ux else None,
        "state_requirements_pass_rate": rate("state_requirements_met"),
        "task_requirements_pass_rate": rate("task_requirements_met"),
        "turns_mean": mean("turns"),
        "tool_calls_mean": mean("tool_calls"),
        "tool_errors_mean": mean("tool_errors"),
        "cost_usd_total": sum(costs) if costs else None,
        "cost_usd_mean": sum(costs) / len(costs) if costs else None,
        "scored": len(scored),
        "expected": len(expected) * num_runs,
        "complete": not failures,
        "failures": failures,
    }


def _run_eval(
    *,
    state_bench_dir: Path,
    domain: str,
    task_ids: tuple[str, ...],
    output_root: Path,
    checkpoint: int,
    num_runs: int,
    agent_class: str,
    agent_model_name: str,
    reasoning_level: str | None,
    num_workers: int,
    api_base: str,
    api_key: str,
    user_id_prefix: str | None = None,
    audit_memory: bool = False,
) -> dict[str, Any]:
    directory = output_root / "eval" / f"checkpoint{checkpoint:02d}"
    memory_before = None
    if audit_memory:
        # Capture the state before the first evaluation request.  Taking both
        # snapshots after _run_batch would only prove that the state was stable
        # between two post-hoc reads, not that eval itself was read-only.
        memory_before = _memory_snapshot(
            api_base=api_base,
            api_key=api_key,
            output_root=output_root / "eval_state_before",
            checkpoint=checkpoint,
        )
    batch = _run_batch(
        state_bench_dir,
        domain=domain,
        task_ids=list(task_ids),
        output_dir=directory,
        log_path=output_root / "logs" / f"eval_checkpoint{checkpoint:02d}.log",
        agent_class=agent_class,
        agent_model_name=agent_model_name,
        reasoning_level=reasoning_level,
        num_workers=num_workers,
        no_score=False,
        role="eval",
        api_base=api_base,
        api_key=api_key,
        num_runs=num_runs,
        user_id_prefix=user_id_prefix,
    )
    summary = _summarize_eval(directory, task_ids, num_runs)
    summary["batch"] = batch
    if audit_memory:
        # The batch itself is read-only for the MindMemOS adapter; the second
        # snapshot is used to make that invariant auditable.
        memory_after = _memory_snapshot(
            api_base=api_base,
            api_key=api_key,
            output_root=output_root / "eval_state_after",
            checkpoint=checkpoint,
        )
        summary["memory_before"] = memory_before
        summary["memory_after"] = memory_after
        if memory_before.get("available") and memory_after.get("available"):
            summary["memory_unchanged"] = memory_before.get("memory_fingerprint") == memory_after.get(
                "memory_fingerprint"
            )
            if not summary["memory_unchanged"]:
                summary.setdefault("failures", []).append(
                    {
                        "stage": "eval_mutation",
                        "status": "failed",
                        "error": "memory fingerprint changed during read-only evaluation",
                    }
                )
        else:
            summary["memory_unchanged"] = None
    else:
        summary["memory_unchanged"] = None
    summary["failures"] = [*(batch.get("failures") or []), *(summary.get("failures") or [])]
    summary["complete"] = not summary["failures"]
    return summary


def _round_report(
    plan: RoundPlan,
    directory: Path,
    feedback: dict[str, Any],
    evolution: dict[str, Any],
    replay_add: list[dict[str, Any]] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = {path.stem: path for path in _trajectory_files(directory)}
    failures = [*(runtime or {}).get("failures", []), *(feedback.get("failures") or [])]
    if feedback.get("expected_event_count") and feedback.get("collected_event_count") != feedback.get(
        "expected_event_count"
    ):
        failures.append(
            {
                "stage": "feedback",
                "status": "failed",
                "error": "not every training trajectory produced a feedback event",
                "expected_event_count": feedback.get("expected_event_count"),
                "collected_event_count": feedback.get("collected_event_count"),
            }
        )
    if (
        evolution.get("status") == "ok"
        and evolution.get("event_selection") == "unconsumed"
        and evolution.get("selected_event_count") != evolution.get("consumed_event_count")
    ):
        failures.append(
            {
                "stage": "evolve",
                "status": "failed",
                "error": "not every selected feedback event was consumed",
                "selected_event_count": evolution.get("selected_event_count"),
                "consumed_event_count": evolution.get("consumed_event_count"),
            }
        )
    if evolution.get("status") == "failed":
        failures.append(
            {
                "stage": "evolve",
                "status": "failed",
                "error": evolution.get("error") or evolution.get("message") or "self-evolve failed",
            }
        )
    trajectory_hashes: list[dict[str, Any]] = []
    for task_id in plan.train_task_ids:
        path = files.get(task_id)
        if path is None:
            failures.append(
                {
                    "stage": "trajectory",
                    "status": "failed",
                    "task_id": task_id,
                    "error": "trajectory file missing while writing round report",
                }
            )
            continue
        try:
            digest = _trajectory_digest(_load(path))
        except Exception as exc:
            failures.append(
                {
                    "stage": "trajectory",
                    "status": "failed",
                    "task_id": task_id,
                    "error": str(exc),
                }
            )
            continue
        trajectory_hashes.append({"task_id": task_id, "sha256": digest})
    return {
        "round_index": plan.round_index,
        "train_task_ids": list(plan.train_task_ids),
        "train_trajectory_hashes": trajectory_hashes,
        "replay_add": replay_add or [],
        "runtime_counts": runtime or {},
        "failures": failures,
        **feedback,
        "evolution": evolution,
    }


def _write_round(report_dir: Path, report: dict[str, Any]) -> None:
    _write_json(report_dir / f"round_{report['round_index']:02d}.json", report)
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "rounds.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")


def _evolution_result(
    feedback: dict[str, Any],
    *,
    enabled: bool,
    api_base: str,
    api_key: str,
    artifact_path: Path | None = None,
    event_selection: str = "all",
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if artifact_path is not None and artifact_path.exists():
        try:
            cached = _load(artifact_path)
            if cached.get("status") != "failed":
                return cached
        except Exception:
            pass
    if enabled:
        result = _evolve(
            api_base,
            api_key,
            event_selection=event_selection,
            idempotency_key=idempotency_key,
        )
    else:
        result = {
            "status": "skipped",
            "evolved": False,
            "version": None,
            "config_unchanged": True,
            "changes": [],
            "signal_count": int(feedback.get("signals_this_round") or 0),
            "selected_event_count": 0,
            "consumed_event_count": 0,
            "cost_usd": 0.0,
            "message": "baseline: self-evolve skipped",
            "event_selection": event_selection,
            "idempotency_key": idempotency_key,
        }
    if artifact_path is not None:
        _write_json(artifact_path, result)
    return result


def _default_project_id(experiment_id: str | None, branch: str) -> str | None:
    if not experiment_id:
        return None
    # Experiment/group IDs retain the STATE-Bench domain spelling (for example
    # customer_support), while the API-key/project namespace uses the
    # documented kebab-case project IDs (customer-support).
    project_namespace = experiment_id.replace("_", "-")
    return f"proj-{project_namespace}-{branch}"


def _current_config_state(*, project_id: str | None, checkpoint: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "project_id": project_id,
        "state_path": None,
        "available": False,
        "config_hash": None,
        "normalized_config": None,
    }
    if not project_id:
        result["error"] = "project_id not supplied; config snapshot unavailable"
    else:
        path = REPO_ROOT / "config" / "evolved" / project_id / "current.json"
        result["state_path"] = str(path)
        try:
            state = _load(path)
            normalized = {
                "add_config": state.get("add_config") or {},
                "search_config": state.get("search_config") or {},
            }
            result.update(
                {
                    "available": True,
                    "version": state.get("version"),
                    "state": state,
                    "normalized_config": normalized,
                    "config_hash": _digest(normalized),
                }
            )
        except Exception as exc:
            result.update({"error_type": type(exc).__name__, "error": str(exc)})
    return result


def _config_snapshot(*, project_id: str | None, output_root: Path, checkpoint: int) -> dict[str, Any]:
    result = _current_config_state(project_id=project_id, checkpoint=checkpoint)
    _write_json(output_root / "config_snapshots" / f"checkpoint{checkpoint:02d}.json", result)
    return result


def _memory_snapshot(*, api_base: str, api_key: str, output_root: Path, checkpoint: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "available": False,
        "memory_count": None,
        "memory_fingerprint": None,
    }
    try:
        body = _post(api_base, api_key, "/v1/memory/get", {"top_k": 100000})
        memories = (body.get("data") or {}).get("memories") or []
        items = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id") or item.get("id") or ""
            content = item.get("memory") or item.get("content") or ""
            items.append({"id": str(memory_id), "content": str(content)})
        items.sort(key=lambda item: (item["id"], item["content"]))
        result.update(
            {
                "available": True,
                "memory_count": len(items),
                "memory_fingerprint": _digest(items),
                "items": items,
            }
        )
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
    _write_json(output_root / "memory_snapshots" / f"checkpoint{checkpoint:02d}.json", result)
    return result


def _capture_checkpoint_state(
    *, project_id: str | None, output_root: Path, checkpoint: int, api_base: str, api_key: str
) -> dict[str, Any]:
    config = _config_snapshot(project_id=project_id, output_root=output_root, checkpoint=checkpoint)
    memory = _memory_snapshot(api_base=api_base, api_key=api_key, output_root=output_root, checkpoint=checkpoint)
    return {"config": config, "memory": memory}


def _apply_evolution_changes(
    initial_config: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the expected config from persisted, successful evolution artifacts."""
    result = copy.deepcopy(initial_config)
    for artifact in artifacts:
        for change in artifact.get("changes") or []:
            path = str(change.get("path") or "")
            parts = path.split(".")
            if not parts or parts[0] not in {"add_config", "search_config"}:
                raise RuntimeError(f"unsupported evolution change path in resume artifact: {path!r}")
            target: dict[str, Any] = result
            for part in parts[:-1]:
                child = target.get(part)
                if not isinstance(child, dict):
                    child = {}
                    target[part] = child
                target = child
            target[parts[-1]] = copy.deepcopy(change.get("after"))
    return result


def _validate_online_resume_state(
    *,
    root: Path,
    baseline_project_id: str,
    evolution_project_id: str,
) -> dict[str, Any] | None:
    """Validate current project configs against the rounds already committed on disk."""
    round_paths = sorted((root / "evolution" / "reports").glob("round_*.json"))
    completed_rounds = [
        int(path.stem.rsplit("_", 1)[-1]) for path in round_paths if _load(path).get("round_complete", True)
    ]
    if not completed_rounds:
        return None

    baseline = _current_config_state(project_id=baseline_project_id)
    evolution = _current_config_state(project_id=evolution_project_id)
    if not baseline.get("available") or not evolution.get("available"):
        raise RuntimeError("resume requires readable baseline and evolution project configs")
    if int(baseline.get("version") or 0) != 1:
        raise RuntimeError(f"resume baseline config must remain at version 1, got {baseline.get('version')!r}")

    artifacts: list[dict[str, Any]] = []
    expected_version = 1
    for round_index in completed_rounds:
        artifact_path = root / "evolution" / "reports" / f"evolution_round{round_index:02d}.json"
        if not artifact_path.exists():
            raise RuntimeError(f"missing evolution artifact for completed round {round_index}")
        artifact = _load(artifact_path)
        if artifact.get("status") != "ok":
            raise RuntimeError(f"evolution artifact for completed round {round_index} is not successful")
        artifacts.append(artifact)
        artifact_version = artifact.get("version")
        if artifact.get("evolved") and artifact_version is not None:
            expected_version = int(artifact_version)

    initial_config = baseline.get("normalized_config") or {}
    expected_config = _apply_evolution_changes(initial_config, artifacts)
    expected_hash = _digest(expected_config)
    actual_hash = evolution.get("config_hash")
    actual_version = int(evolution.get("version") or 0)
    if actual_hash != expected_hash or actual_version != expected_version:
        raise RuntimeError(
            "resume evolution config does not match persisted evolution artifacts: "
            f"expected version/hash {expected_version}/{expected_hash}, "
            f"got {actual_version}/{actual_hash}"
        )
    return {
        "resumed": True,
        "completed_evolution_rounds": completed_rounds,
        "initial_config_hash": baseline.get("config_hash"),
        "expected_evolution_version": expected_version,
        "expected_evolution_hash": expected_hash,
        "baseline": baseline,
        "evolution": evolution,
    }


def _repair_checkpoint_zero_after_resume_probe(
    *,
    root: Path,
    evolution_project_id: str,
    initial_config: dict[str, Any],
    initial_hash: str,
) -> None:
    """Repair checkpoint 0 if an older runner overwrote it while probing a resume."""
    config_path = root / "evolution" / "config_snapshots" / "checkpoint00.json"
    if config_path.exists() and _load(config_path).get("config_hash") != initial_hash:
        state = {
            "project_id": evolution_project_id,
            "mode": "feedback_evo",
            "version": 1,
            "is_current": False,
            **copy.deepcopy(initial_config),
        }
        _write_json(
            config_path,
            {
                "checkpoint": 0,
                "project_id": evolution_project_id,
                "state_path": None,
                "available": True,
                "config_hash": initial_hash,
                "normalized_config": copy.deepcopy(initial_config),
                "version": 1,
                "state": state,
                "restored_from_resume_probe": True,
            },
        )

    memory_path = root / "evolution" / "memory_snapshots" / "checkpoint00.json"
    original_memory_path = root / "evolution" / "eval_state_before" / "memory_snapshots" / "checkpoint00.json"
    if memory_path.exists() and original_memory_path.exists():
        current = _load(memory_path)
        original = _load(original_memory_path)
        if current.get("memory_fingerprint") != original.get("memory_fingerprint"):
            restored = copy.deepcopy(original)
            restored["restored_from_resume_probe"] = True
            _write_json(memory_path, restored)


def _skipped_feedback(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "feedback_skipped": True,
        "feedback_skipped_reason": reason,
        "collected_events": [],
        "collected_event_count": 0,
        "expected_event_count": 0,
        "signals_this_round": 0,
        "signal_evolvable_path_counts": {},
        "failures": [],
    }


def _run_online_branch(
    *,
    branch: str,
    schedule: FeedbackEvoSchedule,
    state_bench_dir: Path,
    domain: str,
    output_root: Path,
    agent_class: str,
    agent_model_name: str,
    reasoning_level: str | None,
    num_workers: int,
    no_score: bool,
    enabled: bool,
    api_base: str,
    api_key: str,
    checkpoints: set[int],
    checkpoint_runs: int,
    final_runs: int,
    train_workers: int | None = None,
    eval_workers: int | None = None,
    collect_enabled: bool = True,
    evolve_enabled: bool | None = None,
    experiment_id: str | None = None,
    project_id: str | None = None,
    user_id_prefix: str | None = None,
    capture_state: bool = False,
) -> dict[str, Any]:
    if evolve_enabled is None:
        evolve_enabled = enabled
    train_workers = train_workers or num_workers
    eval_workers = eval_workers or num_workers
    project_id = project_id or _default_project_id(experiment_id, branch)
    user_id_prefix = (
        user_id_prefix if user_id_prefix is not None else (f"{experiment_id}::{branch}" if experiment_id else None)
    )
    reports = output_root / "reports"
    evaluations: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    max_checkpoint = max(checkpoints)

    def evaluate(checkpoint: int, runs: int) -> None:
        eval_dir = output_root / "eval" / f"checkpoint{checkpoint:02d}"
        config_path = output_root / "config_snapshots" / f"checkpoint{checkpoint:02d}.json"
        memory_path = output_root / "memory_snapshots" / f"checkpoint{checkpoint:02d}.json"
        preserve_snapshot = (
            capture_state
            and _batch_complete(eval_dir, list(schedule.eval_task_ids), runs)
            and config_path.exists()
            and memory_path.exists()
        )
        evaluations[str(checkpoint)] = _run_eval(
            state_bench_dir=state_bench_dir,
            domain=domain,
            task_ids=schedule.eval_task_ids,
            output_root=output_root,
            checkpoint=checkpoint,
            num_runs=runs,
            agent_class=agent_class,
            agent_model_name=agent_model_name,
            reasoning_level=reasoning_level,
            num_workers=eval_workers,
            api_base=api_base,
            api_key=api_key,
            user_id_prefix=user_id_prefix,
            audit_memory=capture_state and not preserve_snapshot,
        )
        if capture_state:
            if preserve_snapshot:
                snapshots[str(checkpoint)] = {
                    "config": _load(config_path),
                    "memory": _load(memory_path),
                    "reused": True,
                }
            else:
                snapshots[str(checkpoint)] = _capture_checkpoint_state(
                    project_id=project_id,
                    output_root=output_root,
                    checkpoint=checkpoint,
                    api_base=api_base,
                    api_key=api_key,
                )

    if 0 in checkpoints:
        evaluate(0, final_runs)
    round_reports: list[dict[str, Any]] = []
    for plan in schedule.rounds:
        label = f"round{plan.round_index:02d}"
        train_dir = output_root / "train" / label
        round_path = reports / f"round_{plan.round_index:02d}.json"
        if round_path.exists():
            round_reports.append(_load(round_path))
            if plan.round_index in checkpoints:
                evaluate(plan.round_index, final_runs if plan.round_index == max_checkpoint else checkpoint_runs)
            continue
        runtime = _run_batch(
            state_bench_dir,
            domain=domain,
            task_ids=list(plan.train_task_ids),
            output_dir=train_dir,
            log_path=output_root / "logs" / f"train_{label}.log",
            agent_class=agent_class,
            agent_model_name=agent_model_name,
            reasoning_level=reasoning_level,
            num_workers=train_workers,
            no_score=no_score,
            role="train",
            api_base=api_base,
            api_key=api_key,
            user_id_prefix=user_id_prefix,
        )
        feedback = (
            _collect_round(
                train_dir,
                plan.train_task_ids,
                api_base,
                api_key,
                reports / f"feedback_round{plan.round_index:02d}.json",
                user_id_prefix,
            )
            if collect_enabled
            else _skipped_feedback("frozen baseline does not collect feedback")
        )
        evolution = _evolution_result(
            feedback,
            enabled=bool(evolve_enabled),
            api_base=api_base,
            api_key=api_key,
            artifact_path=reports / f"evolution_round{plan.round_index:02d}.json",
            event_selection="unconsumed" if evolve_enabled else "all",
            idempotency_key=(
                f"{experiment_id}:evolution:round-{plan.round_index:02d}" if experiment_id and evolve_enabled else None
            ),
        )
        report = _round_report(plan, train_dir, feedback, evolution, runtime=runtime)
        report.update(
            {
                "branch": branch,
                "project_id": project_id,
                "collect_enabled": collect_enabled,
                "evolve_enabled": bool(evolve_enabled),
                "user_id_prefix": user_id_prefix,
            }
        )
        _write_round(reports, report)
        round_reports.append(report)
        if plan.round_index in checkpoints:
            evaluate(plan.round_index, final_runs if plan.round_index == max_checkpoint else checkpoint_runs)
    summary = {
        "branch": branch,
        "project_id": project_id,
        "user_id_prefix": user_id_prefix,
        "collect_enabled": collect_enabled,
        "evolve_enabled": bool(evolve_enabled),
        "rounds": round_reports,
        "checkpoints": evaluations,
        "checkpoint_snapshots": snapshots,
        "failures": [
            *[failure for report in round_reports for failure in report.get("failures", [])],
            *[failure for evaluation in evaluations.values() for failure in evaluation.get("failures", [])],
            *[
                {"stage": "config_snapshot", **item["config"]}
                for item in snapshots.values()
                if not item.get("config", {}).get("available", False)
            ],
            *[
                {"stage": "memory_snapshot", **item["memory"]}
                for item in snapshots.values()
                if not item.get("memory", {}).get("available", False)
            ],
        ],
    }
    _write_json(reports / "summary.json", summary)
    return summary


def _official_trajectory(path: Path, task_id: str) -> dict[str, Any]:
    trajectory = _load(path)
    trajectory.setdefault("task_id", task_id)
    if not trajectory.get("conversation"):
        raise RuntimeError(f"official trajectory has no conversation: {path}")
    return trajectory


def _run_official_baseline(
    *,
    schedule: FeedbackEvoSchedule,
    official_trajectory_dir: Path,
    state_bench_dir: Path,
    domain: str,
    output_root: Path,
    agent_class: str,
    agent_model_name: str,
    reasoning_level: str | None,
    num_workers: int,
    api_base: str,
    api_key: str,
    final_runs: int,
) -> dict[str, Any]:
    """Replay official fixed train trajectories without collect or evolution."""

    reports = output_root / "reports"
    expected_files = {task_id: official_trajectory_dir / f"{task_id}.json" for task_id in schedule.train_task_ids}
    missing = [task_id for task_id, path in expected_files.items() if not path.exists()]
    source_failures = [
        {
            "stage": "official_trajectory",
            "status": "failed",
            "task_id": task_id,
            "error": f"official trajectory file missing: {expected_files[task_id]}",
        }
        for task_id in missing
    ]

    round_reports: list[dict[str, Any]] = []
    for plan in schedule.rounds:
        round_path = reports / f"round_{plan.round_index:02d}.json"
        if round_path.exists():
            round_reports.append(_load(round_path))
            continue
        add_artifact = reports / f"official_add_round{plan.round_index:02d}.json"
        cached_add: dict[str, dict[str, Any]] = {}
        artifact_failures: list[dict[str, Any]] = []
        if add_artifact.exists():
            try:
                cached_add = {
                    str(item.get("task_id")): item
                    for item in (_load(add_artifact).get("adds") or [])
                    if item.get("task_id")
                }
            except Exception as exc:
                artifact_failures.append(
                    {
                        "stage": "add_artifact",
                        "status": "failed",
                        "error": str(exc),
                        "path": str(add_artifact),
                    }
                )
        replay_add: list[dict[str, Any]] = []
        trajectory_hashes: list[dict[str, str]] = []
        for task_id in plan.train_task_ids:
            path = expected_files[task_id]
            if not path.exists():
                result = {
                    "task_id": task_id,
                    "status": "failed",
                    "payload_sha256": None,
                    "error": f"official trajectory file missing: {path}",
                }
                replay_add.append(result)
                _write_json(add_artifact, {"adds": replay_add})
                continue
            try:
                trajectory = _official_trajectory(path, task_id)
                trajectory_hashes.append(
                    {
                        "task_id": task_id,
                        "sha256": _trajectory_digest(trajectory),
                    }
                )
                expected_payload_hash = _digest(_add_payload(trajectory))
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "status": "failed",
                    "payload_sha256": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                replay_add.append(result)
                _write_json(add_artifact, {"adds": replay_add})
                continue
            cached = cached_add.get(task_id)
            if cached is not None and cached.get("payload_sha256") != expected_payload_hash:
                result = {
                    "task_id": task_id,
                    "status": "failed",
                    "payload_sha256": expected_payload_hash,
                    "error": "official add artifact hash mismatch",
                }
            else:
                result = cached if cached is not None else _replay_add(api_base, api_key, trajectory)
            replay_add.append(result)
            _write_json(add_artifact, {"adds": replay_add})
        add_failures = [{"stage": "add", **item} for item in replay_add if item.get("status") != "ok"]
        report = {
            "round_index": plan.round_index,
            "train_task_ids": list(plan.train_task_ids),
            "source": "official_fixed_trajectories",
            "train_trajectory_hashes": trajectory_hashes,
            "replay_add": replay_add,
            "runtime_counts": {
                "add_ok": sum(item.get("status") == "ok" for item in replay_add),
                "add_failed": sum(item.get("status") != "ok" for item in replay_add),
                "payload_bytes": sum(int(item.get("payload_bytes") or 0) for item in replay_add),
                "raw_content_chars": sum(int(item.get("raw_content_chars") or 0) for item in replay_add),
                "mapped_content_chars": sum(int(item.get("mapped_content_chars") or 0) for item in replay_add),
                "content_expansion_chars": sum(int(item.get("content_expansion_chars") or 0) for item in replay_add),
                "content_expansion_ratio_mean": (
                    sum(
                        float(item["content_expansion_ratio"])
                        for item in replay_add
                        if item.get("content_expansion_ratio") is not None
                    )
                    / sum(item.get("content_expansion_ratio") is not None for item in replay_add)
                    if any(item.get("content_expansion_ratio") is not None for item in replay_add)
                    else None
                ),
                "search_results": 0,
                "failures": [*source_failures, *artifact_failures, *add_failures],
            },
            "failures": [*source_failures, *artifact_failures, *add_failures],
            "collected_events": [],
            "signals_this_round": 0,
            "signal_evolvable_path_counts": {},
            "feedback_skipped_reason": "fixed official trajectories are not online feedback",
            "evolution": {
                "evolved": False,
                "version": None,
                "config_unchanged": True,
                "changes": [],
                "signal_count": 0,
                "cost_usd": 0.0,
                "message": "official baseline: collect and self-evolve skipped",
            },
        }
        _write_round(reports, report)
        round_reports.append(report)

    final_checkpoint = schedule.rounds[-1].round_index
    evaluation = _run_eval(
        state_bench_dir=state_bench_dir,
        domain=domain,
        task_ids=schedule.eval_task_ids,
        output_root=output_root,
        checkpoint=final_checkpoint,
        num_runs=final_runs,
        agent_class=agent_class,
        agent_model_name=agent_model_name,
        reasoning_level=reasoning_level,
        num_workers=num_workers,
        api_base=api_base,
        api_key=api_key,
    )
    summary = {
        "branch": "official-baseline",
        "source": "official_fixed_trajectories",
        "rounds": round_reports,
        "checkpoints": {str(final_checkpoint): evaluation},
        "failures": [
            *[failure for report in round_reports for failure in report.get("failures", [])],
            *evaluation.get("failures", []),
        ],
    }
    _write_json(reports / "summary.json", summary)
    return summary


def _paired_comparison(baseline: dict[str, Any], evolution: dict[str, Any]) -> dict[str, Any]:
    def final(summary: dict[str, Any]) -> dict[str, Any]:
        checkpoints = summary.get("checkpoints") or {}
        if not checkpoints:
            return {}
        key = str(max(int(item) for item in checkpoints))
        return checkpoints.get(key) or {}

    base = final(baseline)
    evo = final(evolution)

    def evolution_cost(summary: dict[str, Any]) -> float:
        return sum(float(item.get("evolution", {}).get("cost_usd") or 0) for item in summary.get("rounds", []))

    base_score = base.get("task_completion_pass_at_1")
    evo_score = evo.get("task_completion_pass_at_1")
    return {
        "baseline_final": {
            "pass_at_1": base_score,
            "pass_at_5": base.get("task_completion_pass_at_5"),
            "pass^5": base.get("task_completion_pass^5"),
            "ux_score_mean": base.get("ux_score_mean"),
            "state_requirements_pass_rate": base.get("state_requirements_pass_rate"),
            "task_requirements_pass_rate": base.get("task_requirements_pass_rate"),
            "turns_mean": base.get("turns_mean"),
            "tool_errors_mean": base.get("tool_errors_mean"),
            "cost_usd_total": base.get("cost_usd_total"),
            "evolution_cost_usd": evolution_cost(baseline),
        },
        "evolution_final": {
            "pass_at_1": evo_score,
            "pass_at_5": evo.get("task_completion_pass_at_5"),
            "pass^5": evo.get("task_completion_pass^5"),
            "ux_score_mean": evo.get("ux_score_mean"),
            "state_requirements_pass_rate": evo.get("state_requirements_pass_rate"),
            "task_requirements_pass_rate": evo.get("task_requirements_pass_rate"),
            "turns_mean": evo.get("turns_mean"),
            "tool_errors_mean": evo.get("tool_errors_mean"),
            "cost_usd_total": evo.get("cost_usd_total"),
            "evolution_cost_usd": evolution_cost(evolution),
        },
        "pass_at_1_delta": evo_score - base_score if evo_score is not None and base_score is not None else None,
    }


def _final_eval(summary: dict[str, Any]) -> dict[str, Any]:
    checkpoints = summary.get("checkpoints") or {}
    if not checkpoints:
        return {
            "per_task": [],
            "complete": False,
            "failures": [
                {
                    "stage": "score",
                    "status": "failed",
                    "error": "evaluation summary has no checkpoints",
                }
            ],
        }
    key = str(max(int(item) for item in checkpoints))
    return checkpoints[key]


def _paired_rows(
    baseline: dict[str, Any],
    evolution: dict[str, Any],
    *,
    domain: str,
    seed: int,
) -> list[dict[str, Any]]:
    base_rows = {
        (str(item.get("task_id")), int(item.get("run_index", 1))): item
        for item in _final_eval(baseline).get("per_task", [])
    }
    evo_rows = {
        (str(item.get("task_id")), int(item.get("run_index", 1))): item
        for item in _final_eval(evolution).get("per_task", [])
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base_rows) | set(evo_rows)):
        base = base_rows.get(key)
        evo = evo_rows.get(key)
        base_pass = base.get("task_completion_pass") if base else None
        evo_pass = evo.get("task_completion_pass") if evo else None
        pairing_status = (
            "ok"
            if base is not None and evo is not None
            else "missing_baseline"
            if base is None
            else "missing_evolution"
        )
        rows.append(
            {
                "domain": domain,
                "seed": seed,
                "task_id": key[0],
                "run_index": key[1],
                "pairing_status": pairing_status,
                "baseline": base,
                "evolution": evo,
                "pass_at_1_delta": (
                    int(bool(evo_pass)) - int(bool(base_pass))
                    if base_pass is not None and evo_pass is not None
                    else None
                ),
            }
        )
    return rows


def _bootstrap_task_stratified(
    rows: list[dict[str, Any]],
    *,
    reps: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    observed = [row["pass_at_1_delta"] for row in rows if row["pass_at_1_delta"] is not None]
    if not observed:
        return {"estimate": None, "ci95": [None, None], "reps": reps}
    by_task: dict[str, list[int]] = {}
    for row in rows:
        value = row.get("pass_at_1_delta")
        if value is not None:
            by_task.setdefault(str(row["task_id"]), []).append(int(value))
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(reps):
        sample = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        values = [value for task_id in sample for value in by_task[task_id]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    low = estimates[max(0, int(0.025 * len(estimates)) - 1)]
    high = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return {
        "estimate": sum(observed) / len(observed),
        "ci95": [low, high],
        "reps": reps,
        "strata": "task_id",
    }


def _write_paired_results(
    path: Path,
    baseline: dict[str, Any],
    evolution: dict[str, Any],
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    rows = _paired_rows(baseline, evolution, domain=domain, seed=seed)
    base_final = _final_eval(baseline)
    evo_final = _final_eval(evolution)
    pairing_failures = [
        *[item for item in rows if item.get("pairing_status") != "ok"],
        *[
            {
                "side": "baseline",
                **failure,
            }
            for failure in base_final.get("failures", [])
            if not base_final.get("per_task")
        ],
        *[
            {
                "side": "evolution",
                **failure,
            }
            for failure in evo_final.get("failures", [])
            if not evo_final.get("per_task")
        ],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "rows": len(rows),
        "bootstrap_pass_at_1_delta": _bootstrap_task_stratified(rows),
        "pairing_failures": pairing_failures,
        "pairing_complete": not pairing_failures,
        "path": str(path),
    }


def _parse_checkpoints(value: str, rounds: int) -> set[int]:
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    result.add(0)
    if any(item < 0 or item > rounds for item in result):
        raise ValueError(f"checkpoints must be between 0 and {rounds}")
    return result


def _parse_checkpoint_values(value: str, rounds: int) -> set[int]:
    "Parse a branch-specific checkpoint list without implicitly adding 0."
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not result or any(item < 0 or item > rounds for item in result):
        raise ValueError(f"checkpoints must be between 0 and {rounds}")
    return result


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _manifest(path: Path, args: argparse.Namespace, schedule: FeedbackEvoSchedule) -> None:
    if path.exists():
        existing = _load(path)
        if existing.get("trajectory_mapping_version") != TRAJECTORY_MAPPING_VERSION:
            raise RuntimeError(
                f"output manifest uses {existing.get('trajectory_mapping_version')!r}; "
                f"current runner uses {TRAJECTORY_MAPPING_VERSION!r}; choose a new output directory"
            )
        return
    elif path.parent.exists() and any(path.parent.iterdir()):
        raise RuntimeError(
            f"output directory exists without a compatible manifest: {path.parent}; choose a new output directory"
        )
    _write_json(
        path,
        {
            "protocol": args.protocol,
            "experiment_id": getattr(args, "experiment_id", None),
            "domain": args.domain,
            "seed": args.seed,
            "git_commit": _git_commit(),
            "runner_source_sha256": _file_sha256(Path(__file__)),
            "agent_source_sha256": _file_sha256(AGENT_SOURCE),
            "trajectory_mapping_version": TRAJECTORY_MAPPING_VERSION,
            "search_provenance_version": "statebench-search-provenance-v1",
            "add_provenance_version": "statebench-add-provenance-v1",
            "agent_model_name": args.agent_model_name,
            "agent_model_reasoning_level": args.agent_model_reasoning_level,
            "rounds": args.rounds,
            "train_per_round": args.train_per_round,
            "round_plans": [
                {"round_index": plan.round_index, "train_task_ids": list(plan.train_task_ids)}
                for plan in schedule.rounds
            ],
            "train_task_ids": list(schedule.train_task_ids),
            "eval_task_ids": list(schedule.eval_task_ids),
            "eval_checkpoints": sorted(_parse_checkpoints(args.eval_checkpoints, args.rounds)),
            "branch_order": args.branch_order,
        },
    )


def _schedule_from_args(args: argparse.Namespace) -> FeedbackEvoSchedule:
    split_path = args.state_bench_dir / "state_bench" / "domains" / args.domain / "splits" / "train_test.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    return build_schedule(
        list(splits.get("splits", {}).get("train", [])),
        list(splits.get("splits", {}).get("test", [])),
        rounds=args.rounds,
        train_per_round=args.train_per_round,
        seed=args.seed,
        eval_count=args.eval_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["customer_support", "shopping_assistant", "travel"])
    parser.add_argument(
        "--protocol",
        choices=["online-controlled", "official-baseline"],
        default="online-controlled",
    )
    parser.add_argument(
        "--branch-order",
        choices=["baseline-first", "evolution-first"],
        default="baseline-first",
        help="Execution order for the two online branches; both branches are always run.",
    )
    parser.add_argument("--state-bench-dir", type=Path, default=DEFAULT_STATE_BENCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--experiment-id",
        default=os.environ.get("STATEBENCH_EXPERIMENT_ID", ""),
        help="Namespace prefix for all MindMemOS user IDs in this run.",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--train-per-round", type=int, default=10)
    parser.add_argument("--eval-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--baseline-eval-checkpoints",
        default=None,
        help="Optional checkpoint list for the baseline branch; unlike --eval-checkpoints, 0 is not implicit.",
    )
    parser.add_argument(
        "--evolution-eval-checkpoints",
        default=None,
        help="Optional checkpoint list for the evolution branch; unlike --eval-checkpoints, 0 is not implicit.",
    )
    parser.add_argument("--checkpoint-runs", type=int, default=1)
    parser.add_argument("--final-runs", type=int, default=5)
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("MINDMEMOS_API_KEY", ""))
    parser.add_argument("--baseline-api-key", default=os.environ.get("MINDMEMOS_BASELINE_API_KEY", ""))
    parser.add_argument("--evolution-api-key", default=os.environ.get("MINDMEMOS_EVOLUTION_API_KEY", ""))
    parser.add_argument("--official-trajectory-dir", type=Path, default=None)
    parser.add_argument("--baseline-project-id", default=os.environ.get("MINDMEMOS_BASELINE_PROJECT_ID", ""))
    parser.add_argument("--evolution-project-id", default=os.environ.get("MINDMEMOS_EVOLUTION_PROJECT_ID", ""))
    parser.add_argument("--baseline-key-id", default=os.environ.get("MINDMEMOS_BASELINE_KEY_ID", ""))
    parser.add_argument("--evolution-key-id", default=os.environ.get("MINDMEMOS_EVOLUTION_KEY_ID", ""))
    parser.add_argument("--agent-class", default="MindMemOSAgent")
    parser.add_argument("--agent-model-name", required=True)
    parser.add_argument("--agent-model-reasoning-level", default=None)
    parser.add_argument("--num-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args(argv)
    args.experiment_id = str(args.experiment_id or "").strip()
    if args.experiment_id:
        os.environ["STATEBENCH_EXPERIMENT_ID"] = args.experiment_id
    else:
        os.environ.pop("STATEBENCH_EXPERIMENT_ID", None)
    if args.rounds != 10 and args.eval_checkpoints == DEFAULT_CHECKPOINTS:
        args.eval_checkpoints = f"0,{args.rounds}"
    args.state_bench_dir = args.state_bench_dir.resolve()
    if args.checkpoint_runs < 1 or args.final_runs < 1:
        parser.error("checkpoint and final runs must be >= 1")
    if args.train_workers < 1 or args.eval_workers < 1:
        parser.error("train-workers and eval-workers must be >= 1")
    if args.num_workers is not None:
        args.train_workers = args.num_workers
        args.eval_workers = args.num_workers
    else:
        args.num_workers = 4
    schedule = _schedule_from_args(args)
    checkpoints = _parse_checkpoints(args.eval_checkpoints, args.rounds)
    baseline_checkpoints = (
        _parse_checkpoint_values(args.baseline_eval_checkpoints, args.rounds)
        if args.baseline_eval_checkpoints is not None
        else checkpoints
    )
    evolution_checkpoints = (
        _parse_checkpoint_values(args.evolution_eval_checkpoints, args.rounds)
        if args.evolution_eval_checkpoints is not None
        else checkpoints
    )
    if args.protocol == "online-controlled":
        default_root = REPO_ROOT / "outputs" / "statebench" / args.domain / f"seed{args.seed}"
    else:
        default_root = REPO_ROOT / "outputs" / "statebench" / args.protocol / args.domain / f"seed{args.seed}"
    root = (args.output_dir or default_root).resolve()
    _install_agent(args.state_bench_dir)
    _manifest(root / "manifest.json", args, schedule)

    if args.protocol == "official-baseline":
        if not args.api_key:
            parser.error("official-baseline requires --api-key (or MINDMEMOS_API_KEY)")
        official_dir = (
            args.official_trajectory_dir or args.state_bench_dir / "datasets" / "train_task_trajectories" / args.domain
        ).resolve()
        official = _run_official_baseline(
            schedule=schedule,
            official_trajectory_dir=official_dir,
            state_bench_dir=args.state_bench_dir,
            domain=args.domain,
            output_root=root,
            agent_class=args.agent_class,
            agent_model_name=args.agent_model_name,
            reasoning_level=args.agent_model_reasoning_level,
            num_workers=args.num_workers,
            api_base=args.api_base,
            api_key=args.api_key,
            final_runs=args.final_runs,
        )
        _write_json(
            root / "comparison.json",
            {
                "protocol": args.protocol,
                "domain": args.domain,
                "seed": args.seed,
                "official_baseline": official,
            },
        )
        _write_json(
            root / "reports" / "summary.json",
            {
                "protocol": args.protocol,
                "domain": args.domain,
                "seed": args.seed,
                "official_baseline": official,
            },
        )
        return 0

    if args.protocol == "online-controlled":
        if not (args.baseline_api_key and args.evolution_api_key):
            parser.error("online-controlled requires baseline and evolution API keys")
        if args.baseline_api_key == args.evolution_api_key:
            parser.error("online-controlled requires independent baseline and evolution API keys")
        if not args.experiment_id:
            parser.error("online-controlled requires --experiment-id for memory isolation")
        baseline_project_id = args.baseline_project_id or _default_project_id(args.experiment_id, "baseline")
        evolution_project_id = args.evolution_project_id or _default_project_id(args.experiment_id, "evolution")
        baseline_root = root / "baseline"
        evolution_root = root / "evolution"
        try:
            resume_state = _validate_online_resume_state(
                root=root,
                baseline_project_id=baseline_project_id,
                evolution_project_id=evolution_project_id,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        if resume_state is None:
            baseline_initial = _capture_checkpoint_state(
                project_id=baseline_project_id,
                output_root=baseline_root,
                checkpoint=0,
                api_base=args.api_base,
                api_key=args.baseline_api_key,
            )
            evolution_initial = _capture_checkpoint_state(
                project_id=evolution_project_id,
                output_root=evolution_root,
                checkpoint=0,
                api_base=args.api_base,
                api_key=args.evolution_api_key,
            )
            baseline_hash = baseline_initial["config"].get("config_hash")
            evolution_hash = evolution_initial["config"].get("config_hash")
            if not baseline_hash or not evolution_hash or baseline_hash != evolution_hash:
                parser.error(
                    "baseline/evolution initial normalized add/search config hashes differ "
                    f"({baseline_hash!r} != {evolution_hash!r})"
                )
        else:
            baseline_hash = str(resume_state["initial_config_hash"])
            evolution_hash = baseline_hash
            initial_config = resume_state["baseline"].get("normalized_config") or {}
            _repair_checkpoint_zero_after_resume_probe(
                root=root,
                evolution_project_id=evolution_project_id,
                initial_config=initial_config,
                initial_hash=baseline_hash,
            )
            baseline_initial = {
                "config": _load(baseline_root / "config_snapshots" / "checkpoint00.json"),
                "memory": _load(baseline_root / "memory_snapshots" / "checkpoint00.json"),
            }
            evolution_initial = {
                "config": _load(evolution_root / "config_snapshots" / "checkpoint00.json"),
                "memory": _load(evolution_root / "memory_snapshots" / "checkpoint00.json"),
            }
        manifest = _load(root / "manifest.json")
        manifest.update(
            {
                "experiment_group_id": args.experiment_id,
                "baseline_project_id": baseline_project_id,
                "evolution_project_id": evolution_project_id,
                "key_ids": {
                    "baseline": args.baseline_key_id or None,
                    "evolution": args.evolution_key_id or None,
                },
                "baseline_user_id_prefix": f"{args.experiment_id}::baseline",
                "evolution_user_id_prefix": f"{args.experiment_id}::evolution",
                "initial_config_hash": baseline_hash,
                "initial_config_hashes": {"baseline": baseline_hash, "evolution": evolution_hash},
                "baseline_eval_checkpoints": sorted(baseline_checkpoints),
                "evolution_eval_checkpoints": sorted(evolution_checkpoints),
                "train_workers": args.train_workers,
                "eval_workers": args.eval_workers,
                "prompt_hashes": {
                    "agent_adapter": _file_sha256(AGENT_SOURCE),
                    "trajectory_mapping": _file_sha256(MAPPING_SOURCE),
                },
                "resume_validation": (
                    {
                        "resumed": True,
                        "completed_evolution_rounds": resume_state["completed_evolution_rounds"],
                        "expected_evolution_version": resume_state["expected_evolution_version"],
                        "expected_evolution_hash": resume_state["expected_evolution_hash"],
                    }
                    if resume_state is not None
                    else {"resumed": False}
                ),
            }
        )
        _write_json(root / "manifest.json", manifest)

        def run_branch(branch: str) -> dict[str, Any]:
            is_evolution = branch == "evolution"
            return _run_online_branch(
                branch=branch,
                schedule=schedule,
                state_bench_dir=args.state_bench_dir,
                domain=args.domain,
                output_root=evolution_root if is_evolution else baseline_root,
                agent_class=args.agent_class,
                agent_model_name=args.agent_model_name,
                reasoning_level=args.agent_model_reasoning_level,
                num_workers=args.eval_workers,
                no_score=args.no_score,
                enabled=is_evolution,
                api_base=args.api_base,
                api_key=args.evolution_api_key if is_evolution else args.baseline_api_key,
                checkpoints=evolution_checkpoints if is_evolution else baseline_checkpoints,
                checkpoint_runs=args.checkpoint_runs,
                final_runs=args.final_runs,
                train_workers=args.train_workers,
                eval_workers=args.eval_workers,
                collect_enabled=is_evolution,
                evolve_enabled=is_evolution,
                experiment_id=args.experiment_id,
                project_id=evolution_project_id if is_evolution else baseline_project_id,
                user_id_prefix=f"{args.experiment_id}::{branch}",
                capture_state=True,
            )

        # The order is configurable for a fresh run, but both branches always
        # execute and the final comparison keeps the branch labels stable.
        order = ("evolution", "baseline") if args.branch_order == "evolution-first" else ("baseline", "evolution")
        results = {branch: run_branch(branch) for branch in order}
        baseline = results["baseline"]
        evolution = results["evolution"]
        paired = _write_paired_results(
            root / "paired_results.jsonl", baseline, evolution, domain=args.domain, seed=args.seed
        )
        comparison = _paired_comparison(baseline, evolution)
        comparison.update(paired)
        comparison["bootstrap"] = {"pass_at_1_delta": paired["bootstrap_pass_at_1_delta"]}
        comparison.update(
            {"protocol": args.protocol, "domain": args.domain, "seed": args.seed, "initial_config_hash": baseline_hash}
        )
        _write_json(root / "comparison.json", comparison)
        _write_json(
            root / "reports" / "summary.json",
            {
                "protocol": args.protocol,
                "domain": args.domain,
                "seed": args.seed,
                "baseline": baseline,
                "evolution": evolution,
                "comparison": comparison,
                "initial_state": {"baseline": baseline_initial, "evolution": evolution_initial},
            },
        )
        return 0

    parser.error(f"unsupported STATE-Bench protocol: {args.protocol}")


if __name__ == "__main__":
    sys.exit(main())
