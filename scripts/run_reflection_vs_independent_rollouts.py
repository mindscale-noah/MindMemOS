#!/usr/bin/env python3
"""Paired first-success ablation: reflection retries vs blind independent retries.

The three benchmarks run serially. Within one benchmark, a single shared
operation semaphore bounds both rollouts and reflection calls. Each task has a
shared first rollout; only initially failed tasks branch into two retry arms:

* independent: retry the unchanged task;
* reflection: reflect on the immediately preceding failed trajectory, inject
  that reflection into the next retry, then retry.

Both arms stop at the first success or after ``--max-attempts`` logical
rollouts. The script is intentionally outside the MindMemOS Skill package and
does not register a new experiment method.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_EXPERIMENT_ROOT = REPO_ROOT / "scripts" / "mindmemos_skill"
if str(SCRIPT_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EXPERIMENT_ROOT))

from experiments.skill_evaluation import (  # noqa: E402
    build_agent,
    build_client,
    build_dataset,
    build_skill,
    environment_options,
    reward_of,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.config import RolloutConfig  # noqa: E402
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (  # noqa: E402
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.models import chat_content  # noqa: E402
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.rollout import (  # noqa: E402
    MappingAgentResolver,
    RegistryEnvFactory,
    RolloutScheduler,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.rollout.fixed_group import (  # noqa: E402
    stable_rollout_id,
)
from mindmemos_skill.llm import DatabaseLLMCallSink, llm_run_context  # noqa: E402
from mindmemos_skill.persistence import bootstrap_skill_database  # noqa: E402
from mindmemos_skill.typing import Skill, Task, Trajectory, compute_skill_content_hash  # noqa: E402

ARM_SHARED = "shared"
ARM_INDEPENDENT = "independent"
ARM_REFLECTION = "reflection"
RETRY_ARMS = (ARM_INDEPENDENT, ARM_REFLECTION)
EVENT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "paired-first-rollout-reflection-v1"

REFLECTION_SYSTEM_PROMPT = "You improve a failed attempt using only the supplied task, outcome, and trajectory."
REFLECTION_USER_PROMPT = """Diagnose the failed strategy and produce a concise, actionable plan for the next attempt.
Do not merely summarize. Do not assume access to a gold answer or evaluator explanation. Ground every claim in the trajectory.

## Task
{task}

## Outcome
fail

## Trajectory
{trajectory}

## Reflection and next plan
"""


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    env_ref: str
    data_root: str
    split_dir: str
    skill: str
    max_turns: int
    env_seed: int = 42
    shell_timeout: int = 120
    use_theorem: bool = False
    use_sketch: bool = False
    shuffle_choices: bool = True


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "alfworld": BenchmarkSpec(
        name="alfworld",
        env_ref="alfworld_bounded_history",
        data_root="data/mindmemos_skill/alfworld",
        split_dir="resources/mindmemos_skill/datasets/alfworld/splits",
        skill="resources/mindmemos_skill/skills/alfworld/SKILL.md",
        max_turns=50,
    ),
    "spreadsheetbench": BenchmarkSpec(
        name="spreadsheetbench",
        env_ref="spreadsheetbench",
        data_root="data/mindmemos_skill/spreadsheetbench",
        split_dir="resources/mindmemos_skill/datasets/spreadsheetbench/splits",
        skill="resources/mindmemos_skill/skills/spreadsheetbench/SKILL.md",
        max_turns=30,
    ),
    "livemath": BenchmarkSpec(
        name="livemath",
        env_ref="livemath",
        data_root="data/mindmemos_skill/livemath/raw",
        split_dir="resources/mindmemos_skill/datasets/livemath/splits",
        skill="resources/mindmemos_skill/skills/livemath/SKILL.md",
        max_turns=1,
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=tuple(BENCHMARKS),
        default=list(BENCHMARKS),
        help="benchmarks run serially in the listed order",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".skill.env"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-limit", type=int)

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--reflection-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--api-base")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--reflection-max-completion-tokens", type=int, default=4096)

    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trajectory-chars", type=int, default=60000)
    parser.add_argument("--max-reflection-chars", type=int, default=4000)
    parser.add_argument("--max-previous-answer-chars", type=int, default=8000)
    return parser.parse_args(argv)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid environment assignment at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"invalid environment name at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def apply_env_file(path: Path) -> None:
    for key, value in load_env_file(path).items():
        os.environ[key] = value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for payload in payloads:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(path)


def safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"api_base"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in excluded
    }


def protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmarks": args.benchmarks,
        "split": "train",
        "train_limit": args.train_limit,
        "target_model": args.target_model,
        "reflection_model": args.reflection_model,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout": args.request_timeout,
        "model_retries": args.model_retries,
        "max_completion_tokens": args.max_completion_tokens,
        "reflection_max_completion_tokens": args.reflection_max_completion_tokens,
        "max_attempts": args.max_attempts,
        "max_concurrency": args.max_concurrency,
        "rollout_timeout": args.rollout_timeout,
        "rollout_retries": args.rollout_retries,
        "seed": args.seed,
        "max_trajectory_chars": args.max_trajectory_chars,
        "max_reflection_chars": args.max_reflection_chars,
        "max_previous_answer_chars": args.max_previous_answer_chars,
        "benchmark_specs": [asdict(BENCHMARKS[name]) for name in args.benchmarks],
        "first_rollout": "shared",
        "success_rule": "reward > 0",
        "reflection_evidence": ["original task", "Outcome: fail", "trajectory"],
    }


def protocol_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.events = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines):
                    print(f"warning: ignoring incomplete final event line {index}", file=sys.stderr)
                    continue
                raise ValueError(f"invalid event JSON at {self.path}:{index}")
            if payload.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise ValueError(f"unsupported event schema at {self.path}:{index}")
            events.append(payload)
        return events

    async def append(self, payload: dict[str, Any]) -> None:
        event = {"schema_version": EVENT_SCHEMA_VERSION, "recorded_at": utc_now(), **payload}
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            needs_separator = self.path.is_file() and self.path.stat().st_size > 0
            if needs_separator:
                with self.path.open("rb") as raw:
                    raw.seek(-1, os.SEEK_END)
                    needs_separator = raw.read(1) != b"\n"
            with self.path.open("a", encoding="utf-8") as stream:
                if needs_separator:
                    stream.write("\n")
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self.events.append(event)

    def rollout_events(self, benchmark: str) -> dict[tuple[str, str, int], dict[str, Any]]:
        indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
        for event in self.events:
            if event.get("type") != "rollout" or event.get("benchmark") != benchmark:
                continue
            key = (str(event["arm"]), str(event["task_id"]), int(event["attempt_index"]))
            if key in indexed:
                raise ValueError(f"duplicate rollout event: {key}")
            indexed[key] = event
        return indexed

    def reflection_events(self, benchmark: str) -> dict[tuple[str, int], dict[str, Any]]:
        indexed: dict[tuple[str, int], dict[str, Any]] = {}
        for event in self.events:
            if event.get("type") != "reflection" or event.get("benchmark") != benchmark:
                continue
            key = (str(event["task_id"]), int(event["source_attempt_index"]))
            if key in indexed:
                raise ValueError(f"duplicate reflection event: {key}")
            indexed[key] = event
        return indexed


def rendered_trajectory(trajectory: Trajectory, *, max_chars: int) -> str:
    events = [event for event in trajectory.events if event.get("role") != "system"]
    rendered = json.dumps(events, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > max_chars:
        rendered = "[earlier trajectory content truncated]\n" + rendered[-max_chars:]
    return rendered


def previous_answer(trajectory: Trajectory, *, max_chars: int) -> str:
    for event in reversed(trajectory.events):
        response = event.get("model_response")
        if isinstance(response, str) and response:
            return response[-max_chars:]
        if event.get("role") == "assistant" and isinstance(event.get("content"), str):
            return str(event["content"])[-max_chars:]
    return "(no assistant answer was recorded)"


def reflection_context(answer: str, reflection: str) -> str:
    return (
        "## Evidence from the immediately preceding unsuccessful rollout\n"
        f"Previous answer:\n{answer}\n\n"
        f"Reflection and revised plan:\n{reflection}\n\n"
        "Use this evidence critically, avoid repeating the same failure, and solve the original task again."
    )


def reflected_task(task: Task, context: str) -> Task:
    system_prompt = f"{task.system_prompt.rstrip()}\n\n{context}" if task.system_prompt else context
    return task.model_copy(update={"system_prompt": system_prompt}, deep=True)


def reflected_skill(skill: Skill, context: str, *, task_id: str, attempt_index: int) -> Skill:
    content = f"{skill.content.rstrip()}\n\n{context}\n"
    blob = {"SKILL.md": content}
    return skill.model_copy(
        update={
            "version_id": f"{skill.version_id}:reflection:{task_id}:{attempt_index}",
            "content_hash": compute_skill_content_hash(blob),
            "blob": blob,
        },
        deep=True,
    )


def outcome_success(outcome: RolloutOutcome) -> bool:
    return outcome.trajectory is not None and reward_of(outcome) > 0


def outcome_duration(outcome: RolloutOutcome) -> float:
    return sum((attempt.finished_at - attempt.started_at).total_seconds() for attempt in outcome.attempts)


def outcome_event(
    *,
    benchmark: str,
    arm: str,
    attempt_index: int,
    outcome: RolloutOutcome,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "type": "rollout",
        "benchmark": benchmark,
        "arm": arm,
        "task_id": outcome.spec.task.task_id,
        "attempt_index": attempt_index,
        "success": outcome_success(outcome),
        "reward": reward_of(outcome),
        "duration_seconds": duration_seconds,
        "physical_attempt_duration_seconds": outcome_duration(outcome),
        "outcome": outcome.model_dump(mode="json"),
    }


class TemporaryReflector:
    def __init__(
        self,
        model: Any,
        *,
        max_trajectory_chars: int,
        max_reflection_chars: int,
    ) -> None:
        self._model = model
        self._max_trajectory_chars = max_trajectory_chars
        self._max_reflection_chars = max_reflection_chars

    async def reflect(self, task: Task, trajectory: Trajectory, *, attempt_index: int) -> str:
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REFLECTION_USER_PROMPT.format(
                    task=task.instruction,
                    trajectory=rendered_trajectory(trajectory, max_chars=self._max_trajectory_chars),
                ),
            },
        ]
        content = await chat_content(
            self._model,
            task=f"reflection_ablation.{task.task_id}.attempt-{attempt_index}",
            messages=messages,
        )
        return content[: self._max_reflection_chars].strip()


class ChatModelWithDefaults:
    def __init__(self, client: Any, **defaults: Any) -> None:
        self._client = client
        self._defaults = defaults

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._client.chat(task, messages, **{**self._defaults, **kwargs})


def build_rollout_spec(
    *,
    run_id: str,
    benchmark: BenchmarkSpec,
    task: Task,
    skill: Skill,
    arm: str,
    attempt_index: int,
    task_index: int,
    seed: int,
    env_options_payload: dict[str, Any],
) -> RolloutSpec:
    return RolloutSpec(
        sequence_no=task_index * 10 + attempt_index,
        rollout_id=stable_rollout_id(
            run_id,
            benchmark.name,
            arm,
            task.task_id,
            attempt_index,
        ),
        phase=RolloutPhase.TRAIN,
        task=task,
        skills=[skill],
        sample_index=attempt_index - 1,
        agent_ref="react",
        env_ref=benchmark.env_ref,
        seed=seed + task_index * 100 + attempt_index - 1,
        env_options=env_options_payload,
        metadata={
            "ablation_arm": arm,
            "logical_attempt_index": attempt_index,
            "protocol_version": PROTOCOL_VERSION,
        },
    )


def load_outcome(event: dict[str, Any]) -> RolloutOutcome:
    return RolloutOutcome.model_validate(event["outcome"])


def first_success_attempt(
    task_id: str,
    arm: str,
    rollout_events: dict[tuple[str, str, int], dict[str, Any]],
    *,
    max_attempts: int,
) -> int | None:
    shared = rollout_events.get((ARM_SHARED, task_id, 1))
    if shared is not None and bool(shared["success"]):
        return 1
    for attempt_index in range(2, max_attempts + 1):
        event = rollout_events.get((arm, task_id, attempt_index))
        if event is not None and bool(event["success"]):
            return attempt_index
    return None


def arm_complete(
    task_id: str,
    arm: str,
    rollout_events: dict[tuple[str, str, int], dict[str, Any]],
    *,
    max_attempts: int,
) -> bool:
    if first_success_attempt(task_id, arm, rollout_events, max_attempts=max_attempts) is not None:
        return True
    return (arm, task_id, max_attempts) in rollout_events


def paired_exact_p_value(reflection_only: int, independent_only: int) -> float:
    discordant = reflection_only + independent_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(reflection_only, independent_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_interval(
    reflection_solved: list[bool],
    independent_solved: list[bool],
    *,
    samples: int = 10000,
    seed: int = 0,
) -> list[float]:
    if len(reflection_solved) != len(independent_solved):
        raise ValueError("paired bootstrap inputs must have the same length")
    size = len(reflection_solved)
    if size == 0:
        return [0.0, 0.0]
    differences = [int(left) - int(right) for left, right in zip(reflection_solved, independent_solved, strict=True)]
    rng = random.Random(seed)
    draws = [sum(differences[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples)]
    draws.sort()
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def llm_metrics(database_path: Path, run_ids: list[str]) -> dict[str, Any]:
    if not database_path.is_file() or not run_ids:
        return {
            "logical_calls": 0,
            "succeeded_calls": 0,
            "failed_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "summed_latency_seconds": 0.0,
        }
    placeholders = ",".join("?" for _ in run_ids)
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            f"""SELECT COUNT(*),
                       SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END),
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(total_tokens), 0),
                       COALESCE(SUM(latency_ms), 0)
                FROM llm_calls WHERE run_id IN ({placeholders})""",
            run_ids,
        ).fetchone()
    assert row is not None
    return {
        "logical_calls": int(row[0]),
        "succeeded_calls": int(row[1] or 0),
        "failed_calls": int(row[2] or 0),
        "input_tokens": int(row[3]),
        "output_tokens": int(row[4]),
        "total_tokens": int(row[5]),
        "summed_latency_seconds": float(row[6]) / 1000.0,
    }


def sum_metrics(*payloads: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "logical_calls",
        "succeeded_calls",
        "failed_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "summed_latency_seconds",
    )
    return {key: sum(payload[key] for payload in payloads) for key in keys}


def summarize_benchmark(
    *,
    run_id: str,
    benchmark: str,
    task_ids: list[str],
    store: EventStore,
    max_attempts: int,
    database_path: Path,
) -> dict[str, Any]:
    events = store.rollout_events(benchmark)
    first_success = {
        arm: {task_id: first_success_attempt(task_id, arm, events, max_attempts=max_attempts) for task_id in task_ids}
        for arm in RETRY_ARMS
    }
    arm_summaries: dict[str, Any] = {}
    shared_llm = llm_metrics(database_path, [f"{run_id}:{benchmark}:shared"])
    arm_run_ids = {
        ARM_INDEPENDENT: [f"{run_id}:{benchmark}:independent"],
        ARM_REFLECTION: [
            f"{run_id}:{benchmark}:reflection:rollout",
            f"{run_id}:{benchmark}:reflection:analysis",
        ],
    }
    for arm in RETRY_ARMS:
        solved_attempts = first_success[arm]
        solved = sum(value is not None for value in solved_attempts.values())
        arm_events = [
            event
            for (event_arm, task_id, _attempt), event in events.items()
            if event_arm in {ARM_SHARED, arm} and task_id in solved_attempts
        ]
        attempt_histogram = Counter(
            str(value) if value is not None else "unsolved" for value in solved_attempts.values()
        )
        pass_at_k = {
            str(index): sum(value is not None and value <= index for value in solved_attempts.values()) / len(task_ids)
            if task_ids
            else 0.0
            for index in range(1, max_attempts + 1)
        }
        reflection_events = (
            [
                event
                for event in store.events
                if event.get("type") == "reflection" and event.get("benchmark") == benchmark
            ]
            if arm == ARM_REFLECTION
            else []
        )
        service_seconds = sum(float(event["duration_seconds"]) for event in arm_events) + sum(
            float(event["duration_seconds"]) for event in reflection_events
        )
        arm_specific_llm = llm_metrics(database_path, arm_run_ids[arm])
        arm_summaries[arm] = {
            "tasks": len(task_ids),
            "solved": solved,
            "pass_at_k": pass_at_k,
            "first_success_attempt_histogram": dict(sorted(attempt_histogram.items())),
            "logical_rollouts": len(arm_events),
            "early_stopped_rollouts": len(task_ids) * max_attempts - len(arm_events),
            "reflection_calls": len(reflection_events),
            "summed_task_service_seconds": service_seconds,
            "mean_task_service_seconds": service_seconds / len(task_ids) if task_ids else 0.0,
            "llm": sum_metrics(shared_llm, arm_specific_llm),
        }

    reflection_solved = [first_success[ARM_REFLECTION][task_id] is not None for task_id in task_ids]
    independent_solved = [first_success[ARM_INDEPENDENT][task_id] is not None for task_id in task_ids]
    both = sum(left and right for left, right in zip(reflection_solved, independent_solved, strict=True))
    reflection_only = sum(left and not right for left, right in zip(reflection_solved, independent_solved, strict=True))
    independent_only = sum(
        not left and right for left, right in zip(reflection_solved, independent_solved, strict=True)
    )
    neither = len(task_ids) - both - reflection_only - independent_only
    difference = (
        arm_summaries[ARM_REFLECTION]["pass_at_k"][str(max_attempts)]
        - arm_summaries[ARM_INDEPENDENT]["pass_at_k"][str(max_attempts)]
    )
    complete = all(
        arm_complete(task_id, arm, events, max_attempts=max_attempts) for task_id in task_ids for arm in RETRY_ARMS
    )
    return {
        "benchmark": benchmark,
        "complete": complete,
        "tasks": len(task_ids),
        "max_attempts": max_attempts,
        "shared_first_rollout": True,
        "arms": arm_summaries,
        "paired_pass_at_max": {
            "k": max_attempts,
            "reflection_minus_independent": difference,
            "paired_bootstrap_95_ci": paired_bootstrap_interval(reflection_solved, independent_solved),
            "both_solved": both,
            "reflection_only_solved": reflection_only,
            "independent_only_solved": independent_only,
            "neither_solved": neither,
            "mcnemar_exact_two_sided_p": paired_exact_p_value(reflection_only, independent_only),
        },
        "generated_at": utc_now(),
    }


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sum(int(summary["tasks"]) for summary in summaries)
    result: dict[str, Any] = {"tasks": tasks, "benchmarks": len(summaries), "arms": {}}
    for arm in RETRY_ARMS:
        solved = sum(int(summary["arms"][arm]["solved"]) for summary in summaries)
        max_attempts = max((int(summary["max_attempts"]) for summary in summaries), default=0)
        result["arms"][arm] = {
            "solved": solved,
            "pass_at_max": solved / tasks if tasks else 0.0,
            "logical_rollouts": sum(int(summary["arms"][arm]["logical_rollouts"]) for summary in summaries),
            "reflection_calls": sum(int(summary["arms"][arm]["reflection_calls"]) for summary in summaries),
            "llm": {
                key: sum(summary["arms"][arm]["llm"][key] for summary in summaries)
                for key in summaries[0]["arms"][arm]["llm"]
            }
            if summaries
            else {},
            "k": max_attempts,
        }
    reflection_only = sum(int(summary["paired_pass_at_max"]["reflection_only_solved"]) for summary in summaries)
    independent_only = sum(int(summary["paired_pass_at_max"]["independent_only_solved"]) for summary in summaries)
    result["paired_pass_at_max"] = {
        "reflection_minus_independent": (
            result["arms"][ARM_REFLECTION]["pass_at_max"] - result["arms"][ARM_INDEPENDENT]["pass_at_max"]
        ),
        "reflection_only_solved": reflection_only,
        "independent_only_solved": independent_only,
        "mcnemar_exact_two_sided_p": paired_exact_p_value(reflection_only, independent_only),
    }
    return result


async def run_benchmark(
    *,
    args: argparse.Namespace,
    run_id: str,
    benchmark: BenchmarkSpec,
    root: Path,
    store: EventStore,
    call_sink: DatabaseLLMCallSink,
) -> dict[str, Any]:
    benchmark_dir = root / benchmark.name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(
        benchmark.name,
        data_root=REPO_ROOT / benchmark.data_root,
        split_dir=REPO_ROOT / benchmark.split_dir,
        seed=args.seed,
        shuffle_choices=benchmark.shuffle_choices,
    )
    tasks = dataset.train_tasks()
    if args.train_limit is not None:
        tasks = tasks[: args.train_limit]
    task_ids = [task.task_id for task in tasks]
    manifest_path = benchmark_dir / "tasks.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("task_ids") != task_ids:
            raise ValueError(f"task manifest changed for {benchmark.name}; refusing unsafe resume")
    else:
        write_json(manifest_path, {"benchmark": benchmark.name, "task_ids": task_ids})

    base_skill = build_skill(
        REPO_ROOT / benchmark.skill,
        run_id=run_id,
        benchmark=benchmark.name,
    )
    write_json(benchmark_dir / "base_skill.json", base_skill.model_dump(mode="json"))
    target_client = build_client(
        model=args.target_model,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
        model_retries=args.model_retries,
        call_sink=call_sink,
    )
    reflection_client = build_client(
        model=args.reflection_model,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
        model_retries=args.model_retries,
        call_sink=call_sink,
    )
    agent = build_agent(
        client=target_client,
        model=args.target_model,
        max_turns=benchmark.max_turns,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    reflector = TemporaryReflector(
        ChatModelWithDefaults(
            reflection_client,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.reflection_max_completion_tokens,
        ),
        max_trajectory_chars=args.max_trajectory_chars,
        max_reflection_chars=args.max_reflection_chars,
    )
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        config=RolloutConfig(
            max_concurrent_rollouts=args.max_concurrency,
            timeout_seconds=args.rollout_timeout,
            retry={"max_attempts": args.rollout_retries},
            fail_fast=False,
            workspace_root=benchmark_dir / "workspace_sessions" / session_id,
        ),
    )
    operation_slots = asyncio.Semaphore(args.max_concurrency)
    env_options_payload = environment_options(
        benchmark.name,
        max_turns=benchmark.max_turns,
        env_seed=benchmark.env_seed,
        shell_timeout=benchmark.shell_timeout,
        use_theorem=benchmark.use_theorem,
        use_sketch=benchmark.use_sketch,
    )
    rollout_events = store.rollout_events(benchmark.name)
    reflections = store.reflection_events(benchmark.name)
    task_index = {task.task_id: index for index, task in enumerate(tasks)}
    progress_lock = asyncio.Lock()

    async def report_progress(arm: str, success: bool) -> None:
        async with progress_lock:
            current = store.rollout_events(benchmark.name)
            completed = sum(1 for key in current if key[0] == arm)
            correct = sum(bool(event["success"]) for key, event in current.items() if key[0] == arm)
            print(
                f"[{benchmark.name}] arm={arm} recorded={completed} successes={correct} latest_success={success}",
                file=sys.stderr,
                flush=True,
            )

    async def execute_rollout(
        *,
        task: Task,
        skill: Skill,
        arm: str,
        attempt_index: int,
    ) -> RolloutOutcome:
        spec = build_rollout_spec(
            run_id=run_id,
            benchmark=benchmark,
            task=task,
            skill=skill,
            arm=arm,
            attempt_index=attempt_index,
            task_index=task_index[task.task_id],
            seed=args.seed,
            env_options_payload=env_options_payload,
        )
        context_suffix = arm if arm != ARM_REFLECTION else "reflection:rollout"
        async with operation_slots:
            started_at = monotonic()
            with llm_run_context(f"{run_id}:{benchmark.name}:{context_suffix}"):
                outcome = (await scheduler.run([spec]))[0]
            duration_seconds = monotonic() - started_at
        await store.append(
            outcome_event(
                benchmark=benchmark.name,
                arm=arm,
                attempt_index=attempt_index,
                outcome=outcome,
                duration_seconds=duration_seconds,
            )
        )
        await report_progress(arm, outcome_success(outcome))
        return outcome

    async def ensure_shared(task: Task) -> None:
        key = (ARM_SHARED, task.task_id, 1)
        if key in rollout_events:
            return
        await execute_rollout(task=task, skill=base_skill, arm=ARM_SHARED, attempt_index=1)

    await asyncio.gather(*(ensure_shared(task) for task in tasks))
    rollout_events = store.rollout_events(benchmark.name)

    async def ensure_reflection(
        task: Task, source_attempt_index: int, outcome: RolloutOutcome
    ) -> dict[str, Any] | None:
        key = (task.task_id, source_attempt_index)
        existing = reflections.get(key)
        if existing is not None:
            return existing
        if outcome.trajectory is None:
            return None
        async with operation_slots:
            started_at = monotonic()
            with llm_run_context(f"{run_id}:{benchmark.name}:reflection:analysis"):
                content = await reflector.reflect(task, outcome.trajectory, attempt_index=source_attempt_index)
            duration_seconds = monotonic() - started_at
        answer = previous_answer(outcome.trajectory, max_chars=args.max_previous_answer_chars)
        event = {
            "type": "reflection",
            "benchmark": benchmark.name,
            "task_id": task.task_id,
            "source_attempt_index": source_attempt_index,
            "source_rollout_id": outcome.spec.rollout_id,
            "duration_seconds": duration_seconds,
            "previous_answer": answer,
            "content": content,
        }
        await store.append(event)
        reflections[key] = {"schema_version": EVENT_SCHEMA_VERSION, "recorded_at": utc_now(), **event}
        return reflections[key]

    async def run_arm(task: Task, arm: str) -> None:
        nonlocal rollout_events
        if arm_complete(task.task_id, arm, rollout_events, max_attempts=args.max_attempts):
            return
        previous_event = rollout_events[(ARM_SHARED, task.task_id, 1)]
        previous_outcome = load_outcome(previous_event)
        for attempt_index in range(2, args.max_attempts + 1):
            existing = rollout_events.get((arm, task.task_id, attempt_index))
            if existing is not None:
                previous_event = existing
                previous_outcome = load_outcome(existing)
                if bool(existing["success"]):
                    return
                continue

            attempt_task = task
            attempt_skill = base_skill
            if arm == ARM_REFLECTION:
                reflection = await ensure_reflection(task, attempt_index - 1, previous_outcome)
                if reflection is not None and reflection.get("content"):
                    context = reflection_context(str(reflection["previous_answer"]), str(reflection["content"]))
                    if benchmark.name == "alfworld":
                        attempt_skill = reflected_skill(
                            base_skill,
                            context,
                            task_id=task.task_id,
                            attempt_index=attempt_index,
                        )
                    else:
                        attempt_task = reflected_task(task, context)
            outcome = await execute_rollout(
                task=attempt_task,
                skill=attempt_skill,
                arm=arm,
                attempt_index=attempt_index,
            )
            rollout_events[(arm, task.task_id, attempt_index)] = next(
                event
                for event in reversed(store.events)
                if event.get("type") == "rollout"
                and event.get("benchmark") == benchmark.name
                and event.get("arm") == arm
                and event.get("task_id") == task.task_id
                and event.get("attempt_index") == attempt_index
            )
            previous_outcome = outcome
            if outcome_success(outcome):
                return

    initially_failed = [task for task in tasks if not bool(rollout_events[(ARM_SHARED, task.task_id, 1)]["success"])]
    await asyncio.gather(*(run_arm(task, arm) for task in initially_failed for arm in RETRY_ARMS))
    summary = summarize_benchmark(
        run_id=run_id,
        benchmark=benchmark.name,
        task_ids=task_ids,
        store=store,
        max_attempts=args.max_attempts,
        database_path=root / "state.db",
    )
    write_json(benchmark_dir / "summary.json", summary)
    benchmark_events = [event for event in store.events if event.get("benchmark") == benchmark.name]
    write_jsonl(
        benchmark_dir / "rollouts.jsonl",
        [event for event in benchmark_events if event.get("type") == "rollout"],
    )
    write_jsonl(
        benchmark_dir / "reflections.jsonl",
        [event for event in benchmark_events if event.get("type") == "reflection"],
    )
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if args.max_attempts < 2:
        raise ValueError("--max-attempts must be at least 2")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")
    if args.rollout_retries < 1:
        raise ValueError("--rollout-retries must be at least 1")
    if args.train_limit is not None and args.train_limit < 1:
        raise ValueError("--train-limit must be at least 1")
    if len(args.benchmarks) != len(set(args.benchmarks)):
        raise ValueError("--benchmarks must not contain duplicates")
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires --output-dir")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "outputs" / "reflection_vs_independent" / f"paired_pass4_{timestamp}"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    apply_env_file((REPO_ROOT / args.env_file).resolve() if not args.env_file.is_absolute() else args.env_file)
    args.api_base = args.api_base or os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.dry_run and not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")

    output_dir = resolve_output_dir(args)
    protocol = protocol_payload(args)
    fingerprint = protocol_hash(protocol)
    if args.dry_run:
        counts: dict[str, int] = {}
        for name in args.benchmarks:
            spec = BENCHMARKS[name]
            dataset = build_dataset(
                name,
                data_root=REPO_ROOT / spec.data_root,
                split_dir=REPO_ROOT / spec.split_dir,
                seed=args.seed,
                shuffle_choices=spec.shuffle_choices,
            )
            tasks = dataset.train_tasks()
            counts[name] = len(tasks if args.train_limit is None else tasks[: args.train_limit])
        result = {
            "dry_run": True,
            "output_dir": str(output_dir),
            "protocol_hash": fingerprint,
            "tasks": counts,
            "serial_benchmark_order": args.benchmarks,
            "max_global_operations": args.max_concurrency,
            "max_logical_rollouts_per_task_per_arm": args.max_attempts,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    protocol_path = output_dir / "protocol.json"
    if args.resume:
        if not output_dir.is_dir() or not protocol_path.is_file():
            raise FileNotFoundError(f"resume output is incomplete or missing: {output_dir}")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != fingerprint:
            raise ValueError("resume protocol does not match the existing run")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(protocol_path, {"protocol_hash": fingerprint, "protocol": protocol})
        write_json(output_dir / "arguments.json", safe_arguments(args))

    run_id = output_dir.name
    store = EventStore(output_dir / "events.jsonl")
    database = await bootstrap_skill_database(output_dir / "state.db")
    call_sink = DatabaseLLMCallSink(database)
    summaries: list[dict[str, Any]] = []
    started_at = monotonic()
    try:
        for benchmark_name in args.benchmarks:
            write_json(
                output_dir / "runtime_state.json",
                {
                    "status": "running",
                    "current_benchmark": benchmark_name,
                    "completed_benchmarks": [item["benchmark"] for item in summaries],
                    "updated_at": utc_now(),
                },
            )
            summary = await run_benchmark(
                args=args,
                run_id=run_id,
                benchmark=BENCHMARKS[benchmark_name],
                root=output_dir,
                store=store,
                call_sink=call_sink,
            )
            summaries.append(summary)
            write_json(
                output_dir / "summary.json",
                {
                    "status": "partial",
                    "protocol_hash": fingerprint,
                    "benchmarks": summaries,
                    "aggregate": aggregate_summaries(summaries),
                    "elapsed_seconds": monotonic() - started_at,
                },
            )
    except BaseException as exc:
        write_json(
            output_dir / "runtime_state.json",
            {
                "status": "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_benchmarks": [item["benchmark"] for item in summaries],
                "updated_at": utc_now(),
            },
        )
        raise
    finally:
        await database.close()

    final = {
        "status": "complete",
        "protocol_hash": fingerprint,
        "benchmarks": summaries,
        "aggregate": aggregate_summaries(summaries),
        "elapsed_seconds": monotonic() - started_at,
        "completed_at": utc_now(),
    }
    write_json(output_dir / "summary.json", final)
    write_json(
        output_dir / "runtime_state.json",
        {
            "status": "complete",
            "completed_benchmarks": [item["benchmark"] for item in summaries],
            "updated_at": utc_now(),
        },
    )
    print(json.dumps(final["aggregate"], ensure_ascii=False, indent=2))
    return final


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
