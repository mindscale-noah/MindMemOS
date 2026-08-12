"""STATE-Bench × feedback_evo 10-round iteration runner.

Loop contract (confirmed with the user):

1. Train tasks only accumulate memory: every round runs a fresh, non-overlapping
   chunk of 10 train tasks with ``MINDMEMOS_ROLE=train`` (add + retrieve).
2. A fresh chunk of 4 test tasks runs per round with ``MINDMEMOS_ROLE=feedback``
   (retrieve only). After the chunk, the runner submits each task's conversation
   to ``POST /v1/memory/feedback-evo/collect`` and records the detected signal
   count (evolution is NEVER skipped, even with zero signals).
3. Every round then triggers ``POST /v1/memory/self-evolve`` with ``force=true``
   and records the resulting version / changes / signal count.
4. A reserved subset of the test split (never used as feedback) is evaluated at
   the end with ``MINDMEMOS_ROLE=eval`` (retrieve only, full scoring) for an
   unbiased final measurement.

Task execution shells out to the vendored STATE-Bench repo
(``data/STATE-Bench``); memory operations go through the MindMemOS HTTP API.
The runner loads ``{state-bench-dir}/.env`` so benchmark client credentials are
available to the subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from .schedule import FeedbackEvoSchedule, RoundPlan, build_schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_BENCH_DIR = REPO_ROOT / "data" / "STATE-Bench"
AGENT_SOURCE = Path(__file__).resolve().parent / "mindmemos_agent.py"
AGENT_TARGET_REL = Path("agents") / "mindmemos_agent.py"


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #


def _load_dotenv_file(path: Path) -> dict[str, str]:
    """Minimal ``.env`` loader (KEY=VALUE lines; quotes stripped)."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _subprocess_env(state_bench_dir: Path, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(_load_dotenv_file(state_bench_dir / ".env"))
    env.update(extra)
    return env


# --------------------------------------------------------------------------- #
# STATE-Bench invocation
# --------------------------------------------------------------------------- #


def _run_batch(
    state_bench_dir: Path,
    *,
    domain: str,
    task_ids: list[str],
    output_dir: Path,
    log_path: Path,
    agent_class: str,
    agent_model_name: str,
    agent_model_reasoning_level: str | None,
    num_workers: int,
    no_score: bool,
    role: str,
    api_base: str,
    api_key: str,
) -> None:
    """Run one STATE-Bench batch (chunk) as a subprocess.

    The full subprocess output is saved to ``log_path`` so the chunk can be
    audited after the fact; on failure the log tail is surfaced for debugging.
    """

    python = state_bench_dir / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"STATE-Bench venv not found: {python}")

    command = [
        str(python),
        "-u",
        "-m",
        "state_bench.scripts.run_batch",
        "--domain",
        domain,
        "--tasks",
        ",".join(task_ids),
        "--agent-class",
        agent_class,
        "--agent-model-name",
        agent_model_name,
        "--output-dir",
        str(output_dir),
        "--num-runs",
        "1",
        "--num-workers",
        str(num_workers),
    ]
    if agent_model_reasoning_level:
        command += ["--agent-model-reasoning-level", agent_model_reasoning_level]
    if no_score:
        command += ["--no-score"]

    env = _subprocess_env(
        state_bench_dir,
        {
            "MINDMEMOS_API_BASE": api_base,
            "MINDMEMOS_API_KEY": api_key,
            "MINDMEMOS_ROLE": role,
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[runner] run_batch role={role} tasks={len(task_ids)} -> {output_dir} "
        f"(log: {log_path})",
        flush=True,
    )
    with log_path.open("wb") as log_file:
        result = subprocess.run(
            command,
            cwd=state_bench_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"run_batch failed (exit={result.returncode}) for role={role} "
            f"output={output_dir}; log={log_path}\n--- log tail ---\n{tail}"
        )


# --------------------------------------------------------------------------- #
# MindMemOS HTTP calls
# --------------------------------------------------------------------------- #


def _post(api_base: str, api_key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=600) as client:
        response = client.post(
            f"{api_base.rstrip('/')}{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def _collect_task(
    api_base: str,
    api_key: str,
    *,
    task_id: str,
    conversation: list[dict[str, Any]],
    user_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_messages": conversation,
        "task_id": task_id,
        "user_id": user_id,
    }
    body = _post(api_base, api_key, "/v1/memory/feedback-evo/collect", payload)
    data = body.get("data") or {}
    return {
        "event_id": data.get("event_id"),
        "signal_count": int(data.get("signal_count") or 0),
        "signals": data.get("signals") or [],
        "message": body.get("message") or "",
    }


def _evolve(api_base: str, api_key: str) -> dict[str, Any]:
    body = _post(api_base, api_key, "/v1/memory/self-evolve", {"force": True})
    data = body.get("data") or {}
    return {
        "evolved": bool(data.get("evolved")),
        "version": int(data.get("version") or 0),
        "changes": data.get("changes") or [],
        "signal_count": int(data.get("signal_count") or 0),
        "message": body.get("message") or "",
    }


# --------------------------------------------------------------------------- #
# trajectory helpers
# --------------------------------------------------------------------------- #


def _trajectory_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("run1/*.json"))


def _load_trajectory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _signal_evolvable_paths(signals: list[dict[str, Any]]) -> dict[str, int]:
    """Count collected signals by the evolvable item they implicate."""

    counts: dict[str, int] = {}
    for signal in signals:
        path = signal.get("evolvable_path")
        if isinstance(path, str) and path:
            counts[path] = counts.get(path, 0) + 1
    return counts


def _eval_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": trajectory.get("task_id"),
        "state_requirements_met": trajectory.get("state_requirements_met"),
        "task_requirements_met": trajectory.get("task_requirements_met"),
        "task_completion_pass": trajectory.get("task_completion_pass"),
        "ux_score": trajectory.get("ux_score"),
        "turns": trajectory.get("turns"),
        "tool_calls": trajectory.get("tool_calls"),
        "tool_errors": trajectory.get("tool_errors"),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def _install_agent(state_bench_dir: Path) -> None:
    target = state_bench_dir / AGENT_TARGET_REL
    if target.exists() and target.read_text(encoding="utf-8") == AGENT_SOURCE.read_text(encoding="utf-8"):
        print(f"[runner] agent adapter already installed: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(AGENT_SOURCE, target)
    print(f"[runner] installed agent adapter -> {target}", flush=True)


def _write_round_report(report_dir: Path, round_report: dict[str, Any]) -> None:
    """Persist one round immediately so failures never lose completed rounds.

    Writes ``round_XX.json`` (full report) and appends one line to
    ``rounds.jsonl`` (append-only sequence for incremental recovery).
    """

    report_dir.mkdir(parents=True, exist_ok=True)
    index = round_report["round_index"]
    (report_dir / f"round_{index:02d}.json").write_text(
        json.dumps(round_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (report_dir / "rounds.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(round_report, ensure_ascii=False) + "\n")


def _run_round(
    *,
    round_plan: RoundPlan,
    state_bench_dir: Path,
    domain: str,
    output_root: Path,
    report_dir: Path,
    logs_dir: Path,
    agent_class: str,
    agent_model_name: str,
    agent_model_reasoning_level: str | None,
    num_workers: int,
    no_score: bool,
    no_evolve: bool,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    """Execute one round: train chunk (memory) + feedback chunk + (optionally) evolve."""

    train_dir = output_root / "train" / f"round{round_plan.round_index:02d}"
    feedback_dir = output_root / "feedback" / f"round{round_plan.round_index:02d}"
    round_label = f"round{round_plan.round_index:02d}"

    _run_batch(
        state_bench_dir,
        domain=domain,
        task_ids=list(round_plan.train_task_ids),
        output_dir=train_dir,
        log_path=logs_dir / f"train_{round_label}.log",
        agent_class=agent_class,
        agent_model_name=agent_model_name,
        agent_model_reasoning_level=agent_model_reasoning_level,
        num_workers=num_workers,
        no_score=no_score,
        role="train",
        api_base=api_base,
        api_key=api_key,
    )
    _run_batch(
        state_bench_dir,
        domain=domain,
        task_ids=list(round_plan.feedback_test_task_ids),
        output_dir=feedback_dir,
        log_path=logs_dir / f"feedback_{round_label}.log",
        agent_class=agent_class,
        agent_model_name=agent_model_name,
        agent_model_reasoning_level=agent_model_reasoning_level,
        num_workers=num_workers,
        no_score=no_score,
        role="feedback",
        api_base=api_base,
        api_key=api_key,
    )

    collected: list[dict[str, Any]] = []
    total_signals = 0
    evolvable_path_counts: dict[str, int] = {}
    for path in _trajectory_files(feedback_dir):
        trajectory = _load_trajectory(path)
        task_id = str(trajectory.get("task_id") or path.stem)
        try:
            event = _collect_task(
                api_base,
                api_key,
                task_id=task_id,
                conversation=trajectory.get("conversation") or [],
                user_id=trajectory.get("user_id"),
            )
        except Exception as exc:
            print(f"[runner] collect failed for {task_id}: {exc}", flush=True)
            collected.append({"task_id": task_id, "error": str(exc)})
            continue
        signal_count = event["signal_count"]
        total_signals += signal_count
        for path, count in _signal_evolvable_paths(event["signals"]).items():
            evolvable_path_counts[path] = evolvable_path_counts.get(path, 0) + count
        collected.append({"task_id": task_id, **event})

    if no_evolve:
        evolution = {
            "evolved": False,
            "version": None,
            "changes": [],
            "signal_count": total_signals,
            "message": "baseline: evolution skipped (config pinned at initial version)",
        }
    else:
        evolution = _evolve(api_base, api_key)

    round_report: dict[str, Any] = {
        "round_index": round_plan.round_index,
        "train_task_ids": list(round_plan.train_task_ids),
        "feedback_test_task_ids": list(round_plan.feedback_test_task_ids),
        "train_log": str(logs_dir / f"train_{round_label}.log"),
        "feedback_log": str(logs_dir / f"feedback_{round_label}.log"),
        "collected_events": collected,
        "signals_this_round": total_signals,
        "signal_evolvable_path_counts": evolvable_path_counts,
        "cumulative_signal_count": evolution["signal_count"],
        "evolution": evolution,
    }
    print(
        f"[runner] round {round_plan.round_index}: signals={total_signals} "
        f"({evolvable_path_counts}) | evolution version={evolution['version']} "
        f"changes={len(evolution['changes'])}",
        flush=True,
    )
    _write_round_report(report_dir, round_report)
    return round_report


def _run_final_eval(
    *,
    schedule: FeedbackEvoSchedule,
    state_bench_dir: Path,
    domain: str,
    output_root: Path,
    logs_dir: Path,
    agent_class: str,
    agent_model_name: str,
    agent_model_reasoning_level: str | None,
    num_workers: int,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    """Run the reserved test tasks with retrieval only and full scoring."""

    eval_dir = output_root / "eval" / "final"
    _run_batch(
        state_bench_dir,
        domain=domain,
        task_ids=list(schedule.reserved_test_task_ids),
        output_dir=eval_dir,
        log_path=logs_dir / "eval_final.log",
        agent_class=agent_class,
        agent_model_name=agent_model_name,
        agent_model_reasoning_level=agent_model_reasoning_level,
        num_workers=num_workers,
        no_score=False,
        role="eval",
        api_base=api_base,
        api_key=api_key,
    )
    metrics = [_eval_metrics(_load_trajectory(path)) for path in _trajectory_files(eval_dir)]
    scored = [item for item in metrics if item["task_completion_pass"] is not None]
    summary: dict[str, Any] = {
        "reserved_test_task_ids": list(schedule.reserved_test_task_ids),
        "per_task": metrics,
        "task_completion": {
            "passed": sum(1 for item in scored if item["task_completion_pass"]),
            "total": len(scored),
        },
        "state_requirements_met": {
            "passed": sum(1 for item in scored if item["state_requirements_met"]),
            "total": len(scored),
        },
    }
    print(
        f"[runner] final eval: task_completion {summary['task_completion']['passed']}/"
        f"{summary['task_completion']['total']}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["customer_support", "shopping_assistant", "travel"])
    parser.add_argument("--state-bench-dir", type=Path, default=DEFAULT_STATE_BENCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to outputs/statebench/<domain>")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--train-per-round", type=int, default=10)
    parser.add_argument("--feedback-test-per-round", type=int, default=4)
    parser.add_argument("--reserved-test-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("MINDMEMOS_API_KEY", ""))
    parser.add_argument("--agent-class", default="MindMemOSAgent")
    parser.add_argument("--agent-model-name", required=True)
    parser.add_argument("--agent-model-reasoning-level", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-score", action="store_true", help="Skip judge scoring for iteration chunks (final eval always scores).")
    parser.add_argument(
        "--no-evolve",
        action="store_true",
        help="Baseline mode: skip POST /v1/memory/self-evolve after each feedback chunk "
        "so the project config stays pinned at its initial version.",
    )
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("--api-key (or MINDMEMOS_API_KEY) is required")

    state_bench_dir = args.state_bench_dir.resolve()
    split_path = state_bench_dir / "state_bench" / "domains" / args.domain / "splits" / "train_test.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = list(splits.get("splits", {}).get("train", []))
    test_ids = list(splits.get("splits", {}).get("test", []))

    schedule = build_schedule(
        train_ids,
        test_ids,
        rounds=args.rounds,
        train_per_round=args.train_per_round,
        feedback_test_per_round=args.feedback_test_per_round,
        reserved_test_count=args.reserved_test_count,
        seed=args.seed,
    )

    output_root = (args.output_dir or Path("outputs") / "statebench" / args.domain).resolve()
    report_dir = output_root / "reports"
    logs_dir = output_root / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "schedule.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "rounds": [
                    {
                        "round_index": r.round_index,
                        "train_task_ids": list(r.train_task_ids),
                        "feedback_test_task_ids": list(r.feedback_test_task_ids),
                    }
                    for r in schedule.rounds
                ],
                "reserved_test_task_ids": list(schedule.reserved_test_task_ids),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _install_agent(state_bench_dir)
    round_reports = [
        _run_round(
            round_plan=round_plan,
            state_bench_dir=state_bench_dir,
            domain=args.domain,
            output_root=output_root,
            report_dir=report_dir,
            logs_dir=logs_dir,
            agent_class=args.agent_class,
            agent_model_name=args.agent_model_name,
            agent_model_reasoning_level=args.agent_model_reasoning_level,
            num_workers=args.num_workers,
            no_score=args.no_score,
            no_evolve=args.no_evolve,
            api_base=args.api_base,
            api_key=args.api_key,
        )
        for round_plan in schedule.rounds
    ]
    final_eval = _run_final_eval(
        schedule=schedule,
        state_bench_dir=state_bench_dir,
        domain=args.domain,
        output_root=output_root,
        logs_dir=logs_dir,
        agent_class=args.agent_class,
        agent_model_name=args.agent_model_name,
        agent_model_reasoning_level=args.agent_model_reasoning_level,
        num_workers=args.num_workers,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    summary = {
        "domain": args.domain,
        "schedule": {
            "seed": args.seed,
            "rounds": args.rounds,
            "train_per_round": args.train_per_round,
            "feedback_test_per_round": args.feedback_test_per_round,
            "reserved_test_count": args.reserved_test_count,
        },
        "rounds": round_reports,
        "final_eval": final_eval,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "final_eval.json").write_text(
        json.dumps(final_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[runner] reports written to {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
