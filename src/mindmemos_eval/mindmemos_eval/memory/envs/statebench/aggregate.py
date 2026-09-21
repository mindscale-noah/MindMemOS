"""Aggregate per-seed STATE-Bench online-controlled reports."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append({"pairing_status": "invalid", "line_number": line_number, "error": str(exc)})
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _bootstrap_seed_task(rows: list[dict[str, Any]], *, reps: int = 10_000, seed: int = 42) -> dict[str, Any]:
    """Bootstrap whole (seed, task_id) strata, retaining all run pairs."""
    strata: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        delta = row.get("pass_at_1_delta")
        if delta is None or row.get("pairing_status") != "ok":
            continue
        key = (str(row.get("seed")), str(row.get("task_id")))
        strata.setdefault(key, []).append(int(delta))
    if not strata:
        return {"estimate": None, "ci95": [None, None], "reps": reps, "strata": "seed_task"}
    keys = sorted(strata)
    observed = [value for values in strata.values() for value in values]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(reps):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        values = [value for key in sampled for value in strata[key]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    low_index = max(0, int(0.025 * reps) - 1)
    high_index = min(reps - 1, int(0.975 * reps))
    return {
        "estimate": sum(observed) / len(observed),
        "ci95": [estimates[low_index], estimates[high_index]],
        "reps": reps,
        "strata": "seed_task",
        "stratum_count": len(keys),
    }


def _mean(rows: list[dict[str, Any]], branch: str, field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = (row.get(branch) or {}).get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None


def _pass_rate(rows: list[dict[str, Any]], branch: str) -> float | None:
    values = [
        bool((row.get(branch) or {}).get("task_completion_pass"))
        for row in rows
        if (row.get(branch) or {}).get("task_completion_pass") is not None
    ]
    return sum(values) / len(values) if values else None


def _pass_power(rows: list[dict[str, Any]], branch: str) -> float | None:
    by_task: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        item = row.get(branch) or {}
        value = item.get("task_completion_pass")
        if value is None:
            continue
        key = (str(row.get("seed")), str(row.get("task_id")))
        by_task.setdefault(key, []).append(bool(value))
    if not by_task or not all(len(values) == 5 for values in by_task.values()):
        return None
    return sum(all(values) for values in by_task.values()) / len(by_task)


def aggregate(root: Path, seeds: list[int], *, bootstrap_reps: int = 10_000) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    domain: str | None = None
    for seed in seeds:
        seed_root = root / f"seed{seed}"
        comparison_path = seed_root / "comparison.json"
        paired_path = seed_root / "paired_results.jsonl"
        if not comparison_path.exists():
            failures.append({"seed": seed, "stage": "comparison", "error": "comparison.json missing"})
            continue
        comparison = _load(comparison_path)
        domain = domain or comparison.get("domain")
        per_seed[str(seed)] = comparison
        seed_rows = _rows(paired_path)
        if not seed_rows:
            failures.append({"seed": seed, "stage": "paired_results", "error": "paired_results.jsonl missing or empty"})
        rows.extend(seed_rows)
    valid_rows = [row for row in rows if row.get("pairing_status") == "ok"]
    delta_values = [row["pass_at_1_delta"] for row in valid_rows if row.get("pass_at_1_delta") is not None]
    base_cost = sum(float((item.get("baseline_final") or {}).get("cost_usd_total") or 0) for item in per_seed.values())
    evo_cost = sum(float((item.get("evolution_final") or {}).get("cost_usd_total") or 0) for item in per_seed.values())
    return {
        "protocol": "online-controlled",
        "domain": domain,
        "seeds": seeds,
        "per_seed": per_seed,
        "rows": len(rows),
        "paired_rows": len(valid_rows),
        "pairing_failures": [row for row in rows if row.get("pairing_status") != "ok"] + failures,
        "pairing_complete": not any(row.get("pairing_status") != "ok" for row in rows) and not failures,
        "baseline": {
            "pass_at_1": _pass_rate(valid_rows, "baseline"),
            "pass_at_5": _pass_power(valid_rows, "baseline"),
            "ux_score_mean": _mean(valid_rows, "baseline", "ux_score"),
            "turns_mean": _mean(valid_rows, "baseline", "turns"),
            "tool_errors_mean": _mean(valid_rows, "baseline", "tool_errors"),
            "cost_usd_total": base_cost,
        },
        "evolution": {
            "pass_at_1": _pass_rate(valid_rows, "evolution"),
            "pass_at_5": _pass_power(valid_rows, "evolution"),
            "ux_score_mean": _mean(valid_rows, "evolution", "ux_score"),
            "turns_mean": _mean(valid_rows, "evolution", "turns"),
            "tool_errors_mean": _mean(valid_rows, "evolution", "tool_errors"),
            "cost_usd_total": evo_cost,
        },
        "pass_at_1_delta": sum(delta_values) / len(delta_values) if delta_values else None,
        "bootstrap_pass_at_1_delta": _bootstrap_seed_task(rows, reps=bootstrap_reps),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Formal root containing seed42/seed43/seed44")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    args = parser.parse_args(argv)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    report = aggregate(args.root.resolve(), seeds, bootstrap_reps=args.bootstrap_reps)
    _write_json(args.root / "comparison.json", report)
    with (args.root / "paired_results.jsonl").open("w", encoding="utf-8") as stream:
        for seed in seeds:
            path = args.root / f"seed{seed}" / "paired_results.jsonl"
            if path.exists():
                stream.write(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
