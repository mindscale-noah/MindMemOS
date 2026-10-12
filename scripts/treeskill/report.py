"""Summarize a TreeSkill comparison, including edits that never became patches."""

import csv
from collections import Counter
from pathlib import Path

from mindmemos_skill.algos.train.treeskill.artifacts import model_usage
from mindmemos_skill.common.json import read_json, write_json


def collect_patch_report(training: Path) -> dict:
    """Inspect edit dispositions, content changes, and upstream stage failures.

    Args:
        training: Algorithm artifact directory containing stages and patches.

    Returns:
        Counts and auditable rows distinguishing applied edits from no-ops.
    """
    rows = []
    for path in sorted((training / "patches").glob("*/patch.json")):
        patch = read_json(path)
        result = patch["result"]
        before = (path.parent / "before.md").read_text(encoding="utf-8")
        after = (path.parent / "after.md").read_text(encoding="utf-8")
        changed = before != after
        if result["operation"] == "reject":
            disposition = "model_declined"
        elif not result["accepted"]:
            disposition = "validation_rejected"
        elif not changed:
            disposition = "accepted_no_change"
        else:
            disposition = "applied"
        rows.append(
            {
                "target_node_id": result["target_node_id"],
                "operation": result["operation"],
                "disposition": disposition,
                "changed": changed,
                "reason": result.get("message", ""),
                "artifact": str(path),
            }
        )
    failures = []
    stages = {}
    for name, key in (("analysis", "records"), ("localization", "evidence"), ("fusion", "edits")):
        path = training / "stages" / f"{name}.json"
        if path.exists():
            stage = read_json(path)
            stages[name] = {"records": len(stage[key]), "failures": len(stage["failures"])}
            for item in stage["failures"]:
                detail = item if isinstance(item, dict) else {"trajectory_id": item}
                failures.append({"stage": name, **detail})
    counts = Counter(row["disposition"] for row in rows)
    reasons = Counter(row["reason"] or "unspecified" for row in rows if row["disposition"] != "applied")
    # Model/parse failures have no patch.json and must remain visible separately.
    usage = model_usage(training)
    return {
        "patch_records": len(rows),
        "applied": counts["applied"],
        "model_declined": counts["model_declined"],
        "validation_rejected": counts["validation_rejected"],
        "accepted_no_change": counts["accepted_no_change"],
        "non_applied_reasons": dict(reasons),
        "stages": stages,
        "stage_failures": failures,
        "model_calls": usage["calls"],
        "failed_model_calls": usage["failed_calls"],
        "model_usage_complete": usage["usage_complete"],
        "rows": rows,
    }


def build_comparison(root: Path) -> dict:
    """Write a live comparison and detailed patch tables from committed artifacts.

    Args:
        root: Experiment directory created by the SDK runner.

    Returns:
        Current run state, stage scores, paired outcomes, and patch diagnostics.
    """
    manifest = read_json(root / "manifest.json")
    # Earlier local-service manifests stored task membership only in split.json.
    test_ids = manifest["test_ids"] if "test_ids" in manifest else read_json(root / "split.json")["test"]
    result = {"status": manifest["status"], "protocol": manifest["protocol"], "phases": {}}
    baseline_skipped = manifest["protocol"].get("baseline_skipped", False)
    for name in ("baseline", "evolved"):
        if name == "baseline" and baseline_skipped:
            result["phases"][name] = {"status": "skipped"}
            continue
        path = root / name / "summary.json"
        if path.exists():
            result["phases"][name] = read_json(path)
        elif name == "evolved":
            service_root = root / "training" / "test"
            summary = service_root / "summary.json"
            if summary.exists():
                result["phases"][name] = {
                    **read_json(summary),
                    "items": read_json(service_root / "results.json")["tasks"],
                }
    training = root / "training" / "train"
    summary_path = training / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        result["phases"]["training"] = summary
    result["patches"] = collect_patch_report(training)
    before = {item["task_id"]: item for item in result["phases"].get("baseline", {}).get("items", [])}
    after = {item["task_id"]: item for item in result["phases"].get("evolved", {}).get("items", [])}
    paired = [
        {
            "task_id": task_id,
            "baseline_score": before[task_id]["score"],
            "evolved_score": after[task_id]["score"],
            "score_delta": after[task_id]["score"] - before[task_id]["score"],
            "baseline_success": before[task_id]["success"],
            "evolved_success": after[task_id]["success"],
        }
        for task_id in test_ids
        if task_id in before and task_id in after
    ]
    result["paired"] = {
        "expected": 0 if baseline_skipped else len(test_ids),
        "completed": len(paired),
        "mean_score_delta": sum(row["score_delta"] for row in paired) / len(paired) if paired else None,
        "gained_successes": sum(not row["baseline_success"] and row["evolved_success"] for row in paired),
        "lost_successes": sum(row["baseline_success"] and not row["evolved_success"] for row in paired),
    }
    write_json(root / "comparison.json", result)
    write_json(root / "patch_diagnostics.json", result["patches"])
    for name, rows, fields in (
        (
            "patches.csv",
            result["patches"]["rows"],
            ["target_node_id", "operation", "disposition", "changed", "reason", "artifact"],
        ),
        (
            "paired_results.csv",
            paired,
            ["task_id", "baseline_score", "evolved_score", "score_delta", "baseline_success", "evolved_success"],
        ),
    ):
        with (root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return result
