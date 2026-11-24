"""Invoke the unchanged official evaluator in a dedicated Python process."""

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> None:
    """Validate dependencies and run the official task scorer outside agent execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    # Import all structural-scoring dependencies before any agent is invoked.
    for module in ("pptx", "docx", "fitz", "jsonschema", "pdfplumber", "openpyxl"):
        importlib.import_module(module)
    sys.path.insert(0, str(args.source.resolve()))
    from benchmark.AgentSkillOS_bench import EVALUATOR_REGISTRY, evaluate_task_sync
    from benchmark.AgentSkillOS_bench.types import TaskConfig

    record = json.loads(args.task.read_text(encoding="utf-8"))
    config = TaskConfig.from_dict(record)
    ids = [entry.id for entry in config.evaluators]
    if not ids or len(ids) != len(set(ids)) or sum(entry.weight for entry in config.evaluators) <= 0:
        raise ValueError("Official task must have unique evaluators and positive total weight")
    resolved = set()
    pending = list(config.evaluators)
    while pending:
        ready = [entry for entry in pending if set(entry.depends_on) <= resolved]
        if not ready:
            raise ValueError("Evaluator dependencies contain missing IDs or a cycle")
        for entry in ready:
            if entry.op_func not in EVALUATOR_REGISTRY or entry.weight < 0:
                raise ValueError(f"Unsupported evaluator: {entry.id}")
            resolved.add(entry.id)
            pending.remove(entry)
    if args.preflight:
        return
    result = evaluate_task_sync(str(args.task), str(args.workspace))
    errors = [
        entry.message
        for entry in result.evaluator_results
        if entry.message.startswith(
            (
                "Evaluator exception:",
                "Evaluator error:",
                "Unknown evaluator function:",
                "Dependency not found:",
            )
        )
    ]
    if errors:
        raise RuntimeError(f"Official scorer failed: {errors}")
    args.output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
