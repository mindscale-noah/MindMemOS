#!/usr/bin/env python3
"""Register the AgentSkillOS pool, organize it through the SDK, and score retrieval."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import SkillClient
from mindmemos_sdk.skills.cli_files import collect_resources, read_source
from mindmemos_sdk.skills.typing import (
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    StructuredSkillInfo,
    TaskShowRequest,
)
from mindmemos_sdk.transport import HttpTransport
from mindmemos_skill.envs.builtin_envs.agentskillos.reward import aggregate_metrics, compute_metrics
from pydantic import TypeAdapter

from mindmemos_sdk import MindMemOSClient
from scripts.skill_eval.agentskillos.typing import (
    GraphResult,
    PoolManifest,
    PoolMember,
    QueryRecord,
    RecommendationResult,
    RegistrationResult,
    RelevanceRecord,
)

API_KEY_ENV = "MINDMEMOS_SKILL_API_KEY"
logger = logging.getLogger(__name__)


def _write_json(path: Path, value: object) -> None:
    """Atomically publish an artifact without losing a previously saved result."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_dataset(root: Path) -> tuple[list[QueryRecord], dict[str, RelevanceRecord], list[PoolMember]]:
    """Validate prepared or original legacy files before any remote mutation."""
    prepared = (root / "skill_pool_manifest.json").exists()
    query_root = root / "queries" if prepared else root
    queries = [
        QueryRecord.model_validate_json(line)
        for line in (query_root / "tasks.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    labels = TypeAdapter(dict[str, RelevanceRecord]).validate_json((query_root / "relevance.json").read_bytes())
    ids = [query.task_id for query in queries]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError("Queries and labels must have exactly matching, unique, nonempty task IDs")
    if prepared:
        members = PoolManifest.model_validate_json((root / "skill_pool_manifest.json").read_bytes()).skills
        splits = json.loads((root / "splits.json").read_text())
        if set(splits) != {"test"} or len(splits["test"]) != len(ids) or set(splits["test"]) != set(ids):
            raise ValueError("Expected exactly one test split covering all queries once")
    else:
        join = json.loads((root / "ssl_pool_join.json").read_text())["join"]
        directories = sorted(path.name for path in (root / "skill_pool_200").iterdir() if path.is_dir())
        if set(directories) != set(join) or not directories:
            raise ValueError("Source join must exactly cover the complete skill_pool_200")
        members = [
            PoolMember(source_skill_id=name, member_id=str(join[name]), relative_path=f"skill_pool_200/{name}")
            for name in directories
        ]
    source_ids = {member.source_skill_id for member in members}
    if len(source_ids) != len(members) or len({member.member_id for member in members}) != len(members):
        raise ValueError("Pool source and member identities must each be unique")
    for member in members:
        path = root / member.relative_path
        if not path.resolve().is_relative_to(root) or not path.is_dir():
            raise ValueError(f"Invalid pool directory: {member.relative_path}")
    for label in labels.values():
        core = label.core_gt_ids if label.core_gt_ids is not None else label.gt_skill_ids
        if set(core) - source_ids:
            raise ValueError(f"Core labels contain IDs outside the pool: {sorted(set(core) - source_ids)}")
    return queries, labels, members


def _wait_for_graph(client: SkillClient, submitted: dict, output: Path, timeout: float, interval: float) -> GraphResult:
    """Resolve synchronous publication or poll an accepted asynchronous task."""
    _write_json(output / "organization.json", submitted)
    if "task_id" not in submitted:
        return GraphResult.model_validate(submitted)
    deadline = time.monotonic() + timeout
    previous_status = None
    while time.monotonic() < deadline:
        snapshot = client.show_task(TaskShowRequest(task_id=submitted["task_id"]))
        if snapshot["status"] != previous_status:
            logger.info("Graph task %s status=%s", submitted["task_id"], snapshot["status"])
            previous_status = snapshot["status"]
        if snapshot["status"] == "succeeded":
            return GraphResult.model_validate(snapshot["result"])
        if snapshot["status"] in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Graph task {submitted['task_id']} ended with {snapshot['status']}: {snapshot.get('error_type')}"
            )
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        f"Graph task {submitted['task_id']} exceeded {timeout} seconds; task identity saved in organization.json"
    )


def _evaluate_queries(
    client: SkillClient,
    args: argparse.Namespace,
    graph: GraphResult,
    queries: list[QueryRecord],
    labels: dict[str, RelevanceRecord],
    members: list[PoolMember],
    versions: dict[str, str],
) -> dict:
    """Score all eligible queries and keep retrieval failures separate from misses."""
    source_by_version = {version: source for source, version in versions.items()}
    member_by_source = {member.source_skill_id: member.member_id for member in members}
    rows, skipped, errors = [], [], []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index, query in enumerate(queries, 1):
            label = labels[query.task_id]
            core = label.core_gt_ids if label.core_gt_ids is not None else label.gt_skill_ids
            if label.task_type == "generic_only":
                skipped.append({"task_id": query.task_id, "reason": "generic_only"})
                continue
            try:
                response = RecommendationResult.model_validate(
                    client.recommend(
                        RecommendRequest(
                            graph_ref=graph.graph_id,
                            graph_version_id=graph.graph_version_id,
                            query=query.instruction_text,
                            top_k=args.top_k,
                            strategy=args.strategy,
                            include_scenes=args.include_scenes,
                        )
                    )
                )
                if response.graph_version_id != graph.graph_version_id:
                    raise ValueError("Service returned a different graph snapshot")
                ranked_versions = [score.skill_version_id for score in response.scores]
                if len(ranked_versions) != len(set(ranked_versions)) or len(ranked_versions) > args.top_k:
                    raise ValueError("Service returned duplicate versions or exceeded top_k")
                ranked_sources = [source_by_version[version] for version in ranked_versions]
                metrics = compute_metrics(ranked_sources, core)
                row = {
                    "task_id": query.task_id,
                    "core_gt_ids": core,
                    "ranked_source_ids": ranked_sources,
                    "ranked_member_ids": [member_by_source[source] for source in ranked_sources],
                    "ranked_version_ids": ranked_versions,
                    "metrics": metrics,
                }
                rows.append(row)
            except Exception as exc:
                row = {"task_id": query.task_id, "error_type": type(exc).__name__, "error": str(exc)}
                errors.append(row)
                logger.error("Query %s failed: %s", query.task_id, type(exc).__name__)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            logger.info("Queries processed=%d/%d scored=%d errors=%d", index, len(queries), len(rows), len(errors))
    eligible = len(queries) - len(skipped)
    return {
        "status": "complete" if rows and not errors else "partial",
        "evaluator_version": "agentskillos-sdk-core-v1",
        "graph_id": graph.graph_id,
        "graph_version_id": graph.graph_version_id,
        "candidate_count": len(members),
        "query_count": len(queries),
        "evaluated_query_count": len(rows),
        "skipped_queries": skipped,
        "errors": errors,
        "scoring_coverage": len(rows) / eligible if eligible else 0,
        "strategy": args.strategy,
        "include_scenes": args.include_scenes,
        "top_k": args.top_k,
        "metrics": aggregate_metrics([row["metrics"] for row in rows]) if rows else {},
        "metric_protocol": "Binary core labels; full ground-truth denominator; query macro average; MRR truncated at 10.",
        "dataset_root": str(args.dataset_root),
    }


def _run_evaluation(client: SkillClient, args: argparse.Namespace, *, registrations_path: Path | None = None) -> dict:
    """Register original documents, build the complete pool, and evaluate a pinned snapshot."""
    queries, labels, members = _load_dataset(args.dataset_root)
    if all(label.task_type == "generic_only" for label in labels.values()):
        raise ValueError("Evaluation requires at least one non-generic query")
    # Validate every source before registering any candidate.
    sources = {}
    for member in members:
        root, content, _ = read_source(str(args.dataset_root / member.relative_path))
        sources[member.source_skill_id] = StructuredSkillInfo(
            content=content, resources=collect_resources(root, exclude="*Zone.Identifier")
        )
    logger.info("Loaded candidates=%d queries=%d; registration and retrieval concurrency=1", len(members), len(queries))
    versions = {}
    if registrations_path is not None:
        versions = TypeAdapter(dict[str, str]).validate_json(registrations_path.read_text(), strict=True)
        if set(versions) != set(sources) or len(set(versions.values())) != len(versions) or not all(versions.values()):
            raise ValueError("Reused registrations must map the complete pool to distinct nonempty version IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    if versions:
        _write_json(args.output / "registrations.json", versions)
        logger.info("Reusing exact registered versions=%d from %s", len(versions), registrations_path)
    for index, member in enumerate(members if registrations_path is None else [], 1):
        result = RegistrationResult.model_validate(
            client.register(RegisterRequest(skill=sources[member.source_skill_id]))
        )
        if result.skill_version_id in versions.values():
            raise ValueError("Registration reused a version across different pool sources")
        versions[member.source_skill_id] = result.skill_version_id
        _write_json(args.output / "registrations.json", versions)
        logger.info("Registered candidates=%d/%d source=%s", index, len(members), member.source_skill_id)
    graph = _wait_for_graph(
        client,
        client.organize(
            OrganizeRequest(
                skill_refs=list(versions.values()), alias=args.graph_alias, force_extract_scene=args.force_extract_scene
            )
        ),
        args.output,
        args.task_timeout,
        args.poll_interval,
    )
    if len(graph.skill_version_ids) != len(versions) or set(graph.skill_version_ids) != set(versions.values()):
        raise ValueError("Published graph does not contain exactly the complete candidate pool")
    _write_json(args.output / "graph.json", graph.model_dump())
    _write_json(args.output / "pool.json", [member.model_dump() for member in members])
    logger.info("Graph published snapshot=%s candidates=%d", graph.graph_version_id, len(versions))
    summary = _evaluate_queries(client, args, graph, queries, labels, members, versions)
    _write_json(args.output / "summary.json", summary)
    return summary


def main() -> None:
    """Parse arguments and run the SDK evaluation, returning failure for partial results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Prepared dataset or original agentskillos_eval directory"
    )
    parser.add_argument("--base-url", help="Explicit Skill service URL; otherwise use SDK settings")
    parser.add_argument("--sdk-home", type=Path, help="SDK settings directory")
    parser.add_argument("--graph-alias", help="Optional unique alias for a new graph")
    parser.add_argument("--strategy", choices=("graph", "vector"), default="graph")
    parser.add_argument("--include-scenes", action="store_true")
    parser.add_argument("--force-extract-scene", action="store_true")
    parser.add_argument(
        "--reuse-registrations", type=Path, help="Reuse a complete prior registrations.json without registering again"
    )
    parser.add_argument("--top-k", type=int, default=50, help="At least 50 to report the full legacy metric suite")
    parser.add_argument("--http-timeout", type=float, default=3600)
    parser.add_argument("--task-timeout", type=float, default=7200)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("output") / f"agentskillos_sdk_{datetime.now():%Y%m%d_%H%M%S}"
    )
    args = parser.parse_args()
    if args.top_k < 50 or min(args.http_timeout, args.task_timeout, args.poll_interval) <= 0:
        parser.error("top_k must be >= 50 and all timeouts/intervals must be positive")
    if args.strategy == "graph" and args.include_scenes:
        parser.error("--include-scenes requires --strategy vector")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("Output directory already exists; select a new directory")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.base_url:
        with HttpTransport(
            base_url=args.base_url,
            api_key=os.getenv(API_KEY_ENV),
            timeout_seconds=args.http_timeout,
            max_retries=0,
            require_api_key=False,
        ) as transport:
            summary = _run_evaluation(SkillClient(transport), args, registrations_path=args.reuse_registrations)
    else:
        config = ConfigManager(config_dir=args.sdk_home).load_or_default()
        config.network.timeout_seconds = int(args.http_timeout)
        with MindMemOSClient(config=config) as client:
            summary = _run_evaluation(client.skills, args, registrations_path=args.reuse_registrations)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
