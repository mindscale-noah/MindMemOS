"""Seed deterministic demo memories into a local MindMemOS database.

This is intentionally a database-level seed rather than an API add flow: it
does not need an LLM or embedding endpoint, and it can be rerun safely.  Each
memory is written to Qdrant and mirrored as a ``Memory`` node in Neo4j.  The
script never deletes points and the deterministic UUIDs make the operation
idempotent.

Example::

    UV_CACHE_DIR=/tmp/mindmemos-uv-cache uv run python scripts/seed_demo_memories.py \
      --config config/mindmemos/dev.yaml \
      --project-id proj-dev-vanilla-memory \
      --user-id cyq \
      --api-key-uuid key_dev_0001
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from mindmemos.config import build_config
from mindmemos.infra.db.models import MemoryNode, MemoryPoint
from mindmemos.infra.db.neo4j import Neo4jStore
from mindmemos.infra.db.qdrant import QdrantStore

DEMO_NAMESPACE = "https://mindmemos.ai/demo-memory"
TOPICS = (
    ("偏好", "用户偏好简洁、分步骤的回答，重要结论放在开头。"),
    ("工作流", "用户习惯先查看现有文件和配置，再进行修改，最后运行针对性校验。"),
    ("项目", "当前项目使用 Python SDK 管理本地配置，云端服务通过 API key 和 URL 访问。"),
    ("界面", "用户希望本地 Skill 管理界面与云端 UI 保持一致，并支持 Overview、Memory、Skills、Settings。"),
    ("记忆", "用户希望在 Memory 页面查看自己的记忆，并支持搜索、分页和详情展示。"),
    ("技能", "Skills 页面需要支持新增技能、查看内容以及并排对比两个技能并高亮差异。"),
    ("开发", "用户倾向于减少本地依赖，优先选择只需要 Python 的启动方案。"),
    ("验证", "完成改动后，用户希望看到明确的测试结果和实际运行行为，而不是只看静态代码。"),
)


def _memory_id(index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{DEMO_NAMESPACE}/{index}"))


def _request_id(index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{DEMO_NAMESPACE}/request/{index}"))


def _build_points(
    *,
    count: int,
    project_id: str,
    user_id: str,
    api_key_uuid: str,
    account_id: str,
) -> tuple[list[MemoryPoint], list[MemoryNode]]:
    now = datetime.now(UTC).replace(microsecond=0)
    points: list[MemoryPoint] = []
    nodes: list[MemoryNode] = []

    for offset in range(count):
        index = offset + 1
        memory_id = _memory_id(index)
        topic, template = TOPICS[offset % len(TOPICS)]
        created_at = now - timedelta(minutes=count - index)
        content = f"Demo memory {index:03d}（{topic}）：{template} 这是用于本地 UI 展示的示例记录。"
        payload = {
            "memory_id": memory_id,
            "account_id": account_id,
            "project_id": project_id,
            "api_key_uuid": api_key_uuid,
            "user_id": user_id,
            "request_id": _request_id(index),
            "content": content,
            "mem_type": "fact",
            "mem_extract_type": "vanilla",
            "mem_extract_version": "demo_seed_v1",
            "metadata": {
                "demo": True,
                "seed": "mindmemos-sdk-ui",
                "index": index,
                "topic": topic,
            },
            "validate_from": created_at,
            "status": "active",
            "reinforcement_count": 0,
            "created_at": created_at,
            "parent_ids": [],
            "root_id": [memory_id],
        }
        points.append(MemoryPoint(memory_id=memory_id, payload=payload))
        nodes.append(MemoryNode(project_id=project_id, memory_id=memory_id, content=content))

    return points, nodes


async def _seed(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    cfg = build_config(config_path=config_path)
    qdrant = QdrantStore(cfg.database.qdrant)
    neo4j = Neo4jStore(cfg.database.neo4j)
    try:
        points, nodes = _build_points(
            count=args.count,
            project_id=args.project_id,
            user_id=args.user_id,
            api_key_uuid=args.api_key_uuid,
            account_id=args.account_id,
        )
        await qdrant.ensure_schema()
        await qdrant.upsert_memories(points)

        if not args.skip_graph:
            await neo4j.ensure_schema()
            await neo4j.upsert_nodes(memories=nodes)

        # Verify the exact project/user scope after the write.  This also makes
        # the command fail loudly if a configured collection is not reachable.
        from qdrant_client import models as qmodels

        scoped_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id", match=qmodels.MatchValue(value=args.user_id)
                )
            ]
        )
        records, _ = await qdrant.scroll_memories(
            args.project_id,
            filter_=scoped_filter,
            limit=max(args.count, 200),
        )
        seeded_ids = {_memory_id(index) for index in range(1, args.count + 1)}
        found_ids = {record.payload.get("memory_id") for record in records}
        missing = seeded_ids - found_ids
        if missing:
            raise RuntimeError(f"seed verification failed: {len(missing)} memory points are missing")

        graph_status = "and Neo4j Memory nodes" if not args.skip_graph else "(Neo4j skipped)"
        print(
            f"Seeded {args.count} demo memories into project={args.project_id!r}, "
            f"user={args.user_id!r} {graph_status}."
        )
    finally:
        await qdrant.close()
        await neo4j.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mindmemos/dev.yaml", help="MindMemOS YAML config path")
    parser.add_argument("--project-id", required=True, help="Target project scope")
    parser.add_argument("--user-id", required=True, help="User scope stored on every demo memory")
    parser.add_argument("--api-key-uuid", required=True, help="API-key identity stored on every demo memory")
    parser.add_argument("--account-id", default="memory_standalone", help="Account scope (default: memory_standalone)")
    parser.add_argument("--count", type=int, default=200, help="Number of deterministic demo memories (default: 200)")
    parser.add_argument("--skip-graph", action="store_true", help="Only seed Qdrant; do not write Neo4j nodes")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(_seed(_parse_args()))
