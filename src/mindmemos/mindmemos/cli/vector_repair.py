"""Standalone vector-repair command using the same service as the private API."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from ..config import init_config, init_config_from_env
from ..infra.db import close_database_clients, ensure_database_schema
from ..llm import close_llm_clients, init_embed_client
from ..pipelines.memory_db import VectorRepairRequest, VectorRepairService
from ..typing import MemoryRequestContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair pending MindMemOS memory vectors")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--account-id", default="standalone-repair")
    parser.add_argument("--api-key-uuid", default="standalone-repair")
    parser.add_argument("--memory-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--config-path", help="Path to the static MindMemOS YAML configuration")
    parser.add_argument("--config-name", default="dev", help="Named config used when --config-path is provided")
    return parser


async def run(args: argparse.Namespace) -> dict:
    if args.config_path:
        init_config(config_name=args.config_name, config_path=args.config_path)
    else:
        init_config_from_env()
    await ensure_database_schema()
    init_embed_client()
    ctx = MemoryRequestContext(
        request_id="vector-repair-cli",
        account_id=args.account_id,
        project_id=args.project_id,
        api_key_uuid=args.api_key_uuid,
        scopes=["memory:read", "memory:write"],
    )
    service = VectorRepairService()
    try:
        if args.status:
            service_status = await service.status(ctx)
            return service_status.model_dump()
        result = await service.repair(
            ctx,
            VectorRepairRequest(limit=args.limit, memory_ids=args.memory_id, force=args.force),
        )
        return result.model_dump()
    finally:
        await close_llm_clients()
        await close_database_clients()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
