"""Launch the local workbench."""

import argparse
import asyncio
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from mindmemos_sdk.config import ConfigManager

from .app import create_app


async def _main() -> None:
    """Run a loopback-only workbench, optionally attached to an existing graph database."""
    parser = argparse.ArgumentParser(description="MindMemOS local workbench")
    parser.add_argument("--home", help="SDK configuration directory")
    parser.add_argument("--env-file", type=Path, help="Load model credentials from an explicit environment file")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--graph-db", type=Path, help="Advanced SQLite database override")
    parser.add_argument("--project-id", help="Advanced local scope override")
    args = parser.parse_args()
    from mindmemos_skill.api import create_service

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("--env-file must be an existing file")
        load_dotenv(args.env_file, override=True)
    manager = ConfigManager(args.home)
    load_dotenv(manager.config_path.with_suffix(".env"), override=False, interpolate=False)
    service = await create_service(home=manager.config_dir, database_path=args.graph_db, project_id=args.project_id)
    try:
        await uvicorn.Server(
            uvicorn.Config(
                await create_app(
                    manager,
                    service=service,
                    project_id=service.local_project_id,
                    graph_service=await service.graph_queries(),
                ),
                host="127.0.0.1",
                port=args.port,
            )
        ).serve()
    finally:
        await service.close()


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
