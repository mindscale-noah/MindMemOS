"""Launch the local workbench."""

import argparse
from pathlib import Path

import uvicorn
from mindmemos_sdk.config import ConfigManager

from .app import create_app


def main() -> None:
    """Run a loopback-only workbench, optionally attached to an existing graph database."""
    parser = argparse.ArgumentParser(description="MindMemOS local workbench")
    parser.add_argument("--home", help="SDK configuration directory")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--graph-db", type=Path, help="Advanced SQLite database override")
    parser.add_argument("--project-id", help="Advanced local scope override")
    args = parser.parse_args()
    from mindmemos_skill.api import create_service

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    manager = ConfigManager(args.home)
    service = create_service(home=manager.config_dir, database_path=args.graph_db, project_id=args.project_id)
    try:
        uvicorn.run(
            create_app(
                manager,
                service=service,
                project_id=service.local_project_id,
                graph_service=service.graph_queries(),
            ),
            host="127.0.0.1",
            port=args.port,
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
