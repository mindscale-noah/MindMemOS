"""Launch the local workbench."""

import argparse
import asyncio
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from mindmemos_sdk.config import ConfigManager

from .app import create_app


async def _main() -> None:
    """Run the loopback workbench against configured SDK services."""
    parser = argparse.ArgumentParser(description="MindMemOS local workbench")
    parser.add_argument("--home", help="SDK configuration directory")
    parser.add_argument("--env-file", type=Path, help="Load model credentials from an explicit environment file")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("--env-file must be an existing file")
        load_dotenv(args.env_file, override=True)
    manager = ConfigManager(args.home)
    load_dotenv(manager.config_path.with_suffix(".env"), override=False, interpolate=False)
    await uvicorn.Server(uvicorn.Config(await create_app(manager), host="127.0.0.1", port=args.port)).serve()


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
