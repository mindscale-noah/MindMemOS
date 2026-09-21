"""Assemble the Skill service from local defaults or explicit configuration."""

import argparse
import asyncio
from pathlib import Path

from ..factory import create_service


async def _main() -> None:
    """Build and display the local service selected by the Makefile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Local service YAML file")
    parser.add_argument("--home", type=Path, help="Local workspace root")
    args = parser.parse_args()
    if args.config is not None and args.home is not None:
        parser.error("--home and --config cannot be combined")
    service = await create_service(args.config, home=args.home)
    try:
        if args.config is None:
            print(f"Local project: {service.local_project_id}")
        else:
            print(service)
    finally:
        await service.close()


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
