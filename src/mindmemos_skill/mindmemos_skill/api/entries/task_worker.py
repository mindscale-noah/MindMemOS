"""Run the dedicated Kafka Skill worker against the service's existing database."""

import argparse
import asyncio
import signal
from pathlib import Path

from ...config.loader import read_skill_config
from ..factory import create_service


async def run_worker(config_path: Path) -> None:
    """Consume Skill tasks until the process receives a shutdown signal.

    Args:
        config_path: Service YAML with task_manager.mode=kafka and worker=true.
    """
    deployment = read_skill_config(config_path)
    config = deployment.service
    if config.task_manager is None or config.task_manager.mode != "kafka" or not config.task_manager.worker:
        raise ValueError("Worker config requires task_manager.mode=kafka and worker=true")
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(name, stopping.set)
    async with await create_service(deployment):
        await stopping.wait()


def main() -> None:
    """Parse the shared service configuration and start the Skill worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Skill service YAML")
    args = parser.parse_args()
    asyncio.run(run_worker(args.config))


if __name__ == "__main__":
    main()
