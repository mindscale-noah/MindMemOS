"""Explicit CLI phases; importing the package never starts an experiment."""

import argparse
import asyncio
from pathlib import Path

from .config import load_config
from .credentials import load_credentials
from .data import create_split


def main() -> None:
    """Dispatch a single explicitly requested local experiment phase."""
    parser = argparse.ArgumentParser(description="SWE-bench Verified hierarchical memory experiment")
    parser.add_argument("phase", choices=["split", "train", "extract", "baseline", "test"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, help="Local dotenv credentials for model and memory requests")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase == "split":
        create_split(config)
    else:
        if args.env_file:
            load_credentials(args.env_file, require_models=args.phase != "extract")
        from .runner import extract_memories, run_rollouts

        asyncio.run(extract_memories(config) if args.phase == "extract" else run_rollouts(config, args.phase))


if __name__ == "__main__":
    main()
