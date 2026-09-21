"""Download the official benchmark, assets, and evaluator at a fixed revision."""

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/ynulihao/AgentSkillOS.git"
REVISION = "c3cfae10ea34a13973c461273ad9c5cc4aa907fe"


def download(destination: Path) -> None:
    """Create or verify an immutable sparse source checkout.

    Args:
        destination: Dedicated directory for the benchmark checkout.

    Raises:
        ValueError: An existing checkout differs from the pinned release.
    """
    destination = destination.expanduser().resolve()
    if (destination / ".git").exists():
        origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=destination, text=True).strip()
        if origin != REPOSITORY:
            raise ValueError("Existing checkout has a different source repository")
        head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=destination, text=True, capture_output=True)
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=destination, text=True)
        if status or (head.returncode == 0 and head.stdout.strip() != REVISION):
            raise ValueError("Existing source checkout differs from the pinned revision")
        if head.returncode == 0:
            return
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Destination must be empty")
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=destination, check=True)
        subprocess.run(["git", "remote", "add", "origin", REPOSITORY], cwd=destination, check=True)
    for args in (
        ["sparse-checkout", "set", "benchmark"],
        ["fetch", "--depth=1", "--filter=blob:none", "origin", REVISION],
        ["checkout", "--detach", "FETCH_HEAD"],
    ):
        subprocess.run(["git", *args], cwd=destination, check=True)
    print(f"Benchmark source: {destination}\nRevision: {REVISION}")


def main() -> None:
    """Download the exact source and bundled task data without model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/agentskillos/source"))
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
