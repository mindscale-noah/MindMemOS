"""Assemble the Skill service from local defaults or explicit configuration."""

import argparse
from pathlib import Path

from ..factory import create_service


def main() -> None:
    """Build and display the local service selected by the Makefile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Local service YAML file")
    parser.add_argument("--home", type=Path, help="Local workspace root")
    args = parser.parse_args()
    if args.config is not None and args.home is not None:
        parser.error("--home and --config cannot be combined")
    service = create_service(args.config, home=args.home)
    try:
        if args.config is None:
            print(f"Local project: {service.local_project_id}")
        else:
            print(service)
    finally:
        service.close()


if __name__ == "__main__":
    main()
