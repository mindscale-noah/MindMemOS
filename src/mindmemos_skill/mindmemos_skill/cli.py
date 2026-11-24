"""Local commands for managing versioned Skills."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from . import create_service
from .api.typing import CommitFromLocalRequest, DiffRequest, FileSkillInfo, ListRequest, RegisterFromLocalRequest
from .errors import MindMemosSkillError


def build_parser() -> argparse.ArgumentParser:
    """Build the local Skill command parser.

    Returns:
        Parser for repository operations and the local workspace path.
    """
    parser = argparse.ArgumentParser(prog="mindmemos-skill", description="Manage local MindMemOS Skill versions.")
    parser.add_argument("--config", help="Complete Skill service YAML configuration for repository commands.")
    commands = parser.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema", help="Export YAML editor hints from registered configuration classes.")
    schema.add_argument("--output", type=Path, required=True, help="Destination JSON Schema file.")
    schema.add_argument(
        "--kind",
        choices=("service", "training-request", "environment"),
        default="service",
        help="Configuration or request to describe.",
    )
    schema.add_argument("--include-demo", action="store_true", help="Include the opt-in offline demo registrations.")

    register = commands.add_parser("register", help="Register a local Skill directory or SKILL.md file.")
    register.add_argument("source", help="Skill directory or SKILL.md path.")
    register.add_argument("--alias", help="Project-local alias for later commands.")

    commands.add_parser("list", help="List registered Skills in the local workspace.")

    publish = commands.add_parser("publish", help="Publish edited Skill files as a new local version.")
    publish.add_argument("skill", help="Skill ID or alias.")
    publish.add_argument("source", help="Edited Skill directory or SKILL.md path.")
    publish.add_argument("--message", help="Reason for publishing this version.")
    publish.add_argument("--base-version", help="Explicit parent version ID.")

    diff = commands.add_parser("diff", help="Compare two stored Skill versions.")
    diff.add_argument("skill", help="Skill ID or alias.")
    diff.add_argument("--from", dest="from_version", required=True, help="Starting version ID.")
    diff.add_argument("--to", dest="to_version", required=True, help="Target version ID.")
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Execute one command against the local Skill service."""
    from .config.loader import read_skill_config

    config = read_skill_config(args.config)
    if config.fastapi.auth.mode != "none":
        raise ValueError("Direct local CLI requires none mode; use the SDK client for authenticated services")
    async with await create_service(config) as service:
        if args.command == "register":
            result = await service.register_from_local(
                RegisterFromLocalRequest(
                    project_id=service.default_project_id, skill=FileSkillInfo(path=args.source), alias=args.alias
                )
            )
            print(f"Registered {result.skill_id} {result.skill_version_id}")
        elif args.command == "list":
            for skill in await service.list_skills(ListRequest(project_id=service.default_project_id)):
                print(f"{skill.skill_id}\t{skill.alias or '-'}\t{skill.name}\t{skill.active_version_id}")
        elif args.command == "publish":
            result = await service.commit_from_local(
                CommitFromLocalRequest(
                    project_id=service.default_project_id,
                    skill_id=args.skill,
                    skill=FileSkillInfo(path=args.source, commit_message=args.message),
                    base_version_id=args.base_version,
                )
            )
            print(f"Published {result.skill.skill_id} {result.version.version_id}")
        elif args.command == "diff":
            result = await service.diff_skill(
                DiffRequest(
                    project_id=service.default_project_id,
                    from_version_id=args.from_version,
                    to_version_id=args.to_version,
                )
            )
            print(f"{result.from_version_id} -> {result.to_version_id}")
            if not result.files and not result.implementation_changed:
                print("No changes")
            for change in result.files:
                print(f"{change.change}: {change.path}")
                if change.binary:
                    print("(binary file)")
                elif change.diff:
                    print(change.diff, end="" if change.diff.endswith("\n") else "\n")
            if result.implementation_changed:
                print("Skill implementation changed")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local Skill command line interface.

    Args:
        argv: Optional argument list; defaults to process arguments.

    Returns:
        Zero on success and one for a handled input or repository error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "schema" and args.config is None:
        parser.error("--config is required for repository commands")
    try:
        if args.command == "schema":
            if args.include_demo:
                from . import demos  # noqa: F401

            from .config.schema import environment_config_schema, skill_config_schema, training_request_schema

            generators = {
                "service": skill_config_schema,
                "training-request": training_request_schema,
                "environment": environment_config_schema,
            }
            schema = generators[args.kind]()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Saved schema to {args.output}")
        else:
            asyncio.run(_run(args))
    except (MindMemosSkillError, ValidationError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
