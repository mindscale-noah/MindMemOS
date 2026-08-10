"""Command line interface for the MindMemOS SDK.

This module defines the public CLI shape first. Command handlers are intentionally
thin placeholders until the SDK managers are implemented; they make the command
contract executable and easy to test without performing network or filesystem
mutations yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from contextvars import ContextVar
from getpass import getpass
from pathlib import Path
from typing import Any

from .client import MindMemOSClient
from .config import DEFAULT_BASE_URL, ConfigManager, SDKConfigCompilerV2, mask_secret
from .errors import ConfigError, MindMemOSSDKError
from .memory import DialogueMessage, StatusResult
from .skills import (
    DuplicateSkillAction,
    ExportSkillRequest,
    LocalSkillManifest,
    PublishLocalRequest,
    RegisterLocalRequest,
    SkillManager,
    SkillRecord,
)

_JSON_PARSE_ERROR = object()
_CLI_SKILL_MANAGERS: ContextVar[list[SkillManager] | None] = ContextVar("mindmemos_cli_skill_managers", default=None)


def _without_none(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindmemos",
        description="MindMemOS SDK command line interface.",
    )
    parser.set_defaults(handler=_handle_root)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_auth_command(subparsers)
    _add_config_commands(subparsers)
    _add_ui_command(subparsers)
    _add_skill_commands(subparsers)
    _add_memory_commands(subparsers)
    _add_doctor_command(subparsers)
    return parser


def _add_auth_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "auth",
        help="Configure API key, user id, and API base URL.",
        description="Run the interactive authentication and SDK configuration flow.",
    )
    parser.add_argument("--api-key", help="API key. If omitted, prompt interactively.")
    parser.add_argument("--user-id", help="Default user id. If omitted, prompt interactively.")
    parser.add_argument("--base-url", default=None, help=f"API base URL. Default: {DEFAULT_BASE_URL}")
    parser.set_defaults(handler=_handle_auth)


def _add_config_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("config", help="Show, migrate, or reset local SDK settings.")
    config_subparsers = parser.add_subparsers(dest="config_command", metavar="<config-command>")
    parser.set_defaults(handler=_handle_config_root)

    show = config_subparsers.add_parser("show", help="Show current SDK settings.")
    show.add_argument("--show-secret", action="store_true", help="Show the full API key.")
    show.set_defaults(handler=_handle_config_show)

    reset = config_subparsers.add_parser("reset", help="Reset local SDK settings.")
    reset.add_argument("--yes", "-y", action="store_true", help="Reset without confirmation prompts.")
    reset.set_defaults(handler=_handle_config_reset)

    migrate = config_subparsers.add_parser("migrate", help="Convert settings.json v1 to config.yaml v2.")
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="Write config.yaml and a v1 backup. Without this flag, only show the dry-run plan.",
    )
    migrate.add_argument("--config-dir", default=None, help="Override the SDK configuration directory.")
    migrate.set_defaults(handler=_handle_config_migrate)


def _add_ui_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "ui",
        help="Launch the local MindMemOS management UI.",
        description="Start the local browser-based console for Memory, Skills, and Settings.",
    )
    parser.add_argument("--port", type=int, default=0, help="Bind port. 0 selects a free local port.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the UI in a browser automatically.")
    parser.add_argument("--config-dir", default=None, help="Override the SDK configuration directory.")
    parser.set_defaults(handler=_handle_ui)


def _add_skill_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("skill", help="Manage SDK-registered skills.")
    skill_subparsers = parser.add_subparsers(dest="skill_command", metavar="<skill-command>")
    parser.set_defaults(handler=_handle_skill_root)

    register = skill_subparsers.add_parser("register", help="Import a Skill into the centralized local repository.")
    register.add_argument("path", help="Local skill directory path or SKILL.md file path.")
    register.add_argument("--name", help="Override skill name from manifest.")
    register.add_argument("--alias", help="Local alias for later skill commands.")
    register.add_argument("--version", help="Override skill version from manifest.")
    register.add_argument("-m", "--message", help="Immutable root-version commit message.")
    register.add_argument(
        "--duplicate-action",
        choices=[item.value for item in DuplicateSkillAction],
        help="Explicitly reuse an identical snapshot or create a separate Skill family.",
    )
    register.set_defaults(handler=_handle_skill_register)

    publish = skill_subparsers.add_parser("publish", help="Create an immutable local Skill version.")
    publish.add_argument("skill", help="SDK local skill id or alias.")
    publish.add_argument("--from", dest="source_path", required=True, help="Directory or SKILL.md to snapshot once.")
    publish.add_argument("--base", dest="base_version", help="Parent version. Defaults to the latest available version.")
    publish.add_argument("--version", dest="version_label", help="Human-readable version label.")
    publish.add_argument("-m", "--message", help="Immutable version commit message.")
    publish.set_defaults(handler=_handle_skill_publish)

    list_parser = skill_subparsers.add_parser("list", help="List SDK-registered skills.")
    list_parser.set_defaults(handler=_handle_skill_list)

    show = skill_subparsers.add_parser("show", help="Show one registered skill.")
    show.add_argument("skill", help="SDK local skill id or alias.")
    show.set_defaults(handler=_handle_skill_show)

    pull = skill_subparsers.add_parser(
        "pull",
        help="Pull immutable cloud versions and lifecycle revisions.",
    )
    pull.add_argument("skill", help="SDK local skill id or alias.")
    pull.set_defaults(handler=_handle_skill_pull)

    push = skill_subparsers.add_parser(
        "push",
        help="Upload an existing immutable local version with the same UUID.",
    )
    push.add_argument("skill", help="SDK local skill id or alias.")
    push.add_argument("--version", help="Version UUID. Defaults to the latest available version.")
    push.set_defaults(handler=_handle_skill_push)

    update = skill_subparsers.add_parser(
        "update",
        help="Deprecated alias for `skill sync`.",
    )
    update_target = update.add_mutually_exclusive_group(required=True)
    update_target.add_argument("skill", nargs="?", help="SDK local skill id or alias.")
    update_target.add_argument("--all", action="store_true", help="Update all registered skills.")
    update.add_argument("--yes", "-y", action="store_true", help="Apply update without confirmation prompts.")
    update.set_defaults(handler=_handle_skill_update)

    sync = skill_subparsers.add_parser(
        "sync",
        help="Push pending versions and pull missing versions or lifecycle revisions.",
    )
    sync_target = sync.add_mutually_exclusive_group(required=True)
    sync_target.add_argument("skill", nargs="?", help="SDK local skill id or alias.")
    sync_target.add_argument("--all", action="store_true", help="Sync all registered skills.")
    sync.set_defaults(handler=_handle_skill_update)

    export = skill_subparsers.add_parser("export", help="Export the latest or selected version to a directory.")
    export.add_argument("skill", help="SDK local skill id or alias.")
    export.add_argument("--to", dest="target_path", required=True, help="Target directory.")
    export.add_argument("--version", help="Version to export. Defaults to the latest available version.")
    export.add_argument("--no-replace", action="store_true", help="Fail if the target already exists.")
    export.set_defaults(handler=_handle_skill_export)

    history = skill_subparsers.add_parser("history", help="Show skill version history.")
    history.add_argument("skill", help="SDK local skill id or alias.")
    history.set_defaults(handler=_handle_skill_history)

    diff = skill_subparsers.add_parser("diff", help="Show cached skill version differences.")
    diff.add_argument("skill", help="SDK local skill id or alias.")
    diff.add_argument("--from", dest="from_version", help="Source version. Defaults to current base version.")
    diff.add_argument("--to", dest="version", required=True, help="Target version.")
    diff.set_defaults(handler=_handle_skill_diff)

    unregister = skill_subparsers.add_parser(
        "unregister",
        help="Compatibility command for the legacy external-path registry.",
    )
    unregister.add_argument("skill", help="SDK local skill id or alias.")
    unregister.add_argument(
        "--delete-files",
        action="store_true",
        help="Deprecated and rejected: source directories are not owned by the SDK.",
    )
    unregister.add_argument("--yes", "-y", action="store_true", help="Unregister without confirmation prompts.")
    unregister.set_defaults(handler=_handle_skill_unregister)


def _add_memory_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("memory", help="Run lightweight memory API checks.")
    memory_subparsers = parser.add_subparsers(dest="memory_command", metavar="<memory-command>")
    parser.set_defaults(handler=_handle_memory_root)

    search = memory_subparsers.add_parser("search", help="Search memories using configured credentials.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    search.add_argument("--user-id", help="Override configured user id.")
    search.add_argument("--app-id", help="Request app id.")
    search.add_argument("--agent-id", help="Request agent id.")
    search.add_argument("--session-id", help="Request session id.")
    search.add_argument(
        "--search-strategy",
        default="fast",
        choices=["fast", "agentic"],
        help="Search strategy. Default: fast.",
    )
    search.add_argument("--rerank", action="store_true", help="Enable reranking.")
    search.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum rerank relevance score (0-1). Only effective with --rerank.",
    )
    search.add_argument("--filter", dest="filter_json", help="Filter DSL as a JSON object string.")
    search.add_argument("--json", action="store_true", help="Print a machine-readable JSON result.")
    search.set_defaults(handler=_handle_memory_search)

    add = memory_subparsers.add_parser("add", help="Add a dialogue message as memory using configured credentials.")
    add.add_argument("--content", help="Dialogue message content to add.")
    add.add_argument(
        "--messages-json",
        help="JSON array of memory messages. When provided, --content/--role are ignored.",
    )
    add.add_argument(
        "--messages-json-file",
        help="Path to a JSON file containing an array of memory messages.",
    )
    add.add_argument(
        "--role",
        default="user",
        choices=["user", "assistant", "system", "tool"],
        help="Dialogue role. Default: user.",
    )
    add.add_argument("--user-id", help="Override configured user id.")
    add.add_argument("--app-id", help="Request app id.")
    add.add_argument("--agent-id", help="Request agent id.")
    add.add_argument("--session-id", help="Request session id.")
    add.add_argument("--metadata-json", help="Business metadata as a JSON object string.")
    add.add_argument("--skill-context-json", help="Skill context array as JSON; overrides SDK auto-detection.")
    add.add_argument("--async", dest="async_mode", action="store_true", help="Use async add mode.")
    add.add_argument("--json", action="store_true", help="Print a machine-readable JSON result.")
    add.set_defaults(handler=_handle_memory_add)

    get = memory_subparsers.add_parser("get", help="List or filter memories in the current project.")
    get.add_argument("--filter", dest="filter_json", help="Filter DSL as a JSON object string.")
    get.add_argument("--top-k", type=int, default=None, help="Max number of memories to return.")
    get.set_defaults(handler=_handle_memory_get)

    update = memory_subparsers.add_parser("update", help="Update a memory's content by id.")
    update.add_argument("memory_id", help="Target memory id.")
    update.add_argument("--content", required=True, help="New memory content.")
    update.set_defaults(handler=_handle_memory_update)

    delete = memory_subparsers.add_parser("delete", help="Delete a memory by id.")
    delete.add_argument("memory_id", help="Target memory id.")
    delete.add_argument("--yes", "-y", action="store_true", help="Delete without confirmation prompts.")
    delete.set_defaults(handler=_handle_memory_delete)

    feedback = memory_subparsers.add_parser("feedback", help="Run the feedback flow.")
    feedback.add_argument(
        "--text",
        dest="feedback",
        help="Explicit feedback text. If omitted, the server analyzes recent adds.",
    )
    feedback.add_argument(
        "--messages-json",
        help="JSON array of messages that produced the explicit feedback.",
    )
    feedback.add_argument(
        "--messages-json-file",
        help="Path to a JSON file containing explicit feedback messages (`-` = stdin).",
    )
    feedback.add_argument(
        "--recalled-memories-json",
        help="JSON array of memories recalled in the feedback round.",
    )
    feedback.add_argument(
        "--recalled-memories-json-file",
        help="Path to a JSON file containing recalled memories (`-` = stdin).",
    )
    feedback.add_argument("--user-id", help="Override configured user id.")
    feedback.add_argument("--app-id", help="Request app id.")
    feedback.add_argument("--agent-id", help="Request agent id.")
    feedback.add_argument("--session-id", help="Request session id.")
    feedback.set_defaults(handler=_handle_memory_feedback)

    dreaming = memory_subparsers.add_parser("dreaming", help="Trigger the dreaming pipeline.")
    dreaming.add_argument("--sync", dest="sync_mode", action="store_true", help="Run dreaming synchronously.")
    dreaming.add_argument("--async", dest="async_mode", action="store_true", help="Queue dreaming asynchronously.")
    dreaming.add_argument("--user-id", help="Override configured user id.")
    dreaming.add_argument("--app-id", help="Request app id.")
    dreaming.add_argument("--agent-id", help="Request agent id.")
    dreaming.add_argument("--session-id", help="Request session id.")
    dreaming.set_defaults(handler=_handle_memory_dreaming)


def _add_doctor_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("doctor", help="Check SDK configuration and connectivity.")
    parser.set_defaults(handler=_handle_doctor)


def _handle_root(args: argparse.Namespace) -> int:
    build_parser().print_help()
    return 0


def _handle_auth(args: argparse.Namespace) -> int:
    manager = ConfigManager()
    existing = manager.load_portal() if manager.portal_exists() or manager.legacy_exists() else None
    if existing is not None:
        print(f"Existing configuration detected at {manager.portal_path}.")
        print("This will overwrite the current authentication settings.\n")

    if existing is None:
        default_base_url = DEFAULT_BASE_URL
        default_user_id = None
    else:
        profile = existing.profiles[existing.active_profile]
        default_base_url = profile.connections[profile.default_connection].base_url
        default_user_id = profile.identity.user_id
    base_url = args.base_url or _prompt(f"Base URL [{default_base_url}]: ") or default_base_url

    api_key = args.api_key or _prompt_secret("API key: ")
    if not api_key:
        print("mindmemos auth: API key is required.")
        return 2

    user_id = args.user_id or _prompt(_label("User id", default_user_id)) or default_user_id
    if not user_id:
        print("mindmemos auth: user id is required.")
        return 2

    try:
        manager.update_auth(
            base_url=base_url,
            api_key=api_key,
            user_id=user_id,
        )
    except ConfigError as exc:
        print(f"mindmemos auth: {exc}")
        return 1

    print("Configuration saved.")
    return 0


def _handle_config_root(args: argparse.Namespace) -> int:
    return _missing_subcommand("config")


def _handle_config_show(args: argparse.Namespace) -> int:
    manager = ConfigManager()
    if not manager.portal_exists() and not manager.legacy_exists():
        print(f"No SDK config at {manager.portal_path}. Run `mindmemos auth` to create it.")
        return 1
    try:
        compiled = manager.compile_portal()
    except ConfigError as exc:
        print(f"mindmemos config show: {exc}")
        return 1

    profile = compiled.profile
    connection = profile.connections[profile.default_connection]
    api_key = connection.api_key
    api_key_display = api_key if args.show_secret else mask_secret(api_key)

    print(f"config path: {manager.portal_path}")
    print(f"profile:     {compiled.active_profile}")
    print(f"connection:  {profile.default_connection}")
    print(f"base_url:    {connection.base_url}")
    print(f"api_key:     {api_key_display}")
    print(f"user_id:     {profile.identity.user_id or '(not set)'}")
    print(f"memory route:{profile.memory_connection}")
    print(f"skill route: {_skill_route_display(profile.skill_connection)}")
    return 0


def _handle_config_reset(args: argparse.Namespace) -> int:
    manager = ConfigManager()
    paths = [path for path in (manager.portal_path, manager.config_path) if path.is_file()]
    if not paths:
        print(f"No SDK config at {manager.portal_path}. Nothing to reset.")
        return 0
    if not args.yes:
        answer = _prompt(f"Reset and delete {', '.join(str(path) for path in paths)}? [y/N]: ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    manager.reset()
    print("Configuration reset.")
    return 0


def _handle_config_migrate(args: argparse.Namespace) -> int:
    manager = ConfigManager(args.config_dir)
    try:
        plan = manager.migrate_portal(dry_run=not args.apply)
        compiled = manager.compile_portal() if args.apply and manager.portal_exists() else None
        if compiled is None:
            from .config import SDKConfigCompilerV2

            compiled = SDKConfigCompilerV2().compile(plan.portal)
    except ConfigError as exc:
        print(f"mindmemos config migrate: {exc}")
        return 1

    mode = "applied" if args.apply and plan.changes_required else "dry-run"
    if not plan.changes_required:
        mode = "already-current"
    profile = compiled.profile
    print(f"migration:    {mode}")
    print(f"source:       {plan.source_path}")
    print(f"target:       {plan.target_path}")
    print(f"backup:       {plan.backup_path}")
    print(f"profile:      {compiled.active_profile}")
    print(f"memory route: {profile.memory_connection}")
    print(f"skill route:  {_skill_route_display(profile.skill_connection)}")
    print(f"ignored local Skill entries: {plan.ignored_local_skill_entries}")
    if not args.apply and plan.changes_required:
        print("No files changed. Re-run with --apply to write portal v2.")
    return 0


def _handle_skill_root(args: argparse.Namespace) -> int:
    return _missing_subcommand("skill")


def _handle_ui(args: argparse.Namespace) -> int:
    """Launch the unified local console without adding a web framework dependency."""
    from .ui.server import run_ui

    try:
        run_ui(
            host="127.0.0.1",
            port=args.port,
            open_browser=not args.no_open,
            config_dir=args.config_dir,
        )
    except KeyboardInterrupt:
        print("\nStopped local MindMemOS UI.")
    return 0


def _handle_skill_register(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        result = manager.register_local(
            RegisterLocalRequest(
                source_path=args.path,
                name=args.name,
                alias=args.alias,
                version_label=args.version,
                commit_message=args.message,
                duplicate_action=args.duplicate_action,
            )
        )
        manifest = manager.show_local(result.skill_id)
        version = manager.get_local_version(result.skill_id, result.version_id)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill register", exc)

    verb = "Reused" if result.action == "reused" else "Registered"
    print(f"{verb} {manifest.name} ({manifest.skill_id}).")
    print(f"alias:          {manifest.alias or '(none)'}")
    print(f"cloud_skill_id: {manifest.cloud_skill_id or '(local only)'}")
    print(f"version_id:     {result.version_id}")
    print(f"latest_version: {manifest.latest_version_id}")
    print(f"content_hash:   {version.content_hash}")
    return 0


def _handle_skill_publish(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        result = manager.publish_local(
            PublishLocalRequest(
                skill_id=args.skill,
                base_version_id=args.base_version,
                source_path=args.source_path,
                version_label=args.version_label,
                commit_message=args.message,
            )
        )
        manifest = manager.show_local(result.skill_id)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill publish", exc)
    print(f"Published local version {result.version_id} for {manifest.name} ({manifest.skill_id}).")
    print(f"latest_version: {result.latest_version_id}")
    print(f"snapshot_hash:  {result.local_snapshot_hash}")
    return 0


def _handle_skill_list(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        records = manager.list_local()
    except MindMemOSSDKError as exc:
        return _report_api_error("skill list", exc)
    if not records:
        print("No SDK-registered skills.")
        return 0
    for line in _format_local_skill_manifests_table(records):
        print(line)
    return 0


def _handle_skill_show(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        record = manager.show_local(args.skill)
        latest = manager.get_local_version(record.skill_id, record.latest_version_id)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill show", exc)
    print(f"skill_id:       {record.skill_id}")
    print(f"alias:          {record.alias or '(none)'}")
    print(f"name:           {record.name}")
    print(f"cloud_skill_id: {record.cloud_skill_id or '(none)'}")
    print(f"latest_version: {record.latest_version_id}")
    print(f"version_count:  {len(record.version_ids)}")
    print(f"content_hash:   {latest.content_hash}")
    print(f"snapshot_hash:  {latest.local_snapshot_hash}")
    print(f"sync_state:     {latest.sync_state.value}")
    print(f"last_sync_at:   {record.last_sync_at or '(never)'}")
    print(f"updated_at:     {record.updated_at}")
    return 0


def _handle_skill_pull(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=True)
    if manager is None:
        return 1
    try:
        versions = manager.pull_local(args.skill)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill pull", exc)
    print(f"Pulled {len(versions)} version(s).")
    for version in versions:
        cloud_status = version.cloud_status.value if version.cloud_status else "(unknown)"
        print(f"- {version.version_id} cloud={cloud_status} sync={version.sync_state.value} {version.content_hash}")
    return 0


def _handle_skill_push(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=True)
    if manager is None:
        return 1
    try:
        result = manager.push_local(args.skill, version_id=args.version)
        manifest = manager.show_local(args.skill)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill push", exc)
    print(f"Pushed {manifest.name} ({manifest.skill_id}) version {result.version_id}.")
    print(f"cloud_skill_id: {result.cloud_skill_id}")
    print(f"content_hash:   {result.content_hash}")
    return 0


def _handle_skill_update(args: argparse.Namespace) -> int:
    if not args.skill and not args.all:
        print("mindmemos skill update: provide a skill id/alias or --all.")
        return 2
    manager = _build_skill_manager(require_api_key=True)
    if manager is None:
        return 1
    command = "skill sync" if getattr(args, "skill_command", None) == "sync" else "skill update"
    try:
        skill_refs = [record.skill_id for record in manager.list_local()] if args.all else [args.skill]
        updated = [manager.sync_local(skill_ref) for skill_ref in skill_refs]
    except MindMemOSSDKError as exc:
        return _report_api_error(command, exc)
    for record in updated:
        print(f"Synced {record.name} ({record.skill_id}); latest={record.latest_version_id}.")
    return 0


def _handle_skill_export(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        result = manager.export_local(
            ExportSkillRequest(
                skill_id=args.skill,
                version_id=args.version,
                target_path=args.target_path,
                replace=not args.no_replace,
            )
        )
    except MindMemOSSDKError as exc:
        return _report_api_error("skill export", exc)
    print(f"Exported {result.skill_id}@{result.version_id} to {result.target_path}.")
    print(f"files:         {len(result.exported_files)}")
    print(f"snapshot_hash: {result.local_snapshot_hash}")
    return 0


def _handle_skill_history(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        versions = manager.local_history(args.skill)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill history", exc)
    if not versions:
        print("No local skill history.")
        return 0
    for version in versions:
        parents = ",".join(version.parent_version_ids) or "(root)"
        label = version.version_label or "(none)"
        print(
            f"{version.version_id} parents={parents} sync={version.sync_state.value} "
            f"origin={version.origin.value} version={label}"
        )
        if version.commit_message:
            print(f"  {version.commit_message}")
    return 0


def _handle_skill_diff(args: argparse.Namespace) -> int:
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        result = manager.diff_local(args.skill, from_version_id=args.from_version, to_version_id=args.version)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill diff", exc)
    if result.diff:
        print(result.diff, end="" if result.diff.endswith("\n") else "\n")
    else:
        print(f"No differences between {result.from_version_id} and {result.to_version_id}.")
    return 0


def _handle_skill_unregister(args: argparse.Namespace) -> int:
    if args.delete_files:
        print("mindmemos skill unregister: --delete-files is no longer supported; source directories are not owned by the SDK.")
        return 2
    if not args.yes:
        suffix = " and delete local files" if args.delete_files else ""
        answer = _prompt(f"Unregister skill {args.skill}{suffix}? [y/N]: ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    manager = _build_skill_manager(require_api_key=False)
    if manager is None:
        return 1
    try:
        removed = manager.unregister(args.skill)
    except MindMemOSSDKError as exc:
        return _report_api_error("skill unregister", exc)
    name = getattr(removed, "name", None) or getattr(removed, "skill_name", "unknown")
    print(f"Unregistered {name} ({removed.skill_id}).")
    return 0


def _handle_memory_root(args: argparse.Namespace) -> int:
    return _missing_subcommand("memory")


def _handle_memory_search(args: argparse.Namespace) -> int:
    filters = _parse_json_object(args.filter_json, command="memory search", option="--filter")
    if filters is _JSON_PARSE_ERROR:
        return 2

    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            result = client.memory.search(
                args.query,
                **_without_none(
                    top_k=args.top_k,
                    user_id=args.user_id,
                    search_strategy=args.search_strategy,
                    rerank=args.rerank,
                    score_threshold=args.score_threshold,
                    filters=filters,
                    app_id=args.app_id,
                    agent_id=args.agent_id,
                    session_id=args.session_id,
                ),
            )
    except MindMemOSSDKError as exc:
        return _report_api_error("memory search", exc)

    if args.json:
        _print_json(result.model_dump())
        return 0
    if not result.memories:
        print("No memories found.")
        return 0
    for i, hit in enumerate(result.memories, start=1):
        when = hit.last_update_at or ""
        print(f"{i}. [{hit.id}] {hit.memory}" + (f"  ({when})" if when else ""))
    return 0


def _handle_memory_add(args: argparse.Namespace) -> int:
    messages = _parse_add_messages(args)
    if messages is None:
        return 2
    metadata = _parse_json_object(args.metadata_json, command="memory add", option="--metadata-json")
    if metadata is _JSON_PARSE_ERROR:
        return 2
    skill_context = _parse_json_array(args.skill_context_json, command="memory add", option="--skill-context-json")
    if skill_context is _JSON_PARSE_ERROR:
        return 2

    client = _build_client()
    if client is None:
        return 1
    mode = "async" if args.async_mode else "sync"
    try:
        with client:
            result = client.memory.add(
                messages=messages,
                user_id=args.user_id,
                mode=mode,
                app_id=args.app_id,
                agent_id=args.agent_id,
                session_id=args.session_id,
                metadata=metadata,
                skill_context=skill_context,
            )
    except MindMemOSSDKError as exc:
        return _report_api_error("memory add", exc)

    if args.json:
        _print_json(result.model_dump())
        return 0
    if result.code == "queued":
        print(f"Queued for async processing. request_id={result.request_id or '(none)'}")
        return 0
    if not result.memories:
        print(f"Done. No memories extracted. request_id={result.request_id or '(none)'}")
        return 0
    print(f"Added {len(result.memories)} memory item(s):")
    for item in result.memories:
        mem_id = item.memory_id or "(pending)"
        print(f"- [{item.operation}] {mem_id}: {item.content}")
    return 0


def _parse_add_messages(args: argparse.Namespace) -> list[DialogueMessage | dict[str, Any]] | None:
    """Parse memory add input as a single text message or JSON message list."""
    if args.messages_json is not None or args.messages_json_file is not None:
        messages = _parse_message_json_options(
            command="memory add",
            messages_json=args.messages_json,
            messages_json_file=args.messages_json_file,
        )
        if messages is _JSON_PARSE_ERROR:
            return None
        return messages

    if not args.content:
        print("mindmemos memory add: either --content or --messages-json is required.")
        return None
    return [DialogueMessage(role=args.role, content=args.content, timestamp=_now_millis())]


def _parse_message_json_options(
    *,
    command: str,
    messages_json: str | None,
    messages_json_file: str | None,
) -> list[dict[str, Any]] | object:
    value = _parse_json_array_from_options(
        command=command,
        inline_json=messages_json,
        file_path=messages_json_file,
        inline_option="--messages-json",
        file_option="--messages-json-file",
    )
    if value is _JSON_PARSE_ERROR:
        return _JSON_PARSE_ERROR
    if not value:
        print(f"mindmemos {command}: messages JSON must be a non-empty JSON array.")
        return _JSON_PARSE_ERROR
    if not all(isinstance(message, dict) for message in value):
        print(f"mindmemos {command}: every messages JSON item must be a JSON object.")
        return _JSON_PARSE_ERROR
    return value


def _parse_json_array_from_options(
    *,
    command: str,
    inline_json: str | None,
    file_path: str | None,
    inline_option: str,
    file_option: str,
) -> list[Any] | object:
    if inline_json is not None and file_path is not None:
        print(f"mindmemos {command}: use only one of {inline_option} or {file_option}.")
        return _JSON_PARSE_ERROR
    if inline_json is None and file_path is None:
        return _JSON_PARSE_ERROR

    if file_path is not None:
        if file_path == "-":
            raw = sys.stdin.read()
        else:
            try:
                raw = Path(file_path).read_text(encoding="utf-8")
            except OSError as exc:
                print(f"mindmemos {command}: failed to read {file_option}: {exc}")
                return _JSON_PARSE_ERROR
        source = file_option
    else:
        raw = inline_json or ""
        source = inline_option

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"mindmemos {command}: invalid {source}: {exc}")
        return _JSON_PARSE_ERROR
    if not isinstance(value, list):
        print(f"mindmemos {command}: {source} must be a JSON array.")
        return _JSON_PARSE_ERROR
    return value


def _parse_json_object(raw: str | None, *, command: str, option: str) -> dict[str, Any] | None | object:
    """Parse a CLI JSON object option."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"mindmemos {command}: invalid {option} JSON: {exc}")
        return _JSON_PARSE_ERROR
    if not isinstance(value, dict):
        print(f"mindmemos {command}: {option} must be a JSON object.")
        return _JSON_PARSE_ERROR
    return value


def _parse_json_array(raw: str | None, *, command: str, option: str) -> list[Any] | None | object:
    """Parse a CLI JSON array option."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"mindmemos {command}: invalid {option} JSON: {exc}")
        return _JSON_PARSE_ERROR
    if not isinstance(value, list):
        print(f"mindmemos {command}: {option} must be a JSON array.")
        return _JSON_PARSE_ERROR
    return value


def _handle_memory_get(args: argparse.Namespace) -> int:
    filters = _parse_json_object(args.filter_json, command="memory get", option="--filter")
    if filters is _JSON_PARSE_ERROR:
        return 2

    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            result = client.memory.get(filters=filters, top_k=args.top_k)
    except MindMemOSSDKError as exc:
        return _report_api_error("memory get", exc)

    if not result.memories:
        print("No memories found.")
        return 0
    for i, hit in enumerate(result.memories, start=1):
        when = hit.last_update_at or ""
        print(f"{i}. [{hit.id}] {hit.memory}" + (f"  ({when})" if when else ""))
    return 0


def _handle_memory_update(args: argparse.Namespace) -> int:
    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            result = client.memory.update(args.memory_id, args.content)
    except MindMemOSSDKError as exc:
        return _report_api_error("memory update", exc)
    print(_status_line("Updated", args.memory_id, result))
    return 0


def _handle_memory_delete(args: argparse.Namespace) -> int:
    if not args.yes:
        answer = _prompt(f"Delete memory {args.memory_id}? [y/N]: ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            result = client.memory.delete(args.memory_id)
    except MindMemOSSDKError as exc:
        return _report_api_error("memory delete", exc)
    print(_status_line("Deleted", args.memory_id, result))
    return 0


def _handle_memory_feedback(args: argparse.Namespace) -> int:
    messages: list[dict[str, Any]] | None = None
    recalled_memories: list[dict[str, Any]] | None = None
    has_messages_input = args.messages_json is not None or args.messages_json_file is not None
    has_recalled_input = args.recalled_memories_json is not None or args.recalled_memories_json_file is not None

    if args.feedback:
        if not has_messages_input:
            print(
                "mindmemos memory feedback: --text requires --messages-json or --messages-json-file; "
                "omit --text to run implicit feedback."
            )
            return 2
        parsed_messages = _parse_message_json_options(
            command="memory feedback",
            messages_json=args.messages_json,
            messages_json_file=args.messages_json_file,
        )
        if parsed_messages is _JSON_PARSE_ERROR:
            return 2
        messages = parsed_messages

        if has_recalled_input:
            parsed_recalled = _parse_json_array_from_options(
                command="memory feedback",
                inline_json=args.recalled_memories_json,
                file_path=args.recalled_memories_json_file,
                inline_option="--recalled-memories-json",
                file_option="--recalled-memories-json-file",
            )
            if parsed_recalled is _JSON_PARSE_ERROR:
                return 2
            if not all(isinstance(memory, dict) for memory in parsed_recalled):
                print("mindmemos memory feedback: every recalled memories JSON item must be a JSON object.")
                return 2
            recalled_memories = parsed_recalled
    elif has_messages_input or has_recalled_input:
        print("mindmemos memory feedback: context options require --text; omit them to run implicit feedback.")
        return 2

    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            result = client.memory.feedback(
                **_without_none(
                    feedback=args.feedback,
                    user_id=args.user_id,
                    app_id=args.app_id,
                    agent_id=args.agent_id,
                    session_id=args.session_id,
                    messages=messages,
                    recalled_memories=recalled_memories,
                )
            )
    except MindMemOSSDKError as exc:
        return _report_api_error("memory feedback", exc)
    print(_status_line("Feedback accepted", None, result))
    return 0


def _handle_memory_dreaming(args: argparse.Namespace) -> int:
    if args.sync_mode and args.async_mode:
        print("error: --sync and --async are mutually exclusive", file=sys.stderr)
        return 2
    mode = "sync" if args.sync_mode else "async"
    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            result = client.memory.dreaming(
                **_without_none(
                    mode=mode,
                    user_id=args.user_id,
                    app_id=args.app_id,
                    agent_id=args.agent_id,
                    session_id=args.session_id,
                )
            )
    except MindMemOSSDKError as exc:
        return _report_api_error("memory dreaming", exc)
    print(_status_line("Dreaming triggered", None, result))
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    del args
    manager = ConfigManager()
    if not manager.portal_exists() and not manager.legacy_exists():
        print(f"config: missing ({manager.portal_path})")
        print("Run `mindmemos auth` first.")
        return 1
    try:
        compiled = manager.compile_portal()
    except ConfigError as exc:
        print(f"config: invalid: {exc}")
        return 1
    profile = compiled.profile
    connection = profile.connections[profile.default_connection]
    print(f"config: ok ({manager.portal_path})")
    print(f"profile: {compiled.active_profile}")
    print(f"base_url: {connection.base_url}")
    print(f"api_key: {'configured' if connection.api_key else 'missing'}")
    print(f"user_id: {profile.identity.user_id or '(not set)'}")
    print(f"memory route: {profile.memory_connection}")
    print(f"skill route: {_skill_route_display(profile.skill_connection)}")
    if not connection.api_key:
        return 1
    try:
        client = MindMemOSClient(config_manager=manager)
        client.require_api_key()
        client.close()
    except MindMemOSSDKError as exc:
        print(f"transport: not ready: {exc}")
        return 1
    print("transport: ready")
    return 0


def _build_client() -> MindMemOSClient | None:
    """Build a client from local configuration."""
    manager = ConfigManager()
    if not manager.legacy_exists() and not manager.portal_exists():
        print(f"No SDK config at {manager.config_path}. Run `mindmemos auth` first.")
        return None
    try:
        client = MindMemOSClient(config_manager=manager)
    except ConfigError as exc:
        print(f"mindmemos: {exc}")
        return None
    try:
        client.require_api_key()
    except ConfigError:
        client.close()
        print("No api_key configured. Run `mindmemos auth` first.")
        return None
    return client


def _build_skill_manager(*, require_api_key: bool) -> SkillManager | None:
    """Build the skill manager, optionally requiring cloud credentials."""

    config_manager = ConfigManager()
    try:
        if config_manager.portal_exists() or config_manager.legacy_exists():
            profile = config_manager.compile_portal().profile
        else:
            profile = SDKConfigCompilerV2().compile(config_manager.default_portal()).profile
    except ConfigError as exc:
        print(f"mindmemos skill: {exc}")
        return None
    skill_connection = profile.skill_connection
    if require_api_key and skill_connection is not None and not profile.connections[skill_connection].api_key:
        print("No api_key configured. Run `mindmemos auth` first.")
        return None
    manager = SkillManager.from_config_manager(config_manager)
    active = _CLI_SKILL_MANAGERS.get()
    if active is not None:
        active.append(manager)
    return manager


def _skill_route_display(connection: str | None) -> str:
    return connection if connection is not None else "(disabled)"


def _format_skill_record(record: SkillRecord) -> str:
    cloud = record.cloud_skill_id or "(local)"
    version = record.base_version_id or "(unregistered)"
    alias = record.alias or "(none)"
    return (
        f"{record.skill_id}  {alias}  {record.skill_name}  {version}  {cloud}  {record.hash_state.value}  {record.path}"
    )


def _format_skill_records_table(records: Sequence[SkillRecord]) -> list[str]:
    """Format registered skills as a readable table."""

    headers = ("skill_id", "alias", "name", "base_version_id", "cloud_skill_id", "hash_state", "path")
    rows = [
        (
            record.skill_id,
            record.alias or "(none)",
            record.skill_name,
            record.base_version_id or "(unregistered)",
            record.cloud_skill_id or "(local)",
            record.hash_state.value,
            record.path,
        )
        for record in records
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    return [format_row(headers), format_row(tuple("-" * width for width in widths)), *(format_row(row) for row in rows)]


def _format_local_skill_manifests_table(records: Sequence[LocalSkillManifest]) -> list[str]:
    """Format centralized Skill families without exposing external paths."""

    headers = ("skill_id", "alias", "name", "latest_version_id", "versions", "last_sync_at")
    rows = [
        (
            record.skill_id,
            record.alias or "(none)",
            record.name,
            record.latest_version_id,
            str(len(record.version_ids)),
            record.last_sync_at or "(never)",
        )
        for record in records
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    return [format_row(headers), format_row(tuple("-" * width for width in widths)), *(format_row(row) for row in rows)]


    print(f"content_hash:   {plan.from_content_hash or '(none)'} -> {plan.to_content_hash}")
    print(f"backup_path:    {plan.backup_path or '(pending)'}")
    print(f"files:          {', '.join(plan.files) if plan.files else '(none)'}")


def _report_api_error(command: str, exc: MindMemOSSDKError) -> int:
    """Print an API or SDK error and return exit code 1."""
    request_id = getattr(exc, "request_id", None)
    suffix = f" (request_id={request_id})" if request_id else ""
    print(f"mindmemos {command}: {exc}{suffix}")
    return 1


def _now_millis() -> int:
    """Return the current 13-digit timestamp in milliseconds."""
    return int(time.time() * 1000)


def _status_line(action: str, target: str | None, result: StatusResult) -> str:
    """Build one status-only output line."""
    line = f"{action} {target}." if target else f"{action}."
    if result.message:
        line += f" {result.message}"
    if result.request_id:
        line += f" (request_id={result.request_id})"
    return line


def _print_json(payload: Any) -> None:
    """Print deterministic JSON for plugins and scripts."""
    print(json.dumps(payload, ensure_ascii=False, default=str))


def _prompt(message: str) -> str:
    """Read one line of user input."""
    try:
        return input(message).strip()
    except EOFError:
        return ""


def _prompt_secret(message: str) -> str:
    """Read a secret without echoing it to the terminal."""
    try:
        return getpass(message).strip()
    except EOFError:
        return ""


def _label(field: str, default: str | None) -> str:
    """Handle label."""
    if default:
        return f"{field} [{default}]: "
    return f"{field}: "


def _missing_subcommand(command: str) -> int:
    print(f"mindmemos {command}: missing subcommand. Use --help to see available commands.")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    managers: list[SkillManager] = []
    token = _CLI_SKILL_MANAGERS.set(managers)
    try:
        return args.handler(args)
    finally:
        for manager in reversed(managers):
            close = getattr(manager, "close", None)
            if close is not None:
                close()
        _CLI_SKILL_MANAGERS.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
