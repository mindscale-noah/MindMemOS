"""Command line interface for the MindMemOS SDK.

Commands reuse the SDK clients for configured Memory and Skill services.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from getpass import getpass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .client import MindMemOSClient
from .config import DEFAULT_BASE_URL, ConfigManager, SDKConfig, mask_secret
from .config.models import AuthConfig, MemoryServiceConfig, SkillServiceConfig
from .config.paths import set_default_mindmemos_home
from .errors import ConfigError, InvalidRequestError, MindMemOSSDKError
from .memory import DialogueMessage, ImageRef, StatusResult
from .trajectory import TrajectoryEvent

_JSON_PARSE_ERROR = object()

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class _ImageInputError(Exception):
    """A --image/--query-image value is unreadable or not a supported image."""


def _sniff_image_mime(data: bytes) -> str | None:
    """Detect the image type by magic bytes (the server checks content, not extension)."""

    for magic, mime_type in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime_type
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def _load_image_ref(value: str) -> ImageRef:
    """Build an ImageRef from an http(s) URL or a local file (inlined as base64)."""

    if value.startswith(("http://", "https://")):
        return ImageRef(url=value)
    try:
        data = Path(value).read_bytes()
    except OSError as exc:
        raise _ImageInputError(f"failed to read image {value}: {exc}") from exc
    mime_type = _sniff_image_mime(data) or _EXT_MIME.get(Path(value).suffix.lower())
    if mime_type is None:
        raise _ImageInputError(f"unrecognized image format: {value} (expected png/jpeg/gif/webp/bmp)")
    return ImageRef(data=base64.b64encode(data).decode("ascii"), mime_type=mime_type)


def _load_image_refs(values: Sequence[str] | None) -> list[ImageRef] | None:
    if not values:
        return None
    return [_load_image_ref(value) for value in values]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically (temp file + fsync + replace)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _without_none(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = _CLIParser(
        prog="mindmemos",
        description="MindMemOS SDK command line interface.",
    )
    parser.set_defaults(handler=_handle_root)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_auth_command(subparsers)
    _add_config_commands(subparsers)
    _add_memory_commands(subparsers)
    _add_trajectory_commands(subparsers)
    from .skills.cli import add_skill_commands

    add_skill_commands(subparsers)
    _add_doctor_command(subparsers)
    return parser


class _CLIParser(argparse.ArgumentParser):
    """Keep errors for script-facing commands machine-readable."""

    structured_errors = False
    """Whether the root invocation selected a command with JSON errors."""

    def parse_args(self, args=None, namespace=None):
        """Parse CLI arguments and remember whether to emit JSON errors.

        Args:
            args: Explicit arguments, or None to use the process arguments.
            namespace: Optional argparse destination namespace.

        Returns:
            Parsed argument namespace.
        """
        values = list(sys.argv[1:] if args is None else args)
        self.structured_errors = bool(values and values[0] in {"skill", "trajectory"})
        return super().parse_args(values, namespace)

    def error(self, message):
        """Emit JSON for Skill and trajectory parsing failures.

        Args:
            message: Parser-generated validation explanation.
        """
        if self.prog.startswith(("mindmemos skill", "mindmemos trajectory")) or self.structured_errors:
            print(json.dumps({"code": "invalid_argument", "message": message}), file=sys.stderr)
            self.exit(2)
        super().error(message)


def _add_auth_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "auth",
        help="Configure official or custom Memory and Skill connections.",
        description="Choose a deployment and configure Memory and Skill service credentials.",
    )
    parser.add_argument("--service-type", choices=["official", "custom"], help="Official cloud or custom deployment.")
    parser.add_argument(
        "--memory-base-url", "--base-url", dest="memory_base_url", help="Custom Memory service URL before /v1/memory."
    )
    parser.add_argument(
        "--memory-api-key", "--api-key", dest="memory_api_key", help="Memory API key; prompted if omitted."
    )
    skill_destination = parser.add_mutually_exclusive_group()
    skill_destination.add_argument(
        "--skill-base-url", help="Custom Skill API root, e.g. http://localhost:8001/v1/skills."
    )
    skill_destination.add_argument("--no-skill", action="store_true", help="Clear the custom Skill connection.")
    parser.add_argument("--skill-api-key", help="API key for an independent Skill service.")
    parser.add_argument("--user-id", help="Default user id. If omitted, prompt interactively.")
    parser.set_defaults(handler=_handle_auth)


def _add_config_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("config", help="Manage local SDK settings and the default home.")
    config_subparsers = parser.add_subparsers(dest="config_command", metavar="<config-command>")
    parser.set_defaults(handler=_handle_config_root)

    show = config_subparsers.add_parser("show", help="Show current SDK settings.")
    show.add_argument("--show-secret", action="store_true", help="Show the full API key.")
    show.set_defaults(handler=_handle_config_show)

    reset = config_subparsers.add_parser("reset", help="Reset local SDK settings.")
    reset.add_argument("--yes", "-y", action="store_true", help="Reset without confirmation prompts.")
    reset.set_defaults(handler=_handle_config_reset)

    home = config_subparsers.add_parser("home", help="Manage the default configuration directory.")
    home_subparsers = home.add_subparsers(dest="home_command", metavar="<home-command>")
    home.set_defaults(handler=_handle_config_home_root)
    set_home = home_subparsers.add_parser("set", help="Save the default directory for future processes.")
    set_home.add_argument("directory", help="Absolute SDK configuration directory.")
    set_home.set_defaults(handler=_handle_config_home_set)


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
    search.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Strict token budget; enables token-budget retention post-processing.",
    )
    search.add_argument("--filter", dest="filter_json", help="Filter DSL as a JSON object string.")
    search.add_argument(
        "--query-image",
        dest="query_images",
        action="append",
        metavar="PATH_OR_URL",
        help="Attach an image to the query (local file or http(s) URL). Repeatable.",
    )
    search.add_argument(
        "--multimodal",
        action="store_true",
        help="Also search stored images (implied by --query-image; needs multimodal mode).",
    )
    search.add_argument("--json", action="store_true", help="Print a machine-readable JSON result.")
    search.set_defaults(handler=_handle_memory_search)

    add = memory_subparsers.add_parser("add", help="Add a dialogue message as memory using configured credentials.")
    add.add_argument("--content", help="Dialogue message content to add.")
    add.add_argument(
        "--image",
        dest="images",
        action="append",
        metavar="PATH_OR_URL",
        help="Attach an image to the --content message (local file or http(s) URL). Repeatable.",
    )
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
    add.add_argument("--async", dest="async_mode", action="store_true", help="Use async add mode.")
    add.add_argument("--json", action="store_true", help="Print a machine-readable JSON result.")
    add.set_defaults(handler=_handle_memory_add)

    get = memory_subparsers.add_parser("get", help="List or filter memories in the current project.")
    get.add_argument("--filter", dest="filter_json", help="Filter DSL as a JSON object string.")
    get.add_argument("--top-k", type=int, default=None, help="Max number of memories to return.")
    get.set_defaults(handler=_handle_memory_get)

    get_image = memory_subparsers.add_parser(
        "get-image",
        help="Download a stored image by blob id (metadata.blob_id of a search image result).",
    )
    get_image.add_argument("blob_id", help="Image blob id from a search result.")
    get_image.add_argument(
        "--user-id",
        required=True,
        help="Owner of the image (metadata.user_id of the search image result); verified by the server.",
    )
    get_image.add_argument(
        "-o",
        "--output",
        help="Output file path. Default: ./{blob_id}{extension by content type}.",
    )
    get_image.set_defaults(handler=_handle_memory_get_image)

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


def _add_trajectory_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("trajectory", help="Report one execution to Memory and Skill.")
    commands = parser.add_subparsers(dest="trajectory_command", required=True)
    report = commands.add_parser("report", help="Send a trajectory event and read independent delivery receipts.")
    report.add_argument("--file", required=True, help="Path to the trajectory event JSON, or - for standard input.")
    destinations = report.add_mutually_exclusive_group()
    destinations.add_argument("--skill-only", action="store_true", help="Send only the shared Skill trajectory.")
    destinations.add_argument("--memory-only", action="store_true", help="Send only the Memory projection.")
    report.add_argument(
        "--json", action="store_true", help="Print machine-readable receipts, including partial results."
    )
    report.set_defaults(handler=_handle_trajectory_report)


def _add_doctor_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("doctor", help="Check SDK configuration and connectivity.")
    parser.set_defaults(handler=_handle_doctor)


def _handle_root(args: argparse.Namespace) -> int:
    build_parser().print_help()
    return 0


def _handle_auth(args: argparse.Namespace) -> int:
    manager = ConfigManager()
    existing = manager.load_or_default() if manager.exists() else None
    if existing is not None:
        print(f"Updating SDK connections at {manager.config_path}.")

    default_type = existing.service_type if existing is not None else "official"
    if args.service_type is not None:
        service_type = args.service_type
    else:
        while True:
            service_type = _prompt(f"Deployment [official/custom] [{default_type}]: ").lower() or default_type
            if service_type in {"official", "custom"}:
                break
            print("mindmemos auth: deployment must be official or custom.")

    if service_type == "official":
        if args.memory_base_url is not None or args.skill_base_url is not None or args.no_skill:
            print("mindmemos auth: official deployment does not accept custom service URLs.")
            return 2
        memory_url = DEFAULT_BASE_URL
        skill_url = None
        previous_skill_url = None
    else:
        if args.no_skill and args.skill_api_key is not None:
            print("mindmemos auth: --no-skill cannot be used with --skill-api-key.")
            return 2
        previous_memory_url = (
            existing.memory_service.base_url if existing is not None and existing.service_type == "custom" else None
        )
        try:
            memory_url = _auth_url(
                args.memory_base_url,
                "Memory service URL",
                _label("Memory service URL", previous_memory_url),
                default=previous_memory_url,
            )
        except ValueError as exc:
            print(f"mindmemos auth: {exc}")
            return 2

        previous_skill_url = existing.skill_service.base_url if existing is not None else None
        if args.no_skill:
            skill_url = None
        elif args.service_type is not None and args.skill_base_url is None:
            skill_url = previous_skill_url
        else:
            try:
                skill_url = _auth_url(
                    args.skill_base_url,
                    "Skill service URL",
                    _label("Skill service URL", previous_skill_url)
                    if previous_skill_url
                    else "Skill service URL [skip]: ",
                    default=previous_skill_url,
                    allow_empty=True,
                )
            except ValueError as exc:
                print(f"mindmemos auth: {exc}")
                return 2
        if skill_url is None and args.skill_api_key is not None:
            print("mindmemos auth: a custom Skill API key requires a Skill service URL.")
            return 2

    previous_memory_key = existing.memory_service.auth.api_key if existing is not None else None
    previous_memory_destination = existing.resolve_memory_connection()[0] if existing is not None else None
    matching_memory_key = previous_memory_key if memory_url == previous_memory_destination else None
    has_skill_key = bool((args.skill_api_key or "").strip())
    if has_skill_key and args.memory_api_key is None:
        memory_key = matching_memory_key
    else:
        memory_key = _auth_key(args.memory_api_key, matching_memory_key, "Memory API key")
    if not memory_key and not has_skill_key:
        print("mindmemos auth: Memory API key is required.")
        return 2

    skill_key = None
    previous_skill_auth = existing.skill_service.auth if existing is not None else None
    previous_skill_key = previous_skill_auth.api_key if previous_skill_auth is not None else None
    if skill_url is not None:
        matching_key = previous_skill_key if skill_url == previous_skill_url else None
        if args.service_type is not None and args.skill_base_url is None and args.skill_api_key is None:
            skill_key = matching_key
        else:
            skill_key = _auth_key(args.skill_api_key, matching_key, "Skill API key")
        if not skill_key:
            print("mindmemos auth: independent Skill API key is required.")
            return 2
    elif service_type == "official":
        skill_key = (args.skill_api_key or "").strip() or (
            previous_skill_key if existing is not None and existing.service_type == "official" else None
        )

    default_user_id = existing.defaults.user_id if existing is not None else None
    if args.user_id is not None:
        user_id = args.user_id.strip()
        if not user_id:
            print("mindmemos auth: user id is required.")
            return 2
    else:
        while True:
            user_id = _prompt(_label("User id", default_user_id)) or default_user_id
            if user_id:
                break
            print("mindmemos auth: user id is required.")

    try:
        config = existing.model_copy(deep=True) if existing is not None else SDKConfig()
        config.service_type = service_type
        config.memory_service.base_url = None if service_type == "official" else memory_url
        config.memory_service.auth.api_key = memory_key
        config.skill_service = SkillServiceConfig(
            base_url=skill_url,
            auth=AuthConfig(api_key=skill_key) if skill_key else None,
        )
        config.defaults.user_id = user_id
        manager.save(config)
        saved = manager.load()
    except ConfigError as exc:
        print(f"mindmemos auth: {exc}")
        return 1

    saved_memory_url, _ = saved.resolve_memory_connection()
    saved_skill_url, _ = saved.resolve_skill_connection()
    print(f"Configuration saved at {manager.config_path}.")
    print(f"Deployment: {saved.service_type}")
    print(f"Memory service: {saved_memory_url}")
    print(f"Skill service: {saved_skill_url or '(not configured)'}")
    return 0


def _auth_url(
    provided: str | None,
    label: str,
    prompt: str,
    *,
    default: str | None = None,
    allow_empty: bool = False,
) -> str | None:
    """Validate a service URL before moving to the next auth field."""
    while True:
        raw = provided if provided is not None else _prompt(prompt)
        candidate = raw.strip()
        if not candidate and provided is None:
            candidate = default
        if not candidate and allow_empty and provided is None:
            return None
        if not candidate:
            message = f"{label} is required."
        else:
            try:
                normalized = MemoryServiceConfig(base_url=candidate).base_url
            except ValidationError:
                message = f"Invalid {label}: enter an HTTP URL without credentials, query, or fragment."
            else:
                if normalized is not None:
                    return normalized
                message = f"{label} is required."
        if provided is not None:
            raise ValueError(message)
        print(f"mindmemos auth: {message}")


def _auth_key(provided: str | None, previous: str | None, label: str) -> str:
    """Read a key, keeping the prior key only when its destination is unchanged."""
    if provided is not None:
        return provided.strip()
    suffix = " [Enter keeps current key]" if previous else ""
    while True:
        key = _prompt_secret(f"{label}{suffix}: ") or previous
        if key:
            return key
        print(f"mindmemos auth: {label} is required.")


def _handle_config_root(args: argparse.Namespace) -> int:
    return _missing_subcommand("config")


def _handle_config_home_root(args: argparse.Namespace) -> int:
    return _missing_subcommand("config home")


def _handle_config_home_set(args: argparse.Namespace) -> int:
    try:
        home = set_default_mindmemos_home(args.directory)
    except (OSError, ValueError) as exc:
        print(f"mindmemos config home set: {exc}")
        return 2
    print(f"Default SDK home saved: {home}")
    print("Restart running MindMemOS processes to use the new default.")
    return 0


def _handle_config_show(args: argparse.Namespace) -> int:
    manager = ConfigManager()
    if not manager.exists():
        print(f"No SDK config at {manager.config_path}. Run `mindmemos auth` to create it.")
        return 1
    try:
        config = manager.load()
    except ConfigError as exc:
        print(f"mindmemos config show: {exc}")
        return 1

    memory_url, memory_key = config.resolve_memory_connection()
    skill_url, skill_key = config.resolve_skill_connection()
    memory_key_display = memory_key if args.show_secret else mask_secret(memory_key)
    skill_key_display = skill_key if args.show_secret else mask_secret(skill_key)

    print(f"config path: {manager.config_path}")
    print(f"deployment:  {config.service_type}")
    print(f"memory URL:  {memory_url}")
    print(f"memory key:  {memory_key_display}")
    print(f"skill URL:   {skill_url or '(not configured)'}")
    print(f"skill key:   {skill_key_display}")
    print(f"user_id:     {config.defaults.user_id or '(not set)'}")
    return 0


def _handle_config_reset(args: argparse.Namespace) -> int:
    manager = ConfigManager()
    if not manager.exists():
        print(f"No SDK config at {manager.config_path}. Nothing to reset.")
        return 0
    if not args.yes:
        answer = _prompt(f"Reset the current SDK connection settings at {manager.config_path}? [y/N]: ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    manager.reset()
    print("Configuration reset.")
    return 0


def _handle_memory_root(args: argparse.Namespace) -> int:
    return _missing_subcommand("memory")


def _handle_memory_search(args: argparse.Namespace) -> int:
    filters = _parse_json_object(args.filter_json, command="memory search", option="--filter")
    if filters is _JSON_PARSE_ERROR:
        return 2
    try:
        query_images = _load_image_refs(args.query_images)
    except _ImageInputError as exc:
        print(f"mindmemos memory search: {exc}")
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
                    token_budget=args.token_budget,
                    filters=filters,
                    app_id=args.app_id,
                    agent_id=args.agent_id,
                    session_id=args.session_id,
                    query_images=query_images,
                    multimodal=args.multimodal or None,
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
        image_url = (hit.metadata or {}).get("image_url")
        if image_url:
            print(f"   image: {image_url}")
    return 0


def _handle_memory_add(args: argparse.Namespace) -> int:
    messages = _parse_add_messages(args)
    if messages is None:
        return 2
    if args.images:
        if args.messages_json is not None or args.messages_json_file is not None:
            print("mindmemos memory add: --image attaches to --content; embed images in --messages-json instead.")
            return 2
        try:
            images = _load_image_refs(args.images)
        except _ImageInputError as exc:
            print(f"mindmemos memory add: {exc}")
            return 2
        messages[0] = messages[0].model_copy(update={"images": images})
    metadata = _parse_json_object(args.metadata_json, command="memory add", option="--metadata-json")
    if metadata is _JSON_PARSE_ERROR:
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


def _handle_memory_get_image(args: argparse.Namespace) -> int:
    client = _build_client()
    if client is None:
        return 1
    try:
        with client:
            data, mime_type = client.memory.get_image(args.blob_id, args.user_id)
    except MindMemOSSDKError as exc:
        return _report_api_error("memory get-image", exc)

    output = Path(args.output) if args.output else Path(f"{args.blob_id}{_MIME_EXT.get(mime_type, '.bin')}")
    try:
        _atomic_write_bytes(output, data)
    except OSError as exc:
        print(f"mindmemos memory get-image: failed to write {output}: {exc}")
        return 1
    print(f"Saved {len(data)} bytes to {output} ({mime_type}).")
    return 0


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


def _handle_trajectory_report(args: argparse.Namespace) -> int:
    """Read a trajectory event and report the result of each destination write."""
    try:
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
        event = TrajectoryEvent.model_validate_json(raw)
    except ValidationError as exc:
        details = exc.errors(include_input=False, include_context=False, include_url=False)
        _print_json({"code": "invalid_event", "errors": details}, stream=sys.stderr)
        return 2
    except (OSError, UnicodeError) as exc:
        _print_json({"code": "invalid_input", "message": str(exc)}, stream=sys.stderr)
        return 2

    try:
        config = ConfigManager().load()
        with MindMemOSClient(config=config) as client:
            result = client.report_trajectory(
                event,
                include_memory=not args.skill_only,
                include_skill=not args.memory_only,
            )
    except InvalidRequestError as exc:
        _print_json({"code": "invalid_argument", "message": str(exc)}, stream=sys.stderr)
        return 2
    except (ConfigError, OSError):
        _print_json(
            {"code": "config_error", "message": "Cannot load SDK settings. Run mindmemos auth."}, stream=sys.stderr
        )
        return 1

    if args.json:
        _print_json(result.model_dump(mode="json", exclude_none=True))
    else:
        print(f"Event {result.event_id}")
        if result.memory is not None:
            print(f"Memory: {result.memory.status} ({result.memory.code})")
        if result.skill is not None:
            print(f"Skill: {result.skill.status} ({result.skill.trajectory_id or result.skill.code})")
    return int(result.has_failures)


def _handle_doctor(args: argparse.Namespace) -> int:
    # CLI handlers share the argparse namespace signature; doctor uses persisted configuration.
    manager = ConfigManager()
    if not manager.exists():
        print(f"config: missing ({manager.config_path})")
        print("Run `mindmemos auth` first.")
        return 1
    try:
        config = manager.load()
    except ConfigError as exc:
        print(f"config: invalid: {exc}")
        return 1
    print(f"config: ok ({manager.config_path})")
    base_url, api_key = config.resolve_memory_connection()
    print(f"base_url: {base_url}")
    print(f"api_key: {'configured' if api_key else 'missing'}")
    print(f"user_id: {config.defaults.user_id or '(not set)'}")
    if not api_key:
        return 1
    try:
        client = MindMemOSClient(config=config)
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
    if not manager.exists():
        print(f"No SDK config at {manager.config_path}. Run `mindmemos auth` first.")
        return None
    try:
        config = manager.load()
    except ConfigError as exc:
        print(f"mindmemos: {exc}")
        return None
    if not config.memory_service.auth.api_key:
        print("No api_key configured. Run `mindmemos auth` first.")
        return None
    return MindMemOSClient(config=config)


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


def _print_json(payload: Any, *, stream=None) -> None:
    """Print JSON to stdout or the supplied diagnostic stream."""
    print(json.dumps(payload, ensure_ascii=False, default=str), file=stream)


def _prompt(message: str) -> str:
    """Read one line of user input."""
    return input(message).strip()


def _prompt_secret(message: str) -> str:
    """Read a secret without echoing it to the terminal."""
    return getpass(message).strip()


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
    try:
        return args.handler(args)
    except EOFError:
        print(f"\nmindmemos {args.command}: input cancelled.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
