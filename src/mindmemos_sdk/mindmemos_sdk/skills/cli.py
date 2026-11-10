"""Remote Skill CLI commands backed by the public SDK clients."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..client import MindMemOSClient
from ..config import ConfigManager
from ..errors import ApiError, ConfigError, ConfigValidationError, MindMemOSSDKError
from .cli_files import collect_resources, export_source, read_source
from .client import SkillClient
from .config import EnvironmentConfig
from .typing import (
    ActivateRequest,
    AddGraphMembersRequest,
    CLICommitInput,
    CLIRegisterInput,
    CLISkillInfo,
    CLITrainingRequest,
    CommitRequest,
    DeleteRequest,
    DiffRequest,
    EvaluationQueryRequest,
    EvolveRequest,
    GraphResolveRequest,
    GraphSnapshotRequest,
    GraphSnapshotsRequest,
    HistoryRequest,
    ListRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    RemoveGraphMembersRequest,
    ShowRequest,
    SkillAnalyzeRequest,
    SkillResolveRequest,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingSplitRequest,
    TrajectoryAnalyzeRequest,
    TrajectoryListRequest,
    TrajectoryReportRequest,
    TrajectoryShowRequest,
    UpdateInfoRequest,
)


def _json_object(value: str) -> dict:
    """Parse a command-line JSON object."""
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("Expected valid JSON") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return data


def _positive(value: str) -> int:
    """Parse a strictly positive count."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def _nonnegative(value: str) -> int:
    """Parse a nonnegative offset or sequence."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return number


def add_skill_commands(subparsers: argparse._SubParsersAction) -> None:
    """Add the remote Skill command group to the SDK parser.

    Args:
        subparsers: Root command collection owned by the SDK CLI.
    """
    root = subparsers.add_parser("skill", help="Manage remote Skills, versions, tasks, and graphs.")
    commands = root.add_subparsers(dest="skill_command", required=True)

    def command(name, handler, help_text):
        parser = commands.add_parser(name, help=help_text)
        parser.set_defaults(handler=run_skill_command, skill_handler=handler)
        return parser

    register = command("register", _register, "Register a local Skill or StructuredSkillInfo JSON.")
    register.allow_abbrev = False
    register.description = "Supply exactly one local path or --json-skill; local options cannot be used with JSON."
    source = register.add_mutually_exclusive_group(required=True)
    source.add_argument("source", nargs="?", help="Local Skill directory or SKILL.md; exclusive with --json-skill.")
    source.add_argument("--json-skill", type=_json_object, help="Complete StructuredSkillInfo JSON object.")
    register.add_argument("--alias")
    register.add_argument("--version", help="Local source only; defaults to v0.0.1.")
    register.add_argument("--message", help="Local source only; defaults to Initial registration.")
    register.add_argument("--metadata-json", type=_json_object, help="Local source metadata only.")
    register.add_argument("--include", help="Local resource inclusion patterns only.")
    register.add_argument("--exclude", help="Local resource exclusion patterns only.")
    listing = command("list", _list, "Filter Skill family summaries.")
    listing.add_argument("--filter", type=_json_object, default={})
    listing.add_argument("--name")
    listing.add_argument("--alias")
    listing.add_argument("--user-id")
    listing.add_argument("--agent-id")
    listing.add_argument("--limit", type=_positive, default=20)
    for name, handler in (("show", _show), ("resolve", _resolve), ("export", _export)):
        parser = command(name, handler, f"{name.capitalize()} a Skill reference.")
        parser.add_argument("skill_ref")
        if name == "export":
            parser.add_argument("--to", required=True, help="Parent folder; export creates a <skill-name> subfolder.")
            parser.add_argument(
                "--overwrite", action="store_true", help="Replace only the named Skill subfolder, removing stale files."
            )
    commit = command("commit", _commit, "Publish a child of the selected Skill version.")
    commit.allow_abbrev = False
    commit.description = "Supply exactly one --from or --json-skill; JSON must include a nonempty commit_message."
    commit.add_argument("skill_ref")
    source = commit.add_mutually_exclusive_group(required=True)
    source.add_argument("--from", dest="source", help="Local Skill directory or SKILL.md.")
    source.add_argument("--json-skill", type=_json_object, help="Complete StructuredSkillInfo JSON object.")
    commit.add_argument("--message", help="Required for local sources; use commit_message in JSON mode.")
    labeling = commit.add_mutually_exclusive_group()
    labeling.add_argument("--version", help="Local source version label only.")
    labeling.add_argument("--bump", choices=("patch", "minor", "major"), help="Local source version increment only.")
    resources = commit.add_mutually_exclusive_group()
    resources.add_argument("--resources-dir", help="Local replacement resource directory only.")
    resources.add_argument("--clear-resources", action="store_true", help="Clear resources in local source mode only.")
    commit.add_argument("--activate", action="store_true")
    history = command("history", _history, "Read the owning family's version history.")
    history.add_argument("skill_ref")
    history.add_argument("--limit", type=_positive, default=20)
    history.add_argument("--offset", type=_nonnegative, default=0)
    diff = command("diff", _diff, "Compare two Skill references from A to B.")
    diff.add_argument("skill_ref_a")
    diff.add_argument("skill_ref_b")
    activate = command("activate", _activate, "Activate a family version by label or ID.")
    activate.add_argument("skill_ref")
    selection = activate.add_mutually_exclusive_group(required=True)
    selection.add_argument("--version")
    selection.add_argument("--version-id")
    update = command("update-info", _update_info, "Update family information.")
    update.add_argument("skill_ref")
    update.add_argument("--name")
    alias = update.add_mutually_exclusive_group()
    alias.add_argument("--alias")
    alias.add_argument("--clear-alias", action="store_true")
    update.add_argument("--metadata-json", type=_json_object)
    delete = command("delete", _delete, "Soft-delete the entire owning Skill family.")
    delete.add_argument("skill_ref")
    delete.add_argument("--yes", "-y", action="store_true")
    evolve = command("evolve", _evolve, "Submit evolution of selected Skill versions.")
    evolve.add_argument("skill_refs", nargs="+", help="Skill aliases, family IDs or exact version IDs.")
    evolve.add_argument("--pipeline", help="Configured server pipeline, for example offline.")
    editable = evolve.add_mutually_exclusive_group()
    editable.add_argument("--editable-skill", action="append", help="Editable input Skill reference; repeatable.")
    editable.add_argument("--read-only", action="store_true", help="Keep all input Skills unchanged.")
    evolve.add_argument(
        "--allow-create-skills",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow or forbid new Skills; omission uses the service default.",
    )
    evolve.add_argument("--auto-accept", action="store_true")
    train = command("train", _train, "Submit training of the selected version.")
    train.add_argument("skill_ref")
    train.add_argument("--work-dir", required=True)
    train.add_argument("--pipeline", required=True)
    environment = train.add_mutually_exclusive_group(required=True)
    environment.add_argument("--environment", type=Path, help="YAML or JSON environment configuration file.")
    environment.add_argument("--environment-json", type=_json_object, help="Inline environment configuration object.")
    train.add_argument("--sample-json", type=_json_object, action="append", required=True)
    train.add_argument("--train-id", action="append", required=True)
    train.add_argument("--val-id", action="append", default=[])
    train.add_argument("--test-id", action="append", default=[])
    train.add_argument("--auto-accept", action="store_true")
    _add_task_commands(commands)
    _add_trajectory_commands(commands)
    _add_graph_commands(commands)
    _add_analysis_commands(commands)


def _nested(commands, group, name, handler):
    """Create a nested command with the shared execution boundary."""
    parser = commands.add_parser(name)
    parser.set_defaults(handler=run_skill_command, skill_handler=handler, skill_command=group)
    return parser


def _add_task_commands(commands):
    """Define task lookup and event-following commands."""
    root = commands.add_parser("task", help="Read asynchronous tasks and progress.")
    nested = root.add_subparsers(dest="task_command", required=True)
    show = _nested(nested, "task", "show", _task_show)
    show.add_argument("task_id")
    progress = _nested(nested, "task", "progress", _task_progress)
    progress.add_argument("task_id")
    progress.add_argument("--after", type=_nonnegative, default=0)
    progress.add_argument("--follow", "-f", action="store_true")


def _add_analysis_commands(commands):
    """Expose scoring commands separately from stored evaluation queries."""
    root = commands.add_parser("analyze", help="Score Skills or analyze stored trajectories.")
    nested = root.add_subparsers(dest="analyze_command", required=True)
    skill = _nested(nested, "analyze", "skill", _analyze_skill)
    skill.add_argument("skill_ref")
    trajectory = _nested(nested, "analyze", "trajectory", _analyze_trajectory)
    trajectory.add_argument("trajectory_id")
    trajectory.add_argument("--attribute", action=argparse.BooleanOptionalAction, default=None)
    trajectory.add_argument("--verifier-score", type=float)
    for parser in (skill, trajectory):
        parser.add_argument("--request-id")
        parser.add_argument("--force-refresh", action="store_true")
    root = commands.add_parser("evaluations", help="Read stored evaluations without running models.")
    nested = root.add_subparsers(dest="evaluations_command", required=True)
    query = _nested(nested, "evaluations", "query", _query_evaluations)
    query.add_argument("skill_refs", nargs="+")
    query.add_argument("--judge-type", choices=("static_skill", "trajectory"), action="append")
    query.add_argument("--limit", type=_positive, default=200)


def _add_trajectory_commands(commands):
    """Define trajectory input, filtered listing, and detail lookup."""
    root = commands.add_parser("trajectory", help="Report and read execution trajectories.")
    nested = root.add_subparsers(dest="trajectory_command", required=True)
    report = _nested(nested, "trajectory", "report", _trajectory_report)
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--json-data", type=_json_object)
    source.add_argument("--file")
    listing = _nested(nested, "trajectory", "list", _trajectory_list)
    listing.add_argument("skill_ref")
    listing.add_argument("--status")
    listing.add_argument("--received-from")
    listing.add_argument("--received-to")
    listing.add_argument("--limit", type=_positive, default=20)
    show = _nested(nested, "trajectory", "show", _trajectory_show)
    show.add_argument("trajectory_id")


def _add_graph_commands(commands):
    """Define graph publication, snapshot lookup, and recommendations."""
    root = commands.add_parser("graph", help="Publish and query Skill graphs.")
    nested = root.add_subparsers(dest="graph_command", required=True)
    _nested(nested, "graph", "list", _graph_list)
    for name, handler in (
        ("resolve", _graph_resolve),
        ("history", _graph_history),
        ("show", _graph_show),
        ("preview", _graph_preview),
    ):
        parser = _nested(nested, "graph", name, handler)
        parser.add_argument("graph_ref")
    publish = _nested(nested, "graph", "publish", _graph_publish)
    destination = publish.add_mutually_exclusive_group(required=True)
    destination.add_argument("--alias")
    destination.add_argument("--graph-id")
    members = publish.add_mutually_exclusive_group(required=True)
    members.add_argument("--skill", action="append")
    members.add_argument("--all-skills", action="store_true")
    members.add_argument("--empty", action="store_true")
    publish.add_argument("--message")
    publish.add_argument("--force-extract-scene", action="store_true")
    for name, handler in (("add", _graph_add), ("remove", _graph_remove)):
        members_parser = _nested(nested, "graph", name, handler)
        members_parser.add_argument("graph_ref")
        members_parser.add_argument("--skill", action="append", required=True)
        members_parser.add_argument("--message")
        if name == "add":
            members_parser.add_argument("--force-extract-scene", action="store_true")
    recommend = _nested(nested, "graph", "recommend", _graph_recommend)
    recommend.add_argument("task")
    recommend.add_argument("--by", required=True)
    recommend.add_argument("--top-k", type=_positive, default=5)
    recommend.add_argument("--strategy", choices=("graph", "vector"), default="graph")
    recommend.add_argument("--include-scenes", action="store_true")
    recommend.add_argument("--include-evaluation", action="store_true", help="Attach stored evaluation results.")


def _print_json(data, *, stream=None):
    """Emit one JSON value without incidental status messages."""
    print(json.dumps(data, ensure_ascii=False, default=str), file=stream or sys.stdout, flush=True)


def run_skill_command(args: argparse.Namespace) -> int:
    """Execute one Skill command with JSON results or plain-text progress.

    Args:
        args: Parsed arguments containing a directly bound command handler.

    Returns:
        Zero on success, one on service/local I/O failure, or two on invalid input.
    """
    try:
        config = ConfigManager().load()
        with MindMemOSClient(config=config) as root:
            result = args.skill_handler(root.skills, args)
            if result is not None:
                _print_json(result)
        return 0
    except ApiError as exc:
        error = {"code": exc.code or "api_error", "message": str(exc)}
        if exc.request_id:
            error["request_id"] = exc.request_id
        _print_json(error, stream=sys.stderr)
        return 1
    except ConfigValidationError:
        _print_json(
            {"code": "config_error", "message": "Invalid SDK configuration. Run mindmemos auth to update it."},
            stream=sys.stderr,
        )
        return 1
    except (ConfigError, MindMemOSSDKError, OSError) as exc:
        _print_json({"code": "sdk_error", "message": str(exc)}, stream=sys.stderr)
        return 1
    except (ValueError, ValidationError) as exc:
        message = "Invalid request fields" if isinstance(exc, ValidationError) else str(exc)
        _print_json({"code": "invalid_argument", "message": message}, stream=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        _print_json({"code": "cancelled", "message": "Command cancelled"}, stream=sys.stderr)
        return 1


def _selection(client: SkillClient, reference: str) -> dict:
    """Resolve the owning family and selected immutable version."""
    return client.skill_resolve(SkillResolveRequest(skill_ref=reference))


def _read(client, selection):
    """Read exactly the resolved version without reselecting the active pointer."""
    return client.show(ShowRequest(skill_id=selection["skill_id"], skill_version_id=selection["skill_version_id"]))


def _register(client, args):
    """Register one complete JSON input or a standard local source."""
    source = CLIRegisterInput(
        source=args.source,
        json_skill=args.json_skill,
        version=args.version,
        message=args.message,
        metadata_json=args.metadata_json,
        include=args.include,
        exclude=args.exclude,
    )
    if source.json_skill is not None:
        return client.register(RegisterRequest(skill=source.json_skill, alias=args.alias))
    root, content, frontmatter = read_source(source.source)
    skill = CLISkillInfo(
        content=content,
        resources=collect_resources(root, source.include, source.exclude),
        commit_version=source.version if source.version is not None else "v0.0.1",
        commit_message=source.message if source.message is not None else "Initial registration",
        metadata=source.metadata_json if source.metadata_json is not None else {},
    )
    return client.register(
        RegisterRequest(alias=args.alias if args.alias is not None else frontmatter["name"], skill=skill)
    )


def _list(client, args):
    """Merge explicit family filter arguments without silent overrides."""
    filters = dict(args.filter)
    additions = {"name": args.name, "alias": args.alias, "user_id": args.user_id, "agent_id": args.agent_id}
    for key, value in additions.items():
        if value is not None:
            if key in filters:
                raise ValueError(f"Duplicate filter: {key}")
            filters[key] = value
    if any(key not in additions or not isinstance(value, str) or not value.strip() for key, value in filters.items()):
        raise ValueError("Filters support nonempty name, alias, user_id, and agent_id strings only")
    return client.list(ListRequest(filters=filters, limit=args.limit))


def _resolve(client, args):
    """Resolve a Skill reference without reading the source."""
    return _selection(client, args.skill_ref)


def _show(client, args):
    """Show the full selected Skill version."""
    return _read(client, _selection(client, args.skill_ref))


def _export(client, args):
    """Export the selected version to a local directory."""
    data = _read(client, _selection(client, args.skill_ref))
    return export_source(data["version"], args.to, args.overwrite)


def _commit(client, args):
    """Commit one complete source as a child of the resolved version."""
    source = CLICommitInput(
        source=args.source,
        json_skill=args.json_skill,
        message=args.message,
        version=args.version,
        bump=args.bump,
        resources_dir=args.resources_dir,
        clear_resources=args.clear_resources,
    )
    if source.json_skill is not None:
        skill = source.json_skill
        inherit_resources = False
    else:
        _, content, _ = read_source(source.source)
        resources = collect_resources(Path(source.resources_dir)) if source.resources_dir else {}
        skill = CLISkillInfo(
            content=content,
            resources=resources,
            commit_message=source.message,
            **({"commit_version": source.version} if source.version else {"version_bump": source.bump or "patch"}),
        )
        inherit_resources = not (source.resources_dir or source.clear_resources)
    selected = _selection(client, args.skill_ref)
    return client.commit(
        CommitRequest(
            skill_id=selected["skill_id"],
            skill=skill,
            base_version_id=selected["skill_version_id"],
            active=args.activate,
            inherit_resources=inherit_resources,
        )
    )


def _history(client, args):
    """Read an offset page of the owning family's history."""
    selected = _selection(client, args.skill_ref)
    return client.history(HistoryRequest(skill_id=selected["skill_id"], limit=args.limit, offset=args.offset))


def _diff(client, args):
    """Resolve both sides and compare the exact selected versions."""
    source = _selection(client, args.skill_ref_a)
    target = _selection(client, args.skill_ref_b)
    return client.diff(
        DiffRequest(
            skill_id=source["skill_id"],
            from_skill_version_id=source["skill_version_id"],
            to_skill_id=target["skill_id"],
            to_skill_version_id=target["skill_version_id"],
        )
    )


def _activate(client, args):
    """Select a family-owned target by exact version identity or label."""
    family = _selection(client, args.skill_ref)["skill_id"]
    if args.version_id:
        selected = _selection(client, args.version_id)
        if selected["skill_id"] != family or selected["skill_version_id"] != args.version_id:
            raise ValueError("Target must be an exact version ID belonging to the selected family")
        version_id = selected["skill_version_id"]
    else:
        offset, matches = 0, []
        while True:
            page = client.history(HistoryRequest(skill_id=family, limit=100, offset=offset))
            items = page["items"]
            matches.extend(item["skill_version_id"] for item in items if item["version_label"] == args.version)
            if not page["has_more"]:
                break
            if not items:
                raise ValueError("History pagination returned no progress")
            offset += len(items)
        if len(matches) != 1:
            raise ValueError("Version label must match exactly one version in this family")
        version_id = matches[0]
    return client.activate(ActivateRequest(skill_id=family, skill_version_id=version_id))


def _update_info(client, args):
    """Update only explicitly supplied family fields."""
    fields = {}
    if args.name is not None:
        fields["name"] = args.name
    if args.alias is not None or args.clear_alias:
        fields["alias"] = None if args.clear_alias else args.alias
    if args.metadata_json is not None:
        fields["metadata"] = args.metadata_json
    if not fields:
        raise ValueError("Provide at least one family information change")
    selected = _selection(client, args.skill_ref)
    return client.update_info(UpdateInfoRequest(skill_id=selected["skill_id"], **fields))


def _delete(client, args):
    """Confirm and soft-delete the entire owning family."""
    selected = _selection(client, args.skill_ref)
    if not args.yes:
        print(f"Delete Skill family {selected['skill_id']}? [y/N] ", end="", file=sys.stderr, flush=True)
        if input().strip().lower() not in ("y", "yes"):
            raise OSError("Deletion cancelled")
    return client.delete(DeleteRequest(skill_id=selected["skill_id"]))


def _evolve(client, args):
    """Pin input versions and validate editable families before submitting evolution."""
    selected = [_selection(client, reference) for reference in args.skill_refs]
    families = {item["skill_id"] for item in selected}
    if len(families) != len(selected):
        raise ValueError("Evolution accepts at most one version per Skill family")
    versions = [item["skill_version_id"] for item in selected]
    fields = {"skill_version_ids": versions}
    if args.pipeline is not None:
        fields["pipeline"] = args.pipeline
    if args.read_only:
        fields["editable_skills"] = []
    elif args.editable_skill is not None:
        editable = [_selection(client, reference)["skill_id"] for reference in args.editable_skill]
        if not set(editable) <= families:
            raise ValueError("Editable Skills must belong to the input collection")
        fields["editable_skills"] = editable
    if args.allow_create_skills is not None:
        fields["allow_create_skills"] = args.allow_create_skills
    return client.evolve(EvolveRequest(**fields, auto_accept=args.auto_accept))


def _analyze_skill(client, args):
    """Resolve and score exactly one selected version."""
    selected = _selection(client, args.skill_ref)
    return client.analyze_skill(
        SkillAnalyzeRequest(
            skill_version_id=selected["skill_version_id"],
            request_id=args.request_id,
            force_refresh=args.force_refresh,
        )
    )


def _analyze_trajectory(client, args):
    """Analyze a stored execution with explicit optional overrides."""
    return client.analyze_trajectory(
        TrajectoryAnalyzeRequest(
            trajectory_id=args.trajectory_id,
            request_id=args.request_id,
            attribute=args.attribute,
            verifier_score=args.verifier_score,
            force_refresh=args.force_refresh,
        )
    )


def _query_evaluations(client, args):
    """Read existing evaluations for the selected exact versions."""
    versions = [_selection(client, reference)["skill_version_id"] for reference in args.skill_refs]
    return client.query_evaluations(
        EvaluationQueryRequest(skill_version_ids=versions, judge_types=args.judge_type, limit=args.limit)
    )


def _train(client, args):
    """Validate task assignment and submit the selected training input."""
    environment = args.environment_json
    if environment is None:
        environment = yaml.safe_load(args.environment.read_text(encoding="utf-8"))
    environment = EnvironmentConfig.model_validate(environment)
    selected = _selection(client, args.skill_ref)
    return client.train(
        CLITrainingRequest(
            init_skill_version_id=selected["skill_version_id"],
            work_dir=args.work_dir,
            algo=args.pipeline,
            env=environment,
            tasks=args.sample_json,
            split=TrainingSplitRequest(train=args.train_id, val=args.val_id, test=args.test_id),
            auto_accept=args.auto_accept,
        )
    )


def _task_show(client, args):
    """Read one business task snapshot."""
    return client.show_task(TaskShowRequest(task_id=args.task_id))


def _print_task_log(event):
    """Print the timestamp, logger name, message and optional traceback."""
    payload = event["payload"]
    timestamp = event.get("occurred_at", "-")
    if timestamp != "-":
        timestamp = (
            datetime.fromisoformat(timestamp).astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )
    fields = {key: event.get(key) for key in ("project_id", "task_id", "attempt")}
    fields.update(
        {
            key: payload.get(key)
            for key in ("run_id", "tree_id", "parent_id", "node_id", "group_id", "request_id", "trace_id", "span_id")
        }
    )
    if payload.get("run_id") is not None and payload.get("message", "").split(" | ", 1)[0] != "algorithm.run.started":
        fields.pop("run_id", None)
        fields.pop("request_id", None)
    identity = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    prefix = f"[{timestamp}] [{payload.get('level', 'INFO')}] [{payload.get('name', '-')}]"
    print(f"{prefix}{f' | {identity}' if identity else ''} | {payload.get('message', '')}", flush=True)
    if payload.get("traceback"):
        print(payload["traceback"].rstrip("\n"), flush=True)


def _task_progress(client, args):
    """Print task logs as text, or follow new logs until the task terminates."""
    after = args.after
    while True:
        data = client.task_progress(TaskProgressRequest(task_id=args.task_id, after=after))
        events = data
        for event in events:
            _print_task_log(event)
            after = max(after, event["sequence"])
        if not args.follow:
            return None
        snapshot = client.show_task(TaskShowRequest(task_id=args.task_id))
        if snapshot["status"] in ("succeeded", "failed", "cancelled"):
            # Fetch once more after observing termination to include the final boundary.
            while True:
                final = client.task_progress(TaskProgressRequest(task_id=args.task_id, after=after))
                if not final:
                    break
                for event in final:
                    _print_task_log(event)
                    after = max(after, event["sequence"])
            return None
        time.sleep(1)


def _trajectory_report(client, args):
    """Read trajectory JSON from an argument, file, or standard input."""
    if args.json_data is not None:
        data = args.json_data
    else:
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_bytes().decode("utf-8")
        try:
            data = _json_object(raw)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc
    return client.report_trajectory(TrajectoryReportRequest.model_validate(data))


def _trajectory_list(client, args):
    """Filter the first trajectory page by the resolved version."""
    selected = _selection(client, args.skill_ref)
    filters = {
        key: value
        for key, value in {
            "status": args.status,
            "received_at_from": args.received_from,
            "received_at_to": args.received_to,
        }.items()
        if value is not None
    }
    return client.list_trajectories(
        TrajectoryListRequest(
            skill_id=selected["skill_id"], skill_version_id=selected["skill_version_id"], limit=args.limit, **filters
        )
    )


def _trajectory_show(client, args):
    """Read a complete trajectory attempt."""
    return client.show_trajectory(TrajectoryShowRequest(trajectory_id=args.trajectory_id))


def _graph_selection(client, reference):
    """Resolve a graph family and selected published snapshot."""
    return client.graph_resolve(GraphResolveRequest(graph_ref=reference))


def _graph_list(client, args):
    """List graph families in the authorized project."""
    return client.list_graphs()


def _graph_resolve(client, args):
    """Resolve a graph reference without reading its snapshot."""
    return _graph_selection(client, args.graph_ref)


def _graph_history(client, args):
    """List snapshots of the graph owning the reference."""
    selected = _graph_selection(client, args.graph_ref)
    return client.graph_snapshots(GraphSnapshotsRequest(graph_id=selected["graph_id"]))


def _graph_show(client, args):
    """Read the precisely resolved graph snapshot."""
    selected = _graph_selection(client, args.graph_ref)
    if not selected["graph_version_id"]:
        raise ValueError("Graph has no published snapshot")
    return client.graph_snapshot(
        GraphSnapshotRequest(graph_id=selected["graph_id"], graph_version_id=selected["graph_version_id"])
    )


def _graph_preview(client, args):
    """Read a published graph through the shared preview endpoint."""
    return client.preview_graph(GraphResolveRequest(graph_ref=args.graph_ref))


def _graph_add(client, args):
    """Add explicitly selected families to a graph."""
    return client.add_graph_members(
        AddGraphMembersRequest(
            graph_ref=args.graph_ref,
            skill_refs=args.skill,
            commit_message=args.message,
            force_extract_scene=args.force_extract_scene,
        )
    )


def _graph_remove(client, args):
    """Remove explicitly selected families from a graph."""
    return client.remove_graph_members(
        RemoveGraphMembersRequest(graph_ref=args.graph_ref, skill_refs=args.skill, commit_message=args.message)
    )


def _graph_publish(client, args):
    """Resolve members to exact versions and publish a graph snapshot."""
    references = args.skill or []
    if args.all_skills:
        # The no-limit request lists all families without accepting a CLI cursor.
        references = [item["skill_id"] for item in client.list(ListRequest())["items"]]
    versions, families = [], set()
    for reference in references:
        selected = _selection(client, reference)
        if selected["skill_id"] in families:
            raise ValueError("A graph can contain only one version of each Skill family")
        families.add(selected["skill_id"])
        versions.append(selected["skill_version_id"])
    fields = {}
    if args.graph_id:
        graph = _graph_selection(client, args.graph_id)
        if graph["graph_id"] != args.graph_id:
            raise ValueError("--graph-id requires an exact graph family ID")
        fields["graph_id"] = graph["graph_id"]
    else:
        fields["alias"] = args.alias
    if args.message is not None:
        fields["commit_message"] = args.message
    return client.organize(OrganizeRequest(skill_refs=versions, force_extract_scene=args.force_extract_scene, **fields))


def _graph_recommend(client, args):
    """Recommend using the snapshot selected by the graph reference."""
    selected = _graph_selection(client, args.by)
    if not selected["graph_version_id"]:
        raise ValueError("Graph has no published snapshot")
    return client.recommend(
        RecommendRequest(
            graph_ref=selected["graph_id"],
            graph_version_id=selected["graph_version_id"],
            query=args.task,
            top_k=args.top_k,
            strategy=args.strategy,
            include_scenes=args.include_scenes,
            include_evaluation=args.include_evaluation,
        )
    )
