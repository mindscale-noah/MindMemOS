"""Preview a Skill snapshot or continuously reconcile legacy data in a private worker."""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from mindmemos.config import QdrantConfig
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos_skill.infra.database import Database, PostgresConfig, PostgresDatabase, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence.constants import LEGACY_SKILL_SYNC_PROJECTS
from mindmemos_skill.persistence.factory import create_persistence
from mindmemos_skill.persistence.store import SkillPersistence
from mindmemos_skill.persistence.typing import LegacySkillSyncProject
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .migrate import MigrationError, discover_projects, migrate_project
from .sync import sync_project
from .typing import MigrationFailure, MigrationRun, ProjectMigration

TARGET_DATABASE_ENV = "MINDMEMOS_SKILL_MIGRATION_DATABASE_URL"
PROJECT_REGISTRY_ENV = "MINDMEMOS_SKILL_MIGRATION_PLATFORM_DATABASE_URL"


def _target_database(connection_url: str) -> Database:
    """Open only a configured, existing Skill database."""
    url = make_url(connection_url)
    if url.drivername == "postgresql+psycopg":
        return PostgresDatabase(PostgresConfig(url=connection_url))
    if url.drivername in {"sqlite", "sqlite+aiosqlite"}:
        if not url.database or not Path(url.database).is_absolute() or not Path(url.database).is_file():
            raise MigrationError("invalid_target", "SQLite migration requires an existing absolute database file")
        return SQLiteDatabase(SQLiteConfig(url=connection_url))
    raise MigrationError("invalid_target", "Use postgresql+psycopg or sqlite+aiosqlite for the Skill database")


def _source_config(args: argparse.Namespace) -> QdrantConfig:
    """Use the configured read-only legacy collections for both commands."""
    config = QdrantConfig(url=args.qdrant_url, auto_create=False, batch_upsert_enabled=False)
    if args.version_collection:
        config.skill_version_collection = args.version_collection
    if args.blob_collection:
        config.skill_blob_collection = args.blob_collection
    if args.add_record_collection:
        config.add_record_collection = args.add_record_collection
    if args.trace_summary_collection:
        config.skill_trace_summary_collection = args.trace_summary_collection
    return config


async def _run(args: argparse.Namespace, connection_url: str, run: MigrationRun) -> None:
    """Process all discovered projects and collect partial outcomes."""
    config = _source_config(args)

    async with AsyncExitStack() as cleanup:
        database = _target_database(connection_url)
        cleanup.push_async_callback(database.close)
        source = QdrantEngine(config)
        cleanup.push_async_callback(source.close)
        target = create_persistence(database)
        try:
            projects = await discover_projects(source, config, args.page_size, run.unscoped_skills)
        except Exception as exc:
            raise MigrationError("discovery_failed", f"Project discovery interrupted ({type(exc).__name__})") from exc
        run.projects_discovered = len(projects)
        for project in projects:
            try:
                result = await migrate_project(
                    source, config, target, project, apply=args.apply, page_size=args.page_size
                )
            except Exception as exc:
                result = ProjectMigration(
                    project_id=project,
                    scan_failures=[
                        MigrationFailure(
                            project_id=project,
                            reason="project_processing_failed",
                            detail=f"Project processing interrupted ({type(exc).__name__})",
                        )
                    ],
                )
            run.projects.append(result)


async def _watch(args: argparse.Namespace, connection_url: str, registry_url: str) -> None:
    """Run the scheduled worker against trusted Platform project namespaces."""
    config = _source_config(args)
    registry_dsn = make_url(registry_url)
    if registry_dsn.get_backend_name() == "sqlite":
        registry_dsn = registry_dsn.set(drivername="sqlite+aiosqlite")
    registry = create_async_engine(registry_dsn)
    async with AsyncExitStack() as cleanup:
        cleanup.push_async_callback(registry.dispose)
        database = _target_database(connection_url)
        cleanup.push_async_callback(database.close)
        source = QdrantEngine(config)
        cleanup.push_async_callback(source.close)
        target = create_persistence(database)
        while True:
            try:
                projects = await _watch_projects(registry, source, config, target, args.page_size)
            except Exception as error:
                print(f"Legacy Skill discovery failed ({type(error).__name__})", file=sys.stderr, flush=True)
            else:
                for project in projects:
                    try:
                        result = await sync_project(source, config, target, project, page_size=args.page_size)
                    except Exception as error:
                        print(
                            f"Legacy Skill sync failed for {project} ({type(error).__name__})",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(result.model_dump_json(), flush=True)
            await asyncio.sleep(args.watch)


async def _watch_projects(
    registry: AsyncEngine, source: QdrantEngine, config: QdrantConfig, target: SkillPersistence, page_size: int
) -> list[str]:
    """Check Platform ownership and seed readiness for all discovered projects before backfill."""
    async with registry.connect() as connection:
        allowed = set(
            (await connection.execute(text("SELECT namespace FROM memory_projects WHERE active = true"))).scalars()
        )
    discovered = set(await discover_projects(source, config, page_size, []))
    with target.maintenance() as session:
        checkpoints = await session.database.select(LEGACY_SKILL_SYNC_PROJECTS, fields=["project_id"], limit=None)
        known = {row["project_id"] for row in checkpoints}
        if new_projects := (discovered & allowed) - known:
            await session.database.upsert(
                LEGACY_SKILL_SYNC_PROJECTS,
                [LegacySkillSyncProject(project_id=project).model_dump() for project in sorted(new_projects)],
            )
    return sorted(allowed & (discovered | known))


def _item_summary(total: int, verified: int, new: int, failed: int, apply: bool) -> dict[str, int]:
    """Separate committed writes, validated previews, and reused records."""
    return {
        "total": total,
        "migrated": new if apply else 0,
        "would_migrate": 0 if apply else new,
        "already_migrated": verified - new,
        "failed": failed,
        "unprocessed": total - verified - failed,
    }


def _report(run: MigrationRun, apply: bool) -> dict:
    """Build the final summary and group source IDs by failure type."""
    skills = [*run.unscoped_skills, *(item for project in run.projects for item in project.failed_skills)]
    trajectories = [item for project in run.projects for item in project.failed_trajectories]
    scans = [item for project in run.projects for item in project.scan_failures]
    counts = [project.counts for project in run.projects]
    failure_groups = {
        "failed_skills": skills,
        "failed_trajectories": trajectories,
        "scan_failures": scans,
        "run_failures": run.run_failures,
    }
    reason_counts = {
        key: dict(sorted(Counter(item.reason for item in failures).items())) for key, failures in failure_groups.items()
    }
    summary = {
        "mode": "apply" if apply else "dry-run",
        "status": "incomplete"
        if scans or run.run_failures
        else "completed_with_failures"
        if skills or trajectories
        else "completed",
        "projects": {
            "total": run.projects_discovered,
            "completed": sum(
                not (project.failed_skills or project.failed_trajectories or project.scan_failures)
                for project in run.projects
            ),
            "with_failures": sum(
                not project.scan_failures and bool(project.failed_skills or project.failed_trajectories)
                for project in run.projects
            ),
            "incomplete": sum(bool(project.scan_failures) for project in run.projects),
            "unprocessed": run.projects_discovered - len(run.projects),
        },
        "skills": _item_summary(
            sum(item.families for item in counts) + len(run.unscoped_skills),
            sum(item.verified_families for item in counts),
            sum(item.new_families for item in counts),
            len(skills),
            apply,
        ),
        "trajectories": _item_summary(
            sum(item.trajectories for item in counts),
            sum(item.verified_trajectories for item in counts),
            sum(item.new_trajectories for item in counts),
            len(trajectories),
            apply,
        )
        | {
            "skipped": sum(item.skipped_records for item in counts),
            "scanned": sum(item.trajectories + item.skipped_records for item in counts),
        },
        "failure_reasons": reason_counts,
    }
    return {
        "summary": summary,
        "projects": [
            {
                "project_id": project.project_id,
                **project.counts.model_dump(),
                "failed_skills": len(project.failed_skills),
                "failed_trajectories": len(project.failed_trajectories),
                "scan_complete": not project.scan_failures,
            }
            for project in run.projects
        ],
        **{
            key: [item.model_dump(exclude_defaults=True) for item in failures]
            for key, failures in failure_groups.items()
        },
    }


def main() -> int:
    """Run the server-wide migration and persist a final report."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Set the target connection in {TARGET_DATABASE_ENV}.")
    parser.add_argument("--qdrant-url", required=True, help="Private URL of the legacy Memory Qdrant")
    parser.add_argument("--version-collection", help="Override the Memory Skill version collection")
    parser.add_argument("--blob-collection", help="Override the Memory Skill blob collection")
    parser.add_argument("--add-record-collection", help="Override the Memory add-record collection")
    parser.add_argument("--trace-summary-collection", help="Override the Memory Skill trace summary collection")
    parser.add_argument("--page-size", type=int, default=100, help="Maximum points per source page")
    parser.add_argument("--apply", action="store_true", help="Write records; default is a read-only preview")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Continuously reconcile trusted projects")
    parser.add_argument(
        "--report", type=Path, help="Path for a new JSON report; default includes the UTC run timestamp"
    )
    args = parser.parse_args()
    if not 1 <= args.page_size <= 1000:
        parser.error("Use a page size from 1 to 1000")
    if args.watch is not None:
        if args.watch < 1 or not args.apply or args.report is not None:
            parser.error("--watch requires a positive interval and --apply, without --report")
        connection_url = os.getenv(TARGET_DATABASE_ENV)
        registry_url = os.getenv(PROJECT_REGISTRY_ENV)
        if not connection_url or not registry_url:
            parser.error(f"--watch requires {TARGET_DATABASE_ENV} and {PROJECT_REGISTRY_ENV}")
        try:
            asyncio.run(_watch(args, connection_url, registry_url))
        except KeyboardInterrupt:
            return 0
        except Exception as error:
            print(f"Legacy Skill worker could not start ({type(error).__name__})", file=sys.stderr)
            return 1
        return 0

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (args.report or Path(f"skill-migration-report-{timestamp}.json")).expanduser().absolute()
    run = MigrationRun()
    temporary_path = None
    try:
        if report_path.exists() or report_path.is_symlink():
            raise FileExistsError("Report path already exists")
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=report_path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            try:
                connection_url = os.getenv(TARGET_DATABASE_ENV)
                if not connection_url:
                    raise MigrationError("missing_target", f"{TARGET_DATABASE_ENV} is required")
                asyncio.run(_run(args, connection_url, run))
            except MigrationError as exc:
                run.run_failures.append(MigrationFailure(reason=exc.reason, detail=str(exc)))
            except KeyboardInterrupt:
                run.run_failures.append(MigrationFailure(reason="interrupted", detail="Migration interrupted"))
            except Exception as exc:
                run.run_failures.append(
                    MigrationFailure(reason="run_failed", detail=f"Migration could not finish ({type(exc).__name__})")
                )
            report = _report(run, args.apply)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.link(temporary_path, report_path)
    except OSError as exc:
        print(f"Cannot save migration report at {report_path} ({type(exc).__name__})", file=sys.stderr)
        return 1
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(json.dumps({**report["summary"], "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    return int(report["summary"]["status"] == "incomplete")


if __name__ == "__main__":
    raise SystemExit(main())
