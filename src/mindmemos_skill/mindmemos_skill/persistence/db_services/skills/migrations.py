"""Transactional, repeatable upgrades of existing version display labels."""

from collections import defaultdict

from sqlalchemy import Index, false, select, update

from ....infra.database import Database
from ...constants import SKILL_VERSION_LABEL_INDEX, SKILL_VERSIONS_TABLE
from .versions import parse_label


def migrate_version_labels(database: Database) -> None:
    """Backfill labels and enforce per-Skill uniqueness without changing identities.

    Args:
        database: Initialized repository database. Updates and index creation share
            one transaction, so interruption rolls back the entire upgrade.
    """
    table = database.tables.get(SKILL_VERSIONS_TABLE).table
    with database.engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            quoted = connection.dialect.identifier_preparer.quote(table.name)
            connection.exec_driver_sql(f"LOCK TABLE {quoted} IN SHARE ROW EXCLUSIVE MODE")
        else:
            # Acquire SQLite's writer lock before reading a migration snapshot.
            connection.execute(update(table).where(false()).values(version_label=table.c.version_label))
        rows = connection.execute(select(table).order_by(table.c.created_at, table.c.version_id)).mappings().all()
        groups = defaultdict(list)
        for row in rows:
            groups[(row["project_id"], row["skill_id"])].append(row)
        for versions in groups.values():
            reserved = {row["version_label"] for row in versions if parse_label(row["version_label"]) is not None}
            seen = set()
            patch = 0
            for row in versions:
                old = row["version_label"]
                if parse_label(old) is not None and old not in seen:
                    seen.add(old)
                    continue
                while f"v1.0.{patch}" in reserved:
                    patch += 1
                label = f"v1.0.{patch}"
                reserved.add(label)
                seen.add(label)
                metadata = dict(row["local_metadata"] or {})
                audit_key = "version_label_migration_v1"
                if audit_key in metadata:
                    raise ValueError("Conflicting historical version-label migration metadata")
                metadata[audit_key] = {"previous_label": old, "assigned_label": label}
                connection.execute(
                    update(table)
                    .where(table.c.version_id == row["version_id"])
                    .values(version_label=label, local_metadata=metadata)
                )
        index = next((item for item in table.indexes if item.name == SKILL_VERSION_LABEL_INDEX), None)
        if index is None:
            index = Index(
                SKILL_VERSION_LABEL_INDEX, table.c.project_id, table.c.skill_id, table.c.version_label, unique=True
            )
        index.create(connection, checkfirst=True)
