"""Explicit upgrades for existing local Skill repository tables."""

from sqlalchemy import inspect

from ..infra.database import Database


def migrate_table_names_to_v1(database: Database) -> None:
    """Rename unversioned registered tables while preserving their data.

    Run with repository writers stopped, after any older column migrations and
    before creating missing tables. Repeated calls and partially renamed schemas
    are supported; this does not upgrade columns or cloud collections.

    Args:
        database: Database with the Skill persistence tables already registered.

    Raises:
        RuntimeError: Both the old and new names exist for the same table.
    """
    with database.engine.begin() as connection:
        existing = set(inspect(connection).get_table_names())
        renames = [
            (spec.name.removesuffix("_v1"), spec.name) for spec in database.tables.values() if spec.name.endswith("_v1")
        ]
        for old, new in renames:
            if old in existing and new in existing:
                raise RuntimeError(f"Cannot rename {old}: both {old} and {new} exist")
        quote = connection.dialect.identifier_preparer.quote
        for old, new in renames:
            if old in existing:
                connection.exec_driver_sql(f"ALTER TABLE {quote(old)} RENAME TO {quote(new)}")
