"""v0002 migration: add the ``skill_manage`` graph tables.

The tables are created from the latest catalog for fresh databases and are
materialized through the migration engine's catalog reconciliation for existing
v0001 databases, so this migration intentionally carries no hand-written SQL
statements.
"""

from ...infra.database import SchemaMigration
from ..skill_manage_tables import SKILL_MANAGE_TABLE_NAMES

SKILL_MANAGE_SCHEMA = SchemaMigration(
    namespace="mindmemos-skill",
    version=2,
    name="skill_manage_graph_tables",
    tables=SKILL_MANAGE_TABLE_NAMES,
)

__all__ = ["SKILL_MANAGE_SCHEMA"]
