"""Initial public Skill persistence schema."""

from ...infra.database import SchemaMigration

SKILL_SCHEMA_NAMESPACE = "mindmemos-skill"
CURRENT_SCHEMA_VERSION = 1

INITIAL_SCHEMA = SchemaMigration(
    namespace=SKILL_SCHEMA_NAMESPACE,
    version=CURRENT_SCHEMA_VERSION,
    name="initial_schema",
    tables=(
        "skill_versions",
        "skill_sync_state",
        "trajectories",
        "algorithm_logs",
        "llm_calls",
        "skill_remote_operations",
    ),
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "INITIAL_SCHEMA",
    "SKILL_SCHEMA_NAMESPACE",
]
