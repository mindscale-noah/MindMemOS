"""Storage adapter composition for the ``skill_manage`` capability."""

from .pg_storage import PgSkillGraphStorageAdapter
from .storage import (
    SKILL_MANAGE_GENERATION_NAMESPACE,
    SKILL_MANAGE_RECORD_NAMESPACE,
    SkillGraphStorageAdapter,
    generation_counts_hash,
    generation_id_from_operation,
    head_scope,
    physical_record_id,
    snapshot_scope,
)
from .vector_tables import (
    EMBEDDING_VECTOR_NAME,
    SKILL_MANAGE_EMBEDDINGS,
    build_skill_manage_vector_tables,
)

__all__ = [
    "PgSkillGraphStorageAdapter",
    "SKILL_MANAGE_GENERATION_NAMESPACE",
    "SKILL_MANAGE_RECORD_NAMESPACE",
    "SKILL_MANAGE_EMBEDDINGS",
    "EMBEDDING_VECTOR_NAME",
    "SkillGraphStorageAdapter",
    "build_skill_manage_vector_tables",
    "generation_counts_hash",
    "generation_id_from_operation",
    "head_scope",
    "physical_record_id",
    "snapshot_scope",
]
