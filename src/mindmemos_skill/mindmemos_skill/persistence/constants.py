"""Canonical physical names for the Skill repository business tables."""

from typing import Final

REQUIRED_SCOPE_FIELDS: Final[frozenset[str]] = frozenset({"project_id"})
"""Isolation columns that every Skill persistence session must bind."""

OPTIONAL_SCOPE_FIELDS: Final[frozenset[str]] = frozenset({"agent_id", "user_id"})
"""Additional isolation columns a caller may bind for supported tables."""

SKILLS_TABLE: Final[str] = "skills_v1"
"""Central identities, display metadata, and the selected version pointer."""

SKILL_VERSIONS_TABLE: Final[str] = "skill_versions_v1"
"""Immutable source snapshots, configuration, and parent-version lineage."""

LEGACY_SKILL_SOURCE: Final[str] = "legacy-memory"
"""Provenance of records replicated from the original Memory service."""

LEGACY_SKILL_ID_MAP: Final[str] = "legacy_skill_id_map"
"""Project-scoped mappings from imported legacy Skill and version IDs."""

TRAJECTORIES_TABLE: Final[str] = "trajectories_v1"
"""Inputs, interactions, and results of individual rollout attempts."""

TRAJECTORY_VERSIONS_TABLE: Final[str] = "trajectory_versions_v1"
"""Indexed version membership and query fields for one physical trajectory."""

TRAJECTORY_CONSUMPTIONS_TABLE: Final[str] = "trajectory_consumptions_v1"
"""First successful evolution use of each stored trajectory and version pair."""

TRAJECTORY_REPORT_CURSOR_INDEX: Final[str] = "trajectories_report_cursor_idx"
"""Project and Skill receipt order for scrolling through reported executions."""

TRAJECTORY_REPORT_ACTOR_INDEX: Final[str] = "trajectories_report_actor_idx"
"""Project, Skill, and actor filter order for reported executions."""

SKILL_GRAPHS_TABLE: Final[str] = "skill_graphs_v1"
"""Project-scoped graph identities and selected snapshot pointers."""

SKILL_GRAPH_VERSIONS_TABLE: Final[str] = "skill_graph_versions_v1"
"""Graph content snapshots and validation state."""

SKILL_GRAPH_NODES_TABLE: Final[str] = "skill_graph_nodes_v1"
"""Snapshot nodes, referenced Skill versions, and embeddings."""

SKILL_GRAPH_EDGES_TABLE: Final[str] = "skill_graph_edges_v1"
"""Directed relationships between nodes in a graph snapshot."""

SKILL_TASKS_TABLE: Final[str] = "skill_tasks_v1"
"""Durable Skill execution requests and their latest state."""

SKILL_PROJECT_TASKS_TABLE: Final[str] = "skill_project_tasks_v1"
"""One lock row per project for serializing task admissions."""

SKILL_PROJECT_TASK_LEASES_TABLE: Final[str] = "skill_project_task_leases_v1"
"""Renewable reservations for individual request-owned or local tasks."""

DEFAULT_MAX_ACTIVE_TASKS_PER_PROJECT: Final[int] = 1
"""Default shared project limit for queued and running algorithm tasks."""

SKILL_EVALUATIONS_TABLE: Final[str] = "skill_evaluations_v1"
"""Independent judge results linked to Skill versions or trajectories by business identity."""

SKILL_VERSION_LABEL_INDEX: Final[str] = "skill_versions_label_uq"
"""Unique version display label within one project and Skill."""

SKILL_LIST_CURSOR_INDEX: Final[str] = "skills_project_created_cursor_idx"
"""Project and immutable creation order for scrolling through Skill families."""

SKILL_TASK_LOGS_TABLE: Final[str] = "skill_task_logs_v1"
"""Append-only correlated Python logs for every physical task attempt."""
