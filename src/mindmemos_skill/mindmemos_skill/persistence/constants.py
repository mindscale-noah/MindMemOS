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

TRAJECTORIES_TABLE: Final[str] = "trajectories_v1"
"""Inputs, interactions, and results of individual rollout attempts."""

TRAJECTORY_CONSUMPTIONS_TABLE: Final[str] = "trajectory_consumptions_v1"
"""First successful evolution use of each stored trajectory."""

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

SKILL_GRAPH_REVISIONS_TABLE: Final[str] = "skill_graph_revisions_v1"
"""Legacy publications retained only while older deployments may use them."""

SKILL_TASKS_TABLE: Final[str] = "skill_tasks_v1"
"""Durable Skill execution requests and their latest state."""

SKILL_VERSION_LABEL_INDEX: Final[str] = "skill_versions_label_uq"
"""Unique version display label within one project and Skill."""

SKILL_LIST_CURSOR_INDEX: Final[str] = "skills_project_created_cursor_idx"
"""Project and immutable creation order for scrolling through Skill families."""

SKILL_TASK_LOGS_TABLE: Final[str] = "skill_task_logs_v1"
"""Append-only correlated Python logs for every physical task attempt."""
