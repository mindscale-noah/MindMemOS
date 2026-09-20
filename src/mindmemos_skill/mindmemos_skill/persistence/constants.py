"""Canonical physical names for the Skill repository business tables."""

from typing import Final

SKILLS_TABLE: Final[str] = "skills_v1"
"""Central identities, display metadata, and the selected version pointer."""

SKILL_VERSIONS_TABLE: Final[str] = "skill_versions_v1"
"""Immutable source snapshots, configuration, and parent-version lineage."""

TRAJECTORIES_TABLE: Final[str] = "trajectories_v1"
"""Inputs, interactions, and results of individual rollout attempts."""

REMOTE_SKILL_BINDINGS_TABLE: Final[str] = "remote_skill_bindings_v1"
"""Associations between local Skill families and remote lineages."""

REMOTE_VERSION_BINDINGS_TABLE: Final[str] = "remote_version_bindings_v1"
"""Remote version metadata and its corresponding local snapshot."""

REMOTE_TRAJECTORY_BINDINGS_TABLE: Final[str] = "remote_trajectory_bindings_v1"
"""Associations between local attempts and remote memory records."""

REMOTE_OPERATIONS_TABLE: Final[str] = "remote_operations_v1"
"""Durable remote requests, deduplication keys, and execution state."""

SKILL_GRAPHS_TABLE: Final[str] = "skill_graphs_v1"
"""Project-scoped graph identities and current publication pointers."""

SKILL_GRAPH_VERSIONS_TABLE: Final[str] = "skill_graph_versions_v1"
"""Graph content snapshots and validation state."""

SKILL_GRAPH_NODES_TABLE: Final[str] = "skill_graph_nodes_v1"
"""Snapshot nodes, referenced Skill versions, and embeddings."""

SKILL_GRAPH_EDGES_TABLE: Final[str] = "skill_graph_edges_v1"
"""Directed relationships between nodes in a graph snapshot."""

SKILL_GRAPH_REVISIONS_TABLE: Final[str] = "skill_graph_revisions_v1"
"""Successful graph publications and restore provenance."""

SKILL_VERSION_LABEL_INDEX: Final[str] = "skill_versions_label_uq"
"""Unique version display label within one project and Skill."""
