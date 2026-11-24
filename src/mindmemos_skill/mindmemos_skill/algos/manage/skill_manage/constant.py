"""Current algorithm versions and canonical Skill management metadata keys."""

from typing import Final

SCENE_GRAPH_REPRESENTATION_VERSION: Final[str] = "scene_graph_v2"
"""Graph representation version produced and accepted by the current code."""

JOINT_LINKING_VERSION: Final[str] = "joint_linking_v3"
"""Relationship construction version used by the current code."""

# Metadata containers and graph ownership.

METADATA_KEY: Final[str] = "metadata"
"""Serialized metadata mapping on a document, node, edge, or vector hit."""

METADATA_COLLECTION_SKILL_ID: Final[str] = "collection_skill_id"
"""Collection member identity owning a Skill node and its Scenes."""

METADATA_FAMILY_ID: Final[str] = "family_id"
"""Canonical stored Skill family identity."""

METADATA_EVIDENCE: Final[str] = "evidence"
"""Source-grounded explanation attached to a graph relationship."""

METADATA_INTENT: Final[str] = "intent"
"""Source-grounded goal in derived node metadata."""

METADATA_INTENT_SIGNATURE: Final[str] = "intent_signature"
"""Compact retrieval signature from the document or derived metadata."""

METADATA_LOCAL_SCRIPTS: Final[str] = "local_scripts"
"""Portable local script paths in derived node metadata."""

METADATA_EXTERNAL_DEPENDENCIES: Final[str] = "external_dependencies"
"""Documented external packages in derived node metadata."""

METADATA_COMPATIBILITY: Final[str] = "compatibility"
"""Documented runtime requirements in derived node metadata."""

METADATA_ALLOWED_TOOLS: Final[str] = "allowed_tools"
"""Documented tool declarations in derived node metadata."""

# Document metadata.

METADATA_SUMMARY: Final[str] = "summary"
"""Fallback document capability summary."""

METADATA_DOMAIN: Final[str] = "domain"
"""Single domain declaration in document metadata."""

METADATA_DOMAINS: Final[str] = "domains"
"""Multiple domain declarations in document metadata."""

METADATA_TAGS: Final[str] = "tags"
"""Document tags used for domain retrieval."""

METADATA_TOOLING: Final[str] = "tooling"
"""Document tool declarations used for retrieval."""

METADATA_LIBRARIES: Final[str] = "libraries"
"""Document library declarations used for dependency extraction."""

METADATA_EXAMPLES: Final[str] = "examples"
"""Example tasks declared in document metadata."""

METADATA_EXAMPLE_TASKS: Final[str] = "example_tasks"
"""Alternative example-task declaration in document metadata."""

METADATA_VALUE: Final[str] = "value"
"""Preserved non-mapping frontmatter metadata value."""

# Snapshot metadata and captured construction settings.

METADATA_BUILD_SETTINGS: Final[str] = "build_settings"
"""Immutable graph construction settings captured by the snapshot."""

METADATA_CONTENT_HASH: Final[str] = "content_hash"
"""Logical graph content digest used for integrity validation."""

METADATA_SOURCE_HASH: Final[str] = "source_hash"
"""Exact construction input digest used for graph reuse."""

METADATA_BUILD_WARNINGS: Final[str] = "build_warnings"
"""Allowlisted build warnings exposed during retrieval."""

METADATA_CONFIG: Final[str] = "config"
"""Serialized construction configuration inside build settings."""

METADATA_EMBEDDING_PROFILE: Final[str] = "embedding_profile"
"""Captured vector space and text-template identity."""

METADATA_REPRESENTATION: Final[str] = "representation"
"""Graph representation schema version in build settings."""

METADATA_SCENES: Final[str] = "scenes"
"""Captured Scene content digests or Scene count in build coverage."""

METADATA_BUILD_SCENE: Final[str] = "build_scene"
"""Captured setting controlling Scene construction."""

METADATA_EMBEDDING: Final[str] = "embedding"
"""Captured embedding model connection settings."""

METADATA_CHAT: Final[str] = "chat"
"""Captured chat model connection settings."""

METADATA_ENDPOINT_HASH: Final[str] = "endpoint_hash"
"""Digest of the original model endpoint, without transport secrets."""

METADATA_BASE_URL: Final[str] = "base_url"
"""Runtime endpoint used to validate the captured endpoint digest."""

METADATA_LINKING_VERSION: Final[str] = "linking_version"
"""Captured relationship construction algorithm version."""

METADATA_ENRICHMENT: Final[str] = "enrichment"
"""Captured enrichment prompt and schema identity."""

METADATA_EXTRACTION_USAGE: Final[str] = "extraction_usage"
"""Per-member Scene extraction accounting."""

METADATA_ENRICHMENT_USAGE: Final[str] = "enrichment_usage"
"""Per-member Skill enrichment accounting."""

METADATA_BUILD_COVERAGE: Final[str] = "build_coverage"
"""Counts of constructed and reused graph components."""

METADATA_SKILLS: Final[str] = "skills"
"""Number of Skill members in snapshot build coverage."""

METADATA_EDGES: Final[str] = "edges"
"""Number of graph relationships in snapshot build coverage."""

METADATA_VECTORS: Final[str] = "vectors"
"""Number of node vectors in snapshot build coverage."""

METADATA_REUSED_VECTORS: Final[str] = "reused_vectors"
"""Number of node vectors reused during construction."""

METADATA_CANDIDATE_PAIRS: Final[str] = "candidate_pairs"
"""Number of candidate relationship pairs during construction."""
