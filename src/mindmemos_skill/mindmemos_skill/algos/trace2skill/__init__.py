"""Algorithms that optimize Skills from offline or actively collected traces."""

from .base import Trace2SkillAlgorithm
from .collection import (
    CollectionRetryConfig,
    ScheduledTrajectoryCollector,
    TaskCollectionConfig,
    TrajectoryCollectionResult,
    TrajectoryCollector,
)
from .contracts import AnnotationMode, EvidenceSelection, TraceEvidence
from .trajectory_evidence_patch import (
    TrajectoryEvidencePatch,
    TrajectoryEvidencePatchConfig,
    TrajectoryEvidencePatchOutput,
    TrajectoryEvidencePatchReport,
    TrajectorySummary,
)
from .treeskill import (
    TreeMetadataError,
    TreeSkill,
    TreeSkillConfig,
    TreeSkillModelRequestError,
    TreeSkillOutput,
    TreeSkillReport,
    TreeSkillRouter,
    compile_tree_metadata,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
)

__all__ = [
    "AnnotationMode",
    "CollectionRetryConfig",
    "EvidenceSelection",
    "ScheduledTrajectoryCollector",
    "TaskCollectionConfig",
    "Trace2SkillAlgorithm",
    "TraceEvidence",
    "TreeMetadataError",
    "TreeSkill",
    "TreeSkillConfig",
    "TreeSkillModelRequestError",
    "TreeSkillOutput",
    "TreeSkillReport",
    "TreeSkillRouter",
    "TrajectoryCollectionResult",
    "TrajectoryCollector",
    "TrajectoryEvidencePatch",
    "TrajectoryEvidencePatchConfig",
    "TrajectoryEvidencePatchOutput",
    "TrajectoryEvidencePatchReport",
    "TrajectorySummary",
    "compile_tree_metadata",
    "parse_skill_markdown",
    "parse_tree_with_metadata",
    "render_selected_subtrees",
]
