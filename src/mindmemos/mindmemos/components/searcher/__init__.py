"""Search components: recall, rerank, and fusion operators."""

from .entity_recall import EntityRecall, build_entity_type_filter, combine_entity_results_rrf
from .final_filter import SearchFinalFilter
from .memory_consolidation import ConsolidationResult, MemoryConsolidator
from .memory_retention import MemoryRetentionSelector, RetentionSelection
from .protocols import EntityHydrator, EntityRecallStrategy, SearchStrategy
from .rerank import rerank, rerank_with_scores
from .rrf import reciprocal_rank_fusion
from .scored_candidate import ScoredSearchCandidate, merge_scored_candidates

__all__ = [
    "EntityHydrator",
    "EntityRecall",
    "EntityRecallStrategy",
    "SearchStrategy",
    "SearchFinalFilter",
    "ConsolidationResult",
    "MemoryConsolidator",
    "MemoryRetentionSelector",
    "RetentionSelection",
    "ScoredSearchCandidate",
    "build_entity_type_filter",
    "combine_entity_results_rrf",
    "merge_scored_candidates",
    "reciprocal_rank_fusion",
    "rerank",
    "rerank_with_scores",
]
