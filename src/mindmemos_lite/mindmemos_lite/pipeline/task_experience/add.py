"""Trajectory add pipeline: one task entity + reusable experiences, with LLM dedup."""

from __future__ import annotations

from typing import ClassVar, Literal

from ...components.extractor.task_experience import (
    ExperienceDeduplicator,
    TrajectoryExperienceBuilder,
    TrajectoryExperienceExtractor,
)
from ...components.text import MemoryVectorizer, SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...config import MemoryConfig, MessageChunkerConfig, TrajectoryAddConfig, TextProcessingConfig, get_config
from ...llm import get_embed_client, get_llm_client
from ...logging import get_logger, traced
from ...typing import (
    AddPipelineInput,
    AddPipelineSyncResult,
    MemoryDbMutationPlan,
    MemoryRequestContext,
)
from ..base import MemoryPersistencePipelineMixin
from ..registry import register

Consistency = Literal["fast", "strong"]
logger = get_logger(__name__)
_CLIENT_UNSET = object()


def _try_get_llm():
    """Try to resolve the global LLM client; return None if unavailable."""
    try:
        return get_llm_client()
    except Exception:  # noqa: BLE001
        logger.debug("llm_client_not_available", exc_info=True)
        return None


def _try_get_embed():
    """Try to resolve the global embedding client; return None if unavailable."""
    try:
        return get_embed_client()
    except Exception:  # noqa: BLE001
        logger.debug("embed_client_not_available", exc_info=True)
        return None


@register(type="add", name="trajectory_add")
class TrajectoryAddPipeline(MemoryPersistencePipelineMixin):
    """Distill a trajectory into one task entity plus reusable experience memories.

    The ``task`` input is required (``requires_task``), since every trajectory
    needs a task text to build its task entity. Each extracted experience is
    compared (dense recall + LLM judgment) against existing experiences so
    semantically identical experiences across tasks reuse a single memory node.
    """

    requires_task: ClassVar[bool] = True

    def __init__(
        self,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        trajectory_config: TrajectoryAddConfig | None = None,
        chunker_config: MessageChunkerConfig | None = None,
        extractor: TrajectoryExperienceExtractor | None = None,
        deduplicator: ExperienceDeduplicator | None = None,
        consistency: Consistency | None = None,
        llm_client=_CLIENT_UNSET,
        embed_client=_CLIENT_UNSET,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        cfg = text_config or get_config().algo_config.text_processing
        self._text_preprocessor = text_preprocessor or get_text_preprocessor(cfg)
        self._sparse_encoder = sparse_encoder or SparseVectorEncoder(cfg)
        self._explicit_trajectory_config = trajectory_config
        self._explicit_chunker_config = chunker_config
        self._explicit_consistency = consistency

        resolved_llm = _try_get_llm() if llm_client is _CLIENT_UNSET else llm_client
        resolved_embed = _try_get_embed() if embed_client is _CLIENT_UNSET else embed_client

        vectorizer = MemoryVectorizer(
            sparse_encoder=self._sparse_encoder,
            embed_client=resolved_embed,
            text_preprocessor=self._text_preprocessor,
        )
        self._builder = TrajectoryExperienceBuilder(
            text_preprocessor=self._text_preprocessor,
            extractor=extractor
            or TrajectoryExperienceExtractor(llm_client=resolved_llm),
            deduplicator=deduplicator
            or ExperienceDeduplicator(
                persistence=self.persistence,
                embed_client=resolved_embed,
                llm_client=resolved_llm,
            ),
            vectorizer=vectorizer,
            chunker_config=self._get_chunker_config(),
            llm_client=resolved_llm,
        )

    @classmethod
    def from_config(cls, config: MemoryConfig, **kwargs):
        """Construct the pipeline with its typed trajectory configuration."""

        return cls(
            text_config=config.algo_config.text_processing,
            trajectory_config=getattr(config.algo_config, "trajectory", None),
            chunker_config=config.algo_config.add.chunker,
            consistency=config.database.default_consistency,
            **kwargs,
        )

    def _get_consistency(self) -> Consistency:
        if self._explicit_consistency is not None:
            return self._explicit_consistency
        value = get_config().database.default_consistency
        return value if value in {"fast", "strong"} else "fast"

    def _get_trajectory_config(self) -> TrajectoryAddConfig:
        if self._explicit_trajectory_config is not None:
            return self._explicit_trajectory_config
        return getattr(get_config().algo_config, "trajectory", TrajectoryAddConfig())

    def _get_chunker_config(self) -> MessageChunkerConfig | None:
        if self._explicit_chunker_config is not None:
            return self._explicit_chunker_config
        return get_config().algo_config.add.chunker

    @traced("add.trajectory_add.sync")
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
    ) -> AddPipelineSyncResult:
        consistency = self._get_consistency()
        plan, events, update_commands = await self._builder.build(
            inp,
            context,
            consistency=consistency,
            config=self._get_trajectory_config(),
        )
        mutation_plan = MemoryDbMutationPlan.from_write_plan(plan)
        mutation_plan.memory_updates.extend(update_commands)
        if mutation_plan.has_writes() or mutation_plan.has_updates_or_deletes():
            await self.persistence.apply_mutation_plan(
                context,
                mutation_plan,
                consistency=consistency,
            )
        return AddPipelineSyncResult(status="ok", memories=events)