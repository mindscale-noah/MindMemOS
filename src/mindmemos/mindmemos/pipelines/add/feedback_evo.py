"""Feedback-driven self-evolution add pipeline (``feedback_evo`` mode)."""

from __future__ import annotations

from typing import Literal

from omegaconf import OmegaConf

from ...components.feedback_evo import ensure_evolution_state
from ...components.extractor.feedback_evo import FeedbackEvoMemoryExtractor
from ...components.extractor.vanilla import (
    AddCoreBuilder,
    AddSafetyGate,
    CandidateDeduplicator,
    RelatedMemoryRecall,
)
from ...components.kafka import memory_add_dispatch_key
from ...components.text import MemoryVectorizer, SparseVectorEncoder, get_text_preprocessor
from ...config import VanillaAddConfig, get_config
from ...errors import ConfigNotInitializedError
from ...infra.db import EvolutionStateStore
from ...llm import get_embed_client, get_llm_client
from ...logging import get_logger
from ...typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    MemoryDbMutationPlan,
    MemoryRequestContext,
)
from ..base import MemoryDbPipelineMixin
from ..memory_db import suppress_recording_errors
from ..registry import register

Consistency = Literal["fast", "strong"]
MEMORY_ADD_TOPIC = "memory.add"
logger = get_logger(__name__)


@register(type="add", name="feedback_evo_add")
class FeedbackEvoAddPipeline(MemoryDbPipelineMixin):
    """Independent feedback_evo add pipeline.

    Owns its orchestration (segment → preprocess → recall → extract → plan →
    vectorize → write) using shared components, and reads the evolved
    ``add_config`` live from the evolution state store. Does NOT delegate to the
    vanilla add pipeline — same isolation level as vanilla vs schema.
    """

    def __init__(self, *, state_store: EvolutionStateStore | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state_store = state_store or EvolutionStateStore()

    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        state = await ensure_evolution_state(self._state_store, context.project_id)
        add_cfg = state.add_config
        consistency = _default_consistency()
        return await self._run_add(inp, context, add_cfg, consistency, add_record_id)

    async def _run_add(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        add_cfg: dict,
        consistency: Consistency,
        add_record_id: str | None,
    ) -> AddPipelineSyncResult:
        fe_cfg = get_config().algo_config.add.feedback_evo
        extractor = FeedbackEvoMemoryExtractor(
            llm_client=_try_get_llm(),
            extraction_prompt=add_cfg.get("extraction_prompt") or fe_cfg.extraction_prompt,
            entity_tagging_prompt=add_cfg.get("entity_tagging_prompt") or fe_cfg.entity_tagging_prompt,
            entity_types=add_cfg.get("entity_types") or fe_cfg.entity_types,
        )
        builder = _builder_for(self.db_reader, extractor)
        config = _structural_add_config(
            enable_entities=bool(fe_cfg.enable_entities or extractor.entity_types)
        )
        plan, events, update_commands = await builder.build(
            inp,
            context,
            consistency=consistency,
            config=config,
        )
        mutation_plan = MemoryDbMutationPlan.from_write_plan(plan)
        mutation_plan.memory_updates.extend(update_commands)
        if mutation_plan.has_writes() or mutation_plan.has_updates_or_deletes():
            await self.db_writer.apply_mutation_plan(
                context,
                mutation_plan,
                consistency=consistency,
            )
        result = AddPipelineSyncResult(status="ok", memories=events)
        if add_record_id is not None:
            await suppress_recording_errors(
                self.recorder.mark_add_completed(context, add_record_id, result),
                operation="add.feedback_evo.sync",
            )
        return result

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict | None = None,
    ) -> AddPipelineAsyncResult:
        """Queue feedback_evo add work for background processing.

        The ``memory.add`` worker routes by the algorithm binding, so
        ``feedback_evo`` contexts are processed by this pipeline.
        """

        from ...infra.kafka import get_producer

        cfg = get_config()
        if not cfg.kafka.enabled:
            raise RuntimeError(
                "add_async requires Kafka to be enabled (kafka.enabled=true). "
                "Use mode='sync' or enable Kafka in config."
            )
        message = {
            "context": context.model_dump(),
            "input": inp.model_dump(by_alias=True),
        }
        if add_record_id is not None:
            message["add_record_id"] = add_record_id
        if record_metadata is not None:
            message["record_metadata"] = record_metadata
        await get_producer().send(
            MEMORY_ADD_TOPIC,
            value=message,
            dispatch_key=memory_add_dispatch_key(context),
        )
        return AddPipelineAsyncResult(status="queued")

    async def has_pending(self, context: MemoryRequestContext) -> bool:
        """Return whether this pipeline has pending asynchronous work."""

        del context
        return False


def _builder_for(db_reader, extractor: FeedbackEvoMemoryExtractor):
    """Build the shared flat-memory add components around the feedback_evo extractor."""

    text_cfg = get_config().algo_config.text_processing
    text_preprocessor = get_text_preprocessor(text_cfg)
    sparse_encoder = SparseVectorEncoder(text_cfg)
    llm = _try_get_llm()
    embed = _try_get_embed()
    vectorizer = MemoryVectorizer(
        sparse_encoder=sparse_encoder,
        embed_client=embed,
        text_preprocessor=text_preprocessor,
    )
    return AddCoreBuilder(
        text_preprocessor=text_preprocessor,
        memory_extractor=extractor,
        candidate_deduplicator=CandidateDeduplicator(),
        related_memory_recall=RelatedMemoryRecall(
            db_reader=db_reader,
            sparse_encoder=sparse_encoder,
        ),
        safety_gate=AddSafetyGate(),
        vectorizer=vectorizer,
        llm_client=llm,
    )


def _structural_add_config(*, enable_entities: bool) -> VanillaAddConfig:
    """Return shared flat-memory structural defaults with entity writes toggled."""

    try:
        cfg = get_config().algo_config.add.vanilla
        # ``algo_config`` is an OmegaConf tree, so merge instead of
        # ``dataclasses.replace`` (which requires a dataclass instance).
        return OmegaConf.merge(cfg, {"enable_entities": enable_entities})  # type: ignore[return-value]
    except ConfigNotInitializedError:
        return VanillaAddConfig()


def _default_consistency() -> Consistency:
    value = get_config().database.default_consistency
    return value if value in {"fast", "strong"} else "fast"


def _try_get_llm():
    """Try to resolve the global LLM client; return None if unavailable."""

    try:
        return get_llm_client()
    except Exception:
        logger.debug("llm_client_not_available", exc_info=True)
        return None


def _try_get_embed():
    """Try to resolve the global embedding client; return None if unavailable."""

    try:
        return get_embed_client()
    except Exception:
        logger.debug("embed_client_not_available", exc_info=True)
        return None
