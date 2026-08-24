import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import mindmemos.pipelines.add.schema.schema_add as schema_add
import pytest
from mindmemos.components.extractor.schema.v1 import _schema_higher_order
from mindmemos.components.memory_modeling.schema import EntityManager, EntityType
from mindmemos.config import bind_config_overrides, get_config, init_config, reset_config
from mindmemos.infra import db
from mindmemos.llm import ChatResponse, EmbeddingResponse
from mindmemos.pipelines.add import SchemaAddPipeline
from mindmemos.pipelines.memory_db import AddRecordBuffer, MemoryDbReader, buffer_key
from mindmemos.typing.memory import EntityView, MemoryRequestContext, MemoryView
from mindmemos.typing.memory_db import MemoryDbMutationResult, MemoryDbWriteResult
from mindmemos.typing.service import AddPipelineInput
from qdrant_client import models as qmodels

LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18 = [
    (
        "Caroline",
        "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.",
    ),
    (
        "Melanie",
        "You'd be a great counselor! Your empathy and understanding will really help the people you work with. "
        "By the way, take a look at this.",
    ),
    ("Caroline", "Thanks, Melanie! That's really sweet. Is this your own painting?"),
    ("Melanie", "Yeah, I painted that lake sunrise last year! It's special to me."),
    (
        "Caroline",
        "Wow, Melanie! The colors really blend nicely. Painting looks like a great outlet for expressing yourself.",
    ),
    (
        "Melanie",
        "Thanks, Caroline! Painting's a fun way to express my feelings and get creative. "
        "It's a great way to relax after a long day.",
    ),
    ("Caroline", "Totally agree, Mel. Relaxing and expressing ourselves is key. Well, I'm off to go do some research."),
    (
        "Melanie",
        "Yep, Caroline. Taking care of ourselves is vital. I'm off to go swimming with the kids. Talk to you soon!",
    ),
]


class FakeLLM:
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.episode_boundary":
            content = '[{"start_idx":0,"end_idx":0,"title":"Preference"}]'
        elif task == "memory.add.schema_selection":
            content = '{"selected_entities":[{"entity_type":"person","relevant_properties":["all"]}]}'
        elif task == "memory.add.entity_generation":
            content = """
            {
              "entities": [
                {
                  "name": "User",
                  "entity_type": "person",
                  "description": "Primary user who likes Qdrant.",
                  "properties": [
                    {
                      "property_name": "preference",
                      "value": "As of 2026-05-28, User likes Qdrant for vector search.",
                      "time": "2026-05-28"
                    }
                  ]
                }
              ],
              "edges": []
            }
            """
        elif task == "memory.add.episode_entity":
            content = '{"title":"User Preference About Qdrant","content":"The user said they like Qdrant.","search_fields":["Qdrant","preference"]}'
        elif task == "memory.add.episode_objectify":
            content = "On 2026-05-28, the user said they like Qdrant for vector search."
        else:
            content = '{"action":"create"}'
        parsed = format_parser(content) if format_parser else None
        return ChatResponse(finish_reason="stop", content=content, parsed=parsed)


class ManyPropertiesLLM(FakeLLM):
    """FakeLLM whose generated entity emits more properties than the per-entity cap."""

    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.entity_generation":
            properties = [
                {
                    "property_name": f"p{i}",
                    "value": f"As of 2026-05-28, User fact number {i}.",
                    "time": "2026-05-28",
                }
                for i in range(20)
            ]
            content = json.dumps(
                {
                    "entities": [
                        {
                            "name": "User",
                            "entity_type": "person",
                            "description": "Primary user.",
                            "properties": properties,
                        }
                    ],
                    "edges": [],
                }
            )
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class RecordingConversationTextLLM(FakeLLM):
    def __init__(self) -> None:
        self.prompts_by_task: dict[str, list[str]] = {}

    async def chat(self, task, messages, format_parser=None, **kwargs):
        self.prompts_by_task.setdefault(task, []).append(messages[0]["content"])
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class SingleEpisodeBoundaryRecordingLLM(RecordingConversationTextLLM):
    def __init__(self, *, end_idx: int) -> None:
        super().__init__()
        self.end_idx = end_idx

    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.episode_boundary":
            self.prompts_by_task.setdefault(task, []).append(messages[0]["content"])
            content = f'[{{"start_idx":0,"end_idx":{self.end_idx},"title":"LoCoMo Caroline Melanie exchange"}}]'
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class BoundarySequenceLLM(RecordingConversationTextLLM):
    def __init__(self, boundaries: list[str]) -> None:
        super().__init__()
        self.boundaries = list(boundaries)

    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.episode_boundary":
            self.prompts_by_task.setdefault(task, []).append(messages[0]["content"])
            content = self.boundaries.pop(0) if self.boundaries else "[]"
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class FakeEmbed:
    async def embed(self, task, text, **kwargs):
        texts = text if isinstance(text, list) else [text]
        embeddings = [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]
        return EmbeddingResponse(embeddings=embeddings)


class RecordingFakeEmbed(FakeEmbed):
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, task, text, **kwargs):
        self.calls.append(SimpleNamespace(task=task, text=text))
        return await super().embed(task, text, **kwargs)


class RequestScopedLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if not get_config().chat_model_router.endpoints:
            raise RuntimeError("No chat model endpoint configured")
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class RequestScopedEmbed(FakeEmbed):
    async def embed(self, task, text, **kwargs):
        if not get_config().embed_model_router.endpoints:
            raise RuntimeError("No embed model endpoint configured")
        return await super().embed(task, text, **kwargs)


class FakeQdrant:
    def __init__(self):
        self.add_records = {}
        self.schema_buffer_records = {}
        self.memories = {}
        self.entities = {}
        self.add_record_orders = []
        self.add_record_global_orders = []
        self.schema_buffer_orders = []
        self.schema_buffer_global_orders = []

    async def upsert_add_record(self, point):
        await self.upsert_add_records([point])

    async def upsert_add_records(self, points):
        for point in points:
            self.add_records[point.add_record_id] = dict(point.payload)

    async def scroll_add_records(self, project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        self.add_record_orders.append(order_by)
        records = [
            SimpleNamespace(point_id=record_id, payload=payload)
            for record_id, payload in _sort_add_record_items(self.add_records.items(), order_by)
            if payload.get("project_id") == project_id and _matches_qdrant_filter(payload, filter_)
        ]
        return records[:limit], None

    async def scroll_add_records_global(self, *, filter_=None, limit=50, cursor=None, order_by=None):
        self.add_record_global_orders.append(order_by)
        records = [
            SimpleNamespace(point_id=record_id, payload=payload)
            for record_id, payload in _sort_add_record_items(self.add_records.items(), order_by)
            if _matches_qdrant_filter(payload, filter_)
        ]
        return records[:limit], None

    async def get_add_records_by_ids(self, project_id, add_record_ids):
        return [
            SimpleNamespace(point_id=record_id, payload=payload)
            for record_id, payload in self.add_records.items()
            if record_id in add_record_ids and payload.get("project_id") == project_id
        ]

    async def patch_add_record(self, project_id, add_record_id, payload):
        record = self.add_records.get(add_record_id)
        if record and record.get("project_id") == project_id:
            record.update(payload)

    async def upsert_schema_add_buffer_records(self, points):
        for point in points:
            self.schema_buffer_records[point.schema_buffer_record_id] = dict(point.payload)

    async def scroll_schema_add_buffer_records(self, project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        self.schema_buffer_orders.append(order_by)
        records = [
            SimpleNamespace(point_id=record_id, payload=payload)
            for record_id, payload in _sort_add_record_items(self.schema_buffer_records.items(), order_by)
            if payload.get("project_id") == project_id and _matches_qdrant_filter(payload, filter_)
        ]
        return records[:limit], None

    async def scroll_schema_add_buffer_records_global(self, *, filter_=None, limit=50, cursor=None, order_by=None):
        self.schema_buffer_global_orders.append(order_by)
        records = [
            SimpleNamespace(point_id=record_id, payload=payload)
            for record_id, payload in _sort_add_record_items(self.schema_buffer_records.items(), order_by)
            if _matches_qdrant_filter(payload, filter_)
        ]
        return records[:limit], None

    async def get_schema_add_buffer_records_by_ids(self, project_id, schema_buffer_record_ids):
        return [
            SimpleNamespace(point_id=record_id, payload=payload)
            for record_id, payload in self.schema_buffer_records.items()
            if record_id in schema_buffer_record_ids and payload.get("project_id") == project_id
        ]

    async def patch_schema_add_buffer_record(self, project_id, schema_buffer_record_id, payload):
        record = self.schema_buffer_records.get(schema_buffer_record_id)
        if record and record.get("project_id") == project_id:
            record.update(payload)

    async def search_entity_dense(self, project_id, vector, *, filter_=None, limit=10, score_threshold=None):
        return []

    async def search_memory_dense(self, project_id, vector, *, filter_=None, limit=10, score_threshold=None):
        return []

    async def scroll_memories(self, project_id, *, filter_=None, limit=50, cursor=None):
        records = [
            db.QdrantRecord(point_id=memory_id, payload=payload)
            for memory_id, payload in self.memories.items()
            if payload.get("project_id") == project_id
        ]
        return records[:limit], None

    async def get_memory(self, project_id, memory_id):
        payload = self.memories.get(memory_id)
        if not payload or payload.get("project_id") != project_id:
            return None
        return db.QdrantRecord(point_id=memory_id, payload=payload)

    async def get_memories(self, project_id, memory_ids):
        return [
            db.QdrantRecord(point_id=memory_id, payload=self.memories[memory_id])
            for memory_id in memory_ids
            if memory_id in self.memories and self.memories[memory_id].get("project_id") == project_id
        ]

    async def update_memory_payload(self, project_id, memory_id, payload):
        record = self.memories.get(memory_id)
        if record and record.get("project_id") == project_id:
            record.update(payload)

    async def delete_memory(self, project_id, memory_id):
        record = self.memories.get(memory_id)
        if record and record.get("project_id") == project_id:
            del self.memories[memory_id]


class FakeWriter:
    def __init__(self):
        self.calls = []
        self.mutation_plans = []
        self.deleted: list[str] = []
        self.updated = []
        self.entity_updated = []

    async def apply_mutation_plan(self, ctx, plan, *, consistency="fast"):
        self.mutation_plans.append((ctx, plan, consistency))
        write_plan = plan.to_write_plan()
        if plan.has_writes():
            self.calls.append((ctx, write_plan, consistency))
        for command in plan.entity_updates:
            if command.entity is None:
                continue
            entity_vectors = [vector for vector in [command.core_vector, *command.search_field_vectors] if vector]
            self.entity_updated.append(SimpleNamespace(ctx=ctx, entity=command.entity, entity_vectors=entity_vectors))

        mutations = []
        for command in plan.memory_updates:
            self.updated.append(command)
            mutations.append(MemoryDbMutationResult(memory_id=f"{command.memory_id}-new", changed=True))
        for command in plan.memory_deletes:
            self.deleted.append(command.memory_id)
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in write_plan.memories],
            entity_ids=[entity.entity_id for entity in write_plan.entities],
            mutations=mutations,
        )

    async def write(self, ctx, plan, *, consistency="fast"):
        self.calls.append((ctx, plan, consistency))
        return SimpleNamespace(memory_ids=[memory.memory_id for memory in plan.memories])

    async def update_entity(self, ctx, entity, *, entity_vectors=(), consistency="fast"):
        self.entity_updated.append(SimpleNamespace(ctx=ctx, entity=entity, entity_vectors=list(entity_vectors)))
        return SimpleNamespace(entity_ids=[entity.entity_id], graph_pending=False, errors=[])

    async def update_memory(self, ctx, req):
        self.updated.append(req)
        return SimpleNamespace(status="ok", memory_id=f"{req.memory_id}-new", changed=True)

    async def delete_memory(self, ctx, req):
        self.deleted.append(req.memory_id)
        return SimpleNamespace(status="ok", memory_id=req.memory_id, changed=True)


class FakeRecorder:
    def __init__(self) -> None:
        self.completed = []
        self.appended = []

    async def mark_add_completed(self, ctx, add_record_id, result) -> None:
        self.completed.append((ctx, add_record_id, result))

    async def append_add_output(self, ctx, add_record_id, events) -> None:
        self.appended.append((ctx, add_record_id, events))


class FakeProducer:
    def __init__(self):
        self.calls = []

    async def send(self, topic, value, *, dispatch_key=None):
        self.calls.append(SimpleNamespace(topic=topic, value=value, dispatch_key=dispatch_key))


class FailingProducer:
    async def send(self, topic, value, *, dispatch_key=None):
        raise RuntimeError("kafka unavailable")


def _matches_qdrant_filter(payload, qfilter):
    if qfilter is None:
        return True
    return all(_matches_condition(payload, condition) for condition in qfilter.must or []) and not any(
        _matches_condition(payload, condition) for condition in qfilter.must_not or []
    )


def _matches_condition(payload, condition):
    key = getattr(condition, "key", None)
    if key is None:
        return True
    match = getattr(condition, "match", None)
    if match is None:
        return True
    if getattr(match, "value", None) is not None:
        return payload.get(key) == match.value
    values = getattr(match, "any", None)
    if values is not None:
        return payload.get(key) in values
    excluded = getattr(match, "except", None)
    if excluded is not None:
        return payload.get(key) not in excluded
    return True


def _sort_add_record_items(items, order_by):
    items = list(items)
    if order_by is None:
        return items
    key = getattr(order_by, "key", None)
    if not key:
        return items
    direction = getattr(order_by, "direction", None)
    reverse = direction == qmodels.Direction.DESC or str(direction).lower().endswith("desc")
    return sorted(items, key=lambda item: item[1].get(key, 0), reverse=reverse)


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeReader(MemoryDbReader):
    def __init__(
        self,
        *,
        entity_candidates: list[EntityView] | None = None,
        property_hits: list[MemoryView] | None = None,
        list_memories: list[MemoryView] | None = None,
    ) -> None:
        self.entity_candidates = entity_candidates or []
        self.property_hits = property_hits or []
        self._list_memories = list_memories or []

    async def search_entities_dense(self, ctx, *, query, query_vector, filters=None, limit=10, score_threshold=None):
        hits = [
            SimpleNamespace(entity_id=entity.entity_id, score=0.9, entity=entity, source="fake", rank=index)
            for index, entity in enumerate(self.entity_candidates[:limit], start=1)
        ]
        return SimpleNamespace(query=query, hits=hits, total=len(hits), debug={})

    async def search_entity_property_memories(
        self,
        ctx,
        *,
        query_vector,
        entity_id,
        limit=5,
        score_threshold=None,
    ):
        hits = [
            SimpleNamespace(memory_id=memory.memory_id, score=0.9, memory=memory, source="fake", rank=index)
            for index, memory in enumerate(self.property_hits[:limit], start=1)
            if memory.entity_id == entity_id
        ]
        return SimpleNamespace(query=entity_id, hits=hits, total=len(hits), debug={})

    async def list_memories(self, ctx, *, filters=None, limit=50, cursor=None):
        return self._list_memories[:limit], None


class PropertyMergeLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.property_merge":
            content = '{"existing":[{"id":"p1","op":"delete"}],"new":[{"id":"n1","op":"update","target":"p2","value":"merged preference"}]}'
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class HigherOrderLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.higher_order_generation":
            content = '{"updates":[{"property_name":"preference_summary","action":"update","value":"User consistently prefers vector databases. Evidence: Qdrant memories in 2026. Confidence: high.","reasoning":"Repeated preference evidence."}]}'
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class EmptyBoundaryLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.episode_boundary":
            content = "[]"
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class FailingGenerationLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.entity_generation":
            raise RuntimeError("entity generation failed")
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class DuplicateEdgeLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.schema_selection":
            content = '{"selected_entities":[{"entity_type":"person","relevant_properties":["all"]},{"entity_type":"item","relevant_properties":["all"]}]}'
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        if task == "memory.add.entity_generation":
            content = """
            {
              "entities": [
                {
                  "name": "User",
                  "entity_type": "person",
                  "description": "Primary user who likes Qdrant.",
                  "properties": [
                    {
                      "property_name": "preference",
                      "value": "As of 2026-05-28, User likes Qdrant for vector search.",
                      "time": "2026-05-28"
                    }
                  ]
                },
                {
                  "name": "Qdrant",
                  "entity_type": "item",
                  "description": "Vector database the user likes.",
                  "properties": [
                    {
                      "property_name": "default_property",
                      "value": "As of 2026-05-28, Qdrant is a vector database the user likes.",
                      "time": "2026-05-28"
                    }
                  ]
                }
              ],
              "edges": [
                {
                  "link_entity1_name": "User",
                  "link_entity2_name": "Qdrant",
                  "link_description": "likes"
                },
                {
                  "link_entity1_name": "Qdrant",
                  "link_entity2_name": "User",
                  "link_description": "liked_by"
                }
              ]
            }
            """
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


class MixedSchemaTypeLLM(FakeLLM):
    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.schema_selection":
            content = '{"selected_entities":[{"entity_type":"person","relevant_properties":["all"]},{"entity_type":"task_experience","relevant_properties":["all"]},{"entity_type":"organization","relevant_properties":["all"]}]}'
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        if task == "memory.add.entity_generation":
            content = """
            {
              "entities": [
                {
                  "name": "User",
                  "entity_type": "person",
                  "description": "Primary user who likes Qdrant.",
                  "properties": [
                    {
                      "property_name": "preference",
                      "value": "As of 2026-05-28, User likes Qdrant for vector search.",
                      "time": "2026-05-28"
                    }
                  ]
                },
                {
                  "name": "Qdrant Debugging",
                  "entity_type": "task_experience",
                  "description": "Reusable task experience from debugging Qdrant.",
                  "properties": [
                    {
                      "property_name": "default_property",
                      "value": "As of 2026-05-28, checking Qdrant payload filters before vector scoring avoids wasted debugging time.",
                      "time": "2026-05-28"
                    }
                  ]
                },
                {
                  "name": "Qdrant",
                  "entity_type": "organization",
                  "description": "Vector database organization.",
                  "properties": [
                    {
                      "property_name": "service_info",
                      "value": "As of 2026-05-28, Qdrant provides vector search infrastructure.",
                      "time": "2026-05-28"
                    }
                  ]
                }
              ],
              "edges": []
            }
            """
            parsed = format_parser(content) if format_parser else None
            return ChatResponse(finish_reason="stop", content=content, parsed=parsed)
        return await super().chat(task, messages, format_parser=format_parser, **kwargs)


@pytest.fixture(autouse=True)
def config_context():
    init_config(config_path="config/mindmemos/dev.example.yaml")
    try:
        yield
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_schema_add_pipeline_resolves_request_scoped_clients_when_provider_binding_enabled(monkeypatch, tmp_path):
    config_path = tmp_path / "project-namespaced.yaml"
    config_path.write_text(
        Path("config/mindmemos/dev.example.yaml")
        .read_text(encoding="utf-8")
        .replace("    vector_size: 2560", "    vector_size: 2560\n    project_collection_namespace_enabled: true"),
        encoding="utf-8",
    )
    reset_config()
    init_config(config_path=config_path)
    cfg = get_config()
    cfg.provider_binding.enabled = True
    cfg.chat_model_router.endpoints.clear()
    cfg.embed_model_router.endpoints.clear()
    llm = RequestScopedLLM()
    embed = RequestScopedEmbed()
    monkeypatch.setattr(schema_add, "get_llm_client", lambda: llm)
    monkeypatch.setattr(schema_add, "get_embed_client", lambda: embed)
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    assert pipeline._explicit_llm is None
    assert pipeline._explicit_embed is None

    with bind_config_overrides(
        allow_project_embedding_dimensions=True,
        project_config={
            "chat_model_router": {
                "endpoints": [
                    {
                        "model": "openai/request-chat",
                        "api_key": "sk-request",
                        "api_base": "https://example.test/v1",
                    }
                ]
            },
            "embed_model_router": {
                "endpoints": [
                    {
                        "model": "openai/request-embed",
                        "api_key": "sk-request",
                        "api_base": "https://example.test/v1",
                        "dimensions": 3,
                    }
                ]
            },
        },
    ):
        runtime = pipeline._resolve_add_runtime(make_context())
        assert runtime.chunker.llm_client is llm
        assert runtime.extractor.llm_client is llm
        assert runtime.planner.llm_client is llm
        assert runtime.planner.embed_client is embed
        result = await pipeline.add_sync(
            AddPipelineInput(
                mode="sync",
                timestamp=1770000000000,
                force_generation=True,
                messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
            ),
            make_context(),
        )

    assert result.status == "ok"
    assert writer.calls


def test_schema_add_pipeline_resolves_clients_from_current_runtime_when_provider_binding_disabled(monkeypatch):
    cfg = get_config()
    cfg.provider_binding.enabled = False
    llm = RequestScopedLLM()
    embed = RequestScopedEmbed()
    monkeypatch.setattr(schema_add, "get_llm_client", lambda: llm)
    monkeypatch.setattr(schema_add, "get_embed_client", lambda: embed)
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)

    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    runtime = pipeline._resolve_add_runtime(make_context())
    assert runtime.chunker.llm_client is llm
    assert runtime.extractor.llm_client is llm
    assert runtime.planner.llm_client is llm
    assert runtime.planner.embed_client is embed


@pytest.mark.asyncio
async def test_schema_add_sync_stream_resolves_explicit_consistency():
    class FakeBuffer:
        async def append(self, context, inp, *, force_generation, source_add_record_id):
            return ["buffer-record-1"]

    class FakeRecorder:
        async def mark_add_completed(self, context, add_record_id, result):
            return None

    pipeline = SchemaAddPipeline.__new__(SchemaAddPipeline)
    pipeline.add_buffer = FakeBuffer()
    pipeline.recorder = FakeRecorder()
    pipeline._explicit_consistency = "strong"
    observed = {}

    async def drain(context, *, consistency, force, progress, cancel_check):
        observed["consistency"] = consistency
        return []

    pipeline._ensure_drain_and_wait = drain

    result = await pipeline.add_sync_stream(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            messages=[{"role": "user", "content": "Remember this."}],
        ),
        make_context(),
    )

    assert result.status == "ok"
    assert observed["consistency"] == "strong"


@pytest.mark.asyncio
async def test_schema_add_pipeline_writes_entity_and_property_vectors():
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    embed = RecordingFakeEmbed()
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=embed,
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    before = datetime.now(UTC)
    result = await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[
                {
                    "role": "user",
                    "content": "I like Qdrant for vector search.",
                }
            ],
        ),
        make_context(),
    )
    after = datetime.now(UTC)

    assert result.status == "ok"
    assert len(writer.calls) == 1
    _, plan, consistency = writer.calls[0]
    assert consistency == "fast"
    assert {entity.entity_type for entity in plan.entities} == {"person", "episodes"}
    base_entity_vector_ids = {vector.entity_id for vector in plan.entity_vectors if "#sf" not in vector.entity_id}
    assert base_entity_vector_ids == {entity.entity_id for entity in plan.entities}
    assert len(plan.memories) == 2
    assert len(plan.vectors) == len(plan.memories)
    assert all(vector.semantic_vector for vector in plan.vectors)
    assert all(before <= memory.created_at <= after for memory in plan.memories)
    assert all(before <= entity.created_at <= after for entity in plan.entities)
    validate_from_by_property = {memory.property_name: memory.validate_from for memory in plan.memories}
    assert validate_from_by_property["preference"] == datetime(2026, 5, 28, tzinfo=UTC)
    assert validate_from_by_property["input_messages"] == datetime(2026, 2, 2, tzinfo=UTC)
    assert {memory.mem_type for memory in plan.memories} == {"profile", "episodic"}
    assert any(memory.property_name == "preference" for memory in plan.memories)
    assert any(memory.property_name == "input_messages" for memory in plan.memories)
    property_embed_call = next(call for call in embed.calls if call.task == "memory.add.property")
    assert any(
        text == "User:preference:As of 2026-05-28, User likes Qdrant for vector search."
        for text in property_embed_call.text
    )
    assert any(
        ":input_messages:On 2026-05-28, the user said they like Qdrant for vector search." in text
        for text in property_embed_call.text
    )
    assert "As of 2026-05-28, User likes Qdrant for vector search." in result.memories[0].content
    assert "Entity: User (Type: person)" in result.memories[0].content
    assert result.memories[0].mem_type == "profile"
    assert result.memories[0].memory_type == "profile"
    assert qdrant.add_records == {}
    assert all(payload["buffer_status"] == "processed" for payload in qdrant.schema_buffer_records.values())


@pytest.mark.asyncio
async def test_schema_add_pipeline_caps_properties_per_entity():
    """Properties beyond max_properties_per_entity are dropped before writing/embedding."""
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=ManyPropertiesLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    result = await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
        ),
        make_context(),
    )

    assert result.status == "ok"
    _, plan, _ = writer.calls[0]
    property_names = {memory.property_name for memory in plan.memories}
    # 20 properties emitted, capped to 15 (default max_properties_per_entity).
    # Non-schema property names (p0..p19) are rewritten to "default_property" by the
    # normalizer, so we verify the cap by counting default_property memories on the person
    # entity instead of checking for the raw p0..p14 names.
    default_count = sum(1 for m in plan.memories if m.property_name == "default_property" and m.entity_type == "person")
    assert default_count == 15, f"cap should keep 15, got {default_count}"


@pytest.mark.asyncio
async def test_schema_add_pipeline_passes_locomo_named_speakers_to_extractor():
    """Regression sample from locomo10.json conversation 0, session_1 D1:11-D1:18."""
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    llm = SingleEpisodeBoundaryRecordingLLM(end_idx=len(LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18) - 1)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=llm,
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    result = await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": speaker, "content": text} for speaker, text in LOCOMO_CAROLINE_MELANIE_D1_11_TO_D1_18],
        ),
        make_context(),
    )

    caroline_line = (
        "0. 2026-02-02 02:40:00 speaker=Caroline: "
        "I'm keen on counseling or working in mental health - I'd love to support those with similar issues."
    )
    melanie_line = "1. 2026-02-02 02:40:00 speaker=Melanie: You'd be a great counselor!"
    assert result.status == "ok"
    for task in [
        "memory.add.schema_selection",
        "memory.add.entity_generation",
        "memory.add.episode_objectify",
        "memory.add.episode_entity",
    ]:
        prompt = llm.prompts_by_task[task][0]
        assert caroline_line in prompt
        assert melanie_line in prompt


@pytest.mark.asyncio
async def test_schema_add_pipeline_maps_arbitrary_schema_types_to_display_memory_types():
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    manager = EntityManager("config/presets/entity_modeling_locomo.json")
    manager.register(
        "task_experience",
        entity_description="Reusable task experience.",
        dynamic_property={"default_property": {"type": "string", "desc": "Task experience."}},
    )
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=MixedSchemaTypeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=manager,
    )

    result = await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant and learned a debugging trick."}],
        ),
        make_context(),
    )

    _, plan, _ = writer.calls[0]
    type_by_entity_type_and_property = {
        (memory.entity_type, memory.property_name): memory.mem_type for memory in plan.memories
    }
    assert type_by_entity_type_and_property[("person", "preference")] == "profile"
    assert type_by_entity_type_and_property[("task_experience", "default_property")] == "experience"
    assert type_by_entity_type_and_property[("organization", "service_info")] == "fact"
    assert type_by_entity_type_and_property[("episodes", "input_messages")] == "episodic"
    assert {event.mem_type for event in result.memories} == {"profile", "experience", "fact", "episodic"}
    assert {event.memory_type for event in result.memories} == {"profile", "experience", "fact", "episodic"}


@pytest.mark.asyncio
async def test_schema_add_pipeline_deduplicates_entity_edges_by_pair():
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=DuplicateEdgeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
        ),
        make_context(),
    )

    _, plan, _ = writer.calls[0]
    entity_edges = [
        relationship
        for relationship in plan.relationships
        if relationship.source.kind == "Entity" and relationship.target.kind == "Entity"
    ]
    assert len(entity_edges) == 1
    assert {entity_edges[0].source.node_id, entity_edges[0].target.node_id} <= {
        entity.entity_id for entity in plan.entities
    }


@pytest.mark.asyncio
async def test_schema_add_pipeline_uses_add_input_timestamp_for_text_messages():
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )
    add_input = AddPipelineInput(
        mode="sync",
        timestamp=1770000000000,
        force_generation=True,
        messages=[{"text": "I like Qdrant for vector search."}],
    )

    await pipeline.add_sync(add_input, make_context())

    stored_record = next(iter(qdrant.schema_buffer_records.values()))
    assert stored_record["timestamp"] == add_input.timestamp
    assert stored_record["event_timestamp_ms"] == add_input.timestamp
    assert stored_record["added_timestamp_ms"] != add_input.timestamp
    assert stored_record["buffer_sequence"] != add_input.timestamp * 1000
    assert "timestamp" not in stored_record["messages"][0]


@pytest.mark.asyncio
async def test_schema_add_pipeline_writes_back_to_trigger_add_record_and_links_buffer_source():
    writer = FakeWriter()
    recorder = FakeRecorder()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        recorder=recorder,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    result = await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
        ),
        make_context(),
        add_record_id="request-add-record-1",
    )

    assert qdrant.add_records == {}
    assert {payload["source_add_record_id"] for payload in qdrant.schema_buffer_records.values()} == {
        "request-add-record-1"
    }
    assert [call[1] for call in recorder.completed] == ["request-add-record-1"]
    assert recorder.completed[0][2] is result


@pytest.mark.asyncio
async def test_schema_add_buffer_orders_by_added_time_not_event_time():
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    buffer = AddRecordBuffer(clients=clients)
    inp = AddPipelineInput(
        mode="sync",
        timestamp=1770000000000,
        messages=[
            {"role": "user", "content": "Newer event first.", "timestamp": 1770000001000},
            {"role": "user", "content": "Older event second.", "timestamp": 1700000000000},
        ],
    )
    before = datetime.now(UTC)

    await buffer.append(make_context(), inp, force_generation=False)

    after = datetime.now(UTC)
    records = await buffer.list_buffered(make_context(), limit=10)
    assert [record.payload["messages"][0]["content"] for record in records] == [
        "Newer event first.",
        "Older event second.",
    ]
    assert qdrant.schema_buffer_orders[-1] == qmodels.OrderBy(
        key="buffer_sequence",
        direction=qmodels.Direction.ASC,
    )
    assert [record.payload["event_timestamp_ms"] for record in records] == [1770000000000, 1770000000000]
    assert all(before <= record.payload["added_at"] <= after for record in records)
    assert records[0].buffer_sequence < records[1].buffer_sequence


@pytest.mark.asyncio
async def test_schema_add_buffer_preserves_prompt_language():
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    buffer = AddRecordBuffer(clients=clients)
    inp = AddPipelineInput(
        mode="sync",
        prompt_language="ZH",
        messages=[{"role": "user", "content": "Prefer stable algorithms."}],
    )

    await buffer.append(make_context(), inp, force_generation=True)

    records = await buffer.list_buffered(make_context(), limit=10)
    assert records[0].payload["prompt_language"] == "ZH"
    assert schema_add._reconstruct_input_from_records(records).prompt_language == "ZH"


def test_schema_add_prompt_language_for_records_prefers_explicit_request(monkeypatch):
    records = [
        schema_add.BufferedAddRecord(
            add_record_id="record-1",
            payload={
                "prompt_language": "ZH",
                "messages": [{"role": "user", "content": "Prefer stable algorithms."}],
            },
        )
    ]
    monkeypatch.setattr(
        schema_add,
        "detect_prompt_language",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("language should not be auto-detected")),
    )

    assert schema_add._prompt_language_for_records(records, "Prefer stable algorithms.") == "ZH"


@pytest.mark.asyncio
async def test_schema_add_buffer_uses_message_timestamp_when_request_timestamp_missing():
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    buffer = AddRecordBuffer(clients=clients)
    inp = AddPipelineInput(
        mode="sync",
        messages=[
            {"role": "user", "content": "Newer event first.", "timestamp": 1770000001000},
            {"role": "user", "content": "Older event second.", "timestamp": 1700000000000},
        ],
    )

    await buffer.append(make_context(), inp, force_generation=False)

    records = await buffer.list_buffered(make_context(), limit=10)
    assert [record.payload["event_timestamp_ms"] for record in records] == [1770000001000, 1700000000000]


@pytest.mark.asyncio
async def test_schema_add_buffer_get_and_patch_are_scoped_by_buffer_key():
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    buffer = AddRecordBuffer(clients=clients)
    ctx = make_context()
    other_ctx = ctx.model_copy(update={"user_id": "user-2", "session_id": "session-2"})

    await buffer.append(
        ctx,
        AddPipelineInput(
            mode="async",
            timestamp=1770000000000,
            messages=[{"role": "user", "content": "user one"}],
        ),
        force_generation=False,
    )
    await buffer.append(
        other_ctx,
        AddPipelineInput(
            mode="async",
            timestamp=1770000000001,
            messages=[{"role": "user", "content": "user two"}],
        ),
        force_generation=False,
    )

    record = (await buffer.list_buffered(ctx, limit=10))[0]
    other_record = (await buffer.list_buffered(other_ctx, limit=10))[0]

    assert await buffer.get_by_ids(ctx, [record.add_record_id, other_record.add_record_id]) == [record]

    await buffer.mark_processing(ctx, [record, other_record])

    assert qdrant.schema_buffer_records[record.add_record_id]["buffer_status"] == "processing"
    assert qdrant.schema_buffer_records[other_record.add_record_id]["buffer_status"] == "buffered"


@pytest.mark.asyncio
async def test_schema_add_async_appends_buffer_and_publishes_drain_task(monkeypatch):
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    producer = FakeProducer()
    monkeypatch.setattr(schema_add, "get_producer", lambda: producer)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    ctx = make_context()
    result = await pipeline.add_async(
        AddPipelineInput(
            mode="async",
            timestamp=1770000000000,
            messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
        ),
        ctx,
    )

    assert result.status == "queued"
    assert writer.calls == []
    assert qdrant.add_records == {}
    assert len(qdrant.schema_buffer_records) == 1
    stored_record = next(iter(qdrant.schema_buffer_records.values()))
    assert stored_record["buffer_status"] == "buffered"
    assert len(producer.calls) == 1
    assert producer.calls[0].topic == schema_add.SCHEMA_ADD_DRAIN_TOPIC
    assert producer.calls[0].dispatch_key == buffer_key(ctx)
    assert producer.calls[0].value["context"]["project_id"] == "proj-1"
    assert producer.calls[0].value["force"] is False


@pytest.mark.asyncio
async def test_schema_add_runtime_resolution_failure_releases_processing_state(monkeypatch):
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    pipeline = SchemaAddPipeline(
        db_reader=MemoryDbReader(clients=clients),
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )
    ctx = make_context()
    key = buffer_key(ctx)

    def fail_runtime_resolution(_context):
        raise RuntimeError("runtime initialization failed")

    monkeypatch.setattr(pipeline, "_resolve_add_runtime", fail_runtime_resolution)

    assert await pipeline._try_start_loop(ctx) is True
    with pytest.raises(RuntimeError, match="runtime initialization failed"):
        await pipeline._process_loop(
            ctx,
            consistency="strong",
            force=True,
            inline=True,
        )

    assert pipeline._processing_by_key[key] is False
    assert await pipeline._try_start_loop(ctx) is True
    await pipeline._finish_processing(ctx)


@pytest.mark.asyncio
async def test_schema_add_episode_kafka_dispatch_uses_buffer_key(monkeypatch):
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    producer = FakeProducer()
    monkeypatch.setattr(schema_add, "get_producer", lambda: producer)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )
    ctx = make_context()

    await pipeline.add_buffer.append(
        ctx,
        AddPipelineInput(
            mode="async",
            timestamp=1770000000000,
            messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
        ),
        force_generation=False,
    )
    records = await pipeline.add_buffer.list_buffered(ctx, limit=10)

    await pipeline._dispatch_episodes_kafka(
        [schema_add._EpisodeTask(episode_id="episode-1", records=records)],
        context=ctx,
        consistency="fast",
    )

    assert len(producer.calls) == 1
    assert producer.calls[0].topic == schema_add.SCHEMA_ADD_EPISODE_TOPIC
    assert producer.calls[0].dispatch_key == buffer_key(ctx)
    assert producer.calls[0].value["context"]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_schema_add_async_raises_when_kafka_disabled(monkeypatch):
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    real_get_config = schema_add.get_config

    def fake_get_config():
        cfg = real_get_config()
        return SimpleNamespace(
            kafka=SimpleNamespace(enabled=False),
            algo_config=cfg.algo_config,
            database=cfg.database,
        )

    monkeypatch.setattr(schema_add, "get_config", fake_get_config)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    with pytest.raises(RuntimeError, match="kafka.enabled=true"):
        await pipeline.add_async(
            AddPipelineInput(
                mode="async",
                timestamp=1770000000000,
                messages=[{"role": "user", "content": "Kafka must be enabled."}],
            ),
            make_context(),
        )

    assert qdrant.add_records == {}
    assert qdrant.schema_buffer_records == {}


@pytest.mark.asyncio
async def test_schema_add_async_raises_when_drain_publish_fails(monkeypatch):
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    monkeypatch.setattr(schema_add, "get_producer", lambda: FailingProducer())
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    async def fail_if_inline_drain_attempted(context):
        raise AssertionError("schema add async must publish drain to kafka")

    monkeypatch.setattr(pipeline, "_try_start_loop", fail_if_inline_drain_attempted)

    with pytest.raises(RuntimeError, match="kafka unavailable"):
        await pipeline.add_async(
            AddPipelineInput(
                mode="async",
                timestamp=1770000000000,
                messages=[{"role": "user", "content": "Publish should fail."}],
            ),
            make_context(),
        )

    assert qdrant.add_records == {}
    assert len(qdrant.schema_buffer_records) == 1
    stored_record = next(iter(qdrant.schema_buffer_records.values()))
    assert stored_record["buffer_status"] == "buffered"
    assert writer.calls == []


@pytest.mark.asyncio
async def test_schema_add_drain_uses_buffered_record_context_for_episode_kafka(monkeypatch):
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    producer = FakeProducer()
    monkeypatch.setattr(schema_add, "get_producer", lambda: producer)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )
    original_context = make_context()
    drain_context = original_context.model_copy(update={"request_id": "drain-req", "user_id": "drain-user"})

    await pipeline.add_buffer.append(
        original_context,
        AddPipelineInput(
            mode="async",
            timestamp=1770000000000,
            messages=[{"role": "user", "content": "I like Qdrant for vector search."}],
        ),
        force_generation=True,
    )

    await pipeline.drain_buffer(drain_context, force=True)

    assert writer.calls == []
    assert len(producer.calls) == 1
    assert producer.calls[0].topic == schema_add.SCHEMA_ADD_EPISODE_TOPIC
    assert producer.calls[0].value["context"]["request_id"] == original_context.request_id
    assert producer.calls[0].value["context"]["user_id"] == original_context.user_id


@pytest.mark.asyncio
async def test_schema_add_pipeline_forces_split_when_llm_returns_no_boundary_at_max_length():
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=EmptyBoundaryLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    await pipeline.add_sync(
        AddPipelineInput(
            mode="async",
            timestamp=1770000000000,
            messages=[
                {"role": "user", "content": f"message {index}"}
                for index in range(get_config().algo_config.add.schema.chunker.max_episode_length)
            ],
        ),
        make_context(),
    )

    assert len(writer.calls) == 1
    assert qdrant.add_records == {}
    assert all(payload["buffer_status"] == "processed" for payload in qdrant.schema_buffer_records.values())


@pytest.mark.asyncio
async def test_schema_add_pipeline_property_merge_archives_existing_and_writes_merged_value():
    get_config().algo_config.add.schema.version = "v1"
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    existing_one = MemoryView(
        memory_id="mem-old-1",
        project_id="proj-1",
        content="old plan",
        mem_type="fact",
        status="active",
        property_name="plan_event",
        entity_id="entity-user",
        entity_type="person",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        metadata={"property_time": "2026-05-01"},
    )
    existing_two = MemoryView(
        memory_id="mem-old-2",
        project_id="proj-1",
        content="old preference",
        mem_type="fact",
        status="active",
        property_name="preference",
        entity_id="entity-user",
        entity_type="person",
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
        metadata={"property_time": "2026-05-02"},
    )
    reader = FakeReader(
        entity_candidates=[
            EntityView(
                entity_id="entity-user",
                project_id="proj-1",
                entity_name="User",
                entity_type="person",
                description="Existing user",
            )
        ],
        property_hits=[existing_one, existing_two],
    )
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=PropertyMergeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
        use_property_merge=True,
    )

    await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant even more now."}],
        ),
        make_context(),
    )

    assert writer.deleted == ["mem-old-1"]
    assert len(writer.updated) == 1
    update_command = writer.updated[0]
    assert update_command.memory_id == "mem-old-2"
    assert update_command.content == "merged preference"
    assert update_command.metadata_patch["property_merge_action"] == "update"
    assert update_command.metadata_patch["merged_from_memory_ids"] == ["mem-old-2"]
    assert "merged_from_new_property" not in update_command.metadata_patch
    assert update_command.payload_patch["property_name"] == "preference"
    _, plan, _ = writer.calls[0]
    preference_memories = [memory for memory in plan.memories if memory.property_name == "preference"]
    assert preference_memories == []


@pytest.mark.asyncio
async def test_schema_add_pipeline_higher_order_runs_for_updated_entity(monkeypatch):
    get_config().algo_config.add.schema.version = "v1"
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    first_order = MemoryView(
        memory_id="mem-first",
        project_id="proj-1",
        content="User likes Qdrant for vector search.",
        mem_type="fact",
        status="active",
        property_name="preference",
        entity_id="entity-user",
        entity_type="person",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        metadata={"property_time": "2026-05-01"},
    )
    current_higher = MemoryView(
        memory_id="mem-ho-old",
        project_id="proj-1",
        content="User likes databases.",
        mem_type="fact",
        status="active",
        property_name="preference_summary",
        entity_id="entity-user",
        entity_type="person",
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
        metadata={"property_time": "2026-05-02", "higher_order": True},
    )
    reader = FakeReader(
        entity_candidates=[
            EntityView(
                entity_id="entity-user",
                project_id="proj-1",
                entity_name="User",
                entity_type="person",
                description="Existing user",
            )
        ],
        property_hits=[first_order],
        list_memories=[first_order, current_higher],
    )
    manager = EntityManager("config/presets/entity_modeling_locomo.json")
    user_entity = manager.get("person")
    assert isinstance(user_entity, EntityType)
    user_entity.dynamic_property["preference_summary"] = {
        "type": "string",
        "order": 2,
        "desc": "Higher order user preference summary.",
    }
    # Higher-order generation resolves the project entity manager via the global accessor in
    # _schema_higher_order (not the injected manager), so patch it to return the test manager
    # that has preference_summary registered.
    monkeypatch.setattr(_schema_higher_order, "get_entity_manager", lambda *args, **kwargs: manager)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=HigherOrderLLM(),
        embed_client=FakeEmbed(),
        entity_manager=manager,
        higher_order_enabled=True,
    )

    await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "Qdrant is still my favorite vector database."}],
        ),
        make_context(),
    )

    assert "mem-ho-old" not in writer.deleted
    higher_update = next(command for command in writer.updated if command.memory_id == "mem-ho-old")
    assert higher_update.metadata_patch["higher_order"] is True
    assert higher_update.reason == "schema_add_higher_order_update"
    _, plan, _ = writer.calls[0]
    higher_memories = [memory for memory in plan.memories if memory.property_name == "preference_summary"]
    assert higher_memories == []


@pytest.mark.asyncio
async def test_schema_add_pipeline_marks_episode_failed_when_generation_fails():
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    reader = MemoryDbReader(clients=clients)
    pipeline = SchemaAddPipeline(
        db_reader=reader,
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FailingGenerationLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    result = await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "This message should fail."}],
        ),
        make_context(),
    )

    stored_record = next(iter(qdrant.schema_buffer_records.values()))
    assert result.status == "ok"
    assert writer.calls == []
    assert stored_record["status"] == "error"
    assert stored_record["buffer_status"] == "failed"
    assert stored_record["error"] == "entity generation failed"


@pytest.mark.asyncio
async def test_generate_episode_memory_cancels_stranded_tasks_and_unwraps_group_error(monkeypatch):
    """P1 fix: when a parallel schema-selection task fails, the still-running
    objectify/entity tasks are cancelled (not orphaned) and the original
    exception propagates unwrapped instead of an ExceptionGroup, so the outer
    episode-retry loop keeps its semantics.

    Bypasses SchemaAddPipeline.__init__ to test ``_generate_episode_memory``'s
    TaskGroup behavior in isolation; the test constructs its own ``rt`` with a
    ``TrackingExtractor`` and passes it directly, so full pipeline init is not needed.
    """
    import asyncio

    import mindmemos.pipelines.add.schema.schema_add as schema_add_mod
    from mindmemos.pipelines.add.schema.schema_add import SchemaAddPipeline

    started: set[str] = set()
    finished: set[str] = set()
    both_started = asyncio.Event()

    def _mark_started(name: str) -> None:
        started.add(name)
        if {"objectify", "entity"} <= started:
            both_started.set()

    class TrackingExtractor:
        def schema_for_generation(self, *, entity_manager):
            return entity_manager

        async def select_schema(self, conversation_text, schema, *, prompt_set):
            # Wait until the sibling tasks have actually started their slow
            # LLM calls before failing, so cancellation (not scheduling order)
            # is what prevents them from finishing.
            await both_started.wait()
            raise RuntimeError("schema selection failed")

        async def objectify_conversation(self, conversation_text, dialogue_timestamp, *, prompt_set):
            _mark_started("objectify")
            try:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                raise
            finished.add("objectify")
            return "objectified"

        async def generate_episode_entity(self, conversation_text, dialogue_timestamp, max_fields, *, prompt_set):
            _mark_started("entity")
            try:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                raise
            finished.add("entity")
            return {"title": "t", "content": "c", "search_fields": []}

    class TrackingPlanner:
        async def plan_episode_edges(self, *, episode_name, episode_description, context, prompt_set):
            await asyncio.sleep(0.3)
            return []

        async def recall_reference_entities(self, *, conversation_text, context):
            await asyncio.sleep(0.3)
            return []

    pipeline = SchemaAddPipeline.__new__(SchemaAddPipeline)
    rt = SimpleNamespace(
        version="v2",
        extractor=TrackingExtractor(),
        planner=TrackingPlanner(),
        search_fields_max=10,
        project_em=object(),
    )

    monkeypatch.setattr(
        schema_add_mod,
        "add_record_ops",
        SimpleNamespace(
            to_conversation_text=lambda records: "hello world",
            context=lambda records, ctx: ctx,
            records_datetime=lambda records: "2026-05-28 10:00:00",
            records_added_datetime=lambda records: datetime(2026, 5, 28, tzinfo=UTC),
            dialogue_timestamp=lambda event_at: "2026-05-28 10:00:00",
        ),
    )
    monkeypatch.setattr(schema_add_mod, "detect_prompt_language", lambda text, fallback="en": "en")
    monkeypatch.setattr(schema_add_mod, "get_add_prompts", lambda lang: SimpleNamespace())
    monkeypatch.setattr(schema_add_mod, "get_entity_manager", lambda *, project_id: object())

    with pytest.raises(RuntimeError, match="schema selection failed"):
        await pipeline._generate_episode_memory(
            [SimpleNamespace(payload={})],
            context=make_context(),
            consistency="fast",
            rt=rt,
        )

    # Give any stranded (non-cancelled) tasks time to finish so the
    # assertion reliably distinguishes "cancelled" from "orphaned".
    await asyncio.sleep(0.5)
    assert {"objectify", "entity"} <= started
    assert finished == set()


class BoundaryPromptCaptureLLM(FakeLLM):
    """FakeLLM that records the prompt sent to the episode-boundary task."""

    def __init__(self):
        self.boundary_prompts: list[str] = []

    async def chat(self, task, messages, format_parser=None, **kwargs):
        if task == "memory.add.episode_boundary":
            self.boundary_prompts.append(str(messages[0]["content"]))
        return await super().chat(task, messages, format_parser, **kwargs)


async def _add_one_turn_and_capture_boundary_prompt(llm: BoundaryPromptCaptureLLM) -> str:
    writer = FakeWriter()
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    pipeline = SchemaAddPipeline(
        db_reader=MemoryDbReader(clients=clients),
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=llm,
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant."}],
        ),
        make_context(),
    )

    assert llm.boundary_prompts
    return llm.boundary_prompts[0]


@pytest.mark.asyncio
async def test_schema_add_pipeline_segments_with_version_matched_boundary_prompt():
    """v1 chunking must use the develop boundary prompt (with the reasoning field),
    v2 the token-saving variant without it — the shared chunking path picks the
    prompt set by version instead of always using the latest one."""
    schema_cfg = get_config().algo_config.add.schema
    schema_cfg.version = "v1"
    schema_cfg.chunker.split_mode = "llm"
    v1_prompt = await _add_one_turn_and_capture_boundary_prompt(BoundaryPromptCaptureLLM())
    assert '"reasoning"' in v1_prompt

    schema_cfg.version = "v2"
    v2_prompt = await _add_one_turn_and_capture_boundary_prompt(BoundaryPromptCaptureLLM())
    assert '"reasoning"' not in v2_prompt


async def _add_one_turn_and_get_plan(*, writer: FakeWriter) -> tuple:
    qdrant = FakeQdrant()
    clients = SimpleNamespace(qdrant=qdrant, neo4j=SimpleNamespace())
    pipeline = SchemaAddPipeline(
        db_reader=MemoryDbReader(clients=clients),
        db_writer=writer,
        add_buffer=AddRecordBuffer(clients=clients),
        llm_client=FakeLLM(),
        embed_client=FakeEmbed(),
        entity_manager=EntityManager("config/presets/entity_modeling_locomo.json"),
    )

    await pipeline.add_sync(
        AddPipelineInput(
            mode="sync",
            timestamp=1770000000000,
            force_generation=True,
            messages=[{"role": "user", "content": "I like Qdrant."}],
        ),
        make_context(),
    )

    assert writer.calls
    _, plan, _ = writer.calls[0]
    return plan


@pytest.mark.asyncio
async def test_schema_add_writes_version_matched_labels():
    """v1 must label its output schema_add_v1 (develop parity, keeps eval buckets
    comparable); v2 labels its output schema_add."""
    schema_cfg = get_config().algo_config.add.schema
    schema_cfg.version = "v1"
    plan = await _add_one_turn_and_get_plan(writer=FakeWriter())
    person_entities = [entity for entity in plan.entities if entity.entity_type == "person"]
    assert person_entities and all(
        entity.metadata.get("add_algorithm") == "schema_add_v1" for entity in person_entities
    )
    assert plan.memories and all(memory.mem_extract_version == "schema_add_v1" for memory in plan.memories)

    schema_cfg.version = "v2"
    plan = await _add_one_turn_and_get_plan(writer=FakeWriter())
    person_entities = [entity for entity in plan.entities if entity.entity_type == "person"]
    assert person_entities and all(entity.metadata.get("add_algorithm") == "schema_add" for entity in person_entities)
    assert plan.memories and all(memory.mem_extract_version == "schema_add" for memory in plan.memories)
