"""OTel timing and parentage for graph build stages and concurrent enrichment."""

import asyncio

import pytest
from mindmemos_skill.algos.manage.skill_manage.algorithm import SkillManageAlgorithm
from mindmemos_skill.algos.manage.skill_manage.enrichment import SkillEnricher
from mindmemos_skill.api.typing import OrganizeRequest
from mindmemos_skill.logging.tracing import bind_tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .test_scene_graph_pipeline import graph_service as graph_service
from .test_scene_graph_pipeline import register
from .test_scene_graph_pipeline import scene_record as scene_record


@pytest.mark.parametrize("outcome", ["new", "reuse", "failure"])
async def test_build_stage_spans_capture_timing_hierarchy_and_failures(
    graph_service, scene_record, monkeypatch, outcome
):
    """Export each executed stage with its duration and correct asynchronous parent."""
    service, _, _ = graph_service
    pipeline = service._manage_pipeline
    items = [await register(service, f"traced-member-{index}") for index in range(3)]
    config = pipeline.config.algorithm.model_copy(update={"enrich_skill": True, "llm_linking": True})

    async def enrich(self, node, document):
        await asyncio.sleep(0)
        return node

    async def chat(self, system, payload, allowed):
        return self._parse_relations('{"relations": []}', allowed=allowed)

    monkeypatch.setattr(SkillEnricher, "enrich", enrich)
    monkeypatch.setattr(SkillManageAlgorithm, "_request_relations", chat)
    request = OrganizeRequest(project_id="project", skill_refs=[item.skill.skill_id for item in items[:2]])
    versions = await pipeline._resolve_skill_versions(request.skill_refs)
    sources = pipeline._skills_from_versions(versions)
    store = pipeline._graph_store(request, "skg-traced-build", versions, None)
    algorithm = pipeline._algorithm(store, {key: scene_record for key in sources}, config)
    prior = None
    if outcome == "reuse":
        first = await algorithm._build_joint(sources, "initial", None)
        request = request.model_copy(update={"skill_refs": [item.skill.skill_id for item in items]})
        versions = await pipeline._resolve_skill_versions(request.skill_refs)
        sources = pipeline._skills_from_versions(versions)
        store = pipeline._graph_store(request, "skg-traced-build", versions, None)
        algorithm = pipeline._algorithm(store, {items[2].skill.skill_id: scene_record}, config)
        algorithm.reuse_snapshot(first.snapshot_id, nodes=first.nodes, edges=first.edges, metadata=first.metadata)
        prior = algorithm._state
    elif outcome == "failure":

        async def fail_embedding(self, texts):
            raise RuntimeError("Controlled embedding failure")

        monkeypatch.setattr(SkillManageAlgorithm, "_embed", fail_embedding)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test.graph_build")
    try:
        with bind_tracer(tracer), tracer.start_as_current_span("test.graph_build") as root:
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="Controlled embedding failure"):
                    await algorithm._build_joint(sources, "traced-request", prior)
            else:
                await algorithm._build_joint(sources, "traced-request", prior)
            with tracer.start_as_current_span("test.after_build"):
                pass
        spans = exporter.get_finished_spans()
        by_name = {span.name: span for span in spans}
        build = by_name["skill_manage.build"]
        assert build.parent.span_id == root.get_span_context().span_id
        assert by_name["test.after_build"].parent.span_id == root.get_span_context().span_id
        expected_stages = ["representation", "enrichment", "embedding"]
        if outcome != "failure":
            expected_stages += ["linking", "finalize"]
        for stage in expected_stages:
            assert by_name[f"skill_manage.build.{stage}"].parent.span_id == build.context.span_id
        for span in spans:
            if not span.name.startswith("skill_manage.build"):
                continue
            assert span.context.trace_id == build.context.trace_id
            assert span.attributes["duration_ms"] >= 0
            assert span.end_time >= span.start_time
            assert "arguments" not in span.attributes and "result" not in span.attributes
            if span.name != "skill_manage.build":
                parent_name = span.name.rsplit(".", 1)[0]
                assert span.parent.span_id == by_name[parent_name].context.span_id
        enriched = [span for span in spans if span.name == "skill_manage.build.enrichment.member"]
        assert len(enriched) == (1 if outcome == "reuse" else 2)
        if outcome == "failure":
            assert build.status.status_code.name == "ERROR"
            embedding = by_name["skill_manage.build.embedding"]
            assert embedding.status.status_code.name == "ERROR"
            assert any(event.name == "exception" for event in embedding.events)
            assert "skill_manage.build.linking" not in by_name
            assert "skill_manage.build.finalize" not in by_name
        else:
            assert all(span.status.status_code.name == "OK" for span in spans if span.name.startswith("skill_manage"))
            assert "skill_manage.build.linking.query_embedding" in by_name
            assert "skill_manage.build.linking.inference" in by_name
            assert "skill_manage.build.linking.edge_records" in by_name
            assert "skill_manage.build.representation.build_member" in by_name
            reused = [span for span in spans if span.name == "skill_manage.build.representation.reuse_member"]
            assert len(reused) == (2 if outcome == "reuse" else 0)
    finally:
        provider.shutdown()


@pytest.mark.parametrize("failure", ["http", "no_choices", "private"])
async def test_scene_transport_diagnostics_exclude_provider_payloads(failure, monkeypatch, caplog):
    """Retain actionable error categories in traces without retaining source or credentials."""
    import httpx
    from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
    from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError
    from mindmemos_skill.algos.manage.skill_manage.extraction import SceneExtractor
    from openai import BadRequestError

    private = "private-document-and-credential"
    error = {
        "http": BadRequestError(
            private,
            response=httpx.Response(400, request=httpx.Request("POST", "https://example.com")),
            body={"message": private},
        ),
        "no_choices": RuntimeError("The model returned no choices"),
        "private": RuntimeError(private),
    }[failure]

    async def fail(self, **kwargs):
        raise error

    monkeypatch.setattr(OpenAIChatAgent, "run_messages", fail)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    extractor = SceneExtractor(OpenAIChatAgent(OpenAIChatConfig(model="fake")))
    try:
        with bind_tracer(provider.get_tracer("test.scene")):
            with pytest.raises(SceneGraphError, match="SCENE_GRAPH_MODEL_FAILED"):
                await extractor.extract(private)
        spans = {span.name: span for span in exporter.get_finished_spans()}
        call = spans["skill_manage.scene.model_call"]
        extract = spans["skill_manage.scene.extract"]
        assert call.parent.span_id == extract.context.span_id
        assert call.attributes["error_type"] == type(error).__name__
        assert extract.attributes["extraction.attempts"] == 1
        assert extract.status.status_code.name == "ERROR"
        if failure == "http":
            assert call.attributes["http_status"] == 400
        elif failure == "no_choices":
            assert call.attributes["reason"] == "no_choices"
        assert private not in caplog.text
        assert private not in str([(span.attributes, span.events) for span in spans.values()])
    finally:
        provider.shutdown()


async def test_insufficient_source_is_a_valid_traced_outcome(monkeypatch, caplog):
    """Keep the extraction port exception while treating confirmed absence as a successful call."""
    from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
    from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError, SceneGraphFailure
    from mindmemos_skill.algos.manage.skill_manage.extraction import SceneExtractor

    async def insufficient(self, **kwargs):
        return AgentResult(output='{"status":"insufficient_source","record":null}')

    monkeypatch.setattr(OpenAIChatAgent, "run_messages", insufficient)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    extractor = SceneExtractor(OpenAIChatAgent(OpenAIChatConfig(model="fake")))
    try:
        with bind_tracer(provider.get_tracer("test.scene")):
            with pytest.raises(SceneGraphError) as caught:
                await extractor.extract("Public source without actionable steps")
        assert caught.value.code == SceneGraphFailure.INSUFFICIENT_SOURCE
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert len(spans) == 2
        assert all(span.status.status_code.name == "OK" for span in spans.values())
        extraction = spans["skill_manage.scene.extract"]
        assert extraction.attributes["extraction.outcome"] == "insufficient_source"
        assert extraction.attributes["extraction.error_type"] == "insufficient_source"
        assert not any(event.name == "exception" for span in spans.values() for event in span.events)
        assert not any(record.levelname == "ERROR" for record in caplog.records)
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("entry", "scene_entry_missing"),
        ("status", "scene_status_mismatch"),
        ("terminal", "scene_terminal_outcome_missing"),
    ],
)
def test_scene_validation_diagnostics_identify_contract_failure(scene_record, failure, expected, caplog):
    """Expose stable constraint codes while excluding rejected model contents."""
    import json

    from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
    from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError
    from mindmemos_skill.algos.manage.skill_manage.extraction import SceneExtractor

    record = scene_record.model_dump(mode="json")
    record["skill"]["goal"] = "private-model-response"
    payload = {"status": "ok", "record": record}
    if failure == "entry":
        record["entry_scene_id"] = "private-unknown-entry"
    elif failure == "terminal":
        record["scenes"][0]["next_scene_rules"][0]["target"] = "END_"
    else:
        payload["status"] = "insufficient_source"
    extractor = SceneExtractor(OpenAIChatAgent(OpenAIChatConfig(model="fake")))
    with pytest.raises(SceneGraphError):
        extractor._parse(json.dumps(payload))
    event = caplog.records[-1].event_body
    assert event["validation_types"] == [expected]
    assert "private-model-response" not in json.dumps(event)
    assert "private-unknown-entry" not in json.dumps(event)
