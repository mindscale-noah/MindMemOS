"""Run-scoped OpenTelemetry persistence and Agent capture contracts."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from mindmemos_skill.agents.buitin_agents.openai_chat.agent import OpenAIChatAgent
from mindmemos_skill.agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from mindmemos_skill.agents.typing import AgentResult
from mindmemos_skill.api.typing import AlgoSkillEvolveResult, TrainingRequest
from mindmemos_skill.common.concurrency import map_with_context
from mindmemos_skill.config.config import ServiceConfig
from mindmemos_skill.envs import Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
from mindmemos_skill.logging import get_logger, log_function
from mindmemos_skill.persistence.observability import AlgorithmAuditPersistence, get_log_context
from mindmemos_skill.persistence.observability.constants import OBSERVABILITY_LOGS_TABLE, OBSERVABILITY_SPANS_TABLE
from mindmemos_skill.pipelines import Pipeline, TrainingPipeline
from mindmemos_skill.pipelines.builtin_pipelines.training.training import training_run_id
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.sdk.trace import SpanProcessor
from pydantic import BaseModel

from mindmemos_skill import create_service

from .service_config import deployment_config


class FakeTransport:
    """Return a fixed response without provider or network access."""

    def __init__(self, config):
        self.config = config

    def run_messages(self, **kwargs):
        """Return one response and usage for the supplied messages."""
        return AgentResult(output="saved", usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})


@pytest.fixture(params=["info", "debug"])
def audit(request, tmp_path):
    """Create isolated SQLite observability storage for each capture level."""
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite+pysqlite:///{tmp_path / 'audit.db'}"))
    store = AlgorithmAuditPersistence(database, capture_level=request.param)
    store.initialize_storage()
    yield store
    store.close()
    database.close()


def test_agent_spans_are_tree_children_and_debug_is_separate(audit):
    """Preserve the call tree, usage, project scope, and optional debug payload."""
    config = OpenAIChatConfig(model="offline")
    agent = OpenAIChatAgent(config, model=FakeTransport(config))
    with audit.trace_run(project_id="project", run_id="run", algorithm="forest", base_version_id="v0") as session:
        result = agent.run(messages=[{"role": "user", "content": "secret prompt"}])
        session.published_version_id = "v1"
    assert result.output == "saved"
    run = audit.get_run("project", "run")
    assert run.status == "completed" and run.audit_status == "complete"
    assert run.published_version_id == "v1"
    assert audit.get_run("other", "run") is None
    root, call = audit.spans("project", "run")
    assert root.name == "algorithm.train" and root.parent_span_id is None
    assert call.parent_span_id == root.span_id and call.trace_id == root.trace_id
    assert root.resource["service.name"] == "mindmemos-skill"
    assert call.attributes["gen_ai.usage.input_tokens"] == 3
    assert call.attributes["algorithm.capture_level"] == audit.capture_level
    assert "secret prompt" not in str([root.attributes, call.attributes])
    assert audit.spans("other", "run") == []
    logs = audit.logs("project", "run")
    assert [item.name for item in logs] == [
        "algorithm.run.started",
        "agent.call",
        *(["agent.io"] if audit.capture_level == "debug" else []),
        "algorithm.run.completed",
    ]
    call_log = next(item for item in logs if item.name == "agent.call")
    assert call_log.span_id == call.span_id
    assert call_log.body["usage"]["input_tokens"] == 3
    if audit.capture_level == "debug":
        debug_log = next(item for item in logs if item.name == "agent.io")
        assert debug_log.span_id == call.span_id
        assert "secret prompt" in debug_log.body["input"]
    else:
        assert "secret prompt" not in str(logs)


def test_python_logging_is_forwarded_to_current_otel_trace(audit):
    """Ordinary package logs reach the same run and span without direct OTel calls."""
    logger = get_logger("algorithm.stage")
    logger.info("outside run")
    with audit.trace_run(project_id="project", run_id="logging", algorithm="forest", base_version_id="v0") as session:
        with session.span("algorithm.stage"):
            logger.info("stage ready")
            logger.warning("stage warning")
            logger.debug("stage detail")
    stage = next(item for item in audit.spans("project", "logging") if item.name == "algorithm.stage")
    logs = [item for item in audit.logs("project", "logging") if item.name == "python.log"]
    assert [item.body["message"] for item in logs] == [
        "stage ready",
        "stage warning",
        *(["stage detail"] if audit.capture_level == "debug" else []),
    ]
    assert {item.span_id for item in logs} == {stage.span_id}


def test_log_function_records_sync_inputs_output_and_duration(audit):
    """A decorated method records outcome and bounded debug values in one run."""

    class Worker:
        def __repr__(self):
            return "receiver secret"

        @log_function
        def calculate(self, value: int, token: str) -> dict:
            """Return one result for the given inputs."""
            return {"value": value + 1, "password": "output secret"}

    with audit.trace_run(project_id="project", run_id="decorated-sync", algorithm="forest", base_version_id="v0"):
        assert Worker().calculate(2, token="input secret")["value"] == 3
    logs = audit.logs("project", "decorated-sync")
    call = next(item for item in logs if item.name == "function.call")
    assert call.body["ok"] is True
    assert call.body["duration_ms"] >= 0
    assert call.body["function"].endswith("Worker.calculate")
    debug = [item for item in logs if item.level == "debug"]
    assert [item.name for item in debug] == (
        ["function.input", "function.output"] if audit.capture_level == "debug" else []
    )
    if debug:
        assert "[REDACTED]" in debug[0].body["token"]
        assert "receiver secret" not in str(debug)
        assert "output secret" not in str(debug)
    else:
        assert "input secret" not in str(logs)


@pytest.mark.asyncio
async def test_log_function_records_async_success(audit):
    """A decorated coroutine logs its result after awaited work completes."""

    @log_function
    async def double(value: int) -> int:
        """Return the doubled input."""
        return value * 2

    with audit.trace_run(project_id="project", run_id="decorated-async-ok", algorithm="forest", base_version_id="v0"):
        assert await double(4) == 8
    logs = audit.logs("project", "decorated-async-ok")
    call = next(item for item in logs if item.name == "function.call")
    assert call.body["ok"] is True and call.body["duration_ms"] >= 0
    output = [item for item in logs if item.name == "function.output"]
    assert (len(output) == 1) is (audit.capture_level == "debug")
    if output:
        assert output[0].body["result"] == "8"


@pytest.mark.asyncio
async def test_log_function_records_async_failure_and_traceback(audit):
    """A decorated coroutine records its failure and re-raises the original error."""

    @log_function
    async def fail(value: int) -> None:
        """Raise after one asynchronous invocation."""
        raise RuntimeError(f"failed on {value}")

    with pytest.raises(RuntimeError, match="failed on 7"):
        with audit.trace_run(project_id="project", run_id="decorated-async", algorithm="forest", base_version_id="v0"):
            await fail(7)
    call = next(item for item in audit.logs("project", "decorated-async") if item.name == "function.call")
    assert call.body["ok"] is False
    assert call.body["duration_ms"] >= 0
    assert call.body["error_type"] == "RuntimeError"
    assert "failed on 7" in call.body["traceback"]
    assert "fail" in call.body["traceback"]


def test_log_function_rejects_deferred_generator_execution():
    """A generator cannot be timed accurately when its object is merely created."""

    with pytest.raises(TypeError, match="generator"):

        @log_function
        def stream():
            """Yield one value after invocation."""
            yield 1


def test_parallel_service_handlers_only_capture_their_own_run(audit, tmp_path):
    """Multiple service handlers do not duplicate or cross-write Python logs."""
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite+pysqlite:///{tmp_path / 'other.db'}"))
    other = AlgorithmAuditPersistence(database)
    other.initialize_storage()
    try:
        with audit.trace_run(project_id="project", run_id="first", algorithm="forest", base_version_id="v0"):
            get_logger("algorithm.stage").info("first only")
        with other.trace_run(project_id="project", run_id="second", algorithm="forest", base_version_id="v0"):
            get_logger("algorithm.stage").info("second only")
        assert [item.body["message"] for item in audit.logs("project", "first") if item.name == "python.log"] == [
            "first only"
        ]
        assert [item.body["message"] for item in other.logs("project", "second") if item.name == "python.log"] == [
            "second only"
        ]
        assert audit.logs("project", "second") == []
        assert other.logs("project", "first") == []
    finally:
        other.close()
        database.close()


def test_failure_and_recovery_link(audit):
    """A recovery attempt keeps run identity and links to the previous trace."""
    with pytest.raises(RuntimeError, match="offline failure"):
        with audit.trace_run(project_id="project", run_id="recover", algorithm="forest", base_version_id="v0"):
            raise RuntimeError("offline failure")
    first = audit.get_run("project", "recover")
    assert first.status == "failed" and first.error_type == "RuntimeError"
    with audit.trace_run(project_id="project", run_id="recover", algorithm="forest", base_version_id="v0"):
        pass
    second = audit.get_run("project", "recover")
    assert second.status == "completed" and second.trace_id != first.trace_id
    roots = [span for span in audit.spans("project", "recover") if span.name == "algorithm.train"]
    assert len(roots) == 2
    assert roots[1].links[0]["trace_id"] == first.trace_id
    with pytest.raises(ValueError, match="identity conflicts"):
        with audit.trace_run(project_id="project", run_id="recover", algorithm="other", base_version_id="v0"):
            pass


def test_failed_replay_keeps_previous_publication(audit):
    """A later failed attempt cannot erase the already published version reference."""
    with audit.trace_run(project_id="project", run_id="published", algorithm="forest", base_version_id="v0") as session:
        session.published_version_id = "v1"
    with pytest.raises(RuntimeError, match="replay failed"):
        with audit.trace_run(project_id="project", run_id="published", algorithm="forest", base_version_id="v0"):
            raise RuntimeError("replay failed")
    run = audit.get_run("project", "published")
    assert run.status == "failed" and run.published_version_id == "v1"


def test_span_write_failure_marks_audit_incomplete(audit, monkeypatch):
    """A dropped span is visible in run state instead of silently claiming completeness."""
    original = audit.database.upsert

    def fail_span(table, records, **kwargs):
        if table == OBSERVABILITY_SPANS_TABLE:
            raise OSError("span storage unavailable")
        return original(table, records, **kwargs)

    monkeypatch.setattr(audit.database, "upsert", fail_span)
    with audit.trace_run(project_id="project", run_id="drop", algorithm="forest", base_version_id="v0"):
        pass
    run = audit.get_run("project", "drop")
    assert run.status == "completed" and run.audit_status == "incomplete"
    assert audit.spans("project", "drop") == []


def test_log_write_failure_marks_audit_incomplete(audit, monkeypatch):
    """A failed OTel log sink leaves the run's audit status incomplete."""
    original = audit.database.insert

    def fail_log(table, records, **kwargs):
        if table == OBSERVABILITY_LOGS_TABLE:
            raise OSError("log storage unavailable")
        return original(table, records, **kwargs)

    monkeypatch.setattr(audit.database, "insert", fail_log)
    with audit.trace_run(project_id="project", run_id="drop-log", algorithm="forest", base_version_id="v0"):
        pass
    run = audit.get_run("project", "drop-log")
    assert run.status == "completed" and run.audit_status == "incomplete"
    assert audit.logs("project", "drop-log") == []


def test_worker_threads_keep_the_run_trace(audit):
    """Independent thread copies retain one run without sharing mutable Context objects."""

    def work(index):
        session = get_log_context()
        assert session is not None
        with session.span("algorithm.worker", {"algorithm.task_index": index}):
            return index

    with audit.trace_run(project_id="project", run_id="threads", algorithm="forest", base_version_id="v0"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert map_with_context(executor, work, range(4)) == [0, 1, 2, 3]
    spans = audit.spans("project", "threads")
    root = next(span for span in spans if span.parent_span_id is None)
    workers = [span for span in spans if span.name == "algorithm.worker"]
    assert len(workers) == 4
    assert {span.parent_span_id for span in workers} == {root.span_id}


def test_invalid_observability_configuration_is_rejected():
    """The service accepts one supported backend and explicit capture levels."""
    with pytest.raises(ValueError):
        ServiceConfig.model_validate(
            {"home": ".", "observability": {"backend": {"class_name": "sqlite"}, "capture_level": "verbose"}}
        )
    with pytest.raises(ValueError, match="requires backend or otlp"):
        ServiceConfig.model_validate({"home": ".", "observability": {}})
    with pytest.raises(ValueError):
        ServiceConfig.model_validate({"home": ".", "observability": {"otlp": {"endpoint": "ftp://localhost:4318"}}})


@pytest.mark.parametrize("with_database", [False, True])
@pytest.mark.parametrize("operation", ["train", "evolve"])
def test_otlp_exports_logs_and_spans_with_optional_sql_audit(tmp_path, with_database, operation):
    """Send real OTLP/HTTP payloads while preserving optional local audit writes."""
    received = []

    class Receiver(BaseHTTPRequestHandler):
        """Capture OTLP requests on a local HTTP endpoint."""

        def do_POST(self):
            """Accept one protobuf export request."""
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            """Keep the test receiver silent."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    database = (
        SQLiteDatabase(SQLiteConfig(url=f"sqlite+pysqlite:///{tmp_path / 'audit.db'}")) if with_database else None
    )
    try:
        store = AlgorithmAuditPersistence(
            database,
            service_name="mindmemos-skill-test",
            otlp_endpoint=f"http://127.0.0.1:{server.server_port}",
        )
        try:
            store.initialize_storage()
            with store.trace_run(
                project_id="project", run_id="otlp", algorithm="forest", base_version_id="v0", operation=operation
            ):
                get_logger("otlp_probe").info("probe ready")
            if database is not None:
                assert store.get_run("project", "otlp").status == "completed"
                assert any(log.body.get("message") == "probe ready" for log in store.logs("project", "otlp"))
        finally:
            store.close()
    finally:
        if database is not None:
            database.close()
        server.shutdown()
        server.server_close()
        thread.join()

    paths = {path for path, _ in received}
    assert paths == {"/v1/logs", "/v1/traces"}
    log_request = ExportLogsServiceRequest.FromString(next(body for path, body in received if path == "/v1/logs"))
    trace_request = ExportTraceServiceRequest.FromString(next(body for path, body in received if path == "/v1/traces"))
    logs = [
        record
        for resource in log_request.resource_logs
        for scope in resource.scope_logs
        for record in scope.log_records
    ]
    spans = [
        span for resource in trace_request.resource_spans for scope in resource.scope_spans for span in scope.spans
    ]
    assert any(record.body.string_value == "probe ready" or "probe ready" in str(record.body) for record in logs)
    assert any(span.name == f"algorithm.{operation}" for span in spans)
    assert any(
        attr.key == "service.name" and attr.value.string_value == "mindmemos-skill-test"
        for attr in log_request.resource_logs[0].resource.attributes
    )


@pytest.mark.asyncio
async def test_service_accepts_otlp_without_audit_database():
    """Build and close an OTLP-only service without creating SQL audit storage."""
    service = await create_service(
        ServiceConfig.model_validate({"home": ".", "observability": {"otlp": {"endpoint": "http://127.0.0.1:4318"}}})
    )
    try:
        assert service._observability.database is None
        assert "observability_database" not in service.backends
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_emits_otel_spans_and_logs_without_database(tmp_path, monkeypatch):
    """Disabling the database sink does not disable training instrumentation."""
    monkeypatch.setattr(Pipeline, "_registry", dict(Pipeline._registry))

    class ProbeConfig(BaseModel):
        """Require no pipeline-specific options."""

    @Pipeline.register("otel_probe")
    class ProbeTraining(TrainingPipeline[ProbeConfig]):
        config_type = ProbeConfig

        async def train(self, request, *, env, adapter=None, reporter=None):
            """Invoke an Agent inside the always-active training context."""
            config = OpenAIChatConfig(model="offline")
            OpenAIChatAgent(config, model=FakeTransport(config)).run(messages=[{"role": "user", "content": "task"}])
            return AlgoSkillEvolveResult(
                skill_id="skill",
                base_version_id=request.init_skill_version_id,
                version_id="v1",
                content="Output",
                resources={},
                content_hash="hash",
                skill_hash="hash",
                run_id=training_run_id(request.project_id, request.work_dir),
            )

    span_names = []
    log_names = []

    class CaptureSpans(SpanProcessor):
        """Observe completed spans from the local OTel provider."""

        def on_start(self, span, parent_context=None):
            """Ignore span starts."""

        def on_end(self, span):
            """Collect the completed operation name."""
            span_names.append(span.name)

        def shutdown(self):
            """Release no resources."""

        def force_flush(self, timeout_millis=30000):
            """Report synchronous completion."""
            return True

    class CaptureLogs(LogRecordProcessor):
        """Observe OTel logs before any optional database export."""

        def on_emit(self, record):
            """Collect the emitted event name."""
            log_names.append(record.log_record.event_name)

        def shutdown(self):
            """Release no resources."""

        def force_flush(self, timeout_millis=30000):
            """Report synchronous completion."""
            return True

    service = await create_service(
        deployment_config(
            {
                "home": tmp_path / "home",
                "training_pipelines": {"probe": {"class_name": "otel_probe"}},
            },
            "project",
        )
    )
    try:
        audit = service._observability
        assert audit.database is None
        audit.telemetry.provider.add_span_processor(CaptureSpans())
        audit.telemetry.logger_provider.add_log_record_processor(CaptureLogs())
        request = TrainingRequest(
            project_id="project",
            algo="probe",
            env={"class_name": "searchqa", "config": {"agent": {"model": "offline"}}},
            init_skill_version_id="v0",
            work_dir=str(tmp_path / "run"),
            tasks=[DataSample(task=Task(id="one", instruction="Save"), reward_data=RewardData())],
            split={"train": ["one"]},
        )
        await service.run_training(request)
        assert span_names == ["algorithm.agent.call", "algorithm.train"]
        assert log_names == ["algorithm.run.started", "agent.call", "algorithm.run.completed"]
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_database", [False, True])
async def test_service_uses_one_global_observability_backend(tmp_path, monkeypatch, shared_database):
    """Service training activates the shared trace and persists its publication identity."""
    monkeypatch.setattr(Pipeline, "_registry", dict(Pipeline._registry))

    class ProbeConfig(BaseModel):
        """Require no pipeline-specific options."""

    @Pipeline.register("audit_probe")
    class ProbeTraining(TrainingPipeline[ProbeConfig]):
        config_type = ProbeConfig

        async def train(self, request, *, env, adapter=None, reporter=None):
            """Return a completed version while adding one child operation."""
            session = get_log_context()
            assert session is not None
            with session.span("algorithm.stage.probe"):
                pass
            return AlgoSkillEvolveResult(
                skill_id="skill",
                base_version_id=request.init_skill_version_id,
                version_id="v1",
                content="Output",
                resources={},
                content_hash="hash",
                skill_hash="hash",
                run_id=training_run_id(request.project_id, request.work_dir),
            )

    audit_path = tmp_path / ("skills.db" if shared_database else "audit.db")
    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite+pysqlite:///{tmp_path / 'skills.db'}"}},
            },
            "observability": {
                "backend": {"class_name": "sqlite", "config": {"url": f"sqlite+pysqlite:///{audit_path}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "training_pipelines": {"probe": {"class_name": "audit_probe"}},
        },
        "project",
    )
    service = await create_service(config)
    try:
        await service.initialize_storage()
        request = TrainingRequest(
            project_id="project",
            algo="probe",
            env={"class_name": "searchqa", "config": {"agent": {"model": "offline"}}},
            init_skill_version_id="v0",
            work_dir=str(tmp_path / "run"),
            tasks=[DataSample(task=Task(id="one", instruction="Save"), reward_data=RewardData())],
            split={"train": ["one"]},
        )
        result = await service.run_training(request)
        run = await service.algorithm_run("project", result.run_id)
        assert run.status == "completed" and run.published_version_id == "v1"
        assert [item.name for item in await service.algorithm_spans("project", result.run_id)] == [
            "algorithm.train",
            "algorithm.stage.probe",
        ]
        with pytest.raises(ValueError, match="default project"):
            await service.algorithm_spans("other", result.run_id)
        assert service.backends["observability"].database is service.backends["observability_database"]
    finally:
        await service.close()


def test_latest_evolution_is_scoped_and_excludes_training(audit):
    """Project, exact input version, and configured evolution names bound the timestamp."""
    from datetime import UTC, datetime, timedelta

    from mindmemos_skill.persistence.observability.constants import OBSERVABILITY_RUNS_TABLE
    from mindmemos_skill.persistence.observability.typing import AuditRun

    start = datetime(2026, 10, 9, tzinfo=UTC)
    for index, (project, version, algorithm) in enumerate(
        [
            ("project", "v1", "simple"),
            ("project", "v1", "simple_evolve"),
            ("project", "v1", "training"),
            ("other", "v1", "simple"),
            ("project", "v2", "simple"),
        ]
    ):
        audit.database.insert(
            OBSERVABILITY_RUNS_TABLE,
            [
                AuditRun(
                    project_id=project,
                    run_id=f"run-{index}",
                    algorithm=algorithm,
                    base_version_id=version,
                    status="completed",
                    audit_status="complete",
                    capture_level="info",
                    trace_id="0" * 32,
                    started_at=start + timedelta(hours=index),
                ).model_dump()
            ],
        )
    assert audit.latest_evolution("project", "v1", ["simple", "simple_evolve"]) == start + timedelta(hours=1)
    assert audit.latest_evolution("project", "missing", ["simple"]) is None
    assert audit.latest_evolution("project", "v1", []) is None
