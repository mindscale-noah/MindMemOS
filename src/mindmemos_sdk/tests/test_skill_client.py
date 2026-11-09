"""Exercise every Skill route and shared sync/async request serialization."""

import json

import httpx
import pytest
from mindmemos_sdk.client import MindMemOSClient
from mindmemos_sdk.config.models import AuthConfig, MemoryServiceConfig, SDKConfig, SkillServiceConfig
from mindmemos_sdk.errors import ApiError, TransportError
from mindmemos_sdk.skills import (
    ActivateRequest,
    AddGraphMembersRequest,
    AsyncSkillClient,
    CommitRequest,
    DeleteRequest,
    DiffRequest,
    EvaluationQueryRequest,
    EvolveRequest,
    GraphResolveRequest,
    GraphSnapshotRequest,
    GraphSnapshotsRequest,
    HistoryRequest,
    ListRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    RemoveGraphMembersRequest,
    ShowRequest,
    SkillAnalyzeRequest,
    SkillClient,
    SkillResolveRequest,
    StructuredSkillInfo,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingRequest,
    TrainingSplitRequest,
    TrajectoryAnalyzeRequest,
    TrajectoryListRequest,
    TrajectoryReportRequest,
    TrajectoryShowRequest,
    TrajectoryStatisticsRequest,
    UpdateInfoRequest,
    VersionResolveRequest,
)
from mindmemos_sdk.transport import AsyncHttpTransport, HttpTransport

REQUESTS = [
    (
        SkillClient.analyze_skill,
        AsyncSkillClient.analyze_skill,
        SkillAnalyzeRequest(skill_version_id="v1", request_id="trace", force_refresh=True),
        "algos/analyze/skill",
    ),
    (
        SkillClient.analyze_trajectory,
        AsyncSkillClient.analyze_trajectory,
        TrajectoryAnalyzeRequest(trajectory_id="t1", attribute=False, verifier_score=0.0),
        "algos/analyze/trajectory",
    ),
    (
        SkillClient.query_evaluations,
        AsyncSkillClient.query_evaluations,
        EvaluationQueryRequest(skill_version_ids=["v1", "v2"], judge_types=["static_skill"], limit=10),
        "algos/evaluations/query",
    ),
    (
        SkillClient.add_graph_members,
        AsyncSkillClient.add_graph_members,
        AddGraphMembersRequest(graph_ref="g", skill_refs=["s"], force_extract_scene=True),
        "graphs/members/add",
    ),
    (
        SkillClient.remove_graph_members,
        AsyncSkillClient.remove_graph_members,
        RemoveGraphMembersRequest(graph_ref="g", skill_refs=["s"]),
        "graphs/members/remove",
    ),
    (SkillClient.preview_graph, AsyncSkillClient.preview_graph, GraphResolveRequest(graph_ref="g"), "graphs/preview"),
    (
        SkillClient.trajectory_statistics,
        AsyncSkillClient.trajectory_statistics,
        TrajectoryStatisticsRequest(skill_id="skill-1", skill_version_id="version-1"),
        "trajectories/statistics",
    ),
    (SkillClient.delete_graph, AsyncSkillClient.delete_graph, GraphResolveRequest(graph_ref="graph"), "graphs/delete"),
    (
        SkillClient.register,
        AsyncSkillClient.register,
        RegisterRequest(
            skill=StructuredSkillInfo(
                content="text",
                name="demo",
                description="demo",
                standard_format=False,
                commit_version="v1.0.0",
                commit_message="Initial",
            ),
            alias="demo",
        ),
        "register",
    ),
    (SkillClient.list, AsyncSkillClient.list, ListRequest(filters={"name": "demo*"}, limit=5), "list"),
    (SkillClient.show, AsyncSkillClient.show, ShowRequest(skill_id="skill-1"), "show"),
    (
        SkillClient.commit,
        AsyncSkillClient.commit,
        CommitRequest(
            skill_id="skill-1",
            skill=StructuredSkillInfo(content="new", commit_message="Update", commit_version="v1.0.1"),
            active=False,
        ),
        "commit",
    ),
    (SkillClient.history, AsyncSkillClient.history, HistoryRequest(skill_id="skill-1"), "history"),
    (SkillClient.diff, AsyncSkillClient.diff, DiffRequest(skill_id="skill-1", to_skill_version_id="version-2"), "diff"),
    (
        SkillClient.activate,
        AsyncSkillClient.activate,
        ActivateRequest(skill_id="skill-1", skill_version_id="version-2"),
        "activate",
    ),
    (
        SkillClient.update_info,
        AsyncSkillClient.update_info,
        UpdateInfoRequest(skill_id="skill-1", alias=None),
        "update-info",
    ),
    (SkillClient.delete, AsyncSkillClient.delete, DeleteRequest(skill_id="skill-1"), "delete"),
    (
        SkillClient.version_resolve,
        AsyncSkillClient.version_resolve,
        VersionResolveRequest(content_hash="a" * 64),
        "version-resolve",
    ),
    (
        SkillClient.skill_resolve,
        AsyncSkillClient.skill_resolve,
        SkillResolveRequest(skill_ref="demo"),
        "skill-resolve",
    ),
    (
        SkillClient.report_trajectory,
        AsyncSkillClient.report_trajectory,
        TrajectoryReportRequest(
            user_id="user-1",
            agent_id="agent-1",
            messages=[{"role": "user", "content": "Task"}],
            skill_version_id="version-1",
        ),
        "trajectories/report",
    ),
    (
        SkillClient.show_trajectory,
        AsyncSkillClient.show_trajectory,
        TrajectoryShowRequest(trajectory_id="trajectory-1"),
        "trajectories/show",
    ),
    (
        SkillClient.list_trajectories,
        AsyncSkillClient.list_trajectories,
        TrajectoryListRequest(skill_id="skill-1", limit=3),
        "trajectories/list",
    ),
    (
        SkillClient.organize,
        AsyncSkillClient.organize,
        OrganizeRequest(alias="graph", commit_message="Organize", skill_refs=[]),
        "algos/organize",
    ),
    (
        SkillClient.graph_resolve,
        AsyncSkillClient.graph_resolve,
        GraphResolveRequest(graph_ref="graph"),
        "graph-resolve",
    ),
    (
        SkillClient.graph_snapshots,
        AsyncSkillClient.graph_snapshots,
        GraphSnapshotsRequest(graph_id="graph-1"),
        "graphs/snapshots",
    ),
    (
        SkillClient.graph_snapshot,
        AsyncSkillClient.graph_snapshot,
        GraphSnapshotRequest(graph_id="graph-1", graph_version_id="version-1"),
        "graphs/snapshot",
    ),
    (
        SkillClient.recommend,
        AsyncSkillClient.recommend,
        RecommendRequest(graph_ref="graph", query="Find reports"),
        "algos/recommend",
    ),
    (
        SkillClient.evolve,
        AsyncSkillClient.evolve,
        EvolveRequest(skill_version_id="version-1", auto_accept=False),
        "algos/evolve",
    ),
    (SkillClient.show_task, AsyncSkillClient.show_task, TaskShowRequest(task_id="task-1"), "tasks/show"),
    (
        SkillClient.train,
        AsyncSkillClient.train,
        TrainingRequest(
            algo="treeskill",
            env={"class_name": "test", "config": {}},
            init_skill_version_id="version-1",
            work_dir="runs/demo",
            tasks=[{"task": {"id": "case-1", "instruction": "Help"}, "reward_data": {}}],
            split=TrainingSplitRequest(train=["case-1"]),
        ),
        "algos/train",
    ),
    (
        SkillClient.task_progress,
        AsyncSkillClient.task_progress,
        TaskProgressRequest(task_id="task-1", after=3),
        "tasks/progress",
    ),
]


@pytest.mark.parametrize(("sync_method", "async_method", "body", "path"), REQUESTS)
def test_skill_route_serialization(sync_method, async_method, body, path):
    """Every sync method posts its typed body to the matching service route."""

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == f"/v1/skills/{path}"
        expected = body.model_dump(mode="json", exclude_unset=True)
        if path == "tasks/progress":
            expected.pop("after")
            assert request.url.params["after"] == "3"
        assert json.loads(request.content) == expected
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"code": "ok", "data": {"route": path}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(base_url="http://localhost:8001", api_key=None, client=http, require_api_key=False)
    try:
        assert sync_method(SkillClient(transport), body) == {"route": path}
    finally:
        http.close()


def test_list_graphs_uses_project_scoped_sdk_route():
    """Graph listing sends no caller-selected project identity."""

    def handler(request):
        assert request.url.path == "/v1/skills/graphs/list"
        assert json.loads(request.content) == {}
        return httpx.Response(200, json={"code": "ok", "data": {"items": []}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(base_url="http://localhost:8001", api_key=None, client=http, require_api_key=False)
    try:
        assert SkillClient(transport).list_graphs() == {"items": []}
    finally:
        http.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("sync_method", "async_method", "body", "path"), REQUESTS)
async def test_async_skill_route_serialization(sync_method, async_method, body, path):
    """Async methods use the same route and body contract as sync methods."""

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == f"/v1/skills/{path}"
        expected = body.model_dump(mode="json", exclude_unset=True)
        if path == "tasks/progress":
            expected.pop("after")
            assert request.url.params["after"] == "3"
        assert json.loads(request.content) == expected
        return httpx.Response(200, json={"code": "ok", "data": {"route": path}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(base_url="http://localhost:8001", api_key=None, client=http, require_api_key=False)
    try:
        assert await async_method(AsyncSkillClient(transport), body) == {"route": path}
    finally:
        await http.aclose()


def test_root_client_uses_independent_skill_connection(tmp_path):
    """The root client routes Skill calls using v2 Skill settings."""
    config = SDKConfig(
        service_type="custom",
        memory_service=MemoryServiceConfig(base_url="https://memory.example", auth=AuthConfig(api_key="memory-key")),
        skill_service=SkillServiceConfig(base_url="https://skill.example", auth=AuthConfig(api_key="skill-key")),
    )

    def handler(request):
        assert request.url.host == "skill.example"
        assert request.headers["authorization"] == "Bearer skill-key"
        return httpx.Response(200, json={"code": "ok", "data": {"items": []}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(base_url="https://skill.example", api_key="skill-key", client=http)
    try:
        with MindMemOSClient(config=config, skill_transport=transport) as client:
            assert client.skills.list(ListRequest()) == {"items": []}
    finally:
        http.close()


def test_skill_service_error_code_is_preserved():
    """Nested Skill service error details remain available to callers."""

    def handler(request):
        return httpx.Response(
            422,
            json={"code": "error", "error": {"code": "pipeline_unavailable", "message": "Not configured"}},
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(base_url="http://localhost", api_key=None, client=http, require_api_key=False)
    try:
        with pytest.raises(ApiError) as failure:
            SkillClient(transport).evolve(EvolveRequest(skill_version_id="version-1"))
        assert failure.value.code == "pipeline_unavailable"
        assert str(failure.value) == "Not configured"
    finally:
        http.close()


def test_skill_write_transport_does_not_retry_network_failure():
    """A dropped response cannot cause duplicate Skill registration."""
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection lost")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(
        base_url="http://localhost", api_key=None, client=http, require_api_key=False, max_retries=0
    )
    try:
        with pytest.raises(TransportError):
            SkillClient(transport).register(REQUESTS[0][2])
        assert attempts == 1
    finally:
        http.close()


@pytest.mark.parametrize(
    "model, fields",
    [
        (SkillAnalyzeRequest, {"skill_version_id": " "}),
        (SkillAnalyzeRequest, {"skill_version_id": "v1", "force_refresh": "true"}),
        (TrajectoryAnalyzeRequest, {"trajectory_id": ""}),
        (TrajectoryAnalyzeRequest, {"trajectory_id": "t1", "verifier_score": float("inf")}),
        (EvaluationQueryRequest, {"skill_version_ids": []}),
        (EvaluationQueryRequest, {"skill_version_ids": [" "]}),
        (EvaluationQueryRequest, {"skill_version_ids": ["v1"], "limit": 1001}),
        (EvaluationQueryRequest, {"skill_version_ids": ["v1"], "judge_types": ["unknown"]}),
    ],
)
def test_analysis_requests_reject_invalid_inputs(model, fields):
    """Reject invalid scoring inputs before transport."""
    with pytest.raises(ValueError):
        model(**fields)


def test_recommend_evaluation_is_opt_in():
    """Adding evaluation support leaves legacy request serialization unchanged."""
    request = RecommendRequest(graph_ref="g", query="Task")
    assert "include_evaluation" not in request.model_dump(exclude_unset=True)
    selected = RecommendRequest(graph_ref="g", query="Task", include_evaluation=True)
    assert selected.model_dump(exclude_unset=True)["include_evaluation"] is True
