"""Exercise the SDK against the standalone Skill service HTTP entry."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from mindmemos_sdk.errors import ApiError
from mindmemos_sdk.skills import (
    EvaluationQueryRequest,
    ListRequest,
    RegisterRequest,
    ShowRequest,
    SkillAnalyzeRequest,
    SkillClient,
    SkillResolveRequest,
    StructuredSkillInfo,
    TrajectoryAnalyzeRequest,
    TrajectoryReportRequest,
    TrajectoryShowRequest,
)
from mindmemos_sdk.skills.core import SkillRoute
from mindmemos_sdk.transport import HttpTransport
from mindmemos_skill.api import create_service
from mindmemos_skill.api.entries.fastapi import create_app

from mindmemos_skill import SkillConfig


def test_skill_client_roundtrip_with_local_service(tmp_path):
    """Registration, reading, and trajectory reporting work across the real HTTP entry."""
    service = asyncio.run(
        create_service(
            SkillConfig.model_validate(
                {
                    "service": {
                        "home": tmp_path,
                        "persistence": {"database": {"class_name": "sqlite"}},
                        "repository_pipeline": {"class_name": "repository"},
                    },
                    "fastapi": {"auth": {"mode": "none", "default_project_id": "sdk-test"}},
                }
            )
        )
    )
    with TestClient(create_app(service)) as http:
        service_routes = {route.path for route in http.app.routes if route.path.startswith("/v1/skills/")}
        assert service_routes == {f"/v1/skills/{route.value}" for route in SkillRoute}
        transport = HttpTransport(base_url="http://testserver", api_key=None, client=http, require_api_key=False)
        client = SkillClient(transport)
        content = "---\nname: demo\ndescription: Demo skill\n---\nUse concise answers.\n"
        created = client.register(
            RegisterRequest(skill=StructuredSkillInfo(content=content, commit_message="Initial"), alias="demo")
        )
        family_id = created["skill_id"]
        version_id = created["skill_version_id"]
        assert client.list(ListRequest())["items"][0]["skill_id"] == family_id
        assert client.show(ShowRequest(skill_id=family_id))["version"]["content"] == content
        assert client.skill_resolve(SkillResolveRequest(skill_ref="demo"))["skill_version_id"] == version_id
        reported = client.report_trajectory(
            TrajectoryReportRequest(
                user_id="user-1",
                agent_id="agent-1",
                messages=[{"role": "user", "content": "Help me"}],
                skill_version_id=version_id,
            )
        )
        assert (
            client.show_trajectory(TrajectoryShowRequest(trajectory_id=reported["trajectory_id"]))["skill_version_id"]
            == version_id
        )
        with pytest.raises(ApiError, match="configured"):
            client.analyze_skill(SkillAnalyzeRequest(skill_version_id=version_id))
        with pytest.raises(ApiError, match="configured"):
            client.analyze_trajectory(TrajectoryAnalyzeRequest(trajectory_id=reported["trajectory_id"]))
        with pytest.raises(ApiError, match="configured"):
            client.query_evaluations(EvaluationQueryRequest(skill_version_ids=[version_id]))
