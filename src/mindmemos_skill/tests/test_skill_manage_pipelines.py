"""Service-level organization and recommendation over pinned Skill versions."""

import pytest
from mindmemos_skill.api.typing import CommitRequest, OrganizeRequest, RecommendRequest, RegisterRequest
from mindmemos_skill.errors import GraphNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.infra.embedding import OpenAIEmbeddingClient
from mindmemos_skill.pipelines import GraphManagePipeline, ManagePipeline
from mindmemos_skill.skill import Skill, SkillContent

from mindmemos_skill import create_service

from .service_config import deployment_config
from .skill_sources import register_detail, structured


def service_config(path):
    """Configure one management pipeline without external model calls."""
    algorithm = {
        "embedding": {"model": "test", "base_url": "http://test.invalid/v1", "dimensions": 3},
        "build_scene": False,
        "llm_linking": False,
        "query_rewrite": False,
        "mode": "skill_only",
    }
    return deployment_config(
        {
            "home": path.parent / "home",
            "persistence": {"database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{path}"}}},
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {"class_name": "graph_manage", "config": {"algorithm": algorithm}},
        },
        "project",
    )


@pytest.mark.asyncio
async def test_skill_set_queries_without_algorithm(tmp_path):
    """Read Skill Sets through the management pipeline without model settings."""
    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'queries.db'}"}}
            },
            "manage_pipeline": {"class_name": "graph_manage"},
        },
        "project",
    )
    async with await create_service(config) as service:
        assert await service.list_skill_sets() == []
        assert await service.list_skill_set_snapshots("missing") == []
        with pytest.raises(NotImplementedError, match="algorithm configuration"):
            await service.organize_skills(OrganizeRequest(project_id="project", alias="graph", skill_refs=[]))


@pytest.mark.asyncio
async def test_manage_pipelines_publish_reuse_and_restore_pinned_scope(tmp_path, monkeypatch, caplog):
    """Persist explicit versions and keep old recommendations pinned after republishing."""

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] if "csv" in text.lower() else [0.0, 1.0, 0.0] for text in texts]

    monkeypatch.setattr(OpenAIEmbeddingClient, "embed", embed)
    config = service_config(tmp_path / "manage.db")
    async with await create_service(config) as service:
        assert isinstance(service.backends["manage_pipeline"], GraphManagePipeline)
        assert sum(isinstance(backend, ManagePipeline) for backend in service.backends.values()) == 1
        versions = []
        for name, description in (("csv", "Parse CSV tables"), ("images", "Draw pictures")):
            source = Skill(
                SkillContent(
                    content=f"---\nname: {name}\ndescription: {description}\n---\n{description}",
                    name="test-skill",
                    description="Test instructions",
                    standard_format=False,
                )
            ).snapshot()
            detail = await register_detail(
                service, RegisterRequest(skill=structured(source, name=name), project_id=service.default_project_id)
            )
            versions.append(detail.version.version_id)
        historical = await service.get_skill_version(versions[0])
        newer = await service.commit_skill(
            CommitRequest(
                skill=structured(
                    Skill(
                        SkillContent(
                            content="# Updated CSV workflow",
                            name="test-skill",
                            description="Test instructions",
                            standard_format=False,
                        )
                    ).snapshot()
                ),
                skill_id=historical.skill_id,
                project_id=service.default_project_id,
            )
        )
        assert newer.skill.active_version_id != versions[0]
        request = OrganizeRequest(project_id="project", alias="collection", skill_refs=versions)
        caplog.set_level("INFO", logger="mindmemos_skill")
        caplog.clear()
        first = await service.organize_skills(request)
        assert [
            record.event_body["stage"] for record in caplog.records if record.getMessage() == "algorithm.progress"
        ] == ["organize", "organized"]
        request = request.model_copy(update={"graph_id": first.graph_id, "alias": None})
        assert (await service.organize_skills(request)).graph_version_id == first.graph_version_id
        skill_sets = await service.list_skill_sets()
        assert [(item["skill_set_id"], item["mode"]) for item in skill_sets] == [(first.graph_id, "graph")]
        snapshots = await service.list_skill_set_snapshots("collection")
        assert snapshots[0]["snapshot_id"] == f"skgv-{first.graph_version_id.hex}"
        assert snapshots[0]["request_hash"] is not None
        assert skill_sets[0]["current_version_id"] == f"skgv-{first.graph_version_id.hex}"
        snapshot = await service.get_skill_set_snapshot("collection", first.graph_version_id)
        assert snapshot["mode"] == "graph"
        assert {node["skill_version_id"] for node in snapshot["nodes"]} == set(versions)
        assert (await service.get_skill_version(versions[0])).version_id == versions[0]
        assert len(await service.list_skill_versions(historical.skill_id)) == 2
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})):
            with pytest.raises(PermissionError, match="active scope"):
                await service.repository.list_skill_versions("another", snapshot["nodes"][0]["skill_id"])
        selected = await service.recommend_skills(
            RecommendRequest(
                project_id="project",
                graph_id="collection",
                graph_version_id=first.graph_version_id,
                task="Parse CSV records",
                top_k=1,
            )
        )
        assert [version.version_id for version in selected.skills] == [versions[0]]
        second = await service.organize_skills(request.model_copy(update={"skill_refs": [versions[1]]}))
        assert second.graph_version_id != first.graph_version_id

    async with await create_service(config) as reopened:
        previous = await reopened.recommend_skills(
            RecommendRequest(
                project_id="project",
                graph_id="collection",
                graph_version_id=first.graph_version_id,
                task="Parse CSV records",
                top_k=1,
            )
        )
        assert [version.version_id for version in previous.skills] == [versions[0]]
        current = await reopened.recommend_skills(
            RecommendRequest(
                project_id="project",
                graph_id="collection",
                graph_version_id=second.graph_version_id,
                task="Parse CSV records",
                top_k=1,
            )
        )
        assert [version.version_id for version in current.skills] == [versions[1]]
        assert {item["snapshot_id"] for item in await reopened.list_skill_set_snapshots("collection")} == {
            f"skgv-{first.graph_version_id.hex}",
            f"skgv-{second.graph_version_id.hex}",
        }
        assert await reopened.list_skill_set_snapshots("missing") == []
        with pytest.raises(GraphNotFoundError, match="Graph not found"):
            await reopened.get_skill_set_snapshot("missing", first.graph_version_id)
        with pytest.raises(ValueError, match="default project"):
            await reopened.list_skill_sets("another")
        with pytest.raises(ValueError, match="default project"):
            await reopened.get_skill_set_snapshot("collection", first.graph_version_id, "another")
        with pytest.raises(ValueError, match="default project"):
            await reopened.recommend_skills(
                RecommendRequest(
                    project_id="another",
                    graph_id="collection",
                    graph_version_id=first.graph_version_id,
                    task="CSV",
                )
            )


@pytest.mark.asyncio
async def test_manage_pipeline_preflight_rejects_missing_store_and_wrong_family(tmp_path):
    """Fail invalid assembly before creating any persistence or model resource."""
    config = service_config(tmp_path / "unused.db")
    with pytest.raises(ValueError, match="global persistence"):
        await create_service(config.service.model_copy(update={"persistence": None}))
    with pytest.raises(ValueError, match="ManagePipeline"):
        await create_service(
            config.service.model_copy(
                update={
                    "manage_pipeline": config.service.manage_pipeline.model_copy(update={"class_name": "repository"})
                }
            )
        )
