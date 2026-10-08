"""Offline tests for bounded Skill evolution and durable publication."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.api.typing import RegisterRequest, SkillPipelineRequest, StructuredSkillInfo
from mindmemos_skill.common.json import read_json
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.common.trajectories import collect_skill_trajectories
from mindmemos_skill.errors import SkillConflictError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence import TrajectoryRecord
from mindmemos_skill.persistence.db_services.skills import SkillDataService
from mindmemos_skill.skill import Skill, SkillContent

from mindmemos_skill import ServiceConfig, create_service


class FakeAgent:
    """Return a reflection followed by one exact Markdown edit."""

    calls = []

    def __init__(self, config):
        """Retain the selected offline model configuration."""
        self.config = config

    def run(self, *, messages):
        """Return deterministic reflection and patch text in alternating calls."""
        self.calls.append(messages)
        if len(self.calls) % 2:
            return SimpleNamespace(output="Failed to verify completion in the observed trace.")
        return SimpleNamespace(
            output='{"reasoning":"verify completion","edits":[{"op":"append","content":"Verify completion before responding."}]}'
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("interruption", [None, "model", "before_commit", "after_commit"])
@pytest.mark.asyncio
async def test_simple_evolution_collects_samples_and_publishes(tmp_path, monkeypatch, dynamic, interruption):
    """Exercise both trajectory origins, random sampling and version publication."""
    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'skills.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "evolution_pipelines": {
                "simple": {
                    "class_name": "simple_evolution",
                    "config": {
                        "evolver": {"class_name": "simple_evolve", "config": {"optimizer": {"model": "offline"}}},
                        "work_dir": str(tmp_path / "runs"),
                        "sample_size": 2,
                        "trajectory_limit": 4,
                        "scan_limit": 5,
                        "seed": 7,
                    },
                }
            },
        }
    )
    service = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    try:
        source = Skill(
            SkillContent(
                content="# Workflow\nSave the output.\n",
                name="Workflow",
                description="Workflow instructions",
                standard_format=False,
                resources={"guide.md": b"Keep this file.", "asset.bin": b"unchanged"},
            ),
        )
        if dynamic:
            source = TreeSkill(
                TreeSkillConfig(router={"model": "offline-router"}),
                source=SkillContent(
                    content="---\r\nname: Workflow\r\ndescription: Workflow instructions\r\n---\r\n# Workflow\r\nSave the output.\r\n",
                    resources=source.resources,
                ),
            )
        original = source.snapshot()
        with service.request_principal("test-user"):
            first = await service.register_skill(
                RegisterRequest(
                    project_id="project",
                    skill=StructuredSkillInfo(
                        **source.source.model_dump(exclude={"resources"}), resources=source.snapshot().resources
                    ),
                )
            )
            now = datetime.now(UTC)
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                if dynamic:
                    owner = scoped.skills.resolve_family("project", first.skill_id)
                    version = scoped.skills.commit(owner, original, message="Dynamic source")
                    first = first.model_copy(update={"skill_version_id": version.active_version_id})
                for index in range(3):
                    scoped.trajectories.add(
                        TrajectoryRecord(
                            project_id="project",
                            trajectory_id=f"trace-{index}",
                            trajectory_hash=f"hash-{index}",
                            task_id=f"task-{index}",
                            rollout_id=f"rollout-{index}",
                            task_instruction="Save and verify",
                            env_ref="offline",
                            skill_bindings=[{"version_id": first.skill_version_id}],
                            status="failed",
                            trajectory=[{"role": "assistant", "content": "Saved without checking"}],
                            created_at=now - timedelta(minutes=index),
                        )
                    )
                scoped.trajectories.add(
                    TrajectoryRecord(
                        project_id="project",
                        trajectory_id="reported-trace",
                        trajectory_hash="reported-hash",
                        source="reported",
                        task_id="reported-task",
                        rollout_id="reported-rollout",
                        task_instruction="Save and verify",
                        env_ref="harness",
                        skill_id=first.skill_id,
                        skill_version_id=first.skill_version_id,
                        user_id="user",
                        agent_id="agent",
                        received_at=now - timedelta(minutes=3),
                        created_at=now - timedelta(minutes=3),
                        trajectory=[{"role": "assistant", "content": "No verification"}],
                    )
                )
                selected = collect_skill_trajectories(
                    service.backends["persistence"],
                    skill_id=first.skill_id,
                    limit=2,
                    scan_limit=3,
                    created_from=now - timedelta(minutes=1),
                )
                assert [record.trajectory_id for record in selected] == ["trace-0", "trace-1"]
                both_sources = collect_skill_trajectories(
                    service.backends["persistence"],
                    skill_id=first.skill_id,
                    limit=4,
                    scan_limit=5,
                )
                assert {record.source for record in both_sources} == {"local", "reported"}
                assert not collect_skill_trajectories(
                    service.backends["persistence"],
                    skill_id=first.skill_id,
                    limit=2,
                    scan_limit=3,
                    status="succeeded",
                )
            FakeAgent.calls = []
            pipeline = service.backends["evolution_pipelines.simple"]
            pipeline.evolver.agent_factory = FakeAgent
            reporter = SimpleNamespace(report=AsyncMock())
            request = SkillPipelineRequest(
                project_id="project",
                skill_ref=first.skill_id,
                pipeline="simple",
                run_id="evolve_" + uuid4().hex,
            )
            commit = SkillDataService.commit_algorithm
            evolve = pipeline.evolver.evolve
            if interruption == "model":

                async def interrupted_model(**kwargs):
                    raise RuntimeError("Simulated interrupted model call")

                monkeypatch.setattr(pipeline.evolver, "evolve", interrupted_model)
            elif interruption:

                def interrupted_commit(store, *args, **kwargs):
                    if interruption == "after_commit":
                        commit(store, *args, **kwargs)
                    raise RuntimeError("Simulated interrupted publication")

                monkeypatch.setattr(SkillDataService, "commit_algorithm", interrupted_commit)
            if interruption:
                with pytest.raises(RuntimeError, match="Simulated"):
                    await service.run_evolution(request, reporter=reporter)
                evidence_path = tmp_path / "runs" / request.run_id / "evidence.json"
                evidence_before = evidence_path.read_bytes()
                monkeypatch.setattr(SkillDataService, "commit_algorithm", commit)
                monkeypatch.setattr(pipeline.evolver, "evolve", evolve)
                # A resumed run must not query or resample historical evidence.
                with monkeypatch.context() as replay:
                    replay.setattr(
                        "mindmemos_skill.pipelines.builtin_pipelines.evolve.simple_evolution.collect_skill_trajectories",
                        lambda *args, **kwargs: pytest.fail("Resampled evidence on recovery"),
                    )
                    result = await service.run_evolution(request, reporter=reporter)
                assert evidence_path.read_bytes() == evidence_before
            else:
                result = await service.run_evolution(request, reporter=reporter)
                assert [call.kwargs["stage"] for call in reporter.report.await_args_list] == [
                    "evidence",
                    "sampled",
                    "evolved",
                    "published",
                ]
            calls = len(FakeAgent.calls)
            assert await service.run_evolution(request) == result
            assert len(FakeAgent.calls) == calls == 2
            with pytest.raises(SkillConflictError, match="different inputs"):
                await service.run_evolution(request.model_copy(update={"auto_accept": False}))
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                published = scoped.skills.version_by_id("project", result.version_id)
                assert published.origin == "simple_evolve"
                assert published.parent_version_ids == [first.skill_version_id]
                evolved = Skill.from_snapshot(scoped.skills.get_skill(result.version_id).snapshot())
                assert "Verify completion" in evolved.content
                assert evolved.resources["guide.md"] == b"Keep this file."
                assert evolved.resources["asset.bin"] == b"unchanged"
                assert result.content == evolved.content
                assert result.resources == evolved.snapshot().resources
                assert result.content_hash == published.content_hash
                assert result.skill_hash == published.skill_hash
                restored = load_runtime_skill(evolved.snapshot())
                assert isinstance(restored, TreeSkill) == dynamic
                assert restored.snapshot().config == original.config
                assert source.snapshot() == original
                assert result.content.startswith(original.content.rstrip())
                assert result.content_hash != original.content_hash
                assert result.skill_hash != original.skill_hash
                restored.export(tmp_path / "export")
                assert (tmp_path / "export/SKILL.md").read_bytes() == result.content.encode("utf-8")
                assert (tmp_path / "export/asset.bin").read_bytes() == b"unchanged"
            recorded = read_json(tmp_path / "runs" / result.run_id / "input.json")
            assert len(recorded["sampled_trajectory_ids"]) == 2
            assert len(set(recorded["sampled_trajectory_ids"])) == 2
            assert len(FakeAgent.calls) == 2
            inactive = await service.run_evolution(
                SkillPipelineRequest(
                    project_id="project",
                    skill_ref=first.skill_id,
                    auto_accept=False,
                    pipeline="simple",
                )
            )
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                assert scoped.skills.version_by_id("project", inactive.version_id).parent_version_ids == [
                    result.version_id
                ]
                assert scoped.skills.resolve_family("project", first.skill_id).active_version_id == result.version_id
            evidence = json.loads(FakeAgent.calls[-2][1]["content"])
            assert evidence["skill"] == result.content
            assert all(item["skills_used"][first.skill_id] == original.content for item in evidence["executions"])
            branch = await service.run_evolution(
                SkillPipelineRequest(
                    project_id="project",
                    skill_ref=first.skill_id,
                    base_version_id=first.skill_version_id,
                    pipeline="simple",
                )
            )
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                assert scoped.skills.version_by_id("project", branch.version_id).parent_version_ids == [
                    first.skill_version_id
                ]
                assert scoped.skills.resolve_family("project", first.skill_id).active_version_id == branch.version_id

            assert await service.run_evolution(request) == result
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                assert scoped.skills.resolve_family("project", first.skill_id).active_version_id == branch.version_id

            original_commit = SkillDataService.commit
            original_algorithm_commit = SkillDataService.commit_algorithm
            competing_versions = []

            def changed_parent(store, owner, snapshot, **kwargs):
                competing = original_commit(
                    store,
                    owner,
                    source.with_content(source.content + "\nConcurrent update.").snapshot(),
                    message="Concurrent publication",
                )
                competing_versions.append(competing.active_version_id)
                return original_algorithm_commit(store, owner, snapshot, **kwargs)

            monkeypatch.setattr(SkillDataService, "commit_algorithm", changed_parent)
            with pytest.raises(SkillConflictError):
                await service.run_evolution(
                    SkillPipelineRequest(project_id="project", skill_ref=first.skill_id, pipeline="simple")
                )
            pending = [
                path
                for path in (tmp_path / "runs").iterdir()
                if path.is_dir() and path.name not in {result.run_id, inactive.run_id, branch.run_id}
            ]
            assert len(pending) == 1
            assert (pending[0] / "candidate.json").is_file()
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                assert (
                    scoped.skills.resolve_family("project", first.skill_id).active_version_id == competing_versions[0]
                )
    finally:
        await service.close()
