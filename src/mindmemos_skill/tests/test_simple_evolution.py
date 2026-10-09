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
    """Return one exact Markdown edit for every extraction call."""

    calls = []

    def __init__(self, config):
        """Retain the selected offline model configuration."""
        self.config = config

    def run(self, *, messages):
        """Return deterministic patch text."""
        self.calls.append(messages)
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
            "observability": {
                "backend": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'audit.db'}"}}
            },
            "evolution_pipelines": {
                "simple": {
                    "class_name": "simple_evolution",
                    "config": {
                        "evolver": {"class_name": "simple_evolve", "config": {"optimizer": {"model": "offline"}}},
                        "work_dir": str(tmp_path / "runs"),
                        "traj_collect": {"sample_size": 2, "trajectory_limit": 4, "seed": 7},
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
                            skill_version_id=first.skill_version_id,
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
                for identity, version_id in [("binding-only", None), ("other-version", "different-version")]:
                    scoped.trajectories.add(
                        TrajectoryRecord(
                            project_id="project",
                            trajectory_id=identity,
                            trajectory_hash=identity,
                            task_id=identity,
                            rollout_id=identity,
                            task_instruction="Ignore this record",
                            env_ref="offline",
                            skill_id=first.skill_id,
                            skill_version_id=version_id,
                            skill_bindings=[{"version_id": first.skill_version_id}],
                            created_at=now,
                        )
                    )
                selected = collect_skill_trajectories(
                    service.backends["persistence"],
                    skill_version_id=first.skill_version_id,
                    limit=2,
                    created_from=now - timedelta(minutes=1),
                )
                assert [record.trajectory_id for record in selected] == ["trace-0", "trace-1"]
                both_sources = collect_skill_trajectories(
                    service.backends["persistence"],
                    skill_version_id=first.skill_version_id,
                    limit=4,
                )
                assert {record.source for record in both_sources} == {"local", "reported"}
                assert len(both_sources) == 4
                assert all(record.skill_version_id == first.skill_version_id for record in both_sources)
                assert not collect_skill_trajectories(
                    service.backends["persistence"],
                    skill_version_id=first.skill_version_id,
                    limit=2,
                    status="succeeded",
                )

            def collect_with_defaults(persistence, **kwargs):
                assert set(kwargs) == {"skill_version_id", "limit", "max_page_size"}
                return collect_skill_trajectories(persistence, **kwargs)

            monkeypatch.setattr(
                "mindmemos_skill.pipelines.builtin_pipelines.evolve.evolving.collect_skill_trajectories",
                collect_with_defaults,
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
                with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                    counts = scoped.trajectories.counts(first.skill_version_id)
                    expected = 0 if interruption == "model" else 2
                    assert counts.model_dump() == {"total": 4, "consumed": expected, "unconsumed": 4 - expected}
                evidence_path = tmp_path / "runs" / request.run_id / "evidence.json"
                evidence_before = evidence_path.read_bytes()
                monkeypatch.setattr(SkillDataService, "commit_algorithm", commit)
                monkeypatch.setattr(pipeline.evolver, "evolve", evolve)
                # A resumed run must not query or resample historical evidence.
                with monkeypatch.context() as replay:
                    replay.setattr(
                        "mindmemos_skill.pipelines.builtin_pipelines.evolve.evolving.collect_skill_trajectories",
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
            assert len(FakeAgent.calls) == calls == 1
            audit = service.backends["observability"]
            run = audit.get_run("project", request.run_id)
            assert run.status == "completed"
            assert run.base_version_id == first.skill_version_id
            assert run.published_version_id == result.version_id
            roots = [span for span in audit.spans("project", request.run_id) if span.name == "algorithm.evolve"]
            assert len(roots) == (3 if interruption else 2)
            assert any(span.links for span in roots)
            assert any(log.name == "algorithm.run.completed" for log in audit.logs("project", request.run_id))
            with pytest.raises(SkillConflictError, match="different inputs"):
                await service.run_evolution(request.model_copy(update={"auto_accept": False}))
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                published = scoped.skills.version_by_id("project", result.version_id)
                assert scoped.trajectories.counts(first.skill_version_id).model_dump() == {
                    "total": 4,
                    "consumed": 2,
                    "unconsumed": 2,
                }
                assert scoped.trajectories.counts(result.version_id).total == 0
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
            assert len(FakeAgent.calls) == 1

            def add_version_traces(version_id, prefix):
                with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                    for index in range(2):
                        scoped.trajectories.add(
                            TrajectoryRecord(
                                project_id="project",
                                trajectory_id=f"{prefix}-{index}",
                                trajectory_hash=f"{prefix}-{index}",
                                task_id=f"{prefix}-{index}",
                                rollout_id=f"{prefix}-{index}",
                                task_instruction="Save and verify",
                                env_ref="offline",
                                skill_id=first.skill_id,
                                skill_version_id=version_id,
                                trajectory=[{"role": "assistant", "content": "No verification"}],
                            )
                        )

            add_version_traces(result.version_id, "active-trace")
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
            evidence = json.loads(FakeAgent.calls[-1][1]["content"])
            assert evidence["skill"] == result.content
            assert all(item["skills_used"][first.skill_id] == result.content for item in evidence["executions"])
            assert {item["trajectory_id"] for item in evidence["executions"]} == {"active-trace-0", "active-trace-1"}
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

            add_version_traces(branch.version_id, "branch-trace")

            class NoChangeAgent(FakeAgent):
                def run(self, *, messages):
                    self.calls.append(messages)
                    return SimpleNamespace(output='{"reasoning":"No supported changes","edits":[]}')

            pipeline.evolver.agent_factory = NoChangeAgent
            unchanged_request = SkillPipelineRequest(
                project_id="project", skill_ref=first.skill_id, pipeline="simple", run_id="evolve_" + uuid4().hex
            )
            unchanged_reporter = SimpleNamespace(report=AsyncMock())
            unchanged = await service.run_evolution(unchanged_request, reporter=unchanged_reporter)
            assert unchanged.version_id == branch.version_id
            assert unchanged.base_version_id == branch.version_id
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                assert scoped.trajectories.counts(branch.version_id).model_dump() == {
                    "total": 2,
                    "consumed": 2,
                    "unconsumed": 0,
                }
            assert unchanged_reporter.report.await_args_list[-1].kwargs["stage"] == "unchanged"
            no_change_calls = len(FakeAgent.calls)
            assert await service.run_evolution(unchanged_request) == unchanged
            assert len(FakeAgent.calls) == no_change_calls
            pipeline.evolver.agent_factory = FakeAgent

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
                if path.is_dir() and path.name not in {result.run_id, inactive.run_id, branch.run_id, unchanged.run_id}
            ]
            assert len(pending) == 1
            assert (pending[0] / "candidate.json").is_file()
            with service.backends["persistence"].session(Scope(fields={"project_id": "project"})) as scoped:
                assert (
                    scoped.skills.resolve_family("project", first.skill_id).active_version_id == competing_versions[0]
                )
    finally:
        await service.close()


@pytest.mark.parametrize("max_page_size", [1, 2, 200])
def test_collection_filters_exact_version_and_bounds_pages(max_page_size):
    """Filter before pagination and retain large records without individual loads."""
    rows = []
    for index, (version_id, instruction) in enumerate(
        [
            ("other-version", "Other version of the same Skill"),
            (None, "Binding-only record"),
            ("target-version", "X" * 300_000),
            ("target-version", "First"),
            ("target-version", "Second"),
            ("target-version", "Third"),
        ]
    ):
        record = TrajectoryRecord(
            project_id="project",
            trajectory_id=f"trace-{index}",
            trajectory_hash=str(index),
            task_id=str(index),
            rollout_id=str(index),
            task_instruction=instruction,
            env_ref="offline",
            skill_id="same-family",
            skill_version_id=version_id,
            skill_bindings=[{"version_id": "target-version"}],
        )
        rows.append(record.model_dump(mode="python"))
    queries = []

    def select(table, *, filters, fields, offset, limit, **kwargs):
        version_filter = next(item for item in filters if item.field == "skill_version_id")
        assert version_filter.value == "target-version"
        assert not any(item.field == "skill_id" for item in filters)
        assert fields is None
        assert 0 < limit <= max_page_size
        queries.append((offset, limit))
        matching = [row for row in rows if row["skill_version_id"] == version_filter.value]
        return matching[offset : offset + limit]

    store = SimpleNamespace(database=SimpleNamespace(select=select))
    selected = collect_skill_trajectories(
        store,
        skill_version_id="target-version",
        limit=2,
        max_page_size=max_page_size,
    )
    assert [record.trajectory_id for record in selected] == ["trace-2", "trace-3"]
    assert selected[0].task_instruction == "X" * 300_000
    assert sum(page_size for _, page_size in queries) == 2
    queries.clear()
    exhausted = collect_skill_trajectories(
        store,
        skill_version_id="target-version",
        limit=5,
        max_page_size=max_page_size,
    )
    assert [record.trajectory_id for record in exhausted] == ["trace-2", "trace-3", "trace-4", "trace-5"]
    assert queries[-1][0] == 4


@pytest.mark.parametrize("max_page_size", [0, -1])
def test_collection_rejects_nonpositive_page_size(max_page_size):
    """Reject an invalid page bound before querying storage."""
    with pytest.raises(ValueError, match="positive"):
        collect_skill_trajectories(None, skill_version_id="target", limit=1, max_page_size=max_page_size)


def test_collection_does_not_accept_family_id_parameter():
    """Keep the collector API restricted to an exact version."""
    with pytest.raises(TypeError, match="skill_id"):
        collect_skill_trajectories(None, skill_id="family", limit=1)
