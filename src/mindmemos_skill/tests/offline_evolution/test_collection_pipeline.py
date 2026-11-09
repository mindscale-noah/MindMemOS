"""Verify persisted collection outputs and activation policy through the shared service."""

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
from mindmemos_sdk.skills import ActivateRequest, AsyncSkillClient, EvolveRequest, ShowRequest, TaskShowRequest
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.algos.evolve.offline_evolution import engine as engine_module
from mindmemos_skill.algos.evolve.offline_evolution.evolver import OfflineSkillEvolver
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout, SkillEvolutionResult
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.typing import (
    AlgoSkillEvolveResult,
    CollectionEvolutionRequest,
    ReportTrajectoryRequest,
    SkillPipelineRequest,
)
from mindmemos_skill.errors import SkillConflictError, SkillNotFoundError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.constants import (
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
)
from mindmemos_skill.persistence.db_services.trajectories.typing import TrajectoryWrite
from mindmemos_skill.pipelines import EvolutionPipeline, Pipeline
from mindmemos_skill.pipelines.builtin_pipelines.offline_evolution.config import OfflineTrajectoryConfig
from pydantic import BaseModel

from mindmemos_skill import create_service

from ..service_config import deployment_config
from .helpers import FakeAgent, write_evidence
from .test_collection_publication import capture, skill


@pytest.fixture
async def scenario(tmp_path, monkeypatch):
    """Prepare real inputs while replacing only the algorithm's candidate generation."""
    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {"database": {"class_name": "sqlite"}},
            "repository_pipeline": {"class_name": "repository"},
            "task_manager": {"mode": "local"},
            "evolution_pipelines": {
                "offline": {
                    "class_name": "offline_evolution",
                    "config": {
                        "evolver": {
                            "class_name": "offline_evolution",
                            "config": {"agent": {"class_name": "claude", "model": "offline"}},
                        },
                        "work_dir": str(tmp_path / "runs"),
                    },
                }
            },
        },
        "project",
    )
    config.fastapi.evolution_pipelines = {"offline"}
    service = await create_service(config)
    calls = []

    async def propose(self, request):
        assert self.config.agent.model == "offline"
        calls.append(request)
        keys = {value.source.name: key for key, value in request.skills.items()}
        result = dict(request.skills)
        result[keys["alpha"]] = skill("alpha", "Improved", {"new.bin": b"\x00\xff"})
        return SkillEvolutionResult(
            skills=result,
            created_skills={"new/delta": skill("delta")},
        )

    monkeypatch.setattr(OfflineSkillEvolver, "_evolve", propose)
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        members = [await capture(store, name) for name in ("alpha", "beta", "readonly")]
        for item in members:
            await store.trajectories.add(
                TrajectoryWrite(
                    trajectory_id=uuid4().hex,
                    trajectory_hash="stored-evidence",
                    task_id=uuid4().hex,
                    rollout_id=uuid4().hex,
                    task_instruction="Process a document.",
                    env_ref="offline",
                    status="completed",
                    skill_bindings=[{"skill_id": item.owner.skill_id, "version_id": item.base.version_id}],
                    trajectory=[{"role": "user", "content": "Process a document."}],
                )
            )
    try:
        yield service, members, calls, config
    finally:
        await service.close()


async def counts(service):
    """Inspect repository state separately from candidate and task artifacts."""
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        return {
            table: await store.database.count(table)
            for table in (SKILLS_TABLE, SKILL_VERSIONS_TABLE, SKILL_GRAPH_VERSIONS_TABLE)
        }


def request(members, *, auto_accept=False):
    """Keep all inputs readable while authorizing only the first two mutations."""
    return CollectionEvolutionRequest(
        project_id="project",
        skill_version_ids=[item.base.version_id for item in members],
        editable_skills=[item.owner.skill_id for item in members[:2]],
        auto_accept=auto_accept,
    )


async def test_evolution_saves_versions_without_activating_existing_skills(scenario):
    """Persist every output once while retaining existing active versions."""
    service, members, calls, _ = scenario
    before = await counts(service)
    body = request(members).model_copy(update={"run_id": "evolve_" + uuid4().hex})
    result = await service.run_evolution(body)
    assert list(result.created_skills) == ["new/delta"]
    assert set(result.skills) == {member.owner.skill_id for member in members}
    assert {item.action for item in result.members} == {"updated", "created", "unchanged"}
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        for original in members:
            assert (
                await store.skills.resolve_family(original.owner.skill_id)
            ).active_version_id == original.base.version_id
        updated = next(item for item in result.members if item.action == "updated")
        assert updated.version_id != updated.base_version_id
        assert (await store.skills.get(updated.version_id)).snapshot() == updated.snapshot
        created = next(item for item in result.members if item.action == "created")
        assert (await store.skills.resolve_family(created.skill_id)).active_version_id == created.version_id
    after = await counts(service)
    assert after[SKILLS_TABLE] == before[SKILLS_TABLE] + 1
    assert after[SKILL_VERSIONS_TABLE] == before[SKILL_VERSIONS_TABLE] + 2
    assert after[SKILL_GRAPH_VERSIONS_TABLE] == before[SKILL_GRAPH_VERSIONS_TABLE]
    assert await service.run_evolution(body) == result
    assert await counts(service) == after and len(calls) == 1


async def test_collection_evolution_reads_shared_report_once(scenario):
    """The same physical execution supplies context for all matching input members."""
    service, members, calls, _ = scenario
    versions = [member.base.version_id for member in members[:2]]
    shared = await service.report_trajectory(
        ReportTrajectoryRequest(
            project_id="project",
            skill_version_ids=versions,
            messages=[{"role": "user", "content": "Use both Skills."}],
        )
    )
    await service.run_evolution(request(members))
    recorded = [item for item in calls[0].trajectories if item.trajectory_id == shared.trajectory_id]
    assert len(recorded) == 1
    assert list(recorded[0].skills) == [member.owner.skill_id for member in members[:2]]
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        assert (await store.trajectories.counts(versions[0])).consumed == 2
        assert (await store.trajectories.counts(versions[1])).consumed == 2


@pytest.mark.parametrize("first", [0, 2])
async def test_collection_audit_keeps_input_and_published_version_on_the_same_member(scenario, monkeypatch, first):
    """Never attribute another member's update to a changed or read-only input."""
    service, members, _, _ = scenario
    recorded = []
    trace_run = service._observability.trace_run

    @asynccontextmanager
    async def capture_audit(**parameters):
        async with trace_run(**parameters) as audit:
            yield audit
            recorded.append((parameters["base_version_id"], audit.published_version_id))

    monkeypatch.setattr(service._observability, "trace_run", capture_audit)
    ordered = [members[first], *(member for index, member in enumerate(members) if index != first)]
    body = request(members).model_copy(update={"skill_version_ids": [member.base.version_id for member in ordered]})
    result = await service.run_evolution(body)
    selected = next(member for member in result.members if member.skill_id == ordered[0].owner.skill_id)
    assert recorded == [(ordered[0].base.version_id, selected.version_id if selected.action == "updated" else None)]


async def test_auto_accept_activates_saved_existing_versions(scenario):
    """Use the same publication path and select saved versions when requested."""
    service, members, calls, _ = scenario
    result = await service.run_evolution(request(members, auto_accept=True))
    assert len(calls) == 1 and (await counts(service))[SKILL_VERSIONS_TABLE] == 5
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        for member in result.members:
            assert (await store.skills.resolve_family(member.skill_id)).active_version_id == member.version_id


@pytest.mark.parametrize(
    ("available", "trajectory_limit", "expected"),
    [(3, None, 3), (3, 2, 2), (101, None, 101)],
)
async def test_offline_uses_all_available_evidence_unless_explicitly_capped(
    scenario, monkeypatch, available, trajectory_limit, expected
):
    """Read across pages without an implicit minimum or 100-record pool cap."""
    service, _, _, _ = scenario
    pipeline = service.backends["evolution_pipelines.offline"]
    pipeline.config.traj_collect = OfflineTrajectoryConfig(trajectory_limit=trajectory_limit, max_page_size=2)
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        member = await capture(store, "available-evidence")
    reports = {}
    for index in range(available):
        messages = [{"role": "trajectory", "content": {"schema_version": "ATIF-v1.7", "steps": [index]}}]
        report = await service.report_trajectory(
            ReportTrajectoryRequest(
                project_id="project",
                skill_version_ids=[member.base.version_id],
                messages=messages,
                status="failed" if index == 0 else "completed",
            )
        )
        reports[report.trajectory_id] = report
    observed = []

    async def inspect(self, request):
        observed.extend(request.trajectories)
        return SkillEvolutionResult(skills=request.skills)

    monkeypatch.setattr(OfflineSkillEvolver, "_evolve", inspect)
    before = await counts(service)
    result = await service.run_evolution(
        CollectionEvolutionRequest(project_id="project", skill_version_ids=[member.base.version_id], auto_accept=False)
    )
    assert result.members and len(observed) == expected
    assert {record.trajectory_id for record in observed} <= reports.keys()
    for record in observed:
        assert record.trajectory == reports[record.trajectory_id].trajectory
        assert record.status == reports[record.trajectory_id].status
    if expected == available:
        assert {record.trajectory_id for record in observed} == reports.keys()
    assert await counts(service) == before


async def test_offline_requires_some_evidence_before_calling_the_algorithm(scenario):
    """Reject an empty exact-version collection without invoking a model."""
    service, _, calls, _ = scenario
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        member = await capture(store, "no-evidence")
    with pytest.raises(ValueError, match="No trajectories available"):
        await service.run_evolution(
            CollectionEvolutionRequest(
                project_id="project", skill_version_ids=[member.base.version_id], auto_accept=False
            )
        )
    assert calls == []


async def test_offline_recovery_reuses_frozen_evidence_after_new_uploads(scenario, monkeypatch):
    """Keep the original complete evidence set after an interrupted algorithm call."""
    service, members, _, _ = scenario
    version_id = members[0].base.version_id
    observed = []

    async def interrupt(self, request):
        observed.append([record.trajectory_id for record in request.trajectories])
        if len(observed) == 1:
            raise RuntimeError("Interrupted after collecting evidence")
        return SkillEvolutionResult(skills=request.skills)

    monkeypatch.setattr(OfflineSkillEvolver, "_evolve", interrupt)
    body = CollectionEvolutionRequest(
        project_id="project", skill_version_ids=[version_id], run_id="evolve_" + uuid4().hex, auto_accept=False
    )
    with pytest.raises(RuntimeError, match="Interrupted"):
        await service.run_evolution(body)
    await service.report_trajectory(
        ReportTrajectoryRequest(
            project_id="project", skill_version_ids=[version_id], messages=[{"role": "user", "content": "Later"}]
        )
    )
    result = await service.run_evolution(body)
    assert result.members
    assert len(observed[0]) == 1 and observed[0] == observed[1]


@pytest.mark.parametrize("instruction", [None, "", "  \n"])
async def test_reported_trajectories_without_instruction_remain_eligible(scenario, monkeypatch, instruction):
    """Preserve reported raw evidence without requiring an extracted user instruction."""
    service, _, _, _ = scenario
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        member = await capture(store, "historical")
    raw = {"steps": [{"source": "user", "message": "Process a historical document."}]}
    messages = [{"role": "trajectory", "content": raw}]
    if instruction is not None:
        messages.insert(0, {"role": "user", "content": instruction})
    report = await service.report_trajectory(
        ReportTrajectoryRequest(
            project_id="project",
            skill_version_ids=[member.base.version_id],
            messages=messages,
        )
    )
    observed = []

    async def inspect_evidence(self, request):
        observed.extend(request.trajectories)
        return SkillEvolutionResult(skills=request.skills)

    monkeypatch.setattr(OfflineSkillEvolver, "_evolve", inspect_evidence)
    before = await counts(service)
    result = await service.run_evolution(
        CollectionEvolutionRequest(
            project_id="project",
            skill_version_ids=[member.base.version_id],
            auto_accept=False,
        )
    )
    assert result.members
    assert len(observed) == 1 and observed[0].trajectory_id == report.trajectory_id
    assert observed[0].task.instruction == (instruction or "")
    assert observed[0].trajectory == messages
    assert await counts(service) == before


async def test_persisted_terminal_event_and_failure_status_reach_s1_and_g1(scenario, monkeypatch):
    """Keep the full stored execution visible across collection, export and file snapshots."""
    service, _, _, _ = scenario
    events = [
        {"role": "user", "content": "Process a historical document."},
        {"role": "assistant", "content": "Reading the document."},
        "TERMINATED: execution limit reached.",
    ]
    trajectory_id = uuid4().hex
    with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
        member = await capture(store, "historical-json")
        await store.trajectories.add(
            TrajectoryWrite(
                trajectory_id=trajectory_id,
                trajectory_hash="mixed-events",
                task_id=uuid4().hex,
                rollout_id=uuid4().hex,
                task_instruction="Process a historical document.",
                env_ref="offline",
                status="failed",
                skill_bindings=[{"skill_id": member.owner.skill_id, "version_id": member.base.version_id}],
                trajectory=events,
            )
        )
    observed = []

    def inspect_s1(request):
        layout = RunLayout(root=request.cwd)
        path = next(layout.input_trajectories.glob("*.json"))
        payload = json.loads(path.read_text())
        assert payload["trajectory_id"] == trajectory_id
        assert payload["trajectory"] == events
        assert payload["status"] == "failed"
        observed.append(layout)
        write_evidence(
            layout.evidence_workspace,
            script="""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--inputs")
parser.add_argument("--out")
args = parser.parse_args()
root = Path(args.inputs)
path = next((root / "trajectories").glob("*.json"))
record = json.loads(path.read_text())
relative = path.relative_to(root).as_posix()
findings = [
    {"id": "terminal", "path": relative, "quote": record["trajectory"][-1]},
    {"id": "outcome", "path": relative, "quote": json.dumps(record["status"])},
]
Path(args.out).write_text(json.dumps({"processed": [relative], "findings": findings}))
""",
        )

    # Exercise the real export and gates instead of the fixture's candidate substitute.
    monkeypatch.setattr(OfflineSkillEvolver, "_evolve", OfflineSkillEvolver._run)
    backend = FakeAgent([inspect_s1])
    monkeypatch.setattr(engine_module, "run_agent", backend.run)
    result = await service.run_evolution(
        CollectionEvolutionRequest(
            project_id="project",
            skill_version_ids=[member.base.version_id],
            auto_accept=False,
        )
    )
    assert result.members
    assert len(observed) == 1
    report = json.loads((observed[0].evidence / "report.json").read_text())
    assert report["status"] == "accepted"
    assert {finding["id"] for finding in report["findings"]} == {"terminal", "outcome"}


async def test_changed_current_version_rejects_atomic_activation(scenario, monkeypatch):
    """Roll back every publication when an active input changes during generation."""
    service, members, calls, _ = scenario
    propose = OfflineSkillEvolver._evolve

    async def concurrent(self, algorithm_request):
        result = await propose(self, algorithm_request)
        with service._persistence.session(Scope(fields={"project_id": "project"})) as store:
            await store.skills.commit(
                members[0].owner, skill("alpha", "Concurrent").snapshot(), message="another writer"
            )
        return result

    monkeypatch.setattr(OfflineSkillEvolver, "_evolve", concurrent)
    before = await counts(service)
    with pytest.raises(SkillConflictError):
        await service.run_evolution(request(members, auto_accept=True))
    after = await counts(service)
    assert after[SKILLS_TABLE] == before[SKILLS_TABLE]
    assert after[SKILL_VERSIONS_TABLE] == before[SKILL_VERSIONS_TABLE] + 1
    assert len(calls) == 1


async def test_corrupt_unpublished_candidate_is_not_rerun(scenario, monkeypatch):
    """Validate a retained candidate after a publication failure before writing versions."""
    from pathlib import Path

    from mindmemos_skill.common.json import read_json, write_json
    from mindmemos_skill.pipelines.builtin_pipelines.offline_evolution import pipeline as pipeline_module

    service, members, calls, _ = scenario
    body = request(members).model_copy(update={"run_id": "evolve_" + uuid4().hex})

    async def interrupted(*args):
        raise OSError("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline_module, "publish_candidate", interrupted)
        with pytest.raises(OSError, match="publication interrupted"):
            await service.run_evolution(body)
    path = Path(service.backends["evolution_pipelines.offline"].config.work_dir) / body.run_id / "candidate.json"
    payload = read_json(path)
    payload["payload"]["skills"].pop(members[1].owner.skill_id)
    write_json(path, payload)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        await service.run_evolution(body)
    assert len(calls) == 1
    assert (await counts(service))[SKILL_VERSIONS_TABLE] == 3


async def test_sdk_can_activate_saved_version_after_evolution(scenario):
    """Publish inactive outputs through evolve and select them using the existing activation API."""
    service, members, calls, config = scenario
    app = create_app(service, api_config=config.fastapi)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1") as http:
        client = AsyncSkillClient(
            AsyncHttpTransport(base_url="http://127.0.0.1", api_key=None, client=http, require_api_key=False)
        )
        submitted = await client.evolve(
            EvolveRequest(
                skill_version_ids=[item.base.version_id for item in members],
                editable_skills=[item.owner.skill_id for item in members[:2]],
                auto_accept=False,
            )
        )
        for _ in range(500):
            task = await client.show_task(TaskShowRequest(task_id=submitted["task_id"]))
            if task["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert task["status"] == "succeeded", task
        assert "acceptance_status" not in task["result"]
        updated = next(item for item in task["result"]["members"] if item["action"] == "updated")
        before = await client.show(ShowRequest(skill_id=updated["skill_id"]))
        saved = await client.show(ShowRequest(skill_version_id=updated["version_id"]))
        assert before["version"]["skill_version_id"] == updated["base_version_id"]
        assert saved["version"]["content"] == updated["snapshot"]["content"]
        await client.activate(ActivateRequest(skill_id=updated["skill_id"], skill_version_id=updated["version_id"]))
        after = await client.show(ShowRequest(skill_id=updated["skill_id"]))
        assert after["version"]["skill_version_id"] == updated["version_id"]
        assert (await client.show_task(TaskShowRequest(task_id=submitted["task_id"])))["result"] == task["result"]
        assert len(calls) == 1


async def test_published_outputs_remain_project_scoped(scenario):
    """Prevent another project from reading newly saved collection versions."""
    service, members, _, _ = scenario
    result = await service.run_evolution(request(members))
    updated = next(item for item in result.members if item.action == "updated")
    with service._persistence.session(Scope(fields={"project_id": "foreign"})) as store:
        with pytest.raises(SkillNotFoundError):
            await store.skills.get(updated.version_id)


async def test_concurrent_replay_returns_one_complete_publication(scenario):
    """Serialize repeated execution of one run without duplicating saved versions."""
    service, members, calls, _ = scenario
    body = request(members).model_copy(update={"run_id": "evolve_" + uuid4().hex})
    first, second = await asyncio.gather(service.run_evolution(body), service.run_evolution(body))
    assert first == second and len(calls) == 1
    assert (await counts(service))[SKILL_VERSIONS_TABLE] == 5


async def test_evolve_route_resolves_service_after_factory_startup(scenario):
    """Use the service installed by lifespan rather than capturing its initial None value."""
    service, members, _, config = scenario

    async def factory():
        return service, config.fastapi

    app = create_app(service_factory=factory)
    assert app.state.service is None
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1") as client:
            response = await client.post(
                "/v1/skills/algos/evolve",
                json=request(members).model_dump(exclude={"project_id", "pipeline", "run_id"}, exclude_none=True),
            )
            assert response.status_code == 202, response.text
            task_id = response.json()["data"]["task_id"]
            for _ in range(100):
                task = (await client.post("/v1/skills/tasks/show", json={"task_id": task_id})).json()["data"]
                if task["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.01)
            assert task["status"] == "succeeded", task


async def test_single_and_collection_evolution_share_sdk_route_and_task_kind(scenario, monkeypatch):
    """Keep default single-Skill calls while admitting collections through the same API."""
    service, members, calls, config = scenario
    monkeypatch.setattr(Pipeline, "_registry", Pipeline.registered_types())
    single_calls = []

    class SingleConfig(BaseModel):
        """Configure the test's single-Skill evolution implementation."""

    @Pipeline.register("test_original_single_evolution")
    class SinglePipeline(EvolutionPipeline):
        """Return a stored version without invoking an external optimizer."""

        config_type = SingleConfig
        """No extra runtime parameters are required by this test implementation."""

        async def evolve(self, request):
            request = SkillPipelineRequest.model_validate(request)
            single_calls.append(request)
            version = await self.persistence.skills.get(request.base_version_id)
            return AlgoSkillEvolveResult(
                skill_id=version.skill_id,
                base_version_id=version.version_id,
                version_id=version.version_id,
                run_id=request.run_id,
                content=version.content,
                resources=version.resources,
                content_hash=version.content_hash,
                skill_hash=version.skill_hash,
            )

    single = SinglePipeline(SingleConfig(), persistence=service._persistence)
    monkeypatch.setitem(service._evolution_pipelines, "custom", single)
    monkeypatch.setitem(service._backends, "evolution_pipelines.custom", single)
    config.fastapi.evolution_pipelines = {"custom", "offline"}
    config.fastapi.default_evolution_pipeline = "custom"
    app = create_app(service, api_config=config.fastapi)
    assert "/v1/skills/algos/evolve/offline" not in app.openapi()["paths"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1") as http:
        client = AsyncSkillClient(
            AsyncHttpTransport(base_url="http://127.0.0.1", api_key=None, client=http, require_api_key=False)
        )

        async def completed(task):
            for _ in range(500):
                state = await client.show_task(TaskShowRequest(task_id=task["task_id"]))
                if state["status"] in {"succeeded", "failed"}:
                    assert state["status"] == "succeeded", state
                    assert state["kind"] == "skill.evolve"
                    return state["result"]
                await asyncio.sleep(0.01)
            raise AssertionError("Evolution did not finish")

        original = await completed(
            await client.evolve(EvolveRequest(skill_version_id=members[0].base.version_id, auto_accept=False))
        )
        assert original["version_id"] == members[0].base.version_id
        assert len(single_calls) == 1 and not calls
        collection = await completed(
            await client.evolve(
                EvolveRequest(
                    pipeline="offline",
                    skill_version_ids=[member.base.version_id for member in members],
                    editable_skills=[member.owner.skill_id for member in members[:2]],
                    auto_accept=False,
                )
            )
        )
        assert len(collection["members"]) == 4
        assert len(collection["skills"]) == 3 and len(calls) == 1
        invalid = await http.post(
            "/v1/skills/algos/evolve",
            json={
                "pipeline": "custom",
                "skill_version_ids": [members[0].base.version_id],
                "auto_accept": False,
            },
        )
        assert invalid.status_code == 422
        config.fastapi.default_evolution_pipeline = "offline"
        scalar_offline = await completed(
            await client.evolve(EvolveRequest(skill_version_id=members[0].base.version_id, auto_accept=False))
        )
        assert len(scalar_offline["members"]) == 2
        assert len(scalar_offline["skills"]) == 1 and len(calls) == 2
