"""Local execution and durable Skill task recovery contracts."""

import asyncio
import os
from datetime import timedelta
from uuid import uuid4

import httpx
import jwt
import pytest
from mindmemos_skill.api.auth import SkillHttpIdentity
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.factory import create_service
from mindmemos_skill.api.typing import RegisterRequest, StructuredSkillInfo, TrainingRequest
from mindmemos_skill.config.config import HttpAuthConfig
from mindmemos_skill.envs import Environment, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.errors import ProjectTaskBusyError
from mindmemos_skill.infra.database import PostgresConfig, PostgresDatabase, Scope
from mindmemos_skill.infra.database.builtin_backends.sqlite import SQLiteDatabase
from mindmemos_skill.infra.database.config import SQLiteConfig
from mindmemos_skill.infra.model_access.platform import PLATFORM_MODEL_SECRET_ENV, platform_connection
from mindmemos_skill.infra.task_manager import LocalTaskManager, TaskSpec, TaskStatus
from mindmemos_skill.infra.task_manager.kafka import KafkaTaskManager
from mindmemos_skill.infra.task_manager.typing import EVOLVE_TASK, GRAPH_BUILD_TASK, JUDGE_TASK
from mindmemos_skill.logging import get_logger
from mindmemos_skill.persistence.config import SkillPersistenceConfig
from mindmemos_skill.persistence.db_services.tasks import TaskWorkerService
from mindmemos_skill.persistence.db_services.tasks import admission as task_admission
from mindmemos_skill.persistence.store import SkillPersistence
from mindmemos_skill.pipelines import AlgoSkillEvolveResult, EvolutionPipeline, Pipeline, TrainingPipeline
from pydantic import AnyHttpUrl, BaseModel

from .service_config import deployment_config
from .test_infra import POSTGRES_TEST_URL


async def finished_task(manager, project_id, task_id):
    """Wait until terminal state guarantees that captured logs have drained."""

    async def poll():
        while True:
            snapshot = await manager.get(project_id, task_id)
            if snapshot.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                return snapshot
            await asyncio.sleep(0)

    return await asyncio.wait_for(poll(), timeout=2)


async def test_local_failure_retains_message_traceback_and_logs(caplog):
    async def handler(spec):
        raise ValueError("Invalid training input")

    manager = LocalTaskManager(handler)
    spec = TaskSpec(task_id="failed-local", project_id="p1", kind="skill.train", payload={})
    try:
        await manager.submit(spec)
        snapshot = await finished_task(manager, "p1", spec.task_id)
        assert snapshot.status == TaskStatus.FAILED
        assert snapshot.error_type == "ValueError"
        logs = await manager.logs("p1", spec.task_id)
        failure = next(log for log in logs if log.payload["traceback"])
        assert "ValueError: Invalid training input" in failure.payload["traceback"]
        assert "Traceback" in failure.payload["traceback"]
        assert any(record.exc_text and record.task_id == spec.task_id for record in caplog.records)
    finally:
        await manager.close()


async def test_kafka_failure_diagnostics_survive_storage_reopen(tmp_path):
    path = tmp_path / "failure.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
    store = TaskWorkerService(database)

    async def handler(spec):
        raise ValueError("Training configuration changed after task submission")

    manager = KafkaTaskManager(store=store, bootstrap_servers="unused", topic="test", group_id="test", handler=handler)
    spec = TaskSpec(task_id="failed-kafka", project_id="p1", kind="skill.train", payload={})
    try:
        await store.create(spec)
        await manager._execute(spec.task_id)
    finally:
        await manager.close()
        await database.close()
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    try:
        await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage()
        record = await TaskWorkerService(database).scoped(Scope(fields={"project_id": "p1"})).get(spec.task_id)
        snapshot = record.snapshot()
        assert snapshot.status == TaskStatus.FAILED
        logs = await TaskWorkerService(database).scoped(Scope(fields={"project_id": "p1"})).logs(spec.task_id)
        failure = next(log for log in logs if log.payload["traceback"])
        assert "ValueError: Training configuration changed" in failure.payload["traceback"]
        assert failure.payload["body"]["stage"] == "failed"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_local_tasks_use_standard_logs_without_persistence():
    async def handler(spec):
        get_logger(__name__).info(
            "algorithm.progress",
            extra={"event_body": {"stage": "collect", "message": "Collected one sample", "completed": 1, "total": 1}},
        )
        return {"task_id": spec.task_id}

    manager = LocalTaskManager(handler)
    spec = TaskSpec(task_id="local-1", project_id="p1", kind="skill.train", payload={})
    assert (await manager.submit(spec)).status == TaskStatus.QUEUED
    snapshot = await finished_task(manager, "p1", spec.task_id)
    assert snapshot.status == TaskStatus.SUCCEEDED
    assert snapshot.result == {"task_id": spec.task_id}
    assert await manager.get("p2", spec.task_id) is None
    assert any(log.payload.get("body", {}).get("stage") == "collect" for log in await manager.logs("p1", spec.task_id))
    await manager.close()


@pytest.mark.asyncio
async def test_one_service_selects_environment_and_restores_queued_samples(tmp_path):
    """Select two request-owned environments and preserve their task types through task JSON."""

    class Options(BaseModel):
        """No resources are needed by the test environments."""

    class FirstTask(Task):
        source: str

    class SecondTask(Task):
        choices: list[str]

    class FirstReward(RewardData):
        answer: str

    class SecondReward(RewardData):
        correct_index: int

    @Environment.register("queued_first")
    class FirstEnvironment(Environment[Options]):
        config_type = Options
        task_type = FirstTask
        reward_data_type = FirstReward

        async def _run(self, *, task, reward_data, skills, workspace):
            raise NotImplementedError

    @Environment.register("queued_second")
    class SecondEnvironment(Environment[Options]):
        config_type = Options
        task_type = SecondTask
        reward_data_type = SecondReward

        async def _run(self, *, task, reward_data, skills, workspace):
            raise NotImplementedError

    calls = []

    @Pipeline.register("queued_environment_probe")
    class TrainingProbe(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request, *, env, adapter=None):
            calls.append(
                (
                    type(env),
                    type(request.tasks[0].task),
                    type(request.tasks[0].reward_data),
                    adapter,
                    request.auto_accept,
                )
            )
            return AlgoSkillEvolveResult(
                skill_id="skill",
                base_version_id=request.init_skill_version_id,
                version_id="version-2",
                run_id="run-1",
                content="instructions",
                resources={},
                content_hash="test",
                skill_hash="test",
            )

    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'skill.db'}"}}
            },
            "task_manager": {"mode": "local"},
            "training_pipelines": {"probe": {"class_name": "queued_environment_probe"}},
        },
        "p1",
    )
    service = await create_service(config)
    try:
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            for name, task, reward in (
                ("first", {"source": "passage"}, {"answer": "yes"}),
                ("second", {"choices": ["yes", "no"]}, {"correct_index": 0}),
            ):
                payload = {
                    "project_id": "p1",
                    "algo": "probe",
                    "env": {"class_name": f"queued_{name}"},
                    "init_skill_version_id": "version-1",
                    "auto_accept": name != "first",
                    "work_dir": str(tmp_path / name),
                    "tasks": [{"task": {"id": name, "instruction": "Solve", **task}, "reward_data": reward}],
                    "split": {"train": [name]},
                }
                if name == "first":
                    response = await client.post("/v1/skills/algos/train", json=payload)
                    assert response.status_code == 202, response.text
                    task_id = response.json()["data"]["task_id"]
                else:
                    task_id = (await service.submit_training(payload)).task_id
                for _ in range(50):
                    status = await service.task_status("p1", task_id)
                    if status.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
                        break
                    await asyncio.sleep(0.01)
                assert status.status == TaskStatus.SUCCEEDED
        assert calls == [
            (FirstEnvironment, FirstTask, FirstReward, None, False),
            (SecondEnvironment, SecondTask, SecondReward, None, True),
        ]
        changed = TaskSpec(
            task_id="changed-config",
            project_id="p1",
            kind="skill.train",
            payload={"request": payload, "config_fingerprint": "outdated"},
        )
        with pytest.raises(ValueError, match="configuration changed"):
            await service.execute_task(changed)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_submits_same_training_pipeline_with_progress(tmp_path):
    class Options(BaseModel):
        """Test pipeline has no external dependencies."""

    @Environment.register("async_training_probe_env")
    class TrainingEnvironment(Environment[Options]):
        """Accept the basic task and reward types used by the queued request."""

        config_type = Options

        async def _run(self, *, task, reward_data, skills, workspace):
            raise NotImplementedError

    @Pipeline.register("async_training_probe")
    class TrainingProbe(TrainingPipeline[Options]):
        """Record one algorithm-controlled progress boundary."""

        config_type = Options

        async def train(self, request, *, env, adapter=None):
            get_logger(__name__).info(
                "algorithm.progress", extra={"event_body": {"stage": "selection", "message": "Selected candidate"}}
            )
            return AlgoSkillEvolveResult(
                skill_id="skill",
                base_version_id=request.init_skill_version_id,
                version_id="version-2",
                run_id="run-1",
                content="instructions",
                resources={},
                content_hash="test",
                skill_hash="test",
            )

    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'skill.db'}"}}
            },
            "task_manager": {"mode": "local"},
            "training_pipelines": {"probe": {"class_name": "async_training_probe"}},
        },
        "p1",
    )
    service = await create_service(config)
    try:
        request = TrainingRequest.model_validate(
            {
                "project_id": "p1",
                "algo": "probe",
                "env": {"class_name": "async_training_probe_env"},
                "init_skill_version_id": "version-1",
                "work_dir": str(tmp_path / "run"),
                "tasks": [{"task": {"id": "t1", "instruction": "Solve"}, "reward_data": {}}],
                "split": {"train": ["t1"]},
            }
        )
        submitted = await service.submit_training(request)
        await asyncio.sleep(0)
        for _ in range(20):
            status = await service.task_status("p1", submitted.task_id)
            if status.status == TaskStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)
        assert status.status == TaskStatus.SUCCEEDED
        assert status.result["version_id"] == "version-2"
        assert any(
            log.payload.get("body", {}).get("stage") == "selection"
            for log in await service.task_progress("p1", submitted.task_id)
        )
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2])
async def test_http_evolution_accepts_task_before_algorithm_finishes(tmp_path, limit):
    """Keep Skill writes immediate while evolution runs behind a task identity."""
    started = asyncio.Event()
    release = asyncio.Event()
    choices = []
    pipeline_name = f"task_evolution_probe_{limit}"

    class Options(BaseModel):
        """The probe needs no external settings."""

    @Pipeline.register(pipeline_name)
    class EvolutionProbe(EvolutionPipeline[Options]):
        """Wait for explicit release after the HTTP request is accepted."""

        config_type = Options

        async def evolve(self, request):
            choices.append(request.auto_accept)
            get_logger(__name__).info(
                "algorithm.progress", extra={"event_body": {"stage": "evidence", "message": "Evolution task started"}}
            )
            started.set()
            await release.wait()
            version = await self.persistence.skills.get(request.skill_version_ids[0])
            return AlgoSkillEvolveResult(
                skill_id=version.skill_id,
                base_version_id=version.version_id,
                version_id="result-version",
                run_id="evolution-run",
                content="Evolved",
                resources={},
                content_hash="content-hash",
                skill_hash="skill-hash",
            )

    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'skill.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "task_manager": {"mode": "local"},
            "max_active_tasks_per_project": limit,
            "evolution_pipelines": {"probe": {"class_name": pipeline_name}},
        },
        "p1",
    )
    service = await create_service(config)
    try:
        app = create_app(service, api_config=config.fastapi.model_copy(update={"evolution_pipelines": {"probe"}}))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            registered = await service.register_skill(
                RegisterRequest(
                    project_id="p1",
                    skill=StructuredSkillInfo(
                        content="Example", name="Example", description="Example instructions", standard_format=False
                    ),
                )
            )
            version_id = registered.skill_version_id
            missing = await client.post("/v1/skills/algos/evolve", json={"skill_version_ids": ["missing"]})
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "skill_version_not_found"
            response = await client.post(
                "/v1/skills/algos/evolve", json={"skill_version_ids": [version_id], "auto_accept": False}
            )
            assert response.status_code == 202, response.text
            assert response.json()["code"] == "accepted"
            task_id = response.json()["data"]["task_id"]
            await asyncio.wait_for(started.wait(), timeout=2)
            assert choices == [False]
            assert (await service.task_status("p1", task_id)).result is None
            admitted_ids = {task_id}
            if limit == 2:
                second = await client.post("/v1/skills/algos/evolve", json={"skill_version_ids": [version_id]})
                assert second.status_code == 202, second.text
                admitted_ids.add(second.json()["data"]["task_id"])
            blocked = await client.post("/v1/skills/algos/evolve", json={"skill_version_ids": [version_id]})
            assert blocked.status_code == 429, blocked.text
            error = blocked.json()["error"]
            assert error["task_id"] in admitted_ids
            assert error["task_status"] in {"queued", "running"}
            assert error == {
                "code": "too_many_requests",
                "message": (
                    f"Evolve already has a {error['task_status']} task in this project (task ID: {error['task_id']})."
                ),
                "task_id": error["task_id"],
                "task_kind": "skill.evolve",
                "task_status": error["task_status"],
            }
            with pytest.raises(ValueError, match="default project"):
                await service.task_status("other", task_id)
            release.set()
            for _ in range(20):
                status = await service.task_status("p1", task_id)
                if status.status == TaskStatus.SUCCEEDED:
                    break
                await asyncio.sleep(0.01)
            assert status.status == TaskStatus.SUCCEEDED
            assert status.result["base_version_id"] == version_id
            assert status.result["skill_id"] == registered.skill_id
            response = await client.post("/v1/skills/tasks/progress", json={"task_id": task_id})
            assert response.status_code == 200
            logs = response.json()["data"]
            evidence = next(row for row in logs if (row["payload"].get("body") or {}).get("stage") == "evidence")
            assert evidence["attempt"] == 1 and evidence["task_id"] == task_id
            after = logs[-1]["sequence"]
            response = await client.post(
                "/v1/skills/tasks/progress", json={"task_id": task_id}, params={"after": after}
            )
            assert response.json()["data"] == []
            assert any(
                log.payload.get("body", {}).get("stage") == "evidence"
                for log in await service.task_progress("p1", task_id)
            )
    finally:
        release.set()
        await service.close()


async def test_task_tables_reopen_and_reclaim_expired_attempt(tmp_path):
    path = tmp_path / "existing-skill.db"

    async def open_store():
        database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
        (await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage())
        return database, TaskWorkerService(database)

    database, store = await open_store()
    spec = TaskSpec(task_id="cloud-1", project_id="p1", kind="skill.train", payload={"work_dir": "shared"})
    (await store.create(spec))
    (await database.close())

    database, store = await open_store()
    assert len((await store.due())) == 1
    first, token = await store.claim(spec.task_id, lease_seconds=0)
    assert first.attempt == 1
    assert (await store.claim(spec.task_id)) is not None
    assert not (await store.heartbeat(spec.task_id, token))
    assert not (await store.finish(spec.task_id, token, result={}))
    (await database.close())

    database, store = await open_store()
    assert (await store.scoped(Scope(fields={"project_id": "p2"})).get(spec.task_id)) is None
    assert (await store.scoped(Scope(fields={"project_id": "p1"})).get(spec.task_id)).attempt == 2
    (await database.close())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("limit", [1, 2])
async def test_shared_project_capacity_across_connections(backend, limit, tmp_path, monkeypatch):
    """Count queued work and individual reservations together across database clients."""
    if backend == "postgres":
        url = os.environ.get(POSTGRES_TEST_URL)
        if not url:
            pytest.skip(f"Set {POSTGRES_TEST_URL} to run real PostgreSQL contracts")
        first = PostgresDatabase(PostgresConfig(url=url))
        second = PostgresDatabase(PostgresConfig(url=url))
    else:
        url = f"sqlite:///{tmp_path / 'shared.db'}"
        first = SQLiteDatabase(SQLiteConfig(url=url))
        second = SQLiteDatabase(SQLiteConfig(url=url))
    project = uuid4().hex
    task_id = uuid4().hex
    reservations = []
    try:
        first_store = SkillPersistence(SkillPersistenceConfig(), database=first, max_active_tasks_per_project=limit)
        second_store = SkillPersistence(SkillPersistenceConfig(), database=second, max_active_tasks_per_project=limit)
        await first_store.initialize_storage()
        await second_store.initialize_storage()
        workers = (first_store.task_worker(), second_store.task_worker())
        worker = workers[1]
        gate = first_store.task_admission()
        spec = TaskSpec(task_id=task_id, project_id=project, kind=EVOLVE_TASK, payload={})

        async def submit(index):
            return await workers[index % len(workers)].create(spec.model_copy(update={"task_id": f"{task_id}-{index}"}))

        results = await asyncio.gather(*(submit(index) for index in range(5)), return_exceptions=True)
        accepted = [item for item in results if not isinstance(item, BaseException)]
        assert len(accepted) == limit, results
        task_ids = {task.task_id for task in accepted}
        for rejected in [item for item in results if isinstance(item, BaseException)]:
            assert isinstance(rejected, ProjectTaskBusyError)
            assert rejected.task_id in task_ids
        with pytest.raises(ProjectTaskBusyError) as blocked:
            await gate.reserve(project, "graph", GRAPH_BUILD_TASK, TaskStatus.RUNNING)
        assert blocked.value.task_id in task_ids and blocked.value.status == TaskStatus.QUEUED

        other = await gate.reserve(f"{project}-other", "judge", JUDGE_TASK, TaskStatus.RUNNING)
        await other.close()
        claims = [await worker.claim(task.task_id, lease_seconds=0) for task in accepted]
        with pytest.raises(ProjectTaskBusyError) as blocked:
            await gate.reserve(project, "judge", JUDGE_TASK, TaskStatus.RUNNING)
        assert blocked.value.task_id in task_ids and blocked.value.status == TaskStatus.RUNNING
        for task, token in claims:
            assert await worker.finish(task.task_id, token, error_type="TestFailure")

        for index in range(limit):
            reservations.append(
                await gate.reserve(project, f"{project}-graph-{index}", GRAPH_BUILD_TASK, TaskStatus.RUNNING)
            )
        with pytest.raises(ProjectTaskBusyError) as blocked:
            await worker.create(spec)
        assert blocked.value.kind == GRAPH_BUILD_TASK
        await reservations.pop().close()
        await worker.create(spec)
        with pytest.raises(ProjectTaskBusyError):
            await gate.reserve(project, "judge", JUDGE_TASK, TaskStatus.RUNNING)
        _, token = await worker.claim(spec.task_id)
        assert await worker.finish(spec.task_id, token, result={})

        monkeypatch.setattr(task_admission, "LEASE_DURATION", timedelta(0))
        reservations.append(await gate.reserve(project, f"{project}-expired", GRAPH_BUILD_TASK, TaskStatus.RUNNING))
        new_task_id = f"{task_id}-after-expiry"
        await worker.create(spec.model_copy(update={"task_id": new_task_id}))
        with pytest.raises(ProjectTaskBusyError) as blocked:
            await gate.reserve(project, "judge", JUDGE_TASK, TaskStatus.RUNNING)
        assert blocked.value.task_id in {new_task_id, f"{project}-graph-0"}
    finally:
        for reservation in reservations:
            await reservation.close()
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_kafka_submission_never_waits_for_broker_confirmation(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'skill.db'}"))
    (await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage())
    manager = KafkaTaskManager(
        store=TaskWorkerService(database), bootstrap_servers="unused:9092", topic="skill.train.jobs", group_id="skill"
    )
    confirmation = asyncio.get_running_loop().create_future()

    class Producer:
        async def send(self, topic, value, *, key, headers):
            assert topic == "skill.train.jobs"
            return confirmation

    manager._producer = Producer()
    manager._started = True
    spec = TaskSpec(task_id="cloud-2", project_id="p1", kind="skill.train", payload={})
    snapshot = await asyncio.wait_for(manager.submit(spec), timeout=1)
    assert snapshot.status == TaskStatus.QUEUED
    assert not confirmation.done()
    confirmation.set_result(None)
    await asyncio.sleep(0)
    manager._producer = None
    manager._started = False
    (await database.close())


@pytest.mark.asyncio
async def test_kafka_recovery_republishes_unconfirmed_database_task(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'skill.db'}"))
    (await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage())
    store = TaskWorkerService(database)
    spec = TaskSpec(task_id="recover-1", project_id="p1", kind="skill.train", payload={})
    (await store.create(spec))
    calls = []

    class Producer:
        async def send(self, topic, value, *, key, headers):
            calls.append(key)
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

    manager = KafkaTaskManager(
        store=store,
        bootstrap_servers="unused:9092",
        topic="skill.train.jobs",
        group_id="skill",
        scan_interval_seconds=0.01,
    )
    manager._producer = Producer()
    scanning = asyncio.create_task(manager._scan())
    try:
        for _ in range(20):
            if calls:
                break
            await asyncio.sleep(0.01)
        assert calls == [b"recover-1"]
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get(spec.task_id)).status == TaskStatus.QUEUED
    finally:
        scanning.cancel()
        await asyncio.gather(scanning, return_exceptions=True)
        (await database.close())


@pytest.mark.asyncio
async def test_kafka_disconnected_submission_remains_recoverable(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'skill.db'}"))
    (await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage())
    store = TaskWorkerService(database)
    manager = KafkaTaskManager(store=store, bootstrap_servers="unused:9092", topic="skill.train.jobs", group_id="skill")
    manager._started = True
    spec = TaskSpec(task_id="disconnected-1", project_id="p1", kind="skill.train", payload={})
    assert (await manager.submit(spec)).status == TaskStatus.QUEUED
    assert [item.task_id for item in (await store.due())] == [spec.task_id]
    (await database.close())


@pytest.mark.asyncio
async def test_background_training_preserves_keys_after_database_reopen(tmp_path):
    """Restore explicit nested model credentials from durable task JSON."""
    from mindmemos_skill.agents import OpenAIChatConfig

    from .test_training_request import search_request

    class Options(BaseModel):
        model: OpenAIChatConfig

    received = []

    @Pipeline.register("credential_roundtrip_probe")
    class Probe(TrainingPipeline[Options]):
        config_type = Options

        async def train(self, request, *, env, adapter=None):
            received.append((env.config.agent.api_key, self.config.model.api_key))
            return AlgoSkillEvolveResult(
                skill_id="skill",
                base_version_id=request.init_skill_version_id,
                version_id="version",
                run_id="run",
                content="instructions",
                resources={},
                content_hash="hash",
                skill_hash="hash",
            )

    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'skill.db'}"}}
            },
            "training_pipelines": {
                "probe": {
                    "class_name": "credential_roundtrip_probe",
                    "config": {
                        "model": {
                            "model": "offline",
                            "api_key": "algorithm-test-key",
                            "base_url": "https://test.invalid/v1",
                        }
                    },
                }
            },
        },
        "p1",
    )
    service = await create_service(config)
    database = service.backends["database"]
    manager = KafkaTaskManager(
        store=TaskWorkerService(database), bootstrap_servers="unused", topic="test", group_id="test"
    )
    manager._started = True
    service.attach_task_manager(manager)
    payload = search_request()
    payload.update(project_id="p1", algo="probe", work_dir=str(tmp_path / "run"))
    payload["env"]["config"]["agent"].update(api_key="environment-test-key", base_url="https://test.invalid/v1")
    submitted = await service.submit_training(payload)
    await service.close()
    restored = await create_service(config)
    try:
        record = (
            await TaskWorkerService(restored.backends["database"])
            .scoped(Scope(fields={"project_id": "p1"}))
            .get(submitted.task_id)
        )
        assert record.payload["request"]["env"]["config"]["agent"]["api_key"] == "environment-test-key"
        await restored.execute_task(record.spec())
        assert received == [("environment-test-key", "algorithm-test-key")]
    finally:
        await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("via_platform", [False, True])
async def test_evolution_uses_submitted_configuration_after_worker_restart(tmp_path, monkeypatch, via_platform):
    """Freeze the complete pipeline settings instead of adopting worker changes."""
    from mindmemos_skill.agents import OpenAIChatConfig

    class Options(BaseModel):
        model: OpenAIChatConfig

    received = []
    closed = []
    routes = []
    worker_secret = "test-worker-model-signing-key".ljust(32, "-")
    probe_name = "frozen_gateway_evolution_probe" if via_platform else "frozen_evolution_probe"

    @Pipeline.register(probe_name)
    class Probe(EvolutionPipeline[Options]):
        config_type = Options

        async def evolve(self, request):
            received.append((self.config.model.api_key, request.run_id))
            if self.config.model.transport == "platform_gateway":
                url, token = platform_connection()
                claims = jwt.decode(token, worker_secret, algorithms=["HS256"], audience="platform-skill-model")
                routes.append((url, claims["project_id"]))
            version = await self.persistence.skills.get(request.skill_version_ids[0])
            return AlgoSkillEvolveResult(
                skill_id=version.skill_id,
                base_version_id=version.version_id,
                version_id="version",
                run_id=request.run_id,
                content="instructions",
                resources={},
                content_hash="hash",
                skill_hash="hash",
            )

        async def close(self):
            closed.append(self.config.model.api_key)

    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'skill.db'}"}}
            },
            "observability": {
                "backend": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'audit.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "evolution_pipelines": {
                "probe": {
                    "class_name": probe_name,
                    "config": {
                        "model": {
                            "model": "offline",
                            "transport": "platform_gateway" if via_platform else "direct",
                            "api_key": "submitted-test-key",
                            "base_url": "https://test.invalid/v1",
                        }
                    },
                }
            },
        },
        "p1",
    )
    if via_platform:
        config.service.platform_model_origin = AnyHttpUrl("http://platform.invalid")
        config.fastapi.auth = HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret="inbound-test-key".ljust(32, "-"))
        monkeypatch.setenv(PLATFORM_MODEL_SECRET_ENV, "test-submit-model-signing-key".ljust(32, "-"))
    service = await create_service(config)
    manager = KafkaTaskManager(
        store=TaskWorkerService(service.backends["database"]),
        bootstrap_servers="unused",
        topic="test",
        group_id="test",
    )
    manager._started = True
    service.attach_task_manager(manager)
    with service.request_principal(SkillHttpIdentity(project_id="p1")):
        registered = await service.register_skill(
            RegisterRequest(
                project_id="p1",
                skill=StructuredSkillInfo(
                    content="Instructions",
                    name="Example",
                    description="Example",
                    standard_format=False,
                ),
            )
        )
        submitted = await service.submit_evolution("p1", [registered.skill_version_id])
    await service.close()
    worker_options = Options.model_validate(config.service.evolution_pipelines["probe"].config)
    worker_options.model.api_key = "worker-test-key"
    config.service.evolution_pipelines["probe"].config = worker_options
    if via_platform:
        monkeypatch.setenv(PLATFORM_MODEL_SECRET_ENV, worker_secret)
    restored = await create_service(config)
    try:
        record = (
            await TaskWorkerService(restored.backends["database"])
            .scoped(Scope(fields={"project_id": "p1"}))
            .get(submitted.task_id)
        )
        assert record.payload["pipeline"]["config"]["model"]["api_key"] == "submitted-test-key"
        result = await restored.execute_task(record.spec())
        assert routes == ([("http://platform.invalid/internal/v1/skill-models", "p1")] if via_platform else [])
        assert received == [("submitted-test-key", result["run_id"])]
        assert result["run_id"].startswith("evolve_")
        audit = restored.backends["observability"]
        run = await audit.queries(Scope(fields={"project_id": "p1"})).get_run(result["run_id"])
        assert run.status == "completed"
        assert run.base_version_id == registered.skill_version_id
        assert run.published_version_id == "version"
        assert any(
            span.name == "algorithm.evolve"
            for span in (await audit.queries(Scope(fields={"project_id": "p1"})).spans(result["run_id"]))
        )
        assert closed == ["submitted-test-key", "submitted-test-key"]
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_local_queue_limits_execution_and_cleans_cancelled_submissions():
    """Queued work stays queued and full-buffer submissions remain cancellable."""
    started = asyncio.Event()
    calls = []

    async def handler(spec):
        calls.append(spec.task_id)
        started.set()
        await asyncio.Event().wait()

    manager = LocalTaskManager(handler, max_concurrency=1, max_buffered=2)

    def spec(task_id):
        return TaskSpec(task_id=task_id, project_id="p1", kind="test", payload={})

    try:
        await manager.submit(spec("running"))
        await started.wait()
        await manager.submit(spec("queued"))
        pending = asyncio.create_task(manager.submit(spec("waiting")))
        await asyncio.sleep(0)
        assert not pending.done()
        assert (await manager.get("p1", "queued")).status == TaskStatus.QUEUED
        assert (await manager.get("p1", "queued")).attempt == 0
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert await manager.get("p1", "waiting") is None
        closing_waiter = asyncio.create_task(manager.submit(spec("closing")))
        await asyncio.sleep(0)
        await manager.close()
        with pytest.raises(RuntimeError, match="closed"):
            await closing_waiter
        assert await manager.get("p1", "closing") is None
        assert calls == ["running"]
        for task_id in ("running", "queued"):
            assert (await manager.get("p1", task_id)).status == TaskStatus.CANCELLED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_kafka_consumer_and_recovery_share_slots_without_early_claims(tmp_path):
    """Consume while full, coalesce duplicates, and recover deferred database work."""
    import json
    from types import SimpleNamespace

    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'queue.db'}"))
    (await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage())
    store = TaskWorkerService(database)
    for task_id in ("consumer", "expired", "overflow"):
        (await store.create(TaskSpec(task_id=task_id, project_id="p1", kind="test", payload={})))
    (await store.claim("expired", lease_seconds=0))
    started = asyncio.Queue()
    release = asyncio.Event()
    overflow_done = asyncio.Event()
    active = 0
    peak = 0
    calls = []

    async def handler(spec):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(spec.task_id)
        started.put_nowait(spec.task_id)
        try:
            await release.wait()
            if spec.task_id == "overflow":
                overflow_done.set()
            return {"task_id": spec.task_id}
        finally:
            active -= 1

    messages = asyncio.Queue()
    committed = asyncio.Queue()

    class Consumer:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return SimpleNamespace(value=json.dumps({"task_id": await messages.get()}).encode())

        async def commit(self):
            committed.put_nowait(True)

        async def stop(self):
            pass

    class Producer:
        async def send(self, topic, value, *, key, headers):
            messages.put_nowait(json.loads(value)["task_id"])
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

        async def stop(self):
            pass

    manager = KafkaTaskManager(
        store=store,
        bootstrap_servers="unused",
        topic="test",
        group_id="test",
        handler=handler,
        max_concurrency=1,
        max_buffered=2,
        scan_interval_seconds=0.01,
    )
    manager._consumer = Consumer()
    manager._consumer_task = asyncio.create_task(manager._consume())
    try:
        messages.put_nowait("consumer")
        assert await asyncio.wait_for(started.get(), 1) == "consumer"
        manager._spawn(manager._scan(), name="test-recovery")
        # Both sources submit the same expired task; only one local item is retained.
        messages.put_nowait("expired")
        messages.put_nowait("expired")
        messages.put_nowait("overflow")
        for _ in range(4):
            await asyncio.wait_for(committed.get(), 1)
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get("expired")).attempt == 1
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get("overflow")).attempt == 0
        assert calls == ["consumer"]
        assert started.empty()
        release.set()
        assert await asyncio.wait_for(started.get(), 1) == "expired"
        # Unadmitted work is still queued in the DB and the normal scan republishes it.
        manager._producer = Producer()
        await asyncio.wait_for(overflow_done.wait(), 2)
        assert peak == 1
        assert calls == ["consumer", "expired", "overflow"]
    finally:
        await manager.close()
        (await database.close())


@pytest.mark.asyncio
async def test_kafka_failed_claims_release_slots_and_close_leaves_work_recoverable(tmp_path):
    """Missing and terminal tasks cannot consume slots or advance queued leases."""
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'claims.db'}"))
    (await SkillPersistence(SkillPersistenceConfig(), database=database).initialize_storage())
    store = TaskWorkerService(database)
    for task_id in ("done", "running", "queued"):
        (await store.create(TaskSpec(task_id=task_id, project_id="p1", kind="test", payload={})))
    _, token = await store.claim("done")
    (await store.finish("done", token, result={}))
    started = asyncio.Event()
    calls = []

    async def handler(spec):
        calls.append(spec.task_id)
        started.set()
        await asyncio.Event().wait()

    manager = KafkaTaskManager(
        store=store,
        bootstrap_servers="unused",
        topic="test",
        group_id="test",
        handler=handler,
        max_concurrency=1,
        max_buffered=4,
    )
    try:
        for task_id in ("missing", "done", "running", "queued"):
            assert await manager._dispatch(task_id)
        await asyncio.wait_for(started.wait(), 1)
        assert calls == ["running"]
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get("queued")).attempt == 0
        await manager.close()
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get("running")).status == TaskStatus.RUNNING
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get("queued")).status == TaskStatus.QUEUED
        assert (await store.scoped(Scope(fields={"project_id": "p1"})).get("queued")).attempt == 0
        with pytest.raises(RuntimeError, match="closed"):
            await manager._dispatch("new")
    finally:
        await manager.close()
        (await database.close())
