"""Exercise matched evolution evaluation without model calls."""

import pytest
from mindmemos_eval.cli import main
from mindmemos_eval.skills import service
from mindmemos_eval.skills.algos import offline_evolution as evolve
from mindmemos_eval.skills.config import SkillEvolveConfig
from mindmemos_skill.api.typing import EvolutionPublishedSkill, EvolutionRunResult, RegisterResult
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.typing import Reward, RolloutResult, Usage
from mindmemos_skill.skill import Skill, SkillContent, decode_resources
from pydantic import BaseModel


class FixtureConfig(BaseModel):
    """Configure deterministic evaluation fixtures."""


@Environment.register("evolve_fixture")
class FixtureEnvironment(Environment[FixtureConfig]):
    """Produce deterministic successful, timed-out, and retryable results."""

    config_type = FixtureConfig
    """Fixture configuration accepted by the registry."""

    calls = {}
    """Execution counts keyed by task identity."""

    async def _run(self, *, task, reward_data, skills, workspace):
        self.calls[task.id] = self.calls.get(task.id, 0) + 1
        if task.id == "timeout":
            raise RuntimeError("Harbor AgentTimeoutError: timed out")
        if task.id == "setup":
            raise RuntimeError("Harbor AgentSetupTimeoutError: timed out")
        if task.id == "network" and self.calls[task.id] == 1:
            raise RuntimeError("API error: connection closed")
        score = float(task.id == "perfect" or any("Improved" in skill.content for skill in skills))
        return RolloutResult(
            output="answer",
            trajectory=[{"role": "user", "content": task.instruction}, {"role": "assistant", "content": "answer"}],
            reward=Reward(score=score, success=score == 1, detail="fixture"),
            usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12),
        )


class FakeService:
    """Track exact-source registrations and publication actions."""

    def __init__(self, mode):
        self.mode = mode
        self.skills = {}
        self.reports = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def register_skill(self, request):
        source = request.skill
        skill = Skill(
            SkillContent(
                content=source.content,
                resources=decode_resources(source.resources),
                name=source.name,
                description=source.description,
                standard_format=source.standard_format,
            )
        )
        self.skills[request.alias] = skill
        snapshot = skill.snapshot()
        return RegisterResult(
            skill_id=request.alias,
            skill_version_id=request.alias,
            content_hash=snapshot.content_hash,
            skill_hash=snapshot.skill_hash,
        )

    async def report_trajectory(self, request):
        self.reports.append((request.skill_version_ids, request.messages))

    async def run_evolution(self, request):
        assert request.pipeline == "offline"
        assert request.auto_accept is False
        if self.mode == "fail":
            raise RuntimeError("fixture evolution failure")
        members = []
        for version_id in request.skill_version_ids:
            skill = self.skills[version_id]
            unchanged = self.mode == "unchanged"
            candidate = skill if unchanged else skill.with_content(skill.content + "\nImproved\n")
            members.append(
                EvolutionPublishedSkill(
                    key=version_id,
                    skill_id=version_id,
                    version_id="new",
                    base_version_id=version_id,
                    action="unchanged" if unchanged else "updated",
                    snapshot=candidate.snapshot(),
                )
            )
        return EvolutionRunResult(
            run_id="evolve_" + "0" * 32,
            input_version_ids={version_id: version_id for version_id in request.skill_version_ids},
            skills={member.skill_id: member.snapshot for member in members},
            members=members,
        )


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    FixtureEnvironment.calls = {}
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: fixture\ndescription: Fixture skill\n---\nInitial\n")
    (skill / "resource.bin").write_bytes(b"\x00\xff")
    samples = tmp_path / "samples.json"
    write_json(
        samples,
        [
            {"task": {"id": key, "instruction": "Answer"}, "reward_data": {}}
            for key in ("perfect", "zero", "timeout", "setup", "network")
        ],
    )
    fake_service = FakeService("updated")
    service_config = tmp_path / "service.yaml"
    write_json(
        service_config,
        {
            "service": {
                "home": str(tmp_path / "unused-service-home"),
                "evolution_pipelines": {
                    "offline": {
                        "class_name": "offline_evolution",
                        "config": {
                            "work_dir": "unused-work-dir",
                            "evolver": {"class_name": "offline_evolution"},
                            "traj_collect": {"max_page_size": 23},
                        },
                    },
                },
            },
            "fastapi": {"auth": {"mode": "none", "default_project_id": "unused"}},
        },
    )
    config = SkillEvolveConfig(
        environment={"class_name": "evolve_fixture"},
        samples_file=samples,
        skills=[skill],
        repeats=1,
        service_config=service_config,
        pipeline_overrides={"traj_collect": {"trajectory_limit": 5}},
        output_dir=tmp_path / "run",
    )

    async def create_service(deployment):
        resources = deployment.service
        assert resources.home == config.output_dir / "state"
        options = resources.evolution_pipelines["offline"].config
        assert options["work_dir"] == str(config.output_dir / "evolution_work")
        assert options["traj_collect"] == {"max_page_size": 23, "trajectory_limit": 5, "exclude_consumed": True}
        return fake_service

    monkeypatch.setattr(service, "create_service", create_service)
    return config, fake_service


def test_evaluate_evolve_check_prepares_selected_tasks(setup_run, tmp_path, capsys):
    config, _ = setup_run
    config_file = tmp_path / "evolve.yaml"
    write_json(config_file, config)
    assert (
        main(["skill", "evaluate", "evolve", "--algo", "offline-evolv", "--config", str(config_file), "--check"]) == 0
    )
    assert "Prepared 5 cases" in capsys.readouterr().out
    assert not config.output_dir.exists()


def test_eval_pipeline_can_enable_consumed_evidence_reuse(setup_run):
    config, _ = setup_run
    config.pipeline_overrides["traj_collect"]["exclude_consumed"] = False
    resources = evolve._evolution_resources(config, config.output_dir)
    assert resources.evolution_pipelines["offline"].config["traj_collect"] == {
        "max_page_size": 23,
        "trajectory_limit": 5,
        "exclude_consumed": False,
    }


@pytest.mark.asyncio
async def test_end_to_end_and_resume(setup_run):
    config, service = setup_run
    result = await evolve.run(config)
    rows = {row["case"]: row for row in result["cases"]}
    assert rows["perfect"]["status"] == "perfect"
    assert rows["timeout"]["collect_stats"]["mean_score"] == 0
    assert rows["setup"]["status"] == "excluded"
    assert rows["zero"]["test_stats"]["mean_score"] == 1
    assert FixtureEnvironment.calls["network"] == 3
    assert result["metrics"]["collect"]["mean_score"] == 0.25
    assert result["metrics"]["test"]["mean_score"] == 0.75
    assert len(service.reports) == 2
    initial = Skill.from_path(config.skills[0])
    assert all(skill.source == initial.source for skill in service.skills.values())
    previous = dict(FixtureEnvironment.calls)
    config_path = config.output_dir / "config.json"
    recorded = read_json(config_path)
    write_json(config_path, {**recorded, "connection": "inprocess", "sdk_home": None, "poll_interval": 5})
    await evolve.run(config)
    assert FixtureEnvironment.calls == previous
    assert read_json(config_path) == recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fail", "unchanged"])
async def test_fallback_without_test_calls(setup_run, mode):
    config, service = setup_run
    config.cases = ["zero"]
    service.mode = mode
    result = await evolve.run(config)
    assert result["cases"][0]["status"] == ("evolution_failed" if mode == "fail" else "unchanged")
    assert result["errors"] == int(mode == "fail")
    assert FixtureEnvironment.calls["zero"] == 1
    assert result["metrics"]["test"]["mean_score"] == 0


@pytest.mark.parametrize(
    ("mode", "cases", "expected_errors"),
    [
        ("fail", ["zero"], 1),
        ("fail", ["zero", "network"], 2),
        ("unchanged", ["zero"], 0),
        ("updated", ["zero"], 0),
        ("updated", ["perfect", "timeout", "setup"], 0),
    ],
)
def test_cli_exit_status_includes_evolution_failures_on_resume(setup_run, tmp_path, mode, cases, expected_errors):
    config, service = setup_run
    config.cases = cases
    config.network_retries = 0
    service.mode = mode
    config_file = tmp_path / "evolve.yaml"
    write_json(config_file, config)
    args = ["skill", "evaluate", "evolve", "--config", str(config_file)]

    assert main(args) == int(expected_errors > 0)
    summary = read_json(config.output_dir / "summary.json")
    assert summary["errors"] == expected_errors
    if mode == "fail":
        assert summary["cases"][0]["status"] == "evolution_failed"
        assert summary["cases"][0]["collect"][0]["status"] == "completed"
        assert "fixture evolution failure" in summary["cases"][0]["evolution_error"]
    assert f"execution errors: {expected_errors}" in (config.output_dir / "report.md").read_text()

    previous_calls = dict(FixtureEnvironment.calls)
    previous_reports = len(service.reports)
    service.mode = "updated"
    assert main(args) == int(expected_errors > 0)
    assert read_json(config.output_dir / "summary.json")["errors"] == expected_errors
    assert FixtureEnvironment.calls == previous_calls
    assert len(service.reports) == previous_reports


def test_sanitize_and_error_categories():
    original = [{"content": "x" * 450}]
    assert evolve.sanitize_trajectory(original)[0]["content"] == "[omitted binary]"
    assert original[0]["content"] == "x" * 450
    assert evolve.classify_error("VerifierTimeoutError") == "timeout"
    assert evolve.classify_error("AgentAuthenticationError") == "retryable"
    assert evolve.classify_error("unexpected failure") == "error"


@pytest.mark.parametrize(
    "error",
    [
        "InternalServerError: Error code: 503 - {'error': {'code': 'model_not_found'}}",
        "RateLimitError: Error code: 429 - too many requests",
        "upstream returned HTTP 502",
        "API Error: 503 - service unavailable",
        "status code: 503",
        "HTTP status code: 503",
        "response status code: 429",
        "HTTP/1.1 503 Service Unavailable",
        "HTTP 401 unauthorized",
        "HTTP 403 forbidden",
        "HTTP 408 request timeout",
        "RuntimeError: Harbor NetworkConnectionError: Command failed (exit 28)",
        "Harness exited with status 1: API Error: Connection closed mid-response.",
        "所有分组均无可用渠道",
    ],
)
def test_transient_errors_are_retryable(error):
    assert evolve.classify_error(error) == "retryable"


@pytest.mark.parametrize(
    "error",
    [
        "BadRequestError: Error code: 400 - invalid request",
        "upstream returned HTTP 404",
        "API Error: 400 - invalid request",
        "API Error: 404 - not found",
        "HTTP 400 authentication field is required",
        "HTTP 404: API Error",
        "HTTP 422 authentication field is required",
        "status code: 400 connection reset",
        "HTTP/1.1 404 Not Found",
        "HTTP 503 wrapping API Error: 400 - invalid request",
        "ValueError: missing connection string in task config",
    ],
)
def test_request_errors_are_not_retried(error):
    assert evolve.classify_error(error) == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["register_skill", "report_trajectory"])
async def test_evolution_retry_resumes_failed_step(setup_run, tmp_path, monkeypatch, step):
    config, service = setup_run
    config.cases = ["zero"]
    other = tmp_path / "second"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: second\ndescription: Second fixture\n---\nInitial\n")
    config.skills.append(other)
    calls = []
    original = service.register_skill if step == "register_skill" else service.report_trajectory

    async def flaky(*args):
        calls.append(args)
        failure_call = 2 if step == "register_skill" else 1
        if len(calls) == failure_call:
            raise RuntimeError(f"{step} failure")
        return await original(*args)

    monkeypatch.setattr(service, step, flaky)
    result = await evolve.run(config)
    assert result["cases"][0]["status"] == "evolved"
    assert result["errors"] == 0
    assert len(calls) == (3 if step == "register_skill" else 2)
    assert len(service.skills) == 2
    assert len(service.reports) == 1


def test_missing_skillsbench_skills_directory(tmp_path, monkeypatch):
    sample = type("Sample", (), {})()
    sample.task = type("Task", (), {"id": "case"})()
    sample.reward_data = type("Data", (), {"task_dir": tmp_path})()
    monkeypatch.setattr(evolve, "_load_samples", lambda base, env: [sample])
    monkeypatch.setattr(evolve.Environment, "build", lambda *args: None)
    config = SkillEvolveConfig.model_construct(
        environment=type("Env", (), {"class_name": "skillsbench", "config": {}})(),
        cases=None,
        cases_file=None,
        exclude_cases=[],
        case_skills={},
        skills=[],
        limit=None,
    )
    with pytest.raises(ValueError, match="No initial Skills configured for case"):
        evolve.prepare(config)


@pytest.mark.asyncio
async def test_multiple_skills_share_one_report(setup_run, tmp_path):
    config, service = setup_run
    config.cases = ["zero"]
    other = tmp_path / "second"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: second\ndescription: Second fixture\n---\nInitial\n")
    config.skills.append(other)
    result = await evolve.run(config)
    assert len(service.reports) == 1
    assert len(service.reports[0][0]) == 2
    assert len(result["cases"][0]["members"]) == 2
    assert all("score" not in message for _, messages in service.reports for message in messages)


@pytest.mark.asyncio
async def test_exhausted_network_errors_are_not_scored(setup_run):
    config, service = setup_run
    config.cases = ["network"]
    config.network_retries = 0
    result = await evolve.run(config)
    assert result["errors"] == 1
    assert result["included_cases"] == 0
    assert result["metrics"]["collect"]["mean_score"] is None
    assert not service.reports


def test_optional_usage_preserves_cost_and_missing(tmp_path):
    from mindmemos_eval.skills.rollout import case_from_result as _case

    result = RolloutResult(
        output="answer",
        trajectory=[],
        reward=Reward(score=1, success=True, detail="ok"),
        usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12, cache_read_input_tokens=8, cost_usd=0.125),
    )
    case = _case("fixture", result, tmp_path)
    assert case.usage["cost_usd"] == 0.125
    assert case.usage["cache_read_input_tokens"] == 8
    slots = [
        {"status": "completed", "outcome": {"usage": case.usage}},
        {"status": "timeout", "outcome": {"usage": None}},
    ]
    rows = [
        {
            "case": "fixture",
            "status": "perfect",
            "collect": slots,
            "test": [],
            "collect_stats": {"mean_score": 1, "accuracy": 1},
            "test_stats": {"mean_score": 1, "accuracy": 1},
        }
    ]
    from types import SimpleNamespace

    metrics = evolve._summary(rows, SimpleNamespace(reference_file=None))["metrics"]["collect"]
    assert metrics["reported_usage"]["cost_usd"] == 0.125
    assert metrics["average_usage"]["input_tokens"] == 10
    assert metrics["usage_field_missing"]["cost_usd"] == 1


def test_progress_categories(capsys):
    evolve._progress(
        "collect", [{"status": status} for status in ["completed", "timeout", "retryable", "excluded"]], 5, reused=1
    )
    output = capsys.readouterr().out
    assert "4/5 done" in output
    assert "ok 1 | timeout->0 1 | errors 1 | excluded 1 | reused 1" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_case_concurrency_and_stage_barriers(setup_run, monkeypatch, concurrency):
    import asyncio

    config, service = setup_run
    config.cases = ["zero", "network", "perfect"]
    config.repeats = 2
    config.concurrency = concurrency
    original = evolve._slot
    active = set()
    maximum = {"collect": 0, "test": 0}
    events = []

    async def observed_slot(settings, env, sample, skills, directory):
        phase = directory.parent.name
        assert sample.task.id not in active
        if phase == "test":
            assert len([event for event in events if event == "evolve"]) == 2
        active.add(sample.task.id)
        maximum[phase] = max(maximum[phase], len(active))
        try:
            await asyncio.sleep(0.01)
            return await original(settings, env, sample, skills, directory)
        finally:
            active.remove(sample.task.id)
            events.append(phase)

    original_evolve = service.run_evolution

    async def observed_evolve(request):
        assert not active
        assert events.count("collect") == 6
        assert events.count("test") == 0
        events.append("evolve")
        return await original_evolve(request)

    monkeypatch.setattr(evolve, "_slot", observed_slot)
    monkeypatch.setattr(service, "run_evolution", observed_evolve)
    result = await evolve.run(config)
    assert maximum == {"collect": concurrency, "test": concurrency}
    assert [row["case"] for row in result["cases"]] == config.cases
    assert result["cases"][2]["status"] == "perfect"
    assert (config.output_dir / "network/collect/repeat-000/attempts/attempt-000").exists()
    previous = dict(FixtureEnvironment.calls)
    await evolve.run(config)
    assert FixtureEnvironment.calls == previous
