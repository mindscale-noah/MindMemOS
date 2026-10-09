"""Offline input, service orchestration, and test-report checks for skill local."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from mindmemos_eval.skills.local.preflight import LocalSkillEvalPreflight
from mindmemos_eval.skills.local.runner import LocalSkillEvalRunner
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.config.loader import read_skill_config
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.skill import Skill


def make_inputs(tmp_path: Path, *, val_ids: list[str] | None = None) -> Path:
    """Create a tiny SearchQA run using the checked-in local Service definition."""
    data_path = tmp_path / "items.json"
    data_path.write_text(
        json.dumps(
            [
                {"id": identifier, "question": "Who?", "context": "A passage", "answers": ["Someone"]}
                for identifier in ["train", "test", "val"]
            ]
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"train": ["train"], "val": val_ids or [], "test": ["test"]}))
    skill_path = tmp_path / "initial" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text(
        "---\nname: search\ndescription: Search guidance\n---\n# Search guidance\nRead the passage.\n", encoding="utf-8"
    )
    service_path = Path(__file__).resolve().parents[3] / "config/mindmemos_eval/skill_local_service/service.yaml"
    service_payload = yaml.safe_load(service_path.read_text(encoding="utf-8"))
    service_path = tmp_path / "service.yaml"
    service_path.write_text(yaml.safe_dump(service_payload), encoding="utf-8")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "service_config": str(service_path),
                "algorithm": "treeskill",
                "environment": {"class_name": "searchqa", "config": {"agent": {"model": "offline"}}},
                "data": {"class_name": "searchqa", "config": {"data_path": str(data_path)}},
                "initial_skill": {"path": str(skill_path), "name": "test-initial"},
                "split_file": str(split_path),
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_preflight_selects_exact_splits_without_creating_service(tmp_path: Path) -> None:
    """Preflight only reads local inputs and keeps held-out answers in memory."""
    prepared = LocalSkillEvalPreflight().prepare(make_inputs(tmp_path))
    assert prepared.split.train == ["train"]
    assert prepared.split.test == ["test"]
    assert [sample.task.id for sample in prepared.samples] == ["train", "test"]
    assert not (tmp_path / "manifest.json").exists()


def test_preflight_rejects_external_treeskill_validation(tmp_path: Path) -> None:
    """Keep TreeSkill's external validation split empty as required by its protocol."""
    with pytest.raises(ValueError, match="empty external val"):
        LocalSkillEvalPreflight().prepare(make_inputs(tmp_path, val_ids=["val"]))


def test_cli_preflight_exits_without_creating_a_run(tmp_path: Path) -> None:
    """The public module entry validates a run without invoking training."""
    config_path = make_inputs(tmp_path)
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mindmemos_eval.cli",
            "skill",
            "local",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--preflight-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "preflight passed" in result.stdout
    assert not output.exists()


def test_example_matrix_selects_six_registered_profiles() -> None:
    """Every published command config selects a matching trainer and environment."""
    from mindmemos_eval.skills.local.config import LocalSkillEvalConfig

    root = Path(__file__).resolve().parents[3] / "config/mindmemos_eval"
    service = read_skill_config(root / "skill_local_service/service.yaml").service
    examples = sorted((root / "skill_evaluation").glob("*.yaml"))
    assert len(examples) == 6
    for path in examples:
        config = LocalSkillEvalConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert path.stem == f"{config.algorithm}_{config.data.class_name.removesuffix('_bounded_history')}"
        initial_skill = (path.parent / config.initial_skill.path).resolve()
        split_file = (path.parent / config.split_file).resolve()
        assert initial_skill.is_file()
        assert split_file.is_file()
        assert set(json.loads(split_file.read_text(encoding="utf-8"))) == {"train", "val", "test"}
        assert config.algorithm in service.training_pipelines
        assert "training_environments" not in service.model_dump()
        if config.data.class_name == "spreadsheetbench":
            expected = (
                "spreadsheetbench_text_action" if config.algorithm == "treeskill" else "spreadsheetbench_tool_calling"
            )
            assert config.environment.class_name == expected


@pytest.mark.asyncio
async def test_runner_uses_pipeline_test_and_reuses_registration(tmp_path: Path, monkeypatch) -> None:
    """The CLI delegates test execution to the service and only reads its artifacts."""
    from mindmemos_eval.skills.local import runner as module

    config_path = make_inputs(tmp_path)
    run_dir = tmp_path / "run"
    initial_hash = Skill.from_path(tmp_path / "initial" / "SKILL.md").snapshot().skill_hash
    state = {"registered": False, "registration_calls": 0, "training_calls": 0}

    class FakeService:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def show_skill(self, _request):
            if not state["registered"]:
                raise SkillNotFoundError("missing")
            return SimpleNamespace(
                skill=SimpleNamespace(skill_id="skill-1"),
                version=SimpleNamespace(version_id="base-1", skill_hash=initial_hash),
            )

        async def register_from_local(self, request):
            assert request.alias.startswith("eval-")
            state["registration_calls"] += 1
            state["registered"] = True
            return await self.show_skill(None)

        async def get_skill_version(self, version_id, *, project_id):
            assert project_id == "skill-local-eval"
            assert version_id in {"base-1", "published-1"}
            return SimpleNamespace(skill_id="skill-1")

        async def run_training(self, request):
            state["training_calls"] += 1
            assert request.split.test == ["test"]
            test_dir = Path(request.work_dir) / "test"
            boundary(
                test_dir / "task-000000/completed.json",
                {"task": "test"},
                payload={"result": {"reward": {"score": 0.75, "success": True}}},
            )
            write_json(
                test_dir / "results.json",
                {
                    "tasks": [
                        {
                            "task_id": "test",
                            "score": 0.75,
                            "success": True,
                            "result_path": "task-000000/completed.json",
                        }
                    ]
                },
            )
            write_json(test_dir / "summary.json", {"count": 1, "mean_score": 0.75, "success_rate": 1.0})
            return SimpleNamespace(version_id="published-1", base_version_id="base-1")

    async def fake_create_service(_config):
        return FakeService()

    monkeypatch.setattr(module, "create_service", fake_create_service)
    first = await LocalSkillEvalRunner().run(config_path, run_dir)
    assert first.test_count == 1
    assert first.mean_score == 0.75
    assert read_json(run_dir / "manifest.json")["status"] == "completed"
    await LocalSkillEvalRunner().run(config_path, run_dir)
    assert state["registration_calls"] == 1
    assert state["training_calls"] == 2


def test_report_rejects_missing_test_task(tmp_path: Path) -> None:
    """A published version cannot produce a completed report from partial test data."""
    from mindmemos_eval.skills.local.report import build_report
    from mindmemos_eval.skills.local.typing import LocalRunManifest

    manifest = LocalRunManifest(
        input_digest="digest",
        config_path="eval.yaml",
        project_id="project",
        algorithm="treeskill",
        environment="searchqa",
        train_ids=["train"],
        val_ids=[],
        test_ids=["test"],
        skill_id="skill",
        base_version_id="base",
        published_version_id="child",
    )
    write_json(tmp_path / "test/results.json", {"tasks": []})
    with pytest.raises(ValueError, match="task IDs"):
        build_report(manifest, tmp_path)


@pytest.mark.parametrize("algorithm, adapter", [("skill_forest", "searchqa"), ("treeskill", "generic")])
def test_preflight_selects_adapter_without_profile_override(tmp_path: Path, caplog, algorithm, adapter) -> None:
    """Forest benchmark setup must work with just an execution environment."""
    import logging

    path = make_inputs(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["algorithm"] = algorithm
    path.write_text(yaml.safe_dump(config))
    with caplog.at_level(logging.INFO, logger="mindmemos_skill"):
        prepared = LocalSkillEvalPreflight().prepare(path)
    assert prepared.split.train == ["train"]
    assert prepared.config.environment.class_name == "searchqa"
    assert f"environment=searchqa adapter={adapter}" in caplog.text
