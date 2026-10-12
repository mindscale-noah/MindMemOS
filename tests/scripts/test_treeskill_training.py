"""Verify SDK submission, held-out isolation, recovery and resource export offline."""

import argparse
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
import yaml
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.skills.config import EnvironmentConfig
from mindmemos_sdk.skills.typing import StructuredSkillInfo
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.common.json import read_json, read_jsonl, write_json
from mindmemos_skill.envs import Task
from mindmemos_skill.envs.data import DataSample, RewardData

from scripts.treeskill import run


@pytest.fixture
def inputs(tmp_path):
    """Provide materialized samples and local input paths without external resources."""
    samples = [DataSample(task=Task(id=str(i), instruction="Save"), reward_data=RewardData()) for i in range(4)]
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "service": {
                    "training_pipelines": {
                        "treeskill": {
                            "config": {
                                "trainer": {
                                    "config": {
                                        "optimizer": {"model": "thinking-offline"},
                                        "failure_optimizer": {
                                            "model": "instruct-offline",
                                            "top_p": 1.0,
                                            "presence_penalty": 2.0,
                                            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                                        },
                                        "router": {"model": "offline"},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
    )
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "SKILL.md").write_text("---\nname: example\ndescription: Example\n---\n# Save\nSave output.\n")
    (initial / "helper.bin").write_bytes(b"\x00\xff")
    data = SimpleNamespace(
        load=lambda: samples, split=lambda *args, **kwargs: {"train": samples[:2], "val": [], "test": samples[2:]}
    )
    args = argparse.Namespace(
        service_config=config,
        data_root=tmp_path,
        split_file=tmp_path / "split.json",
        initial_skill=initial,
        output=tmp_path / "output",
        train_count=1,
        test_count=2,
        max_turns=30,
        sdk_home=tmp_path / "sdk",
        poll_interval=0.001,
    )
    return args, data, samples


@pytest.fixture
def submission(tmp_path):
    """Create the report contract and business input for a one-task SDK submission."""
    write_json(
        tmp_path / "manifest.json", {"status": "training", "protocol": {"baseline_skipped": True}, "test_ids": ["test"]}
    )
    return {
        "root": tmp_path,
        "source": StructuredSkillInfo(
            content="---\nname: example\ndescription: Example\n---\n# Save\nSave output.\n",
            resources={"helper.bin": base64.b64encode(b"\x00\xff").decode()},
        ),
        "train": [DataSample(task=Task(id="train", instruction="Save"), reward_data=RewardData())],
        "test": [DataSample(task=Task(id="test", instruction="Verify"), reward_data=RewardData())],
        "env": EnvironmentConfig(class_name="demo_offline", config={}),
        "endpoint": "http://offline",
        "poll_interval": 0.001,
    }


async def test_sdk_submits_splits_and_exports_complete_version(submission):
    requests = []
    source = submission["source"]
    version = {
        **source.model_dump(mode="json"),
        "skill_version_id": "evolved",
        "implementation": "treeskill",
        "config_schema_version": 2,
        "config": {"router": {"model": "offline"}},
    }

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer offline-key"
        if request.url.path.endswith("/register"):
            assert body["skill"]["resources"] == source.resources
            data = {"skill_id": "family", "skill_version_id": "base"}
        elif request.url.path.endswith("/train"):
            assert body["algo"] == "treeskill"
            assert "project_id" not in body
            assert body["split"] == {"train": ["train"], "test": ["test"]}
            assert [sample["task"]["id"] for sample in body["tasks"]] == ["train", "test"]
            assert body["init_skill_version_id"] == "base"
            data = {"task_id": "job", "status": "queued"}
        elif request.url.path.endswith("/progress"):
            data = (
                [{"sequence": 1, "payload": {"body": {"stage": "test"}, "message": "Evaluated test samples"}}]
                if int(request.url.params["after"]) == 0
                else []
            )
        elif request.url.path.endswith("/tasks/show"):
            data = {"status": "succeeded", "result": {"version_id": "evolved", "run_id": "run"}}
        else:
            assert body == {"skill_version_id": "evolved"}
            data = {"version": version}
        return httpx.Response(202 if request.url.path.endswith("/train") else 200, json={"code": "ok", "data": data})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        async with AsyncHttpTransport(base_url="http://offline", api_key="offline-key", client=http) as transport:
            client = AsyncSkillClient(transport)
            assert (await run._train(client, **submission))["version_id"] == "evolved"
            first_count = len(requests)
            await run._train(client, **submission)
    assert not any(request.url.path.endswith(("/register", "/train")) for request in requests[first_count:])
    root = submission["root"]
    assert (root / "evolved_skill/helper.bin").read_bytes() == b"\x00\xff"
    assert read_json(root / "evolved_skill.json")["implementation"] == "treeskill"
    assert read_json(root / "evolved_skill.json")["config"] == version["config"]
    assert len(list(read_jsonl(root / "service_progress.jsonl"))) == 1


async def test_resume_rejects_changed_inputs_before_network(submission):
    skills = SimpleNamespace(
        register=AsyncMock(return_value={"skill_version_id": "base"}),
        train=AsyncMock(return_value={"task_id": "job"}),
        task_progress=AsyncMock(
            side_effect=[
                [
                    {
                        "sequence": 1,
                        "payload": {
                            "message": "Model rejected input",
                            "traceback": "Traceback\nModelError: Model rejected input",
                        },
                    }
                ],
                [],
            ]
        ),
        show_task=AsyncMock(
            return_value={
                "status": "failed",
                "error_type": "ModelError",
            }
        ),
    )
    client = skills
    with pytest.raises(RuntimeError, match="Traceback\nModelError: Model rejected input"):
        await run._train(client, **submission)
    skills.register.reset_mock()
    skills.train.reset_mock()
    skills.show_task.reset_mock()
    submission["env"] = EnvironmentConfig(class_name="demo_offline", config={"changed": True})
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        await run._train(client, **submission)
    skills.register.assert_not_called()
    skills.train.assert_not_called()
    skills.show_task.assert_not_called()


async def test_runner_delegates_train_and_test_and_closes_client(inputs, monkeypatch):
    args, data, samples = inputs
    client = MagicMock()
    training = AsyncMock(return_value={"version_id": "evolved"})
    sdk = SimpleNamespace(
        resolve_skill_connection=lambda: ("http://localhost", "offline"), network=SimpleNamespace(timeout_seconds=30)
    )
    monkeypatch.setattr(run.Data, "build", lambda *args: data)
    monkeypatch.setattr(run, "ConfigManager", lambda path: SimpleNamespace(load=lambda: sdk))
    factory = Mock(return_value=client)
    monkeypatch.setattr(run, "AsyncHttpTransport", factory)
    monkeypatch.setattr(run, "_train", training)
    await run._run(args)
    request = training.call_args.kwargs
    assert request["train"] == samples[:1]
    assert request["test"] == samples[2:]
    assert request["env"].config["transactional_recalculation"]
    policy = request["env"].config["agent"]
    assert policy["model"] == "instruct-offline"
    assert policy["temperature"] == 0.7
    assert policy["stop"] == ["Observation:"]
    assert policy["top_p"] == 1.0
    assert policy["presence_penalty"] == 2.0
    assert policy["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert request["source"].resources == {"helper.bin": base64.b64encode(b"\x00\xff").decode()}
    assert read_json(args.output / "manifest.json")["status"] == "completed"
    assert not (args.output / "baseline").exists()
    factory.assert_called_once_with(
        base_url="http://localhost", api_key="offline", timeout_seconds=30, max_retries=0, require_api_key=False
    )
    client.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("report_fails", [False, True])
async def test_runner_preserves_failure_when_saving_diagnostics(inputs, monkeypatch, report_fails):
    args, data, _ = inputs
    client = MagicMock()
    sdk = SimpleNamespace(
        resolve_skill_connection=lambda: ("http://localhost", "offline"), network=SimpleNamespace(timeout_seconds=30)
    )
    monkeypatch.setattr(run.Data, "build", lambda *args: data)
    monkeypatch.setattr(run, "ConfigManager", lambda path: SimpleNamespace(load=lambda: sdk))
    factory = Mock(return_value=client)
    monkeypatch.setattr(run, "AsyncHttpTransport", factory)
    monkeypatch.setattr(run, "_train", AsyncMock(side_effect=RuntimeError("Training failed")))
    if report_fails:
        monkeypatch.setattr(run, "build_comparison", Mock(side_effect=KeyError("Report failed")))
    with pytest.raises(RuntimeError, match="Training failed"):
        await run._run(args)
    assert read_json(args.output / "manifest.json")["status"] == "failed"
    factory.assert_called_once_with(
        base_url="http://localhost", api_key="offline", timeout_seconds=30, max_retries=0, require_api_key=False
    )
    client.__aexit__.assert_awaited_once()
