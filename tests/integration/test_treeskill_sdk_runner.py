"""Exercise the root SDK client against HTTP, task and publication boundaries offline."""

import base64

import httpx
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.skills.config import EnvironmentConfig
from mindmemos_sdk.skills.typing import StructuredSkillInfo
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs import Task
from mindmemos_skill.envs.data import DataSample, RewardData

from mindmemos_skill import FastAPIConfig, HttpAuthConfig, ServiceConfig, SkillConfig, create_service
from scripts.treeskill.run import _train


async def test_sdk_runner_authenticates_trains_tests_and_exports(tmp_path):
    """Use the offline profile to verify root Client-to-service execution without a model."""
    import mindmemos_skill.demos  # noqa: F401

    keys = tmp_path / "keys.yaml"
    keys.write_text("project_id: sdk-project\napi_key: offline-key\nscopes: [skill:write]\n")
    config = SkillConfig(
        service=ServiceConfig.model_validate(
            {
                "home": tmp_path / "service",
                "persistence": {"database": {"class_name": "sqlite"}},
                "repository_pipeline": {"class_name": "repository"},
                "task_manager": {"mode": "local", "max_concurrency": 1},
                "training_pipelines": {
                    "treeskill": {
                        "class_name": "training",
                        "config": {
                            "trainer": {"class_name": "demo_train", "config": {"delay_seconds": 0}},
                            "evaluation_concurrency": 1,
                        },
                    }
                },
            }
        ),
        fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="local", api_key_file=str(keys))),
    )

    async def factory():
        return await create_service(config), config.fastapi

    root = tmp_path / "run"
    write_json(
        root / "manifest.json", {"status": "training", "protocol": {"baseline_skipped": True}, "test_ids": ["test"]}
    )
    samples = [
        DataSample(task=Task(id=name, instruction="Save and verify"), reward_data=RewardData())
        for name in ("train", "test")
    ]
    app = create_app(service_factory=factory)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:
            transport = AsyncHttpTransport(base_url="http://testserver", api_key="offline-key", client=http)
            client = AsyncSkillClient(transport)
            result = await _train(
                client,
                root=root,
                source=StructuredSkillInfo(
                    content="---\nname: example\ndescription: Example\n---\n# Save\nSave output.\n",
                    resources={"helper.bin": base64.b64encode(b"\x00\xff").decode()},
                ),
                train=samples[:1],
                test=samples[1:],
                env=EnvironmentConfig(class_name="demo_offline", config={"delay_seconds": 0}),
                endpoint="http://testserver",
                poll_interval=0.001,
            )
    assert read_json(root / "training_status.json")["status"] == "succeeded"
    assert read_json(root / "training_result.json")["version_id"] == result["version_id"]
    assert read_json(root / "training/train/report.json")["train_count"] == 1
    assert [row["task_id"] for row in read_json(root / "training/test/results.json")["tasks"]] == ["test"]
    assert read_json(root / "evolved_skill.json")["metadata"]["algorithm_report"]["train_count"] == 1
    assert (root / "evolved_skill/example/helper.bin").read_bytes() == b"\x00\xff"
