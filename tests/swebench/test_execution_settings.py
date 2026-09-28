"""Regression tests for no-memory execution and the explicit thinking contract."""

# ruff: noqa: E402 -- Source checkout imports require path initialization.

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/swebench/collect_train.py"
spec = importlib.util.spec_from_file_location("collect_settings", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

from mindmemos_eval.llm import LLMClient, LLMConfig
from mindmemos_eval.swebench.config import load_config
from mindmemos_eval.swebench.data import create_split
from mindmemos_eval.swebench.environment import TaskContainer
from mindmemos_eval.swebench.typing import Task


@pytest.mark.asyncio
async def test_no_output_token_limit_is_sent():
    create = AsyncMock(return_value=object())
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = LLMClient(
        LLMConfig(
            model="example",
            max_tokens=None,
            max_retries=0,
            extra={"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        client=provider,
    )
    await client._create_completion([{"role": "user", "content": "test"}])
    assert "max_tokens" not in create.call_args.kwargs
    assert create.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_baseline_never_opens_memory_clients(tmp_path, monkeypatch):
    import json

    from mindmemos_eval.swebench import runner

    config = load_config(SCRIPT.parents[2] / "config/eval/swebench_verified.example.yaml")
    data = tmp_path / "tasks.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "instance_id": f"example-{i}",
                    "repo": "example/repo",
                    "base_commit": "a" * 40,
                    "problem_statement": "Fix issue",
                }
            )
            for i in range(100)
        )
    )
    config = config.model_copy(
        update={"dataset_path": data, "image_map_path": tmp_path / "images.json", "output_dir": tmp_path / "run"}
    )
    split = create_split(config)
    config.image_map_path.write_text(json.dumps({t.instance_id: "image" for t in split.test}))
    for task in split.test:
        directory = config.output_dir / "baseline" / task.instance_id
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"state": "completed", "finished": True}))
        (directory / "model.patch").write_text("")
    models = AsyncMock(return_value={})
    memories = AsyncMock(side_effect=AssertionError("Memory client must not be opened"))
    monkeypatch.setattr(runner, "_models", models)
    monkeypatch.setattr(runner, "_memories", memories)
    await runner.run_rollouts(config, "baseline")
    memories.assert_not_called()
    assert len((config.output_dir / "baseline" / "predictions.jsonl").read_text().splitlines()) == 50


@pytest.mark.asyncio
async def test_restore_commit_is_inside_container_and_existing_image_is_preserved(monkeypatch):
    from mindmemos_eval.swebench import environment
    from mindmemos_eval.swebench.config import DockerConfig

    calls = []

    async def fake_docker(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            return 0, "sha256:test"
        if args[-1] == "git rev-parse HEAD":
            return 0, "a" * 40
        return 0, ""

    monkeypatch.setattr(environment, "docker", fake_docker)
    task = Task(instance_id="example-1", repo="example/repo", base_commit="a" * 40, problem_statement="fix")
    async with TaskContainer(task, "existing", DockerConfig(restore_base_commit=True, remove_downloaded=True)):
        pass
    resets = [args for args in calls if "reset" in args]
    assert len(resets) == 1 and resets[0][0] == "exec"
    assert not any(args[:2] == ("image", "rm") for args in calls)


@pytest.mark.asyncio
async def test_missing_image_downloads_are_serialized(monkeypatch):
    import asyncio

    from mindmemos_eval.swebench import environment
    from mindmemos_eval.swebench.config import DockerConfig

    downloaded = set()
    active_pulls = 0
    peak_pulls = 0

    async def fake_docker(*args, **kwargs):
        nonlocal active_pulls, peak_pulls
        if args[:2] == ("image", "inspect"):
            image = args[-1]
            return (0, f"sha256:{image}") if image in downloaded else (1, "missing")
        if args[0] == "pull":
            active_pulls += 1
            peak_pulls = max(peak_pulls, active_pulls)
            await asyncio.sleep(0.01)
            downloaded.add(args[-1])
            active_pulls -= 1
            return 0, "downloaded"
        if args[-1] == "git rev-parse HEAD":
            return 0, "a" * 40
        return 0, ""

    async def open_container(index):
        task = Task(instance_id=f"example-{index}", repo="example/repo", base_commit="a" * 40, problem_statement="fix")
        async with TaskContainer(task, f"image-{index}", DockerConfig(pull_missing=True)):
            pass

    monkeypatch.setattr(environment, "docker", fake_docker)
    await asyncio.gather(*(open_container(index) for index in range(3)))
    assert peak_pulls == 1
    assert downloaded == {"image-0", "image-1", "image-2"}


@pytest.mark.asyncio
async def test_provider_default_omits_all_thinking_controls_and_preserves_reasoning(monkeypatch):
    import json
    from contextlib import AsyncExitStack

    import httpx
    from mindmemos_eval.swebench import runner
    from mindmemos_eval.swebench.constants import (
        CHILD_API_KEY_ENV,
        CHILD_BASE_URL_ENV,
        PARENT_API_KEY_ENV,
        PARENT_BASE_URL_ENV,
    )
    from openai import AsyncOpenAI

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "example",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "OK",
                            "reasoning_content": "Example reasoning",
                        },
                    }
                ],
            },
        )

    def provider(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(runner, "AsyncOpenAI", provider)
    for name in (CHILD_API_KEY_ENV, PARENT_API_KEY_ENV):
        monkeypatch.setenv(name, "test-key")
    for name in (CHILD_BASE_URL_ENV, PARENT_BASE_URL_ENV):
        monkeypatch.setenv(name, "https://example.invalid/v1")
    config = load_config(SCRIPT.parents[2] / "config/eval/swebench_train50_baseline50_default.yaml")
    async with AsyncExitStack() as stack:
        clients = await runner._models(config, stack)
        for client in clients.values():
            response = await client.complete([{"role": "user", "content": "test"}], return_format="message")
            assert response.message["reasoning_content"] == "Example reasoning"
    assert len(requests) == 2
    for request in requests:
        assert not {"thinking", "enable_thinking", "reasoning_effort", "max_tokens"} & request.keys()


@pytest.mark.asyncio
async def test_ten_task_workers_keep_completed_results_and_bound_concurrency(tmp_path, monkeypatch):
    import asyncio
    import json

    from mindmemos_eval.swebench import runner

    config = load_config(SCRIPT.parents[2] / "config/eval/swebench_train50_baseline50_default.yaml")
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "instance_id": f"example-{i}",
                    "repo": "example/repo",
                    "base_commit": "a" * 40,
                    "problem_statement": "Fix issue",
                }
            )
            for i in range(100)
        )
    )
    config = config.model_copy(
        update={
            "dataset_path": source,
            "image_map_path": tmp_path / "images.json",
            "output_dir": tmp_path / "run",
            "task_concurrency": 10,
        }
    )
    split = create_split(config)
    config.image_map_path.write_text(json.dumps({t.instance_id: "image" for t in split.train}))
    completed = config.output_dir / "train" / split.train[0].instance_id
    completed.mkdir(parents=True)
    original = '{"state":"completed","finished":true}'
    (completed / "result.json").write_text(original)
    (completed / "model.patch").write_text("existing patch")
    active = 0
    peak = 0
    invoked = []

    class Container:
        image_id = "fake"

        def __init__(self, *args):
            pass

        async def __aenter__(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            return self

        async def __aexit__(self, *args):
            nonlocal active
            active -= 1

        async def patch(self):
            return "patch"

    class Agent:
        children = 0

        def __init__(self, config, task, *args, **kwargs):
            invoked.append(task.instance_id)

        async def run(self):
            await asyncio.sleep(0.01)
            return SimpleNamespace(finished=True)

    monkeypatch.setattr(runner, "_models", AsyncMock(return_value={}))
    monkeypatch.setattr(runner, "TaskContainer", Container)
    monkeypatch.setattr(runner, "HierarchicalAgent", Agent)
    monkeypatch.setattr(runner, "build_parent_memory_view", lambda path: {})
    await runner.run_rollouts(config, "train")
    assert peak == 10 and active == 0
    assert len(invoked) == len(set(invoked)) == 49
    assert split.train[0].instance_id not in invoked
    assert (completed / "result.json").read_text() == original
    assert len((config.output_dir / "train" / "predictions.jsonl").read_text().splitlines()) == 50
