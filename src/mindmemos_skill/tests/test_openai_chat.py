"""Verify retry policy, exponential waits, and observable retry reasons."""

import logging

import httpx
import openai
import pytest
import tenacity
from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.agents.constants import AGENT_API_KEY_ENV, AGENT_BASE_URL_ENV


@pytest.mark.parametrize(
    "failures,max_retries,expected_waits,expected_calls",
    [
        ([429, 500, 503], 3, [1, 2, 3], 4),
        (["timeout"], 3, [1], 2),
        ([408, 409], 3, [1, 2], 3),
        ([503, 503, 503, 503], 3, [1, 2, 3], 4),
        ([429], 0, [], 1),
        ([400], 3, [], 1),
        ([401], 3, [], 1),
    ],
)
def test_retry_policy_and_logging(monkeypatch, caplog, failures, max_retries, expected_waits, expected_calls):
    """Exercise the real SDK and Tenacity without networking or actual sleeps."""
    calls = []
    sleeps = []
    retrying = tenacity.Retrying
    monkeypatch.setattr(tenacity, "Retrying", lambda **kwargs: retrying(sleep=sleeps.append, **kwargs))

    def respond(request):
        calls.append(request)
        if len(calls) <= len(failures):
            failure = failures[len(calls) - 1]
            if failure == "timeout":
                raise httpx.ReadTimeout("Temporary timeout", request=request)
            return httpx.Response(failure, json={"error": {"message": "Temporary test failure"}})
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            },
        )

    monkeypatch.setattr(
        openai, "DefaultHttpxClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
    )
    agent = OpenAIChatAgent(
        OpenAIChatConfig(
            model="test-model",
            api_key="test-key",
            base_url="https://unit-test.invalid/v1",
            max_retries=max_retries,
            retry_max_wait_s=3,
        )
    )
    with caplog.at_level(logging.WARNING):
        if expected_calls > len(failures):
            assert agent.run(system="system", user="question").output == "OK"
        else:
            with pytest.raises(openai.APIStatusError) as caught:
                agent.run(system="system", user="question")
            assert caught.value.status_code == failures[-1]

    assert len(calls) == expected_calls
    assert sleeps == expected_waits
    messages = [record.getMessage() for record in caplog.records if "OpenAI chat retry" in record.getMessage()]
    assert len(messages) == len(expected_waits)
    for retry_number, (message, wait) in enumerate(zip(messages, expected_waits), start=1):
        assert f"retry {retry_number}/{max_retries}" in message
        assert f"sleep {wait:.3f} seconds" in message
        if failures[retry_number - 1] == "timeout":
            assert "reason: APITimeoutError:" in message
        else:
            assert "reason:" in message and "Temporary test failure" in message


@pytest.mark.parametrize("preset", ["workspace_tools", "bash_only_instruct", "bash_only_thinking"])
def test_spreadsheet_preset_request(monkeypatch, preset):
    """Verify actual SDK payloads from the recorded experiment actor presets."""
    import json
    from pathlib import Path

    import yaml
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import SpreadsheetBenchConfig

    path = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/spreadsheetbench" / f"{preset}.yaml"
    monkeypatch.setenv(AGENT_API_KEY_ENV, "test-key")
    monkeypatch.setenv(AGENT_BASE_URL_ENV, "https://unit-test.invalid/v1")
    config = SpreadsheetBenchConfig.model_validate(yaml.safe_load(path.read_text())).agent
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": config.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "<think>private</think>Done" if config.strip_thinking else "Done",
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(
        openai, "DefaultHttpxClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
    )
    assert OpenAIChatAgent(config).run(system="system", user="task").output == "Done"
    payload = requests[0]
    if preset == "workspace_tools":
        assert payload["max_completion_tokens"] == 16384
        assert payload["reasoning_effort"] == "medium"
        assert "temperature" not in payload
    else:
        assert "max_completion_tokens" not in payload
        assert payload["temperature"] == 0.7
        assert payload["seed"] == 41
        assert payload["stop"] == ["Observation:"]
        assert payload["top_p"] == (1.0 if preset.endswith("instruct") else 0.95)
        assert payload["presence_penalty"] == (2.0 if preset.endswith("instruct") else 1.5)
        assert payload["chat_template_kwargs"] == {"enable_thinking": preset.endswith("thinking")}
        assert payload["top_k"] == (40 if preset.endswith("instruct") else 20)


@pytest.mark.parametrize("wait_mode,waits", [("schedule", [5, 10, 30]), ("equal_jitter", [1.5, 3, 6])])
def test_experiment_retry_waits(monkeypatch, wait_mode, waits):
    """Check explicit retry schedules and equal jitter without real waiting."""
    import random

    sleeps, calls = [], []
    retrying = tenacity.Retrying
    monkeypatch.setattr(tenacity, "Retrying", lambda **kwargs: retrying(sleep=sleeps.append, **kwargs))
    monkeypatch.setattr(random, "random", lambda: 0.5)

    def respond(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "fixture"}})

    monkeypatch.setattr(
        openai, "DefaultHttpxClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
    )
    config = OpenAIChatConfig(
        model="fixture",
        api_key="test-key",
        base_url="https://unit-test.invalid/v1",
        retry_policy="all",
        retry_wait=wait_mode,
        retry_initial_wait_s=2,
    )
    with pytest.raises(openai.BadRequestError):
        OpenAIChatAgent(config).run(system="system", user="task")
    assert len(calls) == 4
    assert sleeps == waits
