"""Structured response forwarding and bounded parser correction contracts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from pydantic import BaseModel


class Answer(BaseModel):
    value: int


class Transport(OpenAIChatAgent):
    def __init__(self, config, outputs):
        super().__init__(config)
        self.outputs = iter(outputs)
        self.requests = []

    def run_messages(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return AgentResult(
            output=output,
            trajectory=[*kwargs["messages"], {"role": "assistant", "content": output}],
            usage={"total_tokens": 7, "input_tokens": 3, "output_tokens": 4},
        )


@pytest.mark.parametrize("mode", ["direct", "feedback"])
def test_parser_retries_are_bounded_preserve_inputs_and_sum_usage(mode):
    config = OpenAIChatConfig(model="test", parse_retry_mode=mode, parse_max_attempts=2)
    transport = Transport(config, ["bad", '{"value": 42}'])
    agent = OpenAIChatAgent(config, model=transport, output_parser=Answer.model_validate_json)
    messages = [{"role": "user", "content": "Return an integer"}]
    result = agent.run(messages=messages, response_format={"type": "json_object"})
    assert result.output == Answer(value=42)
    assert result.usage["total_tokens"] == 14
    assert result.usage["parse_attempts"] == 2
    assert len(messages) == 1
    assert all(call["response_format"] == {"type": "json_object"} for call in transport.requests)
    assert len(transport.requests[1]["messages"]) == (1 if mode == "direct" else 3)
    if mode == "feedback":
        assert transport.requests[1]["messages"][-1]["content"].startswith(
            "Your previous reply could not be applied:\n"
        )


def test_parse_exhaustion_and_provider_errors_propagate():
    config = OpenAIChatConfig(model="test", parse_max_attempts=2)
    transport = Transport(config, ["bad", "bad", '{"value": 1}'])
    with pytest.raises(ValueError):
        OpenAIChatAgent(config, model=transport, output_parser=Answer.model_validate_json).run(system="s", user="u")
    assert len(transport.requests) == 2
    transport = Transport(config, [RuntimeError("offline")])
    with pytest.raises(RuntimeError, match="offline"):
        OpenAIChatAgent(config, model=transport, output_parser=Answer.model_validate_json).run(system="s", user="u")
    assert len(transport.requests) == 1


def test_custom_feedback_and_single_attempt():
    config = OpenAIChatConfig(model="test", parse_max_attempts=2, parse_retry_mode="feedback")
    transport = Transport(config, ["bad", '{"value": 1}'])
    agent = OpenAIChatAgent(
        config,
        model=transport,
        output_parser=Answer.model_validate_json,
        parse_feedback=lambda error: "Exact correction",
    )
    agent.run(system="s", user="u")
    assert transport.requests[1]["messages"][-1]["content"] == "Exact correction"


def test_response_format_reaches_openai_request():
    config = OpenAIChatConfig(
        model="test", api_key="test-secret", base_url="https://test.invalid/v1", response_format={"type": "json_object"}
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(tool_calls=None, refusal=None, content='{"value": 1}')
            )
        ],
        usage=None,
        model="test",
        id="r1",
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = response
    with patch("openai.OpenAI", return_value=client):
        OpenAIChatAgent(config).run(system="s", user="u")
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_output_model_uses_shared_extraction_and_retries_schema_errors():
    config = OpenAIChatConfig(model="test", parse_max_attempts=2, parse_retry_mode="feedback")
    transport = Transport(config, ['{"wrong":1}', 'Answer: ```json\n{"value":42,}\n```'])
    result = OpenAIChatAgent(config, model=transport, output_model=Answer).run(system="s", user="u")
    assert result.output == Answer(value=42)
    assert result.usage["parse_attempts"] == 2
    assert result.usage["total_tokens"] == 14
    assert len(transport.requests) == 2


def test_output_model_rejects_conflicting_parsers():
    config = OpenAIChatConfig(model="test")
    with pytest.raises(ValueError, match="output_model"):
        OpenAIChatAgent(config, output_model=Answer, output_parser=Answer.model_validate_json)
    with pytest.raises(ValueError, match="output_model"):
        OpenAIChatAgent(config, output_model=Answer, parse_response=lambda result, config: None)
