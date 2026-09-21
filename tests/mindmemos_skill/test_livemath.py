"""Exercise raw LiveMath adaptation, private scoring, and the real SDK boundary."""

import json

import httpx
import pytest
from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.agents.typing import AgentResult
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.livemath import LiveMathConfig, LiveMathEnvironment
from mindmemos_skill.envs.builtin_envs.livemath.reward import evaluate
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill, SkillBundle, SkillSet


@pytest.fixture
def data(tmp_path):
    """Mirror the official format where the correct option is stored separately."""
    path = tmp_path / "qa_202601_final.json"
    path.write_text(
        json.dumps(
            [
                {
                    "month": "202601",
                    "no": 1,
                    "theorem": "PRIVATE_THEOREM",
                    "sketch": "PRIVATE_SKETCH",
                    "mcq": {
                        "question": "What is 1+1?",
                        "correct_choice": {"label": "A", "text": "2"},
                        "choices": [{"label": "B", "text": "1"}, {"label": "C", "text": "3"}],
                        "meta": {"private": "PRIVATE_META"},
                    },
                }
            ]
        )
    )
    return Data.build("livemath", {"data_path": str(path), "choice_seed": 42})


def test_raw_loading_completes_choices_and_separates_reference(data):
    """Display every answer option but never disclose which one is correct."""
    sample = data.load()[0]
    assert sample.task.id == "202601:1"
    assert {choice.text for choice in sample.task.choices} == {"1", "2", "3"}
    assert sample.reward_data.correct_choice in sample.task.choices
    assert sample.reward_data.correct_choice.text == "2"
    assert len(sample.reward_data.source_sha256) == 64
    public = sample.task.model_dump_json()
    assert "correct_choice" not in public and "PRIVATE_" not in public
    assert data.load() == [sample]
    assert Environment.resolve("livemath") is LiveMathEnvironment


def test_source_changes_invalidate_provenance(data):
    """Track even private source edits through the source content hash."""
    first = data.load()[0]
    path = data.config.data_path
    rows = json.loads(path.read_text())
    rows[0]["sketch"] = "changed reference material"
    path.write_text(json.dumps(rows))
    second = data.load()[0]
    assert first.task == second.task
    assert first.reward_data.source_sha256 != second.reward_data.source_sha256


@pytest.mark.parametrize(
    "prediction", ["tag", "label", "text", "punctuation", "last_tag", "first_token", "wrong", "empty"]
)
def test_skillopt_scoring_formats(data, prediction):
    """Keep tag, text, and final-line label parsing compatible with skillopt."""
    sample = data.load()[0]
    correct = sample.reward_data.correct_choice
    wrong = next(choice.label for choice in sample.task.choices if choice.label != correct.label)
    outputs = {
        "tag": f"Reasoning. <answer>{correct.label.lower()}</answer>",
        "label": correct.label,
        "text": correct.text,
        "punctuation": f"Reasoning\n{correct.label}.",
        "last_tag": f"<answer>{wrong}</answer>\n<answer>{correct.label}</answer>",
        "first_token": f"{correct.label}. explanation",
        "wrong": f"<answer>{wrong}</answer>",
        "empty": "",
    }
    reward = evaluate(outputs[prediction], sample.task, sample.reward_data)
    assert reward.success is (prediction not in {"wrong", "empty"})
    assert reward.score == float(reward.success)
    assert json.loads(reward.detail)["correct_label"] == correct.label


def test_full_rollout_skill_usage_audit_and_resume(data, tmp_path):
    """Exercise environment integration with a deterministic local agent double."""
    sample = data.load()[0]
    config = LiveMathConfig(agent=OpenAIChatConfig(model="test-model"))

    class CapturingAgent(OpenAIChatAgent):
        calls = 0

        def run(self, *, system: str, user: str) -> AgentResult:
            """Assert only public material is sent and return a scored response."""
            self.calls += 1
            assert "Check quantifiers" in system
            assert "correct_choice" not in user and "PRIVATE_" not in user
            assert all(choice.text in user for choice in sample.task.choices)
            return AgentResult(
                output=f"<answer>{sample.reward_data.correct_choice.label}</answer>",
                trajectory=[
                    {"role": "assistant", "tool_calls": [{"name": "calculate", "arguments": {"x": 1}}]},
                    {"role": "tool", "content": "2"},
                    {"role": "assistant", "content": f"<answer>{sample.reward_data.correct_choice.label}</answer>"},
                ],
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            )

    agent = CapturingAgent(config.agent)
    env = LiveMathEnvironment(config, agent=agent)
    skill = Skill(bundle=SkillBundle(files={"SKILL.md": "Check quantifiers"}))
    root = tmp_path / "rollout"
    result = env.rollout(
        sample.task, root, reward_data=sample.reward_data, skill_set=SkillSet({"0": skill}, max_concurrency=1)
    )
    assert result.reward.success and result.usage.total_tokens == 120
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 20
    assert len(result.trajectory) == 3
    assert result.trajectory[0]["tool_calls"][0]["name"] == "calculate"
    assert result.trajectory[1] == {"role": "tool", "content": "2"}
    assert result.usage.model_dump()["completion_tokens_details"] == {"reasoning_tokens": 8}
    assert result.duration_s is not None
    assert set(path.name for path in (root / "workspace").iterdir()) == {"task.md", "system.md", "response.txt"}
    audit = json.loads((root / "audit/request.json").read_text())
    assert audit["reward_data"]["data"]["correct_choice"]["label"] == sample.reward_data.correct_choice.label
    assert (
        env.rollout(
            sample.task, root, reward_data=sample.reward_data, skill_set=SkillSet({"0": skill}, max_concurrency=1)
        )
        == result
    )
    assert agent.calls == 1

    class TaskSkill(Skill):
        def assemble(self, task):
            return self.with_bundle(SkillBundle(files={"SKILL.md": self.bundle.files["SKILL.md"] + task.instruction}))

    dynamic = TaskSkill(bundle=skill.bundle)
    env.rollout(
        sample.task,
        tmp_path / "dynamic",
        reward_data=sample.reward_data,
        skill_set=SkillSet({"0": dynamic}, max_concurrency=1),
    )
    assert agent.calls == 2


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_sdk_payload_usage_and_incomplete_output(monkeypatch, finish_reason):
    """Test the real SDK request format with an in-memory HTTP transport."""
    import openai

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": "<answer>A</answer>"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    monkeypatch.setattr(
        openai, "DefaultHttpxClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
    )
    agent = OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="test-key", base_url="https://unit-test.invalid/v1")
    )
    if finish_reason == "length":
        with pytest.raises(RuntimeError, match="complete text response"):
            agent.run(system="system", user="question")
    else:
        result = agent.run(system="system", user="question")
        assert result.usage["total_tokens"] == 14
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 4
        assert "prompt_tokens" not in result.usage
        assert result.trajectory[:2] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
        assert result.trajectory[-1]["model"] == "test-model"
        assert result.trajectory[-1]["response_id"] == "test-response"
    assert len(requests) == 1
    assert "tools" not in requests[0]
    assert requests[0]["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]
