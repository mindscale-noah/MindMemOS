from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mindmemos_skill.agents import ReactAgentConfig, Tool, get_agent
from mindmemos_skill.agents.react import ReactAgent, ReactSkillRuntime
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.typing import (
    AgentExecutionRequest,
    AgentType,
    Rollout,
    Skill,
    SkillInjectionMode,
    SkillUsageType,
    Task,
    Trajectory,
    TrajectoryStatus,
)


class FakeChatClient:
    def __init__(self, responses: list[ChatResponse | None | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        task: str,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse | None:
        self.calls.append({"task": task, "messages": messages, "model": model, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_request(
    *,
    skills: list[Skill] | None = None,
    options: dict[str, Any] | None = None,
    system_prompt: str | None = "Be concise.",
) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        trajectory_id="trajectory-react-1",
        task=Task(task_id="task-react-1", instruction="What is 2 + 3?", system_prompt=system_prompt),
        rollout=Rollout(rollout_id="rollout-react-1"),
        skills=skills or [],
        options=options or {},
    )


def assistant(*, content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> ChatResponse:
    return ChatResponse(
        finish_reason="tool_calls" if tool_calls else "stop",
        content=content,
        model="openai/test-model",
        tool_calls=tool_calls or [],
    )


def tool_call(name: str, arguments: str, *, call_id: str = "call-1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.asyncio
async def test_react_agent_runs_openai_tool_loop_and_persists_messages() -> None:
    llm = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("add", '{"left":2,"right":3}')]),
            assistant(content="5"),
        ]
    )

    async def add(left: int, right: int) -> int:
        return left + right

    agent = get_agent(
        agent_type=AgentType.REACT,
        config=ReactAgentConfig(model="test-model", temperature=0.2),
        llm=llm,
        tools=[
            Tool(
                name="add",
                description="Add two integers.",
                parameters={
                    "type": "object",
                    "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
                    "required": ["left", "right"],
                },
                func=add,
            )
        ],
    )

    trajectory = await agent.execute(make_request())

    assert isinstance(agent, ReactAgent)
    assert agent.supported_skill_injection_modes == frozenset(
        {SkillInjectionMode.TOOL, SkillInjectionMode.SYSTEM_PROMPT}
    )
    assert isinstance(agent.get_skill_runtime(SkillInjectionMode.TOOL), ReactSkillRuntime)
    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED
    assert trajectory.execution.n_turn == 2
    assert trajectory.agent.agent_type is AgentType.REACT
    assert trajectory.agent.model == "test-model"
    assert trajectory.agent.temperature == 0.2
    assert trajectory.events[-2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "add",
        "content": "5",
    }
    assert trajectory.events[-1] == {"role": "assistant", "content": "5"}
    assert llm.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
                    "required": ["left", "right"],
                },
            },
        }
    ]
    assert llm.calls[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_react_agent_exposes_injected_skill_as_reserved_tool_and_binds_version() -> None:
    skill = Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="1.0.0",
        content_hash="sha256:skill",
        name="demo",
        description="Demo instructions",
        blob={"SKILL.md": "Always use the persisted instructions."},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    llm = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("skill", '{"name":"demo"}', call_id="skill-call")]),
            assistant(content="done"),
        ]
    )
    agent = ReactAgent({}, llm=llm)

    trajectory = await agent.execute(make_request(skills=[skill]))

    assert llm.calls[0]["messages"][:2] == [
        {
            "role": "system",
            "content": (
                "Be concise.\n\n"
                "<available_skills>\n"
                "  <skill>\n"
                "    <name>demo</name>\n"
                "    <description>Demo instructions</description>\n"
                "  </skill>\n"
                "</available_skills>"
            ),
        },
        {"role": "user", "content": "What is 2 + 3?"},
    ]
    assert sum(message["role"] == "system" for message in llm.calls[0]["messages"]) == 1

    escaped_skill = skill.model_copy(
        update={
            "name": "demo<&>",
            "description": "Use <tool> & verify",
            "resources": {"references/helper.py": "HELPER = True\n"},
        }
    )
    with agent.inject_skills([escaped_skill]) as injection:
        assert injection.system_prompt_suffix == (
            "<available_skills>\n"
            "  <skill>\n"
            "    <name>demo&lt;&amp;&gt;</name>\n"
            "    <description>Use &lt;tool&gt; &amp; verify</description>\n"
            "  </skill>\n"
            "</available_skills>"
        )
        assert injection.workspace is not None
        skill_directory = Path(injection.workspace) / "skills" / "demo"
        assert (skill_directory / "SKILL.md").read_text(encoding="utf-8") == (
            "Always use the persisted instructions.\n"
        )
        assert (skill_directory / "references" / "helper.py").read_text(encoding="utf-8") == "HELPER = True\n"
    assert not Path(injection.workspace).exists()
    skill_schema = llm.calls[0]["tools"][0]
    assert skill_schema["function"]["name"] == "skill"
    assert skill_schema["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill to load. One of: demo.",
            }
        },
        "required": ["name"],
    }
    tool_result = next(event for event in trajectory.events if event.get("tool_call_id") == "skill-call")
    assert tool_result["content"] == "Result of 'skill' delivered in the following user message."
    delivered_result = trajectory.events[trajectory.events.index(tool_result) + 1]
    assert delivered_result["role"] == "user"
    assert "Loaded skill 'demo'." in delivered_result["content"]
    assert "Skill directory (absolute path):" in delivered_result["content"]
    assert "Always use the persisted instructions." in delivered_result["content"]
    assert "version-1" not in delivered_result["content"]
    assert "sha256:skill" not in delivered_result["content"]
    assert trajectory.skill_bindings[0].version_id == "version-1"
    assert trajectory.skill_bindings[0].usage is SkillUsageType.INJECTED
    assert trajectory.skill_bindings[0].injection_mode is SkillInjectionMode.TOOL


@pytest.mark.asyncio
async def test_react_agent_only_binds_a_successfully_loaded_skill_result() -> None:
    skill = Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="1.0.0",
        content_hash="sha256:skill",
        name="demo",
        blob={"SKILL.md": "instructions"},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    llm = FakeChatClient(
        [
            assistant(
                tool_calls=[
                    tool_call("skill", '{"name":"unknown"}', call_id="failed-skill"),
                    tool_call("skill", "not-json", call_id="malformed-skill"),
                ]
            ),
            assistant(content="done"),
        ]
    )

    trajectory = await ReactAgent({}, llm=llm).execute(make_request(skills=[skill]))

    failed_result = next(event for event in trajectory.events if event.get("tool_call_id") == "failed-skill")
    assert failed_result["content"] == "Result of 'skill' delivered in the following user message."
    delivered_error = trajectory.events[trajectory.events.index(failed_result) + 1]
    assert delivered_error["content"] == "Error: unknown skill 'unknown'. Available skills: demo"
    malformed_result = next(event for event in trajectory.events if event.get("tool_call_id") == "malformed-skill")
    assert malformed_result["content"] == "Result of 'skill' delivered in the following user message."
    malformed_error = trajectory.events[trajectory.events.index(malformed_result) + 1]
    assert malformed_error["content"].startswith("Error: TypeError:")
    assert "__raw__" in malformed_error["content"]
    assert trajectory.skill_bindings[0].usage is SkillUsageType.UNUSED


@pytest.mark.asyncio
async def test_react_agent_merges_skill_content_into_existing_system_prompt() -> None:
    skill = Skill(
        skill_id="skill-1",
        version_id="version-system-1",
        version_label="1.0.0",
        content_hash="sha256:system-skill",
        name="prompt-demo",
        blob={"SKILL.md": "Follow the system-injected instructions."},
        resources={"references/example.md": "Persisted reference"},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    llm = FakeChatClient([assistant(content="done")])
    agent = ReactAgent({}, llm=llm)

    trajectory = await agent.execute(
        make_request(
            skills=[skill],
            options={"skill_injection_mode": SkillInjectionMode.SYSTEM_PROMPT},
        )
    )

    prompt_messages = llm.calls[0]["messages"]
    injected_message = prompt_messages[0]
    assert injected_message == {
        "role": "system",
        "content": (
            "Be concise.\n\n"
            "<available_skills>\n"
            "  <skill>\n"
            "    <name>prompt-demo</name>\n"
            "    <description></description>\n"
            "    <content>Follow the system-injected instructions.</content>\n"
            "  </skill>\n"
            "</available_skills>"
        ),
    }
    assert sum(message["role"] == "system" for message in prompt_messages) == 1
    assert "version-system-1" not in injected_message["content"]
    assert "<version_id>" not in injected_message["content"]
    assert "sha256:system-skill" not in injected_message["content"]
    assert "Persisted reference" not in injected_message["content"]
    assert "tools" not in llm.calls[0]
    assert trajectory.events[0] == injected_message
    assert trajectory.agent.skill_injection_mode is SkillInjectionMode.SYSTEM_PROMPT
    assert trajectory.skill_bindings[0].usage is SkillUsageType.INJECTED
    assert trajectory.skill_bindings[0].injection_mode is SkillInjectionMode.SYSTEM_PROMPT

    restored = Trajectory.from_record(trajectory.to_record())
    assert restored.agent.skill_injection_mode is SkillInjectionMode.SYSTEM_PROMPT
    assert restored.skill_bindings[0].injection_mode is SkillInjectionMode.SYSTEM_PROMPT

    escaped_skill = skill.model_copy(update={"blob": {"SKILL.md": "Use <rule> & finish."}})
    with agent.inject_skills([escaped_skill], mode=SkillInjectionMode.SYSTEM_PROMPT) as injection:
        assert injection.system_prompt_suffix is not None
        assert "<content>Use &lt;rule&gt; &amp; finish.</content>" in injection.system_prompt_suffix


@pytest.mark.asyncio
async def test_react_agent_uses_skill_content_as_system_prompt_when_base_prompt_is_absent() -> None:
    skill = Skill(
        skill_id="skill-1",
        version_id="version-system-1",
        version_label="1.0.0",
        content_hash="sha256:system-skill",
        name="prompt-demo",
        blob={"SKILL.md": "Use the learned procedure."},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    llm = FakeChatClient([assistant(content="done")])

    trajectory = await ReactAgent({}, llm=llm).execute(
        make_request(
            skills=[skill],
            options={"skill_injection_mode": SkillInjectionMode.SYSTEM_PROMPT},
            system_prompt=None,
        )
    )

    assert llm.calls[0]["messages"][:2] == [
        {
            "role": "system",
            "content": (
                "<available_skills>\n"
                "  <skill>\n"
                "    <name>prompt-demo</name>\n"
                "    <description></description>\n"
                "    <content>Use the learned procedure.</content>\n"
                "  </skill>\n"
                "</available_skills>"
            ),
        },
        {"role": "user", "content": "What is 2 + 3?"},
    ]
    assert trajectory.events[:2] == llm.calls[0]["messages"]


@pytest.mark.asyncio
async def test_react_agent_returns_tool_errors_to_model_for_recovery() -> None:
    llm = FakeChatClient(
        [
            assistant(
                tool_calls=[
                    tool_call("missing", "{}", call_id="unknown-call"),
                    tool_call("missing", "not-json", call_id="invalid-call"),
                ]
            ),
            assistant(content="recovered"),
        ]
    )
    agent = ReactAgent({}, llm=llm)

    trajectory = await agent.execute(make_request())

    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED
    assert trajectory.events[-3]["content"] == "Error: unknown tool 'missing'"
    assert trajectory.events[-2]["content"] == "Error: unknown tool 'missing'"
    assert llm.calls[1]["messages"][-2:] == trajectory.events[-3:-1]


@pytest.mark.asyncio
async def test_react_agent_marks_max_turns_and_empty_model_response_as_failures() -> None:
    pending_call = assistant(tool_calls=[tool_call("missing", "{}")])
    max_turns_agent = ReactAgent({"max_turns": 1}, llm=FakeChatClient([pending_call]))

    max_turns = await max_turns_agent.execute(make_request())

    assert max_turns.execution.status is TrajectoryStatus.FAILED
    assert max_turns.execution.n_turn == 1
    assert max_turns.execution.error_info == "ReAct agent reached max_turns=1 before producing a final response"

    empty_agent = ReactAgent({}, llm=FakeChatClient([None]))
    empty = await empty_agent.execute(make_request())

    assert empty.execution.status is TrajectoryStatus.FAILED
    assert empty.execution.error_info == "RuntimeError: Chat model returned no response"


def test_react_agent_rejects_duplicate_and_reserved_tool_names() -> None:
    def noop() -> None:
        return None

    duplicate = Tool(name="same", description="", func=noop)

    with pytest.raises(ValueError, match="duplicate tool name"):
        ReactAgent({}, llm=FakeChatClient([]), tools=[duplicate, duplicate])
    with pytest.raises(ValueError, match="reserved"):
        ReactAgent({}, llm=FakeChatClient([]), tools=[Tool(name="skill", description="", func=noop)])


def test_react_skill_tool_materialization_rejects_directory_collisions_and_escaping_paths() -> None:
    skill = Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="1.0.0",
        content_hash="sha256:skill",
        name="demo one",
        blob={"SKILL.md": "instructions"},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    agent = ReactAgent({}, llm=FakeChatClient([]))

    collision = skill.model_copy(update={"skill_id": "skill-2", "version_id": "version-2", "name": "demo@one"})
    with pytest.raises(ValueError, match="duplicate injected Skill directory"):
        with agent.inject_skills([skill, collision]):
            pass

    escaping = skill.model_copy(update={"resources": {"../outside.txt": "blocked"}})
    with pytest.raises(ValueError, match="escapes its ReAct directory"):
        with agent.inject_skills([escaping]):
            pass
