"""Document-tool protocol and generic reflection regression coverage."""

import json

from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter
from mindmemos_skill.algos.train.skill_forest.typing import SkillForestReflectionContext, SkillForestTrajectoryContext
from mindmemos_skill.envs.builtin_envs.officeqa.tools import format_tool_command, run_tool
from mindmemos_skill.envs.builtin_envs.officeqa.typing import OfficeQATask
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task


async def test_officeqa_tool_defaults_and_original_responses(tmp_path):
    path = tmp_path / "public.txt"
    lines = [f"value {index}\n" for index in range(100)]
    path.write_text("".join(lines))
    documents = {"public.txt": path}
    assert (await run_tool("read", {"path": "public.txt"}, documents, 16000)) == "".join(lines[:80])
    assert (await run_tool("read", {"path": "public.txt", "start": 99}, documents, 16000)) == "".join(lines[98:])
    assert (await run_tool("read", {"path": "public.txt", "start": 200}, documents, 16000)) == "[empty file]"
    assert (await run_tool("glob", {"pattern": "*.csv"}, documents, 16000)) == "[no matches]"
    assert (await run_tool("grep", {"path": "public.txt", "pattern": "VALUE"}, documents, 16000)) == "\n".join(
        f"{index + 1}: value {index}" for index in range(20)
    )
    assert (await run_tool("read", {}, documents, 16000)) == "[read error: missing path]"
    assert (await run_tool("read", {"path": "../private"}, documents, 16000)) == "[read error: path not allowed]"
    assert (await run_tool("grep", {"path": "public.txt"}, documents, 16000)) == "[grep error: missing pattern or path]"
    path.write_text("x" * 6000)
    assert len((await run_tool("read", {"path": "public.txt"}, documents, 16000))) == 4000
    assert len((await run_tool("read", {"path": "public.txt"}, documents, 10))) == 10
    assert format_tool_command("read", '{"path":"public.txt"}') == "read(path='public.txt', start=1, limit=80)"
    assert (
        format_tool_command("grep", '{"path":"public.txt","pattern":"VALUE"}')
        == "grep(pattern='value', path='public.txt')"
    )
    assert format_tool_command("read", '{"path":"private"}', "[read error: path not allowed]") == "read(path='private')"


def test_officeqa_and_generic_share_markdown_without_private_reward():
    trajectory = [
        {"role": "system", "content": "Public system."},
        {"role": "user", "content": "Public task."},
        {
            "role": "assistant",
            "content": "Inspect.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "read", "arguments": '{"path":"public.txt","start":5,"limit":10}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Evidence."},
        {"role": "assistant", "content": "Done."},
    ]
    result = RolloutResult(output="Done.", trajectory=trajectory, reward=Reward(score=1, success=True, detail=""))
    context = SkillForestTrajectoryContext(
        task=OfficeQATask(id="example", instruction="Read it.", documents={}), result=result
    )
    office = SkillForestEnvAdapter.build("officeqa", {})
    generic = SkillForestEnvAdapter.build("generic", {})
    assert generic.render_trajectory(context) == office.render_trajectory(context)
    result.reward.detail = "PRIVATE-ANSWER"
    result.trajectory.append({"role": "system", "content": result.reward.detail})
    before = result.model_dump_json()
    messages = generic.build_reflection_messages(
        SkillForestReflectionContext(skill="# Rules", trajectories=(context,), success=True, edit_budget=4)
    )
    text = messages[1]["content"]
    assert text.startswith("## Current Skill\n# Rules\n\n## Edits Budget\n")
    assert "Steps: 3\n" in text
    assert "[action] read(path='public.txt', start=5, limit=10)\n[obs]    Evidence." in text
    assert text.count("Public task.") == 1
    assert "PRIVATE-ANSWER" not in text
    assert result.model_dump_json() == before


def test_generic_retains_partial_and_output_only_evidence():
    adapter = SkillForestEnvAdapter.build("generic", {})
    task = Task(id="partial", instruction="Run it.")
    result = RolloutResult(output="Done.", trajectory=[], reward=Reward(score=0, success=False, detail="private"))
    context = SkillForestTrajectoryContext(task=task, result=result)
    assert "[agent] Done." in adapter.render_trajectory(context)
    result.trajectory = [
        {
            "role": "assistant",
            "content": "Try.",
            "tool_calls": [{"function": {"name": "custom", "arguments": json.dumps({"query": "needle"})}}],
        }
    ]
    text = adapter.render_trajectory(context)
    assert "[action] custom(query='needle')" in text
    assert "private" not in text
