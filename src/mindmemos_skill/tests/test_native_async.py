"""Protect native execution contracts and cancellation cleanup."""

import ast
import asyncio
import inspect
import io
import json
import sys
from pathlib import Path

import pytest
from mindmemos_skill.agents import Agent
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.algos.evolve.base import SkillEvolver
from mindmemos_skill.algos.manage.base import SkillManager
from mindmemos_skill.algos.train.base import SkillTrainer
from mindmemos_skill.envs import Environment
from mindmemos_skill.infra.execution import ExecutionConfig, execute


def test_builtin_execution_interfaces_are_coroutines():
    """A new backend must not silently reintroduce a synchronous execution API."""
    for implementation in Agent.registered_types().values():
        assert inspect.iscoroutinefunction(implementation.run), implementation
    for implementation in Environment.registered_types().values():
        assert inspect.iscoroutinefunction(implementation._run), implementation
    for implementation in Algorithm.registered_types().values():
        if issubclass(implementation, SkillTrainer):
            assert inspect.iscoroutinefunction(implementation.train), implementation
        elif issubclass(implementation, SkillEvolver):
            assert inspect.iscoroutinefunction(implementation.evolve), implementation
        elif issubclass(implementation, SkillManager):
            assert inspect.iscoroutinefunction(implementation.organize), implementation
            assert inspect.iscoroutinefunction(implementation.recommend), implementation


def test_skill_execution_has_no_thread_bridges():
    """Reject application-level thread adapters anywhere in the Skill package."""
    package = Path(__file__).parents[1] / "mindmemos_skill"
    forbidden = {"to_thread", "run_in_executor", "ThreadPoolExecutor", "map_with_context", "map_sync_bounded"}
    violations = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                if name in forbidden:
                    violations.append(f"{path.relative_to(package)}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)


async def test_cancelled_command_is_reaped_before_return(tmp_path, monkeypatch):
    """Cancellation propagates and waits for the actual subprocess to exit."""
    from mindmemos_skill.infra.execution import executor

    launched = asyncio.Event()
    processes = []
    original = asyncio.create_subprocess_exec

    async def launch(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        launched.set()
        return process

    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", launch)
    task = asyncio.create_task(
        execute(
            ["python", "-c", "import time; time.sleep(30)"],
            tmp_path,
            ExecutionConfig(python_command=[sys.executable]),
        )
    )
    await asyncio.wait_for(launched.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert processes[0].returncode is not None


def test_simulator_worker_serializes_only_rollout_fields(monkeypatch, capsys):
    """Opaque upstream game objects never enter the JSON transport."""
    import numpy as np
    from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history import simulator_worker

    closed = []

    class FakeSimulator:
        def reset(self):
            return ["Your task is to: look"], {"admissible_commands": [["look"]], "game": object()}

        def step(self, actions):
            assert actions == ["look"]
            return (
                ["done"],
                np.array([1.0]),
                np.array([True]),
                {
                    "admissible_commands": [[]],
                    "won": np.array([True]),
                    "game": object(),
                },
            )

        def close(self):
            closed.append(True)

    commands = [
        {"operation": "initialize", "config": {}, "gamefile": "game", "seed": 42},
        {"operation": "reset"},
        {"operation": "step", "actions": ["look"]},
    ]
    monkeypatch.setattr(simulator_worker, "create_simulator", lambda *args: FakeSimulator())
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(json.dumps(command) for command in commands)))
    simulator_worker.main()
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[1]["result"] == [["Your task is to: look"], {"admissible_commands": [["look"]]}]
    assert replies[2]["result"] == [["done"], [1.0], [True], {"admissible_commands": [[]], "won": [True]}]
    assert closed == [True]


async def test_simulator_transport_uses_real_async_pipes(tmp_path, monkeypatch):
    """Run the simulator protocol in a real child without the optional game runtime."""
    from textwrap import dedent

    from mindmemos_skill.agents import OpenAIChatConfig
    from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history import simulator
    from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.config import ALFWorldConfig

    original = asyncio.create_subprocess_exec
    script = dedent("""
        from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history import simulator_worker

        class Game:
            def reset(self):
                return ["ready"], {"admissible_commands": [["look"]]}
            def step(self, actions):
                return actions, [1.0], [True], {"admissible_commands": [[]], "won": [True]}
            def close(self):
                pass

        simulator_worker.create_simulator = lambda *args: Game()
        simulator_worker.main()
    """)

    async def launch(*args, **kwargs):
        assert args[1:] == ("-m", "mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.simulator_worker")
        return await original(sys.executable, "-c", script, **kwargs)

    monkeypatch.setattr(simulator.asyncio, "create_subprocess_exec", launch)
    config = ALFWorldConfig(agent=OpenAIChatConfig(model="offline"), simulator={})
    async with asyncio.timeout(10):
        game = await simulator.Simulator.create(config, tmp_path / "game")
        try:
            assert await game.reset() == [["ready"], {"admissible_commands": [["look"]]}]
            transition = await game.step(["look"])
            assert transition[0] == ["look"] and transition[3]["won"] == [True]
        finally:
            await game.close()
    assert game.process.returncode is not None
