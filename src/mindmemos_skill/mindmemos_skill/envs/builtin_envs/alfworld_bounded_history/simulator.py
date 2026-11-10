"""Asynchronous JSON-line transport to a request-owned ALFWorld process."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from ....infra.execution.executor import _terminate
from .config import ALFWorldConfig


class Simulator:
    """Keep blocking simulator work outside the service event loop."""

    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    @classmethod
    async def create(cls, config: ALFWorldConfig, gamefile: Path) -> "Simulator":
        """Start a simulator for the selected game and initialize its seed.

        Args:
            config: Recorded simulator settings.
            gamefile: Selected public game file.

        Returns:
            A simulator whose lifetime belongs to one rollout.
        """
        options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.simulator_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            **options,
        )
        instance = cls(process)
        try:
            await instance._request(
                {"operation": "initialize", "config": config.simulator, "seed": config.seed, "gamefile": str(gamefile)}
            )
        except BaseException:
            await instance.close()
            raise
        return instance

    async def _request(self, payload: dict):
        """Send one command and await its response without blocking a thread."""
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write((json.dumps(payload) + "\n").encode())
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("ALFWorld simulator exited without a response")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    async def reset(self):
        """Reset the selected game and return observations and available actions."""
        return await self._request({"operation": "reset"})

    async def step(self, actions: list[str]):
        """Apply the supplied actions and return the simulator transition.

        Args:
            actions: One action for the single selected game.
        """
        return await self._request({"operation": "step", "actions": actions})

    async def close(self) -> None:
        """Terminate the request-owned simulator process and reap it."""
        await _terminate(self.process)
