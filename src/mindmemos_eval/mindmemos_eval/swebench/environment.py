"""Execution in pre-existing local Docker images without automatic downloads."""

import asyncio
import uuid

from .config import DockerConfig
from .typing import Task

_image_pull_lock = asyncio.Lock()


async def docker(*args: str, timeout: float = 60) -> tuple[int, str]:
    """Run Docker CLI arguments and return combined output.

    Args:
        args: Docker arguments, never interpreted by a host shell.
        timeout: Maximum host-side wait in seconds.

    Returns:
        Exit code and decoded output.
    """
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    return process.returncode or 0, output.decode(errors="replace")


class TaskContainer:
    """Own one isolated repository shared by sequential child invocations."""

    def __init__(self, task: Task, image: str, config: DockerConfig) -> None:
        self.task = task
        self.image = image
        self.config = config
        self.downloaded = False
        self.name = f"mindmemos-swe-{uuid.uuid4().hex}"

    async def __aenter__(self) -> "TaskContainer":
        code, image_id = await docker("image", "inspect", "--format", "{{.Id}}", self.image)
        if code and self.config.pull_missing:
            async with _image_pull_lock:
                code, image_id = await docker("image", "inspect", "--format", "{{.Id}}", self.image)
                if code:
                    print(f"PULL {self.task.instance_id}", flush=True)
                    code, output = await docker("pull", "--platform", self.config.platform, self.image, timeout=3600)
                    if code:
                        raise RuntimeError(f"Cannot download task image: {self.image}: {output[-1000:]}")
                    self.downloaded = True
                    code, image_id = await docker("image", "inspect", "--format", "{{.Id}}", self.image)
        if code:
            raise RuntimeError(f"Required local image unavailable: {self.image}")
        self.image_id = image_id.strip()
        code, output = await docker(
            "run",
            "--platform",
            self.config.platform,
            "--detach",
            "--pull=never",
            "--network=none",
            "--name",
            self.name,
            "--memory",
            self.config.memory_limit,
            "--cpus",
            str(self.config.cpus),
            "--pids-limit",
            "256",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--workdir",
            self.config.workdir,
            "--entrypoint",
            "/bin/bash",
            self.image_id,
            "-c",
            "while true; do sleep 3600; done",
        )
        try:
            if code:
                raise RuntimeError(f"Cannot start task container: {output}")
            if self.config.restore_base_commit:
                code, _ = await docker(
                    "exec",
                    "--workdir",
                    self.config.workdir,
                    self.name,
                    "git",
                    "reset",
                    "--hard",
                    self.task.base_commit,
                )
                if code:
                    raise RuntimeError("Cannot restore task base commit in disposable container")
            head = await self.execute("git rev-parse HEAD")
            status = await self.execute("git status --porcelain")
            if head["exit_code"] or head["output"].strip() != self.task.base_commit:
                raise RuntimeError("Task image HEAD does not match base_commit")
            if status["exit_code"] or status["output"].strip():
                raise RuntimeError("Task image must contain a clean repository")
            return self
        except BaseException:
            await self.__aexit__()
            raise

    async def __aexit__(self, *exc: object) -> None:
        await docker("rm", "--force", self.name)
        if self.downloaded and self.config.remove_downloaded:
            code, _ = await docker("image", "rm", self.image, timeout=180)
            if code:
                raise RuntimeError(f"Cannot remove downloaded image: {self.image}")

    async def execute(self, command: str) -> dict:
        """Execute a bounded shell command inside the task repository.

        Args:
            command: Child-supplied shell command.

        Returns:
            Exit code and truncated combined output.
        """
        if self.config.command_prefix:
            command = f"{self.config.command_prefix} && {{\n{command}\n}}"
        code, output = await docker(
            "exec",
            "--workdir",
            self.config.workdir,
            self.name,
            "timeout",
            "--kill-after=5",
            str(self.config.command_timeout_seconds),
            "/bin/bash",
            "-lc",
            command,
            timeout=self.config.command_timeout_seconds + 15,
        )
        limit = self.config.output_limit_chars
        return {"exit_code": code, "output": output[:limit], "truncated": len(output) > limit}

    async def patch(self) -> str:
        """Export tracked and new-file changes relative to the original task commit.

        Returns:
            A unified binary-capable patch for the official grading harness.
        """
        code, output = await docker(
            "exec",
            "--workdir",
            self.config.workdir,
            self.name,
            "/bin/bash",
            "-lc",
            f"git add -N -- . && git diff --binary {self.task.base_commit} --",
        )
        if code:
            raise RuntimeError(f"Cannot export patch: {output}")
        return output
