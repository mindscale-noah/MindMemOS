"""Execute SkillsBench through the external harbor run command and adapt its native artifacts."""

import os
from pathlib import Path
from tempfile import mkdtemp

from pydantic import JsonValue

from ....agents.buitin_agents.constants import NATIVE_CONNECTION_ENV
from ....common.checkpoint import fingerprint
from ....common.json import read_json, write_json
from ....common.secrets import redact_credentials
from ....infra.execution import ExecutionConfig, execute
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import Reward, RolloutResult, Task, Usage
from .config import SkillsBenchConfig
from .constants import HARBOR_VERSION, SKILLSBENCH_REVISION
from .skills import prepare_task
from .source import regular_file, task_identity
from .typing import AtifTrajectory, HarborTrialResult, SkillsBenchRewardData

MODEL_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_", "CODEX_")
"""Ambient native model aliases replaced by this invocation's explicitly selected variables."""


@Environment.register("skillsbench")
class SkillsBenchEnvironment(Environment[SkillsBenchConfig]):
    """Reuse the rollout lifecycle while Harbor owns containers, agents and official verification."""

    config_type = SkillsBenchConfig
    """Explicit external CLI and model connection variable names."""

    reward_data_type = SkillsBenchRewardData
    """Private task location and immutable source checksum contract."""

    def input_identity(self, task: Task, reward_data: RewardData) -> dict[str, str]:
        """Bind source files and public instructions to both execution and resume.

        Args:
            task: Exact original benchmark ID and instruction.
            reward_data: Original private package and expected file identities.

        Returns:
            Verified source file fingerprints for the base lifecycle.
        """
        if not isinstance(reward_data, SkillsBenchRewardData):
            raise TypeError("SkillsBench requires SkillsBenchRewardData")
        identity = task_identity(reward_data)
        root = reward_data.task_dir.expanduser().resolve()
        if task.id != root.name or task.instruction != (root / "instruction.md").read_text(encoding="utf-8"):
            raise ValueError("SkillsBench public task does not match its original package")
        identity["harbor:version"] = HARBOR_VERSION
        identity["skillsbench:revision"] = SKILLSBENCH_REVISION
        if self.config.agent != "oracle":
            identity["harbor:environment"] = fingerprint(
                redact_credentials(
                    {name: os.environ.get(name) for name in self.config.agent_env},
                    preserve_execution_settings=True,
                )
            )
        return identity

    def _execution(self, timeout_s: float) -> ExecutionConfig:
        """Preserve system/Docker settings while replacing ambient native model connection variables."""
        return ExecutionConfig(
            timeout_s=timeout_s,
            excluded_environment=[
                name for name in os.environ if name in NATIVE_CONNECTION_ENV or name.startswith(MODEL_ENV_PREFIXES)
            ],
            environment={
                name: os.environ[name]
                for name in self.config.agent_env
                if self.config.agent != "oracle" and name in os.environ
            },
        )

    async def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Generate a one-task job, invoke the pinned CLI, and require a real verifier result."""
        if not isinstance(reward_data, SkillsBenchRewardData):
            raise TypeError("SkillsBench requires SkillsBenchRewardData")
        version = await execute([*self.config.harbor_command, "--version"], workspace, self._execution(30))
        if version.returncode or version.stdout.strip() != HARBOR_VERSION:
            raise RuntimeError(f"SkillsBench requires harbor=={HARBOR_VERSION}")
        attempt = Path(mkdtemp(prefix="harbor-", dir=workspace.parent / "audit"))
        prepared = attempt / "tasks" / task.id
        exported = prepare_task(reward_data.task_dir, prepared, skills, oracle=self.config.agent == "oracle")
        agent = {"name": self.config.agent, "skills": [str(path) for path in exported]}
        if self.config.model is not None:
            agent["model_name"] = self.config.model
        if self.config.agent_version is not None:
            agent["kwargs"] = {"version": self.config.agent_version}
        if self.config.agent != "oracle":
            agent["env"] = {name: "${" + name + "}" for name in self.config.agent_env if name in os.environ}
        job = {
            "job_name": "evaluation",
            "jobs_dir": str(attempt / "jobs"),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "quiet": True,
            "retry": {"max_retries": 0},
            "agent_setup_timeout_multiplier": 5,
            "environment": {"type": "docker", "delete": True},
            "agents": [agent],
            "tasks": [{"path": str(prepared)}],
        }
        config_path = attempt / "job.json"
        write_json(config_path, job)
        with (attempt / "stdout.log").open("wb") as output:
            completed = await execute(
                [*self.config.harbor_command, "run", "--config", str(config_path)],
                workspace,
                self._execution(self.config.timeout_s),
                output=output,
            )
        (attempt / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"harbor run failed with status {completed.returncode}; inspect {attempt}")
        return _read_result(attempt / "jobs" / "evaluation", task, self.config.agent, workspace.parent)


def _read_result(job: Path, task: Task, agent: str, root: Path) -> RolloutResult:
    """Require exactly one complete trial and expose only public agent interactions."""
    paths = list(job.glob("*/result.json"))
    if len(paths) != 1:
        raise ValueError("Harbor must produce exactly one trial result")
    path = regular_file(paths[0], job)
    native = HarborTrialResult.model_validate(read_json(path))
    if native.task_name != task.id:
        raise ValueError("Harbor returned a different task identity")
    if native.exception_info is not None:
        error = native.exception_info
        raise RuntimeError(f"Harbor {error.exception_type}: {error.exception_message}")
    rewards = native.verifier_result.rewards if native.verifier_result is not None else None
    if not rewards or "reward" not in rewards or not 0 <= rewards["reward"] <= 1:
        raise ValueError("Harbor did not produce a valid official SkillsBench reward")
    trajectory_path = path.parent / "agent" / "trajectory.json"
    if agent == "oracle":
        messages = [{"role": "user", "content": task.instruction}]
        usage = Usage(reported=True)
    else:
        trajectory = AtifTrajectory.model_validate(read_json(regular_file(trajectory_path, job)))
        messages = _messages(trajectory)
        reported = native.agent_result
        known = reported is not None and reported.n_input_tokens is not None and reported.n_output_tokens is not None
        usage = Usage(
            input_tokens=reported.n_input_tokens if known else 0,
            output_tokens=reported.n_output_tokens if known else 0,
            total_tokens=reported.n_input_tokens + reported.n_output_tokens if known else 0,
            reported=known,
            cache_read_input_tokens=reported.n_cache_tokens if reported is not None else None,
            cost_usd=reported.cost_usd if reported is not None else None,
        )
    response = next((message["content"] for message in reversed(messages) if message["role"] == "assistant"), "")
    return RolloutResult(
        output={"response": response, "harbor_trial_dir": str(path.parent.relative_to(root))},
        trajectory=messages,
        usage=usage,
        reward=Reward(
            score=rewards["reward"],
            success=rewards["reward"] == 1,
            detail="Official SkillsBench verifier reward",
            agent=agent,
            harbor_version=HARBOR_VERSION,
            dataset_revision=SKILLSBENCH_REVISION,
        ),
    )


def _messages(trajectory: AtifTrajectory) -> list[dict[str, JsonValue]]:
    """Convert public ATIF messages and tool results without copying private native fields."""
    messages = []
    for step in trajectory.steps:
        message = {"role": "assistant" if step.source == "agent" else step.source, "content": step.message}
        if step.tool_calls:
            message["tool_calls"] = [call.model_dump(mode="json") for call in step.tool_calls]
        messages.append(message)
        if step.observation is not None:
            for result in step.observation.results:
                messages.append({"role": "tool", "content": result.content, "tool_call_id": result.source_call_id})
    return messages
