"""Base lifecycle for one benchmark rollout attempt."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, JsonValue

from ..agents.base import Agent
from ..typing import AgentExecutionRequest, EnvConfig, Environment, Reward, Rollout, Skill, Task, Trajectory

EnvConfigT = TypeVar("EnvConfigT", bound=EnvConfig)


@dataclass(slots=True)
class EnvRolloutContext:
    """Trainer-owned context for one physical rollout attempt."""

    rollout: Rollout
    """本次物理执行所属的逻辑 rollout 及重试序号；其 ``rollout_id`` 和 ``attempt_no`` 也参与工作目录拼接。"""

    workspace_root: Path | None = None
    """所有 attempt 工作目录的根路径；为 ``None`` 时不创建目录，最终 ``Environment.running_dir`` 也为 ``None``。"""

    workspace_scope: str = "rollout"
    """根路径下的业务隔离层级，例如 ``train/epoch 1`` 会转换为 ``train/epoch_1`` 两级目录。

    最终路径为 ``workspace_root / safe(scope各段) / safe(task_id) /
    safe(rollout_id) / attempt_no``。scope 按 ``/`` 分段，空段、``.`` 和
    ``..`` 被忽略；每段中的非 ``A-Z a-z 0-9 _ . -`` 字符替换为 ``_``。
    """

    agent_options: dict[str, JsonValue] = field(default_factory=dict)
    """仅作用于本次 attempt 的 Agent 执行参数覆盖，原样传入 ``AgentExecutionRequest.options``。"""

    metadata: dict[str, JsonValue] = field(default_factory=dict)
    """Trainer 附加的可序列化上下文；原样传入请求，并写入环境元数据（环境中另补 ``workspace_scope``）。"""

    env_ref: str = "unknown"
    """当前物理 attempt 使用的注册 Env 名称；由 scheduler 从 ``RolloutSpec.env_ref`` 传入。"""


@dataclass(slots=True)
class PreparedRollout:
    """Environment state prepared for one rollout attempt.

    ``runtime_state`` may hold non-serializable objects such as a simulator or
    sidecar client. Only ``environment`` is attached to the persisted
    trajectory.
    """

    agent_request: AgentExecutionRequest
    environment: Environment
    runtime_state: Any = None


class BaseEnv(ABC, Generic[EnvConfigT]):
    """Execute and evaluate one rollout attempt.

    Batch planning, sample fan-out and concurrency belong to the trainer. The
    environment owns per-attempt preparation, execution adaptation, evaluation
    and teardown.
    """

    config_type: type[EnvConfig] = EnvConfig

    def __init__(self, config: EnvConfigT | Mapping[str, Any]) -> None:
        raw_config = config.model_dump() if isinstance(config, BaseModel) else config
        self.config = cast(EnvConfigT, self.config_type.model_validate(raw_config))

    async def setup(self) -> None:
        """Initialize instance-level resources before rollouts begin."""

    async def cleanup(self) -> None:
        """Release instance-level resources after all rollouts finish."""

    async def __aenter__(self) -> BaseEnv[EnvConfigT]:
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.cleanup()

    async def rollout(
        self,
        agent: Agent[Any],
        task: Task,
        skills: Sequence[Skill],
        *,
        context: EnvRolloutContext,
    ) -> Trajectory:
        """Execute and evaluate one physical rollout attempt."""

        prepared = await self._prepare(task=task, skills=skills, context=context)
        try:
            trajectory = await self._execute(agent=agent, prepared=prepared)

            trajectory.reward = await self._evaluate(
                trajectory=trajectory,
                prepared=prepared,
            )
            return trajectory
        finally:
            await self._teardown(prepared)

    async def _prepare(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: EnvRolloutContext,
    ) -> PreparedRollout:
        """Create the default workspace and agent execution request."""

        workspace = self._create_workspace(task=task, context=context)
        running_dir = str(workspace) if workspace is not None else None
        environment = Environment(
            env_ref=context.env_ref,
            running_dir=running_dir,
            metadata={**context.metadata, "workspace_scope": context.workspace_scope},
        )
        return PreparedRollout(
            agent_request=AgentExecutionRequest(
                task=task,
                rollout=context.rollout,
                environment=environment,
                skills=list(skills),
                options=context.agent_options,
                metadata=context.metadata,
            ),
            environment=environment,
        )

    async def _execute(
        self,
        *,
        agent: Agent[Any],
        prepared: PreparedRollout,
    ) -> Trajectory:
        """Execute through the generic Agent interface.

        Interactive simulator environments may override this hook while
        retaining the public rollout lifecycle.
        """

        return await agent.execute(prepared.agent_request)

    @abstractmethod
    async def _evaluate(
        self,
        *,
        trajectory: Trajectory,
        prepared: PreparedRollout,
    ) -> Reward:
        """Return the benchmark-specific reward for one trajectory."""

    async def _teardown(self, prepared: PreparedRollout) -> None:
        """Release resources owned by one attempt without deleting artifacts."""

    def _create_workspace(
        self,
        *,
        task: Task,
        context: EnvRolloutContext,
    ) -> Path | None:
        """Create an isolated workspace for one rollout attempt.

        Existing directories fail fast so retries and concurrent samples cannot
        silently overwrite artifacts.
        """

        if context.workspace_root is None:
            return None

        scope_parts = [
            self._safe_path_part(part)
            for part in context.workspace_scope.split("/")
            if part and part not in {".", ".."}
        ]
        if not scope_parts:
            raise ValueError("workspace_scope must contain at least one safe path part")

        workspace = (
            context.workspace_root
            / Path(*scope_parts)
            / self._safe_path_part(task.task_id)
            / self._safe_path_part(context.rollout.rollout_id)
            / str(context.rollout.attempt_no)
        )
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

    @staticmethod
    def _safe_path_part(value: str) -> str:
        """Return a non-empty filesystem-safe path component."""

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return safe or "item"


__all__ = [
    "BaseEnv",
    "EnvConfigT",
    "EnvRolloutContext",
    "PreparedRollout",
]
