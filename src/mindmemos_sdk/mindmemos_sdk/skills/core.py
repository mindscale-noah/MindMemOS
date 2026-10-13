"""Shared request serialization for synchronous and asynchronous Skill clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .typing import SkillRequestBody, TaskProgressRequest


class SkillRoute(StrEnum):
    """Current public routes of the standalone Skill service."""

    REGISTER = "register"
    LIST = "list"
    SHOW = "show"
    COMMIT = "commit"
    HISTORY = "history"
    DIFF = "diff"
    ACTIVATE = "activate"
    UPDATE_INFO = "update-info"
    DELETE = "delete"
    VERSION_RESOLVE = "version-resolve"
    SKILL_RESOLVE = "skill-resolve"
    TRAJECTORY_REPORT = "trajectories/report"
    TRAJECTORY_SHOW = "trajectories/show"
    TRAJECTORY_LIST = "trajectories/list"
    TRAJECTORY_STATISTICS = "trajectories/statistics"
    ORGANIZE = "algos/organize"
    GRAPH_RESOLVE = "graph-resolve"
    GRAPH_DELETE = "graphs/delete"
    GRAPH_MEMBERS_ADD = "graphs/members/add"
    GRAPH_MEMBERS_REMOVE = "graphs/members/remove"
    GRAPH_PREVIEW = "graphs/preview"
    GRAPHS_LIST = "graphs/list"
    GRAPHS_SNAPSHOTS = "graphs/snapshots"
    GRAPHS_SNAPSHOT = "graphs/snapshot"
    RECOMMEND = "algos/recommend"
    EVOLVE = "algos/evolve"
    TASK_SHOW = "tasks/show"
    TRAIN = "algos/train"
    TASK_PROGRESS = "tasks/progress"


@dataclass(frozen=True)
class SkillRequest:
    """One serialized HTTP request to the Skill service."""

    path: str
    body: dict[str, Any]
    params: dict[str, Any] | None = None


class SkillCore:
    """Build Skill service requests independently of HTTP I/O."""

    @staticmethod
    def prepare(route: SkillRoute, body: SkillRequestBody) -> SkillRequest:
        """Serialize a typed request for one public Skill route.

        Args:
            route: Public operation selected by the caller.
            body: Request body matching that operation.

        Returns:
            Relative route and JSON-compatible body for the transport.
        """
        if route is SkillRoute.TASK_PROGRESS:
            if not isinstance(body, TaskProgressRequest):
                raise TypeError("Task progress requires TaskProgressRequest")
            return SkillRequest(
                path=f"/v1/skills/{route.value}",
                body={"task_id": body.task_id},
                params={"after": body.after},
            )
        return SkillRequest(path=f"/v1/skills/{route.value}", body=body.model_dump(mode="json", exclude_unset=True))
