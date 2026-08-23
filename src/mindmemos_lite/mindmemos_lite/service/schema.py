"""Transport-neutral service commands and results.

The public ``mindmemos_lite`` API is currently exposed through FastAPI/Pydantic
models.  Lite keeps the same capability and business semantics, but the
service boundary is deliberately independent of a transport:

* adapters convert HTTP/SDK input into the command objects below;
* service ports return result objects below;
* adapters convert those results into the legacy API envelope and field names.

This module therefore must not import FastAPI, Pydantic, a database driver, or
an LLM client.  In particular, ``filters`` stays as a public filter mapping;
parsing it into a backend-specific expression belongs to the application
adapter/pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal, TypeAlias

ExecutionMode: TypeAlias = Literal["sync", "async"]
OperationStatus: TypeAlias = Literal["ok", "error", "queued"]
SearchStrategy: TypeAlias = Literal["fast", "agentic"]
MemoryType: TypeAlias = Literal[
    "profile",
    "fact",
    "experience",
    "episodic",
    "tool_trace",
    "skill_candidate",
    "file_knowledge",
]
MemoryOperation: TypeAlias = Literal["add", "delete", "update", "reinforcement", "merge"]
SkillVersionStatus: TypeAlias = Literal[
    "observed",
    "draft",
    "evaluating",
    "published",
    "superseded",
    "rolled_back",
]
SkillOrigin: TypeAlias = Literal["edge", "cloud"]
SkillUsage: TypeAlias = Literal["injected", "modified"]


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_optional_text(value: str | None, field_name: str) -> None:
    if value is not None:
        _require_text(value, field_name)


def _file_type_from_path(file_path: str) -> str:
    path = file_path.split("?", 1)[0].split("#", 1)[0]
    suffix = PurePosixPath(path).suffix
    return suffix[1:].lower() if suffix else ""


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestContext:
    """Security, isolation, and optional actor context for one service call.

    ``request_id`` through ``api_key_uuid`` correspond to the trusted fields
    assembled by the original API dependency.  Actor fields are intentionally
    separate and remain optional because get/delete/update do not accept them.
    The context is shared by memory, skill, and internal ports; a transport
    adapter is responsible for authenticating it before invoking a port.
    """

    request_id: str
    account_id: str
    project_id: str
    api_key_uuid: str
    memory_algorithm: str | None = None
    user_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    scopes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("request_id", "account_id", "project_id", "api_key_uuid"):
            _require_text(getattr(self, field_name), field_name)
        for field_name in ("memory_algorithm", "user_id", "app_id", "session_id", "agent_id"):
            _validate_optional_text(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class TextMessage:
    """A plain text message accepted by add and explicit feedback."""

    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, "text")


@dataclass(frozen=True, slots=True, kw_only=True)
class UrlMessage:
    """A URL reference accepted by add and explicit feedback."""

    url: str

    def __post_init__(self) -> None:
        _require_text(self.url, "url")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileMessage:
    """A file reference accepted by add and explicit feedback."""

    file_name: str
    file_path: str
    file_type: str = ""

    def __post_init__(self) -> None:
        _require_text(self.file_name, "file_name")
        _require_text(self.file_path, "file_path")
        if not self.file_type:
            object.__setattr__(self, "file_type", _file_type_from_path(self.file_path))


@dataclass(frozen=True, slots=True, kw_only=True)
class DialogueMessage:
    """One conversational turn accepted by add and explicit feedback."""

    role: str
    content: str
    timestamp: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.role, "role")
        _require_text(self.content, "content")


MemoryMessage: TypeAlias = TextMessage | UrlMessage | FileMessage | DialogueMessage


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillContext:
    """A skill reference attached to one memory-add trajectory."""

    name: str
    content_hash: str
    base_version_id: str = ""
    version_label: str | None = None
    usage: SkillUsage | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.content_hash, "content_hash")
        _validate_optional_text(self.version_label, "version_label")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryLineage:
    """Version-lineage metadata returned with a memory item."""

    role: Literal["current", "archived"] = "current"
    derived_from_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    derived_to_memory_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryAddEvent:
    """One memory mutation emitted by an add operation."""

    operation: MemoryOperation
    content: str
    memory_id: str | None = None
    memory_type: MemoryType | str | None = None
    confidence: float | None = None
    related_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    graph_edge_count: int = 0

    def __post_init__(self) -> None:
        _require_text(self.content, "content")
        _validate_optional_text(self.memory_id, "memory_id")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.graph_edge_count < 0:
            raise ValueError("graph_edge_count must not be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryItem:
    """A business-visible memory returned by get/search.

    Time values are domain ``datetime`` objects.  The HTTP adapter formats
    them as the legacy ``%Y-%m-%d %H:%M:%S`` strings and maps ``memory_id`` /
    ``content`` to the legacy ``id`` / ``memory`` response fields.
    """

    memory_id: str
    content: str
    memory_type: MemoryType | str = "fact"
    updated_at: datetime | None = None
    event_time: datetime | None = None
    source_timestamp: datetime | None = None
    lineage: MemoryLineage | None = None

    def __post_init__(self) -> None:
        _require_text(self.memory_id, "memory_id")
        _require_text(self.content, "content")


@dataclass(frozen=True, slots=True, kw_only=True)
class AddMemoryRequest:
    """Command for ``POST /v1/memory/add``."""

    messages: tuple[MemoryMessage, ...]
    mode: ExecutionMode = "sync"
    infer: bool = True  # 是否启用大模型推理，不启用直接把原始message拼接后入库
    metadata: Mapping[str, Any] = field(default_factory=dict)
    skill_context: tuple[SkillContext, ...] = field(default_factory=tuple)
    score: float | None = None
    task_id: str | None = None
    task: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty")
        _validate_optional_text(self.task_id, "task_id")
        _validate_optional_text(self.task, "task")


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchMemoryRequest:
    """Command for ``POST /v1/memory/search``."""

    query: str
    memory_mode: str | None = None
    filters: Mapping[str, Any] | None = None
    top_k: int | None = 10
    search_strategy: SearchStrategy = "fast"
    rerank: bool = False
    score_threshold: float | None = None
    max_rounds: int = 3
    task_top_k: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.query, "query")
        _validate_optional_text(self.memory_mode, "memory_mode")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when provided")
        if self.task_top_k is not None and self.task_top_k < 1:
            raise ValueError("task_top_k must be positive when provided")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class GetMemoryRequest:
    """Command for ``POST /v1/memory/get``."""

    filters: Mapping[str, Any] | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when provided")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeleteMemoryRequest:
    """Command for ``POST /v1/memory/delete``."""

    memory_id: str

    def __post_init__(self) -> None:
        _require_text(self.memory_id, "memory_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateMemoryRequest:
    """Command for ``POST /v1/memory/update``."""

    memory_id: str
    content: str

    def __post_init__(self) -> None:
        _require_text(self.memory_id, "memory_id")
        _require_text(self.content, "content")


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackMemoryRequest:
    """Command for ``POST /v1/memory/feedback``."""

    feedback: str | None = None
    messages: tuple[MemoryMessage, ...] = field(default_factory=tuple)
    recalled_memories: tuple[MemoryItem, ...] = field(default_factory=tuple)
    mode: ExecutionMode = "sync"

    def __post_init__(self) -> None:
        _validate_optional_text(self.feedback, "feedback")


@dataclass(frozen=True, slots=True, kw_only=True)
class DreamingMemoryRequest:
    """Command for ``POST /v1/memory/dreaming``."""

    mode: ExecutionMode = "async"


@dataclass(frozen=True, slots=True, kw_only=True)
class AddMemoryResult:
    status: OperationStatus
    memories: tuple[MemoryAddEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryListResult:
    status: OperationStatus
    memories: tuple[MemoryItem, ...] = field(default_factory=tuple)
    message: str | None = None
    task_id: str | None = None
    task_name: str | None = None
    """When the configured search pipeline matched a task entity, its identity."""
    tasks: tuple["MemoryTaskGroup", ...] = field(default_factory=tuple)
    """Task experience search: every matched task plus its one-hop experiences.

    Experiences are scoped per task (shared experiences repeat across tasks).
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryTaskGroup:
    """One matched task together with its one-hop experiences."""

    task_id: str
    task_name: str
    memories: tuple[MemoryItem, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryMutationResult:
    status: OperationStatus
    message: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackAddAction:
    after_content: str
    result_memory_id: str | None = None
    reason: str | None = None
    status: OperationStatus = "ok"
    action: Literal["add"] = "add"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackUpdateAction:
    target_memory_id: str
    before_content: str
    after_content: str
    result_memory_id: str | None = None
    reason: str | None = None
    status: OperationStatus = "ok"
    action: Literal["update"] = "update"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackDeleteAction:
    target_memory_id: str
    before_content: str
    result_memory_id: str | None = None
    reason: str | None = None
    status: OperationStatus = "ok"
    action: Literal["delete"] = "delete"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackNoopAction:
    target_memory_id: str | None = None
    before_content: str | None = None
    reason: str | None = None
    status: OperationStatus = "ok"
    action: Literal["noop"] = "noop"


FeedbackAction: TypeAlias = FeedbackAddAction | FeedbackUpdateAction | FeedbackDeleteAction | FeedbackNoopAction


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackMemoryResult:
    status: OperationStatus
    message: str | None = None
    actions: tuple[FeedbackAction, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillVersion:
    """Metadata for one skill version in the project-scoped lineage."""

    version_id: str
    project_id: str
    cloud_skill_id: str
    skill_name: str
    content_hash: str
    status: SkillVersionStatus
    origin: SkillOrigin
    created_at: datetime
    parent_version_id: str | None = None
    version_label: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSummary:
    cloud_skill_id: str
    skill_name: str
    latest_version: SkillVersion
    published_head: SkillVersion | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillContent:
    version: SkillVersion
    content: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisterSkillRequest:
    """Command for ``POST /v1/skills/register``."""

    name: str
    content: str
    version_label: str | None = None
    parent_version_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.content, "content")
        _validate_optional_text(self.version_label, "version_label")
        _validate_optional_text(self.parent_version_id, "parent_version_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisterSkillResult:
    """The public data returned by ``/v1/skills/register``."""

    cloud_skill_id: str
    version_id: str
    content_hash: str
    status: SkillVersionStatus
    version_label: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvolveSkillRequest:
    """Command for ``POST /v1/skills/evolve``."""

    cloud_skill_id: str
    mode: ExecutionMode = "sync"

    def __post_init__(self) -> None:
        _require_text(self.cloud_skill_id, "cloud_skill_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillEvolveResult:
    cloud_skill_id: str
    evolved: bool
    pending_count: int
    threshold: int
    status: Literal["ok", "queued"] = "ok"
    new_version_id: str | None = None
    new_version_ids: tuple[str, ...] = field(default_factory=tuple)
    summarized_count: int = 0
    consumed_count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSyncItem:
    cloud_skill_id: str
    local_version_id: str

    def __post_init__(self) -> None:
        _require_text(self.cloud_skill_id, "cloud_skill_id")
        _require_text(self.local_version_id, "local_version_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class SyncSkillsRequest:
    """Command for the top-level-array ``POST /v1/skills/sync`` body."""

    items: tuple[SkillSyncItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("items must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSyncResult:
    cloud_skill_id: str
    local_version_id: str
    has_update: bool
    gating_status: str
    published_head: SkillVersion | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InternalMemoryListRequest:
    """Query for ``GET /internal/v1/projects/{project_id}/memories``."""

    project_id: str
    query: str | None = None
    limit: int = 50
    cursor: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.project_id, "project_id")
        if not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        _validate_optional_text(self.cursor, "cursor")


@dataclass(frozen=True, slots=True, kw_only=True)
class InternalMemoryDetailRequest:
    """Query for ``GET /internal/v1/projects/{project_id}/memories/{memory_id}``."""

    project_id: str
    memory_id: str

    def __post_init__(self) -> None:
        _require_text(self.project_id, "project_id")
        _require_text(self.memory_id, "memory_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class InternalMemoryPage:
    items: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    next_cursor: str | None = None


__all__ = [
    "AddMemoryRequest",
    "AddMemoryResult",
    "DialogueMessage",
    "DeleteMemoryRequest",
    "DreamingMemoryRequest",
    "EvolveSkillRequest",
    "ExecutionMode",
    "FeedbackAction",
    "FeedbackAddAction",
    "FeedbackDeleteAction",
    "FeedbackMemoryRequest",
    "FeedbackMemoryResult",
    "FeedbackNoopAction",
    "FeedbackUpdateAction",
    "FileMessage",
    "GetMemoryRequest",
    "InternalMemoryDetailRequest",
    "InternalMemoryListRequest",
    "InternalMemoryPage",
    "MemoryAddEvent",
    "MemoryItem",
    "MemoryLineage",
    "MemoryMessage",
    "MemoryMutationResult",
    "MemoryOperation",
    "MemoryType",
    "MemoryListResult",
    "OperationStatus",
    "RegisterSkillRequest",
    "RegisterSkillResult",
    "RequestContext",
    "SearchMemoryRequest",
    "SearchStrategy",
    "SkillContent",
    "SkillEvolveResult",
    "SkillOrigin",
    "SkillSummary",
    "SkillSyncItem",
    "SkillSyncResult",
    "SkillUsage",
    "SkillVersion",
    "SkillVersionStatus",
    "SkillContext",
    "SyncSkillsRequest",
    "TextMessage",
    "UpdateMemoryRequest",
    "UrlMessage",
]
