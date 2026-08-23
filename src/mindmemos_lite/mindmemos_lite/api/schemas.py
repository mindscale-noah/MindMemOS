"""Pydantic schemas owned by the optional HTTP transport."""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

T = TypeVar("T")
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextMessageInput(_StrictModel):
    text: NonEmptyStr


class UrlMessageInput(_StrictModel):
    url: NonEmptyStr


class FileMessageInput(_StrictModel):
    file_name: NonEmptyStr
    file_path: NonEmptyStr
    file_type: str = ""


class DialogueMessageInput(_StrictModel):
    role: NonEmptyStr
    content: NonEmptyStr
    timestamp: int | None = None


MemoryMessageInput = DialogueMessageInput | UrlMessageInput | FileMessageInput | TextMessageInput


class SkillContextInput(_StrictModel):
    name: NonEmptyStr
    content_hash: NonEmptyStr
    base_version_id: str = ""
    version_label: NonEmptyStr | None = None
    usage: Literal["injected", "modified"] | None = None


class ActorIdentityRequest(_StrictModel):
    user_id: NonEmptyStr | None = None
    app_id: NonEmptyStr | None = None
    session_id: NonEmptyStr | None = None
    agent_id: NonEmptyStr | None = None


class AddRequest(ActorIdentityRequest):
    messages: list[MemoryMessageInput] = Field(min_length=1)
    mode: Literal["sync", "async"] = "sync"
    infer: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_context: list[SkillContextInput] = Field(default_factory=list)
    score: float | None = None
    task_id: NonEmptyStr | None = None
    task: NonEmptyStr | None = None


class SearchRequest(ActorIdentityRequest):
    query: NonEmptyStr
    memory_mode: NonEmptyStr | None = None
    filters: dict[str, Any] | None = None
    top_k: int | None = Field(default=10, ge=1)
    search_strategy: Literal["fast", "agentic"] = "fast"
    rerank: bool = False
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    max_rounds: int = Field(default=3, ge=1)
    task_top_k: int | None = Field(default=None, ge=1)


class GetRequest(_StrictModel):
    filters: dict[str, Any] | None = None
    top_k: int | None = Field(default=None, ge=1)


class DeleteRequest(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    memory_id: NonEmptyStr = Field(alias="id")


class UpdateRequest(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    memory_id: NonEmptyStr = Field(alias="id")
    content: NonEmptyStr


class MemoryLineageInput(_StrictModel):
    role: Literal["current", "archived"] = "current"
    derived_from_memory_ids: list[str] = Field(default_factory=list)
    derived_to_memory_ids: list[str] = Field(default_factory=list)


class RecalledMemoryInput(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    memory_id: NonEmptyStr = Field(alias="id")
    content: NonEmptyStr = Field(alias="memory")
    memory_type: str = "fact"
    last_update_at: str | None = None
    event_time: str | None = None
    source_timestamp: str | None = None
    lineage: MemoryLineageInput | None = None


class FeedbackRequest(ActorIdentityRequest):
    feedback: NonEmptyStr | None = None
    messages: list[MemoryMessageInput] = Field(default_factory=list)
    recalled_memories: list[RecalledMemoryInput] = Field(default_factory=list)
    mode: Literal["sync", "async"] = "sync"


class DreamingRequest(ActorIdentityRequest):
    mode: Literal["sync", "async"] = "async"


class MemoryAddEventResponse(BaseModel):
    operation: str
    content: str
    memory_id: str | None = None
    memory_type: str | None = None
    confidence: float | None = None
    related_memory_ids: list[str] = Field(default_factory=list)
    graph_edge_count: int = 0


class MemoryLineageResponse(BaseModel):
    role: Literal["current", "archived"] = "current"
    derived_from_memory_ids: list[str] = Field(default_factory=list)
    derived_to_memory_ids: list[str] = Field(default_factory=list)


class MemoryItemResponse(BaseModel):
    id: str
    memory: str
    memory_type: str = "fact"
    last_update_at: str | None = None
    event_time: str | None = None
    source_timestamp: str | None = None
    lineage: MemoryLineageResponse | None = None


class AddData(BaseModel):
    memories: list[MemoryAddEventResponse] = Field(default_factory=list)


class MemoryListData(BaseModel):
    memories: list[MemoryItemResponse] = Field(default_factory=list)
    task: TaskEntityResponse | None = None
    """When the configured search pipeline matched a task entity, its identity."""
    tasks: list[TaskSearchGroupData] = Field(default_factory=list)
    """Task experience search: every matched task plus its one-hop experiences."""


class TaskEntityResponse(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str = "task"


class TaskSearchGroupData(BaseModel):
    task: TaskEntityResponse
    memories: list[MemoryItemResponse] = Field(default_factory=list)


class ApiResponse(BaseModel, Generic[T]):
    code: str = "ok"
    message: str = ""
    request_id: str | None = None
    data: T | None = None


__all__ = [
    "AddData",
    "AddRequest",
    "ApiResponse",
    "DeleteRequest",
    "DreamingRequest",
    "FeedbackRequest",
    "GetRequest",
    "MemoryListData",
    "SearchRequest",
    "TaskEntityResponse",
    "TaskSearchGroupData",
    "UpdateRequest",
]
