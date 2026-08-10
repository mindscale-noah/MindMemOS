"""Business Skill aggregates derived from the persistence contract."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts import SkillBundle
from ..persistence.enums import SkillInjectionMode, SkillVersionOrigin, SkillVersionStatus

if TYPE_CHECKING:
    from ..persistence.models import SkillRecord


class SkillUsageType(StrEnum):
    """How a skill was used in one agent trajectory."""

    INJECTED = "injected"
    """技能注入供Agent使用"""

    MODIFIED = "modified"
    """技能修改使用"""

    UNUSED = "unused"
    """技能注入但未使用"""


def normalize_skill_text(content: str) -> str:
    """Normalize text before computing a cross-runtime Skill hash."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.rstrip("\n") + "\n"


def serialize_skill_files(files: dict[str, str]) -> str:
    """Serialize a Skill file mapping deterministically."""

    return json.dumps(files, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def compute_skill_content_hash(blob: dict[str, str]) -> str:
    """Return the canonical SHA-256 hash used by Skill persistence and bindings."""

    return SkillBundle.from_files(blob).content_hash


class Skill(BaseModel):
    """One versioned, executable Skill aggregate.

    A ``Skill`` is always one concrete version: identity, lineage, lifecycle
    and executable files travel together. ``SkillRecord`` persists the same
    aggregate as a flat row; there is deliberately no separate
    ``SkillVersion`` wrapper in the typing layer.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    skill_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    cloud_skill_id: str | None = None
    parent_version_ids: list[str] = Field(default_factory=list)
    version_label: str = Field(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
    content_hash: str = Field(min_length=1)
    status: SkillVersionStatus = SkillVersionStatus.DRAFT
    origin: SkillVersionOrigin = SkillVersionOrigin.LOCAL

    name: str = Field(min_length=1)
    """技能名称"""

    description: str | None = None
    """技能描述"""

    alias: str | None = None
    """供 CLI 或算法检索使用的可选短名称。"""

    blob: dict[str, str]
    """核心 Skill bundle；key 为相对路径，value 为文本内容。"""

    resources: dict[str, str] = Field(default_factory=dict)
    """不属于核心 Skill bundle、但算法执行时需要的辅助文本资源。"""

    commit_message: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    received_at: datetime | None = None
    version_revision: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    local_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("blob")
    @classmethod
    def normalize_bundle(cls, value: dict[str, str]) -> dict[str, str]:
        bundle = SkillBundle.from_files(value)
        return {item.path: item.content for item in bundle.files}

    @property
    def content(self) -> str:
        """Return the canonical ``SKILL.md`` content from the bundle."""

        return self.blob["SKILL.md"]

    @model_validator(mode="after")
    def validate_aggregate(self) -> Skill:
        if set(self.blob) != {"SKILL.md"}:
            raise ValueError("Skill bundle must contain exactly one SKILL.md file")
        invalid_paths = [path for path in (*self.blob, *self.resources) if not path]
        if invalid_paths:
            raise ValueError("Skill file paths must not be empty")
        if self.version_id in self.parent_version_ids:
            raise ValueError("a Skill cannot be its own parent version")
        if len(self.parent_version_ids) != len(set(self.parent_version_ids)):
            raise ValueError("parent_version_ids may not contain duplicates")
        if self.blob.keys() & self.resources.keys():
            raise ValueError("Skill bundle and resources may not contain the same path")
        if self.updated_at is None:
            object.__setattr__(self, "updated_at", self.created_at)
        return self

    def to_record(self) -> SkillRecord:
        """Flatten the aggregate into the canonical persistence row."""

        from ..persistence.models import SkillRecord

        return SkillRecord(
            skill_id=self.skill_id,
            version_id=self.version_id,
            cloud_skill_id=self.cloud_skill_id,
            parent_version_ids=self.parent_version_ids,
            name=self.name,
            description=self.description,
            alias=self.alias,
            bundle=SkillBundle.from_files(self.blob).canonical_json(),
            resources=serialize_skill_files(self.resources),
            content_hash=self.content_hash,
            local_snapshot_hash=str(
                self.local_metadata.get("local_snapshot_hash")
                or self.metadata.get("snapshot", {}).get("local_snapshot_hash")
                or self.content_hash
            ),
            status=self.status,
            version_revision=self.version_revision,
            version_label=self.version_label,
            commit_message=self.commit_message,
            metadata=self.metadata,
            local_metadata=self.local_metadata,
            created_at=self.created_at,
            updated_at=self.updated_at or self.created_at,
            received_at=self.received_at,
            origin=self.origin,
        )

    @classmethod
    def from_record(cls, record: SkillRecord) -> Skill:
        """Rebuild the business aggregate from a validated persistence row."""

        return cls(
            skill_id=record.skill_id,
            version_id=record.version_id,
            cloud_skill_id=record.cloud_skill_id,
            parent_version_ids=record.parent_version_ids,
            version_label=record.version_label,
            content_hash=record.content_hash,
            status=record.status,
            origin=record.origin,
            name=record.name,
            description=record.description,
            alias=record.alias,
            blob={item.path: item.content for item in SkillBundle.model_validate_json(record.bundle).files},
            resources=json.loads(record.resources),
            commit_message=record.commit_message,
            created_at=record.created_at,
            updated_at=record.updated_at,
            received_at=record.received_at,
            version_revision=record.version_revision,
            metadata=record.metadata,
            local_metadata=record.local_metadata,
        )


class SkillBinding(BaseModel):
    """Reference to one skill used by a single execution trajectory.

    The fields mirror the SDK/server skill-reference contract.  ``version_id``
    can be absent while the content is still pending registration; this is
    the same unresolved-binding state used by the skill trace flow.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    """技能名称"""

    content_hash: str = Field(min_length=1)
    """技能内容哈希"""

    skill_id: str | None = None
    """已注册时指向本地 Skill 家族；未注册时为空。"""

    cloud_skill_id: str | None = None
    """已同步时指向云端 Skill family；未 push 时为空。"""

    base_version_id: str | None = None
    """当前 Skill 内容派生自的版本；根版本或未知时为空。"""

    version_id: str | None = None
    """本次轨迹实际使用的不可变版本；尚未注册或绑定时为空。"""

    version_label: str | None = None
    """供算法报告展示的版本标签。"""

    usage: SkillUsageType | None = None
    """技能使用方式"""

    injection_mode: SkillInjectionMode | None = None
    """该 Skill 版本在本次轨迹中的注入机制。"""


__all__ = [
    "Skill",
    "SkillBinding",
    "SkillUsageType",
    "SkillVersionOrigin",
    "SkillVersionStatus",
    "compute_skill_content_hash",
    "normalize_skill_text",
    "serialize_skill_files",
]
