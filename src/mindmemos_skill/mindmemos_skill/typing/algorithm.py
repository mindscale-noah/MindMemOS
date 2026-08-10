"""Algorithm identity and step-report aggregates."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, JsonValue

if TYPE_CHECKING:
    from ..persistence.models import AlgorithmLogRecord


class AlgorithmIdentity(BaseModel):
    """Stable description of the algorithm implementation producing a report."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    """算法名称，例如 trace_summary、skillopt 或 merge_resolver。"""

    version: str | None = None
    """算法实现、配置或 Prompt 协议版本。"""


class AlgorithmStep(BaseModel):
    """One component step emitted during an algorithm execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    component_name: str = Field(min_length=1)
    """上报步骤的算法组件。"""

    name: str = Field(min_length=1)
    """组件内注册的步骤名称。"""

    status: str | None = None
    """started、succeeded、rejected、failed 等组件自定义状态。"""

    payload: dict[str, JsonValue] = Field(default_factory=dict)
    """该步骤的输入、输出、指标、决策、错误和 artifact 引用。"""

    created_at: datetime
    """步骤报告产生时间。"""


class AlgorithmLog(BaseModel):
    """Algorithm-facing report aggregate stored as one flat log row.

    ``AlgorithmLogRecord`` flattens ``algorithm`` and ``step`` into columns.
    This object deliberately represents one step only because the persistence
    contract currently has no algorithm-run identifier for grouping rows.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    log_id: str = Field(min_length=1)
    algorithm: AlgorithmIdentity
    step: AlgorithmStep

    def to_record(self) -> AlgorithmLogRecord:
        """Flatten the aggregate into the canonical persistence row."""

        from ..persistence.models import AlgorithmLogRecord

        return AlgorithmLogRecord(
            log_id=self.log_id,
            algorithm_name=self.algorithm.name,
            algorithm_version=self.algorithm.version,
            component_name=self.step.component_name,
            step_name=self.step.name,
            status=self.step.status,
            payload=self.step.payload,
            created_at=self.step.created_at,
        )

    @classmethod
    def from_record(cls, record: AlgorithmLogRecord) -> AlgorithmLog:
        """Rebuild the business aggregate from a validated persistence row."""

        return cls(
            log_id=record.log_id,
            algorithm=AlgorithmIdentity(name=record.algorithm_name, version=record.algorithm_version),
            step=AlgorithmStep(
                component_name=record.component_name,
                name=record.step_name,
                status=record.status,
                payload=record.payload,
                created_at=record.created_at,
            ),
        )


__all__ = ["AlgorithmIdentity", "AlgorithmLog", "AlgorithmStep"]
