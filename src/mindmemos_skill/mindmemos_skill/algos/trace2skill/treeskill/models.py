"""Strict internal and report contracts for TreeSkill."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....typing import Trace2SkillOutput


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnalysisItem(_StrictModel):
    """One structured finding extracted from a physical trajectory."""

    item_id: str = Field(min_length=1)
    kind: Literal["failure_cause", "failure_memory", "success_memory", "unlabeled_memory"]
    number: int | None = Field(default=None, ge=1)
    title: str = ""
    description: str = ""
    content: str = ""
    relation_to_skill: str = ""
    skill_reflection: str = ""


class TrajectoryAnalysisRecord(_StrictModel):
    """Outcome-aware analysis record consumed by evidence localization."""

    instance_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    record_source: Literal["error", "success", "unlabeled"]
    source_file: str = ""
    items: tuple[AnalysisItem, ...] = ()


class LocatedEvidence(_StrictModel):
    """One atomic reusable lesson assigned to one initial tree node."""

    instance_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    record_source: Literal["error", "success", "unlabeled"]
    reusable_lesson: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class LocalizationResult(_StrictModel):
    """Validated localization result for one trajectory analysis record."""

    instance_id: str = Field(min_length=1)
    evidence: tuple[LocatedEvidence, ...] = ()


class NewChildSpec(_StrictModel):
    """One recursively structured new child requested by node fusion."""

    heading: str = Field(min_length=1)
    content: str = ""
    children: tuple[NewChildSpec, ...] = ()


class NodeFusionEdit(_StrictModel):
    """One model-proposed node-local edit."""

    operation: Literal["update_node", "create_child", "reject"]
    rationale: str = Field(min_length=1)
    content: str | None = None
    new_child: NewChildSpec | None = None

    @model_validator(mode="after")
    def validate_operation_payload(self) -> NodeFusionEdit:
        if self.operation == "update_node":
            if self.content is None or self.new_child is not None:
                raise ValueError("update_node requires content and must not include new_child")
        elif self.operation == "create_child":
            if self.new_child is None or self.content is not None:
                raise ValueError("create_child requires new_child and must not include content")
        elif self.content is not None or self.new_child is not None:
            raise ValueError("reject must not include content or new_child")
        return self


class NodeFusionDecision(_StrictModel):
    """Complete response for one target-node fusion call."""

    rationale: str = Field(min_length=1)
    edits: tuple[NodeFusionEdit, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_edit_set(self) -> NodeFusionDecision:
        operations = [edit.operation for edit in self.edits]
        if "reject" in operations and len(operations) != 1:
            raise ValueError("reject cannot be mixed with another operation")
        if len(operations) != len(set(operations)):
            raise ValueError("node fusion may return each operation at most once")
        return self


class AppliedEditRecord(_StrictModel):
    """Auditable result of validating and applying one fusion edit."""

    target_node_id: str
    operation: str
    accepted: bool
    message: str


class LocalizationFailure(_StrictModel):
    instance_id: str
    error: str


class FusionFailure(_StrictModel):
    target_node_id: str
    error: str


class TreeSkillReport(_StrictModel):
    """Compact audit report for one TreeSkill evolution transaction."""

    algorithm_name: str = "treeskill"
    run_id: str = Field(min_length=1)
    algorithm_version: str
    prompt_version: str
    input_trajectory_ids: tuple[str, ...]
    duplicate_trajectory_ids: tuple[str, ...] = ()
    failed_analysis_trajectory_ids: tuple[str, ...] = ()
    localization_failures: tuple[LocalizationFailure, ...] = ()
    fusion_failures: tuple[FusionFailure, ...] = ()
    analysis_records: tuple[TrajectoryAnalysisRecord, ...] = ()
    located_evidence: tuple[LocatedEvidence, ...] = ()
    applied_edits: tuple[AppliedEditRecord, ...] = ()
    initial_node_count: int = Field(ge=0)
    final_node_count: int = Field(ge=0)
    changed: bool = False
    reason: str | None = None


class TreeSkillOutput(Trace2SkillOutput[TreeSkillReport]):
    """Typed output returned by the TreeSkill trace2skill algorithm."""

    @model_validator(mode="after")
    def validate_report_outcome(self) -> TreeSkillOutput:
        if self.report.changed != self.changed:
            raise ValueError("TreeSkill report and candidate disagree on changed state")
        return self


class TreeRoutingResult(_StrictModel):
    """One ephemeral routing decision used by a Skill runtime."""

    selected_node_ids: tuple[str, ...]
    content_node_ids: tuple[str, ...]
    ancestor_node_ids: tuple[str, ...]
    skill_content: str
    fallback_used: bool = False
    fallback_reason: str = ""
    full_char_count: int = Field(ge=0)
    routed_char_count: int = Field(ge=0)


__all__ = [
    "AnalysisItem",
    "AppliedEditRecord",
    "FusionFailure",
    "LocalizationFailure",
    "LocalizationResult",
    "LocatedEvidence",
    "NewChildSpec",
    "NodeFusionDecision",
    "NodeFusionEdit",
    "TrajectoryAnalysisRecord",
    "TreeRoutingResult",
    "TreeSkillOutput",
    "TreeSkillReport",
]
