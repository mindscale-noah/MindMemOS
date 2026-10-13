"""Validated source-grounded Scene records."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

Nonempty = Annotated[str, Field(min_length=1)]


class SceneContractModel(BaseModel):
    """Reject unknown input and prevent raw payloads appearing in error strings."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, str_strip_whitespace=True)
    """Validation errors must still be sanitized before logging errors() or tracebacks."""


class SceneTransition(SceneContractModel):
    """A conditional Scene or logic-step transition in an in-memory record."""

    target: Nonempty
    """Local peer ID, END_* for Scenes, or YIELD_* for steps."""

    condition: str = ""
    """Source-grounded branch condition; empty means unconditional."""

    @field_validator("target")
    @classmethod
    def validate_terminal(cls, value):
        """Require a named terminal when a reserved terminal prefix is used.

        Args:
            value: Local peer ID or named terminal.
        """
        if value in ("END_", "YIELD_"):
            raise PydanticCustomError("scene_terminal_outcome_missing", "Terminal prefixes require a nonempty outcome")
        return value


class SemanticDependency(SceneContractModel):
    """One documented dependency, classified without accessing the server filesystem."""

    name: Nonempty = Field(max_length=300)
    """Documented package name or portable local script path."""

    version: str = Field(default="", max_length=100)
    """Documented version constraint; the reserved value local identifies a script."""


class SkillSemantics(SceneContractModel):
    """Normalized semantic fields shared by extracted and provided input."""

    name: Nonempty
    """Display name, never a family or graph node identity."""

    goal: Nonempty
    """Source-grounded capability or Scene objective."""

    inputs: tuple[str, ...] = ()
    """Normalized input descriptions used for linking and retrieval."""

    outputs: tuple[str, ...] = ()
    """Normalized output descriptions used for linking and retrieval."""

    tooling: tuple[str, ...] = ()
    """Explicit tools mentioned in the source."""

    intent_signature: str = Field(default="", max_length=1000)
    """Compact source-grounded task signature, independent of the prose goal."""

    tags: tuple[Annotated[str, Field(max_length=200)], ...] = Field(default=(), max_length=30)
    """Source-grounded domain labels merged with document tags."""

    dependencies: tuple[SemanticDependency, ...] = Field(default=(), max_length=50)
    """Documented local scripts and versioned external dependencies; memory-only structure."""


class LogicStepInput(SceneContractModel):
    """One transient logical action; never a persisted graph node or metadata."""

    logic_step_id: Nonempty
    """Unique local identifier within the whole owning record."""

    action: Nonempty
    """Normalized source-grounded action used to derive flow text."""

    next_step_rules: tuple[SceneTransition, ...] = Field(min_length=1)
    """Transitions to steps in the same Scene or explicit YIELD_* terminals."""


class SceneInput(SkillSemantics):
    """Typed in-memory Scene semantics with explicit entry and flow ownership."""

    scene_id: Nonempty
    """Local Scene ID, unique across Scenes and steps in the record."""

    scene_type: Nonempty
    """Source-grounded category, not a graph node type."""

    entry_conditions: tuple[Annotated[str, Field(strict=True, min_length=1)], ...] = ()
    """Source-grounded prerequisites independent of incoming branch conditions; empty when absent."""

    exit_conditions: tuple[Annotated[str, Field(strict=True, min_length=1)], ...] = ()
    """Source-grounded postconditions independent of outgoing branches; empty when absent."""

    entry_logic_step_id: Nonempty
    """Entry step belonging to this Scene."""

    contained_logic_steps: tuple[Nonempty, ...] = Field(min_length=1)
    """Ordered steps owned exclusively by this Scene."""

    next_scene_rules: tuple[SceneTransition, ...] = Field(min_length=1)
    """Conditional peer Scene transitions or explicit END_* terminals."""


class ProvidedSceneRecord(SceneContractModel):
    """Normalized, transient benchmark seam; raw corpus import is outside this contract."""

    skill: SkillSemantics
    """Source-grounded Skill semantics to merge with document-derived fields."""

    entry_scene_id: Nonempty
    """Scene entered when the Skill starts."""

    scenes: tuple[SceneInput, ...] = Field(min_length=1)
    """At least one justified Scene; no artificial minimum of two."""

    logic_steps: tuple[LogicStepInput, ...] = Field(min_length=1, repr=False)
    """Memory-only actions released after deriving Scene flow text."""

    @model_validator(mode="after")
    def validate_ownership(self):
        """Reject duplicate identities, wrong-level transitions and foreign entries."""
        scene_ids = {scene.scene_id for scene in self.scenes}
        steps = {step.logic_step_id: step for step in self.logic_steps}
        all_ids = [scene.scene_id for scene in self.scenes] + [step.logic_step_id for step in self.logic_steps]
        if len(set(all_ids)) != len(all_ids) or any(key.startswith(("END_", "YIELD_")) for key in all_ids):
            raise PydanticCustomError(
                "scene_local_ids_invalid", "Local IDs must be unique and cannot use terminal prefixes"
            )
        if self.entry_scene_id not in scene_ids:
            raise PydanticCustomError("scene_entry_missing", "Skill entry must identify a Scene in this record")
        owned = []
        for scene in self.scenes:
            contained = set(scene.contained_logic_steps)
            owned.extend(scene.contained_logic_steps)
            if scene.entry_logic_step_id not in contained or not contained <= steps.keys():
                raise PydanticCustomError(
                    "scene_step_ownership_invalid", "Scene entry and contained steps must belong to this Scene"
                )
            if any(
                rule.target not in scene_ids and not rule.target.startswith("END_") for rule in scene.next_scene_rules
            ):
                raise PydanticCustomError(
                    "scene_transition_invalid", "Scene transitions require a peer Scene or END_* terminal"
                )
            for key in contained:
                if any(
                    rule.target not in contained and not rule.target.startswith("YIELD_")
                    for rule in steps[key].next_step_rules
                ):
                    raise PydanticCustomError(
                        "scene_step_transition_invalid",
                        "Step transitions require a same-Scene step or YIELD_* terminal",
                    )
        if len(owned) != len(set(owned)) or set(owned) != steps.keys():
            raise PydanticCustomError(
                "scene_step_assignment_invalid", "Every logic step must belong to exactly one Scene"
            )
        self._validate_flow(
            self.entry_scene_id,
            {scene.scene_id: tuple(rule.target for rule in scene.next_scene_rules) for scene in self.scenes},
        )
        for scene in self.scenes:
            self._validate_flow(
                scene.entry_logic_step_id,
                {key: tuple(rule.target for rule in steps[key].next_step_rules) for key in scene.contained_logic_steps},
            )
        return self

    @staticmethod
    def _validate_flow(entry, transitions):
        """Require entry reachability and a possible terminal path, allowing loops with exits."""
        reverse = {key: [] for key in transitions}
        terminating = set()
        for source, targets in transitions.items():
            for target in targets:
                if target in reverse:
                    reverse[target].append(source)
                else:
                    terminating.add(source)

        def reachable(starts, adjacency):
            seen = set(starts)
            pending = list(starts)
            while pending:
                for target in adjacency[pending.pop()]:
                    if target in adjacency and target not in seen:
                        seen.add(target)
                        pending.append(target)
            return seen

        if reachable([entry], transitions) != transitions.keys():
            raise PydanticCustomError("scene_node_unreachable", "Every node must be reachable from its owning entry")
        if reachable(terminating, reverse) != transitions.keys():
            raise PydanticCustomError("scene_terminal_unreachable", "Every node must have a path to a terminal outcome")


class SceneExtractionResponse(SceneContractModel):
    """Model envelope distinguishing insufficient source from invalid structure."""

    status: Literal["ok", "insufficient_source"]
    """Whether the document supports at least one actionable Scene."""

    record: ProvidedSceneRecord | None = Field(repr=False)
    """Transient normalized SSL; null exactly when source information is insufficient."""

    @model_validator(mode="after")
    def validate_status(self):
        """Reject success without a record or insufficient source with invented actions."""
        if (self.status == "ok") != (self.record is not None):
            raise PydanticCustomError("scene_status_mismatch", "Extraction status and record disagree")
        return self
