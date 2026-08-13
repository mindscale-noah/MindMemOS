from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from mindmemos_skill.algos.evolve.task_virtual_skill import (
    TaskVirtualSkillPlan,
    TaskVirtualSkillRunConfig,
    TrajectoryKeyPoints,
    parse_plan,
)
from mindmemos_skill.algos.evolve.task_virtual_skill.algorithm import (
    _build_artifacts,
    _sample_summaries,
    _validate_plan,
)
from mindmemos_skill.algos.evolve.task_virtual_skill.refinement import _apply_merges, _parse_change_response
from mindmemos_skill.algos.evolve.task_virtual_skill.refinement_models import (
    TaskSkillChange,
    VirtualSkillMerge,
)
from mindmemos_skill.algos.evolve.task_virtual_skill.refinement_prompts import CHANGE_SYSTEM
from mindmemos_skill.algos.evolve.task_virtual_skill.summarizer import parse_summary
from mindmemos_skill.registry import ComponentType, get_component
from mindmemos_skill.typing import Skill, compute_skill_content_hash


def _base_skill() -> Skill:
    blob = {"SKILL.md": "# Workbook\n\nInspect first. Preserve formulas. Save and verify.\n"}
    return Skill(
        skill_id="workbook",
        version_id="v1",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="workbook",
        blob=blob,
        created_at=datetime.now(UTC),
    )


def _plan(*, excerpt: str = "Inspect first. Preserve formulas. Save and verify.") -> dict:
    return {
        "skill_name": "Workbook operations",
        "description": "Task-routed workbook operations.",
        "virtual_skills": [
            {
                "skill_id": "edit-workbook",
                "name": "Edit workbook",
                "description": "Select for workbook edits; produces a verified workbook.",
                "supporting_trajectory_ids": ["trajectory-1"],
                "source_excerpts": [excerpt],
            }
        ],
    }


def _summary(index: int) -> TrajectoryKeyPoints:
    return TrajectoryKeyPoints(
        trajectory_id=f"trajectory-{index}",
        task_id=f"task-{index}",
        task_goal="Edit a workbook",
        task_family="Workbook edit",
        key_actions=["Loaded the workbook"],
        outcome="Saved the result",
    )


def test_plan_uses_exact_source_excerpts_to_build_markdown() -> None:
    plan = parse_plan(json.dumps(_plan()))
    _validate_plan(
        plan,
        source_skill=_base_skill().content,
        sampled_trajectory_ids={"trajectory-1"},
        max_virtual_skills=4,
    )

    artifacts = _build_artifacts(plan)

    assert len(artifacts) == 1
    assert artifacts[0].source_excerpts == ["Inspect first. Preserve formulas. Save and verify."]
    assert "Inspect first. Preserve formulas. Save and verify." in artifacts[0].markdown


def test_grounding_rejects_invented_content_and_unsampled_trajectory() -> None:
    invented = TaskVirtualSkillPlan.model_validate(_plan(excerpt="Re-open the workbook after saving."))
    with pytest.raises(ValueError, match="absent from SKILL.md"):
        _validate_plan(
            invented,
            source_skill=_base_skill().content,
            sampled_trajectory_ids={"trajectory-1"},
            max_virtual_skills=4,
        )

    payload_with_generated_content = _plan()
    payload_with_generated_content["virtual_skills"][0]["content"] = "Model-authored instructions"
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TaskVirtualSkillPlan.model_validate(payload_with_generated_content)

    unsupported = TaskVirtualSkillPlan.model_validate(_plan())
    with pytest.raises(ValueError, match="unsampled trajectories"):
        _validate_plan(
            unsupported,
            source_skill=_base_skill().content,
            sampled_trajectory_ids={"trajectory-other"},
            max_virtual_skills=4,
        )


def test_summary_sampling_is_seeded_and_defaults_to_twenty() -> None:
    summaries = [_summary(index) for index in range(30)]

    first = _sample_summaries(summaries, size=20, seed=7)
    second = _sample_summaries(summaries, size=20, seed=7)
    config = TaskVirtualSkillRunConfig.model_validate(
        {"dataset": {"env_ref": "spreadsheetbench", "agent_ref": "react"}}
    )

    assert [item.trajectory_id for item in first] == [item.trajectory_id for item in second]
    assert len(first) == 20
    assert config.summary.sample_size == 20


def test_summary_parser_requires_structured_key_points() -> None:
    payload = {
        "task_goal": "Update values",
        "task_family": "Workbook edit",
        "key_actions": ["Loaded workbook", "Changed cells"],
        "turning_points": [],
        "skill_usage": ["Used openpyxl guidance"],
        "outcome": "Saved output",
    }

    assert parse_summary(f"```json\n{json.dumps(payload)}\n```") == payload
    with pytest.raises(ValueError, match="keys must be exactly"):
        parse_summary(json.dumps({**payload, "invented": True}))


def test_algorithm_is_registered_as_independent_evolve_component() -> None:
    component = get_component(type=ComponentType.ALGO, name="task_virtual_skill")

    assert component.capabilities == frozenset({"evolve"})
    assert component.requirements.required_model_roles == frozenset({"summary", "decomposition"})


def test_task_change_contract_allows_exactly_one_direct_operation() -> None:
    change = TaskSkillChange(
        task_id="task-1",
        operation="update",
        skill_id="edit-workbook",
        name="Edit workbook safely",
        description="Load for literal workbook edits that require output validation.",
        content="# Edit\n\nAdd a general output validation step.",
        diagnosis="The edit guidance omitted a validation step.",
        evidence_trajectory_ids=["trajectory-1", "trajectory-2"],
    )

    assert change.operation == "update"
    assert change.name == "Edit workbook safely"
    content_only = TaskSkillChange.model_validate(
        {
            **change.model_dump(),
            "name": None,
            "description": None,
        }
    )
    assert content_only.name is None
    assert content_only.description is None
    with pytest.raises(ValueError, match="at least one"):
        TaskSkillChange.model_validate({**content_only.model_dump(), "content": None})
    with pytest.raises(ValueError, match="noop must not include Skill fields"):
        TaskSkillChange(
            task_id="task-1",
            operation="noop",
            skill_id="edit-workbook",
            diagnosis="No change.",
            evidence_trajectory_ids=["trajectory-1"],
        )
    with pytest.raises(ValueError, match="create requires"):
        TaskSkillChange(
            task_id="task-1",
            operation="create",
            skill_id="new-subtask",
            diagnosis="A new independent subtask is supported.",
            evidence_trajectory_ids=["trajectory-1"],
        )


def test_change_prompt_uses_one_json_operation_and_conservative_location_principles() -> None:
    assert "semantic ownership" in CHANGE_SYSTEM
    assert "Agent noncompliance" in CHANGE_SYSTEM
    assert "counterfactual evidence" in CHANGE_SYSTEM
    assert "operation is create, update, or noop" in CHANGE_SYSTEM
    assert "not callable tools" in CHANGE_SYSTEM
    assert '"operation": "create"' in CHANGE_SYSTEM
    assert '"operation": "update"' in CHANGE_SYSTEM
    assert '"operation": "noop"' in CHANGE_SYSTEM
    assert "Decide directly what must change" in CHANGE_SYSTEM
    assert "A Skill is not merely a collection of lessons" in CHANGE_SYSTEM
    assert "Multiple coordinated edits are allowed" in CHANGE_SYSTEM
    assert "Distill at most one atomic" not in CHANGE_SYSTEM
    assert "it overwrites the current Markdown" in CHANGE_SYSTEM
    assert '"name": "complete replacement name"' in CHANGE_SYSTEM
    assert '"description": "complete replacement routing description"' in CHANGE_SYSTEM
    assert '"content": "# Complete revised Skill' in CHANGE_SYSTEM
    assert '"rest unchanged"' in CHANGE_SYSTEM


def test_json_update_response_parses_to_direct_skill_change() -> None:
    response = {
        "content": json.dumps(
            {
                "operation": "update",
                "skill_id": "edit-workbook",
                "description": "Load for workbook edits that must be reopened and validated.",
                "diagnosis": "The old guidance was incomplete.",
                "evidence_trajectory_ids": ["trajectory-1"],
            }
        )
    }

    change = _parse_change_response(response, task_id="task-1")

    assert change.operation == "update"
    assert change.skill_id == "edit-workbook"
    assert change.name is None
    assert change.description == "Load for workbook edits that must be reopened and validated."
    assert change.content is None


def test_json_change_response_accepts_markdown_fence() -> None:
    response = """```json
{"operation":"noop","diagnosis":"No safe change.","evidence_trajectory_ids":["trajectory-1"]}
```"""

    change = _parse_change_response(response, task_id="task-1")

    assert change.operation == "noop"


def test_virtual_skill_merges_change_only_the_target_component() -> None:
    base = _base_skill().model_copy(
        update={
            "runtime_type": "virtual_components",
            "runtime_metadata": {
                "max_initial_components": 1,
                "components": [
                    {
                        "component_id": "edit-workbook",
                        "name": "Edit workbook",
                        "description": "Edit cells",
                        "content": "# Edit\n\nOriginal edit guidance.\n",
                    },
                    {
                        "component_id": "inspect-workbook",
                        "name": "Inspect workbook",
                        "description": "Inspect cells",
                        "content": "# Inspect\n\nOriginal inspect guidance.\n",
                    },
                ],
            },
            "resources": {
                "virtual_skills/edit-workbook.md": "# Edit\n\nOriginal edit guidance.\n",
                "virtual_skills/inspect-workbook.md": "# Inspect\n\nOriginal inspect guidance.\n",
            },
        }
    )
    merge = VirtualSkillMerge(
        operation="update",
        skill_id="edit-workbook",
        name="Validated workbook edits",
        description="Load for workbook edits that require validation.",
        source_task_ids=["task-1"],
        original_content="# Edit\n\nOriginal edit guidance.\n",
        revised_content="# Edit\n\nRevised edit guidance.\n",
        change_summary="Add verified guidance",
    )

    evolved = _apply_merges(base, [merge], run_id="refine-1")
    components = {item["component_id"]: item for item in evolved.runtime_metadata["components"]}

    assert components["edit-workbook"]["name"] == "Validated workbook edits"
    assert components["edit-workbook"]["description"] == "Load for workbook edits that require validation."
    assert components["edit-workbook"]["content"] == "# Edit\n\nRevised edit guidance.\n"
    assert components["inspect-workbook"]["name"] == "Inspect workbook"
    assert components["inspect-workbook"]["content"] == "# Inspect\n\nOriginal inspect guidance.\n"
    assert evolved.resources["virtual_skills/edit-workbook.md"] == "# Edit\n\nRevised edit guidance.\n"


def test_virtual_skill_create_merge_adds_one_component_and_resource() -> None:
    base = _base_skill().model_copy(
        update={
            "runtime_type": "virtual_components",
            "runtime_metadata": {
                "max_initial_components": 1,
                "components": [
                    {
                        "component_id": "edit-workbook",
                        "name": "Edit workbook",
                        "description": "Edit cells",
                        "content": "# Edit\n\nOriginal edit guidance.\n",
                    }
                ],
            },
            "resources": {"virtual_skills/edit-workbook.md": "# Edit\n\nOriginal edit guidance.\n"},
        }
    )
    merge = VirtualSkillMerge(
        operation="create",
        skill_id="verify-workbook",
        name="Verify workbook",
        description="Load after edits to verify the saved workbook.",
        source_task_ids=["task-1"],
        original_content=None,
        revised_content="# Verify\n\nReopen and validate the saved workbook.\n",
        change_summary="Created an independent verification subtask.",
    )

    evolved = _apply_merges(base, [merge], run_id="refine-1")
    components = {item["component_id"]: item for item in evolved.runtime_metadata["components"]}

    assert components["verify-workbook"]["name"] == "Verify workbook"
    assert evolved.resources["virtual_skills/verify-workbook.md"] == (
        "# Verify\n\nReopen and validate the saved workbook.\n"
    )


def test_virtual_skill_partial_update_preserves_omitted_fields() -> None:
    base = _base_skill().model_copy(
        update={
            "runtime_type": "virtual_components",
            "runtime_metadata": {
                "max_initial_components": 1,
                "components": [
                    {
                        "component_id": "edit-workbook",
                        "name": "Edit workbook",
                        "description": "Edit cells",
                        "content": "# Edit\n\nOriginal edit guidance.\n",
                    }
                ],
            },
            "resources": {"virtual_skills/edit-workbook.md": "# Edit\n\nOriginal edit guidance.\n"},
        }
    )
    merge = VirtualSkillMerge(
        operation="update",
        skill_id="edit-workbook",
        name=None,
        description="Load for precise literal cell edits.",
        source_task_ids=["task-1"],
        original_content="# Edit\n\nOriginal edit guidance.\n",
        revised_content=None,
        change_summary="Improve routing description only.",
    )

    evolved = _apply_merges(base, [merge], run_id="refine-1")
    component = evolved.runtime_metadata["components"][0]

    assert component["name"] == "Edit workbook"
    assert component["description"] == "Load for precise literal cell edits."
    assert component["content"] == "# Edit\n\nOriginal edit guidance.\n"
    assert evolved.resources["virtual_skills/edit-workbook.md"] == "# Edit\n\nOriginal edit guidance.\n"
