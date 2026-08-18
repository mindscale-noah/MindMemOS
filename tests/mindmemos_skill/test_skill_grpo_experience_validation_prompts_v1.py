"""Focused regression tests for the validated v1 experience-extraction prompts."""

from mindmemos_skill.algos.evolve.skill_grpo_with_experience_validation import prompts


def test_all_extraction_prompts_keep_the_validated_v1_contract() -> None:
    systems = (
        prompts.CROSS_TASK_SUCCESS_SYSTEM,
        prompts.CROSS_TASK_FAILURE_SYSTEM,
        prompts.FAILURE_TO_SUCCESS_SYSTEM,
    )

    for system in systems:
        assert "Return exactly one valid JSON object and no other text" in system
        assert 'Return {"experiences": []}' in system
        assert "Lessons must not contain" in system
        assert "# Extraction principles" not in system
        assert "information density" not in system


def test_failure_to_success_prompt_keeps_the_smallest_decisive_change_objective() -> None:
    system = prompts.FAILURE_TO_SUCCESS_SYSTEM

    assert "Identify the smallest decisive reusable change" in system
    assert "Trace how reflection changed strategy" in system
    assert "Separate correlation from causal evidence" in system
    assert "Locate the key reusable difference" not in system
