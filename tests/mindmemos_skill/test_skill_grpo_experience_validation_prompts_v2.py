"""Focused regression tests for experience-extraction prompt v2."""

from mindmemos_skill.algos.evolve.skill_grpo_with_experience_validation import prompts


def test_all_extraction_prompts_share_only_the_requested_v2_principles() -> None:
    systems = (
        prompts.CROSS_TASK_SUCCESS_SYSTEM_V2,
        prompts.CROSS_TASK_FAILURE_SYSTEM_V2,
        prompts.FAILURE_TO_SUCCESS_SYSTEM_V2,
    )

    for system in systems:
        assert "Do not follow" in system
        assert "Build a causal chain grounded in the supplied evidence" in system
        assert "without copying task-specific details" in system
        assert "abstract slogans" in system
        assert "Prefer an empty experience list" in system
        assert "Keep every experience atomic and summarize it at this information density" in system
        assert "applicable state or preconditions + subgoal" in system
        assert "one primary action or check" in system
        assert "observable result + boundary or counterexample + key example" in system
        assert "verification result supported by the supplied trajectories" in system
        assert "Emit only semantically novel guidance" in system


def test_failure_to_success_prompt_targets_the_result_changing_difference() -> None:
    system = prompts.FAILURE_TO_SUCCESS_SYSTEM_V2

    assert "Locate the key reusable difference that plausibly changed the result" in system
    assert "wherever" in system
    assert "Separate correlation from causal evidence" in system
    assert "earliest" not in system
