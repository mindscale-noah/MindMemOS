from typing import get_type_hints

from mindmemos_skill.algos.evolve import EvolveAlgorithm, EvolveInput, EvolveOutput
from mindmemos_skill.algos.evolve.skill_grpo_with_experience_validation import (
    SkillGrpoWithExperienceValidationEvolveInput,
    SkillGrpoWithExperienceValidationEvolveResult,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import (
    SkillGrpoEvolveInput,
    SkillGrpoEvolveResult,
)
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer import (
    SkillGrpoWithoutReplayBufferEvolveInput,
    SkillGrpoWithoutReplayBufferEvolveResult,
)
from mindmemos_skill.algos.evolve.trajectory_memory import (
    TrajectoryMemoryEvolveInput,
    TrajectoryMemoryEvolveResult,
)
from mindmemos_skill.registry import ComponentType, get_component


def test_evolve_protocol_uses_explicit_operation_contracts() -> None:
    assert set(EvolveInput.model_fields) == {
        "run_id",
        "base_skill",
        "train_tasks",
        "validation_tasks",
        "test_tasks",
    }
    assert set(EvolveOutput.model_fields) == {
        "run_id",
        "final_skill",
        "changed",
        "trajectories",
        "finished_at",
    }
    assert not EvolveAlgorithm.__parameters__
    assert get_type_hints(EvolveAlgorithm.evolve) == {
        "request": EvolveInput,
        "return": EvolveOutput,
    }


def test_all_evolve_contracts_extend_shared_input_and_output() -> None:
    input_types = (
        SkillGrpoEvolveInput,
        SkillGrpoWithoutReplayBufferEvolveInput,
        SkillGrpoWithExperienceValidationEvolveInput,
        TrajectoryMemoryEvolveInput,
    )
    output_types = (
        SkillGrpoEvolveResult,
        SkillGrpoWithoutReplayBufferEvolveResult,
        SkillGrpoWithExperienceValidationEvolveResult,
        TrajectoryMemoryEvolveResult,
    )

    assert all(issubclass(input_type, EvolveInput) for input_type in input_types)
    assert all(issubclass(output_type, EvolveOutput) for output_type in output_types)


def test_builtin_evolve_algorithms_are_application_configurable() -> None:
    for name in (
        "skill_grpo_with_replay_buffer",
        "skill_grpo_without_replay_buffer",
        "skill_grpo_with_experience_validation",
    ):
        component = get_component(type=ComponentType.ALGO, name=name)
        assert component.config_model is not None
        assert component.capabilities == frozenset({"evolve"})
