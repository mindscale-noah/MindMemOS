"""Algorithm contracts and service registration without external resources."""

import pytest
from mindmemos_skill.algos import Algorithm, SkillEvolver, SkillManager, SkillTrainer, SkillTrajectory
from mindmemos_skill.envs import Reward, Task
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import BaseModel, ValidationError


class Options(BaseModel):
    """Empty configuration for contract-only test implementations."""


class Manager(SkillManager[Options]):
    """Minimal manager used only to verify shared registration."""

    config_type = Options

    def organize(self, *, skills):
        self.skills = dict(skills)

    def recommend(self, *, task, limit=5):
        return list(self.skills)[:limit]


class Trainer(SkillTrainer[Options]):
    """Minimal trainer used only to verify shared registration."""

    config_type = Options

    async def train(self, *, skill, env, train_samples, validation_samples, work_dir):
        return skill


class Evolver(SkillEvolver[Options]):
    """Minimal evolver used only to verify shared registration."""

    config_type = Options

    async def evolve(self, *, skill, trajectories):
        return skill


@pytest.mark.parametrize("base", [SkillManager, SkillTrainer, SkillEvolver])
def test_algorithm_contracts_are_abstract(base):
    """Reject incomplete algorithms before they can run or be registered."""
    with pytest.raises(TypeError, match="abstract"):
        base(Options())
    with pytest.raises(TypeError, match="concrete subclass"):
        Algorithm.register("incomplete")(base)


@pytest.mark.parametrize("implementation", [Manager, Trainer, Evolver])
def test_registry_resolves_specialized_algorithms(monkeypatch, implementation):
    """Build all three categories through the existing Algorithm registry."""
    monkeypatch.setattr(Algorithm, "_registry", {})
    Algorithm.register("test_algo")(implementation)
    assert isinstance(Algorithm.build("test_algo", {}), implementation)


def test_trajectory_accepts_unscored_history_and_preserves_subtypes():
    """Keep task and reward extensions, and the historical runtime skill."""

    class HistoricalTask(Task):
        source: str

    class DetailedReward(Reward):
        rationale: str

    skill = Skill(
        SkillContent(
            content="Original instructions", name="original", description="Instructions", standard_format=False
        )
    )
    trajectory = SkillTrajectory(
        task=HistoricalTask(id="task-1", instruction="Solve it", source="archive"),
        skills={"skill-1": skill},
        trajectory=[{"role": "assistant", "content": "An answer"}],
    )
    assert trajectory.reward is None
    assert trajectory.skills["skill-1"] is skill
    assert trajectory.model_dump(exclude={"skills"})["task"]["source"] == "archive"
    scored = trajectory.model_copy(
        update={"reward": DetailedReward(score=1, success=True, detail="", rationale="Verified")}
    )
    assert scored.model_dump(exclude={"skills"})["reward"]["rationale"] == "Verified"


def test_trajectory_rejects_unloaded_skill_ids_and_unknown_fields():
    """Require runtime skills instead of IDs that would need database lookup."""
    values = {"task": Task(id="task-1", instruction="Solve it"), "trajectory": []}
    with pytest.raises(ValidationError):
        SkillTrajectory(**values, skills={"skill-1": "unloaded-id"})
    with pytest.raises(ValidationError):
        SkillTrajectory(**values, skills={}, database="unexpected")
