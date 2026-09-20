"""Business contract for Skill training with environment feedback."""

from abc import ABC, abstractmethod

from ..registry.base import ConfigT
from .base import Pipeline
from .typing import SkillPipelineResult, TrainingRequest


class TrainingPipeline(Pipeline[ConfigT], ABC):
    """Prepare training inputs, invoke a trainer, and persist its selection.

    Concrete implementations load the requested Skill version and partition
    supplied tasks according to the request's train/val/test membership. They
    supply the environment, training/validation samples, and an isolated run
    directory to the trainer. Missing validation means an empty validation set;
    an algorithm requiring validation must reject that input before execution.
    Test samples stay outside the trainer and are evaluated only after final
    candidate selection, without tuning or reselection based on test feedback.
    Missing test samples skip this final evaluation.
    Publishing pipelines pass their fixed algorithm identifier through
    VersionNaming.algorithm when committing; labels are allocated by persistence.
    The trainer controls candidate-dependent rollouts without direct or indirect database
    access. Pipelines manage resources they create and borrow injected backends.
    Private scoring inputs must remain separate from public task material.
    """

    @abstractmethod
    def train(self, request: TrainingRequest) -> SkillPipelineResult:
        """Train one Skill and commit a child of its captured current version.

        Implementations must preserve project scope and source data, avoid
        holding database transactions during training, retain the candidate on
        persistence failure, and reject concurrent version changes. They must
        not mutate the request or return success before persistence.

        Args:
            request: Complete tasks, explicit train/val/test membership, project
                and Skill selection, and the required input version.

        Returns:
            The final selected and persisted Skill version, its parent version,
            and the execution identity, after any requested held-out evaluation.

        Raises:
            ValueError: Inputs or required training/validation data are invalid.
            SkillConflictError: The expected current version changed.
        """
