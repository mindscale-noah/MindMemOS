"""Contract for active skill training with environment feedback."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from ...envs.base import Environment
from ...envs.data.typing import DataSample
from ...infra.task_manager.progress import ProgressReporter
from ...registry.base import ConfigT
from ...skill import Skill
from ..base import Algorithm


class SkillTrainer(Algorithm[ConfigT], ABC):
    """Optimize one skill through active task execution, without database access.

    All skills, environments, and samples must be supplied before training.
    Implementations and their dependencies must not query or write databases,
    including through an environment or lazy data source. Local rollout files
    and model calls are allowed. The caller owns environment lifecycle and
    persists the returned skill. Register implementations with Algorithm.register.
    """

    def build_report(self, work_dir: Path) -> dict | None:
        """Return a compact JSON report for the selected result when available.

        Args:
            work_dir: Trainer-owned run directory containing committed artifacts.

        Returns:
            Algorithm-specific report, or None when the trainer has no report.
        """
        return None

    @abstractmethod
    async def train(
        self,
        *,
        skill: Skill,
        env: Environment,
        train_samples: Sequence[DataSample],
        validation_samples: Sequence[DataSample],
        work_dir: Path,
        progress: ProgressReporter | None = None,
    ) -> Skill:
        """Optimize a supplied skill through environment feedback.

        Args:
            skill: Initial skill snapshot; must not be mutated.
            env: Caller-provided environment for candidate execution and scoring.
            train_samples: Materialized samples available for candidate training.
            validation_samples: Materialized samples for candidate selection,
                separate from training; empty means no validation set supplied.
            work_dir: Stable directory for this run. Implementations automatically
                reuse compatible completed work, return a compatible final result,
                and reject incompatible or unrecognized existing artifacts.
                Use distinct subdirectories for independent environment rollouts.
            progress: Optional report sink; algorithms choose reporting boundaries.

        Returns:
            The selected skill candidate without writing it to a database.

        Raises:
            ValueError: Required training data is missing, or the implementation
                requires validation samples but none were supplied.
        """
