"""Registration family for composable Skill business operations."""

from ..persistence.store import SkillPersistence
from ..registry.base import ConfigT, Registrable


class Pipeline(Registrable[ConfigT]):
    """Compose algorithms, persistence services, and optional environments.

    Register concrete implementations with Pipeline.register. The service passes
    the single global persistence and a validated implementation configuration.
    Each pipeline constructs its own algorithms, agents and environments from
    nested typed configuration; it never resolves global resource names.

    Trainers and evolvers receive detached data, never persistence. Per-call
    state belongs to the call rather than this service-owned pipeline instance.
    """

    def __init__(self, config: ConfigT, *, persistence: SkillPersistence | None = None) -> None:
        """Store the typed options and borrow the service's only persistence.

        Args:
            config: Complete options, including definitions of owned components.
            persistence: Global store, or None for a service without storage.
        """
        super().__init__(config)
        self.persistence = persistence

    async def close(self) -> None:
        """Release pipeline-owned resources; never close the borrowed persistence.

        Implementations owning resources must override this method and clean up
        partial construction themselves if their constructor fails.
        """
