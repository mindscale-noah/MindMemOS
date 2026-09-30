"""Business-independent infrastructure implementations."""

from ..registry.base import ConfigT, Registrable


class Infrastructure(Registrable[ConfigT]):
    """Registration family for business-independent infrastructure implementations."""
