"""Business persistence implementations."""

from ..registry.base import ConfigT, Registrable


class Persistence(Registrable[ConfigT]):
    """Registration family for business persistence implementations."""
