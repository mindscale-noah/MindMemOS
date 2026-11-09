"""Skill algorithm implementations."""

from ..registry.base import ConfigT, Registrable


class Algorithm(Registrable[ConfigT]):
    """Shared registration family for skill algorithm implementations.

    Register concrete managers, trainers, and evolvers with
    Algorithm.register so the service factory can resolve all implementations.
    Configuration describes algorithm behavior; each call supplies its data.
    Database dependencies belong only to concrete SkillManager implementations,
    never to this base, trainers, or evolvers.
    """
