"""Script-side experiment adapters behind the two algorithm-family runners."""

from .registry import (
    EXPERIMENTS,
    ExperimentFamily,
    ExperimentSpec,
    dispatch_experiment,
    get_experiment,
    list_experiments,
)

__all__ = [
    "EXPERIMENTS",
    "ExperimentFamily",
    "ExperimentSpec",
    "dispatch_experiment",
    "get_experiment",
    "list_experiments",
]
