"""Self-evolution (``feedback_evo`` mode) components.

Independent from the implicit/explicit feedback pipelines: owns its signal
detection prompt, planning, evolution-state application, and the task-end
collector.
"""

from __future__ import annotations

from .collector import FeedbackEvoCollector
from .evolution import (
    EvolutionExecutor,
    EvolutionPlanner,
    build_initial_evolution_state,
    ensure_evolution_state,
    is_evolvable_path,
)

__all__ = [
    "EvolutionExecutor",
    "EvolutionPlanner",
    "FeedbackEvoCollector",
    "build_initial_evolution_state",
    "ensure_evolution_state",
    "is_evolvable_path",
]
