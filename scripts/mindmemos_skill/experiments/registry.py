"""Script-side registry for CLI experiment adapters.

Only the algorithm family owns an executable script. Individual algorithms
remain importable package modules selected by this registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any


class ExperimentFamily(StrEnum):
    EVOLVE = "evolve"
    TRACE2SKILL = "trace2skill"


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    family: ExperimentFamily
    module: str
    environments: frozenset[str]
    common_extras: tuple[str, ...] = ("llm",)
    environment_extras: dict[str, tuple[str, ...]] | None = None
    inject_environment_as_benchmark: bool = False

    def extras_for(self, environment: str) -> tuple[str, ...]:
        extras = list(self.common_extras)
        if self.environment_extras is not None:
            extras.extend(self.environment_extras.get(environment, ()))
        return tuple(dict.fromkeys(extras))

    @property
    def implementation_path(self) -> Path:
        return Path(__file__).with_name(f"{self.module.rsplit('.', 1)[-1]}.py")


def _evolve(
    name: str,
    *,
    environments: frozenset[str],
    environment_extras: dict[str, tuple[str, ...]] | None = None,
    common_extras: tuple[str, ...] = ("llm",),
    inject_environment_as_benchmark: bool = False,
) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        family=ExperimentFamily.EVOLVE,
        module=f"experiments.{name}",
        environments=environments,
        common_extras=common_extras,
        environment_extras=environment_extras,
        inject_environment_as_benchmark=inject_environment_as_benchmark,
    )


def _trace2skill(
    name: str,
    *,
    environments: frozenset[str],
    environment_extras: dict[str, tuple[str, ...]] | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        family=ExperimentFamily.TRACE2SKILL,
        module=f"experiments.{name}",
        environments=environments,
        environment_extras=environment_extras,
        inject_environment_as_benchmark=True,
    )


EXPERIMENTS: dict[str, ExperimentSpec] = {
    spec.name: spec
    for spec in (
        _evolve(
            "skill_grpo_with_replay_buffer",
            environments=frozenset({"livemath", "spreadsheetbench"}),
            environment_extras={"spreadsheetbench": ("spreadsheetbench",)},
            inject_environment_as_benchmark=True,
        ),
        _evolve(
            "skill_grpo_without_replay_buffer",
            environments=frozenset({"alfworld", "livemath", "spreadsheetbench"}),
            environment_extras={"alfworld": ("alfworld",), "spreadsheetbench": ("spreadsheetbench",)},
            inject_environment_as_benchmark=True,
        ),
        _evolve(
            "skill_grpo_with_experience_validation",
            environments=frozenset({"alfworld"}),
            environment_extras={"alfworld": ("alfworld",)},
        ),
        _evolve(
            "trajectory_memory",
            environments=frozenset({"alfworld"}),
            environment_extras={"alfworld": ("alfworld",)},
        ),
        _evolve(
            "experience_memory_embedding",
            environments=frozenset({"alfworld"}),
            environment_extras={"alfworld": ("alfworld",)},
        ),
        _evolve(
            "initial_skill_evaluation",
            environments=frozenset({"alfworld"}),
            environment_extras={"alfworld": ("alfworld",)},
        ),
        _evolve(
            "skill_evaluation",
            environments=frozenset({"alfworld", "livemath", "spreadsheetbench"}),
            environment_extras={"alfworld": ("alfworld",), "spreadsheetbench": ("spreadsheetbench",)},
            inject_environment_as_benchmark=True,
        ),
        _evolve(
            "agent_evaluation",
            environments=frozenset({"local"}),
            common_extras=("llm", "claude-sdk"),
        ),
        _trace2skill(
            "trajectory_evidence_patch",
            environments=frozenset({"alfworld", "livemath", "spreadsheetbench"}),
            environment_extras={"alfworld": ("alfworld",), "spreadsheetbench": ("spreadsheetbench",)},
        ),
        _trace2skill(
            "treeskill",
            environments=frozenset({"spreadsheetbench"}),
            environment_extras={"spreadsheetbench": ("spreadsheetbench",)},
        ),
    )
}


def get_experiment(name: str, *, family: ExperimentFamily | None = None) -> ExperimentSpec:
    spec = EXPERIMENTS.get(name)
    if spec is None or (family is not None and spec.family is not family):
        available = ", ".join(list_experiments(family=family)) or "<none>"
        family_label = family.value if family is not None else "experiment"
        raise ValueError(f"unknown {family_label} algorithm {name!r}; available: {available}")
    return spec


def list_experiments(*, family: ExperimentFamily | None = None) -> list[str]:
    return sorted(name for name, spec in EXPERIMENTS.items() if family is None or spec.family is family)


def dispatch_experiment(family: ExperimentFamily, name: str, argv: Sequence[str]) -> int:
    spec = get_experiment(name, family=family)
    module = import_module(spec.module)
    entrypoint: Any = getattr(module, "main", None)
    if not callable(entrypoint):
        raise RuntimeError(f"experiment module {spec.module!r} does not expose main(argv)")
    result = entrypoint(list(argv))
    return result if isinstance(result, int) else 0


__all__ = [
    "EXPERIMENTS",
    "ExperimentFamily",
    "ExperimentSpec",
    "dispatch_experiment",
    "get_experiment",
    "list_experiments",
]
