"""Explicit registry for independently extensible Skill Runtime schemes."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import JsonValue

from ..errors import SkillCapabilityUnavailableError
from ..typing import Skill
from .contracts import SkillRuntime


class SkillRuntimeRegistry:
    def __init__(self, runtimes: Iterable[SkillRuntime] = ()) -> None:
        self._runtimes: dict[str, SkillRuntime] = {}
        for runtime in runtimes:
            self.register(runtime)

    def register(self, runtime: SkillRuntime) -> None:
        runtime_type = runtime.runtime_type
        if not runtime_type:
            raise ValueError("Skill Runtime type must not be empty")
        if runtime_type in self._runtimes:
            raise ValueError(f"Skill Runtime {runtime_type!r} is already registered")
        if not runtime.metadata_models:
            raise ValueError(f"Skill Runtime {runtime_type!r} declares no metadata schema")
        self._runtimes[runtime_type] = runtime

    def resolve(self, runtime_type: str) -> SkillRuntime:
        runtime = self._runtimes.get(runtime_type)
        if runtime is None:
            available = ", ".join(sorted(self._runtimes)) or "<none>"
            raise SkillCapabilityUnavailableError(
                f"Skill Runtime {runtime_type!r} is unavailable; registered: {available}"
            )
        return runtime

    def validate(self, skill: Skill) -> None:
        self.validate_spec(
            runtime_type=skill.runtime_type,
            schema_version=skill.runtime_schema_version,
            metadata=skill.runtime_metadata,
        )

    def validate_spec(
        self,
        *,
        runtime_type: str,
        schema_version: int,
        metadata: dict[str, JsonValue],
    ) -> None:
        self.resolve(runtime_type).parse_metadata(schema_version, metadata)

    @property
    def runtime_types(self) -> frozenset[str]:
        return frozenset(self._runtimes)


__all__ = ["SkillRuntimeRegistry"]
