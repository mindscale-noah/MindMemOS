"""Settings resolution for pipeline configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelineConfig

SUPERVISION_MODES = ("unsupervised", "supervised")


@dataclass(frozen=True)
class PipelineSettings:
    """Typed, frozen view of ``state["settings"]``.

    Per-stage parallelism is deliberately not snapshotted: stages read
    ``parallelism.<stage>`` live from config, so a resumed run may use a
    different width than the original.
    """

    backend: str
    supervision_mode: str
    skill_md_only: bool

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelineSettings":
        """Resolve a :class:`PipelineConfig` against the configured defaults.

        ``skill_md_only`` reads ``pipeline.skill_md_only`` from the config
        file; the CLI flag can only turn it ON (like the fork override).
        It is no longer hardwired to the react backend -- a benchmark that
        wants the restriction states it in its overlay config.
        """
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_flag
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import normalise_backend

        backend = normalise_backend(config.backend)
        skill_md_only = bool(config.skill_md_only) or setting_flag("pipeline.skill_md_only")
        supervision_mode = _resolve_supervision_mode(config)
        return cls(
            backend=backend,
            supervision_mode=supervision_mode,
            skill_md_only=skill_md_only,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineSettings":
        """Rehydrate a settings dict (e.g. from state.json on resume)."""
        return cls(
            backend=str(raw.get("backend", "")),
            supervision_mode=str(raw.get("supervision_mode", "unsupervised")),
            skill_md_only=bool(raw.get("skill_md_only", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def _resolve_supervision_mode(config: PipelineConfig) -> str:
    """Resolve the supervision mode; never inferred from files on disk."""
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_raw

    mode = (config.supervision_mode or setting_raw("pipeline.supervision_mode")).strip().lower()
    if mode not in SUPERVISION_MODES:
        raise ValueError(f"supervision_mode must be one of {', '.join(SUPERVISION_MODES)}, got {mode!r}")
    return mode


__all__ = [
    "SUPERVISION_MODES",
    "PipelineSettings",
    "_resolve_supervision_mode",
]
