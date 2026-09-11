"""Standard LLM API backend with selectable wire protocols."""

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend import (
    BACKEND_NAME,
    invoke,
)

__all__ = [
    "BACKEND_NAME",
    "invoke",
]
