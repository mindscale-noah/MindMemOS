"""Scope-level filesystem edit and diff primitives."""

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.diff import (
    directory_diff,
    scope_tree_diff,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.scope_workflow import (
    run_scope_evolve_agents,
)

__all__ = [
    "directory_diff",
    "scope_tree_diff",
    "run_scope_evolve_agents",
]
