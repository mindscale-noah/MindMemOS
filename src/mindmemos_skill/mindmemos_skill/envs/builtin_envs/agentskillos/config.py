"""Official AgentSkillOS execution and source configuration."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents.buitin_agents.codex.config import CodexConfig
from ....infra.execution import ExecutionConfig
from .source import evaluator_digest


class AgentSkillOSRetrievalDataConfig(BaseModel):
    """Locate legacy retrieval-only files for the standalone comparison tool."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown source settings."""

    tasks_path: Path
    """JSONL task_id and instruction_text records."""

    relevance_path: Path
    """JSON mapping task IDs to private core_gt_ids labels."""


class AgentSkillOSDataConfig(BaseModel):
    """Read the original official task JSON files and bundled input assets."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported source settings."""

    benchmark_root: Path
    """Repository checkout containing benchmark/AgentSkillOS_bench/."""

    relevance_path: Path | None = None
    """Optional legacy core_gt_ids labels; absent uses official task skills for diagnostics."""


class AgentSkillOSConfig(BaseModel):
    """Execute native tool use and score actual artifacts with the official evaluator."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown execution settings and stale retrieval-only configuration."""

    agent: CodexConfig
    """Native Codex execution settings; model must be explicitly selected."""

    benchmark_root: Path
    """Official evaluator checkout, used only by the private scoring subprocess."""

    evaluator: ExecutionConfig = Field(default_factory=ExecutionConfig)
    """Python interpreter and timeout for official deterministic scoring."""

    evaluator_version: Literal["agentskillos-official-pass-v1"] = "agentskillos-official-pass-v1"
    """Binary official passed result; retrieval scores are diagnostic only."""

    @model_validator(mode="after")
    def bind_sources(self) -> Self:
        """Validate scoring sources and require an explicit execution model."""
        if not self.agent.model:
            raise ValueError("AgentSkillOS requires an explicit agent model")
        self.benchmark_root = self.benchmark_root.expanduser().resolve()
        evaluator_digest(self.benchmark_root)
        return self
