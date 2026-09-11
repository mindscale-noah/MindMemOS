"""Typed audit report for one offline evolution optimise() call.

Mirrors the report-field guidance in docs_trac2skill (node ID/hash/counts/
thresholds/decision/artifact URI) without embedding full trajectories or raw
model responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OfflineEvolutionReport(BaseModel):
    algorithm_name: str
    algorithm_version: str
    run_id: str | None = None

    prompt_version: str = Field(default="offline-evolution-v1")

    input_task_ids: list[str] = Field(default_factory=list)
    input_trajectory_ids: list[str] = Field(default_factory=list)
    used_trajectory_ids: list[str] = Field(default_factory=list)
    duplicate_trajectory_ids: list[str] = Field(default_factory=list)

    collection_run_id: str | None = None
    requested_collection_rollout_ids: list[str] = Field(default_factory=list)
    failed_collection_rollout_ids: list[str] = Field(default_factory=list)
    failed_analysis_ids: list[str] = Field(default_factory=list)

    changed: bool = False
    reason: str = ""

    # Vendor pipeline counters and artifact URIs (paths relative to output_dir).
    vendor_counts: dict[str, Any] = Field(default_factory=dict)
    final_skill_path: str | None = None
    report_uri: str | None = None

    config_hash: str | None = None
    vendor_config_hash: str | None = None
