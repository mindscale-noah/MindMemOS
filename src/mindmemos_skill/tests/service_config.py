"""Explicit deployment configurations shared by service behavior tests."""

from pathlib import Path

from mindmemos_skill.config.config import FastAPIConfig, HttpAuthConfig, ServiceConfig, SkillConfig


def deployment_config(resources: dict, project_id: str) -> SkillConfig:
    """Bind test resources to an explicitly configured unauthenticated project.

    Args:
        resources: Complete service resource settings including home.
        project_id: Project selected for all test operations.

    Returns:
        Complete deployment configuration.
    """
    return SkillConfig(
        service=ServiceConfig.model_validate(resources),
        fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id=project_id)),
    )


def repository_config(home: Path, project_id: str = "project") -> SkillConfig:
    """Configure SQLite repository and graph queries for isolated tests.

    Args:
        home: Test-owned service data directory.
        project_id: Explicit unauthenticated project.

    Returns:
        Complete local deployment configuration.
    """
    return deployment_config(
        {
            "home": home,
            "persistence": {"database": {"class_name": "sqlite"}},
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {"class_name": "graph_manage"},
        },
        project_id,
    )
