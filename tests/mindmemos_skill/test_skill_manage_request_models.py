"""Public Application request/result DTO round-trips and frozen invariants."""

from __future__ import annotations

import pytest
from mindmemos_skill.application.skill_manage.models import (
    SkillManageBuildRequest,
    SkillManageRemoveRequest,
    SkillManageRetrieveRequest,
)
from mindmemos_skill.typing import (
    EmbeddingProfile,
    RetrievalMode,
    SkillManageRemoveSelector,
    SkillVersionRef,
)
from pydantic import ValidationError


def test_build_request_freezes_full_mode_and_scope() -> None:
    request = SkillManageBuildRequest(
        project_id="p1",
        graph_id="g1",
        version_refs=(SkillVersionRef(skill_id="s1", version_id="v1"),),
        embedding_profile=EmbeddingProfile(model="m", dimension=3072),
    )

    assert request.build_mode == "full"
    assert request.commit_policy == "persist"
    assert request.project_id == "p1"
    assert request.version_refs[0].version_id == "v1"


def test_dry_run_rejects_idempotency_key() -> None:
    with pytest.raises(ValidationError, match="idempotency_key"):
        SkillManageBuildRequest(
            project_id="p1",
            graph_id="g1",
            commit_policy="dry_run",
            idempotency_key="k",
        )


def test_remove_selector_is_discriminated() -> None:
    by_version = SkillManageRemoveRequest(
        project_id="p1",
        graph_id="g1",
        selector=SkillManageRemoveSelector(kind="version", skill_id="s1", version_id="v1"),
    )
    by_family = SkillManageRemoveRequest(
        project_id="p1",
        graph_id="g1",
        selector=SkillManageRemoveSelector(kind="family", skill_id="s1"),
    )

    assert by_version.selector.kind == "version"
    assert by_family.selector.kind == "family"
    assert by_family.selector.version_id is None


def test_retrieve_request_defaults_to_skill_only() -> None:
    request = SkillManageRetrieveRequest(project_id="p1", graph_id="g1", query="find me", top_k=5)

    assert request.mode is RetrievalMode.SKILL_ONLY
    assert request.top_k == 5
