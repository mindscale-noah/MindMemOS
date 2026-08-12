"""Typed client for the ``/v1/skills/*`` cloud API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from urllib.parse import quote

from mindmemos_skill import SkillBundle as WireSkillBundle
from mindmemos_skill import SkillVersionCore

from ..transport import HttpTransport
from .bundle import compute_content_hash, deserialize_bundle
from .models import (
    CloudSkillsPage,
    EvolveCloudRequest,
    EvolveCloudResult,
    PullVersionContent,
    PullVersionsPage,
    PushVersionRequest,
    PushVersionResult,
    SkillContentData,
    SkillEvolveData,
    SkillEvolveMode,
    SkillListData,
    SkillRegisterData,
    SkillSummary,
    SkillSyncData,
    SkillSyncRequestItem,
    SkillSyncResult,
    SkillVersion,
    SkillVersionsData,
    SyncCloudItem,
    SyncCloudRequest,
    SyncCloudResult,
)


def _path_part(value: str) -> str:
    return quote(value, safe="")


class SkillCloudClient:
    """Skill API resource client over the shared SDK ``HttpTransport``."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def register(
        self,
        *,
        name: str,
        content: str,
        version_label: str | None = None,
        parent_version_id: str | None = None,
    ) -> SkillRegisterData:
        """Register a local skill bundle with the cloud version store."""

        bundle = WireSkillBundle.from_files(deserialize_bundle(content))
        now = datetime.now(UTC)
        version = SkillVersionCore(
            version_id=str(uuid.uuid4()),
            parent_version_ids=[parent_version_id] if parent_version_id else [],
            name=name,
            content_hash=bundle.content_hash,
            version_label=version_label or "0.1.0",
            status="draft",
            origin="local",
            created_at=now,
            updated_at=now,
        )
        body = {
            "operation_id": str(uuid.uuid4()),
            "version": version.model_dump(mode="json"),
            "bundle": bundle.model_dump(mode="json"),
        }
        envelope = self._transport.post_envelope("/v1/skills/register", json=body)
        stored = SkillVersionCore.model_validate((envelope.data or {})["version"])
        return SkillRegisterData(
            cloud_skill_id=stored.cloud_skill_id or "",
            version_id=stored.version_id,
            version_label=stored.version_label,
            content_hash=stored.content_hash,
            status=stored.status.value,
        )

    def push_version(self, request: PushVersionRequest) -> PushVersionResult:
        """Push one client-generated immutable UUID version without private local fields."""

        files = deserialize_bundle(request.content)
        bundle = WireSkillBundle.from_files(files)
        if bundle.content_hash != request.expected_content_hash:
            raise ValueError(
                f"cloud bundle hash mismatch: expected {request.expected_content_hash}, got {bundle.content_hash}"
            )
        created_at = datetime.fromisoformat(request.created_at.replace("Z", "+00:00"))
        version = SkillVersionCore(
            version_id=request.version_id,
            cloud_skill_id=request.cloud_skill_id,
            parent_version_ids=request.parent_version_ids,
            name=request.name,
            content_hash=request.expected_content_hash,
            version_label=request.version_label or "0.1.0",
            commit_message=request.commit_message,
            status=request.status.value,
            version_revision=request.version_revision,
            runtime_type=request.runtime_type,
            runtime_schema_version=request.runtime_schema_version,
            runtime_metadata=request.runtime_metadata,
            origin=request.origin.value,
            metadata=request.metadata,
            created_at=created_at,
            updated_at=created_at,
        )
        envelope = self._transport.post_envelope(
            "/v1/skills/register",
            json={
                "operation_id": request.operation_id,
                "version": version.model_dump(mode="json"),
                "bundle": bundle.model_dump(mode="json"),
            },
        )
        stored = SkillVersionCore.model_validate((envelope.data or {})["version"])
        return PushVersionResult(
            cloud_skill_id=stored.cloud_skill_id or "",
            version_id=stored.version_id,
            content_hash=stored.content_hash,
            status=stored.status.value,
            created_at=stored.created_at.isoformat(),
            received_at=(stored.received_at or stored.created_at).isoformat(),
        )

    def list_cloud_skills(self, *, cursor: str | None = None) -> CloudSkillsPage:
        """List cloud Skill families using the target cursor contract."""

        params = {"cursor": cursor} if cursor else None
        envelope = self._transport.get_envelope("/v1/skills", params=params)
        return CloudSkillsPage.model_validate(envelope.data or {})

    def pull_versions(self, cloud_skill_id: str, *, cursor: str | None = None) -> PullVersionsPage:
        """Read one page of immutable cloud version metadata."""

        params = {"cursor": cursor} if cursor else None
        envelope = self._transport.get_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/versions",
            params=params,
        )
        return PullVersionsPage.model_validate(envelope.data or {})

    def pull_content(self, cloud_skill_id: str, version_id: str) -> PullVersionContent:
        """Download algorithm content only for one cloud version."""

        envelope = self._transport.get_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/versions/{_path_part(version_id)}/content",
        )
        payload = dict(envelope.data or {})
        wire_version = SkillVersionCore.model_validate(payload["version"])
        bundle = WireSkillBundle.model_validate(payload["bundle"])
        result = PullVersionContent(version=_sdk_pull_version(wire_version), content=bundle.canonical_json())
        self._validate_content(result.content, result.version.content_hash)
        return result

    def sync_cloud(self, request: SyncCloudRequest) -> SyncCloudResult:
        """Exchange known immutable version revisions without family pointers."""

        envelope = self._transport.post_envelope(
            "/v1/skills/sync",
            json=request.model_dump(mode="json"),
        )
        return SyncCloudResult.model_validate(envelope.data or {})

    def evolve_cloud(self, request: EvolveCloudRequest) -> EvolveCloudResult:
        """Trigger idempotent cloud evolution from one explicit base version."""

        envelope = self._transport.post_envelope(
            "/v1/skills/evolve",
            json=request.model_dump(mode="json", exclude_none=True, exclude={"operation_id"}),
            headers={"Idempotency-Key": request.operation_id},
        )
        payload = dict(envelope.data or {})
        payload.setdefault("status", envelope.code or "ok")
        return EvolveCloudResult.model_validate(payload)

    def delete_cloud_skill(self, cloud_skill_id: str, *, operation_id: str) -> None:
        """Idempotently soft-delete one cloud Skill family."""

        self._transport.post_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/delete",
            json={},
            headers={"Idempotency-Key": operation_id},
        )

    def list_skills(self) -> list[SkillSummary]:
        """List cloud-managed skills in the current project."""

        envelope = self._transport.get_envelope("/v1/skills")
        return SkillListData.model_validate(envelope.data or {}).skills

    def get_skill(self, cloud_skill_id: str) -> SkillSummary:
        """Return metadata for one cloud-managed skill."""

        envelope = self._transport.get_envelope(f"/v1/skills/{_path_part(cloud_skill_id)}")
        return SkillSummary.model_validate(envelope.data or {})

    def versions_since(
        self,
        cloud_skill_id: str,
        *,
        since: str | None = None,
    ) -> list[SkillVersion]:
        """Return incremental version metadata for one cloud skill."""

        params = {"since": since} if since else None
        envelope = self._transport.get_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/versions",
            params=params,
        )
        return SkillVersionsData.model_validate(envelope.data or {}).versions

    def get_content(
        self,
        cloud_skill_id: str,
        version_id: str,
    ) -> SkillContentData:
        """Download the canonical bundle text for one skill version."""

        envelope = self._transport.get_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/versions/{_path_part(version_id)}/content",
        )
        payload = dict(envelope.data or {})
        wire_version = SkillVersionCore.model_validate(payload["version"])
        bundle = WireSkillBundle.model_validate(payload["bundle"])
        result = SkillContentData(
            version=_sdk_version(wire_version),
            content=bundle.canonical_json(),
        )
        self._validate_content(result.content, result.version.content_hash)
        return result

    def evolve(self, cloud_skill_id: str, *, mode: SkillEvolveMode = "sync") -> SkillEvolveData:
        """Trigger one skill self-evolution pass for ``cloud_skill_id``.

        The server aggregates the injected ``/v1/memory/add`` trajectories bound to
        this skill and, once enough accumulate, mints one or more evolved versions.
        ``evolved`` is false when the pending count is still below the threshold.
        """

        detail = self._transport.get_envelope(f"/v1/skills/{_path_part(cloud_skill_id)}")
        latest = SkillVersionCore.model_validate((detail.data or {})["latest_version"])
        envelope = self._transport.post_envelope(
            "/v1/skills/evolve",
            json={
                "operation_id": str(uuid.uuid4()),
                "cloud_skill_id": cloud_skill_id,
                "base_version_id": latest.version_id,
                "algorithm": "configured",
                "mode": mode,
            },
        )
        payload = dict(envelope.data or {})
        candidate_ids = list(payload.get("candidate_version_ids") or [])
        return SkillEvolveData(
            cloud_skill_id=cloud_skill_id,
            status=str(payload.get("status") or envelope.code or "failed"),
            evolved=payload.get("status") == "succeeded",
            pending_count=0,
            threshold=0,
            new_version_id=payload.get("selected_version_id"),
            new_version_ids=candidate_ids,
        )

    def sync(
        self,
        items: list[SkillSyncRequestItem | dict[str, str]],
    ) -> SkillSyncData:
        """Check whether reported local skills have changed immutable revisions."""

        normalized = [SkillSyncRequestItem.model_validate(item) for item in items]
        result = self.sync_cloud(
            SyncCloudRequest(
                items=[
                    SyncCloudItem(
                        cloud_skill_id=item.cloud_skill_id, known_version_revisions={item.local_version_id: 0}
                    )
                    for item in normalized
                ]
            )
        )
        by_id = {item.cloud_skill_id: item for item in result.items}
        return SkillSyncData(
            results=[
                SkillSyncResult(
                    cloud_skill_id=item.cloud_skill_id,
                    local_version_id=item.local_version_id,
                    has_update=bool(by_id.get(item.cloud_skill_id) and by_id[item.cloud_skill_id].versions),
                    gating_status="changed"
                    if by_id.get(item.cloud_skill_id) and by_id[item.cloud_skill_id].versions
                    else "up_to_date",
                )
                for item in normalized
            ]
        )

    def delete_skill(self, cloud_skill_id: str) -> None:
        """Remove the cloud management relation for one skill."""

        self._transport.post_envelope(f"/v1/skills/{_path_part(cloud_skill_id)}/delete", json=None)

    @staticmethod
    def _validate_content(content: str, expected_hash: str) -> None:
        files = deserialize_bundle(content)
        actual_hash = compute_content_hash(files)
        if actual_hash != expected_hash:
            raise ValueError(f"cloud bundle hash mismatch: expected {expected_hash}, got {actual_hash}")


def _sdk_version(version: SkillVersionCore) -> SkillVersion:
    return SkillVersion(
        version_id=version.version_id,
        cloud_skill_id=version.cloud_skill_id or "",
        name=version.name,
        content_hash=version.content_hash,
        parent_version_ids=version.parent_version_ids,
        version_label=version.version_label,
        status=version.status.value,
        origin=version.origin.value,
        version_revision=version.version_revision,
        runtime_type=version.runtime_type,
        runtime_schema_version=version.runtime_schema_version,
        runtime_metadata=version.runtime_metadata,
        metadata=version.metadata,
        created_at=version.created_at.isoformat(),
        updated_at=version.updated_at.isoformat(),
    )


def _sdk_pull_version(version: SkillVersionCore):
    from .models import PullVersionSummary

    return PullVersionSummary(
        version_id=version.version_id,
        cloud_skill_id=version.cloud_skill_id or "",
        parent_version_ids=version.parent_version_ids,
        name=version.name,
        content_hash=version.content_hash,
        version_label=version.version_label,
        commit_message=version.commit_message,
        origin=version.origin.value,
        status=version.status.value,
        version_revision=version.version_revision,
        runtime_type=version.runtime_type,
        runtime_schema_version=version.runtime_schema_version,
        runtime_metadata=version.runtime_metadata,
        metadata=version.metadata,
        created_at=version.created_at.isoformat(),
        updated_at=version.updated_at.isoformat(),
        received_at=(version.received_at or version.created_at).isoformat(),
    )
