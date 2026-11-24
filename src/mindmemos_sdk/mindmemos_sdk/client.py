"""Root SDK client facade.

Assembles the SDK from local config in the selected SDK home via
:class:`ConfigManager`, builds separate HTTP transports, and exposes resource
clients (``memory`` and ``skills``). Explicit constructor arguments override config so
callers can run without a config file.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from .config import DIRECT_SKILL_API_PATH, ConfigManager, SDKConfig
from .errors import ApiError, AuthRequiredError, InvalidRequestError, MindMemOSSDKError, TransportError
from .memory import MemoryClient
from .skills import SkillClient, TrajectoryReportRequest
from .trajectory import TrajectoryDelivery, TrajectoryEvent, TrajectoryReportResult
from .transport import ACCEPTED_CODE, QUEUED_CODE, HttpTransport


class MindMemOSClient:
    """Public root SDK client composed from config, transport, and resource clients."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        config: SDKConfig | None = None,
        config_manager: ConfigManager | None = None,
        transport: HttpTransport | None = None,
        skill_transport: HttpTransport | None = None,
    ) -> None:
        """Handle init."""
        if config is None:
            manager = config_manager or ConfigManager()
            config = manager.load_or_default()
        self._config = config

        configured_base_url, configured_api_key = config.resolve_memory_connection()
        self._base_url = base_url or configured_base_url
        self._api_key = api_key or configured_api_key
        self._user_id = user_id or config.defaults.user_id
        self._app_id = app_id
        self._agent_id = agent_id
        self._session_id = session_id

        self._transport = transport or HttpTransport(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout_seconds=config.network.timeout_seconds,
            max_retries=config.network.max_retries,
        )

        skill_base_url, skill_api_key = config.resolve_skill_connection(
            memory_base_url=self._base_url, memory_api_key=self._api_key
        )
        self._skill_transport = skill_transport
        if self._skill_transport is None and skill_base_url is not None:
            self._skill_transport = HttpTransport(
                base_url=skill_base_url,
                api_key=skill_api_key,
                timeout_seconds=config.network.timeout_seconds,
                max_retries=0,
            )
        self.skills = SkillClient(
            self._skill_transport, path_prefix="" if skill_transport is None else DIRECT_SKILL_API_PATH
        )
        self.memory = MemoryClient(
            self._transport,
            default_user_id=self._user_id,
            default_app_id=self._app_id,
            default_agent_id=self._agent_id,
            default_session_id=self._session_id,
        )

    def report_trajectory(
        self,
        event: TrajectoryEvent | Mapping[str, object],
        *,
        include_memory: bool = True,
        include_skill: bool = True,
    ) -> TrajectoryReportResult:
        """Send one captured execution to the selected destinations.

        Args:
            event: The captured execution and exact Skill versions used.
            include_memory: Whether to submit the Memory projection.
            include_skill: Whether to submit the shared Skill trajectory when exact versions are known.

        Returns:
            Independent receipts. An unknown outcome may have been stored and
            requires reconciliation before replay.

        Raises:
            ValidationError: The event does not match the trajectory contract.
            InvalidRequestError: Both destinations were excluded.
        """
        event = TrajectoryEvent.model_validate(event)
        if not include_memory and not include_skill:
            raise InvalidRequestError("Select Memory or Skill delivery.")

        user_id = event.user_id or self._user_id
        agent_id = event.agent_id or self._agent_id
        session_id = event.session_id or self._session_id
        result = TrajectoryReportResult(event_id=event.event_id)
        if include_memory:
            result.memory = self._report_memory_trajectory(event, user_id, agent_id, session_id)
        if include_skill:
            result.skill = self._report_skill_trajectory(event, user_id, agent_id, session_id)
        return result

    def _report_memory_trajectory(
        self, event: TrajectoryEvent, user_id: str | None, agent_id: str | None, session_id: str | None
    ) -> TrajectoryDelivery:
        """Project ordered dialogue into a Memory add request."""
        if not user_id:
            return TrajectoryDelivery(status="failed", code="missing_user_id")
        messages = [
            message.model_dump(include={"role", "content", "timestamp"}, exclude_none=True)
            for message in event.messages
            if message.content.strip()
        ]
        if not messages:
            return TrajectoryDelivery(status="failed", code="no_memory_messages")
        metadata = {"source": event.source, "event_id": event.event_id}
        if event.turn_id is not None:
            metadata["turn_id"] = event.turn_id
        try:
            receipt = self.memory.add(
                messages,
                user_id=user_id,
                app_id=event.app_id,
                agent_id=agent_id,
                session_id=session_id,
                mode=event.memory_mode,
                metadata=metadata,
                retry=False,
            )
        except (MindMemOSSDKError, ValidationError) as exc:
            return _delivery_error(exc)
        status = "queued" if receipt.code in {QUEUED_CODE, ACCEPTED_CODE} else "completed"
        return TrajectoryDelivery(status=status, code=receipt.code, request_id=receipt.request_id)

    def _report_skill_trajectory(
        self, event: TrajectoryEvent, user_id: str | None, agent_id: str | None, session_id: str | None
    ) -> TrajectoryDelivery:
        """Submit the full version list as one trajectory."""
        if not event.skill_version_ids:
            return TrajectoryDelivery(status="skipped", code="no_skill_versions")
        if not user_id or not agent_id:
            reason = "missing_user_id" if not user_id else "missing_agent_id"
            return TrajectoryDelivery(status="failed", code=reason)
        request = TrajectoryReportRequest(
            event_id=event.event_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            messages=[message.model_dump(mode="json", exclude_none=True) for message in event.messages],
            skill_version_ids=list(dict.fromkeys(event.skill_version_ids)),
            status=event.status,
            started_at=event.started_at,
            finished_at=event.finished_at,
        )
        try:
            receipt = self.skills.report_trajectory(request, retry=False)
        except (MindMemOSSDKError, ValidationError) as exc:
            return _delivery_error(exc)
        if isinstance(receipt, dict) and receipt.get("skill_version_ids") == request.skill_version_ids:
            trajectory_id = receipt.get("trajectory_id")
            if isinstance(trajectory_id, str) and trajectory_id.strip():
                return TrajectoryDelivery(status="stored", trajectory_id=trajectory_id)
        return TrajectoryDelivery(status="unknown", code="invalid_response")

    def require_api_key(self) -> str:
        """Return the configured API key or raise an auth error."""
        if not self._api_key:
            raise AuthRequiredError("No api_key configured. Run `mindmemos auth` first.")
        return self._api_key

    def close(self) -> None:
        """Release underlying transport resources."""
        self._transport.close()
        if self._skill_transport is not None:
            self._skill_transport.close()

    def __enter__(self) -> MindMemOSClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _delivery_error(exc: MindMemOSSDKError | ValidationError) -> TrajectoryDelivery:
    """Classify a delivery error without exposing request or response content."""
    if isinstance(exc, ApiError):
        status = "unknown"
        if exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code != 408:
            status = "failed"
        return TrajectoryDelivery(
            status=status, code=exc.code or "api_error", request_id=exc.request_id, http_status=exc.status_code
        )
    if isinstance(exc, TransportError):
        return TrajectoryDelivery(status="unknown", code="transport_error")
    if isinstance(exc, ValidationError):
        return TrajectoryDelivery(status="unknown", code="invalid_response")
    if isinstance(exc, AuthRequiredError):
        return TrajectoryDelivery(status="failed", code="auth_required")
    return TrajectoryDelivery(status="failed", code="invalid_request")
