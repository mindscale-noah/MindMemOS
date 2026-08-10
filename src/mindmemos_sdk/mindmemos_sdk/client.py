"""Synchronous root SDK facade composed from portal configuration."""

from __future__ import annotations

from .config import ConfigManager, SDKConfig, SDKConfigCompilerV2, SDKPortalConfigV2
from .errors import AuthRequiredError
from .memory import MemoryClient
from .memory.core import MemoryDefaults
from .skills import SkillManager
from .transport import HttpTransport


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
        config: SDKPortalConfigV2 | SDKConfig | None = None,
        config_manager: ConfigManager | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        """Handle init."""
        manager = config_manager or ConfigManager()
        if config is None:
            config = manager.load_or_default_portal()
        elif isinstance(config, SDKConfig):
            config = manager.convert_legacy_config(config)
        self._config = config
        portal_profile = SDKConfigCompilerV2().compile(config).profile
        connection = portal_profile.connections[portal_profile.memory_connection]
        identity = portal_profile.identity
        memory_config = portal_profile.memory_defaults
        resolved_base_url = connection.base_url
        resolved_api_key = connection.api_key
        timeout_seconds = connection.timeout_seconds
        max_retries = connection.max_retries

        self._base_url = base_url or resolved_base_url
        self._api_key = api_key or resolved_api_key
        self._user_id = user_id or identity.user_id
        self._app_id = app_id or identity.app_id
        self._agent_id = agent_id or identity.agent_id
        self._session_id = session_id or identity.session_id

        self._owns_transport = transport is None
        resolved_transport = transport
        if resolved_transport is None:
            resolved_transport = HttpTransport(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        self._transport = resolved_transport

        self.skills = SkillManager.from_portal_profile(portal_profile)
        self.memory = MemoryClient(
            self._transport,
            default_user_id=self._user_id,
            default_app_id=self._app_id,
            default_agent_id=self._agent_id,
            default_session_id=self._session_id,
            skill_manager=self.skills,
            memory_defaults=MemoryDefaults(
                user_id=self._user_id,
                app_id=self._app_id,
                agent_id=self._agent_id,
                session_id=self._session_id,
                add_mode=memory_config.add_mode,
                add_default_role=memory_config.add_default_role,
                add_auto_skill_context=memory_config.add_auto_skill_context,
                search_top_k=memory_config.search_top_k,
                search_strategy=memory_config.search_strategy,
                search_rerank=memory_config.search_rerank,
                search_score_threshold=memory_config.search_score_threshold,
                search_filters=memory_config.search_filters,
                get_top_k=memory_config.get_top_k,
                get_filters=memory_config.get_filters,
                feedback_mode=memory_config.feedback_mode,
                dreaming_mode=memory_config.dreaming_mode,
            ),
        )

    def require_api_key(self) -> str:
        """Return the configured API key or raise an auth error."""
        if not self._api_key:
            raise AuthRequiredError("No api_key configured. Run `mindmemos auth` first.")
        return self._api_key

    def close(self) -> None:
        """Release resources created by this root client."""
        self.skills.close()
        if self._owns_transport:
            self._transport.close()

    def __enter__(self) -> MindMemOSClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
