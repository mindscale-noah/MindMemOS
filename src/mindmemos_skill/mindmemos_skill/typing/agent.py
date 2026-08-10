"""Agent configuration snapshots used by Skill algorithms."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
)

from ..persistence.enums import AgentType, SkillInjectionMode

_API_KEY_DIGEST_PREFIX = "sha256:"


class AgentProfile(BaseModel):
    """One reproducible Agent configuration snapshot.

    Common provider parameters are first-class fields. ``config`` is reserved
    for implementation-specific options. API keys stay masked in memory and
    become irreversible digests whenever the model is serialized.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    COMMON_CONFIG_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "provider": ("provider",),
        "model": ("model",),
        "base_url": ("base_url", "api_base"),
        "api_key": ("api_key",),
        "temperature": ("temperature",),
        "top_p": ("top_p",),
        "max_tokens": ("max_tokens",),
        "max_completion_tokens": ("max_completion_tokens",),
        "max_retries": ("max_retries", "num_retries", "retry"),
        "timeout_seconds": ("timeout_seconds", "timeout"),
        "reasoning_effort": ("reasoning_effort",),
        "max_turns": ("max_turns",),
        "skill_injection_mode": ("skill_injection_mode",),
    }

    agent_type: AgentType = AgentType.UNKNOWN
    """执行轨迹的 Agent 实现。"""

    provider: str | None = Field(default=None, min_length=1)
    """模型或 Agent 服务提供商，例如 openai、anthropic。"""

    model: str | None = Field(default=None, min_length=1)
    """执行时使用的模型名称或部署别名。"""

    base_url: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("base_url", "api_base"),
    )
    """模型服务的基础 URL。"""

    api_key: SecretStr | None = None
    """模型服务密钥；序列化时仅输出不可逆 SHA-256 摘要。"""

    temperature: float | None = Field(default=None, ge=0)
    """模型采样温度。"""

    top_p: float | None = Field(default=None, ge=0, le=1)
    """模型 nucleus sampling 参数。"""

    max_tokens: int | None = Field(default=None, ge=1)
    """单次生成允许的最大 token 数。"""

    max_completion_tokens: int | None = Field(default=None, ge=1)
    """单次 completion 允许生成的最大 token 数。"""

    max_retries: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("max_retries", "num_retries", "retry"),
    )
    """请求失败后的最大重试次数。"""

    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("timeout_seconds", "timeout"),
    )
    """单次请求或 Agent 执行的超时时间，单位为秒。"""

    reasoning_effort: str | None = Field(default=None, min_length=1)
    """模型推理强度或等价 provider 参数。"""

    max_turns: int | None = Field(default=None, ge=1)
    """Agent 单次执行允许的最大交互轮数。"""

    skill_injection_mode: SkillInjectionMode | None = None
    """Skill 内容进入 Agent 上下文所采用的运行时机制。"""

    config: dict[str, Any] = Field(default_factory=dict)
    """Agent 实现特有、尚未提升为通用字段的扩展配置。"""

    @classmethod
    def from_config(
        cls,
        *,
        agent_type: AgentType,
        config: Mapping[str, Any],
    ) -> AgentProfile:
        """Split a flat runtime config into common fields and Agent-specific extensions."""

        remaining = dict(config)
        common: dict[str, Any] = {}
        for field_name, accepted_names in cls.COMMON_CONFIG_FIELDS.items():
            for name in accepted_names:
                if name in remaining:
                    common[field_name] = remaining.pop(name)
                    break
            for name in accepted_names:
                remaining.pop(name, None)
        return cls.model_validate({"agent_type": agent_type, "config": remaining, **common})

    @classmethod
    def from_serialized(
        cls,
        payload: Mapping[str, Any],
        *,
        agent_type: AgentType = AgentType.UNKNOWN,
    ) -> AgentProfile:
        """Load a current profile or promote one legacy flat ``config`` snapshot."""

        if "config" not in payload:
            return cls.from_config(agent_type=agent_type, config=payload)
        return cls.model_validate({**payload, "agent_type": agent_type})

    @field_serializer("api_key")
    def serialize_api_key(self, value: SecretStr | None) -> str | None:
        """Replace plaintext API keys with a stable, irreversible digest."""

        if value is None:
            return None
        secret = value.get_secret_value()
        if secret.startswith(_API_KEY_DIGEST_PREFIX):
            return secret
        return f"{_API_KEY_DIGEST_PREFIX}{hashlib.sha256(secret.encode()).hexdigest()}"


__all__ = ["AgentProfile", "AgentType", "SkillInjectionMode"]
