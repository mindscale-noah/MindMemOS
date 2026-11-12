"""Agent connection settings and algorithm-scoped configuration references."""

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    InstanceOf,
    SerializeAsAny,
    StringConstraints,
    ValidationInfo,
    field_serializer,
    model_validator,
)
from pydantic_core import CoreSchema

from ..registry.config import serialize_options

AgentName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")]
"""Algorithm-prefixed preset name in snake_case, such as treeskill_qwen3_5_fast."""


class AgentConnectionConfig(BaseModel):
    """Keep model credentials and endpoints exclusively in configuration."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", hide_input_in_errors=True)
    """Reject unknown fields and omit input values from validation error messages."""

    class_name: str | None = Field(default=None, min_length=1)
    """Registered Agent backend; None is reserved for shared connection-only settings."""

    api_key: str | None = Field(default=None, min_length=1)
    """Model credential; required for model calls and preserved in serialized configuration."""

    base_url: str | None = Field(default=None, min_length=1)
    """Model endpoint; required for model calls, with no environment fallback."""

    def require_connection(self) -> None:
        """Require explicit connection settings before invoking a model.

        Raises:
            ValueError: The configuration omits a credential or endpoint.
        """
        if self.api_key is None:
            raise ValueError("Provide api_key in the agent config")
        if self.base_url is None:
            raise ValueError("Provide base_url in the agent config")


class AgentDefinition(BaseModel):
    """Select a registered Agent backend and validate its preset options."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True, hide_input_in_errors=True)
    """Reject unknown preset fields and hide input credentials in validation errors."""

    class_name: str = Field(min_length=1)
    """Registered Agent implementation that owns this preset's configuration model."""

    config: dict[str, Any] | SerializeAsAny[InstanceOf[BaseModel]] = Field(default_factory=dict)
    """Backend-specific options, converted to that backend's config model during validation."""

    serialize_config = field_serializer("config")(serialize_options)
    """Preserve backend-specific fields when serializing the preset."""

    @model_validator(mode="after")
    def validate_backend(self) -> "AgentDefinition":
        """Validate preset options through the existing Agent registry.

        Returns:
            This preset with concrete backend options, without creating an Agent.

        Raises:
            ValueError: The Agent backend is unknown.
            ValidationError: Backend options are invalid.
        """
        from .base import Agent

        implementation = Agent.resolve(self.class_name)
        options = self.config.model_dump() if isinstance(self.config, BaseModel) else dict(self.config)
        options.setdefault("class_name", self.class_name)
        self.config = implementation.config_type.model_validate(options)
        if self.config.class_name != self.class_name:
            raise ValueError("Agent definition and configuration must select the same backend")
        return self


class AgentCatalog(BaseModel):
    """Carry service-local presets through nested Pydantic validation."""

    agents: dict[AgentName, AgentDefinition] = Field(default_factory=dict)
    """Preset names mapped to validated backend definitions; no live Agent instances."""


@dataclass(frozen=True)
class AgentConfig:
    """Annotate a concrete config type with permitted Agent backends and name resolution."""

    allowed: tuple[str, ...]
    """Registered backend names accepted by both inline settings and named references."""

    namespace: str | None = None
    """Owning algorithm prefix; None allows names from any namespace."""

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Add preset resolution before the declared configuration type is validated.

        Args:
            source_type: Concrete config type or union declared by the owning field.
            handler: Pydantic's builder for the field's existing validation schema.

        Returns:
            A schema accepting inline options or a name and producing concrete options.
        """
        if not self.allowed:
            raise ValueError("AgentConfig.allowed must contain at least one Agent backend")
        name_type = AgentName
        if self.namespace is not None:
            name_type = Annotated[str, StringConstraints(pattern=rf"^{self.namespace}_[a-z0-9]+(?:_[a-z0-9]+)*$")]
        validator = BeforeValidator(
            partial(self._resolve, config_type=source_type), json_schema_input_type=source_type | name_type
        )
        return validator.__get_pydantic_core_schema__(source_type, handler)

    def __get_pydantic_json_schema__(self, schema: CoreSchema, handler: GetJsonSchemaHandler) -> dict[str, Any]:
        """Expose permitted backends and algorithm ownership in editor schemas.

        Args:
            schema: Field validation schema including inline and named inputs.
            handler: Pydantic's JSON Schema builder.

        Returns:
            Input schema with Agent backend and namespace metadata.
        """
        result = handler(schema)
        result["x-agent-backends"] = list(self.allowed)
        if self.namespace is not None:
            result["x-agent-namespace"] = self.namespace
        return result

    def _resolve(self, value: object, info: ValidationInfo, *, config_type: Any) -> object:
        """Resolve backend identity and enforce the whitelist without mutating shared inputs."""
        if not isinstance(value, str):
            if isinstance(value, AgentConnectionConfig):
                backend = value.class_name
            elif isinstance(value, dict):
                backend = value.get("class_name")
            else:
                return value
            if backend is None:
                if len(self.allowed) != 1:
                    raise ValueError("Inline Agent configuration requires class_name for multiple allowed backends")
                backend = self.allowed[0]
            if backend not in self.allowed:
                raise ValueError(f"Agent backend {backend!r}; allowed backends: {self.allowed}")
            from .base import Agent

            implementation = Agent.resolve(backend)
            options = value.model_dump() if isinstance(value, BaseModel) else dict(value)
            options["class_name"] = backend
            if isinstance(config_type, type) and issubclass(config_type, implementation.config_type):
                return config_type.model_validate(options)
            return implementation.config_type.model_validate(options)
        if not isinstance(info.context, AgentCatalog):
            raise ValueError("Named Agent configuration requires a service agent catalog")
        definition = info.context.agents.get(value)
        if definition is None:
            raise ValueError(f"Unknown Agent preset {value!r}")
        if self.namespace is not None and not value.startswith(f"{self.namespace}_"):
            raise ValueError(f"Agent preset {value!r} must belong to algorithm {self.namespace!r}")
        if definition.class_name not in self.allowed:
            raise ValueError(f"Agent preset {value!r} uses {definition.class_name!r}; allowed backends: {self.allowed}")
        # Preserve backend identity for union fields, and copy all mutable options.
        options = definition.config.model_copy(deep=True)
        if (
            isinstance(config_type, type)
            and issubclass(config_type, BaseModel)
            and not isinstance(options, config_type)
        ):
            # Algorithm-specific config subclasses still validate their own constraints.
            return options.model_dump()
        return options
