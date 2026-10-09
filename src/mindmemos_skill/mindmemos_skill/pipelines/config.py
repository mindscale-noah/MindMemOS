"""Configuration shared by built-in pipeline components."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, InstanceOf, SerializeAsAny, field_serializer

from ..registry.config import serialize_options


class PipelineComponentConfig(BaseModel):
    """Select an algorithm or environment in its owning registry."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported assembly fields."""

    class_name: str
    """Explicit implementation name in the field's registry."""

    config: dict[str, Any] | SerializeAsAny[InstanceOf[BaseModel]] = Field(default_factory=dict)
    """Implementation-owned model after validation, or raw options before validation."""

    serialize_config = field_serializer("config")(serialize_options)
    """Serialize the actual implementation model without dropping configuration fields."""
