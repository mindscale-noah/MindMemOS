"""Serialize concrete configuration models held by registered component selections."""

from typing import Any

from pydantic import BaseModel, SerializationInfo


def serialize_options(options: BaseModel | dict[str, Any], info: SerializationInfo) -> dict[str, Any]:
    """Serialize the owning implementation's fields using its own model serializer.

    Args:
        options: Validated implementation model or raw configuration fields.
        info: Parent serialization mode, field selection, and omission settings.

    Returns:
        Configuration fields with implementation-owned fields preserved.
    """
    if isinstance(options, dict):
        return options
    return options.model_dump(
        mode=info.mode,
        include=info.include,
        exclude=info.exclude,
        context=info.context,
        by_alias=info.by_alias,
        exclude_unset=info.exclude_unset,
        exclude_defaults=info.exclude_defaults,
        exclude_none=info.exclude_none,
        round_trip=info.round_trip,
    )
