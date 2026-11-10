"""SDK request models must stay in lockstep with the server's typing contracts.

The SDK deliberately mirrors ``mindmemos`` models instead of importing them (it ships without
the server package). These checks fail loudly when one side renames a field or flips its
requiredness, which would otherwise surface only as server-side 422s at runtime.
"""

from __future__ import annotations

from mindmemos.typing.memory import DialogueMessage as ServerDialogueMessage
from mindmemos.typing.memory import ImageRef as ServerImageRef
from mindmemos_sdk.memory.models import DialogueMessage as SdkDialogueMessage
from mindmemos_sdk.memory.models import ImageRef as SdkImageRef
from pydantic import BaseModel


def _shape(model: type[BaseModel]) -> tuple[set[str], set[str]]:
    fields = model.model_fields
    return set(fields), {name for name, field in fields.items() if field.is_required()}


def test_image_ref_fields_match_the_server_contract() -> None:
    server_names, server_required = _shape(ServerImageRef)
    sdk_names, sdk_required = _shape(SdkImageRef)

    assert sdk_names == server_names, f"field sets drifted: {sdk_names ^ server_names}"
    assert sdk_required == server_required, f"requiredness drifted: {sdk_required ^ server_required}"


def test_dialogue_message_fields_match_the_server_contract() -> None:
    server_names, server_required = _shape(ServerDialogueMessage)
    sdk_names, sdk_required = _shape(SdkDialogueMessage)

    assert sdk_names == server_names, f"field sets drifted: {sdk_names ^ server_names}"
    assert sdk_required == server_required, f"requiredness drifted: {sdk_required ^ server_required}"
