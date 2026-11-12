"""Split add inputs into their text part (handled by vanilla) and their attached images."""

from __future__ import annotations

from dataclasses import dataclass

from ...typing import AddPipelineInput, DialogueMessage, ImageRef, SearchPipelineInput


@dataclass(frozen=True)
class AttachedImage:
    """One image reference with the position and text of the turn carrying it."""

    ref: ImageRef
    message_index: int
    role: str
    timestamp_ms: int
    context: str


def has_images(inp: AddPipelineInput) -> bool:
    return any(isinstance(message, DialogueMessage) and message.images for message in inp.messages)


def is_image_only_turn(message: object) -> bool:
    """A dialogue turn carrying images and no text: dropped from the text-only input."""

    return isinstance(message, DialogueMessage) and bool(message.images) and not message.content.strip()


def without_images(inp: AddPipelineInput) -> AddPipelineInput:
    """The request exactly as a text-only client would have sent it.

    Images are removed from every turn, and turns that carried only images are dropped.
    """

    messages = []
    for message in inp.messages:
        if isinstance(message, DialogueMessage) and message.images:
            if is_image_only_turn(message):
                continue
            message = message.model_copy(update={"images": []})
        messages.append(message)
    return inp.model_copy(update={"messages": messages})


def attached_images(inp: AddPipelineInput, *, context_chars: int) -> list[AttachedImage]:
    """Images in message order, each with the text of its turn and of the reply that follows.

    An image-only turn also takes the text of the turn before it, so it is never context-free.
    """

    result: list[AttachedImage] = []
    for index, message in enumerate(inp.messages):
        if not isinstance(message, DialogueMessage) or not message.images:
            continue
        neighbours = [index, index + 1] if message.content.strip() else [index - 1, index + 1]
        parts = []
        for position in neighbours:
            other = inp.messages[position] if 0 <= position < len(inp.messages) else None
            if isinstance(other, DialogueMessage) and other.content.strip():
                parts.append(f"{other.role}: {other.content.strip()}")
        context = "\n".join(parts)[:context_chars]
        timestamp = message.timestamp if message.timestamp else inp.event_timestamp
        result.extend(
            AttachedImage(ref=ref, message_index=index, role=message.role, timestamp_ms=timestamp, context=context)
            for ref in message.images
        )
    return result


def text_only_search(inp: SearchPipelineInput) -> SearchPipelineInput:
    return inp.model_copy(update={"query_images": []}) if inp.query_images else inp
