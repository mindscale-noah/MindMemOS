import pytest
from mindmemos_sdk.memory.models import build_search_body, serialize_messages

from mindmemos_sdk import DialogueMessage, ImageRef


def test_text_only_messages_serialize_unchanged() -> None:
    assert serialize_messages([DialogueMessage(role="user", content="hi")]) == [
        {"role": "user", "content": "hi", "timestamp": None}
    ]


def test_images_and_query_images_are_sent_without_empty_fields() -> None:
    [message] = serialize_messages(
        [DialogueMessage(role="user", content="", images=[ImageRef(data="aGk=", image_id="A")])]
    )
    assert message["images"] == [{"data": "aGk=", "image_id": "A"}]
    body = build_search_body(user_id="u", query="q", query_images=[ImageRef(url="https://x/a.png")])
    assert body["query_images"] == [{"url": "https://x/a.png"}]
    assert "query_images" not in build_search_body(user_id="u", query="q")


def test_image_ref_requires_one_source() -> None:
    with pytest.raises(ValueError):
        ImageRef()
