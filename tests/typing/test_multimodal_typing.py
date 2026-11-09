"""Image fields must not change how text-only messages and searches serialize."""

from mindmemos.mappers.db import _model_list_dump
from mindmemos.typing import IMAGE_DATA_REDACTED, AddPipelineInput, DialogueMessage, SearchPipelineInput


def test_text_only_dialogue_message_serializes_as_before() -> None:
    message = DialogueMessage(role="user", content="hi", timestamp=1)
    assert message.model_dump() == {"role": "user", "content": "hi", "timestamp": 1}
    assert message.model_dump_json() == '{"role":"user","content":"hi","timestamp":1}'
    inp = AddPipelineInput(messages=[{"role": "user", "content": "hi"}], event_timestamp_ms=5)
    assert inp.model_dump(by_alias=True)["messages"] == [{"role": "user", "content": "hi", "timestamp": None}]


def test_text_only_search_input_serializes_as_before() -> None:
    plain = SearchPipelineInput(query="q").model_dump()
    assert "query_images" not in plain and "multimodal" not in plain
    assert SearchPipelineInput(query="q", multimodal=True).model_dump()["multimodal"] is True
    with_image = SearchPipelineInput(query="q", query_images=[{"url": "http://x/a.png"}])
    assert with_image.model_dump()["query_images"][0]["url"] == "http://x/a.png"


def test_records_redact_inline_image_bytes_but_keep_the_reference() -> None:
    message = DialogueMessage(role="user", content="", images=[{"data": "aGVsbG8=", "image_id": "A"}])
    [dumped] = _model_list_dump([message])
    assert dumped["images"] == [{"data": IMAGE_DATA_REDACTED, "image_id": "A"}]
    # Normal dumps (queue messages, pipeline hand-offs) keep the bytes.
    assert message.model_dump()["images"][0]["data"] == "aGVsbG8="
