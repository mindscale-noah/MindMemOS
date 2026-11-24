"""Regression coverage for file integrity and guarded model-response decoding."""

import asyncio
from unittest.mock import patch

import pytest
from mindmemos_skill.common.json import (
    append_jsonl,
    extract_json,
    extract_json_array,
    extract_json_object,
    parse_model,
    read_json,
    read_jsonl,
    strict_json_schema_response_format,
    write_json,
)
from pydantic import BaseModel, RootModel, ValidationError


class Answer(BaseModel):
    """Minimal structured reply used to verify schema validation."""

    value: int
    """Required answer value."""


def test_json_roundtrip_models_unicode_and_atomic_failure(tmp_path):
    path = tmp_path / "nested/summary.json"
    write_json(path, {"answer": Answer(value=3), "text": "中文"})
    assert read_json(path) == {"answer": {"value": 3}, "text": "中文"}
    assert "中文" in path.read_text()
    before = path.read_bytes()
    for invalid in [object(), float("nan")]:
        with pytest.raises((TypeError, ValueError)):
            write_json(path, invalid)
        assert path.read_bytes() == before
    with patch("pathlib.Path.replace", side_effect=OSError("unavailable")):
        with pytest.raises(OSError, match="unavailable"):
            write_json(path, {"new": True})
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_json_files_are_not_repaired(tmp_path):
    path = tmp_path / "invalid.json"
    for text in ['{"value":1,}', "NaN"]:
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


async def test_jsonl_streaming_append_preserves_records_and_reports_line(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"value":0}')
    (await append_jsonl(path, Answer(value=1)))
    (await append_jsonl(path, [{"message": "line\nbreak"}]))
    assert list(read_jsonl(path)) == [{"value": 0}, {"value": 1}, [{"message": "line\nbreak"}]]
    before = path.read_bytes()
    with pytest.raises(TypeError):
        (await append_jsonl(path, object()))
    assert path.read_bytes() == before
    with path.open("a") as stream:
        stream.write('\n{"broken":\n')
    records = read_jsonl(path)
    for _ in range(3):
        next(records)
    with pytest.raises(ValueError, match="line 5"):
        next(records)


async def test_concurrent_jsonl_appends_keep_every_record(tmp_path):
    path = tmp_path / "nested/events.jsonl"
    await asyncio.gather(*(append_jsonl(path, {"id": index, "text": "x" * 10000}) for index in range(32)))
    rows = list(read_jsonl(path))
    assert len(rows) == 32
    assert {row["id"] for row in rows} == set(range(32))
    assert all(row["text"] == "x" * 10000 for row in rows)


@pytest.mark.parametrize(
    "text",
    [
        '{"value":3}',
        'Answer: {"value":3} done.',
        '```json\n{"value":3}\n```',
        '```JSON\n{"value":3,}\n```',
        '```\n{"value":3}\n```',
        "{'value':3}",
        '{"value":3,}',
        'Reasoning "literal {x}" then {"value":3}',
        '```json\nnot json\n```\nAnswer: {"value":3}',
    ],
)
def test_extract_and_validate_model(text):
    assert parse_model(text, Answer) == Answer(value=3)


def test_fence_precedence_and_nested_delimiters():
    assert extract_json_object('```json\n{"value":3}\n``` then {"value":4}') == {"value": 3}
    payload = {"text": 'braces } [ and "quotes" and \\ paths', "nested": [{"value": 1}]}
    import json

    assert extract_json_object("Answer: " + json.dumps(payload)) == payload
    assert extract_json_object('{"text":"a "quoted" word",}') == {"text": 'a "quoted" word'}
    assert extract_json_object("{'text': 'a', 'other': \"it's fine\"}") == {"text": "a", "other": "it's fine"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "nothing",
        '"literal {x: 1}"',
        "Use {op: delete}",
        "Use `[todo]`",
        '{"value":3} then {"value":4}',
        '{"value":3,} then {"value":4,}',
        '```json\n{"value":3}\n```\n```json\n{"value":4}\n```',
        '{"value":',
        '{"value":}',
        '{"value":"truncated}',
        '{"value":3',
        '{"value":3]',
        '{"value":NaN}',
    ],
)
def test_invalid_or_ambiguous_responses_raise(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_arrays_and_wrong_root_types():
    assert extract_json_array('Items: [{"value":3}, 4,]') == [{"value": 3}, 4]
    assert parse_model("[1, 2,]", RootModel[list[int]]).root == [1, 2]
    assert extract_json_object("{}") == {}
    assert extract_json_array("[]") == []
    with pytest.raises(ValueError):
        extract_json_object('[{"value":3}]')
    with pytest.raises(ValueError):
        extract_json_array('{"nested": [1]}')
    with pytest.raises(ValidationError):
        parse_model('{"other":3}', Answer)
    with pytest.raises(ValidationError):
        parse_model('{"value":"bad"}', Answer)


def test_repair_controls_and_schema_request():
    with pytest.raises(ValueError):
        extract_json_object('{"value":3,}', repair=False)
    assert extract_json_object('{"value":3', allow_incomplete=True) == {"value": 3}
    for text in ["{", '{"value":', '{"value":"unfinished']:
        with pytest.raises(ValueError):
            extract_json_object(text, allow_incomplete=True)
    schema = Answer.model_json_schema()
    assert strict_json_schema_response_format("Answer", schema) == {
        "type": "json_schema",
        "json_schema": {"name": "Answer", "strict": True, "schema": schema},
    }


def test_chat_action_shared_parser_preserves_quotes_and_rejects_ambiguity():
    from mindmemos_skill.agents.buitin_agents.openai_chat.callbacks import parse_action

    action = parse_action('Action: {"name":"write","arguments":{"text":"it\'s a test"},}')
    assert action == {"name": "write", "arguments": {"text": "it's a test"}}
    assert parse_action("ACTION: TASK_COMPLETE") is None
    with pytest.raises(ValueError, match="Malformed Action"):
        parse_action('Action: {"name":"first"} {"name":"second"}')
