from types import SimpleNamespace

import pytest
from mindmemos.components.chunker import EpisodesChunker
from mindmemos.config.algo.add.schema.chunker import EpisodesChunkerConfig


@pytest.mark.asyncio
async def test_rule_chunker_keeps_open_tail_until_forced() -> None:
    chunker = EpisodesChunker(mode="rule", max_messages=10, split_on_user_speaker=True)
    entries = [
        {"speaker": "user", "content": "first topic", "timestamp": "2026-05-28 10:00:00"},
        {"speaker": "assistant", "content": "answer", "timestamp": "2026-05-28 10:01:00"},
        {"speaker": "user", "content": "second topic", "timestamp": "2026-05-28 10:02:00"},
    ]

    boundaries = await chunker.detect_boundaries(entries, force=False)
    forced_boundaries = await chunker.detect_boundaries(entries, force=True)

    assert [(item.start_idx, item.end_idx) for item in boundaries] == [(0, 1)]
    assert [(item.start_idx, item.end_idx) for item in forced_boundaries] == [(0, 1), (2, 2)]


@pytest.mark.asyncio
async def test_llm_chunker_leaves_buffer_when_boundary_parse_fails() -> None:
    class BadLLM:
        async def chat(self, *args, **kwargs):
            return SimpleNamespace(parsed=None, content="not json")

    chunker = EpisodesChunker(mode="llm", llm_client=BadLLM())

    boundaries = await chunker.detect_boundaries(
        [{"speaker": "user", "content": "hello", "timestamp": "2026-05-28 10:00:00"}],
        force=True,
    )

    assert boundaries == []


def test_streaming_window_size_defaults_to_develop_value() -> None:
    """The shared default must stay develop-compatible (15); v2 opts into a larger
    window explicitly via config (see the *_v2 eval examples)."""
    assert EpisodesChunkerConfig().streaming_window_size == 15
    assert EpisodesChunker().streaming_window_size == 15
