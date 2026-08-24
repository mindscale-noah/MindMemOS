from datetime import UTC, datetime

import pytest
from mindmemos.components.extractor.schema._schema_utils import (
    entity_write_embedding_text,
    format_reference_entities,
    merge_description,
)
from mindmemos.components.extractor.schema.schema_planner import SchemaAddPlanner
from mindmemos.llm import ChatResponse
from mindmemos.prompts import get_add_prompts
from mindmemos.typing.memory import EntityView, EntityWrite


def _entity_write(*, entity_name: str, latest_raw_entity_name: str | None = None) -> EntityWrite:
    metadata = {"latest_raw_entity_name": latest_raw_entity_name} if latest_raw_entity_name else {}
    return EntityWrite(
        entity_id=f"entity-{entity_name}",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        entity_name=entity_name,
        entity_type="person",
        created_at=datetime(2026, 5, 28, tzinfo=UTC),
        metadata=metadata,
    )


def test_properties_for_alias_update_do_not_leak_from_same_named_create() -> None:
    """An alias update (Bobby -> Robert) must not steal properties from a create
    entity that shares its merge target name (Robert) listed earlier.

    Regression guard for the `_properties_for_entity` name-matching order: the
    update entity stores its own raw name in ``latest_raw_entity_name`` and must
    resolve by that first, before falling back to the target ``entity_name``.
    """
    planner = SchemaAddPlanner.__new__(SchemaAddPlanner)
    planner.max_properties_per_entity = 15

    robert_create = _entity_write(entity_name="Robert")
    bobby_update = _entity_write(entity_name="Robert", latest_raw_entity_name="Bobby")

    raw_entity_list = [
        {"name": "Robert", "properties": [{"property_name": "preference", "value": "age 40"}]},
        {"name": "Bobby", "properties": [{"property_name": "preference", "value": "age 41"}]},
    ]

    # The alias update must resolve its own raw entity, not the same-named create.
    assert planner._properties_for_entity(
        raw_entity_list=raw_entity_list, episode_entity={}, entity_write=bobby_update
    ) == [{"property_name": "preference", "value": "age 41"}]

    # The create entity still resolves its own raw entity via the target-name fallback.
    assert planner._properties_for_entity(
        raw_entity_list=raw_entity_list, episode_entity={}, entity_write=robert_create
    ) == [{"property_name": "preference", "value": "age 40"}]


# --- Bounded entity-description merge (v2 rule fusion) -----------------------


def test_merge_description_concatenates_then_drops_oldest_segments() -> None:
    """Below the cap the merge stays a plain concatenation; above it the oldest
    segments are dropped first so the newest information always survives."""
    assert merge_description("first", "second") == "first\nsecond"
    # A duplicate append is a no-op.
    assert merge_description("first\nsecond", "second") == "first\nsecond"

    old = "\n".join(f"observation {i}" for i in range(10))
    merged = merge_description(old, "newest observation", max_chars=60)

    assert len(merged) <= 60
    assert merged.endswith("newest observation")
    assert "observation 0" not in merged


def test_format_reference_entities_truncates_long_descriptions() -> None:
    entity = EntityView(
        entity_id="entity-user",
        project_id="proj-1",
        entity_name="User",
        entity_type="person",
        description="x" * 600,
    )

    formatted = format_reference_entities([entity], description_max_chars=500)

    assert "x" * 500 + "..." in formatted
    assert "x" * 600 not in formatted


def test_entity_write_embedding_text_truncates_description() -> None:
    write = _entity_write(entity_name="User")
    write.description = "y" * 800

    text = entity_write_embedding_text(write, description_max_chars=500)

    assert "y" * 500 in text
    assert "y" * 501 not in text


class RewriteLLM:
    """LLM fake for the description-rewrite task; fails when *fail* is set."""

    def __init__(self, *, fail: bool = False) -> None:
        self.prompts: list[str] = []
        self.fail = fail

    async def chat(self, task, messages, **kwargs):
        assert task == "memory.add.description_rewrite", f"unexpected task {task}"
        self.prompts.append(str(messages[0]["content"]))
        if self.fail:
            raise RuntimeError("llm down")
        return ChatResponse(finish_reason="stop", content="User prefers Qdrant and enjoys painting.", parsed=None)


def _planner_with_rewrite(llm: RewriteLLM) -> SchemaAddPlanner:
    planner = SchemaAddPlanner.__new__(SchemaAddPlanner)
    planner.llm_client = llm
    planner.prompt_set = get_add_prompts("EN")
    planner.description_rewrite_threshold = 100
    planner.description_max_chars = 2000
    return planner


@pytest.mark.asyncio
async def test_rewrite_oversized_descriptions_compresses_hot_entity() -> None:
    llm = RewriteLLM()
    planner = _planner_with_rewrite(llm)
    write = _entity_write(entity_name="User")
    write.description = "\n".join(f"observation {i}" for i in range(30))

    await planner._rewrite_oversized_descriptions([write])

    assert write.description == "User prefers Qdrant and enjoys painting."
    assert write.metadata["description_rewritten"] is True
    # The prompt is fully substituted and carries the accumulated history.
    assert "{entity_name}" not in llm.prompts[0]
    assert "observation 29" in llm.prompts[0]


@pytest.mark.asyncio
async def test_rewrite_skips_episodes_and_short_descriptions() -> None:
    llm = RewriteLLM()
    planner = _planner_with_rewrite(llm)
    episode = _entity_write(entity_name="Episode 1")
    episode.entity_type = "episodes"
    episode.description = "x" * 500
    short = _entity_write(entity_name="User")
    short.description = "short"

    await planner._rewrite_oversized_descriptions([episode, short])

    assert llm.prompts == []
    assert episode.description == "x" * 500
    assert short.description == "short"


@pytest.mark.asyncio
async def test_rewrite_failure_keeps_capped_concatenation() -> None:
    llm = RewriteLLM(fail=True)
    planner = _planner_with_rewrite(llm)
    write = _entity_write(entity_name="User")
    write.description = "z" * 300

    await planner._rewrite_oversized_descriptions([write])

    assert write.description == "z" * 300
    assert "description_rewritten" not in (write.metadata or {})
