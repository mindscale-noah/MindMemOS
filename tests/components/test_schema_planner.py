from datetime import UTC, datetime

from mindmemos.components.extractor.schema.schema_planner import SchemaAddPlanner
from mindmemos.typing.memory import EntityWrite


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
