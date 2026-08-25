from mindmemos.prompts import get_add_prompts, get_add_prompts_v1


def test_add_prompt_selector_keeps_english_and_chinese_prompts() -> None:
    en_prompts = get_add_prompts("EN")
    zh_prompts = get_add_prompts("ZH")

    assert "conversation analysis expert" in en_prompts.conv_boundary_detection
    assert "professional entity and relationship extraction expert" in en_prompts.entity_generation
    assert "memory editor" in en_prompts.entity_description_rewrite
    assert "episodic memory expert" in en_prompts.episode_entity
    assert "memory relationship expert" in en_prompts.episode_edge
    assert "episodic memory generation expert" in en_prompts.episode_objectify
    assert "memory extraction schema expert" in en_prompts.schema_selection_for_generation
    assert zh_prompts.conv_boundary_detection
    assert zh_prompts.entity_generation
    assert zh_prompts.episode_entity
    assert zh_prompts.episode_edge
    assert zh_prompts.episode_objectify
    assert zh_prompts.schema_selection_for_generation
    assert zh_prompts.conv_boundary_detection != en_prompts.conv_boundary_detection


def test_add_prompt_v1_selector_keeps_english_and_chinese_prompts() -> None:
    en_prompts = get_add_prompts_v1("EN")
    zh_prompts = get_add_prompts_v1("ZH")

    assert "conversation analysis expert" in en_prompts.conv_boundary_detection
    assert "professional entity and relationship extraction expert" in en_prompts.entity_generation
    assert "higher-order personal traits" in en_prompts.higher_order_generation
    assert "memory property merge expert" in en_prompts.property_merge_decision
    assert "search optimization expert" in en_prompts.search_field_generation
    assert zh_prompts.conv_boundary_detection
    assert zh_prompts.entity_generation
    assert zh_prompts.higher_order_generation
    assert zh_prompts.property_merge_decision
    assert zh_prompts.search_field_generation
    assert zh_prompts.conv_boundary_detection != en_prompts.conv_boundary_detection


def test_conv_boundary_detection_prompt_is_versioned() -> None:
    """v1 keeps the develop boundary prompt (with the reasoning output field); v2 drops it."""
    v1_prompts = get_add_prompts_v1("EN")
    v2_prompts = get_add_prompts("EN")

    assert '"reasoning"' in v1_prompts.conv_boundary_detection
    assert '"reasoning"' not in v2_prompts.conv_boundary_detection
    assert '"reasoning"' in v1_prompts.conv_forced_resplit
    assert '"reasoning"' not in v2_prompts.conv_forced_resplit
    assert '"reasoning"' in get_add_prompts_v1("ZH").conv_boundary_detection
    assert '"reasoning"' not in get_add_prompts("ZH").conv_boundary_detection
