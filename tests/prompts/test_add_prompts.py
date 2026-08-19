from mindmemos.prompts import get_add_prompts


def test_add_prompt_selector_keeps_english_and_chinese_prompts() -> None:
    en_prompts = get_add_prompts("EN")
    zh_prompts = get_add_prompts("ZH")

    assert "conversation analysis expert" in en_prompts.conv_boundary_detection
    assert "professional entity and relationship extraction expert" in en_prompts.entity_generation
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
