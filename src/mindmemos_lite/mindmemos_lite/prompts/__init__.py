"""Prompt catalog required by the standalone vanilla algorithm."""

from .EN.add.vanilla import EXTRACTION_SYSTEM_PROMPT
from .EN.add.vanilla_entity import EXTRACTION_SYSTEM_PROMPT_ENTITY
from .EN.add.trajectory_dedup import EXPERIENCE_DEDUP_SYSTEM_PROMPT
from .EN.add.trajectory_experience import EXPERIENCE_EXTRACTION_SYSTEM_PROMPT
from .EN.add.trajectory_plan import PLAN_EXTRACTION_SYSTEM_PROMPT
from .ZH.add.vanilla import EXTRACTION_SYSTEM_PROMPT_ZH
from .ZH.add.vanilla_entity import EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH
from .ZH.add.trajectory_dedup import EXPERIENCE_DEDUP_SYSTEM_PROMPT_ZH
from .ZH.add.trajectory_experience import EXPERIENCE_EXTRACTION_SYSTEM_PROMPT_ZH
from .ZH.add.trajectory_plan import PLAN_EXTRACTION_SYSTEM_PROMPT_ZH


def get_extraction_system_prompt(lang: str, *, enable_entities: bool = False) -> str:
    """Return the same language/entity prompt selected by the full runtime."""

    if enable_entities:
        return EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH if lang == "zh" else EXTRACTION_SYSTEM_PROMPT_ENTITY
    return EXTRACTION_SYSTEM_PROMPT_ZH if lang == "zh" else EXTRACTION_SYSTEM_PROMPT


def get_trajectory_experience_prompt(lang: str, *, extract_type: str = "experience") -> str:
    """Return the selected trajectory extraction prompt.

    Args:
        lang: Use Chinese for ``zh`` and English otherwise.
        extract_type: Select ``experience`` or ``plan`` extraction.

    Returns:
        The system prompt for the selected extraction type and language.

    Raises:
        ValueError: The extraction type is unknown or its prompt is empty.
    """

    if extract_type == "plan":
        prompt = PLAN_EXTRACTION_SYSTEM_PROMPT_ZH if lang == "zh" else PLAN_EXTRACTION_SYSTEM_PROMPT
        if not prompt.strip():
            raise ValueError("plan extraction prompt is empty; fill in the trajectory_plan prompt before use")
        return prompt
    if extract_type != "experience":
        raise ValueError("metadata.extract_type must be 'experience' or 'plan'")

    return EXPERIENCE_EXTRACTION_SYSTEM_PROMPT_ZH if lang == "zh" else EXPERIENCE_EXTRACTION_SYSTEM_PROMPT


def get_trajectory_dedup_prompt(lang: str) -> str:
    """Return the trajectory experience dedup/merge prompt for a language."""

    return EXPERIENCE_DEDUP_SYSTEM_PROMPT_ZH if lang == "zh" else EXPERIENCE_DEDUP_SYSTEM_PROMPT


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT_ENTITY",
    "EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH",
    "EXTRACTION_SYSTEM_PROMPT_ZH",
    "EXPERIENCE_DEDUP_SYSTEM_PROMPT",
    "EXPERIENCE_DEDUP_SYSTEM_PROMPT_ZH",
    "EXPERIENCE_EXTRACTION_SYSTEM_PROMPT",
    "EXPERIENCE_EXTRACTION_SYSTEM_PROMPT_ZH",
    "get_extraction_system_prompt",
    "get_trajectory_dedup_prompt",
    "get_trajectory_experience_prompt",
]
