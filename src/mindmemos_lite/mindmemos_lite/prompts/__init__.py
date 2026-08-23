"""Prompt catalog required by the standalone vanilla algorithm."""

from .EN.add.vanilla import EXTRACTION_SYSTEM_PROMPT
from .EN.add.vanilla_entity import EXTRACTION_SYSTEM_PROMPT_ENTITY
from .EN.add.trajectory_dedup import EXPERIENCE_DEDUP_SYSTEM_PROMPT
from .EN.add.trajectory_experience import EXPERIENCE_EXTRACTION_SYSTEM_PROMPT
from .ZH.add.vanilla import EXTRACTION_SYSTEM_PROMPT_ZH
from .ZH.add.vanilla_entity import EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH
from .ZH.add.trajectory_dedup import EXPERIENCE_DEDUP_SYSTEM_PROMPT_ZH
from .ZH.add.trajectory_experience import EXPERIENCE_EXTRACTION_SYSTEM_PROMPT_ZH


def get_extraction_system_prompt(lang: str, *, enable_entities: bool = False) -> str:
    """Return the same language/entity prompt selected by the full runtime."""

    if enable_entities:
        return EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH if lang == "zh" else EXTRACTION_SYSTEM_PROMPT_ENTITY
    return EXTRACTION_SYSTEM_PROMPT_ZH if lang == "zh" else EXTRACTION_SYSTEM_PROMPT


def get_trajectory_experience_prompt(lang: str) -> str:
    """Return the trajectory experience-extraction prompt for a language."""

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
