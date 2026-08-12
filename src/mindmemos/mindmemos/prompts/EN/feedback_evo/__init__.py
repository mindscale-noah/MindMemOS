"""Prompts for the ``feedback_evo`` self-evolution mode (independent set)."""

from .extraction import FEEDBACK_EVO_ENTITY_TAGGING_PROMPT, FEEDBACK_EVO_EXTRACTION_SYSTEM_PROMPT
from .param_planning import PARAM_PLANNING_PROMPT
from .signal_detection import EVO_SIGNAL_DETECTION_PROMPT

__all__ = [
    "EVO_SIGNAL_DETECTION_PROMPT",
    "FEEDBACK_EVO_ENTITY_TAGGING_PROMPT",
    "FEEDBACK_EVO_EXTRACTION_SYSTEM_PROMPT",
    "PARAM_PLANNING_PROMPT",
]
