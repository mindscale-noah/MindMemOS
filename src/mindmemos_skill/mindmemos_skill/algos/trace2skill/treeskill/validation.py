"""Reference-aligned validation for reusable TreeSkill content."""

from __future__ import annotations

import re

_ANALYSIS_ARTIFACT_RULES = (
    (
        "contains a ground-truth or gold-answer reference",
        r"\b(?:ground[\s-]?truth|gold(?:en)?(?:\s+(?:answer|file|output|workbook)|\.xlsx))\b",
    ),
    (
        "contains evaluator-only guidance",
        r"\b(?:evaluation system|evaluator feedback|official evaluator|reference answer)\b",
    ),
    (
        "contains a run-specific absolute path",
        r"(?:^|[\s`\"'])/(?:mnt|home)/[^\s`\"']*/(?:outputs?|runs?)/",
    ),
)


def analysis_artifact_reason(text: str) -> str:
    """Return why analysis-only text must not enter a reusable Skill."""

    if not text:
        return ""
    for reason, pattern in _ANALYSIS_ARTIFACT_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return ""


__all__ = ["analysis_artifact_reason"]
