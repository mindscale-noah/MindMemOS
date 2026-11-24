"""Shared JSON extraction for judge verdicts.

Reasoning models frequently wrap their JSON verdict in markdown code fences
(`` ```json ... ``` ``) or interleave prose around it. The standalone judge
scripts tolerate this via ``extract_json``; the library judges should too.
"""

from __future__ import annotations

import json


def extract_json_object(text: str) -> dict:
    """Parse a JSON object out of model text, tolerating markdown fences.

    Mirrors ``skill_and_trajectory_evaluation/llm_client.py::extract_json``:
    first strips an enclosing code fence, then falls back to the outermost
    ``{...}`` span so surrounding prose does not break parsing.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(stripped.splitlines()[1:])
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError(f"could not parse judge JSON from model output: {text[:800]}")


__all__ = ["extract_json_object"]
