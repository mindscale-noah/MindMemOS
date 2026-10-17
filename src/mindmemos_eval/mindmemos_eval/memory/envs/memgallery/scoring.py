"""Mem-Gallery scoring as in the benchmark's ``memengine/evaluate/evaluation.py``.

Token F1 over Porter-stemmed normalized tokens, BLEU-1 (NLTK smoothing method 1), and the
Mem-Gallery judge rubric whose score is snapped to 0 / 0.5 / 1 (below 0.25 -> 0, below 0.75 ->
0.5, else 1). Normalization is the benchmark's ``normalize_answer_universal`` (MemEye uses the same).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer

from ..memeye.scoring import bleu1, normalize

JUDGE_PROMPT = (Path(__file__).with_name("judge_prompt.txt")).read_text(encoding="utf-8")
_STEMMER = PorterStemmer()
_JSON = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\"score\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

__all__ = ["bleu1", "f1", "judge_messages", "parse_judge"]


def f1(prediction: str, reference: str) -> float:
    pred = [_STEMMER.stem(w) for w in normalize(prediction).split()]
    ref = [_STEMMER.stem(w) for w in normalize(reference).split()]
    same = sum((Counter(pred) & Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def judge_messages(question: str, ground_truth: str, answer: str) -> list[dict[str, Any]]:
    prompt = (
        JUDGE_PROMPT.replace("{{question}}", question)
        .replace("{{ground_truth}}", ground_truth)
        .replace("{{model_output}}", answer)
    )
    return [{"role": "user", "content": prompt}]


def parse_judge(text: str) -> float | None:
    match = _JSON.search(text)
    if match:
        try:
            score = _snap(json.loads(match.group()).get("score"))
        except (json.JSONDecodeError, ValueError, AttributeError):
            score = None
        if score is not None:
            return score
    for pattern in (r"\"score\"\s*:\s*([0-9.]+)", r"score\s*:\s*([0-9.]+)"):
        found = re.search(pattern, text)
        if found:
            score = _snap(found.group(1))
            if score is not None:
                return score
    return None


def _snap(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return 0.0 if score < 0.25 else 0.5 if score < 0.75 else 1.0
