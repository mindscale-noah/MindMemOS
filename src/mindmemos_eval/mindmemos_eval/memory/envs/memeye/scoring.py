"""MemEye scoring: debiased MCQ exact match, OPEN BLEU-1 and the MemEye LLM-judge rubric.

Mirrors the benchmark protocol: an MCQ question is asked once per option rotation and scores the
fraction of rotations answered correctly; OPEN answers get BLEU-1 (normalized tokens, NLTK
smoothing method 1) and a 0 / 0.25 / 0.5 / 0.75 / 1 judge score.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

JUDGE_PROMPT = (Path(__file__).with_name("judge_prompt.txt")).read_text(encoding="utf-8")
JUDGE_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
_ARTICLES = re.compile(r"\b(a|an|the)\s+(?=\w)")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", text).replace("_", "UNDERSCOREPLACEHOLDER")
    text = _ARTICLES.sub(" ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = text.replace("DOTPLACEHOLDER", ".").replace("UNDERSCOREPLACEHOLDER", "_")
    return " ".join(text.split())


def bleu1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return 0.0
    return float(sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1))


def extract_choice(text: str, valid: set[str]) -> str:
    upper = text.strip().upper()
    if upper in valid:
        return upper
    keys = "|".join(sorted(valid))
    for pattern in (rf"^({keys})[\.\)\:\s,]", rf"(?:ANSWER|CHOICE)\s*(?:IS|:)\s*({keys})\b", rf"\b({keys})\b"):
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return "INVALID"


def judge_messages(question: str, ground_truth: str, answer: str) -> list[dict[str, Any]]:
    prompt = (
        JUDGE_PROMPT.replace("{{question}}", question)
        .replace("{{ground_truth}}", ground_truth)
        .replace("{{model_output}}", answer)
    )
    return [{"role": "user", "content": prompt}]


def parse_judge(text: str) -> float | None:
    for match in re.finditer(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL):
        try:
            score = _snap(json.loads(match.group()).get("score"))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if score is not None:
            return score
    match = re.search(r"\"?score\"?\s*:\s*([0-9.]+)", text, re.IGNORECASE)
    return _snap(match.group(1)) if match else None


def _snap(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if score in JUDGE_SCORES else None
