"""SQuAD normalization and structured answer feedback."""

import re
import string
from collections import Counter

from .typing import SearchQAReward, SearchQARewardData, SearchQATask


def _normalize(text: str) -> str:
    """Remove ASCII punctuation and articles, then collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


async def evaluate(response: str, task: SearchQATask, reward_data: SearchQARewardData) -> SearchQAReward:
    """Score an answer and preserve the exact analyst-facing report.

    Args:
        response: Complete assistant response.
        task: Public question and context.
        reward_data: Private accepted answers and provenance.

    Returns:
        Exact match, token F1, substring match, and original-format feedback.
    """
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    answer = matches[-1].strip() if matches else (lines[-1] if lines else response.strip())
    prediction = _normalize(answer)
    references = [_normalize(gold) for gold in reward_data.answers]
    em = float(prediction in references)
    substring = float(any(gold in prediction or prediction in gold for gold in references))
    predicted_tokens = prediction.split()
    f1 = 0.0
    for gold in references:
        tokens = gold.split()
        if not predicted_tokens:
            f1 = max(f1, float(not tokens))
            continue
        common = sum((Counter(predicted_tokens) & Counter(tokens)).values())
        if common:
            precision, recall = common / len(predicted_tokens), common / len(tokens)
            f1 = max(f1, 2 * precision * recall / (precision + recall))
    detail = (
        f"[EVALUATION RESULT]\n"
        f"Question: {task.instruction}\n"
        f"Predicted answer: {answer!r}\n"
        f"Gold answers: {reward_data.answers!r}\n"
        f"Exact Match: {em}\n"
        f"F1: {f1:.4f}"
    )
    return SearchQAReward(
        score=em,
        success=bool(em),
        detail=detail,
        em=em,
        f1=f1,
        sub_em=substring,
        predicted_answer=answer,
        gold_answers=reward_data.answers,
    )
