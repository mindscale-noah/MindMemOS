"""Deterministic exact-choice evaluation for LiveMath."""

import json
import re

from .typing import LiveMathReward, LiveMathRewardData, LiveMathTask


def evaluate(response: str, task: LiveMathTask, reward_data: LiveMathRewardData) -> LiveMathReward:
    """Compare the extracted choice with the private reference.

    Args:
        response: Complete assistant response.
        task: Public question and displayed options.
        reward_data: Correct displayed choice and dataset provenance.

    Returns:
        Binary exact-choice score and private comparison details.
    """
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    answer = matches[-1].strip() if matches else (lines[-1] if lines else "")
    label = answer.strip().upper().rstrip(".):")
    choices = {choice.label: choice.text for choice in task.choices}
    if label not in choices:
        matched_text = next(
            (choice.label for choice in task.choices if choice.text.strip().lower() == answer.lower()), None
        )
        first_token = answer.split()[0].upper().rstrip(".):") if answer.split() else ""
        if matched_text is not None:
            label = matched_text
        elif first_token in choices:
            label = first_token
    correct = reward_data.correct_choice
    success = label == correct.label
    detail = {
        "predicted_label": label,
        "predicted_text": choices.get(label, ""),
        "correct_label": correct.label,
        "correct_text": correct.text,
    }
    return LiveMathReward(
        score=float(success),
        success=success,
        detail=json.dumps({"exact_match": success, **detail}, ensure_ascii=False),
        **detail,
    )
