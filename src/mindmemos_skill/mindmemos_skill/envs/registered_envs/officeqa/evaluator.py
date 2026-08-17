"""Pinned OfficeQA evaluator matching the official 2026-06-13 scorer."""

from __future__ import annotations

from .official_reward import score_answer

OFFICEQA_REWARD_COMMIT = "309dc01ad21113d240700eecb74d2f3e59f163b6"
OFFICEQA_NUMERIC_TOLERANCE = 0.0


def evaluate(prediction: str, gold: str) -> dict[str, str | float]:
    score = score_answer(
        ground_truth=gold,
        predicted=prediction,
        tolerance=OFFICEQA_NUMERIC_TOLERANCE,
    )
    return {
        "em": score,
        "f1": score,
        "predicted_answer": prediction.strip(),
        "gold_answer": gold,
        "evaluator": "databricks/officeqa:reward.py",
        "evaluator_commit": OFFICEQA_REWARD_COMMIT,
        "tolerance": OFFICEQA_NUMERIC_TOLERANCE,
    }


__all__ = ["OFFICEQA_NUMERIC_TOLERANCE", "OFFICEQA_REWARD_COMMIT", "evaluate"]
