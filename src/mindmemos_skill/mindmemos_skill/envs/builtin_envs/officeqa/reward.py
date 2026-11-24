"""OfficeQA official score and original failure feedback."""

from ...typing import Reward
from .scoring import score_answer
from .typing import OfficeQARewardData


async def evaluate(answer: str, reward_data: OfficeQARewardData, *, fail_reason: str = "") -> Reward:
    """Score the extracted answer with the benchmark failure text.

    Args:
        answer: Model answer after the rollout's tag extraction.
        reward_data: Private expected answer.
        fail_reason: Protocol failure, if the model did not finish answering.

    Returns:
        Official binary reward, evaluation identity, and verbatim failure string.
    """
    score = score_answer(reward_data.ground_truth, answer)
    predicted = answer.strip()
    detail = fail_reason or ("" if score else f"predicted '{predicted}' but expected '{reward_data.ground_truth}'")
    return Reward(
        score=score,
        success=bool(score),
        detail=detail,
        em=score,
        f1=score,
        hard=int(score),
        soft=score,
        predicted_answer=predicted,
        gold_answer=reward_data.ground_truth,
        evaluator="databricks/officeqa:reward.py",
        evaluator_commit="309dc01ad21113d240700eecb74d2f3e59f163b6",
        tolerance=0.0,
        fail_reason=detail,
        agent_ok=not fail_reason,
    )
