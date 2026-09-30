"""Evaluate episode success with deterministic failure feedback."""

from ...typing import Reward


def evaluate(*, won: bool, done: bool, max_steps: int) -> Reward:
    """Build a terminal episode reward.

    Args:
        won: Simulator success flag at a terminal step.
        done: Whether the simulator ended before the action budget was exhausted.
        max_steps: Configured action budget used in timeout feedback.

    Returns:
        Binary success and the episode failure string.
    """
    detail = (
        "" if won else ("Episode ended without completing the task" if done else f"Timeout after {max_steps} steps")
    )
    return Reward(score=float(won), success=won, detail=detail, hard=int(won), soft=float(won), fail_reason=detail)
