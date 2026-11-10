"""Public passages and private SearchQA references."""

from ...data import RewardData
from ...typing import Reward, RolloutResult, Task


class SearchQATask(Task):
    """A question with public retrieved passages."""

    context: str = ""
    """Retrieved context presented to the answering agent."""


class SearchQARewardData(RewardData):
    """Private accepted answers used for answer scoring."""

    answers: list[str]
    """Accepted reference strings in their original order."""


class SearchQAReward(Reward):
    """SearchQA metrics with verbatim analyst feedback."""

    em: float
    """Normalized exact-match score."""

    f1: float
    """Maximum token-overlap F1 across accepted answers."""

    sub_em: float
    """Bidirectional normalized substring score, including upstream empty semantics."""

    predicted_answer: str
    """Final tagged answer or last nonempty response line."""

    gold_answers: list[str]
    """Private references retained for evaluation audit."""


class SearchQAResult(RolloutResult):
    """Restore typed metrics from persisted results."""

    reward: SearchQAReward
    """Answer metrics and exact training feedback."""
