from mindmemos.components.searcher.retention_token_estimator import (
    RETENTION_TOKEN_ESTIMATOR_VERSION,
    estimate_tokens,
)
from mindmemos.components.text.token_estimator import (
    estimate_tokens as chunker_estimate_tokens,
)


def test_estimate_tokens_scales_unbroken_runs_by_length() -> None:
    # A single word cannot bypass budgets by omitting spaces: runs longer
    # than 8 characters scale linearly instead of costing one flat token.
    assert estimate_tokens("x" * 10000) == 1250
    assert estimate_tokens("A1b2C3d4E5" * 800) == 1000  # 8000-char base64 run
    assert estimate_tokens("https://example.com/" + "a" * 500) == 65  # 521 chars


def test_estimate_tokens_keeps_normal_words_at_one_token() -> None:
    # Words up to 8 characters still cost one token each, so ordinary
    # prose and CJK text keep their word-count-level estimates.
    assert estimate_tokens("the user asked about recall budgets today") == 7
    assert estimate_tokens("用户询问记忆保持预算") == 6  # 10 CJK chars / 1.5
    assert RETENTION_TOKEN_ESTIMATOR_VERSION == "heuristic-v2"


def test_retention_estimator_is_independent_of_the_chunker_estimator() -> None:
    # The add pipeline's shared estimator must stay on word-count semantics;
    # only the retention path scales unbroken runs by length. If this test
    # fails, someone re-coupled the two estimators.
    blob = "x" * 10000
    assert estimate_tokens(blob) == 1250
    assert chunker_estimate_tokens(blob) == 1
