from mindmemos.components.text.token_estimator import TOKEN_ESTIMATOR_VERSION, estimate_tokens


def test_estimate_tokens_is_deterministic_for_latin_cjk_and_mixed_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("你好世界") == 2
    assert estimate_tokens("你") == 1
    assert estimate_tokens("你好 hello") == 2
    assert estimate_tokens("   \n\t") == 0
    assert estimate_tokens("hello, world!") == 2
    assert estimate_tokens("hello world") == estimate_tokens("hello world")
    assert TOKEN_ESTIMATOR_VERSION == "heuristic-v1"
