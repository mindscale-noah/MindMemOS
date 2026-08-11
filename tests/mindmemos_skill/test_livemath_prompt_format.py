from mindmemos_skill.envs.registered_envs.livemath import build_system, evaluate, refinement


def test_livemath_prompts_require_visible_reasoning_before_answer() -> None:
    system = build_system("Compare every option.")
    retry = refinement("I am unsure.")

    assert "<think>...</think>" in system
    assert system.index("<think>...</think>") < system.index("<answer>...</answer>")
    assert "Always output both blocks" in system
    assert "<think>...</think>" in retry
    assert retry.index("<think>...</think>") < retry.index("<answer>...</answer>")


def test_livemath_evaluator_ignores_tagged_reasoning() -> None:
    choices = [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}]
    response = "<think>The second option is exact.</think>\n<answer>B</answer>"

    result = evaluate(response, {"label": "B", "text": "second"}, choices)

    assert result["em"] == 1
    assert result["predicted_label"] == "B"
