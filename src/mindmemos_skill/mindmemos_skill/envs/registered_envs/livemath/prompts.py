"""LiveMath Agent execution prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SYSTEM_PROMPT = """You are an expert mathematical reasoning agent solving multiple-choice questions.

{skill_section}## Task Format
You will receive one mathematics multiple-choice question and its answer choices.
Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.

## Answer Format
Write your complete step-by-step reasoning inside <think>...</think> tags.
Then provide your final answer inside <answer>...</answer> tags.
Inside the answer tags, output only the single choice label, such as A or C.
Always output both blocks, in this order.

Example:
<think>Analyze each option carefully and identify the uniquely correct choice.</think>
<answer>B</answer>
"""


def build_system(skill_content: str) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return SYSTEM_PROMPT.format(skill_section=skill_section)


def build_user(item: Mapping[str, Any], use_theorem: bool, use_sketch: bool) -> str:
    choices = "\n".join(f"{choice['label']}. {choice['text']}" for choice in item["choices"])
    parts = [f"## Question\n{item['question']}", f"## Choices\n{choices}"]
    if use_theorem and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if use_sketch and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
    return "\n\n".join(parts)


def refinement(previous_response: str) -> str:
    return (
        f"Your previous answer was:\n{previous_response}\n\n"
        "Re-evaluate the exact option wording. If needed, correct it. "
        "Output your revised reasoning inside <think>...</think>, followed by only the final choice label "
        "inside <answer>...</answer>."
    )


__all__ = ["SYSTEM_PROMPT", "build_system", "build_user", "refinement"]
