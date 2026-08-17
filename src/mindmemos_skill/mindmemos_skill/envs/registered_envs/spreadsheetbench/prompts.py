"""SpreadsheetBench Agent execution prompts."""

from __future__ import annotations

from ....typing import Task

SYSTEM_PROMPT = (
    "You are an expert spreadsheet assistant. Your working directory contains a "
    "source Excel file named 'input.xlsx'. Do NOT modify 'input.xlsx'. Instead, "
    "produce a new file 'output.xlsx' in the same directory that fully satisfies "
    "the user's request (start from a copy of 'input.xlsx' and apply your changes).\n"
    "Work by writing and running Python (openpyxl is available) through the shell "
    "tool — do not answer from memory. Inspect the sheets first, apply the changes, "
    "save to 'output.xlsx', and verify.\n"
    "IMPORTANT: 'output.xlsx' is graded by reading cached cell VALUES, with no "
    "formula recalculation. Write the final computed values into the target cells "
    "(not bare formulas), since an unevaluated formula reads back as empty.\n"
    "When you are done, stop without calling any tool."
)


def build_messages(*, task: Task, skill_names: list[str], expose_answer_position: bool = False) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT
    if skill_names:
        names = ", ".join(skill_names)
        system += (
            f"\n\nA `skill` tool is available with expert skills: {names}. "
            f'Call it first (e.g. skill(name="{skill_names[0]}")) to load detailed '
            "guidance and the absolute path to reusable reference scripts before you start."
        )
    user_parts = [
        "The source file 'input.xlsx' is in your working directory.",
        f"Task:\n{task.instruction}",
    ]
    answer_position = str(task.metadata.get("answer_position") or "").strip() if expose_answer_position else ""
    answer_sheet = str(task.metadata.get("answer_sheet") or "").strip()
    if answer_position and answer_sheet and "!" not in answer_position:
        answer_position = f"{answer_sheet}!{answer_position}"
    if answer_position:
        user_parts.append(
            "Answer position (cells to verify) — these are the cells that will be "
            "graded, so make sure their final computed values are correct:\n"
            f"{answer_position}"
        )
    user_parts.append(
        "Complete the task and save the result as 'output.xlsx' "
        "(do not modify 'input.xlsx')."
    )
    user = "\n\n".join(user_parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = ["SYSTEM_PROMPT", "build_messages"]
