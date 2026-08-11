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


def build_messages(*, task: Task, skill_names: list[str]) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT
    if skill_names:
        names = ", ".join(skill_names)
        system += (
            f"\n\nA `skill` tool is available with expert skills: {names}. "
            f'Call it first (e.g. skill(name="{skill_names[0]}")) to load detailed '
            "guidance and the absolute path to reusable reference scripts before you start."
        )
    user = (
        "The source file 'input.xlsx' is in your working directory.\n\n"
        f"Task:\n{task.instruction}\n\n"
        "Complete the task and save the result as 'output.xlsx' "
        "(do not modify 'input.xlsx')."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = ["SYSTEM_PROMPT", "build_messages"]
