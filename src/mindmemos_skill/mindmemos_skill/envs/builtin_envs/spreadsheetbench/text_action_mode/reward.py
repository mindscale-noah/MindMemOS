"""Score text-action outputs with the released TreeSkill cached-value evaluator."""

from pathlib import Path

from ....typing import Reward
from ..typing import SpreadsheetBenchRewardData
from .reference_reward import compare_workbooks


def evaluate(reward_data: SpreadsheetBenchRewardData, directories: list[Path]) -> Reward:
    """Compare original outputs without recalculating or reading converted files.

    Args:
        reward_data: Private expected workbooks and released scoring ranges.
        directories: Public case directories in reference order.

    Returns:
        Cached-value score and case details using the same comparator as the analyzer.

    Raises:
        ValueError: Case counts differ or an output is a symbolic link.
    """
    if len(directories) != len(reward_data.cases):
        raise ValueError("Output directories must match reference cases")
    position = reward_data.answer_position
    cases, reports = [], []
    passed, executed = 0, 0
    failure = ""
    for reference, directory in zip(reward_data.cases, directories, strict=True):
        output = directory / "output.xlsx"
        if output.is_symlink():
            raise ValueError("Generated output must not be a symbolic link")
        if not output.is_file():
            cases.append({"no": reference.number, "stage": "exec", "ok": False, "error": "output-not-found"})
            failure = failure or "output-not-found"
            reports.append(f"## Execution (case {reference.number})\nERROR: output-not-found")
            continue
        executed += 1
        correct, detail = compare_workbooks(reference.path, output, position, answer_sheet=reward_data.answer_sheet)
        passed += int(correct)
        if not correct:
            failure = failure or f"eval-mismatch: {detail[:200]}"
        cases.append(
            {"no": reference.number, "stage": "eval", "ok": correct, "reason": detail, "scored_file": "output.xlsx"}
        )
        reports.append(f"## Eval Result (case {reference.number}): {'PASS' if correct else 'FAIL'}\n{detail}")
    success = passed == len(cases)
    return Reward(
        score=passed / len(cases),
        success=success,
        detail="[POST-EXECUTION VERIFICATION]\n\n" + "\n\n---\n\n".join(reports),
        hard=int(success),
        soft=passed / len(cases),
        correct=success,
        n_cases=len(cases),
        n_pass=passed,
        n_exec_pass=executed,
        cases=cases,
        exec_ok=executed == len(cases),
        fail_reason="" if success else failure,
    )
