"""Compare adapted rewards with user-supplied reference checkouts, without LLM calls."""

import argparse
import ast
import importlib.util
import itertools
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/mindmemos_skill"))


def _module(path: Path):
    """Load a standalone reference scorer without importing its application."""
    spec = importlib.util.spec_from_file_location("reference_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run independent score and character-level feedback comparisons.

    Raises:
        AssertionError: Any adapted score or feedback differs from its reference.
    """
    import openpyxl
    from mindmemos_skill.envs.builtin_envs.officeqa.scoring import score_answer
    from mindmemos_skill.envs.builtin_envs.searchqa.reward import evaluate
    from mindmemos_skill.envs.builtin_envs.searchqa.typing import SearchQARewardData, SearchQATask
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.reward import compare_workbooks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skillopt-envs", type=Path, required=True, help="Reference skillopt/envs directory")
    parser.add_argument("--treeskill-env", type=Path, help="Reference registered_envs/spreadsheetbench directory")
    args = parser.parse_args()
    root = args.skillopt_envs.resolve()
    office_path = root / "officeqa/official_reward.py"
    office = _module(office_path)
    values = [
        "",
        "0",
        "1",
        "1.5",
        "1,000",
        "−1",
        "$1",
        "($1)",
        "(1977)",
        "1977",
        "March 1977",
        "April 1977",
        "543 million",
        "543 billion",
        "543",
        "543000000",
        "[8, 152260]",
        "[8,152,260]",
        "[2, 1]",
        "[1, 2]",
        "1 2",
        "1 1",
        "one",
        "Federal Trust (OASI)",
        "Federal Trust",
        "Unable to determine",
        "100%",
        "100",
        "1\n2",
        "<FINAL_ANSWER>1</FINAL_ANSWER>",
    ]
    values += [
        f"{number}{unit}"
        for number in (0, -5, 2.66, 1977, 113864)
        for unit in ("", "%", " million", " thousand dollars", " in 2023")
    ]
    for gold, prediction in itertools.product(values, repeat=2):
        actual, expected = score_answer(gold, prediction), office.score_answer(gold, prediction)
        assert actual == expected, (gold, prediction, actual, expected)
    print(f"OfficeQA: {len(values) ** 2} score comparisons passed")

    search_path = root / "searchqa/evaluator.py"
    search = _module(search_path)
    source = ast.parse((root / "searchqa/rollout.py").read_text())
    detail_node = next(
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "eval_detail" for target in node.targets)
    )
    detail_expression = compile(ast.Expression(detail_node), "searchqa-reference-feedback", "eval")
    task = SearchQATask(id="q", instruction="Who?\nUnicode: café")
    count = 0
    for prediction, answers in itertools.product(
        values + ["<answer>wrong</answer><answer>red fox</answer>"],
        [[], [""], ["The red fox"], ["O'Brien", "café"], ["1", "one"]],
    ):
        reference = search.evaluate(prediction, answers)
        actual = evaluate(prediction, task, SearchQARewardData(answers=answers, source_sha256="fixture"))
        assert (actual.em, actual.f1, actual.sub_em) == (reference["em"], reference["f1"], reference["sub_em"])
        expected_detail = eval(
            detail_expression, {"question": task.instruction, "eval_result": reference, "gold_answers": answers}
        )
        assert actual.detail.encode() == expected_detail.encode(), (actual.detail, expected_detail)
        count += 1
    print(f"SearchQA: {count} score and UTF-8 feedback comparisons passed")

    spreadsheet_path = root / "spreadsheetbench/evaluator.py"
    spreadsheet = _module(spreadsheet_path)
    report_path = root / "spreadsheetbench/rollout.py"
    with tempfile.TemporaryDirectory() as temporary:
        gold, output = Path(temporary) / "gold.xlsx", Path(temporary) / "output.xlsx"
        expected_book, actual_book = openpyxl.Workbook(), openpyxl.Workbook()
        expected_sheet, actual_sheet = expected_book.active, actual_book.active
        for index in range(1, 600):
            expected_sheet.cell(index, 1, index)
            actual_sheet.cell(index, 1, -index)
        for index, (expected, actual) in enumerate(
            ((None, ""), (5, "5"), (True, 1), (2.666, 2.67), ("café", "cafe")), 1
        ):
            expected_sheet.cell(index, 2, expected)
            actual_sheet.cell(index, 2, actual)
        expected_book.save(gold)
        actual_book.save(output)
        expected_book.close()
        actual_book.close()
        positions = [
            "Sheet!A1:B10",
            "A1:A599",
            "'Sheet'!B1:B5",
            "Sheet!C1:C3",
            "Missing!A1",
            "Sheet!B1:B5,Sheet!A1:A2",
        ]
        for position in positions:
            assert (
                compare_workbooks(gold, output, position)[0]
                == spreadsheet.compare_workbooks(str(gold), str(output), position)[0]
            )
        if args.treeskill_env:
            tree_path = args.treeskill_env / "evaluator.py"
            parsed = ast.parse(tree_path.read_text())
            parsed.body = [node for node in parsed.body if not (isinstance(node, ast.ImportFrom) and node.level)]
            namespace = {"__name__": "treeskill_reference"}
            exec(compile(parsed, str(tree_path), "exec"), namespace)
            regions = [
                "Sheet!A1:B10",
                "Sheet'!A:B",
                "Sheet!1:5",
                "Sheet!A1:B",
                "Sheet!B1:B5",
                "Sheet!C1:C3",
                "Missing!A1",
            ]
            for position in regions:
                assert compare_workbooks(gold, output, position) == namespace["compare_workbooks"](
                    gold, output, position
                ), position
            print(f"TreeSkill SpreadsheetBench: {len(regions)} score and exact mismatch-text comparisons passed")
    print(f"SpreadsheetBench: {len(positions)} standard-range workbook score comparisons passed")
    for path in (office_path, search_path, root / "searchqa/rollout.py", spreadsheet_path, report_path):
        print(f"reference_sha256 {path.relative_to(root)} {sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
