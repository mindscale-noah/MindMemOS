"""Citations bound skill scope without imposing issue-specific locations or code-matching rules."""

import shutil

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.change_gate import changed_files
from mindmemos_skill.algos.evolve.offline_evolution.change_rules import edit_problems
from mindmemos_skill.algos.evolve.offline_evolution.typing import Change, RunLayout

from .helpers import change, make_inputs, ref

ACCEPTED = {
    "F1": ref(),
    "L1": ref("loading"),
    "R1": ref("requirement"),
    "U1": ref("uncovered", None),
    "A1": ref("contrast"),
    "N1": ref(None),
    "X1": ref("content", "other"),
}


def _layout(tmp_path) -> RunLayout:
    skills, trajectories = make_inputs(tmp_path)
    layout = RunLayout(root=tmp_path / "run")
    layout.root.mkdir()
    layout.snapshot(skills, trajectories)
    shutil.copytree(layout.input_skills, layout.edited_skills)
    return layout


def _write(layout: RunLayout, path: str, text: str) -> None:
    target = layout.edited_skills / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _append(layout: RunLayout, path: str, text: str) -> None:
    _write(layout, path, (layout.edited_skills / path).read_text() + text)


def _check(layout: RunLayout, files: list[str], findings: list[str], **extra) -> list[str]:
    claim = Change.model_validate(change("C1", files, findings, **extra))
    return edit_problems(claim, layout, ACCEPTED, changed_files(layout.input_skills, layout.edited_skills))


@pytest.mark.parametrize("finding", ["F1", "L1", "R1", "A1", "N1"])
def test_related_findings_allow_description_body_and_code_edits(tmp_path, finding):
    layout = _layout(tmp_path)
    main = layout.edited_skills / "pdf/SKILL.md"
    _write(
        layout,
        "pdf/SKILL.md",
        main.read_text().replace("Fill and inspect PDF forms.", "Fill and validate form fields.")
        + "\nInspect required fields before saving.\n",
    )
    _write(
        layout,
        "pdf/scripts/check.py",
        "def inspect_fields(values):\n    return all(value is not None for value in values)\n",
    )
    assert _check(layout, ["pdf/SKILL.md", "pdf/scripts/check.py"], [finding]) == []


def test_skill_attribution_still_bounds_edits(tmp_path):
    layout = _layout(tmp_path)
    _append(layout, "pdf/forms.md", "\nInspect fields.\n")
    assert _check(layout, ["pdf/forms.md"], ["F1"]) == []
    assert any("cites no finding concerning it" in p for p in _check(layout, ["pdf/forms.md"], ["X1"]))


def test_new_code_and_cli_examples_need_no_worked_case_or_line_match(tmp_path):
    layout = _layout(tmp_path)
    _append(layout, "pdf/forms.md", "\n```sh\npython3 scripts/fields.py --input '<file>'\n```\n")
    _write(
        layout,
        "pdf/scripts/fields.py",
        "def inspect_fields(values):\n    return [name for name, value in values.items() if value is None]\n",
    )
    assert _check(layout, ["pdf/forms.md", "pdf/scripts/fields.py"], ["F1"]) == []


def test_new_bundled_resources_must_be_discoverable(tmp_path):
    layout = _layout(tmp_path)
    _write(layout, "pdf/scripts/fields.py", "print('fields')\n")
    assert _check(layout, ["pdf/scripts/fields.py"], ["F1"]) == [
        "pdf/scripts/fields.py is new but no Markdown file of pdf names it where a step uses it",
    ]
    _append(layout, "pdf/forms.md", "\nRun `scripts/fields.py` to inspect fields.\n")
    assert _check(layout, ["pdf/scripts/fields.py", "pdf/forms.md"], ["F1"]) == []


def test_new_skills_need_gap_evidence_but_no_numbered_steps(tmp_path):
    layout = _layout(tmp_path)
    _write(
        layout,
        "notes/SKILL.md",
        "---\nname: notes\ndescription: Keep meeting notes.\n---\n\nCapture decisions and action owners.\n",
    )
    assert any("without citing an uncovered finding" in p for p in _check(layout, ["notes/SKILL.md"], ["F1"]))
    assert _check(layout, ["notes/SKILL.md"], ["U1"]) == []


def test_skill_deletion_and_false_source_quotes_are_rejected(tmp_path):
    layout = _layout(tmp_path)
    shutil.rmtree(layout.edited_skills / "pdf")
    files = ["pdf/SKILL.md", "pdf/forms.md", "pdf/scripts/check.py"]
    assert any("deleting the skill pdf" in p for p in _check(layout, files, ["F1"]))
    shutil.copytree(layout.input_skills / "pdf", layout.edited_skills / "pdf")
    _append(layout, "pdf/forms.md", "\nInspect fields.\n")
    assert any(
        "skill quote not found" in p
        for p in _check(layout, ["pdf/forms.md"], ["F1"], skill_quotes=[{"path": "pdf/forms.md", "quote": "invented"}])
    )


def test_binary_files_must_copy_original_resources(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf/logo.png").write_bytes(b"\x89PNG\0new")
    _append(layout, "pdf/forms.md", "\nThe form uses `logo.png`.\n")
    assert _check(layout, ["pdf/logo.png", "pdf/forms.md"], ["F1"]) == [
        "pdf/logo.png is binary and does not copy a file of the original skills",
    ]
