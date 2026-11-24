"""Change-gate warnings flag prose copied from runs, version conditions, rewrites, dropped resources and bloat,
by comparing characters and file structure so they work for any language."""

import json
import shutil

from mindmemos_skill.algos.evolve.offline_evolution.change_gate import run_change_gate
from mindmemos_skill.algos.evolve.offline_evolution.change_warnings import warning_feedback
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout

from .helpers import change, claim, make_inputs, ref

TASK = "Detect the period of the planet transit in the light curve and write it to the report file"
"""Task instruction recorded in the extra trajectory."""

TASK_ZH = "把本周的工作记录整理成周报，按项目分组，最后附上下周计划，并保存到结果目录里的周报文件中。"
"""A user request in Chinese recorded in the extra trajectory."""

RUN_CODE = "from wotan import flatten\nflat = flatten(time, flux, window_length=0.75, method='biweight')"
"""Code an agent ran in the extra trajectory."""

ACCEPTED = {
    "F1": ref(),
    "L1": ref("loading"),
    "C1": ref(quote=RUN_CODE, case="comparison-1"),
    "C2": ref(quote="release 3.1.0", case="comparison-1"),
}
"""Findings cited by the claims in these tests: content, loading and code the runs executed."""


def _layout(tmp_path) -> RunLayout:
    skills, trajectories = make_inputs(tmp_path)
    steps = [{"cmd": "python3 detrend.py", "out": "ok", "task": TASK, "user": TASK_ZH, "code": RUN_CODE}]
    text = json.dumps({"steps": steps}, ensure_ascii=False)
    (trajectories / "task-a" / "run-3.json").write_text(text, encoding="utf-8")
    layout = RunLayout(root=tmp_path / "run")
    layout.root.mkdir()
    layout.snapshot(skills, trajectories)
    shutil.copytree(layout.input_skills, layout.edited_skills)
    return layout


def _gate(layout: RunLayout, files: list[str], acknowledged: list[dict] | None = None):
    claim(layout.change_workspace, change("C1", files, list(ACCEPTED)))
    path = layout.change_workspace / "changes.json"
    body = {**json.loads(path.read_text(encoding="utf-8")), "acknowledged": acknowledged or []}
    path.write_text(json.dumps(body), encoding="utf-8")
    return run_change_gate(layout, ACCEPTED, {p.name for p in layout.input_skills.iterdir() if p.is_dir()} - (set()))


def _append(layout: RunLayout, path: str, text: str) -> None:
    target = layout.edited_skills / path
    target.write_text(target.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_prose_copied_from_runs_is_flagged_in_any_language(tmp_path):
    layout = _layout(tmp_path)
    code = "from wotan import flatten\nflat = flatten(time, flux, window_length=0.5, method='hspline')"
    _append(layout, "pdf/forms.md", f"\n```python\n{code}\n```\n")
    report = _gate(layout, ["pdf/forms.md"])
    assert report.changes[0].status == "accepted"  # code adaptation is permitted
    assert not [w for w in report.warnings if w.kind == "copied_from_trajectories"]  # this warning examines prose

    shutil.copy2(layout.input_skills / "pdf" / "forms.md", layout.edited_skills / "pdf" / "forms.md")
    _append(layout, "pdf/forms.md", f"\n- 用户要求：{TASK_ZH}\n")
    copied = [w for w in _gate(layout, ["pdf/forms.md"]).warnings if w.kind == "copied_from_trajectories"]
    assert copied and copied[0].actionable


def test_general_text_is_not_flagged(tmp_path):
    layout = _layout(tmp_path)
    _append(
        layout, "pdf/forms.md", "\n- Before finishing, confirm that every required field of the form has a value.\n"
    )
    assert _gate(layout, ["pdf/forms.md"]).warnings == []


def test_prose_version_conditions_are_flagged_but_code_is_not(tmp_path):
    layout = _layout(tmp_path)
    _append(
        layout, "pdf/forms.md", "\nWith the library at 2.4.1 or >= 2.5 the call changes.\n```text\nrelease 3.1.0\n```\n"
    )
    keys = {w.key for w in _gate(layout, ["pdf/forms.md"]).warnings}
    assert {"version_condition:pdf/forms.md:2.4.1", "version_condition:pdf/forms.md:>= 2.5"} <= keys
    assert "version_condition:pdf/forms.md:3.1.0" not in keys


def test_description_change_rewrite_and_length_are_flagged(tmp_path):
    layout = _layout(tmp_path)
    skill = layout.edited_skills / "pdf" / "SKILL.md"
    body = "".join(f"{n}. Check form field {n}.\n" for n in range(1, 520))
    skill.write_text(
        f"---\nname: pdf\ndescription: Fill, check and inspect PDF forms.\n---\n\n{body}", encoding="utf-8"
    )
    original = "".join(f"Step {n}: keep the form field {n} as it is.\n" for n in range(12))
    for root in (layout.input_skills, layout.edited_skills):
        (root / "pdf" / "forms.md").write_text(original, encoding="utf-8")
    (layout.edited_skills / "pdf" / "forms.md").write_text(
        "".join(f"New step {n}: check field {n}.\n" for n in range(12)), encoding="utf-8"
    )
    report = _gate(layout, ["pdf/SKILL.md", "pdf/forms.md"])
    kinds = {w.kind for w in report.warnings}
    assert {"description_changed", "large_rewrite", "long_skill"} <= kinds


def test_acknowledged_warnings_need_no_feedback(tmp_path):
    layout = _layout(tmp_path)
    _append(layout, "pdf/forms.md", "\nThe API changed in 2.4.1.\n")
    first = _gate(layout, ["pdf/forms.md"])
    pending = [w.key for w in first.warnings]
    assert pending and warning_feedback(first.warnings)

    second = _gate(layout, ["pdf/forms.md"], [{"warning": key, "reason": "needed here"} for key in pending])
    assert all(w.acknowledged == "needed here" for w in second.warnings)
    assert warning_feedback(second.warnings) == ""


def test_resources_dropped_from_the_instructions_are_flagged(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "forms.md").write_text("# Forms\n\nInspect the fields by hand.\n", encoding="utf-8")
    report = _gate(layout, ["pdf/forms.md"])
    assert [w.key for w in report.warnings if w.kind == "orphaned_resource"] == [
        "orphaned_resource:pdf/scripts/check.py:scripts/check.py"
    ]

    (layout.edited_skills / "pdf" / "scripts" / "check.py").unlink()
    report = _gate(layout, ["pdf/forms.md", "pdf/scripts/check.py"])
    orphaned = [w for w in report.warnings if w.kind == "orphaned_resource"]
    assert len(orphaned) == 1 and "removed" in orphaned[0].message
