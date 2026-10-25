"""The change gate keeps only claimed, cited and structurally valid edits and returns what is left open."""

import json
import shutil

from mindmemos_skill.algos.evolve.offline_evolution.change_gate import open_items_feedback, run_change_gate
from mindmemos_skill.algos.evolve.offline_evolution.layout import skill_names
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout

from .helpers import change, claim, make_inputs, ref

ACCEPTED = {"F1": ref(), "L1": ref("loading")}
"""Findings the evidence gate accepted in these tests: a content and a loading finding about pdf."""


def _layout(tmp_path) -> RunLayout:
    skills, trajectories = make_inputs(tmp_path)
    layout = RunLayout(root=tmp_path / "run")
    layout.root.mkdir()
    layout.snapshot(skills, trajectories)
    shutil.copytree(layout.input_skills, layout.edited_skills)
    return layout


def _status(report) -> dict[str, str]:
    return {c.id: c.status for c in report.changes}


def _dropped(*findings: str) -> list[dict]:
    return [{"findings": list(findings), "reason": "not needed"}]


def test_unclaimed_edits_are_left_out_and_unknown_findings_rejected(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "forms.md").write_text(
        "# Forms\n\nUse `python3 scripts/check.py`.\n", encoding="utf-8"
    )
    (layout.edited_skills / "pdf" / "notes.md").write_text("unclaimed\n", encoding="utf-8")
    (layout.edited_skills / "pdf" / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: Fill PDF forms with fields.\n---\n\nSee forms.md.\n", encoding="utf-8"
    )
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]), change("C2", ["pdf/SKILL.md"], ["MISSING"]))
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set()))

    assert _status(report) == {"C1": "accepted", "C2": "rejected"}
    assert report.reverted_unclaimed == ["pdf/notes.md"]
    assert report.accepted_files == ["pdf/forms.md"]
    assert not (layout.candidate_skills / "pdf" / "notes.md").exists()
    assert "`pdf/notes.md`" in open_items_feedback(report)


def test_unclaimed_symlink_does_not_block_claimed_edits(tmp_path):
    """Leave an unclaimed symlink out while accepting an independently justified edit."""
    layout = _layout(tmp_path)
    forms = layout.edited_skills / "pdf" / "forms.md"
    forms.write_text("# Forms\nRun `python3 scripts/check.py`.\n", encoding="utf-8")
    outside = tmp_path / "unclaimed.txt"
    outside.write_text("Unclaimed file outside the edited skills.\n", encoding="utf-8")
    (layout.edited_skills / "pdf" / "notes.md").symlink_to(outside)
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]))

    report = run_change_gate(layout, ACCEPTED, {"pdf"})

    assert _status(report) == {"C1": "accepted"}
    assert report.accepted_files == ["pdf/forms.md"]
    assert report.reverted_unclaimed == ["pdf/notes.md"]
    candidate = layout.candidate_skills / "pdf"
    assert (candidate / "forms.md").read_bytes() == forms.read_bytes()
    assert not (candidate / "notes.md").exists()
    assert not (candidate / "notes.md").is_symlink()


def test_changes_sharing_a_file_fail_together(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "forms.md").write_text("# Forms\nchanged\n", encoding="utf-8")
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]), change("C2", ["pdf/forms.md"], ["MISSING"]))
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set()))

    assert set(_status(report).values()) == {"rejected"}
    assert any("shares files" in reason for reason in report.changes[0].reasons)
    assert report.accepted_files == []


def test_shared_file_combines_description_and_body_authorizations(tmp_path):
    layout = _layout(tmp_path)
    main = layout.edited_skills / "pdf" / "SKILL.md"
    main.write_text(
        "---\nname: pdf\ndescription: Fill and validate form fields.\n---\n\nSee forms.md.\nCheck required fields.\n",
        encoding="utf-8",
    )
    claim(
        layout.change_workspace,
        change("C1", ["pdf/SKILL.md"], ["L1"]),
        change("C2", ["pdf/SKILL.md"], ["F1"]),
    )
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set()))
    assert _status(report) == {"C1": "accepted", "C2": "accepted"}
    assert report.accepted_files == ["pdf/SKILL.md"]


def test_shared_file_authorization_does_not_spread_to_other_files(tmp_path):
    layout = _layout(tmp_path)
    for base in (layout.input_skills, layout.edited_skills):
        (base / "other").mkdir()
        (base / "other/SKILL.md").write_text(
            "---\nname: other\ndescription: Inspect reports.\n---\n\nInspect reports.\n"
        )
    main = layout.edited_skills / "pdf" / "SKILL.md"
    main.write_text(main.read_text().replace("Fill and inspect PDF forms.", "Fill and validate form fields."))
    other = layout.edited_skills / "other/SKILL.md"
    other.write_text(other.read_text() + "\nCheck report fields.\n")
    claim(
        layout.change_workspace,
        change("C1", ["pdf/SKILL.md", "other/SKILL.md"], ["L1"]),
        change("C2", ["pdf/SKILL.md"], ["X1"]),
    )
    report = run_change_gate(
        layout, {**ACCEPTED, "X1": ref("content", "other")}, skill_names(layout.input_skills) - (set())
    )
    assert set(_status(report).values()) == {"rejected"}
    assert any("edits other but cites no finding concerning it" in r for c in report.changes for r in c.reasons)


def test_requirement_check_can_accompany_a_script_change(tmp_path):
    layout = _layout(tmp_path)
    forms = layout.edited_skills / "pdf" / "forms.md"
    forms.write_text(forms.read_text() + "\n- Inspect required fields before saving.\n")
    (layout.edited_skills / "pdf" / "scripts" / "check.py").write_text("print('checked')\n")
    claim(layout.change_workspace, change("C1", ["pdf/forms.md", "pdf/scripts/check.py"], ["R1"]))
    report = run_change_gate(layout, {"R1": ref("requirement")}, skill_names(layout.input_skills) - (set()))
    assert report.changes[0].status == "accepted"
    assert report.accepted_files == ["pdf/forms.md", "pdf/scripts/check.py"]


def test_new_broken_reference_reverts_and_description_change_warns(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "forms.md").write_text("# Forms\nRun `scripts/missing.py`.\n", encoding="utf-8")
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]))
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set()))
    assert report.changes[0].status == "rejected"
    assert "scripts/missing.py does not exist" in report.changes[0].reasons[0]

    shutil.copy2(layout.input_skills / "pdf" / "forms.md", layout.edited_skills / "pdf" / "forms.md")
    (layout.edited_skills / "pdf" / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: Use when filling or checking PDF forms.\n---\n\nSee forms.md.\n", encoding="utf-8"
    )
    claim(layout.change_workspace, change("C1", ["pdf/SKILL.md"], ["L1"]))
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set()))
    assert report.changes[0].status == "accepted"
    assert [w.kind for w in report.warnings] == ["description_changed"]
    assert not report.warnings[0].actionable


def test_unsafe_script_reverts_every_change_of_the_skill(tmp_path):
    layout = _layout(tmp_path)
    accepted = {**ACCEPTED, "C1": ref(quote="curl -O https://x/y", case="comparison-1")}
    (layout.edited_skills / "pdf" / "scripts" / "fetch.sh").write_text("curl -O https://x/y\n", encoding="utf-8")
    (layout.edited_skills / "pdf" / "forms.md").write_text(
        "# Forms\nRun `python3 scripts/check.py form.pdf`, then `scripts/fetch.sh`.\n", encoding="utf-8"
    )
    claim(
        layout.change_workspace,
        change("C1", ["pdf/scripts/fetch.sh"], ["C1"]),
        change("C2", ["pdf/forms.md"]),
    )
    report = run_change_gate(layout, accepted, skill_names(layout.input_skills) - (set()))

    assert _status(report) == {"C1": "rejected", "C2": "rejected"}
    assert any("unsafe operation" in reason for reason in report.changes[0].reasons)


def test_baseline_problems_do_not_block_changes(tmp_path):
    layout = _layout(tmp_path)
    for root in (layout.input_skills, layout.edited_skills):
        (root / "pdf" / "SKILL.md").write_text("# PDF\nNo frontmatter here.\n", encoding="utf-8")
    (layout.edited_skills / "pdf" / "forms.md").write_text(
        "# Forms\nRun `python3 scripts/check.py`.\n", encoding="utf-8"
    )
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]))
    assert run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set())).changes[0].status == "accepted"


def test_changes_to_protected_skills_are_rejected(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "forms.md").write_text(
        "# Forms\n\nUse `python3 scripts/check.py`.\n", encoding="utf-8"
    )
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]))
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - ({"pdf"}))

    assert _status(report) == {"C1": "rejected"} and "skill pdf is read-only" in report.changes[0].reasons
    assert report.accepted_files == []
    assert "`python scripts" in (layout.candidate_skills / "pdf" / "forms.md").read_text(encoding="utf-8")


def test_description_repeating_trajectory_text_is_rejected(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: Check the form fields and write the result to report.json\n---\n\nSee forms.md.\n",
        encoding="utf-8",
    )
    claim(layout.change_workspace, change("C1", ["pdf/SKILL.md"], ["L1"]))
    report = run_change_gate(layout, ACCEPTED, skill_names(layout.input_skills) - (set()))
    assert report.changes[0].status == "rejected"
    assert "description repeats trajectory text" in report.changes[0].reasons[0]


def test_findings_with_an_issue_must_be_answered_directly_or_through_their_case(tmp_path):
    layout = _layout(tmp_path)
    (layout.edited_skills / "pdf" / "forms.md").write_text(
        "# Forms\nUse `python3 scripts/check.py`.\n", encoding="utf-8"
    )
    accepted = {
        "F1": ref(case="comparison-1"),
        "F2": ref(case="comparison-1"),
        "F3": ref(),
        "S1": ref("loading"),
        "O1": ref(None, None),
    }
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]))
    report = run_change_gate(layout, accepted, skill_names(layout.input_skills) - (set()))
    assert report.unaddressed_findings == ["F3", "S1"]
    assert "`F3`, `S1`" in open_items_feedback(report)

    changes = json.loads((layout.change_workspace / "changes.json").read_text(encoding="utf-8"))
    changes["dropped"] = _dropped("F3", "S1")
    (layout.change_workspace / "changes.json").write_text(json.dumps(changes), encoding="utf-8")
    report = run_change_gate(layout, accepted, skill_names(layout.input_skills) - (set()))
    assert report.unaddressed_findings == [] and open_items_feedback(report) == ""
