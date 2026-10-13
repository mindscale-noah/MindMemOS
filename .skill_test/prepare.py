"""Create local Skill fixtures for the UI functional testing guide."""

from datetime import datetime
from pathlib import Path
from textwrap import dedent

# Paths are relative to this script; values are complete UTF-8 file contents.
FIXTURE_FILES: dict[str, str] = {
    "skills/qa-meeting-notes/SKILL.md": """
        ---
        name: qa-meeting-notes
        description: Summarize meeting notes and extract action items.
        ---
        # Meeting notes
        1. Read resources/meeting.md.
        2. Write a summary using resources/template.md.
        3. Check the result against references/checklist.md.
        """,
    "skills/qa-meeting-notes/resources/meeting.md": """
        # Release planning meeting
        Alice will prepare the release notes by Friday.
        Bob will verify the installation guide by Thursday.
        The team agreed to release next Monday.
        """,
    "skills/qa-meeting-notes/resources/template.md": """
        # Summary
        Describe the decisions made during the meeting.

        # Action items
        | Task | Owner |
        | --- | --- |
        """,
    "skills/qa-meeting-notes/references/checklist.md": """
        # Checklist
        - Include a summary.
        - Include an owner for each action item.
        """,
    "skills/qa-expense-review/SKILL.md": """
        ---
        name: qa-expense-review
        description: Review expense records against the reimbursement policy.
        ---
        # Expense review
        1. Read resources/expenses.csv.
        2. Apply the limits in references/policy.md.
        3. List approved and rejected expenses with reasons.
        """,
    "skills/qa-expense-review/resources/expenses.csv": """
        employee,category,amount
        Alice,meal,80
        Bob,taxi,120
        Carol,meal,200
        """,
    "skills/qa-expense-review/references/policy.md": """
        # Reimbursement policy
        - Meal limit: 100 per expense.
        - Taxi limit: 150 per expense.
        """,
    "skills/qa-release-check/SKILL.md": """
        ---
        name: qa-release-check
        description: Check whether a release is ready for testing.
        ---
        # Release check
        1. Read resources/release.json.
        2. Check every item in references/checklist.md.
        3. Report missing requirements.
        """,
    "skills/qa-release-check/resources/release.json": """
        {"version": "1.0.0", "installation_checked": true, "smoke_test_passed": false}
        """,
    "skills/qa-release-check/references/checklist.md": """
        # Release checklist
        - Installation must succeed.
        - Smoke tests must pass.
        """,
    "exports/qa-meeting-notes/obsolete.md": """
        # Obsolete file
        A confirmed export should remove this file when replacing the folder.
        """,
}


def prepare_skill_fixtures() -> None:
    """Create missing fixture files and print absolute paths for UI input.

    Existing files are preserved so reruns do not discard manual test edits.
    """
    fixture_root = Path(__file__).resolve().parent
    created = 0
    for relative_path, content in FIXTURE_FILES.items():
        target = fixture_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(dedent(content).lstrip(), encoding="utf-8")
            created += 1
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Created {created} files; existing files preserved.")
    for skill_name in ("qa-meeting-notes", "qa-expense-review", "qa-release-check"):
        print(f"{skill_name}: {fixture_root / 'skills' / skill_name}")
    print(f"Export destination: {fixture_root / 'exports' / 'qa-meeting-notes'}")
    print("Local files only: register each Skill in the UI to upload it to the service.")


if __name__ == "__main__":
    prepare_skill_fixtures()
