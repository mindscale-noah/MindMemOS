"""Stage prompts stay general: they render, their JSON examples parse, and they name nothing from the datasets."""

import json
import re
import string
from pathlib import Path

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.prompts import PROMPT_DIR, render_prompt

DATA = Path(__file__).resolve().parents[2] / "data"
"""Trajectory and skill datasets the method is developed and evaluated on."""


def _dataset_names() -> set[str]:
    """Skill names and task names (directories holding a skills/ collection) found in the datasets."""
    names = set()
    for main in DATA.rglob("SKILL.md"):
        names.add(main.parent.name)
        if main.parent.parent.name == "skills":
            names.add(main.parent.parent.parent.name)
    return names - {"skills", DATA.name}


def test_prompts_render_with_valid_json_examples():
    for path in PROMPT_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        fields = {field for _, field, _, _ in string.Formatter().parse(text) if field}
        rendered = render_prompt(path.stem, **{field: f"<{field}>" for field in fields})
        for block in re.findall(r"```json\n(.*?)```", rendered, re.S):
            json.loads(re.sub(r"<[a-z_]+>", "0", block))


def test_prompts_name_no_dataset_task_or_skill():
    names = _dataset_names()
    if not names:
        pytest.skip("datasets not present")
    for path in PROMPT_DIR.rglob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        found = sorted(n for n in names if re.search(rf"(?<![\w-]){re.escape(n.lower())}(?![\w-])", text))
        assert not found, f"{path.name} names dataset tasks or skills: {found}"
