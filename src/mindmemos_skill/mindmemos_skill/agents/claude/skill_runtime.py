"""Claude-native filesystem Skill runtime."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager

from ...typing import Skill, SkillBinding, SkillInjectionMode, Trajectory
from ..skill_runtime import SkillInjection, SkillRuntime
from .support import extract_used_skill_names


class ClaudeSkillRuntime(SkillRuntime):
    """Materialize Skills for Claude and interpret Claude-native load evidence."""

    supported_modes = frozenset({SkillInjectionMode.FILESYSTEM})

    @contextmanager
    def inject(self, skills: list[Skill]) -> Iterator[SkillInjection]:
        if not skills:
            yield SkillInjection(mode=self.mode)
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        workspace = tempfile.mkdtemp(prefix=f"mindmemos_skills_{timestamp}_")
        try:
            skills_dir = os.path.join(workspace, ".claude", "skills")
            os.makedirs(skills_dir, exist_ok=True)

            for skill in skills:
                if "SKILL.md" not in skill.blob:
                    raise ValueError(f"Claude agents require {skill.name!r} to contain SKILL.md in blob")
                duplicate_paths = skill.blob.keys() & skill.resources.keys()
                if duplicate_paths:
                    duplicates = ", ".join(sorted(duplicate_paths))
                    raise ValueError(f"Claude agents cannot materialize overlapping Skill paths: {duplicates}")
                safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in skill.name)
                skill_dir = os.path.join(skills_dir, safe_name)
                os.makedirs(skill_dir, exist_ok=True)
                abs_workspace = os.path.abspath(workspace)
                for rel_path, content in {**skill.blob, **skill.resources}.items():
                    abs_path = os.path.abspath(os.path.join(skill_dir, rel_path))
                    if not abs_path.startswith(abs_workspace + os.sep):
                        raise ValueError(f"Skill file path escapes its workspace: {rel_path!r}")
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as file:
                        file.write(content)

            yield SkillInjection(
                mode=self.mode,
                skill_names={skill.name for skill in skills},
                workspace=workspace,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def bind(self, trajectory: Trajectory) -> list[SkillBinding]:
        loaded_names = extract_used_skill_names(trajectory.events)
        return self._build_bindings(trajectory, loaded_names)


__all__ = ["ClaudeSkillRuntime"]
