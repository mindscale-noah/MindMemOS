"""Shared fixtures: a tiny skill collection, trajectories and a scripted fake agent."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from mindmemos_skill.agents.typing import AgentResult
from mindmemos_skill.algos.evolve.offline_evolution import engine as engine_module
from mindmemos_skill.algos.evolve.offline_evolution.agents.typing import SessionOptions
from mindmemos_skill.algos.evolve.offline_evolution.config import OfflineEvolutionConfig
from mindmemos_skill.algos.evolve.offline_evolution.engine import OfflineEvolution
from mindmemos_skill.algos.evolve.offline_evolution.typing import FindingRef, Issue
from pydantic import ConfigDict


class AgentRequest(SessionOptions):
    """Captured invocation containing prompt and workspace alongside session scopes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Keep captured calls immutable."""

    prompt: str
    """Actual stage instruction supplied to the shared Agent."""

    cwd: Path
    """Actual child working directory."""


def build_engine(agent, config=None, options=None):
    """Replace only native sessions while exercising the complete file engine."""
    return _ScriptedEvolution(agent, config or OfflineEvolutionConfig(), options)


class _ScriptedEvolution(OfflineEvolution):
    """Keep the native-session substitute active across asynchronous stage execution."""

    def __init__(self, agent, config, options):
        super().__init__(config, options)
        self.backend = agent

    async def evolve(self, **inputs):
        with patch.object(engine_module, "run_agent", side_effect=self.backend.run):
            return await super().evolve(**inputs)

    async def resume(self, *, work_dir):
        with patch.object(engine_module, "run_agent", side_effect=self.backend.run):
            return await super().resume(work_dir=work_dir)


FORMS = "# Forms\n\nRun `python scripts/check.py <file.pdf>` from the skill directory.\n"
SKILL = "---\nname: pdf\ndescription: Fill and inspect PDF forms.\n---\n\nSee forms.md.\n"

EVIDENCE_SCRIPT = """
import argparse, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--inputs"); parser.add_argument("--out")
args = parser.parse_args()
root = pathlib.Path(args.inputs)
processed, findings, records = [], [], {}
for path in sorted((root / "trajectories").rglob("*.json")):
    rel = path.relative_to(root).as_posix()
    processed.append(rel)
    records[rel] = json.loads(path.read_text())
    for step in records[rel]["steps"]:
        if "command not found" in step["out"]:
            findings.append({"id": f"F{len(findings) + 1}", "path": rel, "quote": step["out"], "value": step["cmd"],
                             "issue": "content", "skill": "pdf"})
output = {"processed": processed, "failed": [], "summary": {"failures": len(findings)}, "findings": findings}
json.dump(output, open(args.out, "w"))
"""
"""Shared collector producing located observations without mandatory audit views."""

INSTRUCTION = "Check the form fields and write the result to report.json"
"""Task instruction recorded in every fixture trajectory."""

ISSUE_FINDINGS = ["F1", "F2"]
"""Accepted fixture findings that call for an edit."""


def make_inputs(tmp: Path) -> tuple[Path, Path]:
    """Create one pdf skill with a script and two runs that hit `python: command not found`."""
    skills, trajectories = tmp / "src-skills", tmp / "src-trajectories"
    (skills / "pdf" / "scripts").mkdir(parents=True)
    (skills / "pdf" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (skills / "pdf" / "forms.md").write_text(FORMS, encoding="utf-8")
    (skills / "pdf" / "scripts" / "check.py").write_text("print('ok')\n", encoding="utf-8")
    (trajectories / "task-a").mkdir(parents=True)
    for name in ("run-1", "run-2"):
        steps = [
            {"cmd": "python scripts/check.py form.pdf", "out": "/bin/bash: python: command not found"},
            {"cmd": "python3 scripts/check.py form.pdf", "out": "ok"},
        ]
        run = {"instruction": INSTRUCTION, "steps": steps}
        (trajectories / "task-a" / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")
    return skills, trajectories


def write_evidence(workspace: Path, script: str = EVIDENCE_SCRIPT) -> None:
    """Write the collection manifest and its single entry point."""
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "collect.py").write_text(script, encoding="utf-8")
    manifest = {"collector": "scripts/collect.py"}
    (workspace / "collection.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_change(workspace: Path, findings: list[str], dropped: list[str] | None = None) -> None:
    """Edit forms.md to use python3 and claim it, as the change agent would.

    Fixture findings with an issue that the change does not cite are listed under `dropped` unless
    `dropped` is given.
    """
    forms = workspace / "skills" / "pdf" / "forms.md"
    forms.write_text(forms.read_text(encoding="utf-8").replace("`python scripts", "`python3 scripts"), encoding="utf-8")
    changes = {
        "changes": [
            {
                "id": "C1",
                "files": ["pdf/forms.md"],
                "findings": findings,
                "rationale": "Documented interpreter is missing in the environment.",
            }
        ],
    }
    rest = [f for f in ISSUE_FINDINGS if f not in findings] if dropped is None else dropped
    if rest:
        changes["dropped"] = [{"findings": rest, "reason": "same failure as the cited finding"}]
    (workspace / "changes.json").write_text(json.dumps(changes), encoding="utf-8")


class FakeAgent:
    """Runs scripted steps in order; each step writes files the way a real session would."""

    def __init__(self, steps: list[Callable[[AgentRequest], None]]) -> None:
        """Queue the steps of successive sessions."""
        self.steps = list(steps)
        self.requests: list[AgentRequest] = []

    async def run(self, config, *, user, workspace, session) -> AgentResult:
        """Record the request and execute the next step."""
        assert config.class_name == "claude"
        request = AgentRequest(prompt=user, cwd=workspace, **session.model_dump())
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("unexpected agent session")
        self.steps.pop(0)(request)
        return AgentResult(
            output="done",
            trajectory=[{"session_id": f"session-{len(self.requests)}"}],
            usage={"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 5},
        )


def failing(session_id: str) -> Callable[[AgentRequest], None]:
    """A step that fails like an interrupted session."""

    def step(request: AgentRequest) -> None:
        request.log_path.write_text(json.dumps({"type": "system", "session_id": session_id}) + "\n")
        raise RuntimeError("simulated interruption")

    return step


def ref(
    issue: Issue | None = "content",
    skill: str | None = "pdf",
    quote: str = "q",
    path: str = "trajectories/task-a/run-1.json",
    **extra,
) -> FindingRef:
    """An accepted finding as the change gate receives it from the run state."""
    return FindingRef(path=path, quote=quote, issue=issue, skill=skill, **extra)


def claim(workspace: Path, *changes: dict, dropped: list[dict] | None = None) -> None:
    """Write changes.json with the given changes."""
    body = {"changes": list(changes), "dropped": dropped or []}
    (workspace / "changes.json").write_text(json.dumps(body), encoding="utf-8")


def change(identity: str, files: list[str], findings: list[str] | None = None, **extra) -> dict:
    """One claimed change citing `F1` unless other findings are given."""
    return {"id": identity, "files": files, "findings": findings or ["F1"], "rationale": "r", **extra}
