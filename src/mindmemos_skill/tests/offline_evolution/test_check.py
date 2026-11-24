"""Preflight invokes the real gates on copies and reports feedback without changing the run."""

import json
import shutil

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.change_gate import run_change_gate
from mindmemos_skill.algos.evolve.offline_evolution.check import main
from mindmemos_skill.algos.evolve.offline_evolution.config import (
    EvidenceGateConfig,
    EvolutionOptions,
    OfflineEvolutionConfig,
)
from mindmemos_skill.algos.evolve.offline_evolution.layout import write_text
from mindmemos_skill.algos.evolve.offline_evolution.typing import PreflightContext, RunLayout, RunState

from .helpers import change, claim, make_inputs, ref, write_evidence


def _layout(tmp_path) -> RunLayout:
    skills, trajectories = make_inputs(tmp_path)
    layout = RunLayout(root=tmp_path / "run")
    layout.root.mkdir()
    layout.snapshot(skills, trajectories)
    state = RunState(config=OfflineEvolutionConfig(), inputs_digest=layout.inputs_digest())
    state.evidence_gate.accepted_findings = {"F1": ref()}
    write_text(layout.state_path, state.model_dump_json(indent=1))
    _preview(layout, state)
    return layout


def _preview(layout, state):
    """Seed the public preview inputs without exposing connection settings."""
    context = PreflightContext(
        gate=EvidenceGateConfig(),
        options=state.options,
        inputs_digest=state.inputs_digest,
        accepted_findings=state.evidence_gate.accepted_findings,
    )
    write_text(layout.preflight_path, context.model_dump_json())


def _files(layout: RunLayout) -> dict[str, bytes]:
    return {p.relative_to(layout.root).as_posix(): p.read_bytes() for p in layout.root.rglob("*") if p.is_file()}


def test_evidence_preflight_runs_on_copies_and_returns_rejections(tmp_path, capsys):
    layout = _layout(tmp_path)
    write_evidence(layout.evidence_workspace)
    before = _files(layout)
    assert main(["evidence", "--run", str(layout.root)]) == 0
    assert json.loads(capsys.readouterr().out)["feedback"] == ""
    assert _files(layout) == before

    manifest = layout.evidence_workspace / "collection.json"
    manifest.write_text("invalid")
    before = _files(layout)
    assert main(["evidence", "--run", str(layout.root)]) == 1
    assert "collection.json" in json.loads(capsys.readouterr().out)["feedback"]
    assert _files(layout) == before


def test_change_preflight_preserves_existing_reports_and_candidate(tmp_path, capsys):
    layout = _layout(tmp_path)
    shutil.copytree(layout.input_skills, layout.edited_skills)
    forms = layout.edited_skills / "pdf" / "forms.md"
    forms.write_text(forms.read_text().replace("python scripts", "python3 scripts"))
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]))
    write_text(layout.candidate / "gate.json", '{"previous": true}')
    write_text(layout.candidate_skills / "keep.txt", "previous candidate")
    before = _files(layout)
    assert main(["change", "--run", str(layout.root)]) == 0
    assert json.loads(capsys.readouterr().out)["accepted_files"] == ["pdf/forms.md"]
    assert _files(layout) == before

    claim(layout.change_workspace, change("C1", ["pdf/forms.md"], ["d:missing"]))
    before = _files(layout)
    assert main(["change", "--run", str(layout.root)]) == 1
    assert "not accepted" in json.loads(capsys.readouterr().out)["feedback"]
    assert _files(layout) == before


def test_preflight_rejects_modified_inputs(tmp_path, capsys):
    layout = _layout(tmp_path)
    (layout.input_skills / "pdf" / "forms.md").write_text("changed")
    assert main(["evidence", "--run", str(layout.root)]) == 2
    assert "inputs/ changed" in json.loads(capsys.readouterr().out)["feedback"]


def test_preflight_honors_configured_collector_interpreter(tmp_path, capsys):
    """An unavailable configured interpreter must fail instead of silently using another Python."""
    layout = _layout(tmp_path)
    write_evidence(layout.evidence_workspace)
    context = json.loads(layout.preflight_path.read_text())
    context["gate"]["python"] = str(tmp_path / "missing-collector-python")
    write_text(layout.preflight_path, json.dumps(context))
    before = _files(layout)
    assert main(["evidence", "--run", str(layout.root)]) == 2
    assert "missing-collector-python" in json.loads(capsys.readouterr().out)["feedback"]
    assert _files(layout) == before


@pytest.mark.parametrize("editable", [[], ["pdf"]])
@pytest.mark.parametrize("allow_create", [False, True])
def test_preflight_and_formal_gate_enforce_the_same_permissions(tmp_path, capsys, editable, allow_create):
    """Preview each permission combination without mutating any formal artifacts."""
    layout = _layout(tmp_path)
    state = RunState.model_validate_json(layout.state_path.read_text())
    state.options = EvolutionOptions(editable_skills=editable, allow_create_skills=allow_create)
    state.evidence_gate.accepted_findings["U1"] = ref("uncovered", None)
    write_text(layout.state_path, state.model_dump_json())
    _preview(layout, state)
    shutil.copytree(layout.input_skills, layout.edited_skills)
    forms = layout.edited_skills / "pdf/forms.md"
    forms.write_text(forms.read_text().replace("python scripts", "python3 scripts"))
    notes = layout.edited_skills / "notes"
    notes.mkdir()
    (notes / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Record reusable observations.\n---\n\nRecord observations.\n"
    )
    claim(layout.change_workspace, change("C1", ["pdf/forms.md"]), change("C2", ["notes/SKILL.md"], ["U1"]))
    before = _files(layout)
    assert main(["change", "--run", str(layout.root)]) == (0 if editable and allow_create else 1)
    preview = json.loads(capsys.readouterr().out)
    assert _files(layout) == before
    report = run_change_gate(layout, state.evidence_gate.accepted_findings, set(editable), allow_create)
    assert preview["accepted_files"] == report.accepted_files
    assert ("pdf/forms.md" in report.accepted_files) is bool(editable)
    assert ("notes/SKILL.md" in report.accepted_files) is allow_create


def test_preflight_requires_public_gate_settings(tmp_path, capsys):
    """Reject incomplete previews without reading a private-state fallback."""
    layout = _layout(tmp_path)
    context = json.loads(layout.preflight_path.read_text())
    context.pop("gate")
    write_text(layout.preflight_path, json.dumps(context))
    before = _files(layout)
    assert main(["evidence", "--run", str(layout.root)]) == 2
    assert "gate" in json.loads(capsys.readouterr().out)["feedback"]
    assert _files(layout) == before


def test_preflight_requires_no_access_to_private_host_state(tmp_path, capsys):
    """Preview still works when every Host file is unavailable to the agent."""
    layout = _layout(tmp_path)
    write_evidence(layout.evidence_workspace)
    shutil.rmtree(layout.host)
    before = _files(layout)
    assert main(["evidence", "--run", str(layout.root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "accepted"
    assert _files(layout) == before and not layout.host.exists()
