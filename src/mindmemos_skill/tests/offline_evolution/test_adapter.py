"""Verify complete collections, recovery and the real file gates around native sessions."""

import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mindmemos_skill.algos.evolve import SkillTrajectory
from mindmemos_skill.algos.evolve.offline_evolution import OfflineEvolutionConfig, OfflineSkillEvolver
from mindmemos_skill.algos.evolve.offline_evolution import engine as engine_module
from mindmemos_skill.algos.evolve.offline_evolution import evolver as adapter_module
from mindmemos_skill.algos.evolve.offline_evolution.config import EvolutionOptions
from mindmemos_skill.algos.evolve.offline_evolution.evolver import _read_candidate
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout
from mindmemos_skill.envs.typing import Reward, Task
from mindmemos_skill.infra.database import Database
from mindmemos_skill.skill import Skill, SkillContent

from .helpers import EVIDENCE_SCRIPT, FakeAgent, failing, make_inputs, write_change, write_evidence
from .test_collection_publication import skill


async def test_public_evidence_changes_invalidate_completed_adapter_cache(tmp_path, monkeypatch):
    """Reject cached candidates produced with a different S1 evidence export."""
    original = skill("context")
    traces = [
        SkillTrajectory(
            task=Task(id="task", instruction="Process the document."),
            skills={"context": original},
            trajectory=[{"role": "user", "content": "Process the document."}],
            status="failed",
        )
    ]
    public_trajectory = adapter_module._public_trajectory

    def old_export(record):
        payload = public_trajectory(record)
        payload.pop("status")
        return payload

    async def keep_inputs(self, *, skills, trajectories, work_dir):
        return SimpleNamespace(skills=skills)

    monkeypatch.setattr(adapter_module.OfflineEvolution, "evolve", keep_inputs)
    evolver = OfflineSkillEvolver(OfflineEvolutionConfig())
    with monkeypatch.context() as patch:
        patch.setattr(adapter_module, "_public_trajectory", old_export)
        await evolver.evolve(skills={"context": original}, trajectories=traces, work_dir=tmp_path / "algorithm")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        await evolver.evolve(skills={"context": original}, trajectories=traces, work_dir=tmp_path / "algorithm")


def test_nonstandard_existing_candidates_preserve_metadata_and_binary_resources(tmp_path):
    original = Skill(
        SkillContent(
            content="# Original\r\n", name="plain-source", description="Plain text workflow", standard_format=False
        )
    )
    directory = original.export(tmp_path)
    (directory / "SKILL.md").write_bytes(b"# Updated\r\n")
    (directory / "asset.bin").write_bytes(b"\x00\xff")
    candidate = _read_candidate(original, directory)
    assert candidate.content == "# Updated\r\n" and candidate.resources == {"asset.bin": b"\x00\xff"}
    assert candidate.source.name == original.source.name
    assert candidate.source.description == original.source.description
    assert not candidate.source.standard_format


def test_nonstandard_candidates_reject_resource_symlinks(tmp_path):
    original = Skill(
        SkillContent(
            content="# Original\n", name="plain-source", description="Plain text workflow", standard_format=False
        )
    )
    directory = original.export(tmp_path)
    outside = tmp_path / "private.bin"
    outside.write_bytes(b"outside")
    try:
        (directory / "linked.bin").symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are unavailable")
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        _read_candidate(original, directory)


@pytest.mark.parametrize("allow_create", [False, True])
@pytest.mark.parametrize("delete", [False, True])
async def test_adapter_enforces_scopes_and_preserves_complete_resources(tmp_path, monkeypatch, allow_create, delete):
    """Run the standalone algorithm without a database while enforcing mutation scopes."""
    monkeypatch.setattr(
        Database, "__init__", Mock(side_effect=AssertionError("The algorithm must not open a database"))
    )
    sources, _ = make_inputs(tmp_path)
    pdf = Skill.from_path(sources / "pdf")
    pdf = pdf.with_resources({**pdf.resources, "asset.bin": b"\x00\xff\r\n"})
    readonly = skill("readonly", "Original context", {"asset.bin": b"\xff\x00"})
    inputs = {"pdf-id": pdf, "readonly-id": readonly}
    before = {key: value.snapshot() for key, value in inputs.items()}

    def analyze(request):
        layout = RunLayout(root=request.cwd)
        script = EVIDENCE_SCRIPT.replace('["steps"]', '["trajectory"]')
        script = script.replace(
            "json.dump(output,",
            'output["findings"].append({"id": "U1", "path": "trajectories/trajectory-000001.json", "quote": "Process the PDF", "issue": "uncovered"})\njson.dump(output,',
        )
        write_evidence(layout.evidence_workspace, script=script)

    def change(request):
        layout = RunLayout(root=request.cwd)
        write_change(layout.change_workspace, ["F1"])
        claims = json.loads((layout.change_workspace / "changes.json").read_text())
        if delete:
            files = sorted(
                path.relative_to(layout.edited_skills).as_posix()
                for path in (layout.edited_skills / "pdf").rglob("*")
                if path.is_file()
            )
            shutil.rmtree(layout.edited_skills / "pdf")
            claims["changes"][0]["files"] = files
        context = layout.edited_skills / "readonly" / "SKILL.md"
        context.write_text(context.read_text() + "\nUnauthorized edit.\n")
        new = layout.edited_skills / "delta"
        new.mkdir()
        (new / "SKILL.md").write_text(skill("delta").content)
        for identity, files in (
            ("readonly-edit", ["readonly/SKILL.md"]),
            ("new-skill", ["delta/SKILL.md"]),
        ):
            claims["changes"].append(
                {
                    "id": identity,
                    "files": files,
                    "findings": ["U1"] if identity == "new-skill" else ["F1"],
                    "rationale": "Observed evidence.",
                }
            )
        (layout.change_workspace / "changes.json").write_text(json.dumps(claims))

    backend = FakeAgent([analyze, change])
    monkeypatch.setattr(engine_module, "run_agent", backend.run)
    traces = [
        SkillTrajectory(
            trajectory_id="original",
            task=Task(id="task", instruction="Process the PDF"),
            skills={"pdf-id": pdf},
            trajectory=[{"cmd": "python scripts/check.py", "out": "/bin/bash: python: command not found"}],
            reward=Reward(score=1, success=True, detail="PRIVATE_REFERENCE_ANSWER"),
            observed_score=1,
            feedback_detail="PRIVATE_VERIFIER_DETAIL",
        )
    ]
    evolver = OfflineSkillEvolver(OfflineEvolutionConfig(retries=0))
    permissions = EvolutionOptions(editable_skills=["pdf-id"], allow_create_skills=allow_create)
    result = await evolver.evolve(
        skills=inputs,
        trajectories=traces,
        work_dir=tmp_path / "algorithm",
        options=permissions,
    )
    assert result.skills["readonly-id"].snapshot() == before["readonly-id"]
    assert bool(result.created_skills) is allow_create
    if delete:
        assert result.skills["pdf-id"].snapshot() == before["pdf-id"]
    else:
        assert result.skills["pdf-id"].resources["asset.bin"] == b"\x00\xff\r\n"
        assert b"python3 scripts" in result.skills["pdf-id"].resources["forms.md"]
    raw = (tmp_path / "algorithm/source_trajectories/trajectory-000001.json").read_text()
    assert "PRIVATE_REFERENCE_ANSWER" not in raw and "PRIVATE_VERIFIER_DETAIL" not in raw
    assert "observed_score" not in raw and '"reward"' not in raw
    assert {key: value.snapshot() for key, value in inputs.items()} == before
    replay = await evolver.evolve(
        skills=inputs, trajectories=traces, work_dir=tmp_path / "algorithm", options=permissions
    )
    assert replay.snapshot() == result.snapshot()
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        await evolver.evolve(
            skills=inputs,
            trajectories=traces,
            work_dir=tmp_path / "algorithm",
            options=EvolutionOptions(editable_skills=[], allow_create_skills=allow_create),
        )
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        await evolver.evolve(
            skills={**inputs, "readonly-id": readonly.with_content(readonly.content + "\nChanged context.\n")},
            trajectories=traces,
            work_dir=tmp_path / "algorithm",
            options=permissions,
        )
    assert len(backend.requests) == 2


async def test_adapter_rejects_completed_cache_with_different_gate_settings(tmp_path, monkeypatch):
    """Changing collector settings invalidates an otherwise identical completed result."""
    sources, _ = make_inputs(tmp_path)
    pdf = Skill.from_path(sources / "pdf")

    def analyze(request):
        layout = RunLayout(root=request.cwd)
        write_evidence(layout.evidence_workspace, script=EVIDENCE_SCRIPT.replace('["steps"]', '["trajectory"]'))

    backend = FakeAgent([analyze, lambda request: write_change(RunLayout(root=request.cwd).change_workspace, ["F1"])])
    monkeypatch.setattr(engine_module, "run_agent", backend.run)
    algorithm = OfflineSkillEvolver(OfflineEvolutionConfig(retries=0))
    arguments = {
        "skills": {"pdf-id": pdf},
        "trajectories": [
            SkillTrajectory(
                task=Task(id="task", instruction="Process forms"),
                skills={"pdf-id": pdf},
                trajectory=[{"out": "/bin/bash: python: command not found", "cmd": "python scripts/check.py"}],
            )
        ],
        "work_dir": tmp_path / "algorithm",
    }
    await algorithm.evolve(**arguments)
    assert (arguments["work_dir"] / "result.json").is_file()
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        changed = OfflineSkillEvolver(OfflineEvolutionConfig(retries=0, script_timeout_s=901))
        await changed.evolve(**arguments)
    assert len(backend.requests) == 2


@pytest.mark.parametrize("interrupted_export", ["skills", "trajectories"])
async def test_interrupted_exports_are_rebuilt_completely_before_the_engine(tmp_path, monkeypatch, interrupted_export):
    """A partially written source directory cannot reduce the engine's input collection."""
    sources, _ = make_inputs(tmp_path)
    pdf = Skill.from_path(sources / "pdf")
    readonly = skill("readonly", "Original context", {"asset.bin": b"\x00\xff"})
    inputs = {"pdf-id": pdf, "readonly-id": readonly}
    traces = [
        SkillTrajectory(
            trajectory_id=f"trace-{index}",
            task=Task(id=f"task-{index}", instruction="Process the PDF"),
            skills=inputs,
            trajectory=[{"cmd": "python scripts/check.py", "out": "/bin/bash: python: command not found"}],
        )
        for index in range(2)
    ]
    work = tmp_path / "algorithm"
    algorithm = OfflineSkillEvolver(OfflineEvolutionConfig(retries=0))

    def analyze(request):
        layout = RunLayout(root=request.cwd)
        write_evidence(layout.evidence_workspace, script=EVIDENCE_SCRIPT.replace('["steps"]', '["trajectory"]'))

    backend = FakeAgent(
        [analyze, lambda request: write_change(RunLayout(root=request.cwd).change_workspace, ["F1", "F2"])]
    )
    monkeypatch.setattr(engine_module, "run_agent", backend.run)
    export_skill = Skill.export
    write_json = adapter_module.write_json

    def interrupted_skill(self, directory, **options):
        if self.source.name == "readonly":
            raise OSError("source export interrupted")
        return export_skill(self, directory, **options)

    def interrupted_trajectory(path, value):
        if path.name == "trajectory-000002.json":
            raise OSError("source export interrupted")
        return write_json(path, value)

    with monkeypatch.context() as patch:
        if interrupted_export == "skills":
            patch.setattr(Skill, "export", interrupted_skill)
        else:
            patch.setattr(adapter_module, "write_json", interrupted_trajectory)
        for _ in range(2):
            with pytest.raises(OSError, match="source export interrupted"):
                await algorithm.evolve(skills=inputs, trajectories=traces, work_dir=work)
            assert not backend.requests
            assert not (work / "engine").exists() and not (work / "result.json").exists()

    result = await algorithm.evolve(skills=inputs, trajectories=traces, work_dir=work)
    report = json.loads((work / "engine/evidence/report.json").read_text())
    assert report["status"] == "accepted" and report["trajectory_files"] == report["processed"] == 2
    assert set(result.skills) == set(inputs)
    assert result.skills["readonly-id"].snapshot() == readonly.snapshot()
    assert len(list((work / "source_trajectories").glob("*.json"))) == 2
    assert len(backend.requests) == 2


async def test_reexporting_inputs_preserves_an_interrupted_engine_session(tmp_path, monkeypatch):
    """Rebuilding adapter exports must retain the engine checkpoint and native session identity."""
    sources, _ = make_inputs(tmp_path)
    pdf = Skill.from_path(sources / "pdf")
    traces = [
        SkillTrajectory(
            task=Task(id="task", instruction="Process the PDF"),
            skills={"pdf-id": pdf},
            trajectory=[{"cmd": "python scripts/check.py", "out": "/bin/bash: python: command not found"}],
        )
    ]

    def analyze(request):
        layout = RunLayout(root=request.cwd)
        write_evidence(layout.evidence_workspace, script=EVIDENCE_SCRIPT.replace('["steps"]', '["trajectory"]'))

    backend = FakeAgent(
        [
            failing("preserved-engine-session"),
            analyze,
            lambda request: write_change(RunLayout(root=request.cwd).change_workspace, ["F1"]),
        ]
    )
    monkeypatch.setattr(engine_module, "run_agent", backend.run)
    algorithm = OfflineSkillEvolver(OfflineEvolutionConfig(retries=0))
    work = tmp_path / "algorithm"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        await algorithm.evolve(skills={"pdf-id": pdf}, trajectories=traces, work_dir=work)
    result = await algorithm.evolve(skills={"pdf-id": pdf}, trajectories=traces, work_dir=work)
    assert backend.requests[1].session_id == "preserved-engine-session"
    assert "python3 scripts" in result.skills["pdf-id"].resources["forms.md"].decode()
    assert len(backend.requests) == 3
