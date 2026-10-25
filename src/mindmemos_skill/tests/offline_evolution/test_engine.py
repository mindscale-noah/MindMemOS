"""The evolver runs both agent stages, gates their output and publishes only verified edits."""

import json
from pathlib import Path

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.config import EvolutionOptions, OfflineEvolutionConfig
from mindmemos_skill.algos.evolve.offline_evolution.engine import read_state
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout

from .helpers import (
    EVIDENCE_SCRIPT,
    FakeAgent,
    build_engine,
    failing,
    make_inputs,
    write_change,
    write_evidence,
)

BROKEN_SCRIPT = EVIDENCE_SCRIPT.replace(
    "json.dump(output,",
    'output = {"processed": ["trajectories/task-a/run-1.json"], "failed": [], '
    '"findings": [{"id": "F1", "path": "trajectories/task-a/run-1.json", "quote": "invented text"}]}\n'
    "json.dump(output,",
)


async def _evolve(tmp_path, agent: FakeAgent, config: OfflineEvolutionConfig | None = None, options=None):
    skills, trajectories = make_inputs(tmp_path)
    evolver = build_engine(agent, config=config or OfflineEvolutionConfig(), options=options)
    return await evolver.evolve(skills=skills, trajectories=trajectories, work_dir=tmp_path / "run")


async def test_full_run_publishes_verified_change(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F1", "F2"]),
        ]
    )
    result = await _evolve(tmp_path, agent)

    assert result.outcome == "updated" and result.skills == layout.output_skills
    assert result.state.evidence_gate.accepted is True
    assert result.state.change_gate.accepted == ["C1"]
    assert "`python3 scripts/check.py" in (result.skills / "pdf" / "forms.md").read_text(encoding="utf-8")
    assert "`python scripts" in (layout.input_skills / "pdf" / "forms.md").read_text(encoding="utf-8")
    assert list(result.state.evidence_gate.accepted_findings) == [
        "F1",
        "F2",
    ]
    evidence_request, change_request = agent.requests
    assert all(r.deny_paths == (layout.host,) and r.config_dir.parent == layout.host for r in agent.requests)
    assert layout.change_workspace in change_request.write_dirs
    assert layout.state_path.parent == layout.host and (layout.logs / "evidence-1.jsonl").parent.parent == layout.host
    assert "Comparison signals" in evidence_request.prompt and "How to write" in change_request.prompt
    assert "Read-only skills: (none)" in change_request.prompt and change_request.readonly_dirs == ()


async def test_full_run_allows_cited_description_and_new_code_without_loading_or_worked_labels(tmp_path):
    layout = RunLayout(root=tmp_path / "run")

    def adapt(_):
        write_change(layout.change_workspace, ["F1", "F2"])
        main = layout.edited_skills / "pdf/SKILL.md"
        main.write_text(main.read_text().replace("Fill and inspect PDF forms.", "Fill and validate form fields."))
        (layout.edited_skills / "pdf/scripts/check.py").write_text(
            "def inspect_fields(fields):\n    return [name for name, value in fields.items() if value is None]\n"
        )
        manifest = layout.change_workspace / "changes.json"
        changes = json.loads(manifest.read_text())
        changes["changes"][0]["files"] += ["pdf/SKILL.md", "pdf/scripts/check.py"]
        manifest.write_text(json.dumps(changes))

    agent = FakeAgent([lambda _: write_evidence(layout.evidence_workspace), adapt])
    result = await _evolve(tmp_path, agent)
    assert result.outcome == "updated" and len(agent.requests) == 2
    assert result.state.change_gate.accepted_files == ["pdf/SKILL.md", "pdf/forms.md", "pdf/scripts/check.py"]
    assert all(
        finding.issue == "content" and finding.case is None
        for finding in result.state.evidence_gate.accepted_findings.values()
    )


async def test_protected_skill_is_read_only_for_the_change_stage(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F1"]),
            lambda r: None,
        ]
    )
    result = await _evolve(tmp_path, agent, options=EvolutionOptions(editable_skills=[]))

    change_request, feedback = agent.requests[1:]
    assert change_request.readonly_dirs == (layout.edited_skills / "pdf",)
    assert "Read-only skills: pdf" in change_request.prompt
    assert "C1: skill pdf is read-only" in feedback.prompt and feedback.session_id == "session-2"
    assert result.state.change_gate.rejected == ["C1"] and result.outcome == "unchanged"


async def test_unknown_protected_skill_stops_before_the_run(tmp_path):
    with pytest.raises(ValueError, match="outside the input collection"):
        await _evolve(tmp_path, FakeAgent([]), options=EvolutionOptions(editable_skills=["missing"]))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("artifact", ["report", "preflight"])
async def test_change_gate_ignores_findings_added_to_public_artifacts(tmp_path, artifact):
    layout = RunLayout(root=tmp_path / "run")

    def forge_and_cite(_) -> None:
        path = layout.evidence / "report.json" if artifact == "report" else layout.preflight_path
        content = json.loads(path.read_text(encoding="utf-8"))
        if artifact == "report":
            content["findings"].append(dict(content["findings"][0], id="F99"))
        else:
            content["accepted_findings"]["F99"] = content["accepted_findings"]["F1"]
        path.write_text(json.dumps(content), encoding="utf-8")
        write_change(layout.change_workspace, ["F99"])

    agent = FakeAgent([lambda r: write_evidence(layout.evidence_workspace), forge_and_cite, lambda r: None])
    result = await _evolve(tmp_path, agent)

    assert "cites findings that are not accepted" in agent.requests[2].prompt
    assert result.state.change_gate.rejected == ["C1"] and result.outcome == "unchanged"


async def test_rejected_change_fixed_after_feedback_is_accepted(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F99"]),
            lambda r: write_change(layout.change_workspace, ["F1"]),
        ]
    )
    result = await _evolve(tmp_path, agent)

    record = result.state.change_gate
    assert record.feedback_rounds == 1 and record.accepted == ["C1"] and result.outcome == "updated"


async def test_rejected_evidence_is_returned_once_then_accepted(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace, script=BROKEN_SCRIPT),
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F1"]),
        ]
    )
    result = await _evolve(tmp_path, agent)

    first = json.loads((layout.evidence / "outputs" / "attempt-1.json").read_text())
    assert first["findings"][0]["quote"] == "invented text"
    retry = agent.requests[1]
    assert retry.session_id == "session-1"
    assert "neither processed nor reported" in retry.prompt and "quote not found" in retry.prompt
    assert result.state.evidence_gate.attempts == 2 and result.outcome == "updated"
    assert all((layout.evidence / "reports" / f"attempt-{n}.json").is_file() for n in (1, 2))


async def test_no_accepted_evidence_skips_change_and_keeps_skills(tmp_path):
    layout = RunLayout(root=tmp_path / "run")

    def invalid(_):
        write_evidence(layout.evidence_workspace)
        (layout.evidence_workspace / "collection.json").write_text("invalid")

    agent = FakeAgent([invalid] * 2)
    result = await _evolve(tmp_path, agent, OfflineEvolutionConfig(retries=1))

    assert result.state.evidence_gate.accepted is False
    assert result.state.change.skipped == "no accepted finding calls for an edit"
    assert result.outcome == "unchanged" and result.skills == layout.output_skills
    copied = sorted(p.relative_to(result.skills) for p in result.skills.rglob("*"))
    assert copied == sorted(p.relative_to(layout.input_skills) for p in layout.input_skills.rglob("*"))
    assert len(agent.requests) == 2


async def test_interrupted_session_resumes_by_id(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    with pytest.raises(RuntimeError):
        await _evolve(tmp_path, FakeAgent([failing("abc")]))
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F1"]),
        ]
    )
    result = await build_engine(agent, config=OfflineEvolutionConfig()).resume(work_dir=layout.root)

    assert agent.requests[0].session_id == "abc"
    assert "previous session stopped" in agent.requests[0].prompt
    assert result.outcome == "updated"


@pytest.mark.parametrize("stage", ["evidence", "change"])
@pytest.mark.parametrize("repaired_before_failure", [False, True])
@pytest.mark.parametrize("recorded_error", [False, True])
async def test_interrupted_feedback_resumes_before_gating_without_spending_another_retry(
    tmp_path, stage, repaired_before_failure, recorded_error
):
    layout = RunLayout(root=tmp_path / "run")
    config = OfflineEvolutionConfig(retries=1)

    def repair(_):
        if stage == "evidence":
            write_evidence(layout.evidence_workspace)
        else:
            write_change(layout.change_workspace, ["F1", "F2"])

    def interrupt(request):
        saved = read_state(layout.root)
        assert saved.outcome is None and not layout.output_skills.exists()
        pending = getattr(saved, stage).pending_feedback
        assert pending and pending in request.prompt
        assert saved.evidence_gate.attempts == 1
        assert saved.change_gate.feedback_rounds == (1 if stage == "change" else 0)
        if repaired_before_failure:
            repair(request)
        if recorded_error:
            request.log_path.write_text(json.dumps({"session_id": "feedback-session"}) + "\n")
            raise RuntimeError("feedback interrupted")
        # A process interruption can leave queued feedback without a recorded RuntimeError.
        raise RuntimeError("feedback interrupted")

    steps = [lambda _: write_evidence(layout.evidence_workspace, script=BROKEN_SCRIPT), interrupt]
    if stage == "change":
        steps = [
            lambda _: write_evidence(layout.evidence_workspace),
            lambda _: write_change(layout.change_workspace, ["F99"]),
            interrupt,
        ]
    first_agent = FakeAgent(steps)
    with pytest.raises(RuntimeError, match="feedback interrupted"):
        await _evolve(tmp_path, first_agent, config)

    # Even a valid partial edit must not let the gate skip the interrupted session.
    again = FakeAgent([failing("feedback-session-again")])
    with pytest.raises(RuntimeError, match="simulated interruption"):
        await build_engine(again, config=config).resume(work_dir=layout.root)
    expected_session = "feedback-session" if recorded_error else first_agent.requests[-1].session_id
    assert again.requests[0].session_id == expected_session
    assert again.requests[0].prompt == first_agent.requests[-1].prompt
    saved = read_state(layout.root)
    assert saved.evidence_gate.attempts == 1
    assert saved.change_gate.feedback_rounds == (1 if stage == "change" else 0)
    assert saved.outcome is None and not layout.output_skills.exists()

    remaining = [repair]
    if stage == "evidence":
        remaining.append(lambda _: write_change(layout.change_workspace, ["F1", "F2"]))
    resumed = FakeAgent(remaining)
    result = await build_engine(resumed, config=config).resume(work_dir=layout.root)

    assert resumed.requests[0].session_id == "feedback-session-again"
    assert resumed.requests[0].prompt == first_agent.requests[-1].prompt
    assert len(resumed.requests) == len(remaining)
    assert result.state.evidence_gate.attempts == (2 if stage == "evidence" else 1)
    assert result.state.change_gate.feedback_rounds == (1 if stage == "change" else 0)
    assert result.state.evidence.error is None and result.state.change.error is None
    assert result.state.evidence.pending_feedback is None and result.state.change.pending_feedback is None
    assert result.outcome == "updated" and result.state.completed[-1] == "publish"


@pytest.mark.parametrize("stage", ["evidence", "change"])
@pytest.mark.parametrize("feedback", [False, True])
async def test_session_input_changes_stop_before_the_next_gate_or_publish(tmp_path, stage, feedback):
    layout = RunLayout(root=tmp_path / "run")

    def alter_inputs(_):
        if stage == "evidence":
            write_evidence(layout.evidence_workspace)
        else:
            write_change(layout.change_workspace, ["F1", "F2"])
        original = layout.input_skills / "pdf/forms.md"
        original.write_text(original.read_text() + "\nINPUT_WAS_ALTERED\n")

    steps = []
    if stage == "change":
        steps.append(lambda _: write_evidence(layout.evidence_workspace))
    if feedback:
        if stage == "evidence":
            steps.append(lambda _: write_evidence(layout.evidence_workspace, script=BROKEN_SCRIPT))
        else:
            steps.append(lambda _: write_change(layout.change_workspace, ["F99"]))
    steps.append(alter_inputs)
    agent = FakeAgent(steps)
    with pytest.raises(RuntimeError, match="inputs/ changed"):
        await _evolve(tmp_path, agent)

    saved = read_state(layout.root)
    assert layout.inputs_digest() != saved.inputs_digest
    assert len(agent.requests) == len(steps)
    assert saved.outcome is None and not layout.output_skills.exists()
    assert f"{stage}_gate" not in saved.completed
    if stage == "evidence":
        assert saved.evidence_gate.attempts == (1 if feedback else 0)
        assert not layout.change_workspace.exists()
    else:
        assert saved.change_gate.accepted_files == []


@pytest.mark.parametrize("stage", ["evidence", "change"])
async def test_inputs_changed_during_a_gate_stop_before_the_next_stage(tmp_path, monkeypatch, stage):
    from mindmemos_skill.algos.evolve.offline_evolution import engine as evolver

    layout = RunLayout(root=tmp_path / "run")

    def alter_inputs():
        original = layout.input_skills / "pdf/forms.md"
        original.write_text(original.read_text() + "\nINPUT_WAS_ALTERED\n")

    if stage == "evidence":
        evidence_gate = evolver.run_evidence_gate

        async def changed_evidence(*args):
            report = await evidence_gate(*args)
            alter_inputs()
            return report

        monkeypatch.setattr(evolver, "run_evidence_gate", changed_evidence)
    else:
        change_gate = evolver.run_change_gate

        def changed_change(*args):
            report = change_gate(*args)
            alter_inputs()
            return report

        monkeypatch.setattr(evolver, "run_change_gate", changed_change)

    agent = FakeAgent(
        [
            lambda _: write_evidence(layout.evidence_workspace),
            lambda _: write_change(layout.change_workspace, ["F1", "F2"]),
        ]
    )
    with pytest.raises(RuntimeError, match="inputs/ changed"):
        await _evolve(tmp_path, agent)
    saved = read_state(layout.root)
    assert f"{stage}_gate" not in saved.completed
    assert len(agent.requests) == (1 if stage == "evidence" else 2)
    assert saved.outcome is None and not layout.output_skills.exists()


async def test_modified_inputs_stop_the_run(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    with pytest.raises(RuntimeError):
        await _evolve(tmp_path, FakeAgent([failing("abc")]))
    (layout.input_skills / "pdf" / "SKILL.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inputs/ changed"):
        await build_engine(FakeAgent([]), config=OfflineEvolutionConfig()).resume(work_dir=layout.root)


async def test_relative_run_directory_works(tmp_path, monkeypatch):
    skills, trajectories = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F1"]),
        ]
    )
    result = await build_engine(agent, config=OfflineEvolutionConfig()).evolve(
        skills=skills.relative_to(tmp_path), trajectories=trajectories.relative_to(tmp_path), work_dir=Path("run")
    )

    assert result.state.evidence_gate.accepted is True and result.outcome == "updated"
    assert all(r.cwd.is_absolute() for r in agent.requests)


def _risky_change(layout: RunLayout) -> None:
    write_change(layout.change_workspace, ["F1"])
    forms = layout.change_workspace / "skills" / "pdf" / "forms.md"
    forms.write_text(forms.read_text(encoding="utf-8") + "The reader changed in 4.2.0.\n", encoding="utf-8")


def _acknowledge(layout: RunLayout, request) -> None:
    path = layout.change_workspace / "changes.json"
    changes = json.loads(path.read_text(encoding="utf-8"))
    keys = [line.split("`")[1] for line in request.prompt.splitlines() if line.startswith("- `")]
    changes["acknowledged"] = [{"warning": key, "reason": "the bundled script requires it"} for key in keys]
    path.write_text(json.dumps(changes), encoding="utf-8")


async def test_warnings_are_returned_once_to_the_change_session(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: _risky_change(layout),
            lambda r: _acknowledge(layout, r),
        ]
    )
    result = await _evolve(tmp_path, agent)

    feedback = agent.requests[2]
    assert feedback.session_id == "session-2" and "version_condition:pdf/forms.md:4.2.0" in feedback.prompt
    record = result.state.change_gate
    assert record.feedback_rounds == 1 and result.outcome == "updated"
    assert {w.kind for w in record.warnings} == {"version_condition"}
    assert all(w.acknowledged for w in record.warnings)


async def test_warnings_without_retries_stay_in_the_report(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent([lambda r: write_evidence(layout.evidence_workspace), lambda r: _risky_change(layout)])
    result = await _evolve(tmp_path, agent, OfflineEvolutionConfig(retries=0))

    record = result.state.change_gate
    assert len(agent.requests) == 2 and record.feedback_rounds == 0
    assert result.outcome == "updated" and not any(w.acknowledged for w in record.warnings)
    assert "version_condition" in (layout.candidate / "gate.md").read_text(encoding="utf-8")
    assert result.state.evidence.tokens.output == 5 and result.state.change.tokens.cache_read == 100


async def test_repeated_finding_ids_are_renamed_not_rejected(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    script = EVIDENCE_SCRIPT.replace('f"F{len(findings) + 1}"', '"F1"')
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace, script=script),
            lambda r: write_change(layout.change_workspace, ["F1-2"]),
        ]
    )
    result = await _evolve(tmp_path, agent)

    assert result.state.evidence_gate.attempts == 1
    assert list(result.state.evidence_gate.accepted_findings) == [
        "F1",
        "F1-2",
    ]
    assert "1 repeated finding ids renamed" in (layout.evidence / "report.md").read_text(encoding="utf-8")
    assert result.outcome == "updated"


async def test_unaddressed_findings_are_returned_to_the_change_session(tmp_path):
    layout = RunLayout(root=tmp_path / "run")
    agent = FakeAgent(
        [
            lambda r: write_evidence(layout.evidence_workspace),
            lambda r: write_change(layout.change_workspace, ["F1"], dropped=[]),
            lambda r: write_change(layout.change_workspace, ["F1"]),
        ]
    )
    result = await _evolve(tmp_path, agent)

    assert "`F2`" in agent.requests[2].prompt and agent.requests[2].session_id == "session-2"
    record = result.state.change_gate
    assert record.feedback_rounds == 1 and record.unaddressed_findings == [] and result.outcome == "updated"


async def test_cancelled_invocation_recovers_identity_from_its_partial_event_log(tmp_path):
    import asyncio

    layout = RunLayout(root=tmp_path / "run")

    def cancelled(request):
        observed = json.dumps({"type": "system", "session_id": "streamed-before-cancellation"}) + "\n"
        request.log_path.write_bytes(observed.encode("utf-8") + b'{"type":"assistant","content":"\xe4')
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _evolve(tmp_path, FakeAgent([cancelled]))
    backend = FakeAgent(
        [
            lambda request: write_evidence(layout.evidence_workspace),
            lambda request: write_change(layout.change_workspace, ["F1"]),
        ]
    )
    result = await build_engine(backend).resume(work_dir=layout.root)
    assert backend.requests[0].session_id == "streamed-before-cancellation"
    assert backend.requests[0].log_path.name == "evidence-2.jsonl"
    assert result.outcome == "updated"


async def test_readonly_inputs_without_creation_skip_the_change_session(tmp_path):
    """Analyze the readable collection without invoking an unauthorized editing session."""
    layout = RunLayout(root=tmp_path / "run")
    backend = FakeAgent([lambda _: write_evidence(layout.evidence_workspace)])
    result = await _evolve(tmp_path, backend, options=EvolutionOptions(editable_skills=[], allow_create_skills=False))
    assert result.state.evidence_gate.accepted is True
    assert result.state.change.skipped == "all inputs are read-only and creation is forbidden"
    assert result.outcome == "unchanged" and len(backend.requests) == 1
    assert not layout.change_workspace.exists()
