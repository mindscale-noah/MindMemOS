"""The unified evidence gate checks sources and coverage without grading comparisons or audit tables."""

import json

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.config import OfflineEvolutionConfig
from mindmemos_skill.algos.evolve.offline_evolution.evidence_gate import run_evidence_gate
from mindmemos_skill.algos.evolve.offline_evolution.provenance import quoted
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout

from .helpers import EVIDENCE_SCRIPT, make_inputs, write_evidence


def _layout(tmp_path) -> RunLayout:
    skills, trajectories = make_inputs(tmp_path)
    layout = RunLayout(root=tmp_path / "run")
    layout.root.mkdir()
    layout.snapshot(skills, trajectories)
    return layout


def _with_output(code: str) -> str:
    return EVIDENCE_SCRIPT.replace("json.dump(output,", code + "\njson.dump(output,")


def test_quotes_match_crlf_and_json_escapes():
    data = b'{\r\n  "skill": "pdf",\r\n  "out": "line one\\nline two"\r\n}'
    assert quoted('"skill": "pdf",\n  "out"', data)
    assert quoted("line one\nline two", data)
    assert not quoted('"skill": "docx"', data)


async def test_minimal_collection_needs_no_definitions_views_or_success_labels(tmp_path):
    layout = _layout(tmp_path)
    script = _with_output('findings[0]["case"] = "local-comparison"')
    write_evidence(layout.evidence_workspace, script)
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "accepted" and report.errors == []
    assert report.processed == 2 and len(report.findings) == 2
    assert set(report.accepted_findings()) == {"F1", "F2"}
    assert report.accepted_findings()["F1"].case == "local-comparison"
    assert not (layout.evidence_workspace / "definitions.md").exists()
    assert (layout.evidence / "outputs" / "attempt-1.json").is_file()


async def test_optional_statistics_and_notes_are_displayed_as_unverified(tmp_path):
    layout = _layout(tmp_path)
    script = _with_output(
        'output["summary"] = {"arbitrary_view": "author interpretation"}\noutput["notes"] = "unknown starting conditions"'
    )
    write_evidence(layout.evidence_workspace, script)
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "accepted"
    assert report.summary == {"arbitrary_view": "author interpretation"}
    rendered = (layout.evidence / "report.md").read_text()
    assert "Author summary (unverified)" in rendered and "Author notes (unverified)" in rendered
    assert "unknown starting conditions" in rendered


async def test_every_input_must_be_accounted_for_and_unknown_paths_rejected(tmp_path):
    layout = _layout(tmp_path)
    write_evidence(layout.evidence_workspace, _with_output('output["processed"] = processed[:1]'))
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "rejected" and any("neither processed nor reported" in e for e in report.errors)
    assert report.accepted_findings() == {}
    script = _with_output('output["processed"] = processed + ["trajectories/missing.json"]')
    write_evidence(layout.evidence_workspace, script)
    report = await run_evidence_gate(layout, 2, OfflineEvolutionConfig())
    assert any("not trajectory inputs" in e for e in report.errors)


async def test_an_unparsed_input_can_be_accounted_for_with_a_reason(tmp_path):
    layout = _layout(tmp_path)
    script = _with_output(
        'output["processed"] = processed[:1]\noutput["failed"] = [{"path": processed[1], "reason": "unrecorded format"}]\noutput["findings"] = findings[:1]'
    )
    write_evidence(layout.evidence_workspace, script)
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "accepted" and report.processed == 1
    assert len(report.failed) == 1 and report.failed[0].reason == "unrecorded format"


async def test_invalid_findings_are_excluded_and_excess_invalid_findings_reject_collection(tmp_path):
    layout = _layout(tmp_path)
    invalid = 'findings.append({"id": "X", "path": processed[0], "quote": "invented"})'
    write_evidence(layout.evidence_workspace, _with_output(invalid))
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "accepted" and len(report.invalid_findings) == 1
    assert set(report.accepted_findings()) == {"F1", "F2"}
    invalid += '\nfindings.append({"id": "Y", "path": processed[0], "quote": ""})'
    write_evidence(layout.evidence_workspace, _with_output(invalid))
    report = await run_evidence_gate(layout, 2, OfflineEvolutionConfig())
    assert report.status == "rejected" and "2/4 findings failed verification" in report.errors[-1]
    assert report.accepted_findings() == {}
    assert {f.problem.split(":")[0] for f in report.invalid_findings} == {
        "quote not found verbatim in the cited file",
        "malformed finding",
    }


async def test_skill_attribution_and_input_boundaries_are_checked(tmp_path):
    layout = _layout(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside quote")
    script = _with_output(
        'findings[0]["skill"] = "missing"\nfindings[1]["path"] = "trajectories/../../outside.txt"\nfindings[1]["quote"] = "outside quote"'
    )
    write_evidence(layout.evidence_workspace, script)
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "rejected"
    assert {f.problem for f in report.invalid_findings} == {
        "skill 'missing' is not one of the input skills",
        "cited file does not exist",
    }


async def test_collector_runs_once_and_does_not_reuse_stale_output(tmp_path):
    layout = _layout(tmp_path)
    marker = tmp_path / "calls.txt"
    write_evidence(
        layout.evidence_workspace,
        _with_output(
            f'marker = pathlib.Path({str(marker)!r})\nmarker.write_text(marker.read_text() + "call\\n" if marker.exists() else "call\\n")'
        ),
    )
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "accepted" and marker.read_text().splitlines() == ["call"]
    write_evidence(layout.evidence_workspace, "raise RuntimeError('collection failed')\n")
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "rejected" and any("collection failed" in e for e in report.errors)
    assert report.accepted_findings() == {}
    assert not (layout.evidence / "outputs" / "attempt-1.json").exists()


@pytest.mark.parametrize("manifest", [{}, {"collector": "scripts/collect.py", "directions": []}])
async def test_only_the_current_collection_manifest_is_accepted(tmp_path, manifest):
    layout = _layout(tmp_path)
    write_evidence(layout.evidence_workspace, "raise AssertionError('must not execute')\n")
    (layout.evidence_workspace / "collection.json").write_text(json.dumps(manifest))
    report = await run_evidence_gate(layout, 1, OfflineEvolutionConfig())
    assert report.status == "rejected" and "collection.json missing or invalid" in report.errors[0]
    assert report.findings == []
