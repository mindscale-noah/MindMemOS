"""The current contracts use a unified collector and optional, ungraded comparison groups."""

import json

import pytest
from mindmemos_skill.algos.evolve.offline_evolution.typing import EvidenceManifest, Finding, ScriptOutput
from pydantic import ValidationError


def test_collection_manifest_has_only_the_single_entry_point():
    with pytest.raises(ValidationError, match="collector"):
        EvidenceManifest.model_validate({})
    with pytest.raises(ValidationError, match="directions"):
        EvidenceManifest.model_validate({"collector": "scripts/collect.py", "directions": []})
    assert EvidenceManifest.model_validate({"collector": "scripts/collect.py"}).collector == "scripts/collect.py"


def test_windows_paths_are_normalized():
    path = "trajectories\\task\\run.json"
    output = ScriptOutput.model_validate_json(
        json.dumps(
            {
                "processed": [path],
                "failed": [{"path": path, "reason": "r"}],
                "findings": [{"id": "F1", "path": path, "quote": "q"}],
            }
        )
    )
    expected = "trajectories/task/run.json"
    assert output.processed == [expected] and output.failed[0].path == expected
    assert Finding.model_validate(output.findings[0]).path == expected


def test_comparison_group_needs_no_roles_or_outcome_labels():
    finding = Finding(id="F1", path="p", quote="q", case="comparison-1")
    assert finding.case == "comparison-1"
    assert Finding(id="F2", path="p", quote="q").case is None


@pytest.mark.parametrize("bad", [{"issue": "uncovered", "skill": "pdf"}, {"issue": "loading"}])
def test_issue_skill_attribution_is_consistent(bad):
    with pytest.raises(ValidationError):
        Finding.model_validate({"id": "F1", "path": "p", "quote": "q", **bad})
