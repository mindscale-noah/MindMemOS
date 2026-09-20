"""AC-P0-003: Repeatable GOS baseline fixture verification.

This module is the verifier contract test for the GOS minimal baseline
fixture. It validates every hash recorded in ``baseline_manifest.json``
against the actual on-disk files, so a drift in either the inputs or the
expected golden output cannot pass silently:

- manifest schema (schema_version=1, complete status, real version fields)
- ``baseline.lock`` SHA-256 matches the manifest
- every input file (skills + ssl_records + generator script) SHA-256 matches
- ``golden.json`` SHA-256 matches the manifest
- golden contains the required sections (parsed/nodes/edges/scenes/seeds/ppr/
  top_k/context) and covers dependency/semantic/containment/isolated/tie
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures/skill_manage/gos_minimal"
BASELINE_MANIFEST = FIXTURES_DIR / "baseline_manifest.json"
BASELINE_LOCK = FIXTURES_DIR / "baseline.lock"
EXPECTED_GOLDEN = FIXTURES_DIR / "expected/golden.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert BASELINE_MANIFEST.exists(), f"baseline_manifest.json not found at {BASELINE_MANIFEST}"
    return json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Manifest structure
# --------------------------------------------------------------------------
def test_baseline_manifest_exists():
    assert BASELINE_MANIFEST.exists()


def test_baseline_manifest_schema_and_completeness(manifest):
    """schema_version=1, status=complete, real versions (no TBD)."""
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"

    # Documented dependency versions must be present and real
    assert manifest["python_version"].startswith("3.12")
    assert manifest["fast_graphrag_version"] == "0.0.4"
    assert manifest["numpy_version"] not in ("", "TBD")
    assert manifest["openai_version"] not in ("", "TBD")
    assert manifest["gos_commit"] not in ("", "TBD")

    # baseline.lock hash is real
    lock_sha = manifest["baseline_lock_sha256"]
    assert lock_sha not in ("", "TBD", "Not calculated")
    assert len(lock_sha) == 64

    # Golden hash is real
    golden_sha = manifest["golden_sha256"]
    assert golden_sha not in ("", "TBD")
    assert len(golden_sha) == 64


def test_baseline_manifest_lock_exists(manifest):
    assert BASELINE_LOCK.exists()
    assert sha256(BASELINE_LOCK) == manifest["baseline_lock_sha256"], (
        "baseline.lock does not match the manifest's recorded SHA-256"
    )


def test_baseline_manifest_lists_all_input_files(manifest):
    """Manifest documents every fixture input with its SHA-256."""
    input_files = manifest["input_files"]
    assert len(input_files) >= 8  # 6 skills + 1 ssl record + 1 generator

    for entry in input_files:
        assert entry["path"], f"input file entry missing path: {entry}"
        assert len(entry["sha256"]) == 64, f"input {entry['path']} missing real hash"


def test_baseline_manifest_hashes_match_disk(manifest):
    """Every recorded input SHA-256 must match the actual on-disk file."""
    for entry in manifest["input_files"]:
        path = FIXTURES_DIR / entry["path"]
        assert path.exists(), f"manifest references missing file: {entry['path']}"
        assert sha256(path) == entry["sha256"], f"input file {entry['path']} changed since manifest was frozen"


def test_baseline_golden_matches_manifest(manifest):
    """golden.json bytes must match the manifest's recorded SHA-256."""
    assert EXPECTED_GOLDEN.exists()
    assert sha256(EXPECTED_GOLDEN) == manifest["golden_sha256"], (
        "golden.json does not match the manifest's recorded SHA-256"
    )


def test_baseline_golden_contains_required_sections(manifest):
    """Golden output contains all required comparison sections."""
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    required = {"parsed", "nodes", "edges", "scenes", "seeds", "ppr", "top_k", "context"}
    assert required.issubset(golden.keys()), f"golden output missing required sections: {required - golden.keys()}"
    assert golden["schema_version"] == 1


# --------------------------------------------------------------------------
# Fixture coverage: dependency / semantic / scene / isolated / tie
# --------------------------------------------------------------------------
def test_golden_contains_dependency_edges(manifest):
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    dep_edges = {e["type"] for e in golden["edges"]}
    assert "dependency" in dep_edges
    dependency_edges = [e for e in golden["edges"] if e["type"] == "dependency"]
    assert len(dependency_edges) >= 2


def test_golden_contains_semantic_edge(manifest):
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    semantic_edges = [e for e in golden["edges"] if e["type"] == "semantic"]
    assert semantic_edges, "golden must contain at least one semantic edge"
    # The semantic edge must be LLM-validated content-driven (source/target pair)
    edge = semantic_edges[0]
    assert edge["source"] and edge["target"]


def test_golden_contains_scene(manifest):
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    # Scene node present + containment edge
    scene_nodes = [n for n in golden["nodes"] if n.get("type") != "Skill"]
    assert scene_nodes, "golden must contain at least one Scene node"
    containment = [e for e in golden["edges"] if e["type"] == "containment"]
    assert containment, "golden must contain a containment edge"
    assert golden["scenes"], "golden must document the scene structure"


def test_golden_contains_isolated_skill(manifest):
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    edge_endpoints = {e["source"] for e in golden["edges"]} | {e["target"] for e in golden["edges"]}
    skill_names = {n["name"] for n in golden["nodes"] if n.get("type") == "Skill"}
    isolated = skill_names - edge_endpoints
    assert isolated, "golden must contain at least one Skill with zero incident edges"


def test_golden_contains_same_score_tie(manifest):
    """The fixture must exercise the stable tie-break path."""
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    scores = list(golden["ppr"]["scores"].values())
    # At least two distinct skills share the same score
    from collections import Counter

    counts = Counter(scores)
    tied = {score for score, count in counts.items() if count >= 2}
    assert tied, "golden PPR must contain at least one same-score tie"


def test_golden_nodes_and_edges_consistent(manifest):
    golden = json.loads(EXPECTED_GOLDEN.read_text(encoding="utf-8"))
    node_names = {n["name"] for n in golden["nodes"]}
    for e in golden["edges"]:
        assert e["source"] in node_names, f"edge source {e['source']} not a node"
        assert e["target"] in node_names, f"edge target {e['target']} not a node"


@pytest.mark.parametrize("with_missing_source", [False, True])
def test_generator_requires_an_explicit_reference_checkout(tmp_path: Path, with_missing_source: bool):
    """A checkout without the external GOS engine must fail promptly, without optional dependencies."""
    command = [sys.executable, "-S", str(FIXTURES_DIR / "scripts/generate_baseline.py")]
    if with_missing_source:
        command.extend(["--gos-source", str(tmp_path)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)

    assert result.returncode == 2
    assert "--gos-source" in result.stderr
    if with_missing_source:
        assert "must contain gos/core/engine.py" in result.stderr
    else:
        assert "required" in result.stderr
