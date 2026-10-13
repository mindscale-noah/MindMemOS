"""Check metric parity and source identity isolation through the real SDK transport."""

import argparse
import json

import httpx
import pytest
from mindmemos_sdk.skills import SkillClient
from mindmemos_sdk.transport import HttpTransport

from scripts.skill_eval.agentskillos.evaluate import _load_dataset, _run_evaluation, _wait_for_graph


def _create_dataset(root):
    """Create a raw source pool with distinct source IDs, member IDs, and names."""
    root.mkdir()
    for source in ("source-a", "source-b", "source-c"):
        directory = root / "skill_pool_200" / source
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: name-{source}\ndescription: Example skill\n---\n# Instructions\n"
        )
        (directory / "resource.txt").write_text("Auxiliary resource")
    (root / "ssl_pool_join.json").write_text(
        json.dumps({"join": {source: f"member-{source}" for source in ("source-a", "source-b", "source-c")}})
    )
    queries = [
        {"task_id": "query-a", "instruction_text": "Public instruction"},
        {"task_id": "generic", "instruction_text": "Generic query"},
    ]
    (root / "tasks.jsonl").write_text("\n".join(json.dumps(query) for query in queries))
    (root / "relevance.json").write_text(
        json.dumps(
            {
                "query-a": {"core_gt_ids": ["source-a", "source-c"]},
                "generic": {"core_gt_ids": [], "task_type": "generic_only"},
            }
        )
    )


@pytest.mark.parametrize("async_build", [False, True])
@pytest.mark.parametrize("retrieval_failure", [None, "snapshot", "unknown", "duplicate", "http"])
@pytest.mark.parametrize("reuse", [False, True])
def test_sdk_registration_build_and_scoring(tmp_path, async_build, retrieval_failure, reuse):
    """Verify exact pool, pinned retrieval, macro metrics, and incomplete-run accounting."""
    dataset = tmp_path / "dataset"
    _create_dataset(dataset)
    args = argparse.Namespace(
        dataset_root=dataset,
        output=tmp_path / "output",
        graph_alias=None,
        force_extract_scene=False,
        task_timeout=1,
        poll_interval=0.001,
        top_k=50,
        strategy="graph",
        include_scenes=False,
    )
    registered = []
    queries = []
    graph = {"graph_id": "graph-1", "graph_version_id": "snapshot-1", "skill_version_ids": ["v1", "v2", "v3"]}
    registrations_path = None
    if reuse:
        registrations_path = tmp_path / "registrations.json"
        registrations_path.write_text(json.dumps({"source-a": "v1", "source-b": "v2", "source-c": "v3"}))

    def handle_request(request):
        """Validate SDK wire requests and return offline service responses."""
        body = json.loads(request.content)
        route = request.url.path.rsplit("/", 1)[-1]
        if route == "register":
            assert "core_gt_ids" not in json.dumps(body)
            assert body["skill"]["resources"]["resource.txt"]
            registered.append(body)
            result = {"skill_id": f"s{len(registered)}", "skill_version_id": f"v{len(registered)}"}
        elif route == "organize":
            assert body["skill_refs"] == ["v1", "v2", "v3"]
            result = {"task_id": "task-1", "status": "queued"} if async_build else graph
        elif route == "show":
            assert body == {"task_id": "task-1"}
            result = {"task_id": "task-1", "status": "succeeded", "result": graph}
        elif route == "recommend":
            queries.append(body)
            assert body["query"] == "Public instruction"
            assert body["graph_version_id"] == "snapshot-1"
            assert "core_gt_ids" not in body
            if retrieval_failure == "http":
                return httpx.Response(500, json={"code": "error", "message": "Offline failure"})
            versions = ["v2", "v1"]
            if retrieval_failure == "unknown":
                versions = ["outside-pool"]
            if retrieval_failure == "duplicate":
                versions = ["v1", "v1"]
            result = {
                "graph_version_id": "wrong" if retrieval_failure == "snapshot" else "snapshot-1",
                "scores": [{"skill_version_id": version} for version in versions],
            }
        else:
            raise AssertionError(f"Unexpected route {route}")
        return httpx.Response(200, json={"code": "ok", "data": result})

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as http:
        transport = HttpTransport(base_url="http://offline.test", api_key="test", client=http, max_retries=0)
        summary = _run_evaluation(SkillClient(transport), args, registrations_path=registrations_path)
    assert len(registered) == (0 if reuse else 3)
    assert len(queries) == 1
    assert summary["candidate_count"] == 3
    assert len(summary["skipped_queries"]) == 1
    assert json.loads((args.output / "summary.json").read_text()) == summary
    if retrieval_failure:
        assert summary["status"] == "partial"
        assert summary["scoring_coverage"] == 0
        assert summary["metrics"] == {}
        assert len(summary["errors"]) == 1
    else:
        assert summary["status"] == "complete"
        assert summary["metrics"]["MRR@10"] == 0.5
        assert summary["metrics"]["Recall@10"] == 0.5
        assert summary["metrics"]["Recall@20"] == 0.5
        assert summary["metrics"]["Recall@50"] == 0.5
        row = json.loads((args.output / "results.jsonl").read_text())
        assert row["ranked_source_ids"] == ["source-b", "source-a"]
        assert row["ranked_member_ids"] == ["member-source-b", "member-source-a"]


def test_invalid_labels_are_rejected_before_mutations(tmp_path):
    """Out-of-pool labels never shrink the recall denominator silently."""
    root = tmp_path / "data"
    _create_dataset(root)
    labels = json.loads((root / "relevance.json").read_text())
    labels["query-a"]["core_gt_ids"] = ["missing-source"]
    (root / "relevance.json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="outside the pool"):
        _load_dataset(root)


def test_prepared_dataset_split_is_validated(tmp_path):
    """Prepared inputs must retain the full fixed test split."""
    root = tmp_path / "data"
    _create_dataset(root)
    (root / "queries").mkdir()
    for name in ("tasks.jsonl", "relevance.json"):
        (root / name).rename(root / "queries" / name)
    members = [
        {"source_skill_id": source, "member_id": f"member-{source}", "relative_path": f"skill_pool_200/{source}"}
        for source in ("source-a", "source-b", "source-c")
    ]
    (root / "skill_pool_manifest.json").write_text(json.dumps({"skills": members}))
    (root / "splits.json").write_text(json.dumps({"test": ["query-a", "generic"]}))
    assert len(_load_dataset(root)[2]) == 3
    (root / "splits.json").write_text(json.dumps({"test": ["query-a", "query-a"]}))
    with pytest.raises(ValueError, match="test split"):
        _load_dataset(root)


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued"])
def test_async_failure_or_timeout_preserves_submission(tmp_path, status):
    """A failed or uncompleted build remains traceable by its saved task identity."""

    def handle_request(request):
        """Return a terminal failure or an uncompleted task."""
        return httpx.Response(200, json={"code": "ok", "data": {"status": status, "error_type": "BuildError"}})

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as http:
        client = SkillClient(
            HttpTransport(base_url="http://offline.test", api_key=None, client=http, require_api_key=False)
        )
        with pytest.raises(TimeoutError if status == "queued" else RuntimeError):
            _wait_for_graph(client, {"task_id": "task-1"}, tmp_path, timeout=0.01, interval=0.001)
    assert json.loads((tmp_path / "organization.json").read_text())["task_id"] == "task-1"


def test_cli_uses_selected_sdk_settings_directory(tmp_path, monkeypatch, capsys):
    """Run the settings-based CLI path without network or saved credentials."""
    from scripts.skill_eval.agentskillos import evaluate

    home = tmp_path / "sdk"
    home.mkdir()
    (home / "settings.json").write_text(json.dumps({"network": {"timeout_seconds": 12, "max_retries": 0}}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--dataset-root",
            str(tmp_path / "data"),
            "--sdk-home",
            str(home),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    def run_offline(client, args, *, registrations_path):
        """Return a completed CLI result after connection settings are loaded."""
        assert isinstance(client, SkillClient)
        assert args.sdk_home == home
        assert registrations_path is None
        return {"status": "complete", "metrics": {"MRR@10": 1.0}}

    monkeypatch.setattr(evaluate, "_run_evaluation", run_offline)
    evaluate.main()
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "mapping",
    [
        {"source-a": "v1"},
        {"source-a": "v1", "source-b": "v1", "source-c": "v3"},
        {"source-a": "", "source-b": "v2", "source-c": "v3"},
    ],
)
def test_reused_registration_pool_is_validated_before_mutation(tmp_path, mapping):
    """Never register or build from partial, duplicate, or empty reused identities."""
    dataset = tmp_path / "dataset"
    _create_dataset(dataset)
    registrations = tmp_path / "registrations.json"
    registrations.write_text(json.dumps(mapping))
    args = argparse.Namespace(dataset_root=dataset, output=tmp_path / "output")
    with pytest.raises(ValueError, match="complete pool"):
        _run_evaluation(None, args, registrations_path=registrations)
    assert not args.output.exists()
