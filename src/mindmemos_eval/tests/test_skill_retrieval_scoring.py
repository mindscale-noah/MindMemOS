"""Exercise graph retrieval and official scoring through the evaluation CLI."""

import json
from types import SimpleNamespace

import pytest
from mindmemos_eval.cli import main
from mindmemos_eval.skills import service
from mindmemos_eval.skills.algos import skill_graph
from mindmemos_skill.api.typing import OrganizeResult
from mindmemos_skill.config.config import ImplementationConfig


@pytest.mark.parametrize("failure", [None, "snapshot", "unknown", "duplicate"])
def test_skill_graph_generates_rankings_before_scoring(tmp_path, monkeypatch, capsys, failure):
    """Keep graph outputs separate from gold labels and reject invalid recommendations."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for task_id, skills in (("task-a", ["docx", "pptx"]), ("task-b", ["xlsx"])):
        (tasks / f"{task_id}.json").write_text(
            json.dumps({"task_id": task_id, "prompt": f"Public {task_id}", "skills": skills})
        )

    skill_dir = tmp_path / "skills"
    for source_id in ("docx", "unrelated"):
        source = skill_dir / source_id
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(f"---\nname: display-{source_id}\ndescription: Test Skill\n---\nBody")
        (source / "resource.txt").write_text("resource")

    closed = []

    class Service:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            closed.append(True)
            return False

        async def register_skill(self, request):
            assert request.skill.resources["resource.txt"]
            return SimpleNamespace(skill_version_id=request.skill.name)

        async def organize_skills(self, request):
            assert request.skill_refs == ["display-docx", "display-unrelated"]
            return OrganizeResult(graph_id="graph", graph_version_id="snapshot", skill_version_ids=request.skill_refs)

        async def recommend_skills(self, request):
            assert request.graph_version_id == "snapshot"
            assert request.task in {"Public task-a", "Public task-b"}
            ids = ["display-unrelated", "display-docx"] if request.task == "Public task-a" else []
            if failure == "unknown":
                ids = ["outside"]
            elif failure == "duplicate":
                ids = ["display-docx", "display-docx"]
            return SimpleNamespace(
                graph_version_id="other" if failure == "snapshot" else "snapshot",
                scores=[SimpleNamespace(skill_version_id=skill_id) for skill_id in ids],
            )

    async def create_service(deployment):
        assert deployment.service.home == output / "state"
        assert deployment.fastapi.auth.default_project_id == skill_graph.PROJECT_ID
        return Service()

    config = SimpleNamespace(agents={}, manage_pipeline=ImplementationConfig(class_name="graph_manage"))
    monkeypatch.setattr(skill_graph, "read_skill_config", lambda path: SimpleNamespace(service=config))
    monkeypatch.setattr(service, "create_service", create_service)
    output = tmp_path / "run"
    args = [
        "skill",
        "evaluate",
        "manage",
        "--algo",
        "skill-graph",
        "--eval-mode",
        "retrieval",
        "--tasks-dir",
        str(tasks),
        "--skills-dir",
        str(skill_dir),
        "--output-dir",
        str(output),
    ]
    assert main(args) == int(failure is not None)
    assert closed == [True]
    if failure:
        assert "Retrieval failed for task-a" in capsys.readouterr().out
        assert not (output / "rankings.json").exists()
        assert not (output / "summary.json").exists()
        return

    assert json.loads((output / "rankings.json").read_text()) == {"task-a": ["unrelated", "docx"], "task-b": []}
    report = json.loads((output / "summary.json").read_text())
    assert report["label_source"] == "official_task_skills"
    assert report["per_task"]["task-a"]["Recall@10"] == 0.5
    assert report["per_task"]["task-a"]["MRR@10"] == 0.5
    assert report["per_task"]["task-b"]["Recall@10"] == 0
    assert report["metrics"]["Recall@10"] == 0.25
