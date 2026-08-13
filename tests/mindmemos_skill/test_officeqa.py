from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.datasets import OfficeQAIdSplitDataset
from mindmemos_skill.envs import EnvRolloutContext, OfficeQAEnv
from mindmemos_skill.envs.registered_envs.officeqa.env import OfficeQATools
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.typing import Rollout, Skill, compute_skill_content_hash


class AnswerClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def chat(self, task: str, messages: list[dict], *, model=None, **kwargs):
        del task, model, kwargs
        self.calls.append(messages)
        return ChatResponse(finish_reason="stop", content="<answer>2,602</answer>", model="fake")


def _fixture_dataset(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "officeqa"
    data_root.mkdir()
    with (data_root / "officeqa_full.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["uid", "question", "answer", "source_docs", "source_files", "difficulty"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uid": "UID0001",
                "question": "What was the total?",
                "answer": "2,602 million dollars",
                "source_docs": "https://example.test/?page=15",
                "source_files": "treasury_bulletin_1941_01.txt",
                "difficulty": "hard",
            }
        )
    split_root = tmp_path / "splits"
    for split in ("train", "val", "test"):
        path = split_root / split / "items.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "UID0001",
                        "uid": "UID0001",
                        "category": "hard",
                        "source_files": "treasury_bulletin_1941_01.txt",
                        "source_docs": "https://example.test/?page=15",
                        "source_split": split,
                    }
                ]
            ),
            encoding="utf-8",
        )
    return data_root, split_root


def test_officeqa_dataset_joins_skillopt_split_to_full_payload(tmp_path: Path) -> None:
    data_root, split_root = _fixture_dataset(tmp_path)
    dataset = OfficeQAIdSplitDataset(data_root=data_root, split_dir=split_root)

    task = dataset.train_tasks()[0]

    assert task.task_id == "UID0001"
    assert task.instruction == "What was the total?"
    assert task.metadata["ground_truth"] == "2,602 million dollars"
    assert task.metadata["source_files"] == ["treasury_bulletin_1941_01.txt"]


def test_officeqa_tools_restrict_reads_to_candidate_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    allowed = docs / "allowed.txt"
    denied = docs / "denied.txt"
    allowed.write_text("alpha\nneedle value\nomega\n", encoding="utf-8")
    denied.write_text("secret\n", encoding="utf-8")
    tools = OfficeQATools(docs_roots=[docs.resolve()], allowed_files=["allowed.txt"])

    assert str(allowed) in tools.glob("*.txt")
    assert "needle value" in tools.grep("needle", str(allowed))
    assert "path not allowed" in tools.read(str(denied))


@pytest.mark.asyncio
async def test_officeqa_env_uses_skillopt_answer_normalization(tmp_path: Path) -> None:
    data_root, split_root = _fixture_dataset(tmp_path)
    docs = data_root / "officeqa_docs_official"
    transformed = docs / "transformed"
    parsed = docs / "jsons"
    transformed.mkdir(parents=True)
    parsed.mkdir()
    (transformed / "treasury_bulletin_1941_01.txt").write_text("reported total: 2,602\n", encoding="utf-8")
    (parsed / "treasury_bulletin_1941_01.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"bbox": [{"page_id": 15}], "content": "Oracle reported total: 2,602 million dollars."}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    task = OfficeQAIdSplitDataset(data_root=data_root, split_dir=split_root).test_tasks()[0]
    client = AnswerClient()
    agent = ReactAgent(
        {"model": "fake", "max_turns": 2, "skill_injection_mode": "tool"},
        llm=client,
    )
    blob = {"SKILL.md": "# OfficeQA Skill\n\nVerify every extracted value."}
    skill = Skill(
        skill_id="officeqa-skill",
        version_id="officeqa-skill-v1",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash(blob),
        name="officeqa",
        blob=blob,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
    )

    trajectory = await OfficeQAEnv({"max_turns": 2, "docs_root": str(docs)}).rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="officeqa-rollout"),
            workspace_root=tmp_path / "workspace",
            env_ref="officeqa",
        ),
    )

    assert trajectory.reward.score == 1.0
    assert trajectory.reward.metadata["f1"] == 1.0
    assert trajectory.environment.env_ref == "officeqa"
    assert trajectory.metadata["oracle_parsed_pages_chars"] > 0
    assert "Verify every extracted value" in client.calls[0][0]["content"]
    assert "Oracle reported total" in client.calls[0][1]["content"]
    assert trajectory.skill_bindings[0].injection_mode.value == "system_prompt"
    assert Path(trajectory.environment.running_dir, "prediction.json").is_file()
