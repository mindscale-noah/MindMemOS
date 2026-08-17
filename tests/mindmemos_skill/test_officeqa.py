from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.datasets import OfficeQAIdSplitDataset
from mindmemos_skill.envs import EnvRolloutContext, OfficeQAEnv
from mindmemos_skill.envs.registered_envs.officeqa import search
from mindmemos_skill.envs.registered_envs.officeqa.env import (
    _MAX_ORACLE_PAGE_CHARS,
    OfficeQATools,
    _build_oracle_context,
)
from mindmemos_skill.envs.registered_envs.officeqa.evaluator import (
    OFFICEQA_NUMERIC_TOLERANCE,
    OFFICEQA_REWARD_COMMIT,
    evaluate,
)
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.typing import Rollout, Skill, Task, compute_skill_content_hash


class AnswerClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def chat(self, task: str, messages: list[dict], *, model=None, **kwargs):
        del task, model, kwargs
        self.calls.append(messages)
        return ChatResponse(
            finish_reason="stop",
            content="<answer>2,602 million dollars</answer>",
            model="fake",
        )


class SearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat(self, task: str, messages: list[dict], *, model=None, **kwargs):
        del task, model
        self.calls.append((messages, kwargs))
        content = (
            '<search_queries>["Treasury Bulletin total"]</search_queries>'
            if len(self.calls) == 1
            else "<answer>2,602 million dollars</answer>"
        )
        return ChatResponse(finish_reason="stop", content=content, model="fake")


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


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


def test_officeqa_oracle_context_uses_skillopt_pairing_deduplication_and_truncation(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    transformed = docs / "transformed"
    parsed = docs / "jsons"
    transformed.mkdir(parents=True)
    parsed.mkdir()
    for name in ("a.txt", "b.txt"):
        (transformed / name).write_text(name, encoding="utf-8")
    (parsed / "a.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"bbox": [{"page_id": 1}], "content": "page one"},
                        {"bbox": [{"page_id": 3}], "content": "must be skipped"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (parsed / "b.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"bbox": [{"page_id": 2}], "content": "x" * (_MAX_ORACLE_PAGE_CHARS + 10)}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    task = Task(
        task_id="officeqa-context",
        instruction="question",
        metadata={
            "source_files": ["a.txt", "b.txt"],
            "source_docs": [
                "https://example.test/a?page=1",
                "https://example.test/b?page=2",
                "https://example.test/a?page=3",
            ],
        },
    )

    context = _build_oracle_context(task, [transformed.resolve()])

    assert context.count("### a.txt page 1") == 1
    assert "### b.txt page 2" in context
    assert "must be skipped" not in context
    assert "[... 10 characters omitted from this parsed page ...]" in context

    duplicate_task = Task(
        task_id="officeqa-dedup",
        instruction="question",
        metadata={
            "source_files": ["a.txt"],
            "source_docs": [
                "https://example.test/first?page=1",
                "https://example.test/second?page=1",
            ],
        },
    )
    duplicate_context = _build_oracle_context(duplicate_task, [transformed.resolve()])
    assert duplicate_context.count("### a.txt page 1") == 1


@pytest.mark.asyncio
async def test_officeqa_env_uses_official_reward(tmp_path: Path) -> None:
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
    assert trajectory.reward.metadata["evaluator_commit"] == OFFICEQA_REWARD_COMMIT
    assert trajectory.reward.metadata["tolerance"] == OFFICEQA_NUMERIC_TOLERANCE
    assert trajectory.environment.env_ref == "officeqa"
    assert trajectory.metadata["oracle_parsed_pages_chars"] > 0
    system_prompt = client.calls[0][0]["content"]
    assert "\n\n## Skill\n# OfficeQA Skill\n\nVerify every extracted value.\n\n## Rules\n" in system_prompt
    assert "<available_skills>" not in system_prompt
    assert "Oracle reported total" in client.calls[0][1]["content"]
    assert f"## Candidate Files\n- {(transformed / 'treasury_bulletin_1941_01.txt').resolve()}" in client.calls[0][1][
        "content"
    ]
    assert trajectory.skill_bindings[0].injection_mode.value == "system_prompt"
    assert Path(trajectory.environment.running_dir, "prediction.json").is_file()


def test_officeqa_official_reward_matches_2026_06_13_contract() -> None:
    assert evaluate("113,864 thousand dollars", "113864")["em"] == 1.0
    assert evaluate("<FINAL_ANSWER>73%</FINAL_ANSWER>", "73")["em"] == 1.0
    assert evaluate("The inputs were 10 and 20.\nThe final result is 30.", "30")["em"] == 0.0


def test_officeqa_custom_search_uses_configurable_official_serper_protocol(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "organic": [
                    {
                        "title": f"Result {index}",
                        "link": f"https://example.test/{index}",
                        "snippet": "Evidence",
                    }
                    for index in range(1, 6)
                ]
            }
        )

    monkeypatch.setenv("OFFICEQA_SERPAPI_API_KEY", "dummy-secret")
    monkeypatch.setattr(search, "urlopen", fake_urlopen)

    result = search.custom_search(
        "Treasury Bulletin August 1982 yield",
        base_url="https://search-gateway.example.test/custom",
        max_num_results=4,
        timeout=19,
    )

    request = captured["request"]
    assert request.full_url == "https://search-gateway.example.test/custom/search"
    assert request.method == "POST"
    assert request.get_header("X-api-key") == "dummy-secret"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode("utf-8")) == {
        "q": "Treasury Bulletin August 1982 yield",
        "num": 4,
    }
    assert captured["timeout"] == 19
    assert "Result 4" in result
    assert "Result 5" not in result


def test_officeqa_custom_search_supports_yibu_protocol(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "organic_results": [
                    {
                        "title": "Treasury Bulletin",
                        "link": "https://example.test/treasury",
                        "snippet": "Evidence",
                    }
                ]
            }
        )

    monkeypatch.setenv("OFFICEQA_SERPAPI_API_KEY", "dummy-secret")
    monkeypatch.setattr(search, "urlopen", fake_urlopen)

    result = search.custom_search(
        "Treasury Bulletin August 1982 yield",
        base_url="https://yibuapi.example/serpapi",
        protocol="yibu",
        max_num_results=4,
        timeout=17,
    )

    request = captured["request"]
    assert request.method == "GET"
    assert request.data is None
    assert request.get_header("Authorization") == "Bearer dummy-secret"
    assert parse_qs(urlparse(request.full_url).query) == {
        "q": ["Treasury Bulletin August 1982 yield"],
        "output": ["json"],
        "num": ["4"],
    }
    assert urlparse(request.full_url).path == "/serpapi/search"
    assert captured["timeout"] == 17
    assert "Treasury Bulletin" in result


@pytest.mark.asyncio
async def test_officeqa_env_runs_controller_managed_custom_search(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setenv("OFFICEQA_SERPAPI_BASE_URL", "https://google.serper.dev")
    monkeypatch.setenv("OFFICEQA_SERPAPI_API_KEY", "dummy-secret")
    monkeypatch.setenv("OFFICEQA_SERPAPI_PROTOCOL", "serper")

    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeResponse(
            {
                "organic": [
                    {
                        "title": "Treasury evidence",
                        "link": "https://example.test/evidence",
                        "snippet": "The reported total was 2,602 million dollars.",
                    }
                ]
            }
        )

    monkeypatch.setattr(search, "urlopen", fake_urlopen)
    task = OfficeQAIdSplitDataset(data_root=data_root, split_dir=split_root).test_tasks()[0]
    client = SearchClient()
    agent = ReactAgent({"model": "fake", "max_turns": 2}, llm=client)

    trajectory = await OfficeQAEnv(
        {
            "max_turns": 2,
            "docs_root": str(docs),
            "search_mode": "custom_search",
        }
    ).rollout(
        agent,
        task,
        [],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="officeqa-search-rollout"),
            workspace_root=tmp_path / "search-workspace",
            env_ref="officeqa",
        ),
    )

    assert trajectory.reward.score == 1.0
    assert trajectory.metadata["search_mode"] == "custom_search"
    assert len(client.calls) == 2
    assert client.calls[0][1]["tools"] == []
    assert "<search_queries>" in client.calls[0][0][0]["content"]
    assert "The reported total was 2,602 million dollars." in client.calls[1][0][-1]["content"]
