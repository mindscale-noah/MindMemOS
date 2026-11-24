"""Exercise dataset loading, reproducible splitting, and private scoring inputs."""

import json
import random
from pathlib import Path

import pytest
from mindmemos_skill.envs import Environment, Task
from mindmemos_skill.envs.data import Data, DataSample, RewardData
from pydantic import BaseModel, ValidationError


class FileConfig(BaseModel):
    """Local source configuration for the test loader."""

    path: Path
    """JSON data file containing task IDs, instructions, and answers."""


class AnswerData(RewardData):
    """Private evaluation inputs for a simple answer task."""

    answer: str
    """Expected answer withheld from the agent."""


class JsonData(Data[FileConfig]):
    """Adapt a local JSON dataset into typed samples."""

    config_type = FileConfig
    """Configuration used by the shared registry."""

    def _load(self) -> list[DataSample]:
        """Map raw records without leaking answers into Task."""
        return [
            DataSample(
                task=Task(id=row["id"], instruction=row["question"]), reward_data=AnswerData(answer=row["answer"])
            )
            for row in json.loads(self.config.path.read_text())
        ]


@pytest.fixture
def dataset(tmp_path):
    """Create a deterministic local source with eleven independently identified tasks."""
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps([{"id": f"q-{i:02}", "question": f"Compute {i}+1", "answer": str(i + 1)} for i in range(11)])
    )
    return JsonData(FileConfig(path=path))


def membership(splits):
    """Extract ordered IDs for comparison and expected file content."""
    return {name: [sample.task.id for sample in samples] for name, samples in splits.items()}


def test_load_private_inputs_and_registration(dataset, monkeypatch):
    """Reuse the registry and retain typed private inputs without exposing them in Task."""
    with monkeypatch.context() as patch:
        patch.setattr(Data, "_registry", {})
        Data.register("json_test")(JsonData)
        loaded = Data.build("json_test", {"path": str(dataset.config.path)}).load()
        assert Data.resolve("json_test") is JsonData
        assert "json_test" not in Environment._registry
    assert len(loaded) == 11
    assert isinstance(loaded[0].reward_data, AnswerData)
    assert loaded[0].reward_data.answer == "1"
    assert "answer" not in loaded[0].task.model_dump()
    assert loaded[0].model_dump()["reward_data"] == {"answer": "1"}


def test_load_rejects_duplicate_ids(dataset):
    """Duplicate source IDs cannot silently overwrite samples."""
    dataset.config.path.write_text(
        json.dumps(
            [
                {"id": "same", "question": "a", "answer": "1"},
                {"id": "same", "question": "b", "answer": "2"},
            ]
        )
    )
    with pytest.raises(ValueError, match="Duplicate task ID"):
        dataset.load()


@pytest.mark.parametrize("identifier", ["", " ", "\n"])
def test_blank_task_ids_are_rejected(identifier):
    """Require stable nonblank task identities."""
    with pytest.raises(ValidationError):
        Task(id=identifier, instruction="task")


def test_seeded_split_is_order_independent_and_does_not_mutate_inputs(dataset):
    """Preserve reproducibility, global RNG state, full coverage, and sample identity."""
    samples = dataset.load()
    before = [sample.task.id for sample in samples]
    state = random.getstate()
    first = dataset.split(samples, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    second = dataset.split(list(reversed(samples)), ratios={"test": 0.2, "val": 0.2, "train": 0.6}, seed=42)
    assert membership(first) == membership(second)
    assert random.getstate() == state
    assert [sample.task.id for sample in samples] == before
    assert {name: len(group) for name, group in first.items()} == {"train": 7, "val": 2, "test": 2}
    assert sorted(sample.task.id for group in first.values() for sample in group) == before
    assert all(any(sample is original for original in samples) for group in first.values() for sample in group)
    assert membership(first) != membership(
        dataset.split(samples, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=43)
    )


def test_rounding_zero_ratios_and_empty_dataset(dataset):
    """Allocate tied remainders by name and handle empty or zero-size splits."""
    samples = dataset.load()[:1]
    splits = dataset.split(samples, ratios={"val": 0.5, "train": 0.5, "unused": 0}, seed=0)
    assert {name: len(group) for name, group in splits.items()} == {"train": 1, "val": 0, "unused": 0}
    assert dataset.split([], ratios={"train": 1}, seed=0) == {"train": []}


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"ratios": {"train": 1}},
        {"ratios": {}, "seed": 1},
        {"ratios": {"train": -0.1, "test": 1.1}, "seed": 1},
        {"ratios": {"train": 0.5}, "seed": 1},
        {"ratios": {"train": float("nan")}, "seed": 1},
        {"ratios": {" ": 1}, "seed": 1},
        {"ratios": {"train": 1}, "seed": True},
        {"split_file": "unused.json", "seed": 1},
        {"split_file": "unused.json", "ratios": {"train": 1}},
    ],
)
def test_invalid_split_options(dataset, options):
    """Fail invalid modes and proportions before file IO."""
    with pytest.raises(ValueError):
        dataset.split(dataset.load(), **options)


def test_save_and_reload_membership_only(dataset, tmp_path):
    """Persist exact membership without private answers or task contents."""
    samples = dataset.load()
    splits = dataset.split(samples, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    path = tmp_path / "splits" / "fixed.json"
    dataset.save_splits(splits, path)
    assert json.loads(path.read_text()) == membership(splits)
    assert membership(dataset.split(list(reversed(samples)), split_file=path)) == membership(splits)
    assert not path.with_name("fixed.json.tmp").exists()


def test_file_split_preserves_order_and_allows_subsets(dataset, tmp_path):
    """Explicit IDs select a subset in the file's order."""
    path = tmp_path / "subset.json"
    expected = {"train": ["q-09", "q-01"], "test": [], "dev": ["q-03"]}
    path.write_text(json.dumps(expected))
    assert membership(dataset.split(dataset.load(), split_file=path)) == expected


@pytest.mark.parametrize(
    "manifest",
    [
        {"train": ["unknown"]},
        {"train": ["q-00", "q-00"]},
        {"train": ["q-00"], "test": ["q-00"]},
        {"": []},
        {},
        {"train": "q-00"},
        {"train": [0]},
        {"train": [" "]},
    ],
)
def test_invalid_manifests(dataset, tmp_path, manifest):
    """Reject unknown, repeated, overlapping, and malformed membership."""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        dataset.split(dataset.load(), split_file=path)


def test_reject_invalid_samples_and_overlapping_save(dataset, tmp_path):
    """Validate both direct split inputs and exported membership."""
    sample = dataset.load()[0]
    with pytest.raises(ValueError, match="Duplicate task ID"):
        dataset.split([sample, sample], ratios={"train": 1}, seed=1)
    with pytest.raises(ValueError):
        dataset.split([{}], ratios={"train": 1}, seed=1)
    with pytest.raises(ValueError, match="Duplicate task ID"):
        dataset.save_splits({"train": [sample], "val": [sample]}, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
